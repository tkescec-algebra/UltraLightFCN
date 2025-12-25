"""
Complete HPO script (finetune-only) aligned with methodology:
    - Phase2 encoder init (fixed)
    - Nondeterministic HPO (fast), deterministic VAL via transforms (valid mode)
    - Selection metric: soft Dice (thr=None)
    - Best checkpoint chosen by max avg_last_k (K=10)
    - No checkpoint saving (store Optuna user_attrs for reproducibility)
    - MedianPruner
    - Logs soft + hard@0.5 Dice (hard only for monitoring, not selection)
    - Option: fixed TRAIN+VAL subsets via file lists (auto-generated if missing),
      later top-10 retrain on full dataset
"""

import os
import random
from collections import deque

import cv2
import torch
import optuna
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from models.UltraLightFCN_base import UltraLightFCN
from utils.helpers import get_loss_function, clear_cuda_cache
from utils.load_simclr_pretrain_encoder import load_phase2_encoder_into_ultralight
from utils.metrics import calculate_dice
from utils.repro import set_global_seed
from utils.dataset import SolarPanelDataset


# -----------------------
# Global config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/workspace/UltraLightFCN/dataset"
TRAIN_SPLIT = "train"
VAL_SPLIT = "valid"

PHASE2_CKPT = "/workspace/UltraLightFCN/pretrain/checkpoints/simclr_phase2/phase2_seed13_best.pth"

CHANNELS = 3
EPOCHS = 30
AVG_LAST_K = 10
HARD_THR_MONITOR = 0.5

NUM_WORKERS = 8
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Option: use fixed reduced subsets for TRAIN and VAL during HPO
USE_HPO_SUBSET = True
HPO_SUBSET_DIR = "hpo_subsets"
HPO_TRAIN_LIST = os.path.join(HPO_SUBSET_DIR, "hpo_train_files.txt")
HPO_VAL_LIST   = os.path.join(HPO_SUBSET_DIR, "hpo_val_files.txt")

# Subset fractions (only used if lists do not exist yet)
HPO_TRAIN_FRAC = 0.20
HPO_VAL_FRAC   = 0.50

# Dataset mask naming (must match SolarPanelDataset defaults)
MASK_SUFFIX = "_label"
MASK_EXT = ".png"
IMG_EXTS = (".png", ".jpg", ".jpeg")

set_global_seed(42, deterministic=False)


# -----------------------
# Encoder/decoder param split
# -----------------------
ENC_PREFIXES = ("block1", "dsconv2", "dsconv3", "dilconv4", "dilconv5")


def split_encoder_decoder_params(model):
    enc_params, dec_params = [], []
    for name, p in model.named_parameters():
        if name.startswith(ENC_PREFIXES):
            enc_params.append(p)
        else:
            dec_params.append(p)
    return enc_params, dec_params


# -----------------------
# Subset list utilities (SimCLR-style)
# -----------------------
def read_list(path: str):
    if not os.path.isfile(path):
        raise RuntimeError(f"Subset list not found: {path}")
    with open(path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    if len(files) == 0:
        raise RuntimeError(f"Empty subset list: {path}")
    return files


def _list_images(data_dir: str):
    mask_tail = f"{MASK_SUFFIX}{MASK_EXT}".lower()
    imgs = []
    for f in os.listdir(data_dir):
        fl = f.lower()
        if fl.endswith(IMG_EXTS) and (not fl.endswith(mask_tail)):
            imgs.append(f)
    imgs.sort()
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in: {data_dir}")
    return imgs


def _mask_path_for(img_name: str, data_dir: str) -> str:
    stem, _ = os.path.splitext(img_name)
    return os.path.join(data_dir, f"{stem}{MASK_SUFFIX}{MASK_EXT}")


def _is_positive_mask(mask_path: str) -> bool:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    return bool((m > 0).any())


def _stratified_subsample(data_dir: str, frac: float, seed: int):
    assert 0.0 < frac <= 1.0
    rng = random.Random(seed)

    imgs = _list_images(data_dir)
    pos, neg = [], []

    for name in imgs:
        mp = _mask_path_for(name, data_dir)
        if not os.path.isfile(mp):
            raise RuntimeError(f"Missing mask for {name}: expected {mp}")
        (pos if _is_positive_mask(mp) else neg).append(name)

    n_total = int(round(frac * len(imgs)))
    pos_ratio = len(pos) / max(1, len(imgs))
    n_pos = int(round(n_total * pos_ratio))
    n_neg = n_total - n_pos

    rng.shuffle(pos)
    rng.shuffle(neg)

    chosen = pos[:min(n_pos, len(pos))] + neg[:min(n_neg, len(neg))]
    rng.shuffle(chosen)

    stats = {
        "total": len(imgs),
        "pos": len(pos),
        "neg": len(neg),
        "frac": frac,
        "chosen_total": len(chosen),
        "chosen_pos": sum(1 for x in chosen if x in set(pos)),
        "chosen_neg": sum(1 for x in chosen if x in set(neg)),
    }
    return chosen, stats


def _write_list(path: str, files):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for name in files:
            f.write(name + "\n")


def ensure_hpo_lists():
    """
    SimCLR-style: ensure fixed subset lists exist.
    If missing, generate them once (stratified by mask positivity) and reuse.
    """
    if not USE_HPO_SUBSET:
        return

    os.makedirs(HPO_SUBSET_DIR, exist_ok=True)

    have_train = os.path.isfile(HPO_TRAIN_LIST)
    have_val = os.path.isfile(HPO_VAL_LIST)

    if have_train and have_val:
        return

    train_dir = os.path.join(DATA_ROOT, TRAIN_SPLIT)
    val_dir   = os.path.join(DATA_ROOT, VAL_SPLIT)

    train_files, train_stats = _stratified_subsample(train_dir, HPO_TRAIN_FRAC, seed=42)
    val_files,   val_stats   = _stratified_subsample(val_dir,   HPO_VAL_FRAC,   seed=43)

    _write_list(HPO_TRAIN_LIST, train_files)
    _write_list(HPO_VAL_LIST,   val_files)

    print("✅ Generated HPO subset lists:")
    print("  ", HPO_TRAIN_LIST, train_stats)
    print("  ", HPO_VAL_LIST,   val_stats)


# -----------------------
# Search spaces
# -----------------------
def build_model_params(trial):
    mini_aspp = trial.suggest_categorical("mini_aspp", [True, False])
    mini_aspp_gpool = trial.suggest_categorical("mini_aspp_gpool", [True, False]) if mini_aspp else False

    use_sa = trial.suggest_categorical("use_sa", [True, False])
    sa_window_size = trial.suggest_categorical("sa_window_size", [8, 16, 32]) if use_sa else 16
    sa_dropout = trial.suggest_categorical("sa_dropout", [0.0, 0.05, 0.10]) if use_sa else 0.0

    return {
        "enc_channels": [16, 16, 32, 32, 64],
        "enc_kernel_sizes": [3, 3, 3, 3, 3],
        "enc_strides": [1, 2, 2, 1, 1],
        "dilations": [2, 4],

        "dec_channels": [32, 16, 16],
        "dec_kernel_sizes": [3, 3],
        "dec_strides": [1, 1],
        "upscale": [2, 2],

        "mini_aspp": mini_aspp,
        "mini_aspp_gpool": mini_aspp_gpool,

        "use_sa": use_sa,
        "sa_windowed": True,
        "sa_window_size": sa_window_size,
        "sa_shifted": True,
        "sa_heads": 4,
        "sa_dropout": sa_dropout,
    }


def build_loss(trial):
    loss_name = trial.suggest_categorical("loss", ["BCEDiceLoss", "BCEDiceFocalLoss"])

    if loss_name == "BCEDiceLoss":
        bce_w = trial.suggest_categorical("bce_w", [0.3, 0.4, 0.5])
        dice_w = 1.0 - bce_w
        return loss_name, get_loss_function("BCEDiceLoss", bce_weight=bce_w, dice_weight=dice_w)

    bce_w = trial.suggest_categorical("bce_w", [0.3, 0.4, 0.5])
    dice_w = trial.suggest_categorical("dice_w", [0.1, 0.2, 0.3])
    focal_w = 1.0 - (bce_w + dice_w)
    if focal_w <= 0:
        raise optuna.TrialPruned()

    alpha = trial.suggest_categorical("alpha_focal", [0.25, 0.5, 0.75])
    gamma = trial.suggest_categorical("gamma_focal", [2.0, 3.0, 4.0])

    return loss_name, get_loss_function(
        "BCEDiceFocalLoss",
        bce_weight=bce_w,
        dice_weight=dice_w,
        focal_weight=focal_w,
        alpha_focal=alpha,
        gamma_focal=gamma,
    )


def build_loaders(bs: int):
    train_dir = os.path.join(DATA_ROOT, TRAIN_SPLIT)
    val_dir   = os.path.join(DATA_ROOT, VAL_SPLIT)

    if USE_HPO_SUBSET:
        train_files = read_list(HPO_TRAIN_LIST)
        val_files   = read_list(HPO_VAL_LIST)
    else:
        train_files = None
        val_files = None

    train_ds = SolarPanelDataset(train_dir, mode="train", files=train_files, return_extra=False)
    val_ds   = SolarPanelDataset(val_dir,   mode="valid", files=val_files,   return_extra=False)

    pw = PERSISTENT_WORKERS and (NUM_WORKERS > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=pw,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=pw,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    return train_loader, val_loader


# -----------------------
# Optuna objective
# -----------------------
def objective(trial: optuna.Trial) -> float:
    bs = trial.suggest_categorical("batch_size", [8, 16, 32])
    base_lr = trial.suggest_float("base_lr", 1e-4, 1e-2, log=True)
    enc_lr_mult = trial.suggest_float("enc_lr_mult", 0.01, 0.30, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model_params = build_model_params(trial)
    loss_name, criterion = build_loss(trial)

    train_loader, val_loader = build_loaders(bs)

    model = UltraLightFCN(in_channels=CHANNELS, num_classes=1, params=model_params)
    load_phase2_encoder_into_ultralight(model, PHASE2_CKPT, verbose=(trial.number == 0))
    model = model.to(DEVICE)

    use_amp = (DEVICE.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    enc_params, dec_params = split_encoder_decoder_params(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": base_lr * enc_lr_mult},
            {"params": dec_params, "lr": base_lr},
        ],
        lr=base_lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )

    last_k = deque(maxlen=AVG_LAST_K)
    best_avg_last_k = -1.0
    best_epoch = -1
    best_val_soft = -1.0
    best_val_hard05 = -1.0

    for epoch in range(EPOCHS):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"[trial {trial.number}] Train {epoch+1}/{EPOCHS}", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        soft_sum = 0.0
        hard05_sum = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks  = masks.to(DEVICE, non_blocking=True)

                with autocast(device_type="cuda", enabled=use_amp):
                    logits = model(images)

                soft_sum += float(calculate_dice(logits, masks, thr=None))
                hard05_sum += float(calculate_dice(logits, masks, thr=HARD_THR_MONITOR))

        val_soft = soft_sum / max(1, len(val_loader))
        val_hard05 = hard05_sum / max(1, len(val_loader))

        scheduler.step(val_soft)

        last_k.append(val_soft)
        avg_last_k = float(sum(last_k) / len(last_k))

        if avg_last_k > best_avg_last_k:
            best_avg_last_k = avg_last_k
            best_epoch = epoch + 1
            best_val_soft = val_soft
            best_val_hard05 = val_hard05

        report_metric = avg_last_k if len(last_k) >= 3 else val_soft
        trial.report(report_metric, step=epoch + 1)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("phase2_ckpt", PHASE2_CKPT)
    trial.set_user_attr("avg_last_k", AVG_LAST_K)
    trial.set_user_attr("selection_metric", "avg_last_k_soft")
    trial.set_user_attr("best_epoch", int(best_epoch))
    trial.set_user_attr("best_val_soft", float(best_val_soft))
    trial.set_user_attr("best_val_hard05", float(best_val_hard05))
    trial.set_user_attr("loss_name", str(loss_name))
    trial.set_user_attr("use_hpo_subset", bool(USE_HPO_SUBSET))
    if USE_HPO_SUBSET:
        trial.set_user_attr("hpo_train_list", HPO_TRAIN_LIST)
        trial.set_user_attr("hpo_val_list", HPO_VAL_LIST)
        trial.set_user_attr("hpo_train_frac", float(HPO_TRAIN_FRAC))
        trial.set_user_attr("hpo_val_frac", float(HPO_VAL_FRAC))

    return float(best_avg_last_k)


# -----------------------
# Main
# -----------------------
def main():
    # Ensure fixed subset lists exist (SimCLR-style)
    ensure_hpo_lists()

    study = optuna.create_study(
        direction="maximize",
        study_name="UltraLightFCN_seg_finetune_softdice_RGB",
        storage="sqlite:///UltraLightFCN_study.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=8,
            n_min_trials=10,
        ),
    )

    study.optimize(
        objective,
        n_trials=200,
        callbacks=[clear_cuda_cache],
    )

    print("\n📈 Best trial (maximize best avg_last_k soft dice):")
    print("  Best value:", study.best_value)
    for k, v in study.best_trial.params.items():
        print(f"  {k:18s}: {v}")


if __name__ == "__main__":
    main()
