"""phase4_seg_study_UltraLightFCN_no_simclr.py

Phase 4 (ablation) — Segmentation HPO screening without SimCLR initialization.

Controlled relative to the main Phase-4 script:
  - Same UltraLightFCN architecture via SEG_PARAMS.
  - Same TRAIN subset logic, full VALID policy, losses, optimizer split, scheduler, metrics, and objective.
  - Same dataset code, transforms, and reproducibility controls.

Scientific difference:
  - Encoder is initialized from scratch/random init.
  - No Phase-3 SimCLR checkpoint is loaded anywhere in this script.
"""

from __future__ import annotations

import os
import platform
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import optuna
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import SEG_PARAMS
from models.UltraLightFCN_base import UltraLightFCN
from utils.dataset import SolarPanelDataset
from utils.helpers import clear_cuda_cache, get_loss_function, split_encoder_decoder_params
from utils.metrics import calculate_dice
from utils.no_simclr_guard import (
    NO_SIMCLR_ABLATION,
    NO_SIMCLR_ENCODER_INIT,
    NO_SIMCLR_PRETRAINING,
    assert_no_pretrained_checkpoint,
    build_no_simclr_metadata,
)
from utils.repro import seed_worker, set_global_seed, GLOBAL_SEED


@dataclass(frozen=True)
class Phase4NoSimCLRConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root: str = "../dataset"
    train_split: str = "train"
    val_split: str = "valid"

    # Explicitly forbidden in this branch. Kept only as a guard surface.
    phase3_last_ckpt: str | None = None

    in_channels: int = 3
    num_classes: int = 1
    seg_params: dict = field(default_factory=lambda: dict(SEG_PARAMS))

    epochs: int = 30
    avg_last_k: int = 10
    hard_thr_monitor: float = 0.5

    num_workers: int = 8
    pin_memory: bool = True
    drop_last_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    global_seed: int = GLOBAL_SEED
    deterministic: bool = False
    strict: bool = False

    use_hpo_subset: bool = True
    use_hpo_val_subset: bool = False
    hpo_subset_dir: str = "runs/hpo_subsets"
    hpo_train_list: str = os.path.join(hpo_subset_dir, "hpo_train_files.txt")
    hpo_val_list: str = os.path.join(hpo_subset_dir, "hpo_val_files.txt")
    hpo_train_frac: float = 0.20
    hpo_val_frac: float = 0.50
    subset_seed_train: int = 42
    subset_seed_val: int = 43

    mask_suffix: str = "_label"
    mask_ext: str = ".png"
    img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")

    study_name: str = "UltraLightFCN_seg_softdice_no_simclr"
    storage: str = "sqlite:///UltraLightFCN_study.db"
    n_trials: int = 100
    sampler_seed: int = 123

    pruner_warmup_steps: int = 8
    pruner_min_trials: int = 10

    pretraining: str = NO_SIMCLR_PRETRAINING
    encoder_init: str = NO_SIMCLR_ENCODER_INIT
    ablation: str = NO_SIMCLR_ABLATION


def _resolve_dataloader_runtime(cfg: Phase4NoSimCLRConfig) -> tuple[int, bool, int | None]:
    # Windows multiprocessing compatibility safeguard:
    # spawned workers re-import torch and can hit DLL init failures, so force single-process loading.
    if platform.system().lower().startswith("win"):
        return 0, False, None
    return cfg.num_workers, cfg.persistent_workers, cfg.prefetch_factor


def _read_list(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise RuntimeError(f"Subset list not found: {path}")
    with open(path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    if len(files) == 0:
        raise RuntimeError(f"Empty subset list: {path}")
    return files


def _list_images(cfg: Phase4NoSimCLRConfig, data_dir: str) -> List[str]:
    mask_tail = f"{cfg.mask_suffix}{cfg.mask_ext}".lower()
    imgs: List[str] = []
    for f in os.listdir(data_dir):
        fl = f.lower()
        if fl.endswith(cfg.img_exts) and (not fl.endswith(mask_tail)):
            imgs.append(f)
    imgs.sort()
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in: {data_dir}")
    return imgs


def _mask_path_for(cfg: Phase4NoSimCLRConfig, img_name: str, data_dir: str) -> str:
    stem, _ = os.path.splitext(img_name)
    return os.path.join(data_dir, f"{stem}{cfg.mask_suffix}{cfg.mask_ext}")


def _is_positive_mask(mask_path: str) -> bool:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    return bool((m > 0).any())


def _stratified_subsample(cfg: Phase4NoSimCLRConfig, data_dir: str, frac: float, seed: int):
    assert 0.0 < frac <= 1.0
    rng = random.Random(seed)

    imgs = _list_images(cfg, data_dir)
    pos, neg = [], []

    for name in imgs:
        mp = _mask_path_for(cfg, name, data_dir)
        if not os.path.isfile(mp):
            raise RuntimeError(f"Missing mask for {name}: expected {mp}")
        (pos if _is_positive_mask(mp) else neg).append(name)

    n_total = int(round(frac * len(imgs)))
    pos_ratio = len(pos) / max(1, len(imgs))
    n_pos = int(round(n_total * pos_ratio))
    n_neg = n_total - n_pos

    rng.shuffle(pos)
    rng.shuffle(neg)

    chosen = pos[: min(n_pos, len(pos))] + neg[: min(n_neg, len(neg))]
    rng.shuffle(chosen)

    pos_set = set(pos)
    stats = {
        "total": len(imgs),
        "pos": len(pos),
        "neg": len(neg),
        "frac": frac,
        "chosen_total": len(chosen),
        "chosen_pos": sum(1 for x in chosen if x in pos_set),
        "chosen_neg": sum(1 for x in chosen if x not in pos_set),
    }
    return chosen, stats


def _write_list(path: str, files: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for name in files:
            f.write(name + "\n")


def ensure_hpo_lists(cfg: Phase4NoSimCLRConfig) -> None:
    if not cfg.use_hpo_subset:
        return

    os.makedirs(cfg.hpo_subset_dir, exist_ok=True)

    have_train = os.path.isfile(cfg.hpo_train_list)
    have_val = os.path.isfile(cfg.hpo_val_list)

    if have_train and (have_val or (not cfg.use_hpo_val_subset)):
        return

    train_dir = os.path.join(cfg.data_root, cfg.train_split)
    val_dir = os.path.join(cfg.data_root, cfg.val_split)

    if not have_train:
        train_files, train_stats = _stratified_subsample(
            cfg, train_dir, cfg.hpo_train_frac, seed=cfg.subset_seed_train
        )
        _write_list(cfg.hpo_train_list, train_files)
        print("Generated HPO TRAIN subset list:")
        print("  ", cfg.hpo_train_list, train_stats)

    if cfg.use_hpo_val_subset and (not have_val):
        val_files, val_stats = _stratified_subsample(
            cfg, val_dir, cfg.hpo_val_frac, seed=cfg.subset_seed_val
        )
        _write_list(cfg.hpo_val_list, val_files)
        print("Generated HPO VALID subset list:")
        print("  ", cfg.hpo_val_list, val_stats)


def build_loss(trial: optuna.Trial):
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


def build_loaders(cfg: Phase4NoSimCLRConfig, batch_size: int, *, seed: int):
    train_dir = os.path.join(cfg.data_root, cfg.train_split)
    val_dir = os.path.join(cfg.data_root, cfg.val_split)

    if cfg.use_hpo_subset:
        train_files = _read_list(cfg.hpo_train_list)
    else:
        train_files = None

    if cfg.use_hpo_subset and cfg.use_hpo_val_subset:
        val_files = _read_list(cfg.hpo_val_list)
    else:
        val_files = None

    train_ds = SolarPanelDataset(train_dir, mode="train", files=train_files, return_extra=False)
    val_ds = SolarPanelDataset(val_dir, mode="valid", files=val_files, return_extra=False)

    num_workers, persistent_workers, prefetch_factor = _resolve_dataloader_runtime(cfg)
    pw = persistent_workers and (num_workers > 0)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last_train,
        persistent_workers=pw,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=pw,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    return train_loader, val_loader


def _val_epoch(model: torch.nn.Module, val_loader: DataLoader, cfg: Phase4NoSimCLRConfig, *, use_amp: bool):
    model.eval()
    soft_sum, hard_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)

            bs = int(images.shape[0])
            soft = float(calculate_dice(logits, masks, thr=None))
            hard = float(calculate_dice(logits, masks, thr=cfg.hard_thr_monitor))
            soft_sum += soft * bs
            hard_sum += hard * bs
            n += bs

    if n == 0:
        return 0.0, 0.0
    return soft_sum / n, hard_sum / n


def objective(trial: optuna.Trial, cfg: Phase4NoSimCLRConfig) -> float:
    trial_seed = int(cfg.global_seed + 1000 * trial.number)
    set_global_seed(trial_seed, deterministic=cfg.deterministic, strict=cfg.strict)
    assert_no_pretrained_checkpoint(
        branch_name="phase4_no_simclr",
        phase3_last_ckpt=cfg.phase3_last_ckpt,
        init_metadata=build_no_simclr_metadata(phase=4),
    )

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    base_lr = trial.suggest_float("base_lr", 1e-4, 1e-2, log=True)
    # Random-initialized encoder and decoder should learn under the same base LR.
    # This avoids carrying over the conservative pretrained-encoder LR policy from the SimCLR branch.
    enc_lr_mult = 1.0
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    rlop_factor = trial.suggest_categorical("rlop_factor", [0.3, 0.5, 0.7])
    rlop_patience = trial.suggest_categorical("rlop_patience", [2, 3, 4])

    loss_name, criterion = build_loss(trial)
    train_loader, val_loader = build_loaders(cfg, batch_size, seed=trial_seed)

    # No-SimCLR ablation: random initialization only. Do not load any pretrained checkpoint.
    model = UltraLightFCN(in_channels=cfg.in_channels, num_classes=cfg.num_classes, params=cfg.seg_params)
    model = model.to(cfg.device)

    use_amp = (cfg.device.type == "cuda")
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
        factor=float(rlop_factor),
        patience=int(rlop_patience),
        threshold=1e-4,
        min_lr=1e-6,
    )

    last_k = deque(maxlen=cfg.avg_last_k)
    best_avg_last_k = -1.0
    best_epoch = -1
    best_val_soft = -1.0
    best_val_hard05 = -1.0

    for epoch in range(cfg.epochs):
        model.train()
        for images, masks in tqdm(
            train_loader,
            desc=f"[trial {trial.number}] Train {epoch + 1}/{cfg.epochs}",
            leave=False,
        ):
            images = images.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_soft, val_hard05 = _val_epoch(model, val_loader, cfg, use_amp=use_amp)
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

    use_val_subset = bool(cfg.use_hpo_subset and cfg.use_hpo_val_subset)
    trial.set_user_attr("trial_seed", trial_seed)
    trial.set_user_attr("avg_last_k", cfg.avg_last_k)
    trial.set_user_attr("selection_metric", "avg_last_k_soft")
    trial.set_user_attr("best_epoch", int(best_epoch))
    trial.set_user_attr("best_val_soft", float(best_val_soft))
    trial.set_user_attr("best_val_hard05", float(best_val_hard05))
    trial.set_user_attr("loss_name", str(loss_name))
    trial.set_user_attr("enc_lr_mult", float(enc_lr_mult))
    trial.set_user_attr("enc_lr_policy", "fixed_equal_lr_random_init")
    trial.set_user_attr("pretraining", cfg.pretraining)
    trial.set_user_attr("encoder_init", cfg.encoder_init)
    trial.set_user_attr("ablation", cfg.ablation)
    trial.set_user_attr("phase3_last_ckpt", None)

    trial.set_user_attr("use_hpo_subset", bool(cfg.use_hpo_subset))
    trial.set_user_attr("use_hpo_val_subset", use_val_subset)
    trial.set_user_attr("use_full_valid", not use_val_subset)
    trial.set_user_attr("val_files_mode", "subset_list" if use_val_subset else "full_valid_dir")

    if cfg.use_hpo_subset:
        trial.set_user_attr("hpo_train_list", cfg.hpo_train_list)
        trial.set_user_attr("hpo_train_frac", float(cfg.hpo_train_frac))
        if cfg.use_hpo_val_subset:
            trial.set_user_attr("hpo_val_list", cfg.hpo_val_list)
            trial.set_user_attr("hpo_val_frac", float(cfg.hpo_val_frac))

    return float(best_avg_last_k)


def main() -> None:
    cfg = Phase4NoSimCLRConfig()
    assert_no_pretrained_checkpoint(
        branch_name="phase4_no_simclr",
        phase3_last_ckpt=cfg.phase3_last_ckpt,
        init_metadata=build_no_simclr_metadata(phase=4),
    )

    set_global_seed(cfg.global_seed, deterministic=cfg.deterministic, strict=cfg.strict)
    ensure_hpo_lists(cfg)

    sampler = optuna.samplers.TPESampler(seed=cfg.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=cfg.pruner_warmup_steps,
        n_min_trials=cfg.pruner_min_trials,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=cfg.study_name,
        storage=cfg.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    print(
        f"[no-simclr] study={cfg.study_name} pretraining={cfg.pretraining} "
        f"encoder_init={cfg.encoder_init} ablation={cfg.ablation}"
    )

    study.optimize(
        lambda t: objective(t, cfg),
        n_trials=cfg.n_trials,
        callbacks=[clear_cuda_cache],
    )

    print("\nBest trial (maximize best avg_last_k soft dice):")
    print("  Best value:", study.best_value)
    for k, v in study.best_trial.params.items():
        print(f"  {k:18s}: {v}")


if __name__ == "__main__":
    main()
