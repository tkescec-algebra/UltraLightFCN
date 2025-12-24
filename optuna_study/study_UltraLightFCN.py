"""
Complete HPO script (finetune-only) aligned with methodology:
    - Phase2 encoder init (fixed)
    - Nondeterministic HPO (fast), deterministic VAL via transforms (valid mode)
    - Selection metric: soft Dice (thr=None)
    - Best checkpoint chosen by max avg_last_k (K=10)
    - Save only "last" and "best"
    - MedianPruner
    - Logs soft + hard@0.5 Dice (hard only for monitoring, not selection)
"""

import os
from collections import deque

import torch
import optuna
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast

from models.UltraLightFCN_base import UltraLightFCN
from utils.helpers import get_loss_function, clear_cuda_cache
from utils.load_simclr_pretrain_encoder import load_phase2_encoder_into_ultralight
from utils.metrics import calculate_dice
from utils.repro import set_global_seed

from utils.dataset import SolarPanelDataset  # updated dataset (SimCLR-style listing, sorted, return_extra)



# -----------------------
# Global config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/workspace/UltraLightFCN_snakemake/dataset"
TRAIN_SPLIT = "train"
VAL_SPLIT = "valid"

# Choose ONE Phase2 "best" checkpoint as fixed init for HPO
PHASE2_CKPT = "/workspace/UltraLightFCN/pretrain/checkpoints/simclr_phase2/phase2_seed13_best.pth"

CHANNELS = 3
EPOCHS = 30

# Selection stability
AVG_LAST_K = 10

# Hard Dice threshold only for monitoring (not for selection)
HARD_THR_MONITOR = 0.5

# DataLoader speed knobs (HPO is nondeterministic by design)
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# HPO seeding (sampling consistency); training itself remains nondeterministic
set_global_seed(42, deterministic=False)


# -----------------------
# Encoder/decoder param split (same logic as ablation)
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


def build_model_params(trial):
    """
    Architecture knobs to tune (aligned with your ablation variants).
    Keep encoder/decoder topology fixed; tune only context/attention switches.
    """
    mini_aspp = trial.suggest_categorical("mini_aspp", [True, False])
    mini_aspp_gpool = trial.suggest_categorical("mini_aspp_gpool", [True, False]) if mini_aspp else False

    use_sa = trial.suggest_categorical("use_sa", [True, False])
    sa_window_size = trial.suggest_categorical("sa_window_size", [8, 16, 32]) if use_sa else 16

    # Tune attention dropout (small discrete space)
    sa_dropout = trial.suggest_categorical("sa_dropout", [0.0, 0.05, 0.10]) if use_sa else 0.0

    return {
        # Encoder configuration (fixed)
        "enc_channels": [16, 16, 32, 32, 64],
        "enc_kernel_sizes": [3, 3, 3, 3, 3],
        "enc_strides": [1, 2, 2, 1, 1],
        "dilations": [2, 4],

        # Decoder configuration (fixed)
        "dec_channels": [32, 16, 16],
        "dec_kernel_sizes": [3, 3],
        "dec_strides": [1, 1],
        "upscale": [2, 2],

        # Context (tuned)
        "mini_aspp": mini_aspp,
        "mini_aspp_gpool": mini_aspp_gpool,

        # Attention (tuned except heads hardcoded)
        "use_sa": use_sa,
        "sa_windowed": True,
        "sa_window_size": sa_window_size,
        "sa_shifted": True,
        "sa_heads": 4,           # hardcoded as agreed
        "sa_dropout": sa_dropout,
    }


def build_loss(trial):
    """
    Controlled loss search space (kept small to reduce val overfitting).
    """
    loss_name = trial.suggest_categorical("loss", ["BCEDiceLoss", "BCEDiceFocalLoss"])

    if loss_name == "BCEDiceLoss":
        # Stable, small discrete weight space
        bce_w = trial.suggest_categorical("bce_w", [0.3, 0.4, 0.5])
        dice_w = 1.0 - bce_w
        return loss_name, get_loss_function("BCEDiceLoss", bce_weight=bce_w, dice_weight=dice_w)

    # BCEDiceFocalLoss
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


def build_loaders(bs):
    """
    Fast (nondeterministic) HPO loaders.
    Validation determinism is ensured by transforms (mode='valid' has no random aug).
    """
    train_ds = SolarPanelDataset(os.path.join(DATA_ROOT, TRAIN_SPLIT), mode="train", return_extra=False)
    val_ds = SolarPanelDataset(os.path.join(DATA_ROOT, VAL_SPLIT), mode="valid", return_extra=False)

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


def objective(trial: optuna.Trial) -> float:
    # -----------------------
    # Training HP search space
    # -----------------------
    bs = trial.suggest_categorical("batch_size", [8, 16])
    base_lr = trial.suggest_float("base_lr", 1e-4, 5e-3, log=True)
    enc_lr_mult = trial.suggest_float("enc_lr_mult", 0.01, 0.30, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model_params = build_model_params(trial)
    loss_name, criterion = build_loss(trial)

    train_loader, val_loader = build_loaders(bs)

    # -----------------------
    # Model init + Phase2 encoder load
    # -----------------------
    model = UltraLightFCN(in_channels=CHANNELS, num_classes=1, params=model_params)

    # Fixed init from Phase2
    load_phase2_encoder_into_ultralight(model, PHASE2_CKPT, verbose=(trial.number == 0))

    # Move to device
    model = model.to(DEVICE)

    # Finetune-only optimizer: encoder smaller LR, decoder base LR
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

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # -----------------------
    # Checkpointing (best/last) by soft avg_last_k
    # -----------------------
    ckpt_dir = os.path.join(
        "optuna_outputs",
        "seg_finetune_softdice",
        f"trial_{trial.number:04d}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pth")
    last_path = os.path.join(ckpt_dir, "last.pth")

    last_k = deque(maxlen=AVG_LAST_K)
    best_avg_last_k = -1.0

    # -----------------------
    # Train loop
    # -----------------------
    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        for images, masks in tqdm(
            train_loader,
            desc=f"[trial {trial.number}] Train {epoch+1}/{EPOCHS}",
            leave=False,
        ):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ---- Validation (soft Dice for selection; hard Dice@0.5 for monitoring) ----
        model.eval()
        soft_sum = 0.0
        hard05_sum = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
                    logits = model(images)

                soft_sum += float(calculate_dice(logits, masks, thr=None))
                hard05_sum += float(calculate_dice(logits, masks, thr=HARD_THR_MONITOR))

        val_soft = soft_sum / max(1, len(val_loader))
        val_hard05 = hard05_sum / max(1, len(val_loader))

        scheduler.step(val_soft)

        last_k.append(val_soft)
        avg_last_k = float(sum(last_k) / len(last_k))

        # Save last checkpoint (overwrite every epoch)
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "trial_params": dict(trial.params),
                "metrics": {
                    "val_soft_dice": val_soft,
                    "val_hard_dice@0.5": val_hard05,
                    "avg_last_k_soft": avg_last_k,
                    "avg_last_k_k": AVG_LAST_K,
                },
                "hpo": {
                    "selection_metric": "avg_last_k_soft_dice",
                    "epochs": EPOCHS,
                    "loss_name": loss_name,
                    "phase2_ckpt": PHASE2_CKPT,
                },
            },
            last_path,
        )

        # Save best by avg_last_k (soft)
        if avg_last_k > best_avg_last_k:
            best_avg_last_k = avg_last_k
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "trial_params": dict(trial.params),
                    "metrics": {
                        "val_soft_dice": val_soft,
                        "val_hard_dice@0.5": val_hard05,
                        "avg_last_k_soft": avg_last_k,
                        "avg_last_k_k": AVG_LAST_K,
                    },
                    "hpo": {
                        "selection_metric": "avg_last_k_soft_dice",
                        "epochs": EPOCHS,
                        "loss_name": loss_name,
                        "phase2_ckpt": PHASE2_CKPT,
                    },
                },
                best_path,
            )

        # ---- Optuna report / prune ----
        # Start pruning only after a warmup window to reduce noise sensitivity.
        # We report avg_last_k once we have a few points; otherwise report val_soft.
        report_metric = avg_last_k if len(last_k) >= 3 else val_soft
        trial.report(report_metric, step=epoch + 1)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_avg_last_k


def main():
    os.makedirs("optuna_outputs", exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"UltraLightFCN_seg_finetune_softdice_RGB",
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
