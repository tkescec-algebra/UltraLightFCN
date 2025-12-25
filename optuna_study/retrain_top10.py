"""
Segmentation Top-10 Retrain (FULL dataset) after Seg-HPO — CSV logging (BEST-only)

What it does:
- Loads COMPLETE Optuna trials from sqlite DB (your Seg HPO study)
- Selects Top-K by Optuna objective value (maximize best_avg_last_k soft Dice)
- Retrains each candidate on FULL TRAIN + FULL VALID with FIXED seed
- Uses same selection logic as HPO:
    selection_metric = avg_last_k of soft Dice (thr=None), K=AVG_LAST_K
- Saves:
    - best.pth (best avg_last_k only)
    - retrain_top10_results.csv (incremental write each candidate)
    - retrain_top10_ranking.csv (final ranking after retrain)

Notes:
- Uses SolarPanelDataset on FULL data (files=None)
- Uses Phase2 encoder init (PHASE2_CKPT) same as HPO
"""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import Dict, Any, List
from collections import deque

import torch
import optuna
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from models.UltraLightFCN_base import UltraLightFCN
from utils.helpers import get_loss_function, clear_cuda_cache
from utils.load_simclr_pretrain_encoder import load_phase2_encoder_into_ultralight
from utils.metrics import calculate_dice
from utils.repro import set_global_seed, GLOBAL_SEED
from utils.dataset import SolarPanelDataset


# -----------------------------
# Config (match HPO)
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STORAGE = "sqlite:///UltraLightFCN_study.db"
STUDY_NAME = "UltraLightFCN_seg_finetune_softdice_RGB"

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

TOPK = 10

OUT_DIR = Path("checkpoints/seg_top10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "retrain_top10_results.csv"
RANKING_CSV_PATH = OUT_DIR / "retrain_top10_ranking.csv"


# -----------------------------
# Reproducibility (fixed seed across candidates)
# -----------------------------
set_global_seed(GLOBAL_SEED, deterministic=False)


# -----------------------------
# Encoder/decoder param split (same as HPO/ablation)
# -----------------------------
ENC_PREFIXES = ("block1", "dsconv2", "dsconv3", "dilconv4", "dilconv5")


def split_encoder_decoder_params(model: torch.nn.Module):
    enc_params, dec_params = [], []
    for name, p in model.named_parameters():
        if name.startswith(ENC_PREFIXES):
            enc_params.append(p)
        else:
            dec_params.append(p)
    return enc_params, dec_params


# -----------------------------
# Build model/loss from Optuna params (must match HPO)
# -----------------------------
def build_model_params_from_optuna(params: Dict[str, Any]) -> Dict[str, Any]:
    mini_aspp = bool(params["mini_aspp"])
    mini_aspp_gpool = bool(params.get("mini_aspp_gpool", False)) if mini_aspp else False

    use_sa = bool(params["use_sa"])
    sa_window_size = int(params.get("sa_window_size", 16)) if use_sa else 16
    sa_dropout = float(params.get("sa_dropout", 0.0)) if use_sa else 0.0

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


def build_loss_from_optuna(params: Dict[str, Any]):
    loss_name = str(params["loss"])

    if loss_name == "BCEDiceLoss":
        bce_w = float(params["bce_w"])
        dice_w = 1.0 - bce_w
        crit = get_loss_function("BCEDiceLoss", bce_weight=bce_w, dice_weight=dice_w)
        return loss_name, crit

    if loss_name == "BCEDiceFocalLoss":
        bce_w = float(params["bce_w"])
        dice_w = float(params["dice_w"])
        focal_w = 1.0 - (bce_w + dice_w)
        if focal_w <= 0:
            raise RuntimeError(f"Invalid focal_w={focal_w:.3f} for params={params}")

        alpha = float(params["alpha_focal"])
        gamma = float(params["gamma_focal"])

        crit = get_loss_function(
            "BCEDiceFocalLoss",
            bce_weight=bce_w,
            dice_weight=dice_w,
            focal_weight=focal_w,
            alpha_focal=alpha,
            gamma_focal=gamma,
        )
        return loss_name, crit

    raise RuntimeError(f"Unknown loss: {loss_name}")


# -----------------------------
# Full loaders (no subset lists)
# -----------------------------
def build_full_loaders(bs: int):
    train_dir = os.path.join(DATA_ROOT, TRAIN_SPLIT)
    val_dir = os.path.join(DATA_ROOT, VAL_SPLIT)

    train_ds = SolarPanelDataset(train_dir, mode="train", files=None, return_extra=False)
    val_ds = SolarPanelDataset(val_dir, mode="valid", files=None, return_extra=False)

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


# -----------------------------
# One retrain run (BEST-only)
# -----------------------------
def retrain_one_candidate(
    rank: int,
    trial_number: int,
    optuna_value: float,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Retrain one candidate on FULL dataset with fixed seed.
    Saves ONLY best checkpoint by avg_last_k soft Dice.
    Returns a dict row for CSV logging.
    """
    bs = int(params["batch_size"])
    base_lr = float(params["base_lr"])
    enc_lr_mult = float(params["enc_lr_mult"])
    weight_decay = float(params["weight_decay"])

    model_params = build_model_params_from_optuna(params)
    loss_name, criterion = build_loss_from_optuna(params)

    train_loader, val_loader = build_full_loaders(bs)

    model = UltraLightFCN(in_channels=CHANNELS, num_classes=1, params=model_params)
    load_phase2_encoder_into_ultralight(model, PHASE2_CKPT, verbose=(rank == 1))
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

    run_dir = OUT_DIR / f"rank{rank:02d}_trial{trial_number:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = str(run_dir / "best.pth")

    last_k = deque(maxlen=AVG_LAST_K)

    best_avg_last_k = -1.0
    best_epoch = -1
    best_val_soft = -1.0
    best_val_hard05 = -1.0

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        for images, masks in tqdm(
            train_loader,
            desc=f"[Seg Top{TOPK}] Rank{rank:02d} Trial{trial_number:04d} Train {epoch+1}/{EPOCHS}",
            leave=False,
        ):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ---- Validation (soft selection + hard monitor) ----
        model.eval()
        soft_sum = 0.0
        hard05_sum = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                with autocast(device_type="cuda", enabled=use_amp):
                    logits = model(images)

                soft_sum += float(calculate_dice(logits, masks, thr=None))
                hard05_sum += float(calculate_dice(logits, masks, thr=HARD_THR_MONITOR))

        val_soft = soft_sum / max(1, len(val_loader))
        val_hard05 = hard05_sum / max(1, len(val_loader))

        scheduler.step(val_soft)

        last_k.append(val_soft)
        avg_last_k = float(sum(last_k) / len(last_k))

        # Save best checkpoint by avg_last_k
        if avg_last_k > best_avg_last_k:
            best_avg_last_k = avg_last_k
            best_epoch = epoch + 1
            best_val_soft = val_soft
            best_val_hard05 = val_hard05

            torch.save(
                {
                    "epoch": int(epoch + 1),
                    "model": model.state_dict(),
                    "params": dict(params),
                    "phase2_ckpt": PHASE2_CKPT,
                    "metrics": {
                        "val_soft": float(val_soft),
                        "val_hard05": float(val_hard05),
                        "avg_last_k_soft": float(avg_last_k),
                        "avg_last_k_k": int(AVG_LAST_K),
                    },
                    "source": {
                        "study_name": STUDY_NAME,
                        "trial_number": int(trial_number),
                        "optuna_value": float(optuna_value),
                        "rank": int(rank),
                        "seed": int(GLOBAL_SEED),
                    },
                },
                best_ckpt_path,
            )

        print(
            f"[Rank{rank:02d} Trial{trial_number:04d}] "
            f"Epoch {epoch+1:02d}/{EPOCHS} | val_soft={val_soft:.4f} | avg_last_k={avg_last_k:.4f}"
        )

    clear_cuda_cache()

    # CSV row (no last checkpoint)
    row: Dict[str, Any] = {
        # identifiers
        "rank": int(rank),
        "trial_number": int(trial_number),
        "optuna_value": float(optuna_value),
        "seed": int(GLOBAL_SEED),

        # retrain selection outcome
        "best_avg_last_k_soft": float(best_avg_last_k),
        "best_epoch": int(best_epoch),
        "best_val_soft": float(best_val_soft),
        "best_val_hard05": float(best_val_hard05),

        # checkpoint paths
        "best_ckpt_path": best_ckpt_path,

        # training HPs
        "batch_size": int(bs),
        "base_lr": float(base_lr),
        "enc_lr_mult": float(enc_lr_mult),
        "weight_decay": float(weight_decay),

        # model knobs
        "mini_aspp": bool(params["mini_aspp"]),
        "mini_aspp_gpool": bool(params.get("mini_aspp_gpool", False)),
        "use_sa": bool(params["use_sa"]),
        "sa_window_size": int(params.get("sa_window_size", 16)),
        "sa_dropout": float(params.get("sa_dropout", 0.0)),

        # loss knobs
        "loss": str(params["loss"]),
        "bce_w": float(params.get("bce_w", "")) if "bce_w" in params else "",
        "dice_w": float(params.get("dice_w", "")) if "dice_w" in params else "",
        "alpha_focal": float(params.get("alpha_focal", "")) if "alpha_focal" in params else "",
        "gamma_focal": float(params.get("gamma_focal", "")) if "gamma_focal" in params else "",
        "loss_name_reconstructed": str(loss_name),

        # metadata
        "phase2_ckpt": PHASE2_CKPT,
        "study_name": STUDY_NAME,
        "storage": STORAGE,
    }

    return row


# -----------------------------
# Optuna: load Top-K COMPLETE trials
# -----------------------------
def load_topk_trials(study: optuna.Study, k: int) -> List[optuna.trial.FrozenTrial]:
    complete = [t for t in study.trials if (t.value is not None and t.state == optuna.trial.TrialState.COMPLETE)]
    complete.sort(key=lambda t: float(t.value), reverse=True)  # maximize
    return complete[:k]


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    top_trials = load_topk_trials(study, TOPK)

    print(f"Loaded {len(study.trials)} trials, {len(top_trials)} top COMPLETE candidates (TOPK={TOPK}).")

    fieldnames = [
        # identifiers
        "rank", "trial_number", "optuna_value", "seed",
        # outcome
        "best_avg_last_k_soft", "best_epoch", "best_val_soft", "best_val_hard05",
        # checkpoint
        "best_ckpt_path",
        # train HPs
        "batch_size", "base_lr", "enc_lr_mult", "weight_decay",
        # model HPs
        "mini_aspp", "mini_aspp_gpool", "use_sa", "sa_window_size", "sa_dropout",
        # loss HPs
        "loss", "bce_w", "dice_w", "alpha_focal", "gamma_focal", "loss_name_reconstructed",
        # meta
        "phase2_ckpt", "study_name", "storage",
    ]

    rows: List[Dict[str, Any]] = []

    for rank, t in enumerate(top_trials, start=1):
        print(f"\n=== Candidate #{rank}/{TOPK}: trial {t.number} (optuna={t.value:.6f}) ===")
        row = retrain_one_candidate(
            rank=rank,
            trial_number=int(t.number),
            optuna_value=float(t.value),
            params=dict(t.params),
        )
        rows.append(row)

        # Incremental CSV write (safe if interrupted)
        write_csv(CSV_PATH, rows, fieldnames)

        # Current best after retrain
        best = max(rows, key=lambda r: r["best_avg_last_k_soft"])
        print(
            f"[Current best after retrain] trial {best['trial_number']} "
            f"avg_last_k_soft={best['best_avg_last_k_soft']:.4f}"
        )

    # Final ranking by retrain metric
    rows_sorted = sorted(rows, key=lambda r: r["best_avg_last_k_soft"], reverse=True)
    write_csv(RANKING_CSV_PATH, rows_sorted, fieldnames)

    print("\n=== Final ranking by retrain best_avg_last_k_soft ===")
    for i, r in enumerate(rows_sorted, start=1):
        print(
            f"#{i:2d} trial {r['trial_number']:4d}  "
            f"retrain_avg_last_k={r['best_avg_last_k_soft']:.4f}  optuna={r['optuna_value']:.6f}"
        )

    print(f"\n[CSV] incremental results: {CSV_PATH}")
    print(f"[CSV] final ranking:       {RANKING_CSV_PATH}")


if __name__ == "__main__":
    main()
