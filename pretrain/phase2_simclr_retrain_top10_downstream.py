"""
retrain_top10_downstream_phase2.py

Phase 2:
- Load top-K SimCLR hyperparameter sets from the Phase 1 Optuna study.
- Retrain each candidate SimCLR model on the FULL downstream TRAIN split (80%) with no internal validation.
- Select the best pretrained encoder using a controlled downstream-aware evaluation on the official VALID split (10%):
    * a short "mini segmentation warm-up" (few epochs or a capped number of steps) with a frozen encoder,
      evaluated by soft Dice on VALID.

Notes:
- This script intentionally does NOT use the TEST split.
- All candidates receive the same training budget.
- RNG is reset per candidate + DataLoader generators/workers are seeded for fair comparison.
"""

from __future__ import annotations

import os
import math
import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

from utils.config import ENCODER_PARAMS, SEG_PARAMS
from utils.repro import set_global_seed, GLOBAL_SEED, seed_worker
from utils.loss_functions import NTXentLoss, BCEDiceLoss
from utils.dataset import SimCLRSolarPanelDataset, SolarPanelDataset
from utils.metrics import calculate_dice
from pretrain.utils.metrics_simclr import simclr_alignment, simclr_uniformity
from utils.helpers import estimate_pos_weight_from_masks
from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel
from models.UltraLightFCN_base import UltraLightFCN


# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Phase 1 Optuna study (SimCLR HPO)
STORAGE = "sqlite:////workspace/UltraLightFCN/optuna_study/UltraLightFCN_study.db"
STUDY_NAME = "UltraLightFCN_SimCLR_pretrain_RGB"
TOPK = 10

# Dataset paths
DATA_ROOT = "/workspace/UltraLightFCN/dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")   # downstream TRAIN (80%)
VAL_DIR   = os.path.join(DATA_ROOT, "valid")   # downstream VALID (10%)

# SimCLR retrain budget
SIMCLR_IMAGE_SIZE = 256
SIMCLR_EPOCHS = 40
SIMCLR_BS_DEFAULT = 256
DROP_LAST = True


# SimCLR representation diagnostics (do NOT use for model selection)
SIMCLR_METRIC_BS = 256          # smaller batch for O(B^2) uniformity computation
SIMCLR_METRIC_NUM_BATCHES = 2   # number of batches to estimate metrics (fixed compute)
SIMCLR_UNIFORMITY_T = 2.0

# Output
OUT_DIR = Path("/workspace/UltraLightFCN/pretrain/checkpoints/simclr_topk_retrain_downstream")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mini segmentation warm-up (controlled downstream evaluation)
MINI_SEG_EPOCHS = 5
MINI_SEG_MAX_STEPS = 1200   # cap steps to keep compute fixed across candidates
MINI_SEG_BS = 16
MINI_SEG_LR = 1e-3
MINI_SEG_WD = 1e-4

# Mini warm-up loss settings (should match the main segmentation recipe as closely as possible)
MINI_SEG_BCE_WEIGHT = 0.4
MINI_SEG_DICE_WEIGHT = 0.6

# Freeze policy for mini seg warm-up:
# Freeze the encoder/bottleneck; train only decoder/head to measure representation quality.
ENCODER_PREFIXES = (
    "block1", "dsconv2", "dsconv3", "dilconv4", "dilconv5", "mini_aspp", "sa"
)


# -----------------------------
# Repro helpers
# -----------------------------
def list_images_no_masks(data_dir: str) -> List[str]:
    """
    List image files in a directory, excluding mask files (*_label.*).
    Returns a sorted list for reproducibility.
    """
    exts = (".png", ".jpg", ".jpeg")
    files = []
    for f in os.listdir(data_dir):
        lf = f.lower()
        if not lf.endswith(exts):
            continue
        if "_label" in os.path.splitext(lf)[0]:
            continue
        files.append(f)
    files.sort()
    if not files:
        raise RuntimeError(f"No images found in: {data_dir}")
    return files


def steps_per_epoch(n_samples: int, batch_size: int, drop_last: bool) -> int:
    if n_samples <= 0:
        return 0
    if drop_last:
        return n_samples // batch_size
    return math.ceil(n_samples / batch_size)


# -----------------------------
# SimCLR build / train
# -----------------------------
def build_encoder_and_model(proj_hidden_dim: int, proj_out_dim: int) -> Tuple[UltraLightEncoder, SimCLRModel]:
    encoder = UltraLightEncoder(in_channels=3, params=ENCODER_PARAMS).to(DEVICE)
    proj = ProjectionHead(in_dim=encoder.out_channels, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim).to(DEVICE)
    model = SimCLRModel(encoder, proj).to(DEVICE)
    return encoder, model


def retrain_simclr_candidate(
    trial_number: int,
    optuna_value: float,
    params: Dict[str, Any],
    simclr_train_files: List[str],
) -> Dict[str, Any]:
    """
    Retrain a SimCLR candidate on FULL downstream TRAIN split (no internal val).
    Returns a row dict with checkpoint path and metadata (no downstream metric here).
    """
    # IMPORTANT: fair comparison across candidates
    set_global_seed(GLOBAL_SEED, deterministic=False)
    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    simclr_lr = float(params.get("simclr_lr", 3e-4))
    wd = float(params.get("weight_decay", 1e-4))
    temperature = float(params.get("simclr_temperature", 0.2))
    warmup_ratio = float(params.get("warmup_ratio", 0.05))
    proj_hidden_dim = int(params.get("proj_hidden_dim", 128))
    proj_out_dim = int(params.get("proj_out_dim", 64))
    max_grad_norm = float(params.get("max_grad_norm", 1.0))
    simclr_bs = int(params.get("batch_size", SIMCLR_BS_DEFAULT))

    simclr_train_ds = SimCLRSolarPanelDataset(
        data_dir=TRAIN_DIR,
        image_size=SIMCLR_IMAGE_SIZE,
        files=simclr_train_files,
    )
    n_train = len(simclr_train_ds)

    drop_last_trial = DROP_LAST
    if DROP_LAST and n_train < simclr_bs:
        drop_last_trial = False

    g_train = torch.Generator().manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(
        simclr_train_ds,
        batch_size=simclr_bs,
        shuffle=True,
        pin_memory=True,
        drop_last=drop_last_trial,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g_train,
    )

    # Scheduling in update-steps
    spe = steps_per_epoch(n_train, simclr_bs, drop_last_trial)
    total_steps = SIMCLR_EPOCHS * spe
    if total_steps <= 0:
        raise RuntimeError(f"Invalid total_steps={total_steps} (n_train={n_train}, bs={simclr_bs})")

    warmup_steps = int(warmup_ratio * total_steps)
    warmup_steps = min(max(1, warmup_steps), max(1, total_steps - 1))

    encoder, model = build_encoder_and_model(proj_hidden_dim, proj_out_dim)

    opt = torch.optim.AdamW(model.parameters(), lr=simclr_lr, weight_decay=wd)
    scheduler = CosineLRScheduler(
        optimizer=opt,
        t_initial=total_steps,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    crit = NTXentLoss(temperature=temperature, device=DEVICE)

    global_step = 0
    last_epoch_loss = None

    for epoch in range(SIMCLR_EPOCHS):
        model.train()
        running = 0.0
        seen = 0

        for xi, xj, *_ in tqdm(train_loader, desc=f"Trial{trial_number} SimCLR {epoch+1}/{SIMCLR_EPOCHS}", leave=False):
            B = xi.size(0)
            if B < 2:
                continue

            xi = xi.to(DEVICE, non_blocking=True)
            xj = xj.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", enabled=(DEVICE.type == "cuda")):
                zi = model(xi)
                zj = model(xj)
                loss = crit(zi, zj)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(opt)
            scaler.update()

            global_step += 1
            scheduler.step_update(num_updates=global_step)

            running += float(loss.detach().cpu())
            seen += 1

        if seen == 0:
            raise RuntimeError(f"Empty epoch in retrain trial {trial_number}")

        last_epoch_loss = running / seen
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[Retrain trial {trial_number}] Epoch {epoch+1}/{SIMCLR_EPOCHS} | "
            f"lr={lr_now:.3e} | train_loss={last_epoch_loss:.4f}"
        )


    # SimCLR diagnostics (alignment/uniformity) on a fixed compute budget
    align_val, uni_val = compute_simclr_alignment_uniformity(
        model=model,
        dataset=simclr_train_ds,
        metric_bs=SIMCLR_METRIC_BS  ,
        num_batches=SIMCLR_METRIC_NUM_BATCHES,
        t_uniform=SIMCLR_UNIFORMITY_T,
    )
    print(f"[Metrics] trial {trial_number} | alignment={align_val:.6f} | uniformity={uni_val:.6f}")

    # Save a compatible checkpoint (similar to simclr_phase2.py)
    ckpt_path = OUT_DIR / f"trial{trial_number}_encoder_retrain.pth"
    meta = {
        "trial_number": int(trial_number),
        "optuna_proxy_value": float(optuna_value),
        "params": dict(params),
        "simclr_epochs": int(SIMCLR_EPOCHS),
        "train_dir": str(TRAIN_DIR),
        "n_train_files": int(len(simclr_train_files)),
        "global_seed": int(GLOBAL_SEED),
    }
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "proj_head_state_dict": model.proj_head.state_dict(),
            "meta": meta,
        },
        str(ckpt_path),
    )
    print(f"[ckpt] Saved retrained encoder: {ckpt_path}")

    return {
        "trial_number": int(trial_number),
        "optuna_rank_proxy_value": float(optuna_value),
        "ckpt_path": str(ckpt_path),
        "retrain_train_loss": float(last_epoch_loss) if last_epoch_loss is not None else None,
        "simclr_alignment": float(align_val),
        "simclr_uniformity": float(uni_val),
        "simclr_lr": simclr_lr,
        "weight_decay": wd,
        "simclr_temperature": temperature,
        "warmup_ratio": warmup_ratio,
        "proj_hidden_dim": proj_hidden_dim,
        "proj_out_dim": proj_out_dim,
        "max_grad_norm": max_grad_norm,
        "simclr_batch_size": simclr_bs,
    }


@torch.no_grad()
def compute_simclr_alignment_uniformity(
    model: SimCLRModel,
    dataset: SimCLRSolarPanelDataset,
    metric_bs: int,
    num_batches: int,
    t_uniform: float,
) -> Tuple[float, float]:
    """
    Compute SimCLR alignment and uniformity on a fixed, small compute budget.

    Notes:
    - Alignment is computed over positive pairs (z_i, z_j). Lower is better.
    - Uniformity is computed on the combined embedding set [z_i; z_j]. Lower (more negative) is better.
    - This is a diagnostic only; we do NOT use it for candidate selection.
    """
    model.eval()

    metric_bs = int(max(2, metric_bs))
    g_metric = torch.Generator().manual_seed(GLOBAL_SEED + 999)

    loader = DataLoader(
        dataset,
        batch_size=metric_bs,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g_metric,
    )

    align_sum = 0.0
    uni_sum = 0.0
    n_align = 0
    n_uni = 0

    for b_idx, (xi, xj, *_) in enumerate(loader):
        if b_idx >= num_batches:
            break
        if xi.size(0) < 2:
            continue

        xi = xi.to(DEVICE, non_blocking=True)
        xj = xj.to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", enabled=(DEVICE.type == "cuda")):
            zi = model(xi)
            zj = model(xj)

        align = simclr_alignment(zi, zj)
        align_sum += float(align) * xi.size(0)
        n_align += xi.size(0)

        z = torch.cat([zi, zj], dim=0)
        uni = simclr_uniformity(z, t=t_uniform)
        uni_sum += float(uni)
        n_uni += 1

    align_avg = align_sum / max(1, n_align)
    uni_avg = uni_sum / max(1, n_uni)
    return float(align_avg), float(uni_avg)


# -----------------------------
# Mini segmentation warm-up eval
# -----------------------------
def freeze_encoder_params(model: UltraLightFCN) -> None:
    """
    Freeze encoder + bottleneck so the warm-up trains only decoder/head.
    This makes the metric more indicative of representation quality.
    """
    for name, p in model.named_parameters():
        if name.startswith(ENCODER_PREFIXES):
            p.requires_grad = False


def assert_encoder_frozen(model: torch.nn.Module) -> None:
    """
    Sanity-check that encoder parameters are frozen.
    Raises AssertionError if no encoder params were frozen (likely prefix mismatch).
    """
    enc_total = 0
    enc_trainable = 0

    for name, p in model.named_parameters():
        if name.startswith(ENCODER_PREFIXES):
            enc_total += p.numel()
            if p.requires_grad:
                enc_trainable += p.numel()

    if enc_total == 0:
        raise AssertionError(
            "No parameters matched encoder_prefixes. "
            "Check ENCODER_PREFIXES vs model.named_parameters() names."
        )

    if enc_trainable > 0:
        raise AssertionError(
            f"Encoder not fully frozen: {enc_trainable}/{enc_total} encoder parameters are still trainable."
        )


def load_encoder_into_ultralight(model: UltraLightFCN, encoder_ckpt_path: str) -> None:
    """
    Load pretrained encoder weights into UltraLightFCN.
    This assumes encoder parameter names match (block1/dsconv*/dilconv*/mini_aspp/sa).
    """
    ckpt = torch.load(encoder_ckpt_path, map_location="cpu")
    if "encoder_state_dict" in ckpt:
        enc_sd = ckpt["encoder_state_dict"]
    elif "encoder" in ckpt:
        enc_sd = ckpt["encoder"]
    else:
        raise RuntimeError(f"Unsupported ckpt format keys={list(ckpt.keys())}")
    # Load with strict=False so decoder keys are ignored
    missing, unexpected = model.load_state_dict(enc_sd, strict=False)
    # Only report large mismatches
    if len(unexpected) > 0:
        print(f"[warn] Unexpected keys when loading encoder: {len(unexpected)}")
    if len(missing) > 0:
        # Missing keys likely correspond to decoder/head; OK.
        pass


@torch.no_grad()
def evaluate_soft_dice(model: UltraLightFCN, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    n = 0
    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        bs = images.size(0)

        with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            d = calculate_dice(logits, masks)

        total += d * bs
        n += bs

    return total / max(1, n)


def mini_seg_warmup_eval(encoder_ckpt_path: str, pos_weight: float | None) -> float:
    """
    Controlled downstream evaluation:
    - Load encoder into UltraLightFCN.
    - Freeze encoder.
    - Train decoder/head for a short budget on downstream TRAIN.
    - Report soft Dice on official downstream VALID.
    """
    set_global_seed(GLOBAL_SEED, deterministic=False)
    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    train_ds = SolarPanelDataset(TRAIN_DIR, mode="train", return_extra=False)
    val_ds = SolarPanelDataset(VAL_DIR, mode="valid", return_extra=False)

    g_train = torch.Generator().manual_seed(GLOBAL_SEED)
    g_val = torch.Generator().manual_seed(GLOBAL_SEED + 12345)

    train_loader = DataLoader(
        train_ds,
        batch_size=MINI_SEG_BS,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=MINI_SEG_BS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g_val,
    )

    model = UltraLightFCN(in_channels=3, num_classes=1, params=SEG_PARAMS).to(DEVICE)
    load_encoder_into_ultralight(model, encoder_ckpt_path)
    freeze_encoder_params(model)
    assert_encoder_frozen(model)

    criterion = BCEDiceLoss(
        pos_weight=pos_weight,
        bce_weight=MINI_SEG_BCE_WEIGHT,
        dice_weight=MINI_SEG_DICE_WEIGHT,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=MINI_SEG_LR, weight_decay=MINI_SEG_WD)

    step = 0
    for epoch in range(MINI_SEG_EPOCHS):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"MiniSeg {epoch+1}/{MINI_SEG_EPOCHS}", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", enabled=(DEVICE.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            step += 1
            if step >= MINI_SEG_MAX_STEPS:
                break
        if step >= MINI_SEG_MAX_STEPS:
            break

    val_soft_dice = evaluate_soft_dice(model, val_loader)
    return float(val_soft_dice)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Load top-K trials from Optuna DB
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    complete = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
    complete.sort(key=lambda t: t.value)  # minimize proxy

    top_trials = complete[:TOPK]
    print(f"Loaded {len(complete)} complete trials. Running retrain + mini-seg eval for top-{TOPK}.\n")

    # FULL downstream TRAIN file list for SimCLR retrain (no internal val)
    train_files_full = list_images_no_masks(TRAIN_DIR)

    # Estimate pos_weight from TRAIN masks for mini seg warm-up
    mini_pos_weight = estimate_pos_weight_from_masks(
        train_dir=TRAIN_DIR,
        max_images=1000,  # 300–1000
        seed=GLOBAL_SEED
    )

    csv_path = OUT_DIR / "phase2_topk_results.csv"
    fieldnames = [
        "trial_number", "optuna_rank_proxy_value", "ckpt_path",
        "mini_val_soft_dice",
        "mini_seg_pos_weight", "mini_seg_bce_weight", "mini_seg_dice_weight",
        "retrain_train_loss",
        "simclr_alignment", "simclr_uniformity",
        "simclr_lr", "weight_decay", "simclr_temperature", "warmup_ratio",
        "proj_hidden_dim", "proj_out_dim", "max_grad_norm", "simclr_batch_size",
    ]

    rows: List[Dict[str, Any]] = []
    for rank, t in enumerate(top_trials, start=1):
        print(f"\n=== Candidate #{rank}/{TOPK}: trial {t.number} (proxy={t.value:.6f}) ===")

        retrain_row = retrain_simclr_candidate(
            trial_number=t.number,
            optuna_value=float(t.value),
            params=t.params,
            simclr_train_files=train_files_full,
        )

        # Controlled downstream evaluation on official VALID
        mini_dice = mini_seg_warmup_eval(retrain_row["ckpt_path"], pos_weight=mini_pos_weight)
        retrain_row["mini_val_soft_dice"] = float(mini_dice)
        retrain_row["mini_seg_pos_weight"] = mini_pos_weight
        retrain_row["mini_seg_bce_weight"] = float(MINI_SEG_BCE_WEIGHT)
        retrain_row["mini_seg_dice_weight"] = float(MINI_SEG_DICE_WEIGHT)

        print(f"[MiniSeg] trial {t.number} | val_soft_dice={mini_dice:.4f}")

        rows.append(retrain_row)

        # Incremental CSV (safe if interrupted)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, None) for k in fieldnames})

        best = max(rows, key=lambda r: r["mini_val_soft_dice"])
        print(f"[Current best] trial {best['trial_number']} with mini-val Dice={best['mini_val_soft_dice']:.4f}")

    rows_sorted = sorted(rows, key=lambda r: r["mini_val_soft_dice"], reverse=True)
    print("\n=== Final ranking by mini-val soft Dice ===")
    for i, r in enumerate(rows_sorted, start=1):
        print(f"#{i:2d} trial {r['trial_number']:3d}  Dice={r['mini_val_soft_dice']:.4f}  proxy={r['optuna_rank_proxy_value']:.6f}")


if __name__ == "__main__":
    main()
