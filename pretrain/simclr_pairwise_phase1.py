"""
Phase1: Augmentation sensitivity heatmap for SimCLR pretraining (UltraLightEncoder).

Methodology-safe alignment with HPO/study code:
- Uses the SAME reduced pretraining pool and the SAME fixed train/val split as HPO
  (pretrain_train_files.txt / pretrain_val_files.txt).
- Uses the SAME proxy metric as HPO for ranking:
    val_ratio = NTXentLoss / ln(2B - 1)
  and objective = average of last K epochs (K=10).
- Keeps hyperparameters FIXED to the final HP set selected after HPO + kNN verification.
- Uses SimCLR-faithful "pairwise policy":
  For each pair (t1, t2), we build ONE augmentation distribution (policy).
  Both views are sampled independently from that SAME policy (dataset calls tf(img) twice).

Dependencies:
- utils/transforms_simclr_pairwise.py  (SimCLR-faithful pairwise policy builder)
- utils/dataset.py -> SimCLRSolarPanelDataset (supports files=...)
- utils/loss_functions.py -> NTXentLoss
- utils/repro.py -> set_global_seed, seed_worker, GLOBAL_SEED
- models/UltraLightFCN_SimCLR.py -> UltraLightEncoder, ProjectionHead, SimCLRModel
- utils/metrics_simclr.py -> simclr_alignment, simclr_uniformity, batch_ssim_windowed
- utils/transforms.py -> RGBOnlyColorJitter
"""

from __future__ import annotations

import os
import math
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from timm.scheduler import CosineLRScheduler

from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel
from utils.dataset import SimCLRSolarPanelDataset
from utils.loss_functions import NTXentLoss
from utils.repro import set_global_seed, seed_worker, GLOBAL_SEED

from utils.transforms_simclr_pairwise import BaseAugCfg, build_simclr_pairwise_transforms
from utils.transforms import RGBOnlyColorJitter

from utils.metrics_simclr import simclr_alignment, simclr_uniformity, batch_ssim_windowed


# -------------------------------------------------------------------------
# 0) Reproducibility (match HPO/study style)
# -------------------------------------------------------------------------
set_global_seed(GLOBAL_SEED, deterministic=False)


# -------------------------------------------------------------------------
# 1) Constants / configuration (align with HPO where relevant)
# -------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "UltraLightFCN"
CHANNELS = 3
IMAGE_SIZE = 256

# IMPORTANT: Must match the folder used to create the split lists in HPO
DATA_ROOT = "../dataset/train"

# Fixed HPO split lists (must exist)
TRAIN_LIST_PATH = "../optuna_study/runs/simclr_hpo/pretrain_train_files.txt"
VAL_LIST_PATH = "../optuna_study/runs/simclr_hpo/pretrain_val_files.txt"

# Training budget (match HPO)
SIMCLR_BS = 256
EPOCHS = 3
DROP_LAST = True

# -------------------------------------------------------------------------
# 2) Best hyperparameters (FILL with your chosen final HP set)
# -------------------------------------------------------------------------
# NOTE: keep these synchronized with the best-trial selected after top10 retrain + kNN.
SIMCLR_LR = 0.0025675349295728707
SIMCLR_WD = 0.00012114033209597344
SIMCLR_TEMPERATURE = 0.05060246552597808

WARMUP_RATIO = 0.23653367813158288   # <-- replace with best_trial warmup_ratio
LR_MIN = 1e-6         # match HPO
MAX_GRAD_NORM = 2.0   # <-- replace with best_trial max_grad_norm

PROJ_HIDDEN_DIM = 256  # <-- replace with best_trial proj_hidden_dim
PROJ_OUT_DIM = 128      # <-- replace with best_trial proj_out_dim

# Objective: avg last K validation ratios (match HPO)
LAST_K_EPOCHS = 10


# -------------------------------------------------------------------------
# 3) Phase1 grid definition (includes grayscale)
# -------------------------------------------------------------------------
OPS = ["identity", "color", "gray", "blur", "hflip", "vflip", "rotate"]
SEEDS = [GLOBAL_SEED]  # Phase1: 1 seed for full grid


# -------------------------------------------------------------------------
# 4) Diagnostics cadence (does not affect objective)
# -------------------------------------------------------------------------
METRICS_EVERY = 20
SSIM_EVERY = 100

SSIM_ON_CPU = False
SSIM_DATA_RANGE = 1.0
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03

UNIFORMITY_T = 2.0
UNIFORMITY_SUBSAMPLE = 512


# -------------------------------------------------------------------------
# 5) Output
# -------------------------------------------------------------------------
RESULTS_DIR = Path("pairwise_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RESULTS_DIR / "simclr_pairwise_phase1.csv"
CKPT_DIR = RESULTS_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FIELDNAMES = [
    "seed",
    "t1",
    "t2",
    "epochs",
    "steps_per_epoch",
    "total_updates",
    "warmup_steps",
    # Objective/proxy
    "val_ratio_avg_last_k",
    "val_ratio_best",
    "val_ratio_last",
    # Diagnostics (train)
    "train_loss_last",
    "train_alignment_last",
    "train_uniformity_last",
    "train_ssim_last",
    # Runtime/paths
    "wall_time_sec",
    "final_ckpt_path",
]


# -------------------------------------------------------------------------
# 6) Helpers
# -------------------------------------------------------------------------
def _load_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def steps_per_epoch(n_samples: int, batch_size: int, drop_last: bool) -> int:
    if n_samples <= 0:
        return 0
    if drop_last:
        return max(1, n_samples // batch_size)
    return max(1, math.ceil(n_samples / batch_size))


def canonical_pair(t1: str, t2: str) -> Tuple[str, str]:
    """(color, blur) == (blur, color)."""
    return tuple(sorted([t1, t2]))


def _load_completed(csv_path: Path) -> set[tuple[int, str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    done: set[tuple[int, str, str]] = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                seed = int(row.get("seed", "0"))
                t1 = row.get("t1", "")
                t2 = row.get("t2", "")
                if t1 and t2:
                    done.add((seed, t1, t2))
            except Exception:
                continue
    return done


def _append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})


def build_model() -> SimCLRModel:
    # Must match the encoder definition used in HPO
    model_params = {
        "enc_channels": [16, 16, 32, 32, 64],
        "enc_kernel_sizes": [3, 3, 3, 3, 3],
        "enc_strides": [1, 2, 2, 1, 1],
        "dilations": [2, 4],
        "mini_aspp": True,
        "mini_aspp_gpool": True,
        "use_sa": False,
        "sa_windowed": True,
        "sa_window_size": 16,
        "sa_shifted": True,
        "sa_heads": 4,
        "sa_dropout": 0.1,
    }

    encoder = UltraLightEncoder(in_channels=CHANNELS, params=model_params).to(DEVICE)
    proj_head = ProjectionHead(
        in_dim=encoder.out_channels,
        hidden_dim=PROJ_HIDDEN_DIM,
        out_dim=PROJ_OUT_DIM,
    ).to(DEVICE)

    model = SimCLRModel(encoder, proj_head).to(DEVICE)

    return model


@torch.no_grad()
def evaluate_val_ratio(model: SimCLRModel, loader: DataLoader, crit: NTXentLoss) -> float:
    """
    Computes mean val_ratio over validation:
      val_ratio = NTXentLoss / ln(2B - 1)
    """
    model.eval()
    running, seen = 0.0, 0

    for xi, xj, *_ in tqdm(loader, desc="Val proxy", leave=False):
        B = xi.size(0)
        if B < 2:
            continue

        xi = xi.to(DEVICE, non_blocking=True)
        xj = xj.to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
            zi = model(xi)
            zj = model(xj)
            loss = float(crit(zi, zj).item())

        baseline = math.log(2 * B - 1)
        running += (loss / baseline) * B
        seen += B

    return float(running / max(1, seen))


# -------------------------------------------------------------------------
# 7) One run: train+val for a single pair
# -------------------------------------------------------------------------
def run_one_pair(
    t1: str,
    t2: str,
    seed: int,
    train_files: List[str],
    val_files: List[str],
) -> Dict[str, Any]:
    # Repro: match HPO style (fast, non-deterministic)
    set_global_seed(seed, deterministic=False)

    # Canonicalize pair to avoid duplicates
    t1, t2 = canonical_pair(t1, t2)
    run_tag = f"PAIR_{t1}+{t2}"

    # Build SimCLR-faithful policy for this pair (HPO-aligned probabilities)
    base_cfg = BaseAugCfg(crop_scale_min=0.4)  # MUST match HPO scale min=0.4
    tf = build_simclr_pairwise_transforms(
        image_size=IMAGE_SIZE,
        cfg=base_cfg,
        t1=t1,
        t2=t2,
        RGBOnlyColorJitter=RGBOnlyColorJitter,
    )

    # Datasets/loaders using FIXED HPO lists
    train_ds = SimCLRSolarPanelDataset(DATA_ROOT, image_size=IMAGE_SIZE, transform=tf, files=train_files)
    val_ds = SimCLRSolarPanelDataset(DATA_ROOT, image_size=IMAGE_SIZE, transform=tf, files=val_files)

    n_train = len(train_ds)
    drop_last_train = DROP_LAST
    if DROP_LAST and n_train < SIMCLR_BS:
        drop_last_train = False

    g = torch.Generator().manual_seed(seed)
    num_workers = 8

    train_loader = DataLoader(
        train_ds,
        batch_size=SIMCLR_BS,
        shuffle=True,
        pin_memory=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=SIMCLR_BS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )

    spe = steps_per_epoch(n_train, SIMCLR_BS, drop_last_train)
    total_updates = EPOCHS * spe
    if total_updates <= 0:
        raise RuntimeError(f"total_updates<=0 for {run_tag} (n_train={n_train}, bs={SIMCLR_BS})")

    warmup_steps = max(1, int(WARMUP_RATIO * total_updates))
    warmup_steps = min(warmup_steps, max(1, total_updates - 1))

    # Model/opt/sched/loss (AdamW without fused to match HPO)
    model = build_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SIMCLR_LR,
        weight_decay=SIMCLR_WD,
    )

    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=total_updates,
        lr_min=LR_MIN,
        warmup_lr_init=1e-6,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    criterion = NTXentLoss(temperature=SIMCLR_TEMPERATURE, device=DEVICE)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    # Tracking
    global_update = 0
    val_hist: List[float] = []
    best_val = float("inf")

    # Diagnostics (train, last epoch averages)
    last_train_loss = float("nan")
    last_train_align = float("nan")
    last_train_uni = float("nan")
    last_train_ssim = float("nan")

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()

        run_loss = 0.0
        run_align = 0.0
        run_uni = 0.0
        run_ssim = 0.0

        iters = 0
        metrics_count = 0
        ssim_count = 0

        for xi, xj, *_ in tqdm(train_loader, desc=f"{run_tag} | Train {epoch+1}/{EPOCHS}", leave=False):
            iters += 1
            B = xi.size(0)
            if B < 2:
                continue

            xi = xi.to(DEVICE, non_blocking=True)
            xj = xj.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
                zi = model(xi)
                zj = model(xj)
                loss = criterion(zi, zj)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            global_update += 1
            scheduler.step_update(num_updates=global_update)

            run_loss += float(loss.item())

            # Alignment / uniformity diagnostics
            if (METRICS_EVERY > 0) and (global_update % METRICS_EVERY) == 0:
                with torch.no_grad():
                    align = simclr_alignment(zi.detach(), zj.detach())

                    z_cat = torch.cat([zi.detach(), zj.detach()], dim=0)
                    if UNIFORMITY_SUBSAMPLE is not None and z_cat.size(0) > UNIFORMITY_SUBSAMPLE:
                        idx = torch.randperm(z_cat.size(0), device=z_cat.device)[:UNIFORMITY_SUBSAMPLE]
                        z_cat = z_cat[idx]
                    uni = simclr_uniformity(z_cat, t=UNIFORMITY_T)

                run_align += float(align.item())
                run_uni += float(uni.item())
                metrics_count += 1

            # SSIM diagnostic
            if (SSIM_EVERY > 0) and (global_update % SSIM_EVERY) == 0:
                with torch.no_grad():
                    if SSIM_ON_CPU:
                        ssim_val = batch_ssim_windowed(
                            xi.detach().float().cpu(),
                            xj.detach().float().cpu(),
                            data_range=SSIM_DATA_RANGE,
                            window_size=SSIM_WINDOW,
                            sigma=SSIM_SIGMA,
                            k1=SSIM_K1,
                            k2=SSIM_K2,
                        )
                    else:
                        ssim_val = batch_ssim_windowed(
                            xi.detach().float(),
                            xj.detach().float(),
                            data_range=SSIM_DATA_RANGE,
                            window_size=SSIM_WINDOW,
                            sigma=SSIM_SIGMA,
                            k1=SSIM_K1,
                            k2=SSIM_K2,
                        )
                run_ssim += float(ssim_val.item())
                ssim_count += 1

        # Train epoch averages (for logging only)
        last_train_loss = run_loss / max(1, iters)
        last_train_align = (run_align / max(1, metrics_count)) if metrics_count > 0 else float("nan")
        last_train_uni = (run_uni / max(1, metrics_count)) if metrics_count > 0 else float("nan")
        last_train_ssim = (run_ssim / max(1, ssim_count)) if ssim_count > 0 else float("nan")

        # Validation proxy (HPO-compatible)
        val_ratio = evaluate_val_ratio(model, val_loader, criterion)
        val_hist.append(val_ratio)
        best_val = min(best_val, val_ratio)

        print(
            f"[{run_tag}] Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={last_train_loss:.4f} | val_ratio={val_ratio:.4f} | "
            f"lr_now={optimizer.param_groups[0]['lr']:.3e}"
        )

    # Objective: avg last K epochs
    k = min(LAST_K_EPOCHS, len(val_hist))
    avg_last_k = float(sum(val_hist[-k:]) / k)
    last_val = float(val_hist[-1]) if val_hist else float("nan")

    # Save final checkpoint (optional but useful)
    final_path = CKPT_DIR / f"simclr_{MODEL_NAME}_seed{seed}_{t1}+{t2}_final.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "proj_head_state_dict": model.proj_head.state_dict(),
            "meta": {
                "seed": seed,
                "t1": t1,
                "t2": t2,
                "image_size": IMAGE_SIZE,
                "channels": CHANNELS,
                "model_name": MODEL_NAME,
                "epochs": EPOCHS,
                "bs": SIMCLR_BS,
                "hp": {
                    "lr": SIMCLR_LR,
                    "wd": SIMCLR_WD,
                    "temperature": SIMCLR_TEMPERATURE,
                    "warmup_ratio": WARMUP_RATIO,
                    "lr_min": LR_MIN,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "proj_hidden_dim": PROJ_HIDDEN_DIM,
                    "proj_out_dim": PROJ_OUT_DIM,
                },
                "objective": {
                    "avg_last_k_val_ratio": avg_last_k,
                    "best_val_ratio": best_val,
                    "last_val_ratio": last_val,
                    "k": k,
                },
            },
        },
        str(final_path),
    )

    wall_time = time.time() - start_time

    return {
        "seed": seed,
        "t1": t1,
        "t2": t2,
        "epochs": EPOCHS,
        "steps_per_epoch": spe,
        "total_updates": total_updates,
        "warmup_steps": warmup_steps,
        "val_ratio_avg_last_k": avg_last_k,
        "val_ratio_best": best_val,
        "val_ratio_last": last_val,
        "train_loss_last": last_train_loss,
        "train_alignment_last": last_train_align,
        "train_uniformity_last": last_train_uni,
        "train_ssim_last": last_train_ssim,
        "wall_time_sec": wall_time,
        "final_ckpt_path": str(final_path),
    }


# -------------------------------------------------------------------------
# 8) Main
# -------------------------------------------------------------------------
def main():
    # Safety checks (match study/HPO assumptions)
    if not os.path.exists(TRAIN_LIST_PATH) or not os.path.exists(VAL_LIST_PATH):
        raise FileNotFoundError(
            f"Missing split lists. Expected:\n"
            f"  {TRAIN_LIST_PATH}\n  {VAL_LIST_PATH}\n"
            f"Run HPO split creation first."
        )

    train_files = _load_list(TRAIN_LIST_PATH)
    val_files = _load_list(VAL_LIST_PATH)

    if len(train_files) == 0 or len(val_files) == 0:
        raise RuntimeError("Empty train/val file list(s).")

    # Prepare unique canonical pairs only (i <= j) to avoid duplicates
    pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(OPS):
        for j, b in enumerate(OPS):
            if j < i:
                continue
            pairs.append(canonical_pair(a, b))
    pairs = sorted(list(set(pairs)))

    completed = _load_completed(OUT_CSV)

    print(f"[Info] DATA_ROOT: {DATA_ROOT}")
    print(f"[Info] Train files: {len(train_files)} | Val files: {len(val_files)}")
    print(f"[Info] Unique pairs: {len(pairs)} | Seeds: {SEEDS}")
    print(f"[Info] CSV: {OUT_CSV}")

    for seed in SEEDS:
        for (t1, t2) in pairs:
            key = (seed, t1, t2)
            if key in completed:
                print(f"[SKIP] seed={seed} {t1}+{t2} already in CSV")
                continue

            print(f"\n=== RUN seed={seed} pair={t1}+{t2} ===")
            row = run_one_pair(
                t1=t1,
                t2=t2,
                seed=seed,
                train_files=train_files,
                val_files=val_files,
            )

            _append_csv_row(OUT_CSV, row)
            completed.add(key)

            print(
                f"[DONE] seed={seed} {t1}+{t2} | "
                f"avg_last_k_val_ratio={row['val_ratio_avg_last_k']:.6f} "
                f"(best={row['val_ratio_best']:.6f}, last={row['val_ratio_last']:.6f}) | "
                f"time={row['wall_time_sec']:.1f}s"
            )

    print(f"\n✓ Completed. Results saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
