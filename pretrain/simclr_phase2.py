"""
simclr_phase2.py

Phase2: Full SimCLR pretraining (UltraLightEncoder) using FINAL HP set selected by HPO -> top10 retrain -> kNN.

Methodology-safe alignment:
- HP fixed to the best kNN-selected trial (here: trial 32).
- Adds a validation split using the existing VALID folder.
- Proxy metric matches HPO:
    val_ratio = NTXentLoss / ln(2B - 1)
- "Best checkpoint" criterion is HPO-aligned:
    best_score = min avg_last_k_val_ratio (K=10)
  with the minimal correction:
    - BEST updates only start once len(val_hist) >= K.

Transforms:
- TRAIN: standard SimCLR augmentations (two stochastic views).
- VAL: deterministic transform; val-proxy computed using two identical deterministic views.

Logging:
- CSV per epoch (train_loss, val_loss, val_ratio, avg_last_k, alignment, uniformity, lr, timing)
- Checkpoints:
    - "last" overwritten every epoch
    - "best" overwritten only when avg_last_k improves AND len(val_hist) >= K
"""

from __future__ import annotations

import math
import csv
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from timm.scheduler import CosineLRScheduler

from torchvision import transforms
from PIL import Image
import numpy as np

from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel
from utils.dataset import SimCLRSolarPanelDataset
from utils.loss_functions import NTXentLoss
from utils.repro import set_global_seed
from utils.metrics_simclr import simclr_alignment, simclr_uniformity

# Must return the same SimCLR TRAIN augmentation pipeline used in HPO.
from utils.transforms import get_simclr_transforms

torch.multiprocessing.set_sharing_strategy("file_system")


# -------------------------------------------------------------------------
# 1) Paths / setup
# -------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "/workspace/UltraLightFCN/dataset/train"
VAL_DIR   = "/workspace/UltraLightFCN/dataset/valid"

IMAGE_SIZE = 256
CHANNELS = 3

SIMCLR_BS = 256
DROP_LAST = True

TOTAL_EPOCHS = 200

# Multi-seed runs (Phase2)
SEEDS = [13, 37, 73]


# -------------------------------------------------------------------------
# 2) FINAL HP (trial 32, selected by best kNN-F1)
# -------------------------------------------------------------------------
SIMCLR_LR = 0.0025675349295728707
SIMCLR_WD = 0.00012114033209597344
SIMCLR_TEMPERATURE = 0.05060246552597808

WARMUP_RATIO = 0.23653367813158288
LR_MIN = 1e-6
MAX_GRAD_NORM = 2.0

PROJ_HIDDEN_DIM = 256
PROJ_OUT_DIM = 128


# -------------------------------------------------------------------------
# 3) Best-checkpoint criterion (HPO-aligned)
# -------------------------------------------------------------------------
LAST_K_EPOCHS = 10
BEST_EPS = 1e-4  # avoids tiny/noisy best updates


# -------------------------------------------------------------------------
# 4) Diagnostics cadence (does not affect objective)
# -------------------------------------------------------------------------
METRICS_EVERY = 20
UNIFORMITY_T = 2.0
UNIFORMITY_SUBSAMPLE = 512


# -------------------------------------------------------------------------
# 5) Output
# -------------------------------------------------------------------------
OUT_DIR = Path("checkpoints/simclr_phase2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "phase2_metrics.csv"
CSV_FIELDNAMES = [
    "seed",
    "epoch",
    "steps_per_epoch",
    "global_step",
    "wall_time_sec",
    # Train
    "train_loss",
    # Val proxy
    "val_loss",
    "val_ratio",
    "avg_last_k_val_ratio",
    # Diagnostics
    "alignment",
    "uniformity",
    # LR
    "lr_now",
    # Checkpoints
    "ckpt_last_path",
    "ckpt_best_path",
    # Best info (tracked)
    "best_epoch",
    "best_avg_last_k_val_ratio",
]


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def steps_per_epoch(n: int, bs: int, drop_last: bool) -> int:
    if n <= 0:
        return 0
    if drop_last:
        return max(1, n // bs)
    return max(1, math.ceil(n / bs))


def _append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})


def build_model() -> SimCLRModel:
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

    return SimCLRModel(encoder, proj_head).to(DEVICE)


def save_ckpt(path: Path, model: SimCLRModel, meta: Dict[str, Any]) -> str:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "proj_head_state_dict": model.proj_head.state_dict(),
            "meta": meta,
        },
        str(path),
    )
    return str(path)


# -------------------------------------------------------------------------
# Deterministic VAL dataset (two identical views)
# -------------------------------------------------------------------------
class DeterministicTwoViewDataset(torch.utils.data.Dataset):
    """
    Returns (view1, view2, filename) where view2 == view1 (identical deterministic view).
    Intended for validation proxy to remove augmentation stochasticity.
    """
    def __init__(self, data_dir: str, image_size: int = 256, transform=None):
        self.data_dir = data_dir
        self.tf = transform if transform is not None else transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        exts = (".png", ".jpg", ".jpeg")
        self.images = [
            f for f in sorted(list(Path(data_dir).iterdir()))
            if f.name.lower().endswith(exts) and (not f.name.lower().endswith("_label.png"))
        ]
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        p = self.images[idx]
        img = Image.open(str(p)).convert("RGB")
        x = self.tf(img).float()
        # identical views
        return x, x, p.name


@torch.no_grad()
def evaluate_val_proxy(model: SimCLRModel, loader: DataLoader, crit: NTXentLoss) -> Dict[str, float]:
    """
    Returns:
      - val_loss: mean NT-Xent over val
      - val_ratio: mean NT-Xent / ln(2B-1)
    """
    model.eval()
    running_loss = 0.0
    running_ratio = 0.0
    seen = 0

    for xi, xj, *_ in tqdm(loader, desc="Val", leave=False, mininterval=1.0):
        B = xi.size(0)
        if B < 2:
            continue

        xi = xi.to(DEVICE, non_blocking=True)
        xj = xj.to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
            zi = model(xi)
            zj = model(xj)
            loss = crit(zi, zj)

        loss_f = float(loss.item())
        baseline = math.log(2 * B - 1)
        ratio = loss_f / baseline

        running_loss += loss_f * B
        running_ratio += ratio * B
        seen += B

    if seen == 0:
        return {"val_loss": float("nan"), "val_ratio": float("nan")}

    return {
        "val_loss": float(running_loss / seen),
        "val_ratio": float(running_ratio / seen),
    }


# -------------------------------------------------------------------------
# One seed run
# -------------------------------------------------------------------------
def run_one_seed(seed: int) -> None:
    set_global_seed(seed, deterministic=False, strict=False)

    # TRAIN: same SimCLR stochastic transform policy as HPO
    train_tf = get_simclr_transforms(image_size=IMAGE_SIZE)

    # VAL: deterministic transform + identical views
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_ds = SimCLRSolarPanelDataset(TRAIN_DIR, image_size=IMAGE_SIZE, transform=train_tf)
    val_ds   = DeterministicTwoViewDataset(VAL_DIR, image_size=IMAGE_SIZE, transform=val_tf)

    n_train = len(train_ds)
    n_val = len(val_ds)

    drop_last_train = DROP_LAST
    if DROP_LAST and n_train < SIMCLR_BS:
        drop_last_train = False

    train_loader = DataLoader(
        train_ds,
        batch_size=SIMCLR_BS,
        shuffle=True,
        pin_memory=True,
        drop_last=drop_last_train,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=SIMCLR_BS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
    )

    spe = steps_per_epoch(n_train, SIMCLR_BS, drop_last_train)
    total_steps = TOTAL_EPOCHS * spe
    if total_steps <= 0:
        raise RuntimeError("total_steps<=0 (check dataset/batch).")

    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    print(
        f"\n[Phase2] seed={seed} | train={n_train} val={n_val} | "
        f"BS={SIMCLR_BS} | spe={spe} | total_steps={total_steps} | warmup_steps={warmup_steps}"
    )

    model = build_model()

    opt = torch.optim.AdamW(model.parameters(), lr=SIMCLR_LR, weight_decay=SIMCLR_WD)
    scheduler = CosineLRScheduler(
        optimizer=opt,
        t_initial=total_steps,
        lr_min=LR_MIN,
        warmup_lr_init=1e-6,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    criterion = NTXentLoss(temperature=SIMCLR_TEMPERATURE, device=DEVICE)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    global_step = 0

    # Best tracking (avg_last_k of val_ratio)
    val_hist: List[float] = []
    best_score = float("inf")
    best_epoch = -1

    # Paths
    last_ckpt_path = OUT_DIR / f"phase2_seed{seed}_last.pth"
    best_ckpt_path = OUT_DIR / f"phase2_seed{seed}_best.pth"

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        start = time.time()

        run_loss = 0.0
        iters = 0

        run_align = 0.0
        run_uni = 0.0
        metrics_count = 0

        for xi, xj, *_ in tqdm(train_loader, desc=f"Train {epoch}/{TOTAL_EPOCHS}", leave=False, mininterval=1.0):
            B = xi.size(0)
            if B < 2:
                continue

            xi = xi.to(DEVICE, non_blocking=True)
            xj = xj.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
                zi = model(xi)
                zj = model(xj)
                loss = criterion(zi, zj)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            scheduler.step_update(num_updates=global_step)

            run_loss += float(loss.item())
            iters += 1

            # Diagnostics: alignment/uniformity
            if (METRICS_EVERY > 0) and (global_step % METRICS_EVERY == 0):
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

        train_loss = run_loss / max(1, iters)
        alignment = (run_align / metrics_count) if metrics_count > 0 else float("nan")
        uniformity = (run_uni / metrics_count) if metrics_count > 0 else float("nan")

        # Val proxy (HPO-compatible), but with deterministic identical views
        val_metrics = evaluate_val_proxy(model, val_loader, criterion)
        val_ratio = float(val_metrics["val_ratio"])
        val_loss = float(val_metrics["val_loss"])

        val_hist.append(val_ratio)
        k = min(LAST_K_EPOCHS, len(val_hist))
        avg_last_k = float(sum(val_hist[-k:]) / k)

        lr_now = opt.param_groups[0]["lr"]
        wall = time.time() - start

        # Meta for checkpoint
        meta = {
            "phase": "phase2_full_pretrain",
            "seed": seed,
            "epoch": epoch,
            "global_step": global_step,
            "train_dir": TRAIN_DIR,
            "val_dir": VAL_DIR,
            "image_size": IMAGE_SIZE,
            "channels": CHANNELS,
            "bs": SIMCLR_BS,
            "steps_per_epoch": spe,
            "total_steps": total_steps,
            "hp": {
                "lr": SIMCLR_LR,
                "wd": SIMCLR_WD,
                "temperature": SIMCLR_TEMPERATURE,
                "warmup_ratio": WARMUP_RATIO,
                "warmup_steps": warmup_steps,
                "lr_min": LR_MIN,
                "max_grad_norm": MAX_GRAD_NORM,
                "proj_hidden_dim": PROJ_HIDDEN_DIM,
                "proj_out_dim": PROJ_OUT_DIM,
            },
            "metrics": {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ratio": val_ratio,
                "avg_last_k_val_ratio": avg_last_k,
                "k": k,
                "alignment": alignment,
                "uniformity": uniformity,
                "lr_now": lr_now,
                "wall_time_sec": wall,
            },
            "best_tracking": {
                "criterion": f"min avg_last_k(val_ratio), K={LAST_K_EPOCHS}",
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_eps": BEST_EPS,
                "best_updates_start_after_k": True,
            },
        }

        # ------------------------------------------------------------
        # Checkpointing policy:
        #  - "last": overwrite every epoch
        #  - "best": overwrite only when avg_last_k improves,
        #            and only after we have at least K epochs in history
        # ------------------------------------------------------------
        ckpt_last = save_ckpt(last_ckpt_path, model, meta)

        ckpt_best = str(best_ckpt_path) if best_ckpt_path.exists() else ""
        if len(val_hist) >= LAST_K_EPOCHS:
            if avg_last_k < (best_score - BEST_EPS):
                best_score = avg_last_k
                best_epoch = epoch

                meta["best_tracking"]["best_epoch"] = best_epoch
                meta["best_tracking"]["best_score"] = best_score

                ckpt_best = save_ckpt(best_ckpt_path, model, meta)
                print(
                    f"[seed={seed}] NEW BEST @ epoch {epoch}: avg_last_{k}={avg_last_k:.6f} "
                    f"(val_ratio={val_ratio:.6f}) -> {ckpt_best}"
                )

        # CSV logging (per-epoch)
        row = {
            "seed": seed,
            "epoch": epoch,
            "steps_per_epoch": spe,
            "global_step": global_step,
            "wall_time_sec": wall,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ratio": val_ratio,
            "avg_last_k_val_ratio": avg_last_k,
            "alignment": alignment,
            "uniformity": uniformity,
            "lr_now": lr_now,
            "ckpt_last_path": ckpt_last,
            "ckpt_best_path": ckpt_best,
            "best_epoch": best_epoch,
            "best_avg_last_k_val_ratio": best_score,
        }
        _append_csv_row(CSV_PATH, row)

        print(
            f"[seed={seed}] Epoch {epoch}/{TOTAL_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_ratio={val_ratio:.4f} | "
            f"avg_last_{k}={avg_last_k:.4f} | align={alignment:.4f} uni={uniformity:.4f} | "
            f"lr={lr_now:.3e} | step={global_step}"
        )

    print(f"[seed={seed}] DONE. Best avg_last_k={best_score:.6f} @ epoch {best_epoch}.")
    print(f"[seed={seed}] Best ckpt: {best_ckpt_path}")
    print(f"[seed={seed}] Last ckpt: {last_ckpt_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print(f"[Phase2] DEVICE={DEVICE}")
    print(f"[Phase2] TRAIN_DIR={TRAIN_DIR}")
    print(f"[Phase2] VAL_DIR={VAL_DIR}")
    print(f"[Phase2] CSV={CSV_PATH}")
    print(f"[Phase2] Seeds={SEEDS}")
    print(
        f"[Phase2] Best criterion: min avg_last_k(val_ratio), K={LAST_K_EPOCHS} "
        f"(BEST_EPS={BEST_EPS}) | best updates start after K epochs"
    )
    print("[Phase2] VAL is deterministic: identical views per image for proxy evaluation")

    for s in SEEDS:
        run_one_seed(s)

    print(f"\n✓ Done. Metrics saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
