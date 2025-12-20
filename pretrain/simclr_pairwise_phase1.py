import os
import math
import csv
import time

import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from timm.scheduler import CosineLRScheduler

from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel
from utils.dataset import SimCLRSolarPanelDataset
from utils.loss_functions import NTXentLoss
from utils.repro import set_global_seed, seed_worker

from utils.transforms_simclr_pairwise import BaseAugCfg, build_simclr_pairwise_transforms
from utils.metrics_simclr import simclr_alignment, simclr_uniformity, batch_ssim_windowed
from utils.transforms import RGBOnlyColorJitter


# =========================
# Speed / Numerical knobs
# =========================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

torch.multiprocessing.set_sharing_strategy("file_system")

# Metrics cadence
METRICS_EVERY = 20
SSIM_EVERY = 100  # SSIM every 100 updates

# SSIM on GPU (no GPU->CPU copies)
SSIM_ON_CPU = False

# Standard SSIM settings (windowed)
SSIM_DATA_RANGE = 1.0
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT     = "../dataset"

MODEL_NAME    = "UltraLightFCN"
EDGE_DETECTOR = None
CHANNELS      = 3
IMAGE_SIZE    = 256

SIMCLR_BS     = 256
DROP_LAST     = True
TOTAL_EPOCHS  = 50

SIMCLR_LR          = 0.0024554497897962607
SIMCLR_TEMPERATURE = 0.060705096152236224
SIMCLR_WD          = 1.584361249525883e-06

WARMUP_PCT    = 0.10
LR_MIN        = 1e-5
MAX_GRAD_NORM = 1.0

UNIFORMITY_T = 2.0
UNIFORMITY_SUBSAMPLE = 512


RESULTS_DIR = "pairwise_results"
OUT_CSV = os.path.join(RESULTS_DIR, "simclr_pairwise_results.csv")

CSV_FIELDNAMES = [
    "seed",
    "t1",
    "t2",
    "epoch",
    "loss",
    "alignment",
    "uniformity",
    "ssim",
    "steps_per_epoch",
    "total_updates",
    "metrics_count",
    "wall_time_sec",
]


def steps_per_epoch(n_samples: int, batch_size: int, drop_last: bool) -> int:
    return max(1, n_samples // batch_size) if drop_last else max(1, math.ceil(n_samples / batch_size))


def build_model():
    encoder = UltraLightEncoder(in_channels=CHANNELS, params={
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
    }).to(DEVICE)

    proj_head = ProjectionHead(in_dim=encoder.out_channels, hidden_dim=128, out_dim=64).to(DEVICE)
    model = SimCLRModel(encoder, proj_head).to(DEVICE)

    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    return model


def _load_completed_pairs(csv_path: str) -> set[tuple[int, str, str]]:
    """Return a set of (seed, t1, t2) already present in the CSV."""
    if not os.path.exists(csv_path):
        return set()

    completed = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seed = int(row.get("seed", "0"))
                t1 = row.get("t1", "")
                t2 = row.get("t2", "")
                if t1 and t2:
                    completed.add((seed, t1, t2))
            except Exception:
                continue
    return completed


def _append_csv_row(csv_path: str, fieldnames: list[str], row: dict):
    """Append a single row to CSV; write header if file does not exist or is empty."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_one_pair(t1: str, t2: str, base_cfg: BaseAugCfg, seed: int = 42):
    set_global_seed(seed, deterministic=False, strict=False)
    training_type = "edge" if CHANNELS == 4 else "rgb"
    run_tag = f"PAIR_{t1}+{t2}"

    wandb.init(
        mode="offline",
        name=f"{MODEL_NAME}({training_type})-{run_tag}-seed{seed}",
        project="UltraLightFCN_SimCLR-phase1-pairwise",
        entity="tomislav-kescec-algebra",
        config={
            "phase": "phase1_pairwise",
            "t1": t1,
            "t2": t2,
            "base_cfg": vars(base_cfg),
            "seed": seed,
            "bs": SIMCLR_BS,
            "epochs": TOTAL_EPOCHS,
            "metrics_every": METRICS_EVERY,
            "ssim_every": SSIM_EVERY,
            "tf32": True,
            "fused_adamw": (DEVICE.type == "cuda"),
            "ssim_on_cpu": SSIM_ON_CPU,
            "ssim_window": SSIM_WINDOW,
            "ssim_sigma": SSIM_SIGMA,
            "ssim_k1": SSIM_K1,
            "ssim_k2": SSIM_K2,
        },
        reinit=True,
    )

    wandb.define_metric("global_step")
    wandb.define_metric("simclr/*", step_metric="global_step")
    wandb.define_metric("simclr/loss", summary="min")
    wandb.define_metric("simclr/alignment", summary="min")
    wandb.define_metric("simclr/uniformity", summary="min")
    wandb.define_metric("simclr/ssim", summary="mean")

    tf = build_simclr_pairwise_transforms(
        image_size=IMAGE_SIZE,
        cfg=base_cfg,
        t1=t1,
        t2=t2,
        RGBOnlyColorJitter=RGBOnlyColorJitter,
    )

    g = torch.Generator().manual_seed(seed)
    train_ds = SimCLRSolarPanelDataset(
        f"{DATA_ROOT}/train",
        edge_detector=EDGE_DETECTOR,
        channels=CHANNELS,
        image_size=IMAGE_SIZE,
        transform=tf,
    )
    n_full = len(train_ds)

    num_workers = 8
    train_loader = DataLoader(
        train_ds,
        batch_size=SIMCLR_BS,
        shuffle=True,
        pin_memory=True,
        drop_last=DROP_LAST,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )

    spe = steps_per_epoch(n_full, SIMCLR_BS, DROP_LAST)
    total_updates = TOTAL_EPOCHS * spe
    warmup_steps = max(1, int(WARMUP_PCT * total_updates))

    model = build_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SIMCLR_LR,
        weight_decay=SIMCLR_WD,
        fused=(DEVICE.type == "cuda"),
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
    scaler = GradScaler()

    global_update = 0
    last = None

    for epoch in range(TOTAL_EPOCHS):
        model.train()

        run_loss = 0.0
        run_align = 0.0
        run_uni = 0.0
        run_ssim = 0.0

        # counts:
        metrics_count = 0         # alignment/uniformity count
        ssim_count = 0            # ssims count
        iters = 0

        for xi, xj, *_ in tqdm(train_loader, desc=f"{run_tag} | {epoch+1}/{TOTAL_EPOCHS}", leave=False):
            iters += 1

            xi = xi.to(DEVICE, non_blocking=True)
            xj = xj.to(DEVICE, non_blocking=True)

            if DEVICE.type == "cuda":
                xi = xi.to(memory_format=torch.channels_last)
                xj = xj.to(memory_format=torch.channels_last)

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

            # =========================
            # Metrics (alignment/uniformity) every 20
            # SSIM every 100
            # =========================
            if (global_update % METRICS_EVERY) == 0:
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

            if (global_update % SSIM_EVERY) == 0:
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
                        # Keep fp32 for stability; still on GPU and fast enough at 1/100 updates
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

        avg_loss = run_loss / max(1, iters)
        avg_align = run_align / max(1, metrics_count)
        avg_uni = run_uni / max(1, metrics_count)
        avg_ssim = run_ssim / max(1, ssim_count)

        wandb.log({
            "global_step": global_update,
            "simclr/epoch": epoch + 1,
            "simclr/loss": avg_loss,
            "simclr/alignment": avg_align,
            "simclr/uniformity": avg_uni,
            "simclr/ssim": avg_ssim,
            "simclr/lr": optimizer.param_groups[0]["lr"],
            "simclr/metrics_count": metrics_count,
            "simclr/ssim_count": ssim_count,
        })

        last = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "alignment": avg_align,
            "uniformity": avg_uni,
            "ssim": avg_ssim,
            "steps_per_epoch": spe,
            "total_updates": total_updates,
            "metrics_count": metrics_count,
        }

    # =========================
    # Save final weights (.pth)
    # =========================
    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    final_path = os.path.join(
        ckpt_dir,
        f"simclr_{MODEL_NAME}_seed{seed}_{t1}+{t2}_final.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),  # full SimCLR (encoder + projection head)
            "encoder_state_dict": model.encoder.state_dict(),  # just encoder (often enough for embeddings)
            "proj_head_state_dict": model.proj_head.state_dict(),
            "meta": {
                "seed": seed,
                "t1": t1,
                "t2": t2,
                "image_size": IMAGE_SIZE,
                "channels": CHANNELS,
                "model_name": MODEL_NAME,
            },
        },
        final_path,
    )
    print(f"[SAVE] Final weights saved to: {final_path}")
    wandb.finish()
    return last


def main():
    ops = ["identity", "color", "blur", "hflip", "vflip", "rotate"]

    base_cfg = BaseAugCfg(
        crop_scale_min=0.2,
        cj_strength=0.4,
        cj_hue=0.04,
        blur_k=5,
        blur_sigma_min=0.1,
        blur_sigma_max=2.0,
        rot_deg=15,
        rot_min_abs_deg=1.0,
    )

    seed = 42

    completed = _load_completed_pairs(OUT_CSV)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for t1 in ops:
        for t2 in ops:
            key = (seed, t1, t2)
            if key in completed:
                print(f"[SKIP] {t1}+{t2} (seed={seed}) already in CSV")
                continue

            start = time.time()
            metrics = run_one_pair(t1, t2, base_cfg, seed=seed)
            wall_time_sec = time.time() - start

            row = {
                "seed": seed,
                "t1": t1,
                "t2": t2,
                **metrics,
                "wall_time_sec": wall_time_sec,
            }

            _append_csv_row(OUT_CSV, CSV_FIELDNAMES, row)
            completed.add(key)

            print(
                f"[DONE] {t1}+{t2} | loss={metrics['loss']:.4f} "
                f"align={metrics['alignment']:.4f} uni={metrics['uniformity']:.4f} "
                f"ssim={metrics['ssim']:.4f} | wrote CSV row"
            )

    print(f"✓ CSV (incremental): {OUT_CSV}")


if __name__ == "__main__":
    main()
