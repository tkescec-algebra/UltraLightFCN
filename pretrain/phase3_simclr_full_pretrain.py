"""
phase3_simclr_full_pretrain.py

Phase 3: Final SimCLR full pretraining on the FULL downstream TRAIN split (80%),
using the best hyperparameters selected in Phase 2.

Methodology alignment with Phase 1/2:
- Train-only: uses ONLY the downstream TRAIN split (no official VALID, no TEST).
- No proxy validation and no "best checkpoint" selection: checkpoint is the final epoch ("last").
- Phase 2 provides the *only* selection step via controlled downstream warm-up on the official VALID split.
  Phase 3 simply takes the best Phase 2 candidate and performs full-budget train-only pretraining.
- Controlled reproducibility: deterministic, sorted file list + seeded DataLoader
  (generator + worker_init_fn) + per-seed RNG reset (same philosophy as Phase 1/2).
- Encoder definition matches your SimCLR model: backbone + MiniASPP + Self-Attention ("sa")
  are pretrained jointly and transferred downstream.
- Optional representation diagnostics (alignment/uniformity) are computed with a fixed,
  compute-capped protocol and logged for analysis (NOT used for selection).

"""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from timm.scheduler import CosineLRScheduler
from tqdm import tqdm


from utils.config import ENCODER_PARAMS
# Project imports (keep consistent with Phase 1/2)
from utils.dataset import SimCLRSolarPanelDataset
from utils.helpers import steps_per_epoch
from utils.loss_functions import NTXentLoss
from utils.repro import set_global_seed, seed_worker
from utils.metrics_simclr import simclr_alignment, simclr_uniformity
from pretrain.utils.transforms import get_simclr_transforms

# SimCLR model components
from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel

torch.multiprocessing.set_sharing_strategy("file_system")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class Phase3Config:
    # Data
    train_dir: str = "../dataset/train"
    image_size: int = 256

    # Output
    out_dir: str = "checkpoints/simclr_phase3"
    run_tag: str = "phase3_full_pretrain"

    # Phase 2 results (used to pull the best SimCLR hyperparameters)
    # IMPORTANT: this CSV must come from Phase 2 (top-K retrain + downstream warm-up ranking).
    phase2_results_csv: str = "checkpoints/simclr_topk_retrain_downstream/phase2_topk_results.csv"

    # Seeds
    # Using 1 seed is acceptable for SSL full pretrain if compute is limited.
    # Prefer reporting multi-seed variability at the downstream segmentation stage.
    seeds: Tuple[int, ...] = (13,)

    # Full pretrain budget (fixed, no selection)
    epochs: int = 200

    # Scheduler
    lr_min: float = 1e-6

    # Optimization stability
    amp: bool = True

    # Logging / checkpoints
    save_every_epochs: int = 0  # 0 disables periodic checkpoints; only saves "last"
    log_every_steps: int = 50

    # Alignment/uniformity diagnostics (optional)
    compute_rep_metrics: bool = True
    rep_metric_bs: int = 256
    rep_metric_num_batches: int = 4

    # Encoder architecture (must match downstream usage!)
    # If you want to override defaults, set encoder_params explicitly here.
    encoder_params: Optional[Dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(ENCODER_PARAMS))


# -----------------------------------------------------------------------------
# Deterministic file list (critical for reproducibility)
# -----------------------------------------------------------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_images_no_masks_sorted(folder: str) -> List[str]:
    """
    Return a deterministic, sorted list of image filenames excluding mask files.
    Assumes mask files end with '_label.png' (adjust if needed).
    """
    p = Path(folder)
    files: List[str] = []
    for fp in p.iterdir():
        name = fp.name
        lname = name.lower()
        if not lname.endswith(IMG_EXTS):
            continue
        if lname.endswith("_label.png"):
            continue
        files.append(name)
    return sorted(files)


# -----------------------------------------------------------------------------
# Phase 2 → Phase 3: Load best candidate hyperparameters from CSV
# -----------------------------------------------------------------------------
def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_best_phase2_row(csv_path: str) -> Dict[str, str]:
    """
    Select the best candidate from Phase 2 according to the Phase 2 selection rule:
    - primary: maximize mini_val_soft_dice
    - tie-break: minimize optuna_rank_proxy_value (smaller contrastive proxy is better)
    """
    csv_path = str(csv_path)
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Phase2 results CSV not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError(f"Phase2 results CSV is empty: {csv_path}")

    # Check required columns early (fail fast)
    required = [
        "mini_val_soft_dice",
        "optuna_rank_proxy_value",
        "simclr_batch_size",
        "simclr_lr",
        "weight_decay",
        "simclr_temperature",
        "warmup_ratio",
        "max_grad_norm",
        "proj_hidden_dim",
        "proj_out_dim",
    ]
    missing_cols = [c for c in required if c not in rows[0]]
    if missing_cols:
        raise RuntimeError(
            f"Phase2 CSV missing required columns: {missing_cols}\n"
            f"Found columns: {list(rows[0].keys())}"
        )

    rows.sort(
        key=lambda r: (
            -_to_float(r.get("mini_val_soft_dice")),
            _to_float(r.get("optuna_rank_proxy_value"), default=1e9),
        )
    )
    return rows[0]


@dataclass
class SimCLRHP:
    """
    The SimCLR hyperparameters we reuse from Phase 2.
    """
    batch_size: int
    lr: float
    weight_decay: float
    temperature: float
    warmup_ratio: float
    grad_clip: float
    proj_hidden_dim: int
    proj_out_dim: int

    # Optional provenance fields for audit trail
    trial_number: Optional[int] = None
    mini_val_soft_dice: Optional[float] = None
    optuna_rank_proxy_value: Optional[float] = None


def parse_simclr_hp_from_phase2(row: Dict[str, str]) -> SimCLRHP:
    """
    Convert the best Phase 2 row into typed hyperparameters for Phase 3.
    """
    hp = SimCLRHP(
        batch_size=int(float(row["simclr_batch_size"])),
        lr=float(row["simclr_lr"]),
        weight_decay=float(row["weight_decay"]),
        temperature=float(row["simclr_temperature"]),
        warmup_ratio=float(row["warmup_ratio"]),
        grad_clip=float(row["max_grad_norm"]),
        proj_hidden_dim=int(float(row["proj_hidden_dim"])),
        proj_out_dim=int(float(row["proj_out_dim"])),
        trial_number=int(row["trial_number"]) if "trial_number" in row and row["trial_number"] != "" else None,
        mini_val_soft_dice=_to_float(row.get("mini_val_soft_dice")),
        optuna_rank_proxy_value=_to_float(row.get("optuna_rank_proxy_value")),
    )
    return hp


# -----------------------------------------------------------------------------
# Model building + checkpointing
# -----------------------------------------------------------------------------
def build_simclr_model(
    encoder_params: Optional[Dict[str, Any]],
    proj_hidden_dim: int,
    proj_out_dim: int,
) -> SimCLRModel:
    """
    Build the SimCLR model used in Phase 3.

    IMPORTANT:
    - encoder_params must match downstream usage to keep transfer consistent.
    - projection head dims are taken from Phase 2 best candidate for consistency.
    """
    encoder = UltraLightEncoder(in_channels=3, params=encoder_params)
    proj_head = ProjectionHead(in_dim=64, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim)
    model = SimCLRModel(encoder=encoder, proj_head=proj_head)
    return model


def save_checkpoint(path: Path, model: SimCLRModel, meta: Dict[str, Any]) -> None:
    """
    Save both full SimCLR weights and encoder-only weights for downstream transfer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "proj_head_state_dict": model.proj_head.state_dict(),
            "meta": meta,
        },
        str(path),
    )


# -----------------------------------------------------------------------------
# Representation diagnostics (alignment/uniformity) — diagnostics only
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_alignment_uniformity(
    model: SimCLRModel,
    dataset: SimCLRSolarPanelDataset,
    device: torch.device,
    batch_size: int,
    num_batches: int,
    seed: int,
    num_workers: int = 8,
) -> Tuple[float, float]:
    """
    Compute alignment and uniformity on a fixed, compute-capped protocol:
    - fixed batch_size
    - deterministic shuffle (seeded generator)
    - drop_last=True (constant batch size)
    - only first `num_batches` batches
    """
    model.eval()

    g = torch.Generator().manual_seed(seed + 999)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    align_sum = 0.0
    uni_sum = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        x1, x2, _ = batch
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        z1 = model(x1)
        z2 = model(x2)

        align_sum += float(simclr_alignment(z1, z2))
        uni_sum += float(simclr_uniformity(torch.cat([z1, z2], dim=0)))
        n += 1

    if n == 0:
        return float("nan"), float("nan")

    return align_sum / n, uni_sum / n


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def run_one_seed(cfg: Phase3Config, hp: SimCLRHP, seed: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fairness/reproducibility: reset RNG per run
    set_global_seed(seed, deterministic=False, strict=False)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic file list (critical)
    train_files = list_images_no_masks_sorted(cfg.train_dir)
    if len(train_files) == 0:
        raise RuntimeError(f"No training images found in: {cfg.train_dir}")

    # Persist file list + chosen Phase2 HP for audit trail
    (out_dir / f"phase3_train_files_seed{seed}.txt").write_text("\n".join(train_files), encoding="utf-8")
    (out_dir / "phase3_best_phase2_candidate.json").write_text(
        json.dumps(asdict(hp), indent=2), encoding="utf-8"
    )

    # SimCLR training transforms (same family as Phase 1/2)
    train_tf = get_simclr_transforms(image_size=cfg.image_size)

    train_ds = SimCLRSolarPanelDataset(
        cfg.train_dir, image_size=cfg.image_size, transform=train_tf, files=train_files
    )

    drop_last = (len(train_ds) >= hp.batch_size)
    g_train = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        generator=g_train,
        worker_init_fn=seed_worker,
        num_workers=8,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Model
    model = build_simclr_model(cfg.encoder_params, hp.proj_hidden_dim, hp.proj_out_dim).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    # Scheduler (cosine, per-iteration)
    spe = steps_per_epoch(len(train_ds), hp.batch_size, drop_last)
    total_steps = cfg.epochs * spe

    warmup_steps = int(hp.warmup_ratio * total_steps)
    warmup_steps = min(max(1, warmup_steps), max(1, total_steps - 1))

    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=total_steps,
        lr_min=cfg.lr_min,
        warmup_lr_init=1e-6,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    # Loss
    criterion = NTXentLoss(temperature=hp.temperature, device=device)

    # AMP scaler per run (do not reuse across runs)
    scaler = GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    # Logging CSV (train-only)
    csv_path = out_dir / f"phase3_train_metrics_seed{seed}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("seed,epoch,step,lr,train_loss\n")

    global_step = 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0

        for batch in tqdm(train_loader, desc=f"[Phase3][seed={seed}] epoch {epoch}/{cfg.epochs}", leave=False):
            x1, x2, _ = batch
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=scaler.is_enabled()):
                z1 = model(x1)
                z2 = model(x2)
                loss = criterion(z1, z2)

            scaler.scale(loss).backward()

            # Gradient clipping (unscale first)
            if hp.grad_clip and hp.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step_update(global_step)

            bs = x1.size(0)
            epoch_loss += float(loss.detach()) * bs
            seen += bs

            global_step += 1

            if cfg.log_every_steps > 0 and (global_step % cfg.log_every_steps == 0):
                lr_now = optimizer.param_groups[0]["lr"]
                avg_loss = epoch_loss / max(1, seen)
                print(
                    f"[Phase3][seed={seed}] epoch={epoch}/{cfg.epochs} "
                    f"step={global_step} lr={lr_now:.3e} train_loss={avg_loss:.5f}"
                )

        lr_now = optimizer.param_groups[0]["lr"]
        avg_epoch_loss = epoch_loss / max(1, seen)

        with csv_path.open("a", encoding="utf-8") as f:
            f.write(f"{seed},{epoch},{global_step},{lr_now:.8e},{avg_epoch_loss:.8f}\n")

        # Optional periodic checkpointing (not used for selection)
        if cfg.save_every_epochs and cfg.save_every_epochs > 0 and (epoch % cfg.save_every_epochs == 0):
            meta = {
                "phase": "phase3_full_pretrain",
                "run_tag": cfg.run_tag,
                "seed": seed,
                "epoch": epoch,
                "global_step": global_step,
                "config": asdict(cfg),
                "simclr_hp": asdict(hp),
                "train_files_count": len(train_files),
            }
            ckpt_path = out_dir / f"phase3_seed{seed}_epoch{epoch:04d}.pth"
            save_checkpoint(ckpt_path, model, meta)

    # Final checkpoint ("last") — this is the one you should transfer downstream
    meta_last = {
        "phase": "phase3_full_pretrain",
        "run_tag": cfg.run_tag,
        "seed": seed,
        "epoch": cfg.epochs,
        "global_step": global_step,
        "config": asdict(cfg),
        "simclr_hp": asdict(hp),
        "train_files_count": len(train_files),
        "elapsed_sec": time.time() - t0,
        "selection_note": "No official VALID/TEST used in Phase 3. Last checkpoint saved (fixed budget).",
    }
    last_path = out_dir / f"phase3_seed{seed}_last.pth"
    save_checkpoint(last_path, model, meta_last)
    print(f"[Phase3][seed={seed}] Saved LAST checkpoint: {last_path}")

    # Optional representation diagnostics (diagnostics only; not used for selection)
    if cfg.compute_rep_metrics:
        align, uni = compute_alignment_uniformity(
            model=model,
            dataset=train_ds,
            device=device,
            batch_size=cfg.rep_metric_bs,
            num_batches=cfg.rep_metric_num_batches,
            seed=seed,
            num_workers=8,
        )
        diag = {
            "seed": seed,
            "alignment": align,
            "uniformity": uni,
            "rep_metric_bs": cfg.rep_metric_bs,
            "rep_metric_num_batches": cfg.rep_metric_num_batches,
        }
        (out_dir / f"phase3_rep_metrics_seed{seed}.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        print(f"[Phase3][seed={seed}] alignment={align:.6f} uniformity={uni:.6f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = Phase3Config()

    # Load best Phase 2 candidate once (single source of truth for Phase 3 HP)
    best_row = load_best_phase2_row(cfg.phase2_results_csv)
    hp = parse_simclr_hp_from_phase2(best_row)

    print(
        "[Phase3] Selected best Phase2 candidate: "
        f"trial={hp.trial_number} "
        f"mini_val_soft_dice={hp.mini_val_soft_dice:.6f} "
        f"optuna_proxy={hp.optuna_rank_proxy_value:.6f} "
        f"(bs={hp.batch_size}, lr={hp.lr}, wd={hp.weight_decay}, temp={hp.temperature}, warmup={hp.warmup_ratio})"
    )

    for seed in cfg.seeds:
        run_one_seed(cfg, hp, seed)


if __name__ == "__main__":
    main()
