"""
SimCLR pretraining hyperparameter optimization (Optuna) for UltraLightEncoder.

Optuna-tuned hyperparameters:
  - simclr_lr: AdamW learning rate
  - simclr_temperature: NT-Xent temperature
  - weight_decay: AdamW weight decay
  - warmup_ratio: fraction of total update steps used for LR warmup
  - proj_hidden_dim: MLP hidden dim in projection head
  - proj_out_dim: output embedding dim of projection head
  - max_grad_norm: gradient clipping threshold (small categorical set)

Keeps the stability improvements:
  - TPESampler(seed=GLOBAL_SEED) for reproducibility
  - MedianPruner for noisy SSL metrics
  - Manual warmup-epoch guard before pruning
  - Objective = average of last K validation ratios

Validation proxy metric:
    val_ratio = NTXentLoss / ln(2B - 1)
Lower is better; ~1.0 is approximately random baseline.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

from utils.dataset import SimCLRSolarPanelDataset
from utils.repro import set_global_seed, GLOBAL_SEED
from utils.helpers import clear_cuda_cache, infer_subset_from_filename, make_reduced_file_list
from utils.loss_functions import NTXentLoss
from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel

torch.multiprocessing.set_sharing_strategy("file_system")

# -------------------------------------------------------------------------
# 0) Reproducibility
# -------------------------------------------------------------------------
set_global_seed(GLOBAL_SEED, deterministic=False)


# -------------------------------------------------------------------------
# 1) Constants / configuration
# -------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/workspace/UltraLightFCN/dataset/train"   # image-only folder for SimCLR

SIMCLR_BS = 256
EPOCHS = 40
DROP_LAST = True

# Reduced pool settings (speed-up for HPO)
REDUCE_MAX_TOTAL = 5120  # max total images in reduced pool

# Pretrain validation split (within the reduced pool)
PRETRAIN_VAL_FRAC = 0.10

# HPO stability knobs
WARMUP_EPOCHS = 8      # do not allow pruning before this epoch index is reached
LAST_K_EPOCHS = 10     # objective = average of last K validation ratios

# Store deterministic file lists (pool/train/val) for reproducibility
RUN_DIR = Path("runs/simclr_hpo")
RUN_DIR.mkdir(parents=True, exist_ok=True)
POOL_LIST_PATH = RUN_DIR / "pretrain_pool_files.txt"
TRAIN_LIST_PATH = RUN_DIR / "pretrain_train_files.txt"
VAL_LIST_PATH = RUN_DIR / "pretrain_val_files.txt"

# -------------------------------------------------------------------------
# 2) Helper: save/load file lists to guarantee reproducibility
# -------------------------------------------------------------------------
def _save_list(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x}\n")


def _load_list(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stratified_split_files(files: List[str], val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    Pure-Python stratified split by subset label (PVxx prefix).
    Keeps the PV01/PV03/PV08 distribution (and OTHER) roughly consistent
    between pretrain-train and pretrain-val.
    """
    assert 0.0 < val_frac < 1.0
    rng = random.Random(seed)

    buckets: Dict[str, List[str]] = {}
    for f in files:
        buckets.setdefault(infer_subset_from_filename(f), []).append(f)

    train_files: List[str] = []
    val_files: List[str] = []

    for subset, subset_files in buckets.items():
        rng.shuffle(subset_files)
        n = len(subset_files)
        n_val = max(1, int(round(val_frac * n))) if n > 1 else 0
        val_files.extend(subset_files[:n_val])
        train_files.extend(subset_files[n_val:])

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    return train_files, val_files


def get_or_create_pretrain_splits() -> Tuple[List[str], List[str]]:
    """
    Creates or loads a deterministic reduced pool and its train/val split.
    All Optuna trials use the SAME lists -> fair comparison and reproducibility.
    """
    if POOL_LIST_PATH.exists() and TRAIN_LIST_PATH.exists() and VAL_LIST_PATH.exists():
        train_files = _load_list(TRAIN_LIST_PATH)
        val_files = _load_list(VAL_LIST_PATH)
        return train_files, val_files

    pool = make_reduced_file_list(
        DATA_ROOT,
        max_total=REDUCE_MAX_TOTAL,
        seed=GLOBAL_SEED,
    )
    if len(pool) == 0:
        raise RuntimeError(f"No images found in {DATA_ROOT}. Check your path/extensions.")

    train_files, val_files = stratified_split_files(pool, val_frac=PRETRAIN_VAL_FRAC, seed=GLOBAL_SEED)

    _save_list(POOL_LIST_PATH, pool)
    _save_list(TRAIN_LIST_PATH, train_files)
    _save_list(VAL_LIST_PATH, val_files)

    print(f"[Split] Reduced pool: {len(pool)} files")
    print(f"[Split] Pretrain-train: {len(train_files)} files")
    print(f"[Split] Pretrain-val:   {len(val_files)} files")
    print(f"[Split] Saved lists to: {RUN_DIR.resolve()}")

    return train_files, val_files


PRETRAIN_TRAIN_FILES, PRETRAIN_VAL_FILES = get_or_create_pretrain_splits()


# -------------------------------------------------------------------------
# 3) Helper: steps per epoch calculation and worker seeding
# -------------------------------------------------------------------------
def steps_per_epoch(n: int, bs: int, drop_last: bool) -> int:
    """Number of optimizer updates per epoch."""
    if n <= 0:
        return 0
    if drop_last:
        return max(1, n // bs)
    return max(1, math.ceil(n / bs))

def seed_worker(worker_id: int):
    """
    Ensure each DataLoader worker has a deterministic RNG state.
    This makes augmentations + any numpy/random usage reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -------------------------------------------------------------------------
# 4) Optuna objective (one trial)
# -------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """
    Tuned hyperparameters:
      - simclr_lr: AdamW learning rate
      - simclr_temperature: NT-Xent temperature
      - weight_decay: AdamW weight decay
      - warmup_ratio: fraction of total update steps used for warmup
      - proj_hidden_dim: projection MLP hidden dimension
      - proj_out_dim: projection output dimension (embedding dim)
      - max_grad_norm: gradient clipping threshold
    """
    # Reproducibility per trial
    set_global_seed(GLOBAL_SEED, deterministic=False)

    # 4.1 Hyperparameters to tune
    simclr_lr = trial.suggest_float("simclr_lr", 3e-5, 3e-3, log=True)
    simclr_temperature = trial.suggest_float("simclr_temperature", 0.05, 1.0, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.30)
    proj_hidden_dim = trial.suggest_categorical("proj_hidden_dim", [128, 256])
    proj_out_dim = trial.suggest_categorical("proj_out_dim", [64, 128])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0])

    # 4.2 Dataset / loaders (fixed file lists)
    simclr_train_ds = SimCLRSolarPanelDataset(DATA_ROOT, image_size=256, files=PRETRAIN_TRAIN_FILES)
    simclr_val_ds = SimCLRSolarPanelDataset(DATA_ROOT, image_size=256, files=PRETRAIN_VAL_FILES)

    n_train = len(simclr_train_ds)
    drop_last_trial = DROP_LAST
    if DROP_LAST and n_train < SIMCLR_BS:
        drop_last_trial = False

    # Deterministic sampling + deterministic worker RNG for THIS trial
    g_train = torch.Generator()
    g_train.manual_seed(GLOBAL_SEED)

    g_val = torch.Generator()
    g_val.manual_seed(GLOBAL_SEED + 10)

    train_loader = DataLoader(
        simclr_train_ds,
        batch_size=SIMCLR_BS,
        shuffle=True,
        pin_memory=True,
        drop_last=drop_last_trial,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g_train,
    )

    val_loader = DataLoader(
        simclr_val_ds,
        batch_size=SIMCLR_BS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g_val,
    )

    # 4.3 Step-based schedule setup
    spe = steps_per_epoch(n_train, SIMCLR_BS, drop_last_trial)
    total_steps = EPOCHS * spe
    if total_steps <= 0:
        raise optuna.TrialPruned()

    # NEW: warmup_steps derived from tuned warmup_ratio
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    # Safety: do not let warmup exceed total_steps-1 (Cosine scheduler expects warmup < total cycle)
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    # 4.4 Model setup
    model_params = {
        'enc_channels': [16, 16, 32, 32, 64],
        'enc_kernel_sizes': [3, 3, 3, 3, 3],
        'enc_strides': [1, 2, 2, 1, 1],
        'dilations': [2, 4],
        'mini_aspp': True,
        'mini_aspp_gpool': True,
        'use_sa': False,
        'sa_windowed': True,
        'sa_window_size': 16,
        'sa_shifted': True,
        'sa_heads': 4,
        'sa_dropout': 0.1,
    }

    encoder = UltraLightEncoder(in_channels=3, params=model_params).to(DEVICE)

    # IMPORTANT: ProjectionHead signature is (in_dim, hidden_dim, out_dim)
    proj_head = ProjectionHead(encoder.out_channels, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim).to(DEVICE)

    model = SimCLRModel(encoder, proj_head).to(DEVICE)

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

    crit = NTXentLoss(temperature=simclr_temperature, device=DEVICE)

    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    # Log metadata (useful for debugging / thesis reproducibility appendix)
    trial.set_user_attr("n_train", int(len(simclr_train_ds)))
    trial.set_user_attr("n_val", int(len(simclr_val_ds)))
    trial.set_user_attr("batch_size", int(SIMCLR_BS))
    trial.set_user_attr("drop_last_trial", bool(drop_last_trial))
    trial.set_user_attr("steps_per_epoch", int(spe))
    trial.set_user_attr("total_steps", int(total_steps))
    trial.set_user_attr("warmup_steps", int(warmup_steps))
    trial.set_user_attr("warmup_ratio", float(warmup_ratio))
    trial.set_user_attr("proj_hidden_dim", int(proj_hidden_dim))
    trial.set_user_attr("proj_out_dim", int(proj_out_dim))
    trial.set_user_attr("max_grad_norm", float(max_grad_norm))

    # Global seed for reproducibility
    trial.set_user_attr("global_seed", int(GLOBAL_SEED))
    trial.set_user_attr("n_train_files", len(PRETRAIN_TRAIN_FILES))
    trial.set_user_attr("n_val_files", len(PRETRAIN_VAL_FILES))

    # 4.5 Train + val proxy per epoch
    epoch_val_ratios: List[float] = []
    global_step = 0

    for epoch in range(EPOCHS):
        # ---- TRAIN ----
        model.train()
        seen = 0

        for xi, xj, *_ in tqdm(
                train_loader,
                desc=f"[Trial {trial.number}] Train {epoch + 1}/{EPOCHS}",
                leave=False,
                mininterval=1.0,  # NEW: reduces tqdm overhead a bit
        ):
            B = xi.size(0)
            if B < 2:
                continue

            xi = xi.to(DEVICE, non_blocking=True)
            xj = xj.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
                zi = model(xi)
                zj = model(xj)
                loss = crit(zi, zj)

            scaler.scale(loss).backward()

            # Unscale before clipping when using GradScaler
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(opt)
            scaler.update()

            global_step += 1
            scheduler.step_update(num_updates=global_step)

            seen += B

        if seen == 0:
            raise optuna.TrialPruned()

        # ---- VALIDATION PROXY ----
        model.eval()
        running = 0.0
        val_seen = 0

        with torch.no_grad():
            for xi, xj, *_ in tqdm(
                    val_loader,
                    desc=f"[Trial {trial.number}] Val {epoch + 1}/{EPOCHS}",
                    leave=False,
                    mininterval=1.0,  # NEW
            ):
                B = xi.size(0)
                if B < 2:
                    continue

                xi = xi.to(DEVICE, non_blocking=True)
                xj = xj.to(DEVICE, non_blocking=True)

                # CHANGED: autocast only on CUDA
                with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):  # CHANGED
                    zi = model(xi)
                    zj = model(xj)
                    val_loss = crit(zi, zj).item()

                baseline = math.log(2 * B - 1)
                ratio = val_loss / baseline

                running += ratio * B
                val_seen += B

        if val_seen == 0:
            raise optuna.TrialPruned()

        val_ratio = float(running / val_seen)
        epoch_val_ratios.append(val_ratio)

        # ---- OPTUNA REPORT / PRUNING ----
        trial.report(val_ratio, step=epoch)

        # Manual warm-up guard: do not prune too early (SSL metrics can be noisy at start).
        if epoch >= WARMUP_EPOCHS and trial.should_prune():
            raise optuna.TrialPruned()

        current_lr = opt.param_groups[0]["lr"]
        print(
            f"[Trial {trial.number}] Epoch {epoch+1}/{EPOCHS} | "
            f"LR_now={current_lr:.3e} | simclr_lr={simclr_lr:.3e} | WD={wd:.2e} | T={simclr_temperature:.3g} | "
            f"warmup_ratio={warmup_ratio:.2f} | proj=({proj_hidden_dim}->{proj_out_dim}) | "
            f"clip={max_grad_norm} | val_ratio={val_ratio:.4f}"
        )

    # Objective: average of LAST_K_EPOCHS validation ratios
    k = min(LAST_K_EPOCHS, len(epoch_val_ratios))
    avg_last_k = float(sum(epoch_val_ratios[-k:]) / k)

    trial.set_user_attr("objective_avg_last_k_val_ratio", avg_last_k)
    trial.set_user_attr("best_val_ratio", float(min(epoch_val_ratios)))
    trial.set_user_attr("final_val_ratio", float(epoch_val_ratios[-1]))

    return avg_last_k


# -------------------------------------------------------------------------
# 5) Study runner
# -------------------------------------------------------------------------
def main():
    # Reproducible sampler
    sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)

    # Robust pruner for noisy SSL objectives
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=8,
        interval_steps=1,
    )

    study = optuna.create_study(
        direction="minimize",
        study_name="UltraLightFCN_SimCLR_pretrain_RGB",
        storage="sqlite:///UltraLightFCN_study.db",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=100,
        timeout=24 * 60 * 60,
        callbacks=[clear_cuda_cache],
    )

    print("\nBest hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k:24s}: {v}")
    print("Min objective (avg last-k val_ratio):", study.best_value)


if __name__ == "__main__":
    main()
