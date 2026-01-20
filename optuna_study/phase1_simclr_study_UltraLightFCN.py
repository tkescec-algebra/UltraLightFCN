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

import torch
import optuna
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

from utils.config import ENCODER_PARAMS
from utils.dataset import SimCLRSolarPanelDataset
from utils.repro import set_global_seed, GLOBAL_SEED, seed_worker
from utils.helpers import clear_cuda_cache, infer_subset_from_filename, make_reduced_file_list, steps_per_epoch
from utils.loss_functions import NTXentLoss
from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel

torch.multiprocessing.set_sharing_strategy("file_system")

# -------------------------------------------------------------------------
# 0) Reproducibility
# -------------------------------------------------------------------------
set_global_seed(GLOBAL_SEED, deterministic=False)

# -------------------------------------------------------------------------
# 1) Configuration
# -------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class Phase1Config:
    # Data
    data_root: str = "../dataset/train"  # image-only folder for SimCLR

    # Train budget
    simclr_bs: int = 256
    epochs: int = 40
    drop_last: bool = True

    # Reduced pool settings (speed-up for HPO)
    reduce_max_total: int = 5120  # max total images in reduced pool

    # Pretrain validation split (within the reduced pool)
    pretrain_val_frac: float = 0.10

    # HPO stability knobs
    warmup_epochs: int = 8      # do not allow pruning before this epoch index is reached
    last_k_epochs: int = 10     # objective = average of last K validation ratios

    # Store deterministic file lists (pool/train/val) for reproducibility
    run_dir: str = "runs/simclr_hpo"

    # Optuna
    study_name: str = "UltraLightFCN_SimCLR_pretrain_RGB"
    storage: str = "sqlite:///UltraLightFCN_study.db"
    n_trials: int = 70
    timeout_sec: int = 24 * 60 * 60

    # Pruner
    pruner_n_startup_trials: int = 10
    pruner_n_warmup_steps: int = 8
    pruner_interval_steps: int = 1


CFG = Phase1Config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Derived paths
RUN_DIR = Path(CFG.run_dir)
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
        CFG.data_root,
        max_total=CFG.reduce_max_total,
        seed=GLOBAL_SEED,
    )
    if len(pool) == 0:
        raise RuntimeError(f"No images found in {CFG.data_root}. Check your path/extensions.")

    train_files, val_files = stratified_split_files(pool, val_frac=CFG.pretrain_val_frac, seed=GLOBAL_SEED)

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
# 3) Optuna objective (one trial)
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

    # Hyperparameters to tune
    simclr_lr = trial.suggest_float("simclr_lr", 3e-5, 3e-3, log=True)
    simclr_temperature = trial.suggest_float("simclr_temperature", 0.05, 1.0, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.30)
    proj_hidden_dim = trial.suggest_categorical("proj_hidden_dim", [128, 256])
    proj_out_dim = trial.suggest_categorical("proj_out_dim", [64, 128])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0])

    # Dataset / loaders (fixed file lists)
    simclr_train_ds = SimCLRSolarPanelDataset(CFG.data_root, image_size=256, files=PRETRAIN_TRAIN_FILES)
    simclr_val_ds = SimCLRSolarPanelDataset(CFG.data_root, image_size=256, files=PRETRAIN_VAL_FILES)

    n_train = len(simclr_train_ds)
    drop_last_trial = CFG.drop_last
    if CFG.drop_last and n_train < CFG.simclr_bs:
        drop_last_trial = False

    # Deterministic sampling + deterministic worker RNG for THIS trial
    g_train = torch.Generator()
    g_train.manual_seed(GLOBAL_SEED)

    g_val = torch.Generator()
    g_val.manual_seed(GLOBAL_SEED + 10)

    train_loader = DataLoader(
        simclr_train_ds,
        batch_size=CFG.simclr_bs,
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
        batch_size=CFG.simclr_bs,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g_val,
    )

    # Step-based schedule setup
    spe = steps_per_epoch(n_train, CFG.simclr_bs, drop_last_trial)
    total_steps = CFG.epochs * spe
    if total_steps <= 0:
        raise optuna.TrialPruned()

    warmup_steps = int(warmup_ratio * total_steps)
    warmup_steps = min(max(1, warmup_steps), max(1, total_steps - 1))

    encoder = UltraLightEncoder(in_channels=3, params=ENCODER_PARAMS).to(DEVICE)

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
    trial.set_user_attr("batch_size", int(CFG.simclr_bs))
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

    for epoch in range(CFG.epochs):
        # ---- TRAIN ----
        model.train()
        seen = 0

        for xi, xj, *_ in tqdm(
                train_loader,
                desc=f"[Trial {trial.number}] Train {epoch + 1}/{CFG.epochs}",
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
                    desc=f"[Trial {trial.number}] Val {epoch + 1}/{CFG.epochs}",
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
        if epoch >= CFG.warmup_epochs and trial.should_prune():
            raise optuna.TrialPruned()

        current_lr = opt.param_groups[0]["lr"]
        print(
            f"[Trial {trial.number}] Epoch {epoch+1}/{CFG.epochs} | "
            f"LR_now={current_lr:.3e} | simclr_lr={simclr_lr:.3e} | WD={wd:.2e} | T={simclr_temperature:.3g} | "
            f"warmup_ratio={warmup_ratio:.2f} | proj=({proj_hidden_dim}->{proj_out_dim}) | "
            f"clip={max_grad_norm} | val_ratio={val_ratio:.4f}"
        )

    # Objective: average of CFG.last_k_epochs validation ratios
    k = min(CFG.last_k_epochs, len(epoch_val_ratios))
    avg_last_k = float(sum(epoch_val_ratios[-k:]) / k)

    trial.set_user_attr("objective_avg_last_k_val_ratio", avg_last_k)
    trial.set_user_attr("best_val_ratio", float(min(epoch_val_ratios)))
    trial.set_user_attr("final_val_ratio", float(epoch_val_ratios[-1]))

    return avg_last_k


# -------------------------------------------------------------------------
# 4) Study runner
# -------------------------------------------------------------------------
def main():
    # Reproducible sampler
    sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)

    # Robust pruner for noisy SSL objectives
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=CFG.pruner_n_startup_trials,
        n_warmup_steps=CFG.pruner_n_warmup_steps,
        interval_steps=CFG.pruner_interval_steps,
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=CFG.study_name,
        storage=CFG.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=CFG.n_trials,
        timeout=CFG.timeout_sec,
        callbacks=[clear_cuda_cache],
    )

    print("\nBest hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k:24s}: {v}")
    print("Min objective (avg last-k val_ratio):", study.best_value)


if __name__ == "__main__":
    main()
