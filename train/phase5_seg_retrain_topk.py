"""phase5_seg_retrain_topk.py

Phase 5 — Top-K retrain (confirmation) on FULL TRAIN (80%) with FULL VALID (10%),
aligned with Phase-4 segmentation HPO.

Adds:
  - automatic winner selection and saving to phase5_winner.json for Phase-6
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import optuna
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


from models.UltraLightFCN_base import UltraLightFCN
from utils.config import SEG_PARAMS
from utils.dataset import SolarPanelDataset
from utils.helpers import (
    clear_cuda_cache,
    get_loss_function,
    save_json,
    save_csv_rows,
    split_encoder_decoder_params,
    build_loss_from_params,
)
from utils.load_simclr_pretrain_encoder import load_pretrained_encoder_into_ultralight
from utils.metrics import calculate_dice
from utils.repro import seed_worker, set_global_seed


@dataclass(frozen=True)
class Phase5Config:
    # --------- Data
    data_root: str = "../dataset"
    train_split: str = "train"
    val_split: str = "valid"

    # --------- Pretrain init (Phase-3 LAST)
    phase3_last_ckpt: str = "../pretrain/checkpoints/simclr_phase3/phase3_seed13_last.pth"

    # --------- Model (fixed)
    in_channels: int = 3
    num_classes: int = 1
    seg_params: dict = field(default_factory=lambda: dict(SEG_PARAMS))

    # --------- Training budget (fixed across candidates)
    epochs: int = 30
    avg_last_k: int = 10
    hard_thr_monitor: float = 0.5

    # --------- Dataloaders (FULL TRAIN + FULL VALID)
    num_workers: int = 8
    pin_memory: bool = True
    drop_last_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # --------- Runtime / Repro
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deterministic: bool = False
    strict: bool = False
    seeds: Tuple[int, ...] = (13,37,71)  # e.g. (13, 37, 71) for robustness

    # --------- Optuna study source (Phase-4)
    study_name: str = "UltraLightFCN_seg_softdice"
    storage: str = "sqlite:///../optuna_study/UltraLightFCN_study.db"
    top_k: int = 10

    # --------- Output
    out_root: str = "seg_phase5/topk_retrain"
    save_last_name: str = "last.pth"
    save_epoch_log_name: str = "epoch_log.csv"
    summary_csv_name: str = "phase5_topk_results.csv"
    winner_json_name: str = "phase5_winner.json"


def _build_loaders(cfg: Phase5Config, batch_size: int, *, seed: int) -> Tuple[DataLoader, DataLoader]:
    """FULL TRAIN + FULL VALID loaders (no subset lists in Phase-5)."""
    train_dir = os.path.join(cfg.data_root, cfg.train_split)
    val_dir = os.path.join(cfg.data_root, cfg.val_split)

    train_ds = SolarPanelDataset(train_dir, mode="train", files=None, return_extra=False)
    val_ds = SolarPanelDataset(val_dir, mode="valid", files=None, return_extra=False)

    pw = cfg.persistent_workers and (cfg.num_workers > 0)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last_train,
        persistent_workers=pw,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=pw,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader


def _val_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    cfg: Phase5Config,
    *,
    use_amp: bool,
    criterion,
) -> Tuple[float, float, float]:
    """Return (val_loss, soft_dice, hard_dice@thr) aggregated per-image across full validation."""
    model.eval()
    loss_sum, soft_sum, hard_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            bs = int(images.shape[0])
            loss_sum += float(loss.detach().cpu()) * bs
            soft = float(calculate_dice(logits, masks, thr=None))
            hard = float(calculate_dice(logits, masks, thr=cfg.hard_thr_monitor))
            soft_sum += soft * bs
            hard_sum += hard * bs
            n += bs

    if n == 0:
        return 0.0, 0.0, 0.0
    return loss_sum / n, soft_sum / n, hard_sum / n


def run_one_candidate_one_seed(
    cfg: Phase5Config,
    candidate_id: int,
    params: Dict[str, Any],
    optuna_value: float,
    seed: int,
) -> Dict[str, Any]:
    """Train one candidate config on FULL TRAIN for one seed. Saves LAST + logs. Returns summary stats."""
    set_global_seed(seed, deterministic=cfg.deterministic, strict=cfg.strict)

    batch_size = int(params["batch_size"])
    base_lr = float(params["base_lr"])
    enc_lr_mult = float(params["enc_lr_mult"])
    weight_decay = float(params["weight_decay"])
    rlop_factor = float(params["rlop_factor"])
    rlop_patience = int(params["rlop_patience"])

    loss_name, criterion = build_loss_from_params(params, get_loss_function)
    train_loader, val_loader = _build_loaders(cfg, batch_size, seed=seed)

    model = UltraLightFCN(in_channels=cfg.in_channels, num_classes=cfg.num_classes, params=cfg.seg_params)
    load_pretrained_encoder_into_ultralight(model, cfg.phase3_last_ckpt, verbose=(seed == cfg.seeds[0]))
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
        factor=rlop_factor,
        patience=rlop_patience,
        threshold=1e-4,
        min_lr=1e-6,
    )

    last_k = deque(maxlen=cfg.avg_last_k)
    best_avg_last_k = -1.0
    best_epoch = -1
    best_val_soft = -1.0
    best_val_hard05 = -1.0
    epoch_rows: List[Dict[str, Any]] = []

    run_dir = os.path.join(cfg.out_root, f"trial_{candidate_id}", f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0

        for images, masks in tqdm(
            train_loader,
            desc=f"[trial {candidate_id} seed {seed}] Train {epoch + 1}/{cfg.epochs}",
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

            bs = int(images.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_n += bs

        train_loss = train_loss_sum / max(1, train_n)

        val_loss, val_soft, val_hard05 = _val_epoch(model, val_loader, cfg, use_amp=use_amp, criterion=criterion)
        scheduler.step(val_soft)

        last_k.append(val_soft)
        avg_last_k = float(sum(last_k) / len(last_k))

        if avg_last_k > best_avg_last_k:
            best_avg_last_k = avg_last_k
            best_epoch = epoch + 1
            best_val_soft = val_soft
            best_val_hard05 = val_hard05

        lr_enc = optimizer.param_groups[0]["lr"]
        lr_dec = optimizer.param_groups[1]["lr"]

        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_soft": val_soft,
                "val_hard05": val_hard05,
                "avg_last_k_soft": avg_last_k,
                "lr_enc": lr_enc,
                "lr_dec": lr_dec,
            }
        )

    # Save LAST (no checkpoint selection)
    ckpt_path = os.path.join(run_dir, cfg.save_last_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "candidate_id": candidate_id,
            "seed": seed,
            "params": params,
            "phase3_last_ckpt": cfg.phase3_last_ckpt,
            "epochs": cfg.epochs,
            "avg_last_k": cfg.avg_last_k,
            "timestamp": datetime.now().isoformat(),
        },
        ckpt_path,
    )

    # Save per-epoch log
    save_csv_rows(os.path.join(run_dir, cfg.save_epoch_log_name), epoch_rows)

    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "optuna_value_phase4": float(optuna_value),
        "loss_name": str(loss_name),
        "best_avg_last_k_soft": float(best_avg_last_k),
        "best_epoch": int(best_epoch),
        "best_val_soft": float(best_val_soft),
        "best_val_hard05": float(best_val_hard05),
        "final_val_soft": float(epoch_rows[-1]["val_soft"]),
        "final_val_hard05": float(epoch_rows[-1]["val_hard05"]),
        "ckpt_last_path": ckpt_path,
        "params": params,  # store so winner JSON can include exact params
    }


def load_topk_candidates(cfg: Phase5Config) -> List[Tuple[int, float, Dict[str, Any]]]:
    """Return list of (trial_number, trial_value, trial_params) for Top-K COMPLETE trials."""
    study = optuna.load_study(study_name=cfg.study_name, storage=cfg.storage)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("No COMPLETE trials found in Phase-4 study.")
    complete.sort(key=lambda t: float(t.value), reverse=True)
    top = complete[: cfg.top_k]
    return [(int(t.number), float(t.value), dict(t.params)) for t in top]


def write_phase5_summary(cfg: Phase5Config, rows: List[Dict[str, Any]]) -> str:
    os.makedirs(cfg.out_root, exist_ok=True)
    path = os.path.join(cfg.out_root, cfg.summary_csv_name)

    rows_sorted = sorted(rows, key=lambda r: float(r["best_avg_last_k_soft"]), reverse=True)
    keys = [
        "candidate_id",
        "seed",
        "optuna_value_phase4",
        "loss_name",
        "best_avg_last_k_soft",
        "best_epoch",
        "best_val_soft",
        "best_val_hard05",
        "final_val_soft",
        "final_val_hard05",
        "ckpt_last_path",
    ]
    save_csv_rows(path, rows_sorted, fieldnames=keys)
    return path


def pick_winner_and_save(cfg: Phase5Config, all_rows: List[Dict[str, Any]]) -> str:
    """Pick winner by mean(best_avg_last_k_soft) across seeds; tie-break by lower std, then higher best_val_soft."""
    per_cand: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        per_cand[int(r["candidate_id"])].append(r)

    scored: List[Dict[str, Any]] = []
    for cand_id, runs in per_cand.items():
        scores = [float(x["best_avg_last_k_soft"]) for x in runs]
        mu = mean(scores)
        sd = pstdev(scores) if len(scores) > 1 else 0.0
        best_val_soft_max = max(float(x["best_val_soft"]) for x in runs)

        # pick representative run for Phase-6: best score among seeds
        best_run = max(runs, key=lambda x: float(x["best_avg_last_k_soft"]))

        scored.append(
            {
                "candidate_id": cand_id,
                "mean_best_avg_last_k_soft": float(mu),
                "std_best_avg_last_k_soft": float(sd),
                "best_val_soft_max": float(best_val_soft_max),
                "best_run": {
                    "seed": int(best_run["seed"]),
                    "best_avg_last_k_soft": float(best_run["best_avg_last_k_soft"]),
                    "best_epoch": int(best_run["best_epoch"]),
                    "ckpt_last_path": str(best_run["ckpt_last_path"]),
                },
                "params": dict(best_run["params"]),
            }
        )

    # sort: higher mean, lower std, higher best_val_soft_max
    scored.sort(
        key=lambda x: (
            -float(x["mean_best_avg_last_k_soft"]),
            float(x["std_best_avg_last_k_soft"]),
            -float(x["best_val_soft_max"]),
        )
    )
    winner = scored[0]

    winner_obj = {
        "phase": 5,
        "selection_metric": "mean(best_avg_last_k_soft) over seeds; tie-break: min std, then max(best_val_soft)",
        "timestamp": datetime.now().isoformat(),
        "study_source": {"study_name": cfg.study_name, "storage": cfg.storage, "top_k": cfg.top_k},
        "data": {"train": cfg.train_split, "valid": cfg.val_split, "valid_full": True},
        "init": {"phase3_last_ckpt": cfg.phase3_last_ckpt},
        "training": {"epochs": cfg.epochs, "avg_last_k": cfg.avg_last_k, "fine_tune": True},
        "winner": winner,
        "ranking_topk": scored,  # full ranking for traceability (still compact)
    }

    path = os.path.join(cfg.out_root, cfg.winner_json_name)
    save_json(path, winner_obj)
    return path


def main() -> None:
    cfg = Phase5Config()
    os.makedirs(cfg.out_root, exist_ok=True)

    if not os.path.isfile(cfg.phase3_last_ckpt):
        raise RuntimeError(f"Phase-3 LAST checkpoint not found: {cfg.phase3_last_ckpt}")

    candidates = load_topk_candidates(cfg)
    print(f"Loaded Top-{len(candidates)} candidates from Phase-4 study '{cfg.study_name}'.")

    all_rows: List[Dict[str, Any]] = []

    for trial_num, trial_value, trial_params in candidates:
        print(f"\n=== Phase-5 retrain: candidate trial {trial_num} (Phase-4 value={trial_value:.6f}) ===")
        for seed in cfg.seeds:
            row = run_one_candidate_one_seed(cfg, trial_num, trial_params, trial_value, seed)
            # remove big 'params' from per-epoch csv rows but keep in row dict; OK as-is
            all_rows.append(row)
            clear_cuda_cache()  # works both as callback and as direct call (see helpers patch below)

    summary_path = write_phase5_summary(cfg, all_rows)
    print(f"\n✅ Phase-5 summary written to: {summary_path}")

    winner_path = pick_winner_and_save(cfg, all_rows)
    print(f"🏁 Winner written to: {winner_path}")

    # quick print
    top = sorted(all_rows, key=lambda r: float(r["best_avg_last_k_soft"]), reverse=True)[:3]
    print("\nTop-3 runs by best_avg_last_k_soft:")
    for r in top:
        print(
            f"  trial={r['candidate_id']} seed={r['seed']} "
            f"best_avg_last_k_soft={r['best_avg_last_k_soft']:.6f} best_val_soft={r['best_val_soft']:.6f}"
        )


if __name__ == "__main__":
    main()
