"""phase6_seg_final_retrain90_test.py

Phase 6 — Final retrain of Phase-5 winner on TRAIN+VALID (90%), 60 epochs, 10 seeds (13, 37, 71, 101, 131, 151, 181, 211, 241, 271).
- No checkpoint selection: save LAST only.
- No TEST usage during training.
- After training finishes: evaluate ONCE on TEST and report mean±std across seeds.

Notes:
- We initialize encoder from Phase-3 LAST checkpoint (same as Phase-5 winner init).
- Since there is no VALID split in Phase-6, ReduceLROnPlateau is driven by train_loss (mode="min").
  This keeps scheduler hyperparams consistent without introducing a validation proxy.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
from dataclasses import dataclass, field, replace
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
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
from utils.metrics import calculate_dice, calculate_iou, calculate_precision_recall
from utils.repro import seed_worker, set_global_seed


@dataclass(frozen=True)
class Phase6Config:
    # --------- Data
    data_root: str = "../dataset"
    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"

    # --------- Seeds
    seeds: Tuple[int, ...] = (13, 37, 71, 101, 131, 151, 181, 211, 241, 271, 307, 353, 409, 457, 521, 601, 701, 809, 907, 997)
    resume_existing: bool = True
    overwrite_existing_seeds: bool = False

    # --------- Training budget
    epochs: int = 60
    hard_thr_monitor: float = 0.5

    # --------- Model (fixed)
    in_channels: int = 3
    num_classes: int = 1
    seg_params: dict = field(default_factory=lambda: dict(SEG_PARAMS))

    # --------- Dataloaders
    num_workers: int = 8
    pin_memory: bool = True
    drop_last_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # --------- Runtime / Repro
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deterministic: bool = False
    strict: bool = False

    # --------- Inputs
    phase5_winner_json: str = "seg_phase5/topk_retrain/phase5_winner.json"

    # --------- Outputs
    out_root: str = "seg_phase6/final_retrain90"
    save_last_name: str = "last.pth"
    save_epoch_log_name: str = "epoch_log.csv"
    per_seed_test_csv_name: str = "phase6_test_per_seed.csv"
    report_json_name: str = "phase6_test_report.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _resolve_device_from_arg(device_arg: str | None, default_device: torch.device) -> torch.device:
    if device_arg is None:
        return default_device

    try:
        device = torch.device(device_arg)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Invalid device argument: {device_arg}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device is unavailable because CUDA is not available: {device_arg}")
        if device.index is not None and not (0 <= device.index < torch.cuda.device_count()):
            raise RuntimeError(
                f"Requested CUDA device is unavailable: {device_arg}. "
                f"Available CUDA device count: {torch.cuda.device_count()}"
            )

    return device


def _read_json(path: str) -> Dict[str, Any]:
    import json
    with open(path, "r") as f:
        return json.load(f)


def _read_existing_seed_rows(csv_path: str) -> Dict[int, Dict[str, Any]]:
    if not os.path.isfile(csv_path):
        return {}

    rows_by_seed: Dict[int, Dict[str, Any]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            try:
                seed = int(row.get("seed", ""))
            except (TypeError, ValueError):
                continue
            rows_by_seed[seed] = dict(row)
    return rows_by_seed


def _row_has_existing_checkpoint(row: Dict[str, Any]) -> bool:
    try:
        int(row.get("seed", ""))
    except (TypeError, ValueError):
        return False

    ckpt_last_path = str(row.get("ckpt_last_path", "")).strip()
    if not ckpt_last_path:
        return False
    return os.path.exists(ckpt_last_path)


def _merge_seed_rows(
    existing_rows: Dict[int, Dict[str, Any]],
    new_rows: Dict[int, Dict[str, Any]],
    seeds: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    merged_by_seed: Dict[int, Dict[str, Any]] = {}
    merged_by_seed.update(existing_rows)
    merged_by_seed.update(new_rows)

    requested_seeds = sorted({int(seed) for seed in seeds})
    return [merged_by_seed[seed] for seed in requested_seeds if seed in merged_by_seed]


def _resolve_dataloader_runtime(cfg: Phase6Config) -> tuple[int, bool, int | None]:
    # Windows multiprocessing compatibility safeguard:
    # spawned workers re-import torch and can hit DLL init failures, so force single-process loading.
    if platform.system().lower().startswith("win"):
        return 0, False, None
    return cfg.num_workers, cfg.persistent_workers, cfg.prefetch_factor


def _build_report(
    cfg: Phase6Config,
    *,
    candidate_id: int,
    phase3_last_ckpt: str,
    rows: List[Dict[str, Any]],
    skipped_existing_seeds: List[int],
    newly_trained_rows: Dict[int, Dict[str, Any]],
    per_seed_csv: str,
) -> Dict[str, Any]:
    completed_seeds = [int(r["seed"]) for r in rows]
    newly_trained_seeds = sorted(newly_trained_rows.keys())
    report: Dict[str, Any] = {
        "phase": 6,
        "timestamp": datetime.now().isoformat(),
        "data": {"train": "train+valid (90%)", "test": cfg.test_split, "test_used_during_training": False},
        "training": {
            "epochs": cfg.epochs,
            "seeds": list(cfg.seeds),
            "save": "LAST only",
            "resume_existing": cfg.resume_existing,
            "overwrite_existing_seeds": cfg.overwrite_existing_seeds,
        },
        "seed_status": {
            "requested_seeds": list(cfg.seeds),
            "completed_seeds": completed_seeds,
            "skipped_existing_seeds": sorted(skipped_existing_seeds),
            "newly_trained_seeds": newly_trained_seeds,
            "resume_existing": cfg.resume_existing,
            "overwrite_existing_seeds": cfg.overwrite_existing_seeds,
        },
        "winner_source": {"phase5_winner_json": cfg.phase5_winner_json, "candidate_id": candidate_id},
        "init": {"phase3_last_ckpt": phase3_last_ckpt},
        "runs": rows,
        "artifacts": {"per_seed_csv": per_seed_csv},
    }

    if rows:
        report["metrics_test"] = {
            "soft_dice": _mean_std([float(r["test_soft_dice"]) for r in rows]),
            "hard_dice@0.5": _mean_std([float(r["test_hard_dice@0.5"]) for r in rows]),
            "soft_iou": _mean_std([float(r["test_soft_iou"]) for r in rows]),
            "hard_iou@0.5": _mean_std([float(r["test_hard_iou@0.5"]) for r in rows]),
            "precision@0.5": _mean_std([float(r["test_precision@0.5"]) for r in rows]),
            "recall@0.5": _mean_std([float(r["test_recall@0.5"]) for r in rows]),
            "loss": _mean_std([float(r["test_loss"]) for r in rows]),
        }

    return report


def _save_incremental_outputs(
    cfg: Phase6Config,
    *,
    candidate_id: int,
    phase3_last_ckpt: str,
    reusable_existing_rows: Dict[int, Dict[str, Any]],
    newly_trained_rows: Dict[int, Dict[str, Any]],
    skipped_existing_seeds: List[int],
    per_seed_csv: str,
    report_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = _merge_seed_rows(reusable_existing_rows, newly_trained_rows, cfg.seeds)

    save_csv_rows(
        per_seed_csv,
        rows,
        fieldnames=[
            "seed",
            "candidate_id",
            "loss_name",
            "test_loss",
            "test_soft_dice",
            "test_hard_dice@0.5",
            "test_soft_iou",
            "test_hard_iou@0.5",
            "test_precision@0.5",
            "test_recall@0.5",
            "ckpt_last_path",
        ],
    )

    report = _build_report(
        cfg,
        candidate_id=candidate_id,
        phase3_last_ckpt=phase3_last_ckpt,
        rows=rows,
        skipped_existing_seeds=skipped_existing_seeds,
        newly_trained_rows=newly_trained_rows,
        per_seed_csv=per_seed_csv,
    )
    save_json(report_path, report)
    return rows, report


def _build_train_loader_90(cfg: Phase6Config, batch_size: int, *, seed: int) -> DataLoader:
    """TRAIN+VALID (90%) concatenated, both using TRAIN transforms."""
    train_dir = os.path.join(cfg.data_root, cfg.train_split)
    val_dir = os.path.join(cfg.data_root, cfg.val_split)

    ds_train = SolarPanelDataset(train_dir, mode="train", files=None, return_extra=False)
    ds_val_as_train = SolarPanelDataset(val_dir, mode="train", files=None, return_extra=False)
    train90_ds = ConcatDataset([ds_train, ds_val_as_train])

    num_workers, persistent_workers, prefetch_factor = _resolve_dataloader_runtime(cfg)
    pw = persistent_workers and (num_workers > 0)
    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        train90_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last_train,
        persistent_workers=pw,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
    return loader


def _build_test_loader(cfg: Phase6Config, batch_size: int) -> DataLoader:
    test_dir = os.path.join(cfg.data_root, cfg.test_split)
    test_ds = SolarPanelDataset(test_dir, mode="test", files=None, return_extra=False)

    num_workers, persistent_workers, prefetch_factor = _resolve_dataloader_runtime(cfg)
    pw = persistent_workers and (num_workers > 0)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=pw,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    return loader


def _eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Phase6Config,
    *,
    use_amp: bool,
    criterion,
) -> Dict[str, float]:
    """Aggregated per-image metrics over full loader (bs-weighted)."""
    model.eval()

    loss_sum = 0.0
    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    iou_soft_sum = 0.0
    iou_hard_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    n = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            # Safety: should not happen if training is stable, but avoids crashing eval
            if not torch.isfinite(loss):
                continue

            bs = int(images.shape[0])
            loss_sum += float(loss.detach().cpu()) * bs

            # Dice
            dice_soft = float(calculate_dice(logits, masks, thr=None))
            dice_hard = float(calculate_dice(logits, masks, thr=cfg.hard_thr_monitor))
            dice_soft_sum += dice_soft * bs
            dice_hard_sum += dice_hard * bs

            # IoU
            iou_soft = float(calculate_iou(logits, masks, thr=None))
            iou_hard = float(calculate_iou(logits, masks, thr=cfg.hard_thr_monitor))
            iou_soft_sum += iou_soft * bs
            iou_hard_sum += iou_hard * bs

            # Precision / Recall (hard threshold)
            prec, rec = calculate_precision_recall(logits, masks, thr=cfg.hard_thr_monitor)
            prec_sum += float(prec) * bs
            rec_sum += float(rec) * bs

            n += bs

    n = max(1, n)
    return {
        "loss": loss_sum / n,
        "soft_dice": dice_soft_sum / n,
        "hard_dice@0.5": dice_hard_sum / n,
        "soft_iou": iou_soft_sum / n,
        "hard_iou@0.5": iou_hard_sum / n,
        "precision@0.5": prec_sum / n,
        "recall@0.5": rec_sum / n,
    }


def _mean_std(xs: List[float]) -> Dict[str, float]:
    if len(xs) == 1:
        return {"mean": float(xs[0]), "std": 0.0}
    return {"mean": float(mean(xs)), "std": float(pstdev(xs))}


def run_one_seed(
    cfg: Phase6Config,
    *,
    candidate_id: int,
    params: Dict[str, Any],
    phase3_last_ckpt: str,
    seed: int,
) -> Dict[str, Any]:
    set_global_seed(seed, deterministic=cfg.deterministic, strict=cfg.strict)

    batch_size = int(params["batch_size"])
    base_lr = float(params["base_lr"])
    enc_lr_mult = float(params["enc_lr_mult"])
    weight_decay = float(params["weight_decay"])
    rlop_factor = float(params["rlop_factor"])
    rlop_patience = int(params["rlop_patience"])

    loss_name, criterion = build_loss_from_params(params, get_loss_function)

    train_loader = _build_train_loader_90(cfg, batch_size, seed=seed)
    test_loader = _build_test_loader(cfg, batch_size=batch_size)

    model = UltraLightFCN(in_channels=cfg.in_channels, num_classes=cfg.num_classes, params=cfg.seg_params)
    load_pretrained_encoder_into_ultralight(model, phase3_last_ckpt, verbose=(seed == cfg.seeds[0]))
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    run_dir = os.path.join(cfg.out_root, f"trial_{candidate_id}", f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    epoch_rows: List[Dict[str, Any]] = []

    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0

        for images, masks in tqdm(
            train_loader,
            desc=f"[Phase6 trial {candidate_id} seed {seed}] Train {epoch + 1}/{cfg.epochs}",
            leave=False,
        ):
            images = images.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            # ---- stability guard: do not propagate NaNs/Infs
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            # ---- AMP-safe grad clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()

            bs = int(images.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_n += bs

        train_loss = train_loss_sum / max(1, train_n)
        scheduler.step()

        lr_enc = optimizer.param_groups[0]["lr"]
        lr_dec = optimizer.param_groups[1]["lr"]

        epoch_rows.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "lr_enc": lr_enc, "lr_dec": lr_dec}
        )

    # Save LAST only
    ckpt_path = os.path.join(run_dir, cfg.save_last_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "phase": 6,
            "candidate_id": candidate_id,
            "seed": seed,
            "params": params,
            "phase3_last_ckpt": phase3_last_ckpt,
            "epochs": cfg.epochs,
            "timestamp": datetime.now().isoformat(),
            "train_data": "train+valid (90%)",
            "no_checkpoint_selection": True,
        },
        ckpt_path,
    )

    save_csv_rows(os.path.join(run_dir, cfg.save_epoch_log_name), epoch_rows)

    # 1× TEST evaluation after training
    test_metrics = _eval_epoch(model, test_loader, cfg, use_amp=use_amp, criterion=criterion)

    return {
        "seed": seed,
        "candidate_id": candidate_id,
        "loss_name": str(loss_name),
        "ckpt_last_path": ckpt_path,
        "test_loss": float(test_metrics["loss"]),
        "test_soft_dice": float(test_metrics["soft_dice"]),
        "test_hard_dice@0.5": float(test_metrics["hard_dice@0.5"]),
        "test_soft_iou": float(test_metrics["soft_iou"]),
        "test_hard_iou@0.5": float(test_metrics["hard_iou@0.5"]),
        "test_precision@0.5": float(test_metrics["precision@0.5"]),
        "test_recall@0.5": float(test_metrics["recall@0.5"]),
    }


def main() -> None:
    args = _parse_args()
    cfg = Phase6Config()
    cfg = replace(cfg, device=_resolve_device_from_arg(args.device, cfg.device))
    print(f"Using device: {cfg.device}")
    os.makedirs(cfg.out_root, exist_ok=True)
    per_seed_csv = os.path.join(cfg.out_root, cfg.per_seed_test_csv_name)

    w = _read_json(cfg.phase5_winner_json)
    winner = w["winner"]
    candidate_id = int(winner["candidate_id"])
    params = dict(winner["params"])
    phase3_last_ckpt = str(w["init"]["phase3_last_ckpt"])

    if not os.path.isfile(phase3_last_ckpt):
        raise RuntimeError(f"Phase-3 LAST checkpoint not found: {phase3_last_ckpt}")

    existing_rows = _read_existing_seed_rows(per_seed_csv)
    reusable_existing_rows = {
        seed: row
        for seed, row in existing_rows.items()
        if cfg.resume_existing and _row_has_existing_checkpoint(row)
    }

    report_path = os.path.join(cfg.out_root, cfg.report_json_name)
    skipped_existing_seeds: List[int] = [
        int(seed)
        for seed in cfg.seeds
        if (not cfg.overwrite_existing_seeds) and (int(seed) in reusable_existing_rows)
    ]
    newly_trained_rows: Dict[int, Dict[str, Any]] = {}
    if reusable_existing_rows:
        _save_incremental_outputs(
            cfg,
            candidate_id=candidate_id,
            phase3_last_ckpt=phase3_last_ckpt,
            reusable_existing_rows=reusable_existing_rows,
            newly_trained_rows=newly_trained_rows,
            skipped_existing_seeds=skipped_existing_seeds,
            per_seed_csv=per_seed_csv,
            report_path=report_path,
        )

    for seed in cfg.seeds:
        seed = int(seed)
        if (not cfg.overwrite_existing_seeds) and (seed in reusable_existing_rows):
            continue

        newly_trained_rows[seed] = run_one_seed(
            cfg,
            candidate_id=candidate_id,
            params=params,
            phase3_last_ckpt=phase3_last_ckpt,
            seed=seed,
        )
        rows, report = _save_incremental_outputs(
            cfg,
            candidate_id=candidate_id,
            phase3_last_ckpt=phase3_last_ckpt,
            reusable_existing_rows=reusable_existing_rows,
            newly_trained_rows=newly_trained_rows,
            skipped_existing_seeds=skipped_existing_seeds,
            per_seed_csv=per_seed_csv,
            report_path=report_path,
        )
        clear_cuda_cache()

    rows, report = _save_incremental_outputs(
        cfg,
        candidate_id=candidate_id,
        phase3_last_ckpt=phase3_last_ckpt,
        reusable_existing_rows=reusable_existing_rows,
        newly_trained_rows=newly_trained_rows,
        skipped_existing_seeds=skipped_existing_seeds,
        per_seed_csv=per_seed_csv,
        report_path=report_path,
    )

    print(f"✅ Phase-6 per-seed TEST CSV: {per_seed_csv}")
    print(f"✅ Phase-6 TEST report JSON: {report_path}")
    if "metrics_test" in report:
        print(f"📌 TEST soft Dice mean±std: {report['metrics_test']['soft_dice']}")


if __name__ == "__main__":
    main()
