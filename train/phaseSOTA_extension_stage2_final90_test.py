from __future__ import annotations

import os
import csv
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from utils.dataset import SolarPanelDataset
from utils.helpers import (
    clear_cuda_cache,
    get_model,
    get_loss_function,
    save_csv_rows,
    save_json,
    build_loss_from_params,
)
from utils.metrics import calculate_dice, calculate_iou, calculate_precision_recall
from utils.repro import seed_worker, set_global_seed
from utils.sota_registry_extension import (
    SOTA_EXTENSION_MODELS,
    SOTA_REGIMES,
    split_smp_encoder_decoder_params,
)


TEST_METRIC_KEYS: Tuple[str, ...] = (
    "dice_soft",
    "dice_hard05",
    "iou_soft",
    "iou_hard05",
    "precision_hard05",
    "recall_hard05",
)

CSV_TEST_METRIC_KEYS: Tuple[str, ...] = tuple(f"test_{k}" for k in TEST_METRIC_KEYS)
SEED_CSV_FIELDNAMES: Tuple[str, ...] = (
    "model_name",
    "regime",
    "seed",
    "status",
    *CSV_TEST_METRIC_KEYS,
    "last_ckpt",
    "error",
    "trace_path",
)


# Extension Stage2 is restricted to minft-only because manuscript SOTA
# comparator reporting uses minft-only results; this avoids mixing
# fine-tuning regimes across comparator families.
SOTA_EXTENSION_STAGE2_REGIMES: Tuple[str, ...] = ("minft",)


@dataclass(frozen=True)
class SOTAExtensionStage2Config:
    data_root: str = "../dataset"
    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"

    epochs: int = 60
    hard_thr: float = 0.5

    num_workers: int = 8
    pin_memory: bool = True
    drop_last_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deterministic: bool = False
    strict: bool = False
    seeds: Tuple[int, ...] = (13, 37, 71)

    amp: bool = True
    grad_clip_norm: float = 5.0
    finite_loss_guard: bool = True

    lr_min: float = 1e-6

    # unified recipe
    phase5_winner_json: str = "seg_phase5/topk_retrain/phase5_winner.json"

    out_root: str = "seg_sota_extension/stage2_final90_test"
    seed_csv: str = "sota_extension_seed_runs.csv"
    report_json: str = "sota_extension_test_report.json"


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"JSON not found: {path}")
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _test_metrics_are_finite(metrics: Dict[str, Any]) -> bool:
    return all(_finite_float(metrics.get(k)) for k in TEST_METRIC_KEYS)


def _read_existing_seed_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return (str(row["model_name"]), str(row["regime"]), int(row["seed"]))


def _csv_row_has_valid_completed_run(row: Dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False

    last_ckpt = row.get("last_ckpt", "")
    if not last_ckpt or not os.path.isfile(last_ckpt):
        return False

    return all(_finite_float(row.get(k)) for k in CSV_TEST_METRIC_KEYS)


def _seed_result_from_valid_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": row["model_name"],
        "regime": row["regime"],
        "seed": int(row["seed"]),
        "status": "ok",
        "last_ckpt": row["last_ckpt"],
        "test": {k: float(row[f"test_{k}"]) for k in TEST_METRIC_KEYS},
    }


def _seed_result_to_csv_row(r: Dict[str, Any]) -> Dict[str, Any]:
    base = {"model_name": r["model_name"], "regime": r["regime"], "seed": r["seed"], "status": r["status"]}
    if r["status"] == "ok":
        base.update(
            {
                "test_dice_soft": r["test"]["dice_soft"],
                "test_dice_hard05": r["test"]["dice_hard05"],
                "test_iou_soft": r["test"]["iou_soft"],
                "test_iou_hard05": r["test"]["iou_hard05"],
                "test_precision_hard05": r["test"]["precision_hard05"],
                "test_recall_hard05": r["test"]["recall_hard05"],
                "last_ckpt": r["last_ckpt"],
            }
        )
    else:
        base.update(
            {
                "error": r.get("error", ""),
                "trace_path": r.get("trace_path", ""),
                "last_ckpt": r.get("last_ckpt", ""),
            }
        )
    return base


def _population_std(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def _build_train90_loader(cfg: SOTAExtensionStage2Config, batch_size: int, *, seed: int) -> DataLoader:
    train_dir = os.path.join(cfg.data_root, cfg.train_split)
    val_dir = os.path.join(cfg.data_root, cfg.val_split)

    ds_train = SolarPanelDataset(train_dir, mode="train", files=None, return_extra=False)
    ds_val_as_train = SolarPanelDataset(val_dir, mode="train", files=None, return_extra=False)
    ds_90 = ConcatDataset([ds_train, ds_val_as_train])

    pw = cfg.persistent_workers and (cfg.num_workers > 0)
    g = torch.Generator().manual_seed(seed)

    return DataLoader(
        ds_90,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last_train,
        persistent_workers=pw,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=g,
    )


def _build_test_loader(cfg: SOTAExtensionStage2Config, batch_size: int) -> DataLoader:
    test_dir = os.path.join(cfg.data_root, cfg.test_split)
    ds_test = SolarPanelDataset(test_dir, mode="test", files=None, return_extra=False)

    pw = cfg.persistent_workers and (cfg.num_workers > 0)

    return DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=pw,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
    )


def _build_smp_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    ModelClass = get_model(model_cfg["model_key"])
    model = ModelClass(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model


@torch.no_grad()
def eval_test_failfast(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: SOTAExtensionStage2Config,
) -> Dict[str, float]:
    model.eval()
    n = 0

    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    iou_soft_sum = 0.0
    iou_hard_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0

    use_amp = (cfg.amp and cfg.device.type == "cuda")
    for images, masks in loader:
        images = images.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)

        bs = int(images.shape[0])
        dice_soft_sum += float(calculate_dice(logits, masks, thr=None)) * bs
        dice_hard_sum += float(calculate_dice(logits, masks, thr=cfg.hard_thr)) * bs
        iou_soft_sum += float(calculate_iou(logits, masks, thr=None)) * bs
        iou_hard_sum += float(calculate_iou(logits, masks, thr=cfg.hard_thr)) * bs
        p, r = calculate_precision_recall(logits, masks, thr=cfg.hard_thr)
        prec_sum += float(p) * bs
        rec_sum += float(r) * bs
        n += bs

    if n == 0:
        raise RuntimeError("TEST FAIL-FAST: eval_n==0 (no samples processed).")

    return {
        "dice_soft": dice_soft_sum / n,
        "dice_hard05": dice_hard_sum / n,
        "iou_soft": iou_soft_sum / n,
        "iou_hard05": iou_hard_sum / n,
        "precision_hard05": prec_sum / n,
        "recall_hard05": rec_sum / n,
        "n": float(n),
    }


def train_one(
    cfg: SOTAExtensionStage2Config,
    phase5_winner: Dict[str, Any],
    model_name: str,
    regime_name: str,
    seed: int,
) -> Dict[str, Any]:
    set_global_seed(seed, deterministic=cfg.deterministic, strict=cfg.strict)

    params = dict(phase5_winner["winner"]["params"])
    batch_size = int(params["batch_size"])
    base_lr = float(params["base_lr"])
    weight_decay = float(params["weight_decay"])

    # FullFT uses Phase-5 enc_lr_mult; MinFT uses fixed 0.1
    enc_lr_mult = float(params["enc_lr_mult"])
    if regime_name == "minft":
        enc_lr_mult = float(SOTA_REGIMES["minft"]["enc_lr_mult"])

    loss_name, criterion = build_loss_from_params(params, get_loss_function)

    model_cfg = dict(SOTA_EXTENSION_MODELS[model_name])
    model = _build_smp_model(model_cfg).to(cfg.device)

    enc_params, dec_params = split_smp_encoder_decoder_params(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": base_lr * enc_lr_mult},
            {"params": dec_params, "lr": base_lr},
        ],
        lr=base_lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.epochs),
        eta_min=float(cfg.lr_min),
    )

    train_loader = _build_train90_loader(cfg, batch_size, seed=seed)
    test_loader = _build_test_loader(cfg, batch_size=batch_size)

    use_amp = (cfg.amp and cfg.device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    run_dir = os.path.join(cfg.out_root, model_name, regime_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        skipped_nonfinite = 0

        for images, masks in tqdm(
            train_loader,
            desc=f"[SOTA-Extension-Stage2] {model_name}/{regime_name} seed={seed} ep={epoch}",
            leave=False,
        ):
            images = images.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if cfg.finite_loss_guard and (not torch.isfinite(loss).all()):
                skipped_nonfinite += int(images.shape[0])
                continue

            scaler.scale(loss).backward()

            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))

            scaler.step(optimizer)
            scaler.update()

            bs = int(images.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_n += bs

        if train_n == 0:
            raise RuntimeError(f"TRAIN FAIL-FAST: train_n==0 at epoch {epoch} (skipped_nonfinite={skipped_nonfinite})")

        train_loss = train_loss_sum / train_n
        scheduler.step()

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_n": train_n,
                "skipped_nonfinite": skipped_nonfinite,
                "lr_enc": optimizer.param_groups[0]["lr"],
                "lr_dec": optimizer.param_groups[1]["lr"],
            }
        )

    # Save LAST (official artifact)
    last_path = os.path.join(run_dir, "last.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat(),
            "stage": 2,
            "split": "train+valid (90%)",
            "seed": seed,
            "model_name": model_name,
            "regime": regime_name,
            "model_cfg": model_cfg,
            "loss_name": loss_name,
            "params_unified": params,
            "enc_lr_mult_used": enc_lr_mult,
            "scheduler": {"name": "CosineAnnealingLR", "eta_min": cfg.lr_min, "T_max": cfg.epochs},
        },
        last_path,
    )

    save_csv_rows(os.path.join(run_dir, "epoch_log.csv"), epoch_rows)

    # Locked-box TEST evaluation
    test_metrics = eval_test_failfast(model, test_loader, cfg)
    if not _test_metrics_are_finite(test_metrics):
        bad_keys = [k for k in TEST_METRIC_KEYS if not _finite_float(test_metrics.get(k))]
        return {
            "model_name": model_name,
            "regime": regime_name,
            "seed": seed,
            "status": "failed_nonfinite_test_metric",
            "last_ckpt": last_path,
            "error": f"Non-finite TEST metric(s): {', '.join(bad_keys)}",
            "test": test_metrics,
        }

    return {
        "model_name": model_name,
        "regime": regime_name,
        "seed": seed,
        "status": "ok",
        "last_ckpt": last_path,
        "test": test_metrics,
    }


def main() -> None:
    cfg = SOTAExtensionStage2Config()
    os.makedirs(cfg.out_root, exist_ok=True)

    phase5_winner = _read_json(cfg.phase5_winner_json)
    seed_csv_path = os.path.join(cfg.out_root, cfg.seed_csv)
    existing_rows = _read_existing_seed_csv(seed_csv_path)
    existing_by_key = {_row_key(r): r for r in existing_rows if r.get("model_name") and r.get("regime") and r.get("seed")}

    seed_rows: List[Dict[str, Any]] = []
    for model_name in SOTA_EXTENSION_MODELS.keys():
        for regime_name in SOTA_EXTENSION_STAGE2_REGIMES:
            for seed in cfg.seeds:
                key = (model_name, regime_name, seed)
                existing_row = existing_by_key.get(key)
                if existing_row is not None and _csv_row_has_valid_completed_run(existing_row):
                    print(f"[SOTA Extension Stage2] Skip existing valid run: {model_name}/{regime_name}/seed_{seed}")
                    seed_rows.append(_seed_result_from_valid_csv_row(existing_row))
                    continue

                try:
                    row = train_one(cfg, phase5_winner, model_name, regime_name, seed)
                except Exception as e:
                    run_dir = os.path.join(cfg.out_root, model_name, regime_name, f"seed_{seed}")
                    os.makedirs(run_dir, exist_ok=True)
                    trace_path = os.path.join(run_dir, "error_trace.txt")
                    with open(trace_path, "w", encoding="utf-8") as f:
                        f.write(traceback.format_exc())
                    row = {
                        "model_name": model_name,
                        "regime": regime_name,
                        "seed": seed,
                        "status": "failed",
                        "error": str(e),
                        "trace_path": trace_path,
                    }
                seed_rows.append(row)
                clear_cuda_cache()

    # Seed CSV (flatten key metrics)
    csv_rows = [_seed_result_to_csv_row(r) for r in seed_rows]

    save_csv_rows(seed_csv_path, csv_rows, fieldnames=SEED_CSV_FIELDNAMES)

    # Aggregated report by (model, regime)
    def agg(ok_rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
        vals = [float(r["test"][key]) for r in ok_rows if _finite_float(r.get("test", {}).get(key))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": sum(vals) / len(vals), "std": _population_std(vals), "n": len(vals)}

    report = {
        "timestamp": datetime.now().isoformat(),
        "locked_box_test": True,
        "per_seed": seed_rows,
        "aggregate": {},
    }

    for model_name in SOTA_EXTENSION_MODELS.keys():
        for regime_name in SOTA_EXTENSION_STAGE2_REGIMES:
            ok = [
                r
                for r in seed_rows
                if (
                    r["model_name"] == model_name
                    and r["regime"] == regime_name
                    and r["status"] == "ok"
                    and _test_metrics_are_finite(r.get("test", {}))
                )
            ]
            report["aggregate"][f"{model_name}::{regime_name}"] = {
                "dice_soft": agg(ok, "dice_soft"),
                "dice_hard05": agg(ok, "dice_hard05"),
                "iou_soft": agg(ok, "iou_soft"),
                "iou_hard05": agg(ok, "iou_hard05"),
                "precision_hard05": agg(ok, "precision_hard05"),
                "recall_hard05": agg(ok, "recall_hard05"),
            }

    save_json(os.path.join(cfg.out_root, cfg.report_json), report)
    print(f"[SOTA Extension Stage2] Wrote: {os.path.join(cfg.out_root, cfg.seed_csv)}")
    print(f"[SOTA Extension Stage2] Wrote: {os.path.join(cfg.out_root, cfg.report_json)}")


if __name__ == "__main__":
    main()
