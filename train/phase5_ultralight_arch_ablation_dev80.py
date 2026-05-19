from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import SolarPanelDataset
from utils.helpers import (
    build_loss_from_params,
    clear_cuda_cache,
    get_loss_function,
    save_csv_rows,
    save_json,
    split_encoder_decoder_params,
)
from utils.load_simclr_pretrain_encoder_experimental import (
    load_pretrained_encoder_into_ultralight_experimental,
)
from utils.metrics import calculate_dice
from utils.repro import seed_worker, set_global_seed
from utils.ultralight_variant_registry import (
    build_ultralight_variant,
    get_ultralight_variant_spec,
    list_ultralight_variant_names,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKBONE_PREFIXES: tuple[str, ...] = ("block1", "dsconv2", "dsconv3", "dilconv4", "dilconv5")
DEFAULT_ABLATION_VARIANTS: tuple[str, ...] = (
    "no_mini_aspp",
    "no_shifted_sa",
    "no_mini_aspp_no_sa",
    "no_shallow_skip",
    "no_dilation",
    "decoder_narrow",
    "decoder_wide",
)
PHASE5_BASELINE_REFERENCE_DIR = REPO_ROOT / "train" / "seg_phase5" / "topk_retrain" / "trial_54"


@dataclass(frozen=True)
class AblationStage1Config:
    data_root: Path = REPO_ROOT / "dataset"
    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"

    phase5_winner_json: Path = REPO_ROOT / "train" / "seg_phase5" / "topk_retrain" / "phase5_winner.json"
    phase3_last_ckpt: Path | None = None

    in_channels: int = 3
    num_classes: int = 1
    variants: Tuple[str, ...] = DEFAULT_ABLATION_VARIANTS
    seeds: Tuple[int, ...] = (13, 37, 71)

    epochs: int = 2
    avg_last_k: int = 10
    hard_thr_monitor: float = 0.5

    num_workers: int = 8
    pin_memory: bool = True
    drop_last_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deterministic: bool = False
    strict: bool = False
    amp: bool = True
    grad_clip_norm: float = 5.0
    finite_loss_guard: bool = True

    out_root: Path = REPO_ROOT / "train" / "seg_experimental_ablation" / "stage1_dev80"
    save_last_name: str = "last.pth"
    save_epoch_log_name: str = "epoch_log.csv"
    save_audit_name: str = "simclr_load_audit.json"
    seed_metrics_name: str = "seed_metrics.json"
    seed_runs_csv_name: str = "arch_ablation_stage1_seed_runs.csv"
    aggregate_csv_name: str = "arch_ablation_stage1_aggregate.csv"
    report_json_name: str = "arch_ablation_stage1_report.json"


STAGE1_REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "variant_name",
    "seed",
    "status",
    "loss_name",
    "best_avg_last_k_soft",
    "best_epoch",
    "best_val_soft",
    "best_val_hard05",
    "final_val_loss",
    "final_val_soft",
    "final_val_hard05",
    "ckpt_last_path",
    "simclr_load_audit_path",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _resolve_device(device_arg: str | None, default_device: torch.device) -> torch.device:
    if device_arg is None:
        return default_device

    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device is unavailable: {device_arg}")
        if device.index is not None and not (0 <= device.index < torch.cuda.device_count()):
            raise RuntimeError(
                f"Requested CUDA device is unavailable: {device_arg}. "
                f"Available CUDA device count: {torch.cuda.device_count()}"
            )
    return device


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_read_json(path: Path) -> Dict[str, Any] | None:
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError, RuntimeError):
        return None


def _resolve_existing_path(raw_path: str, *, relative_anchor: Path) -> Path:
    path_obj = Path(raw_path)
    candidates = []
    if path_obj.is_absolute():
        candidates.append(path_obj)
    else:
        candidates.extend(
            [
                (relative_anchor / path_obj),
                (REPO_ROOT / path_obj),
                (REPO_ROOT / "train" / path_obj),
                (cfg_path := REPO_ROOT / raw_path.lstrip("./")),
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    return (relative_anchor / path_obj).resolve()


def _resolve_phase3_ckpt(cfg: AblationStage1Config, winner_obj: Dict[str, Any]) -> Path:
    if cfg.phase3_last_ckpt is not None:
        return cfg.phase3_last_ckpt

    phase3 = winner_obj.get("init", {}).get("phase3_last_ckpt")
    if not phase3:
        raise RuntimeError("Phase-5 winner JSON does not contain init.phase3_last_ckpt.")
    return _resolve_existing_path(str(phase3), relative_anchor=REPO_ROOT / "train")


def _validate_required_paths(cfg: AblationStage1Config, phase3_last_ckpt: Path) -> None:
    if not cfg.phase5_winner_json.is_file():
        raise RuntimeError(f"Phase-5 winner JSON not found: {cfg.phase5_winner_json}")
    if not phase3_last_ckpt.is_file():
        raise RuntimeError(f"Phase-3 LAST checkpoint not found: {phase3_last_ckpt}")

    required_dirs = [
        cfg.data_root / cfg.train_split,
        cfg.data_root / cfg.val_split,
        cfg.data_root / cfg.test_split,
    ]
    for directory in required_dirs:
        if not directory.is_dir():
            raise RuntimeError(f"Required dataset directory not found: {directory}")


def _dataloader_kwargs(
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    worker_init_fn,
    generator: torch.Generator | None = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["worker_init_fn"] = worker_init_fn
    if generator is not None:
        kwargs["generator"] = generator
    return kwargs


def _build_loaders(cfg: AblationStage1Config, batch_size: int, *, seed: int) -> tuple[DataLoader, DataLoader]:
    train_ds = SolarPanelDataset(
        str(cfg.data_root / cfg.train_split),
        mode="train",
        files=None,
        return_extra=False,
    )
    val_ds = SolarPanelDataset(
        str(cfg.data_root / cfg.val_split),
        mode="valid",
        files=None,
        return_extra=False,
    )

    pw = cfg.persistent_workers and (cfg.num_workers > 0)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        **_dataloader_kwargs(
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last_train,
            persistent_workers=pw,
            prefetch_factor=cfg.prefetch_factor,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
    )
    val_loader = DataLoader(
        val_ds,
        **_dataloader_kwargs(
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
            persistent_workers=pw,
            prefetch_factor=cfg.prefetch_factor,
            worker_init_fn=seed_worker,
        ),
    )
    return train_loader, val_loader


def _validate_loader_audit(audit: Dict[str, Any]) -> None:
    if int(audit.get("skipped_shape_key_count", 0)) > 0:
        raise RuntimeError(f"Loader audit reported shape-mismatched keys: {audit}")

    loaded_prefixes = set(audit.get("loaded_prefixes", []))
    missing_backbone = [prefix for prefix in BACKBONE_PREFIXES if prefix not in loaded_prefixes]
    if missing_backbone:
        raise RuntimeError(
            f"Loader audit missing required backbone prefixes {missing_backbone}: {audit}"
        )


def _eval_val(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: AblationStage1Config,
    *,
    criterion,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    soft_sum = 0.0
    hard_sum = 0.0
    n = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                continue

            bs = int(images.shape[0])
            loss_sum += float(loss.detach().cpu()) * bs
            soft_sum += float(calculate_dice(logits, masks, thr=None)) * bs
            hard_sum += float(calculate_dice(logits, masks, thr=cfg.hard_thr_monitor)) * bs
            n += bs

    if n == 0:
        raise RuntimeError("VALID FAIL-FAST: eval_n==0 (no samples processed).")

    return {
        "val_loss": loss_sum / n,
        "val_soft_dice": soft_sum / n,
        "val_hard_dice@0.5": hard_sum / n,
        "val_n": float(n),
    }


def _stage1_run_dir(cfg: AblationStage1Config, variant_name: str, seed: int) -> Path:
    return cfg.out_root / variant_name / f"seed_{seed}"


def _count_csv_data_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)
    except OSError:
        return None


def _read_completed_seed_row(
    cfg: AblationStage1Config,
    *,
    variant_name: str,
    seed: int,
) -> Dict[str, Any] | None:
    run_dir = _stage1_run_dir(cfg, variant_name, seed)
    ckpt_path = run_dir / cfg.save_last_name
    epoch_log_path = run_dir / cfg.save_epoch_log_name
    audit_path = run_dir / cfg.save_audit_name
    metrics_path = run_dir / cfg.seed_metrics_name

    required_paths = (ckpt_path, epoch_log_path, audit_path, metrics_path)
    if not all(path.is_file() for path in required_paths):
        return None

    row_count = _count_csv_data_rows(epoch_log_path)
    if row_count != cfg.epochs:
        return None

    metrics = _try_read_json(metrics_path)
    if not isinstance(metrics, dict):
        return None

    missing_fields = [field for field in STAGE1_REQUIRED_ROW_FIELDS if field not in metrics]
    if missing_fields:
        return None

    if str(metrics.get("variant_name")) != variant_name:
        return None
    try:
        if int(metrics.get("seed")) != int(seed):
            return None
    except (TypeError, ValueError):
        return None

    return dict(metrics)


def _build_stage1_aggregate_rows(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregate_rows: List[Dict[str, Any]] = []
    for variant_name in DEFAULT_ABLATION_VARIANTS:
        rows = [row for row in seed_rows if row.get("variant_name") == variant_name]
        if not rows:
            continue
        aggregate_rows.append(
            {
                "variant_name": variant_name,
                "n_seeds": len(rows),
                "mean_best_avg_last_k_soft": _mean_std([float(r["best_avg_last_k_soft"]) for r in rows])["mean"],
                "std_best_avg_last_k_soft": _mean_std([float(r["best_avg_last_k_soft"]) for r in rows])["std"],
                "mean_best_val_soft": _mean_std([float(r["best_val_soft"]) for r in rows])["mean"],
                "std_best_val_soft": _mean_std([float(r["best_val_soft"]) for r in rows])["std"],
                "mean_best_val_hard05": _mean_std([float(r["best_val_hard05"]) for r in rows])["mean"],
                "std_best_val_hard05": _mean_std([float(r["best_val_hard05"]) for r in rows])["std"],
                "mean_final_val_soft": _mean_std([float(r["final_val_soft"]) for r in rows])["mean"],
                "std_final_val_soft": _mean_std([float(r["final_val_soft"]) for r in rows])["std"],
                "mean_final_val_hard05": _mean_std([float(r["final_val_hard05"]) for r in rows])["mean"],
                "std_final_val_hard05": _mean_std([float(r["final_val_hard05"]) for r in rows])["std"],
            }
        )
    return aggregate_rows


def _refresh_stage_outputs(
    cfg: AblationStage1Config,
    *,
    winner_params: Dict[str, Any],
    phase3_last_ckpt: Path,
    seed_rows: List[Dict[str, Any]],
) -> None:
    aggregate_rows = _build_stage1_aggregate_rows(seed_rows)
    save_csv_rows(
        str(cfg.out_root / cfg.seed_runs_csv_name),
        seed_rows,
        fieldnames=list(STAGE1_REQUIRED_ROW_FIELDS),
    )
    save_csv_rows(str(cfg.out_root / cfg.aggregate_csv_name), aggregate_rows)
    save_json(
        str(cfg.out_root / cfg.report_json_name),
        _build_stage_report(
            cfg,
            winner_params=winner_params,
            phase3_last_ckpt=phase3_last_ckpt,
            seed_rows=seed_rows,
            aggregate_rows=aggregate_rows,
        ),
    )


def _save_run_artifacts(
    run_dir: Path,
    *,
    epoch_rows: List[Dict[str, Any]],
    audit: Dict[str, Any],
    checkpoint_obj: Dict[str, Any],
    seed_row: Dict[str, Any],
    cfg: AblationStage1Config,
) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_csv_rows(str(run_dir / cfg.save_epoch_log_name), epoch_rows)
    save_json(str(run_dir / cfg.save_audit_name), audit)
    ckpt_path = run_dir / cfg.save_last_name
    torch.save(checkpoint_obj, ckpt_path)
    save_json(str(run_dir / cfg.seed_metrics_name), seed_row)
    return str(ckpt_path)


def _mean_std(values: List[float]) -> Dict[str, float]:
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "n": 1}
    return {"mean": float(mean(values)), "std": float(pstdev(values)), "n": len(values)}


def run_one_variant_seed(
    cfg: AblationStage1Config,
    *,
    variant_name: str,
    winner_params: Dict[str, Any],
    phase3_last_ckpt: Path,
    seed: int,
) -> Dict[str, Any]:
    set_global_seed(seed, deterministic=cfg.deterministic, strict=cfg.strict)

    batch_size = int(winner_params["batch_size"])
    base_lr = float(winner_params["base_lr"])
    enc_lr_mult = float(winner_params["enc_lr_mult"])
    weight_decay = float(winner_params["weight_decay"])
    rlop_factor = float(winner_params["rlop_factor"])
    rlop_patience = int(winner_params["rlop_patience"])

    loss_name, criterion = build_loss_from_params(winner_params, get_loss_function)
    train_loader, val_loader = _build_loaders(cfg, batch_size, seed=seed)

    model = build_ultralight_variant(
        variant_name=variant_name,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
    )
    audit = load_pretrained_encoder_into_ultralight_experimental(
        model=model,
        ckpt_path=str(phase3_last_ckpt),
        variant_name=variant_name,
        verbose=(seed == cfg.seeds[0]),
    )
    _validate_loader_audit(audit)
    model = model.to(cfg.device)

    use_amp = bool(cfg.amp and cfg.device.type == "cuda")
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
    best_val_hard = -1.0
    epoch_rows: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        skipped_nonfinite = 0

        for images, masks in tqdm(
            train_loader,
            desc=f"[ArchAblation-Stage1] {variant_name} seed={seed} ep={epoch}",
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

            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))

            scaler.step(optimizer)
            scaler.update()

            bs = int(images.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_n += bs

        if train_n == 0:
            raise RuntimeError(
                f"TRAIN FAIL-FAST: train_n==0 for variant={variant_name} seed={seed} epoch={epoch} "
                f"(skipped_nonfinite={skipped_nonfinite})"
            )

        train_loss = train_loss_sum / train_n
        val_metrics = _eval_val(model, val_loader, cfg, criterion=criterion, use_amp=use_amp)
        scheduler.step(float(val_metrics["val_soft_dice"]))

        last_k.append(float(val_metrics["val_soft_dice"]))
        avg_last_k_soft = float(sum(last_k) / len(last_k))
        if avg_last_k_soft > best_avg_last_k:
            best_avg_last_k = avg_last_k_soft
            best_epoch = epoch
            best_val_soft = float(val_metrics["val_soft_dice"])
            best_val_hard = float(val_metrics["val_hard_dice@0.5"])

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_n": train_n,
                "skipped_nonfinite": skipped_nonfinite,
                "val_loss": float(val_metrics["val_loss"]),
                "val_soft_dice": float(val_metrics["val_soft_dice"]),
                "val_hard_dice@0.5": float(val_metrics["val_hard_dice@0.5"]),
                "avg_last_k_soft": avg_last_k_soft,
                "lr_enc": float(optimizer.param_groups[0]["lr"]),
                "lr_dec": float(optimizer.param_groups[1]["lr"]),
            }
        )

    spec = get_ultralight_variant_spec(variant_name)
    run_dir = _stage1_run_dir(cfg, variant_name, seed)
    seed_row = {
        "variant_name": variant_name,
        "seed": seed,
        "status": "ok",
        "loss_name": str(loss_name),
        "best_avg_last_k_soft": float(best_avg_last_k),
        "best_epoch": int(best_epoch),
        "best_val_soft": float(best_val_soft),
        "best_val_hard05": float(best_val_hard),
        "final_val_loss": float(epoch_rows[-1]["val_loss"]),
        "final_val_soft": float(epoch_rows[-1]["val_soft_dice"]),
        "final_val_hard05": float(epoch_rows[-1]["val_hard_dice@0.5"]),
        "ckpt_last_path": str(run_dir / cfg.save_last_name),
        "simclr_load_audit_path": str(run_dir / cfg.save_audit_name),
    }
    checkpoint_path = _save_run_artifacts(
        run_dir,
        epoch_rows=epoch_rows,
        audit=audit,
        checkpoint_obj={
            "model_state_dict": model.state_dict(),
            "variant_name": variant_name,
            "seed": seed,
            "params": spec.params,
            "phase5_winner_json": str(cfg.phase5_winner_json),
            "phase3_last_ckpt": str(phase3_last_ckpt),
            "epochs": cfg.epochs,
            "timestamp": datetime.now().isoformat(),
            "simclr_load_audit_summary": {
                "loaded_key_count": audit["loaded_key_count"],
                "skipped_missing_key_count": audit["skipped_missing_key_count"],
                "skipped_shape_key_count": audit["skipped_shape_key_count"],
                "loaded_prefixes": audit["loaded_prefixes"],
                "skipped_prefixes": audit["skipped_prefixes"],
            },
            "fixed_recipe_params": dict(winner_params),
            "stage": "stage1_dev80",
            "train_split": cfg.train_split,
            "val_split": cfg.val_split,
            "test_used_during_training": False,
            "selection_note": "This run summarizes VALID behavior only and does not auto-select an architecture for TEST.",
        },
        seed_row=seed_row,
        cfg=cfg,
    )
    seed_row["ckpt_last_path"] = checkpoint_path
    return seed_row


def _build_stage_report(
    cfg: AblationStage1Config,
    *,
    winner_params: Dict[str, Any],
    phase3_last_ckpt: Path,
    seed_rows: List[Dict[str, Any]],
    aggregate_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "stage": "stage1_dev80",
        "timestamp": datetime.now().isoformat(),
        "locked_box_test": True,
        "test_used_during_training": False,
        "selection_note": (
            "This report summarizes VALID-only architecture ablation behavior under the fixed "
            "Phase-5 winner recipe. This runner trains only ablated variants by default, reuses "
            "the full UltraLightFCN SimCLR baseline reference from train/seg_phase5/topk_retrain/trial_54/, "
            "and does not auto-select architectures for TEST."
        ),
        "baseline_reference": {
            "reused": True,
            "source_dir": str(PHASE5_BASELINE_REFERENCE_DIR),
            "note": "The full UltraLightFCN SimCLR baseline is reused from the existing Phase-5 baseline runs.",
        },
        "data": {
            "data_root": str(cfg.data_root),
            "train_split": cfg.train_split,
            "val_split": cfg.val_split,
            "test_split": cfg.test_split,
        },
        "recipe_source": {
            "phase5_winner_json": str(cfg.phase5_winner_json),
            "phase3_last_ckpt": str(phase3_last_ckpt),
            "winner_params": dict(winner_params),
        },
        "training": {
            "epochs": cfg.epochs,
            "avg_last_k": cfg.avg_last_k,
            "hard_thr_monitor": cfg.hard_thr_monitor,
            "seeds": list(cfg.seeds),
            "save": "LAST only",
        },
        "variants": list(cfg.variants),
        "seed_runs": seed_rows,
        "aggregate": aggregate_rows,
    }


def main() -> None:
    args = _parse_args()
    base_cfg = AblationStage1Config()
    cfg = AblationStage1Config(device=_resolve_device(args.device, base_cfg.device))
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    winner_obj = _read_json(cfg.phase5_winner_json)
    winner_params = dict(winner_obj.get("winner", {}).get("params", {}))
    if not winner_params:
        raise RuntimeError("Phase-5 winner JSON does not contain winner.params.")

    phase3_last_ckpt = _resolve_phase3_ckpt(cfg, winner_obj)
    _validate_required_paths(cfg, phase3_last_ckpt)

    seed_rows: List[Dict[str, Any]] = []
    for variant_name in cfg.variants:
        for seed in cfg.seeds:
            completed_row = _read_completed_seed_row(cfg, variant_name=variant_name, seed=seed)
            if completed_row is not None:
                print(f"[resume] Skipping completed run: variant={variant_name} seed={seed}")
                row = completed_row
            else:
                print(f"[resume] Rerunning incomplete run: variant={variant_name} seed={seed}")
                row = run_one_variant_seed(
                    cfg,
                    variant_name=variant_name,
                    winner_params=winner_params,
                    phase3_last_ckpt=phase3_last_ckpt,
                    seed=seed,
                )
            seed_rows.append(row)
            _refresh_stage_outputs(
                cfg,
                winner_params=winner_params,
                phase3_last_ckpt=phase3_last_ckpt,
                seed_rows=seed_rows,
            )
            clear_cuda_cache()


if __name__ == "__main__":
    main()
