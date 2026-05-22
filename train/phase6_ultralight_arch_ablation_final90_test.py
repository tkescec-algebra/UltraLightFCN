from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterator, List, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
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
from utils.metrics import calculate_dice, calculate_iou, calculate_precision_recall
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
SHARD_VARIANTS: dict[str, tuple[str, ...]] = {
    "all": DEFAULT_ABLATION_VARIANTS,
    "part1": (
        "no_mini_aspp",
        "no_shifted_sa",
        "no_mini_aspp_no_sa",
        "no_shallow_skip",
    ),
    "part2": (
        "no_dilation",
        "decoder_narrow",
        "decoder_wide",
    ),
}
PHASE6_BASELINE_REFERENCE_DIR = REPO_ROOT / "train" / "seg_phase6" / "final_retrain90" / "trial_54"


@dataclass(frozen=True)
class AblationStage2Config:
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
    amp: bool = True
    grad_clip_norm: float = 5.0
    finite_loss_guard: bool = True
    lr_min: float = 1e-6

    out_root: Path = REPO_ROOT / "train" / "seg_experimental_ablation" / "stage2_final90_test"
    save_last_name: str = "last.pth"
    save_epoch_log_name: str = "epoch_log.csv"
    save_audit_name: str = "simclr_load_audit.json"
    test_metrics_name: str = "test_metrics.json"
    seed_csv_name: str = "arch_ablation_stage2_test_per_seed.csv"
    report_json_name: str = "arch_ablation_stage2_test_report.json"
    report_lock_name: str = "arch_ablation_stage2_report.lock"


STAGE2_REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "variant_name",
    "seed",
    "status",
    "loss_name",
    "dice_soft",
    "dice_hard05",
    "iou_soft",
    "iou_hard05",
    "precision_hard05",
    "recall_hard05",
    "n",
    "ckpt_last_path",
    "simclr_load_audit_path",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--shard", type=str, default="all", choices=tuple(SHARD_VARIANTS.keys()))
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


def _resolve_phase3_ckpt(cfg: AblationStage2Config, winner_obj: Dict[str, Any]) -> Path:
    if cfg.phase3_last_ckpt is not None:
        return cfg.phase3_last_ckpt

    phase3 = winner_obj.get("init", {}).get("phase3_last_ckpt")
    if not phase3:
        raise RuntimeError("Phase-5 winner JSON does not contain init.phase3_last_ckpt.")
    return _resolve_existing_path(str(phase3), relative_anchor=REPO_ROOT / "train")


def _validate_required_paths(cfg: AblationStage2Config, phase3_last_ckpt: Path) -> None:
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


def _build_train_loader_90(cfg: AblationStage2Config, batch_size: int, *, seed: int) -> DataLoader:
    train_ds = SolarPanelDataset(
        str(cfg.data_root / cfg.train_split),
        mode="train",
        files=None,
        return_extra=False,
    )
    val_as_train_ds = SolarPanelDataset(
        str(cfg.data_root / cfg.val_split),
        mode="train",
        files=None,
        return_extra=False,
    )
    train90_ds = ConcatDataset([train_ds, val_as_train_ds])

    pw = cfg.persistent_workers and (cfg.num_workers > 0)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        train90_ds,
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


def _build_test_loader(cfg: AblationStage2Config, batch_size: int) -> DataLoader:
    test_ds = SolarPanelDataset(
        str(cfg.data_root / cfg.test_split),
        mode="test",
        files=None,
        return_extra=False,
    )

    pw = cfg.persistent_workers and (cfg.num_workers > 0)
    return DataLoader(
        test_ds,
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


def _validate_loader_audit(audit: Dict[str, Any]) -> None:
    if int(audit.get("skipped_shape_key_count", 0)) > 0:
        raise RuntimeError(f"Loader audit reported shape-mismatched keys: {audit}")

    loaded_prefixes = set(audit.get("loaded_prefixes", []))
    missing_backbone = [prefix for prefix in BACKBONE_PREFIXES if prefix not in loaded_prefixes]
    if missing_backbone:
        raise RuntimeError(
            f"Loader audit missing required backbone prefixes {missing_backbone}: {audit}"
        )


@torch.no_grad()
def _eval_test(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: AblationStage2Config,
    *,
    criterion,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    iou_soft_sum = 0.0
    iou_hard_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    n = 0

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
        dice_soft_sum += float(calculate_dice(logits, masks, thr=None)) * bs
        dice_hard_sum += float(calculate_dice(logits, masks, thr=cfg.hard_thr)) * bs
        iou_soft_sum += float(calculate_iou(logits, masks, thr=None)) * bs
        iou_hard_sum += float(calculate_iou(logits, masks, thr=cfg.hard_thr)) * bs
        precision, recall = calculate_precision_recall(logits, masks, thr=cfg.hard_thr)
        prec_sum += float(precision) * bs
        rec_sum += float(recall) * bs
        n += bs

    if n == 0:
        raise RuntimeError("TEST FAIL-FAST: eval_n==0 (no samples processed).")

    return {
        "loss": loss_sum / n,
        "dice_soft": dice_soft_sum / n,
        "dice_hard05": dice_hard_sum / n,
        "iou_soft": iou_soft_sum / n,
        "iou_hard05": iou_hard_sum / n,
        "precision_hard05": prec_sum / n,
        "recall_hard05": rec_sum / n,
        "n": float(n),
    }


def _mean_std(values: List[float]) -> Dict[str, float]:
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "n": 1}
    return {"mean": float(mean(values)), "std": float(pstdev(values)), "n": len(values)}


def _stage2_run_dir(cfg: AblationStage2Config, variant_name: str, seed: int) -> Path:
    return cfg.out_root / variant_name / f"seed_{seed}"


def _count_csv_data_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)
    except OSError:
        return None


def _read_completed_seed_row(
    cfg: AblationStage2Config,
    *,
    variant_name: str,
    seed: int,
) -> Dict[str, Any] | None:
    run_dir = _stage2_run_dir(cfg, variant_name, seed)
    ckpt_path = run_dir / cfg.save_last_name
    epoch_log_path = run_dir / cfg.save_epoch_log_name
    audit_path = run_dir / cfg.save_audit_name
    metrics_path = run_dir / cfg.test_metrics_name

    required_paths = (ckpt_path, epoch_log_path, audit_path, metrics_path)
    if not all(path.is_file() for path in required_paths):
        return None

    row_count = _count_csv_data_rows(epoch_log_path)
    if row_count != cfg.epochs:
        return None

    metrics = _try_read_json(metrics_path)
    if not isinstance(metrics, dict):
        return None

    missing_fields = [field for field in STAGE2_REQUIRED_ROW_FIELDS if field not in metrics]
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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_name = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _atomic_save_json(path: Path, obj: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _atomic_save_csv_rows(
    path: Path,
    rows: List[Dict[str, Any]],
    *,
    fieldnames: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_name = f.name
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


@contextmanager
def _stage_report_lock(cfg: AblationStage2Config) -> Iterator[None]:
    lock_path = cfg.out_root / cfg.report_lock_name
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    deadline = time.monotonic() + 3600.0
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(
                    fd,
                    f"pid={os.getpid()} acquired={datetime.now().isoformat()}\n".encode("utf-8"),
                )
            except OSError:
                os.close(fd)
                fd = None
                lock_path.unlink(missing_ok=True)
                raise
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for stage report lock: {lock_path}")
            time.sleep(0.25)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _collect_completed_seed_rows_from_disk(cfg: AblationStage2Config) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant_name in DEFAULT_ABLATION_VARIANTS:
        for seed in cfg.seeds:
            completed_row = _read_completed_seed_row(cfg, variant_name=variant_name, seed=seed)
            if completed_row is not None:
                rows.append(completed_row)
    return rows


def _refresh_stage_outputs(
    cfg: AblationStage2Config,
    *,
    winner_params: Dict[str, Any],
    phase3_last_ckpt: Path,
    shard: str,
    active_variants: Tuple[str, ...],
) -> None:
    with _stage_report_lock(cfg):
        seed_rows = _collect_completed_seed_rows_from_disk(cfg)
        _atomic_save_csv_rows(
            cfg.out_root / cfg.seed_csv_name,
            seed_rows,
            fieldnames=list(STAGE2_REQUIRED_ROW_FIELDS),
        )
        _atomic_save_json(
            cfg.out_root / cfg.report_json_name,
            _build_stage_report(
                cfg,
                winner_params=winner_params,
                phase3_last_ckpt=phase3_last_ckpt,
                seed_rows=seed_rows,
                shard=shard,
                active_variants=active_variants,
            ),
        )


def run_one_variant_seed(
    cfg: AblationStage2Config,
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

    loss_name, criterion = build_loss_from_params(winner_params, get_loss_function)
    train_loader = _build_train_loader_90(cfg, batch_size, seed=seed)
    test_loader = _build_test_loader(cfg, batch_size=batch_size)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.epochs),
        eta_min=float(cfg.lr_min),
    )

    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        skipped_nonfinite = 0

        for images, masks in tqdm(
            train_loader,
            desc=f"[ArchAblation-Stage2] {variant_name} seed={seed} ep={epoch}",
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
        scheduler.step()

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_n": train_n,
                "skipped_nonfinite": skipped_nonfinite,
                "lr_enc": float(optimizer.param_groups[0]["lr"]),
                "lr_dec": float(optimizer.param_groups[1]["lr"]),
            }
        )

    run_dir = _stage2_run_dir(cfg, variant_name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_csv_rows(str(run_dir / cfg.save_epoch_log_name), epoch_rows)
    save_json(str(run_dir / cfg.save_audit_name), audit)

    ckpt_path = run_dir / cfg.save_last_name
    variant_spec = get_ultralight_variant_spec(variant_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant_name": variant_name,
            "seed": seed,
            "params": dict(variant_spec.params),
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
            "stage": "stage2_final90_test",
            "train_data": "train+valid (90%)",
            "test_used_during_training": False,
            "test_evaluated_once_after_training": True,
        },
        ckpt_path,
    )

    test_metrics = _eval_test(model, test_loader, cfg, criterion=criterion, use_amp=use_amp)

    seed_row = {
        "variant_name": variant_name,
        "seed": seed,
        "status": "ok",
        "loss_name": str(loss_name),
        "dice_soft": float(test_metrics["dice_soft"]),
        "dice_hard05": float(test_metrics["dice_hard05"]),
        "iou_soft": float(test_metrics["iou_soft"]),
        "iou_hard05": float(test_metrics["iou_hard05"]),
        "precision_hard05": float(test_metrics["precision_hard05"]),
        "recall_hard05": float(test_metrics["recall_hard05"]),
        "n": float(test_metrics["n"]),
        "ckpt_last_path": str(ckpt_path),
        "simclr_load_audit_path": str(run_dir / cfg.save_audit_name),
    }
    save_json(str(run_dir / cfg.test_metrics_name), seed_row)
    return seed_row


def _build_stage_report(
    cfg: AblationStage2Config,
    *,
    winner_params: Dict[str, Any],
    phase3_last_ckpt: Path,
    seed_rows: List[Dict[str, Any]],
    shard: str,
    active_variants: Tuple[str, ...],
) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for variant_name in DEFAULT_ABLATION_VARIANTS:
        rows = [row for row in seed_rows if row["variant_name"] == variant_name]
        if not rows:
            continue
        aggregate[variant_name] = {
            "dice_soft": _mean_std([float(r["dice_soft"]) for r in rows]),
            "dice_hard05": _mean_std([float(r["dice_hard05"]) for r in rows]),
            "iou_soft": _mean_std([float(r["iou_soft"]) for r in rows]),
            "iou_hard05": _mean_std([float(r["iou_hard05"]) for r in rows]),
            "precision_hard05": _mean_std([float(r["precision_hard05"]) for r in rows]),
            "recall_hard05": _mean_std([float(r["recall_hard05"]) for r in rows]),
            "n": _mean_std([float(r["n"]) for r in rows]),
        }

    return {
        "stage": "stage2_final90_test",
        "timestamp": datetime.now().isoformat(),
        "locked_box_test": True,
        "test_used_during_training": False,
        "test_used_for_selection": False,
        "test_used_for_tuning": False,
        "baseline_reference": {
            "reused": True,
            "source_dir": str(PHASE6_BASELINE_REFERENCE_DIR),
            "overlapping_seeds": [13, 37, 71],
            "full_baseline_seed_count": 20,
            "note": (
                "The full UltraLightFCN SimCLR baseline is reused from the existing Phase-6 baseline runs. "
                "Comparisons should use overlapping seeds 13, 37, 71 unless ablation variants are later trained "
                "with the full 20-seed set."
            ),
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
            "hard_thr": cfg.hard_thr,
            "seeds": list(cfg.seeds),
            "save": "LAST only",
            "scheduler": {"name": "CosineAnnealingLR", "eta_min": cfg.lr_min, "T_max": cfg.epochs},
        },
        "variants": list(DEFAULT_ABLATION_VARIANTS),
        "sharding": {
            "shard": shard,
            "default_full_variant_list": list(DEFAULT_ABLATION_VARIANTS),
            "active_variant_list": list(active_variants),
            "note": (
                "Shared stage-level reports are rebuilt from completed per-seed artifacts on disk. "
                "Variants with no completed rows yet are omitted from aggregate."
            ),
        },
        "per_seed": seed_rows,
        "aggregate": aggregate,
    }


def main() -> None:
    args = _parse_args()
    base_cfg = AblationStage2Config()
    active_variants = SHARD_VARIANTS[args.shard]
    cfg = AblationStage2Config(
        device=_resolve_device(args.device, base_cfg.device),
        variants=active_variants,
    )
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    winner_obj = _read_json(cfg.phase5_winner_json)
    winner_params = dict(winner_obj.get("winner", {}).get("params", {}))
    if not winner_params:
        raise RuntimeError("Phase-5 winner JSON does not contain winner.params.")

    phase3_last_ckpt = _resolve_phase3_ckpt(cfg, winner_obj)
    _validate_required_paths(cfg, phase3_last_ckpt)

    for variant_name in cfg.variants:
        for seed in cfg.seeds:
            completed_row = _read_completed_seed_row(cfg, variant_name=variant_name, seed=seed)
            if completed_row is not None:
                print(f"[resume] Skipping completed run: variant={variant_name} seed={seed}")
            else:
                print(f"[resume] Rerunning incomplete run: variant={variant_name} seed={seed}")
                run_one_variant_seed(
                    cfg,
                    variant_name=variant_name,
                    winner_params=winner_params,
                    phase3_last_ckpt=phase3_last_ckpt,
                    seed=seed,
                )
            _refresh_stage_outputs(
                cfg,
                winner_params=winner_params,
                phase3_last_ckpt=phase3_last_ckpt,
                shard=args.shard,
                active_variants=active_variants,
            )
            clear_cuda_cache()


if __name__ == "__main__":
    main()
