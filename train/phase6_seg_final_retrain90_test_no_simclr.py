"""phase6_seg_final_retrain90_test_no_simclr.py

Phase 6 (ablation) — Final retrain of the no-SimCLR Phase-5 winner on TRAIN+VALID,
followed by one locked-box TEST evaluation.

Controlled relative to the main Phase-6 script:
  - Same seeds, same 60-epoch budget, same loaders, same metrics, same TEST aggregation logic.

Scientific difference:
  - UltraLightFCN starts from random initialization.
  - No SimCLR checkpoint is loaded anywhere in this script.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from models.UltraLightFCN_base import UltraLightFCN
from utils.config import SEG_PARAMS
from utils.dataset import SolarPanelDataset
from utils.helpers import (
    build_loss_from_params,
    clear_cuda_cache,
    get_loss_function,
    save_csv_rows,
    save_json,
    split_encoder_decoder_params,
)
from utils.metrics import calculate_dice, calculate_iou, calculate_precision_recall
from utils.no_simclr_guard import (
    NO_SIMCLR_ABLATION,
    NO_SIMCLR_ENCODER_INIT,
    NO_SIMCLR_PRETRAINING,
    NO_SIMCLR_TEST_POLICY,
    assert_no_pretrained_checkpoint,
    build_no_simclr_metadata,
)
from utils.repro import seed_worker, set_global_seed


@dataclass(frozen=True)
class Phase6NoSimCLRConfig:
    data_root: str = "../dataset"
    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"

    seeds: Tuple[int, ...] = (13, 37, 71)

    epochs: int = 60
    hard_thr_monitor: float = 0.5

    in_channels: int = 3
    num_classes: int = 1
    seg_params: dict = field(default_factory=lambda: dict(SEG_PARAMS))

    num_workers: int = 8
    pin_memory: bool = True
    drop_last_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deterministic: bool = False
    strict: bool = False

    phase5_winner_json: str = "seg_phase5/topk_retrain_no_simclr/phase5_winner_no_simclr.json"

    out_root: str = "seg_phase6/final_retrain90_no_simclr"
    save_last_name: str = "last.pth"
    save_epoch_log_name: str = "epoch_log.csv"
    per_seed_test_csv_name: str = "phase6_test_per_seed_no_simclr.csv"
    report_json_name: str = "phase6_test_report_no_simclr.json"

    pretraining: str = NO_SIMCLR_PRETRAINING
    encoder_init: str = NO_SIMCLR_ENCODER_INIT
    ablation: str = NO_SIMCLR_ABLATION
    test_policy: str = NO_SIMCLR_TEST_POLICY


def _resolve_dataloader_runtime(cfg: Phase6NoSimCLRConfig) -> tuple[int, bool, int | None]:
    # Windows multiprocessing compatibility safeguard:
    # spawned workers re-import torch and can hit DLL init failures, so force single-process loading.
    if platform.system().lower().startswith("win"):
        return 0, False, None
    return cfg.num_workers, cfg.persistent_workers, cfg.prefetch_factor


def _read_json(path: str) -> Dict[str, Any]:
    import json

    with open(path, "r") as f:
        return json.load(f)


def _build_train_loader_90(cfg: Phase6NoSimCLRConfig, batch_size: int, *, seed: int) -> DataLoader:
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


def _build_test_loader(cfg: Phase6NoSimCLRConfig, batch_size: int) -> DataLoader:
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
    cfg: Phase6NoSimCLRConfig,
    *,
    use_amp: bool,
    criterion,
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

            dice_soft = float(calculate_dice(logits, masks, thr=None))
            dice_hard = float(calculate_dice(logits, masks, thr=cfg.hard_thr_monitor))
            dice_soft_sum += dice_soft * bs
            dice_hard_sum += dice_hard * bs

            iou_soft = float(calculate_iou(logits, masks, thr=None))
            iou_hard = float(calculate_iou(logits, masks, thr=cfg.hard_thr_monitor))
            iou_soft_sum += iou_soft * bs
            iou_hard_sum += iou_hard * bs

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
    cfg: Phase6NoSimCLRConfig,
    *,
    candidate_id: int,
    params: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    set_global_seed(seed, deterministic=cfg.deterministic, strict=cfg.strict)
    assert_no_pretrained_checkpoint(
        branch_name="phase6_no_simclr",
        phase3_last_ckpt=None,
        init_metadata=build_no_simclr_metadata(phase=6),
    )

    params = dict(params)
    params.setdefault("enc_lr_mult", 1.0)

    batch_size = int(params["batch_size"])
    base_lr = float(params["base_lr"])
    enc_lr_mult = float(params.get("enc_lr_mult", 1.0))
    weight_decay = float(params["weight_decay"])
    rlop_factor = float(params["rlop_factor"])
    rlop_patience = int(params["rlop_patience"])

    _ = (rlop_factor, rlop_patience)  # preserved in candidate params for parity/audit
    loss_name, criterion = build_loss_from_params(params, get_loss_function)

    train_loader = _build_train_loader_90(cfg, batch_size, seed=seed)
    test_loader = _build_test_loader(cfg, batch_size=batch_size)

    # No-SimCLR ablation: random initialization only. Do not load any pretrained checkpoint.
    model = UltraLightFCN(in_channels=cfg.in_channels, num_classes=cfg.num_classes, params=cfg.seg_params)
    model = model.to(cfg.device)

    use_amp = (cfg.device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    enc_params, dec_params = split_encoder_decoder_params(model)
    optimizer = torch.optim.AdamW(
        [
            # No-SimCLR branch keeps encoder/decoder grouping for compatibility, but uses equal LR.
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
            desc=f"[Phase6 no-SimCLR trial {candidate_id} seed {seed}] Train {epoch + 1}/{cfg.epochs}",
            leave=False,
        ):
            images = images.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
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
        epoch_rows.append({"epoch": epoch + 1, "train_loss": train_loss, "lr_enc": lr_enc, "lr_dec": lr_dec})

    ckpt_path = os.path.join(run_dir, cfg.save_last_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "phase": 6,
            "candidate_id": candidate_id,
            "seed": seed,
            "params": params,
            "phase3_last_ckpt": None,
            "epochs": cfg.epochs,
            "timestamp": datetime.now().isoformat(),
            "train_data": "train+valid (90%)",
            "no_checkpoint_selection": True,
            "pretraining": cfg.pretraining,
            "encoder_init": cfg.encoder_init,
            "ablation": cfg.ablation,
            "test_policy": cfg.test_policy,
        },
        ckpt_path,
    )

    save_csv_rows(os.path.join(run_dir, cfg.save_epoch_log_name), epoch_rows)
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
    cfg = Phase6NoSimCLRConfig()
    os.makedirs(cfg.out_root, exist_ok=True)

    w = _read_json(cfg.phase5_winner_json)
    winner = w["winner"]
    candidate_id = int(winner["candidate_id"])
    params = dict(winner["params"])
    params.setdefault("enc_lr_mult", 1.0)

    assert_no_pretrained_checkpoint(
        branch_name="phase6_no_simclr",
        phase3_last_ckpt=w.get("init", {}).get("phase3_last_ckpt"),
        init_metadata=w.get("init"),
    )

    rows: List[Dict[str, Any]] = []
    for seed in cfg.seeds:
        rows.append(run_one_seed(cfg, candidate_id=candidate_id, params=params, seed=int(seed)))
        clear_cuda_cache()

    per_seed_csv = os.path.join(cfg.out_root, cfg.per_seed_test_csv_name)
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

    report = {
        "phase": 6,
        "timestamp": datetime.now().isoformat(),
        "data": {"train": "train+valid (90%)", "test": cfg.test_split, "test_used_during_training": False},
        "training": {"epochs": cfg.epochs, "seeds": list(cfg.seeds), "save": "LAST only"},
        "winner_source": {"phase5_winner_json": cfg.phase5_winner_json, "candidate_id": candidate_id},
        "init": {"phase3_last_ckpt": None, **build_no_simclr_metadata(phase=6)},
        "metrics_test": {
            "soft_dice": _mean_std([float(r["test_soft_dice"]) for r in rows]),
            "hard_dice@0.5": _mean_std([float(r["test_hard_dice@0.5"]) for r in rows]),
            "soft_iou": _mean_std([float(r["test_soft_iou"]) for r in rows]),
            "hard_iou@0.5": _mean_std([float(r["test_hard_iou@0.5"]) for r in rows]),
            "precision@0.5": _mean_std([float(r["test_precision@0.5"]) for r in rows]),
            "recall@0.5": _mean_std([float(r["test_recall@0.5"]) for r in rows]),
            "loss": _mean_std([float(r["test_loss"]) for r in rows]),
        },
        "runs": rows,
        "artifacts": {"per_seed_csv": per_seed_csv},
        "pretraining": cfg.pretraining,
        "encoder_init": cfg.encoder_init,
        "ablation": cfg.ablation,
        "test_policy": cfg.test_policy,
    }

    report_path = os.path.join(cfg.out_root, cfg.report_json_name)
    save_json(report_path, report)

    print(f"[no-simclr] pretraining={cfg.pretraining} encoder_init={cfg.encoder_init} ablation={cfg.ablation}")
    print(f"Phase-6 no-SimCLR per-seed TEST CSV: {per_seed_csv}")
    print(f"Phase-6 no-SimCLR TEST report JSON: {report_path}")
    print(f"TEST soft Dice mean±std: {report['metrics_test']['soft_dice']}")


if __name__ == "__main__":
    main()
