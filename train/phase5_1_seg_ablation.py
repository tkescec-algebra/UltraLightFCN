"""phase5.1_seg_ablation.py

Phase A — Ablation study (development-only; NO TEST).

Variants (all on TRAIN 80% / VALID 10%, seeds 13/37/71, 30 epochs):
  - baseline_pretrained_finetune: NOT rerun (taken from Phase-5 winner summary).
  - ablation1_pretrained_freeze: load SimCLR Phase-3 LAST into encoder, freeze encoder stack.
  - ablation2_scratch_finetune: random init (no pretrain), train encoder+decoder.

Evaluation protocol matches Phase-5:
  - selection metric: avg_last_k of validation soft Dice (thr=None)
  - hard Dice@0.5 is logged only (monitoring)

Outputs:
  - per-run folder with last.pth and epoch_log.csv
  - global CSV with per-run summary
  - JSON report with mean±std (baseline from phase5_winner.json + ablations from runs)
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.UltraLightFCN_base import UltraLightFCN
from utils.config import SEG_PARAMS, ENCODER_PREFIXES
from utils.dataset import SolarPanelDataset
from utils.helpers import (
    build_loss_from_params,
    clear_cuda_cache,
    get_loss_function,
    save_csv_rows,
    save_json,
    split_encoder_decoder_params,
)
from utils.load_simclr_pretrain_encoder import load_pretrained_encoder_into_ultralight
from utils.metrics import calculate_dice
from utils.repro import seed_worker, set_global_seed


@dataclass(frozen=True)
class PhaseAConfig:
    # Data
    data_root: str = "../dataset"
    train_split: str = "train"
    val_split: str = "valid"

    # Seeds
    seeds: Tuple[int, ...] = (13, 37, 71)

    # Budget / metric
    epochs: int = 30
    avg_last_k: int = 10
    hard_thr_monitor: float = 0.5

    # Model (fixed)
    in_channels: int = 3
    num_classes: int = 1
    seg_params: dict = field(default_factory=lambda: dict(SEG_PARAMS))

    # Runtime / loaders
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 8
    pin_memory: bool = True
    drop_last_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Repro
    deterministic: bool = False
    strict: bool = False

    # Inputs (Phase-5 artifacts)
    phase5_winner_json: str = "seg_phase5/topk_retrain/phase5_winner.json"

    # Outputs
    out_root: str = "seg_phase5/ablation"
    save_last_name: str = "last.pth"
    save_epoch_log_name: str = "epoch_log.csv"
    summary_csv_name: str = "phase5.1_ablation_results.csv"
    report_json_name: str = "phase5.1_ablation_report.json"


def _read_json(path: str) -> Dict[str, Any]:
    import json
    with open(path, "r") as f:
        return json.load(f)


def _build_loaders(cfg: PhaseAConfig, batch_size: int, *, seed: int) -> Tuple[DataLoader, DataLoader]:
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
    cfg: PhaseAConfig,
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


def _freeze_encoder_stack(model: torch.nn.Module) -> None:
    """Freeze encoder params and keep encoder modules in eval mode (avoid BN/stat updates)."""
    for name, p in model.named_parameters():
        if name.startswith(ENCODER_PREFIXES):
            p.requires_grad = False
    for name, m in model.named_modules():
        if name and name.startswith(ENCODER_PREFIXES):
            m.eval()


def run_one(
    cfg: PhaseAConfig,
    *,
    variant_name: str,
    candidate_id: int,
    params: Dict[str, Any],
    phase3_last_ckpt: str,
    seed: int,
    init_mode: str,
    freeze_encoder: bool,
) -> Dict[str, Any]:
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
    if init_mode == "pretrained":
        load_pretrained_encoder_into_ultralight(model, phase3_last_ckpt, verbose=(seed == cfg.seeds[0]))
    elif init_mode == "scratch":
        pass
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    model = model.to(cfg.device)
    use_amp = (cfg.device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    enc_params_list, dec_params_list = split_encoder_decoder_params(model)
    if freeze_encoder:
        _freeze_encoder_stack(model)
        enc_params_list = [p for p in enc_params_list if p.requires_grad]

    if freeze_encoder:
        optimizer = torch.optim.AdamW(
            [{"params": dec_params_list, "lr": base_lr}],
            lr=base_lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": enc_params_list, "lr": base_lr * enc_lr_mult},
                {"params": dec_params_list, "lr": base_lr},
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

    run_dir = os.path.join(cfg.out_root, variant_name, f"trial_{candidate_id}", f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        if freeze_encoder:
            _freeze_encoder_stack(model)

        train_loss_sum, train_n = 0.0, 0
        for images, masks in tqdm(
            train_loader,
            desc=f"[{variant_name} trial {candidate_id} seed {seed}] Train {epoch + 1}/{cfg.epochs}",
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

        if freeze_encoder:
            lr_enc = 0.0
            lr_dec = optimizer.param_groups[0]["lr"]
        else:
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

    ckpt_path = os.path.join(run_dir, cfg.save_last_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "phase": "A",
            "variant": variant_name,
            "candidate_id": candidate_id,
            "seed": seed,
            "params": params,
            "init_mode": init_mode,
            "freeze_encoder": bool(freeze_encoder),
            "phase3_last_ckpt": phase3_last_ckpt if init_mode == "pretrained" else None,
            "epochs": cfg.epochs,
            "avg_last_k": cfg.avg_last_k,
            "timestamp": datetime.now().isoformat(),
        },
        ckpt_path,
    )

    save_csv_rows(os.path.join(run_dir, cfg.save_epoch_log_name), epoch_rows)

    return {
        "variant": variant_name,
        "candidate_id": candidate_id,
        "seed": seed,
        "loss_name": str(loss_name),
        "best_avg_last_k_soft": float(best_avg_last_k),
        "best_epoch": int(best_epoch),
        "best_val_soft": float(best_val_soft),
        "best_val_hard05": float(best_val_hard05),
        "final_val_soft": float(epoch_rows[-1]["val_soft"]),
        "final_val_hard05": float(epoch_rows[-1]["val_hard05"]),
        "ckpt_last_path": ckpt_path,
    }


def _mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "std": float("nan")}
    if len(xs) == 1:
        return {"mean": float(xs[0]), "std": 0.0}
    return {"mean": float(mean(xs)), "std": float(pstdev(xs))}


def main() -> None:
    cfg = PhaseAConfig()
    os.makedirs(cfg.out_root, exist_ok=True)

    w = _read_json(cfg.phase5_winner_json)
    winner = w["winner"]
    candidate_id = int(winner["candidate_id"])
    params = dict(winner["params"])
    phase3_last_ckpt = str(w["init"]["phase3_last_ckpt"])

    rows: List[Dict[str, Any]] = []

    # Ablation 1: Pretrained + Freeze encoder
    for seed in cfg.seeds:
        rows.append(
            run_one(
                cfg,
                variant_name="ablation1_pretrained_freeze",
                candidate_id=candidate_id,
                params=params,
                phase3_last_ckpt=phase3_last_ckpt,
                seed=int(seed),
                init_mode="pretrained",
                freeze_encoder=True,
            )
        )
        clear_cuda_cache()

    # Ablation 2: Scratch + Fine-tune
    for seed in cfg.seeds:
        rows.append(
            run_one(
                cfg,
                variant_name="ablation2_scratch_finetune",
                candidate_id=candidate_id,
                params=params,
                phase3_last_ckpt=phase3_last_ckpt,
                seed=int(seed),
                init_mode="scratch",
                freeze_encoder=False,
            )
        )
        clear_cuda_cache()

    # Write per-run summary CSV
    summary_path = os.path.join(cfg.out_root, cfg.summary_csv_name)
    keys = [
        "variant",
        "candidate_id",
        "seed",
        "loss_name",
        "best_avg_last_k_soft",
        "best_epoch",
        "best_val_soft",
        "best_val_hard05",
        "final_val_soft",
        "final_val_hard05",
        "ckpt_last_path",
    ]
    save_csv_rows(summary_path, rows, fieldnames=keys)

    # Build report JSON with mean±std on VALID (soft Dice avg_last_k)
    report = {
        "phase": "A",
        "timestamp": datetime.now().isoformat(),
        "data": {"train": cfg.train_split, "valid": cfg.val_split, "test_used": False},
        "fixed_protocol": {"epochs": cfg.epochs, "avg_last_k": cfg.avg_last_k, "seeds": list(cfg.seeds)},
        "winner_source": {
            "phase5_winner_json": cfg.phase5_winner_json,
            "candidate_id": candidate_id,
            "params": params,
            "phase3_last_ckpt": phase3_last_ckpt,
        },
        "baseline_pretrained_finetune": {
            "mean_best_avg_last_k_soft": float(winner.get("mean_best_avg_last_k_soft", float("nan"))),
            "std_best_avg_last_k_soft": float(winner.get("std_best_avg_last_k_soft", float("nan"))),
            "note": "Baseline is taken from Phase-5 (not rerun in Phase A).",
        },
        "ablations": {},
        "artifacts": {"summary_csv": summary_path},
    }

    for vname in ["ablation1_pretrained_freeze", "ablation2_scratch_finetune"]:
        scores = [float(r["best_avg_last_k_soft"]) for r in rows if r["variant"] == vname]
        report["ablations"][vname] = {
            "best_avg_last_k_soft": _mean_std(scores),
            "runs": [r for r in rows if r["variant"] == vname],
        }

    report_path = os.path.join(cfg.out_root, cfg.report_json_name)
    save_json(report_path, report)

    print(f"✅ Phase A ablation results saved to: {summary_path}")
    print(f"✅ Phase A ablation report saved to: {report_path}")


if __name__ == "__main__":
    main()
