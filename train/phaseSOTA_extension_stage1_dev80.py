from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
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
from utils.metrics import calculate_dice
from utils.repro import seed_worker, set_global_seed
from utils.sota_registry_extension import (
    SOTA_EXTENSION_MODELS,
    SOTA_REGIMES,
    split_smp_encoder_decoder_params,
)


# Extension Stage1 is restricted to minft-only because manuscript SOTA
# comparator reporting uses minft-only results; this avoids mixing
# fine-tuning regimes across comparator families.
SOTA_EXTENSION_STAGE1_REGIMES: Tuple[str, ...] = ("minft",)


@dataclass(frozen=True)
class SOTAExtensionStage1Config:
    data_root: str = "../dataset"
    train_split: str = "train"
    val_split: str = "valid"

    # Training budget
    epochs: int = 30
    avg_last_k: int = 10
    hard_thr_monitor: float = 0.5

    # Dataloaders
    num_workers: int = 8
    pin_memory: bool = True
    drop_last_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Runtime / Repro
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deterministic: bool = False
    strict: bool = False
    seeds: Tuple[int, ...] = (13, 37, 71)

    # Stability (paper-safe, apply uniformly)
    amp: bool = True
    grad_clip_norm: float = 5.0
    finite_loss_guard: bool = True

    # Unified recipe source (Phase-5 winner)
    phase5_winner_json: str = "seg_phase5/topk_retrain/phase5_winner.json"

    # Output
    out_root: str = "seg_sota_extension/stage1_dev80"
    summary_csv: str = "sota_extension_stage1_results.csv"
    aggregate_csv: str = "sota_extension_stage1_aggregate.csv"
    winners_json: str = "stage1_extension_winners.json"


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"JSON not found: {path}")
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_loaders(cfg: SOTAExtensionStage1Config, batch_size: int, *, seed: int) -> Tuple[DataLoader, DataLoader]:
    train_dir = os.path.join(cfg.data_root, cfg.train_split)
    val_dir = os.path.join(cfg.data_root, cfg.val_split)

    train_ds = SolarPanelDataset(train_dir, mode="train", files=None, return_extra=False)
    val_ds = SolarPanelDataset(val_dir, mode="valid", files=None, return_extra=False)

    pw = cfg.persistent_workers and (cfg.num_workers > 0)
    generator = torch.Generator().manual_seed(seed)

    train_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last_train,
        persistent_workers=pw,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=generator,
    )
    val_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=pw,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
    )

    # Only set prefetch_factor when workers > 0 (avoids None-related issues on some PyTorch versions)
    if cfg.num_workers > 0:
        train_kwargs["prefetch_factor"] = cfg.prefetch_factor
        val_kwargs["prefetch_factor"] = cfg.prefetch_factor

    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader = DataLoader(val_ds, **val_kwargs)
    return train_loader, val_loader


@torch.no_grad()
def _eval_val(model: torch.nn.Module, loader: DataLoader, cfg: SOTAExtensionStage1Config) -> Dict[str, float]:
    model.eval()
    soft_sum = 0.0
    hard_sum = 0.0
    n = 0

    use_amp = (cfg.amp and cfg.device.type == "cuda")
    for images, masks in loader:
        images = images.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)

        bs = int(images.shape[0])
        soft_sum += float(calculate_dice(logits, masks, thr=None)) * bs
        hard_sum += float(calculate_dice(logits, masks, thr=cfg.hard_thr_monitor)) * bs
        n += bs

    if n == 0:
        raise RuntimeError("VAL FAIL-FAST: eval_n==0 (no samples processed).")

    return {
        "val_dice_soft": soft_sum / n,
        "val_dice_hard05": hard_sum / n,
        "val_n": float(n),
    }


def _build_smp_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    ModelClass = get_model(model_cfg["model_key"])  # smp.DeepLabV3Plus or smp.Unet
    # SMP expects classes=1 for binary; activation=None -> logits
    model = ModelClass(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model


def run_one(
    cfg: SOTAExtensionStage1Config,
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

    # FullFT uses Phase-5 winner enc_lr_mult; MinFT uses fixed 0.1
    if regime_name == "fullft":
        enc_lr_mult = float(params["enc_lr_mult"])
    else:
        enc_lr_mult = float(SOTA_REGIMES["minft"]["enc_lr_mult"])

    loss_name, criterion = build_loss_from_params(params, get_loss_function)

    train_loader, val_loader = _build_loaders(cfg, batch_size, seed=seed)

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

    # VALID exists in Stage1 -> keep Plateau (as in Phase-5 spirit)
    rlop_factor = float(params.get("rlop_factor", 0.5))
    rlop_patience = int(params.get("rlop_patience", 3))
    rlop_min_lr = float(params.get("rlop_min_lr", 1e-6))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=rlop_factor,
        patience=rlop_patience,
        min_lr=rlop_min_lr,
    )

    use_amp = (cfg.amp and cfg.device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    # output dir
    run_dir = os.path.join(cfg.out_root, model_name, regime_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    epoch_rows: List[Dict[str, Any]] = []
    lastk = deque(maxlen=cfg.avg_last_k)
    best_avg_lastk = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        skipped_nonfinite = 0

        for images, masks in tqdm(
            train_loader,
            desc=f"[SOTA-Extension-Stage1] {model_name}/{regime_name} seed={seed} ep={epoch}",
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
            raise RuntimeError(
                f"TRAIN FAIL-FAST: train_n==0 at epoch {epoch} (skipped_nonfinite={skipped_nonfinite})"
            )

        train_loss = train_loss_sum / train_n

        val_metrics = _eval_val(model, val_loader, cfg)
        val_soft = float(val_metrics["val_dice_soft"])
        val_hard = float(val_metrics["val_dice_hard05"])

        lastk.append(val_soft)
        avg_lastk = float(sum(lastk) / len(lastk))
        best_avg_lastk = max(best_avg_lastk, avg_lastk)

        scheduler.step(val_soft)

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_n": train_n,
                "skipped_nonfinite": skipped_nonfinite,
                "val_soft": val_soft,
                "val_hard05": val_hard,
                "avg_last_k_soft": avg_lastk,
                "lr_enc": optimizer.param_groups[0]["lr"],
                "lr_dec": optimizer.param_groups[1]["lr"],
            }
        )

    # save LAST ckpt
    last_path = os.path.join(run_dir, "last.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat(),
            "stage": 1,
            "split": "train80/valid10",
            "seed": seed,
            "model_name": model_name,
            "regime": regime_name,
            "model_cfg": model_cfg,
            "loss_name": loss_name,
            "params_unified": params,
            "enc_lr_mult_used": enc_lr_mult,
        },
        last_path,
    )

    # epoch log
    save_csv_rows(os.path.join(run_dir, "epoch_log.csv"), epoch_rows)

    return {
        "model_name": model_name,
        "regime": regime_name,
        "seed": seed,
        "status": "ok",
        "last_ckpt": last_path,
        "best_avg_last_k_soft": best_avg_lastk,
        "best_val_soft": max(r["val_soft"] for r in epoch_rows),
        "best_val_hard05": max(r["val_hard05"] for r in epoch_rows),
    }


def main() -> None:
    cfg = SOTAExtensionStage1Config()
    os.makedirs(cfg.out_root, exist_ok=True)

    phase5_winner = _read_json(cfg.phase5_winner_json)

    rows: List[Dict[str, Any]] = []
    for model_name in SOTA_EXTENSION_MODELS.keys():
        for regime_name in SOTA_EXTENSION_STAGE1_REGIMES:
            for seed in cfg.seeds:
                try:
                    row = run_one(cfg, phase5_winner, model_name, regime_name, seed)
                except Exception as e:
                    row = {
                        "model_name": model_name,
                        "regime": regime_name,
                        "seed": seed,
                        "status": "failed",
                        "error": str(e),
                    }
                rows.append(row)
                clear_cuda_cache()

    # Save summary
    save_csv_rows(os.path.join(cfg.out_root, cfg.summary_csv), rows)

    # Also save a compact aggregate by (model, regime)
    agg_rows = []
    for model_name in SOTA_EXTENSION_MODELS.keys():
        for regime_name in SOTA_EXTENSION_STAGE1_REGIMES:
            ok = [
                r
                for r in rows
                if r["model_name"] == model_name and r["regime"] == regime_name and r["status"] == "ok"
            ]
            if not ok:
                agg_rows.append({"model_name": model_name, "regime": regime_name, "n_ok": 0})
                continue

            vals = [float(r["best_avg_last_k_soft"]) for r in ok]
            agg_rows.append(
                {
                    "model_name": model_name,
                    "regime": regime_name,
                    "n_ok": len(vals),
                    "mean_best_avg_last_k_soft": mean(vals),
                    "std_best_avg_last_k_soft": pstdev(vals) if len(vals) > 1 else 0.0,
                }
            )

    save_csv_rows(os.path.join(cfg.out_root, cfg.aggregate_csv), agg_rows)
    print(f"[SOTA Extension Stage1] Wrote: {os.path.join(cfg.out_root, cfg.summary_csv)}")
    print(f"[SOTA Extension Stage1] Wrote: {os.path.join(cfg.out_root, cfg.aggregate_csv)}")

    # --- Build stage1_extension_winners.json (Phase-5 style selection: mean metric, tie-break std, then best_val_soft)
    per_group_stats: Dict[str, Any] = {}
    per_model_winner: Dict[str, Any] = {}

    # Build per (model, regime) stats
    for model_name in SOTA_EXTENSION_MODELS.keys():
        for regime_name in SOTA_EXTENSION_STAGE1_REGIMES:
            ok = [
                r
                for r in rows
                if r["model_name"] == model_name and r["regime"] == regime_name and r["status"] == "ok"
            ]

            key = f"{model_name}::{regime_name}"
            if not ok:
                per_group_stats[key] = {
                    "model_name": model_name,
                    "regime": regime_name,
                    "n_ok": 0,
                    "mean_best_avg_last_k_soft": float("nan"),
                    "std_best_avg_last_k_soft": float("nan"),
                    "mean_best_val_soft": float("nan"),
                    "mean_best_val_hard05": float("nan"),
                    "seeds_ok": [],
                }
                continue

            vals_avgk = [float(r["best_avg_last_k_soft"]) for r in ok]
            vals_best_soft = [float(r["best_val_soft"]) for r in ok]
            vals_best_hard = [float(r["best_val_hard05"]) for r in ok]
            seeds_ok = [int(r["seed"]) for r in ok]

            per_group_stats[key] = {
                "model_name": model_name,
                "regime": regime_name,
                "n_ok": len(ok),
                "seeds_ok": seeds_ok,
                "mean_best_avg_last_k_soft": float(mean(vals_avgk)),
                "std_best_avg_last_k_soft": float(pstdev(vals_avgk) if len(vals_avgk) > 1 else 0.0),
                "mean_best_val_soft": float(mean(vals_best_soft)),
                "mean_best_val_hard05": float(mean(vals_best_hard)),
            }

    # Pick winner regime per model using:
    # 1) max mean_best_avg_last_k_soft
    # 2) min std_best_avg_last_k_soft
    # 3) max mean_best_val_soft
    def _regime_score(d: Dict[str, Any]) -> Tuple[float, float, float]:
        mean_metric = d["mean_best_avg_last_k_soft"]
        std_metric = d["std_best_avg_last_k_soft"]
        mean_best_soft = d["mean_best_val_soft"]
        # sort reverse=True => mean higher is better, std lower is better, best_val_soft higher is better
        return (mean_metric, -std_metric, mean_best_soft)

    for model_name in SOTA_EXTENSION_MODELS.keys():
        candidates = []
        for regime_name in SOTA_EXTENSION_STAGE1_REGIMES:
            key = f"{model_name}::{regime_name}"
            d = per_group_stats[key]
            if d["n_ok"] > 0:
                candidates.append((key, d))

        if not candidates:
            per_model_winner[model_name] = {"status": "no_ok_runs"}
            continue

        # select by Phase-5 tie-break logic
        candidates.sort(key=lambda x: _regime_score(x[1]), reverse=True)
        winner_key, winner_stats = candidates[0]

        per_model_winner[model_name] = {
            "winner_key": winner_key,
            "winner_regime": winner_stats["regime"],
            "n_ok": winner_stats["n_ok"],
            "seeds_ok": winner_stats["seeds_ok"],
            "mean_best_avg_last_k_soft": winner_stats["mean_best_avg_last_k_soft"],
            "std_best_avg_last_k_soft": winner_stats["std_best_avg_last_k_soft"],
            "mean_best_val_soft": winner_stats["mean_best_val_soft"],
            "mean_best_val_hard05": winner_stats["mean_best_val_hard05"],
        }

    # Optional audit trail (does not affect training/selection)
    unified_params = dict(phase5_winner.get("winner", {}).get("params", {}))
    audit_unified = {
        "phase5_winner_json": cfg.phase5_winner_json,
        "batch_size": unified_params.get("batch_size", None),
        "base_lr": unified_params.get("base_lr", None),
        "enc_lr_mult_fullft": unified_params.get("enc_lr_mult", None),
        "weight_decay": unified_params.get("weight_decay", None),
        "loss_name": phase5_winner.get("winner", {}).get("loss_name", None),
    }

    stage1_winners = {
        "stage": 1,
        "timestamp": datetime.now().isoformat(),
        "split": "train80/valid10",
        "seeds_requested": list(cfg.seeds),
        "selection_metric": "val_soft_dice(thr=None) with avg_last_k; select by mean(best_avg_last_k_soft)",
        "tie_break": ["min std(best_avg_last_k_soft)", "max mean(best_val_soft)"],
        "models": list(SOTA_EXTENSION_MODELS.keys()),
        "regimes": list(SOTA_EXTENSION_STAGE1_REGIMES),
        "audit_unified_recipe": audit_unified,
        "per_group_stats": per_group_stats,
        "per_model_winner": per_model_winner,
    }

    save_json(os.path.join(cfg.out_root, cfg.winners_json), stage1_winners)
    print(f"[SOTA Extension Stage1] Wrote: {os.path.join(cfg.out_root, cfg.winners_json)}")


if __name__ == "__main__":
    main()
