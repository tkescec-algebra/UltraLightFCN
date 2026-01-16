from __future__ import annotations

import json
import time
import copy
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

from models.UltraLightFCN_SimCLR import SimCLRModel, UltraLightEncoder, ProjectionHead
from pretrain.utils.metrics_simclr import simclr_alignment, simclr_uniformity
from pretrain.utils.transforms import SimCLRAugConfig, build_simclr_transforms
from utils.dataset import SimCLRSolarPanelDataset
from utils.loss_functions import NTXentLoss
from utils.repro import set_global_seed, seed_worker

torch.multiprocessing.set_sharing_strategy("file_system")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class AugRunConfig:
    """Augmentation sensitivity runner config (Phase3-aligned)."""

    # Plan + Phase3 artifacts
    plan_json_path: str = "aug_run_plan.json"
    phase3_best_json: str = "../checkpoints/simclr_phase3/phase3_best_phase2_candidate.json"
    phase3_last_ckpt: str = "../checkpoints/simclr_phase3/phase3_seed13_last.pth"

    # Data
    train_dir: str = "../../dataset/train"

    # Output
    out_root: str = "runs/aug_sensitivity"

    # DataLoader (Phase3-style)
    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    drop_last: bool = True

    # Execution / reproducibility
    skip_if_done: bool = True
    deterministic: bool = False  # aligned with Phase3: seeded but not strict deterministic

    # Logging
    log_every_steps: int = 50

    # Rep metrics (diagnostics only; fixed compute)
    compute_rep_metrics: bool = True
    rep_metric_bs: int = 256
    rep_metric_num_batches: int = 4
    uniformity_t: float = 2.0


# Single source of truth
CFG = AugRunConfig()

# =============================================================================
# 1) IO helpers
# =============================================================================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_plan() -> List[Dict[str, Any]]:
    plan = load_json(CFG.plan_json_path)
    if not isinstance(plan, list) or len(plan) == 0:
        raise RuntimeError("PLAN_JSON_PATH must contain a non-empty JSON list.")
    return plan


def read_file_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# =============================================================================
# 2) Phase3 HP + encoder_params from artifacts
# =============================================================================
def load_phase3_hp_from_best_json() -> Dict[str, Any]:
    hp = load_json(CFG.phase3_best_json)
    # Expected schema from your best-candidate json :contentReference[oaicite:1]{index=1}
    return {
        "batch_size": int(hp["batch_size"]),
        "lr": float(hp["lr"]),
        "weight_decay": float(hp["weight_decay"]),
        "temperature": float(hp["temperature"]),
        "warmup_ratio": float(hp["warmup_ratio"]),
        "grad_clip": float(hp["grad_clip"]),
        "proj_hidden_dim": int(hp["proj_hidden_dim"]),
        "proj_out_dim": int(hp["proj_out_dim"]),
    }


def load_encoder_params_from_phase3_ckpt(device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(CFG.phase3_last_ckpt, map_location=device)
    meta = ckpt.get("meta", {})
    cfg = meta.get("config", {})
    enc_params = cfg.get("encoder_params", None)
    if not isinstance(enc_params, dict) or len(enc_params) == 0:
        raise RuntimeError("encoder_params not found at meta.config.encoder_params in Phase3 checkpoint")
    return enc_params


# =============================================================================
# 3) Convert aug dict -> SimCLRAugConfig (transforms.py is the source of truth)
# =============================================================================
def aug_dict_to_config(aug: Dict[str, Any], image_size: int) -> SimCLRAugConfig:
    # jitter stored as list [b,c,s,h] in your run plan
    jitter = aug.get("jitter", [0.4, 0.4, 0.4, 0.1])
    jb, jc, js, jh = map(float, jitter)

    crop_scale = aug.get("crop_scale", [0.4, 1.0])
    cs0, cs1 = map(float, crop_scale)

    blur_sigma = aug.get("blur_sigma", [0.1, 0.8])
    bs0, bs1 = map(float, blur_sigma)

    return SimCLRAugConfig(
        image_size=int(image_size),

        use_crop=bool(aug.get("use_crop", True)),
        crop_scale=(cs0, cs1),

        p_hflip=float(aug.get("p_hflip", 0.5)),
        p_vflip=float(aug.get("p_vflip", 0.5)),

        rot_deg=float(aug.get("rot_deg", 10.0)),

        p_jitter=float(aug.get("p_jitter", 0.8)),
        jitter_b=jb,
        jitter_c=jc,
        jitter_s=js,
        jitter_h=jh,

        p_gray=float(aug.get("p_gray", 0.1)),

        p_blur=float(aug.get("p_blur", 0.5)),
        blur_kernel=int(aug.get("blur_kernel", 3)),
        blur_sigma=(bs0, bs1),
    )


# =============================================================================
# 4) Rep metrics compute-capped (diagnostics)
# =============================================================================
@torch.no_grad()
def compute_alignment_uniformity_fixed(
    model: SimCLRModel,
    dataset: SimCLRSolarPanelDataset,
    device: torch.device,
    seed: int,
) -> Tuple[float, float]:
    """
    Fixed compute protocol: seeded shuffle + drop_last=True.
    This matches the Phase3 rep-metrics spirit: comparable diagnostics.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed + 999)

    loader = DataLoader(
        dataset,
        batch_size=CFG.rep_metric_bs,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        drop_last=True,
        persistent_workers=CFG.persistent_workers,
        prefetch_factor=CFG.prefetch_factor,
    )

    align_sum, uni_sum, n = 0.0, 0.0, 0
    for i, (x1, x2, _) in enumerate(loader):
        if i >= CFG.rep_metric_num_batches:
            break
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        z1 = model(x1)
        z2 = model(x2)

        align_sum += float(simclr_alignment(z1, z2))
        uni_sum += float(simclr_uniformity(torch.cat([z1, z2], dim=0), t=CFG.uniformity_t))
        n += 1

    if n == 0:
        return float("nan"), float("nan")
    return align_sum / n, uni_sum / n


# =============================================================================
# 5) Main
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plan = load_plan()
    hp = load_phase3_hp_from_best_json()
    encoder_params = load_encoder_params_from_phase3_ckpt(device=device)

    out_root = Path(CFG.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Hard sanity: all runs should use the same file_list for Phase3 alignment
    file_lists = {r.get("file_list") for r in plan}
    if len(file_lists) != 1:
        raise RuntimeError(f"Non-identical file_list values in plan: {file_lists}")

    print(f"[plan] runs={len(plan)} | out_root={out_root} | device={device.type}")
    print(f"[hp] bs={hp['batch_size']} lr={hp['lr']:.6g} wd={hp['weight_decay']:.3g} T={hp['temperature']:.4g}")

    for run in plan:
        run_id = str(run["run_id"])
        group = str(run.get("group", ""))
        base = str(run.get("base", ""))
        seed = int(run["seed"])
        max_steps = int(run["max_steps"])
        image_size = int(run.get("image_size", 256))
        aug = dict(run["aug"])
        notes = str(run.get("notes", ""))

        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        done_flag = run_dir / "DONE"
        if CFG.skip_if_done and done_flag.exists():
            print(f"[skip] {run_id}")
            continue

        # Phase3-aligned: seed reset per run
        set_global_seed(seed, deterministic=CFG.deterministic, strict=False)

        # Deterministic file list from plan
        file_list_path = str(run["file_list"])
        train_files = read_file_list(file_list_path)
        (run_dir / f"train_files_seed{seed}.txt").write_text("\n".join(train_files), encoding="utf-8")

        # Build transforms ONLY through transforms.py config factory (source of truth) :contentReference[oaicite:3]{index=3}
        cfg = aug_dict_to_config(aug, image_size=image_size)
        train_tf = build_simclr_transforms(cfg)

        # Dataset
        train_ds = SimCLRSolarPanelDataset(CFG.train_dir, image_size=image_size, transform=train_tf, files=train_files)

        # Batch size: plan wins, else Phase3 HP
        bs = int(run.get("batch_size", hp["batch_size"]))
        if bs != hp["batch_size"]:
            print(f"[warn] {run_id}: plan batch_size={bs} differs from Phase3 hp batch_size={hp['batch_size']}")

        drop_last = CFG.drop_last and (len(train_ds) >= bs)
        g_train = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            generator=g_train,
            worker_init_fn=seed_worker,
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
            drop_last=drop_last,
            persistent_workers=CFG.persistent_workers,
            prefetch_factor=CFG.prefetch_factor,
        )

        # Model (Phase3 aligned)
        encoder = UltraLightEncoder(in_channels=3, params=encoder_params).to(device)

        proj = ProjectionHead(in_dim=encoder.out_channels, hidden_dim=hp["proj_hidden_dim"], out_dim=hp["proj_out_dim"]).to(device)
        model = SimCLRModel(encoder, proj).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

        warmup_steps = int(hp["warmup_ratio"] * max_steps)
        warmup_steps = min(max(1, warmup_steps), max(1, max_steps - 1))

        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=max_steps,
            lr_min=1e-6,
            warmup_lr_init=1e-6,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

        criterion = NTXentLoss(temperature=hp["temperature"], device=device)
        scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

        # Manifest
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "group": group,
                    "base": base,
                    "seed": seed,
                    "notes": notes,
                    "train_dir": CFG.train_dir,
                    "image_size": image_size,
                    "batch_size": bs,
                    "max_steps": max_steps,
                    "drop_last": drop_last,
                    "log_every_steps": CFG.log_every_steps,
                    "hp": hp,
                    "aug_config": cfg.__dict__,
                    "encoder_params_source": "phase3_last_ckpt.meta.config.encoder_params",
                    "encoder_out_channels": int(encoder.out_channels),
                    "time_started": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Train metrics CSV (Phase3 schema)
        csv_path = run_dir / "train_metrics.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("seed,epoch,step,lr,train_loss\n")

        # Step-level log for dynamics / heatmaps (Phase3-style granularity)
        step_csv_path = run_dir / "train_steps.csv"
        with step_csv_path.open("w", encoding="utf-8") as f:
            f.write("seed,epoch,step,lr,train_loss\n")

        # NOTE: additional per-step logs are appended below during training.

        # Train loop (compute-capped by max_steps)
        global_step = 0
        epoch = 0
        t0 = time.time()

        while global_step < max_steps:
            epoch += 1
            model.train()
            epoch_loss_sum = 0.0
            seen = 0

            for (x1, x2, _) in tqdm(train_loader, desc=f"[A][{run_id}] epoch {epoch}", leave=False):
                if global_step >= max_steps:
                    break

                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    z1 = model(x1)
                    z2 = model(x2)
                    loss = criterion(z1, z2)

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])

                scaler.step(optimizer)
                scaler.update()

                scheduler.step_update(global_step)

                bsz = x1.size(0)
                epoch_loss_sum += float(loss.detach()) * bsz
                seen += bsz
                global_step += 1

                # Phase3-style log granularity: write every log_every_steps
                if CFG.log_every_steps > 0 and (global_step % CFG.log_every_steps == 0):
                    lr_now = optimizer.param_groups[0]["lr"]
                    avg_loss = epoch_loss_sum / max(1, seen)

                    print(
                        f"[A][{run_id}][seed={seed}] epoch={epoch} "
                        f"step={global_step} lr={lr_now:.3e} train_loss={avg_loss:.5f}"
                    )

                    with step_csv_path.open("a", encoding="utf-8") as f:
                        f.write(f"{seed},{epoch},{global_step},{lr_now:.8e},{avg_loss:.8f}\n")

                # (intentionally only one step-level logging block)

            lr_now = optimizer.param_groups[0]["lr"]
            avg_epoch_loss = epoch_loss_sum / max(1, seen)

            with csv_path.open("a", encoding="utf-8") as f:
                f.write(f"{seed},{epoch},{global_step},{lr_now:.8e},{avg_epoch_loss:.8f}\n")

        elapsed = time.time() - t0
        print(f"[done] {run_id} | steps={global_step}/{max_steps} | wall={elapsed/60:.1f} min")

        # Rep diagnostics (fixed compute)
        if CFG.compute_rep_metrics:
            align, uni = compute_alignment_uniformity_fixed(model, train_ds, device, seed)
            (run_dir / "rep_metrics.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "alignment": float(align),
                        "uniformity": float(uni),
                        "rep_metric_bs": CFG.rep_metric_bs,
                        "rep_metric_num_batches": CFG.rep_metric_num_batches,
                        "uniformity_t": CFG.uniformity_t,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        # Save LAST only
        ckpt = {
            "encoder_state_dict": model.encoder.state_dict(),
            "proj_head_state_dict": model.proj_head.state_dict(),
            "meta": {
                "phase": "aug_sensitivity_phase3_aligned",
                "run_id": run_id,
                "seed": seed,
                "epoch": epoch,
                "global_step": global_step,
                "max_steps": max_steps,
                "simclr_hp": hp,
                "train_files_count": len(train_files),
                "elapsed_sec": elapsed,
                "selection_note": "Train-only; last checkpoint saved; rep-metrics are diagnostics only.",
            },
        }
        torch.save(ckpt, str(run_dir / "last.pth"))

        done_flag.write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
