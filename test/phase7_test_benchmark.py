"""phase7_test_benchmark.py

Phase 7 — Final TEST benchmarking + deployment efficiency.

Paper-safe locked-box TEST rules:
  - No tuning / selection / decisions on TEST.
  - Evaluate ONLY already trained FINAL checkpoints (LAST).
  - Phase 7 recomputes TEST quality metrics deterministically (hard @0.5) for reporting:
      * Dice/F1 @0.5 (mean over images)
      * IoU @0.5 (mean over images)
      * Precision/Recall @0.5 (mean over images)
    and optionally resolution subgroups PV01/PV03/PV08 (+ OTHER).
  - Phase 7 also adds:
      * per-image metrics artifact (npz, for later plots)
      * efficiency: #params (total + encoder/decoder split), FLOPs/MACs @ 256×256, batch=1
      * deployment timing: latency (ms/img), FPS, peak RAM/VRAM (CPU/GPU), with warmup + repeats
      * secondary composite score: hardDice@0.5 / log10(1 + Params)

No CLI args by design:
  - Edit BenchmarkConfig below (Desktop vs Jetson Nano).
  - Select active config in __main__.

Progress logging:
  - The script prints phase markers and per-model progress to the console.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from fvcore.nn import FlopCountAnalysis

from models.UltraLightFCN_base import UltraLightFCN
from utils.config import ENCODER_PREFIXES, SEG_PARAMS
from utils.dataset import SolarPanelDataset


# -------------------------
# Configuration (no argparse)
# -------------------------

@dataclass(frozen=True)
class BenchmarkConfig:
    # Roster sources containing FINAL checkpoints to benchmark.
    # Supported formats:
    #   (A) generic manifest: list[dict] or {'models':[...]} with checkpoint_path etc.
    #   (B) Phase 6 test report JSON (has key 'runs' with 'ckpt_last_path').
    #   (C) Phase SOTA test report JSON (has key 'per_seed' with 'last_ckpt').
    roster_paths: Tuple[str, ...]

    # Dataset
    data_root: str = "../dataset"
    test_split: str = "test"
    input_size: int = 256
    threshold: float = 0.5

    # Quality evaluation (deterministic)
    recompute_quality: bool = True
    compute_per_image_metrics: bool = True  # store per-image vectors (for plots)
    per_image_include_iou: bool = True
    eval_batch: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    subgroups: bool = True  # PV01/PV03/PV08 (+ OTHER)

    # Efficiency
    compute_flops_macs: bool = True

    # Deployment benchmark
    measure_cpu: bool = True
    measure_gpu: bool = True
    use_amp_timing: bool = False  # AMP can change numerics; keep False unless explicitly needed
    warmup: int = 30
    iters: int = 200
    repeats: int = 10

    # Output
    out_root: str = "bench_phase7"


# Example configs (edit paths as needed)
CFG_DESKTOP = BenchmarkConfig(
    roster_paths=(
        "../train/seg_phase6/final_retrain90/phase6_test_report.json",
        "../train/seg_sota/stage2_final90_test/phaseSOTA_test_report.json",
    ),
    data_root="../dataset",
    test_split="test",
    eval_batch=16,
    num_workers=0,
    measure_cpu=True,
    measure_gpu=True,
    use_amp_timing=False,
    warmup=30,
    iters=200,
    repeats=10,
    out_root="bench_phase7",
)

CFG_JETSON = BenchmarkConfig(
    roster_paths=(
        "../train/seg_phase6/final_retrain90/phase6_test_report.json",
        "../train/seg_sota/stage2_final90_test/phaseSOTA_test_report.json",
    ),
    data_root="../dataset",
    test_split="test",
    eval_batch=8,
    num_workers=0,
    measure_cpu=True,
    measure_gpu=True,
    use_amp_timing=False,
    warmup=30,
    iters=200,
    repeats=10,
    out_root="bench_phase7_jetson",
)


# -------------------------
# Model roster
# -------------------------

@dataclass
class ModelEntry:
    """Single checkpoint to benchmark."""

    model_id: str
    seed: Optional[int]
    checkpoint_path: str

    # Model family
    family: str = "ultralight"  # "ultralight" | "smp"

    # SMP metadata
    arch: Optional[str] = None  # "DeepLabV3Plus" | "Unet"
    encoder_name: Optional[str] = None
    encoder_weights: Optional[str] = "imagenet"

    # Input/output
    in_channels: int = 3
    num_classes: int = 1

    # UltraLightFCN parameters
    seg_params: Optional[dict] = None


# -------------------------
# Determinism + audit
# -------------------------

def set_eval_determinism() -> None:
    """Deterministic inference for TEST evaluation."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_versions() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "smp": getattr(smp, "__version__", None),
    }


def device_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_device_index": int(idx),
                "cuda_name": props.name,
                "cuda_total_memory_bytes": int(props.total_memory),
                "cuda_multi_processor_count": int(props.multi_processor_count),
            }
        )
    return info


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write list-of-dicts to CSV with stable union-of-keys header (order preserved by first occurrence)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# Quality utilities
# -------------------------

def infer_resolution_group(name: str) -> str:
    """Infer PVxx group from filename."""
    base = Path(str(name)).name
    if base.startswith("PV01"):
        return "PV01"
    if base.startswith("PV03"):
        return "PV03"
    if base.startswith("PV08"):
        return "PV08"
    return "OTHER"


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def per_image_metrics_hard(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    thr: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-image (dice, iou, precision, recall) tensors shape (B,)."""
    prob = sigmoid(logits)
    pred = (prob > thr).float()

    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.float()

    B = pred.shape[0]
    p = pred.view(B, -1)
    g = mask.view(B, -1)

    inter = (p * g).sum(dim=1)
    p_sum = p.sum(dim=1)
    g_sum = g.sum(dim=1)

    dice = (2.0 * inter + eps) / (p_sum + g_sum + eps)
    union = p_sum + g_sum - inter
    iou = (inter + eps) / (union + eps)

    tp = inter
    fp = (p * (1.0 - g)).sum(dim=1)
    fn = ((1.0 - p) * g).sum(dim=1)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return dice, iou, precision, recall


def summarize_array(x: np.ndarray) -> Dict[str, float]:
    x = x.astype(np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def quality_from_vectors(
    dice: np.ndarray,
    iou: Optional[np.ndarray],
    precision: np.ndarray,
    recall: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dice_hard05": summarize_array(dice),
        "precision_hard05": summarize_array(precision),
        "recall_hard05": summarize_array(recall),
    }
    if iou is not None:
        out["iou_hard05"] = summarize_array(iou)
    return out


@torch.no_grad()
def compute_per_image_on_test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    *,
    device: torch.device,
    thr: float,
    include_iou: bool,
    subgroups: bool,
) -> Dict[str, Any]:
    """Compute per-image hard Dice/IoU/Prec/Rec on TEST (deterministic)."""
    model.eval()
    model.to(device)

    dice_all: List[float] = []
    iou_all: List[float] = []
    prec_all: List[float] = []
    rec_all: List[float] = []
    group_all: List[str] = []

    total_batches = len(test_loader)
    for bi, batch in enumerate(test_loader, start=1):
        # Dataset may return: (x, y) or (x, y, orig, names)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0]
            y = batch[1]
            names = batch[-1] if len(batch) >= 4 else None
        else:
            raise RuntimeError("Unexpected batch format from SolarPanelDataset")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        dice, iou, prec, rec = per_image_metrics_hard(logits, y, thr=thr)

        dice_all.extend(dice.detach().cpu().numpy().tolist())
        if include_iou:
            iou_all.extend(iou.detach().cpu().numpy().tolist())
        prec_all.extend(prec.detach().cpu().numpy().tolist())
        rec_all.extend(rec.detach().cpu().numpy().tolist())

        if subgroups:
            if names is None:
                group_all.extend(["OTHER"] * int(x.shape[0]))
            else:
                group_all.extend([infer_resolution_group(str(n)) for n in list(names)])

        # Progress print every ~10% or every 50 batches (whichever is smaller)
        if bi == 1 or bi == total_batches or (bi % max(1, min(50, total_batches // 10)) == 0):
            print(f"      [quality] batches {bi:>4d}/{total_batches}  (accumulated images={len(dice_all)})")

    dice_np = np.asarray(dice_all, dtype=np.float32)
    prec_np = np.asarray(prec_all, dtype=np.float32)
    rec_np = np.asarray(rec_all, dtype=np.float32)
    iou_np = np.asarray(iou_all, dtype=np.float32) if include_iou else None

    out: Dict[str, Any] = {
        "n": int(dice_np.shape[0]),
        "threshold": float(thr),
        "dice": dice_np,
        "precision": prec_np,
        "recall": rec_np,
    }
    if include_iou and iou_np is not None:
        out["iou"] = iou_np
    if subgroups:
        out["group"] = np.asarray(group_all, dtype=object)

    # Summaries (overall + optional subgroups)
    summaries: Dict[str, Any] = {"overall": {"n": int(out["n"]), **quality_from_vectors(dice_np, iou_np, prec_np, rec_np)}}

    if subgroups:
        groups = out["group"]
        for g in ["PV01", "PV03", "PV08", "OTHER"]:
            idx = (groups == g)
            if not np.any(idx):
                continue
            g_dice = dice_np[idx]
            g_prec = prec_np[idx]
            g_rec = rec_np[idx]
            g_iou = iou_np[idx] if iou_np is not None else None
            summaries[g] = {"n": int(np.sum(idx)), **quality_from_vectors(g_dice, g_iou, g_prec, g_rec)}

    out["summary"] = summaries
    return out


def composite_score(dice_hard05_mean: float, params_total: int) -> float:
    return float(dice_hard05_mean) / max(1e-12, math.log10(1.0 + float(params_total)))


# -------------------------
# Params + FLOPs/MACs
# -------------------------

def count_params(params: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params if p is not None))


def split_params_ultralight(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (encoder_params, decoder_params) for UltraLightFCN via ENCODER_PREFIXES."""
    enc_params: List[torch.nn.Parameter] = []
    dec_params: List[torch.nn.Parameter] = []
    prefixes = tuple(str(p) for p in ENCODER_PREFIXES)
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            enc_params.append(p)
        else:
            dec_params.append(p)
    return count_params(enc_params), count_params(dec_params)


def split_params_smp(model: torch.nn.Module) -> Tuple[int, int]:
    enc = count_params(model.encoder.parameters()) if hasattr(model, "encoder") else 0
    total = count_params(model.parameters())
    return enc, max(0, total - enc)


def compute_flops_macs(
    model: torch.nn.Module,
    device: torch.device,
    *,
    input_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Compute FLOPs/MACs for a single forward pass at fixed resolution."""
    model = model.to(device)
    model.eval()
    C, H, W = 3, input_size[0], input_size[1]
    x = torch.zeros(1, C, H, W, device=device)

    with torch.no_grad():
        flops = FlopCountAnalysis(model, x)
        total_flops = int(flops.total())

    # MACs is commonly FLOPs/2 for conv-like ops; keep this convention for reporting.
    return {"tool": "fvcore", "flops": total_flops, "macs": int(total_flops // 2)}


# -------------------------
# Latency + memory benchmark
# -------------------------

@torch.no_grad()
def time_inference(
    model: torch.nn.Module,
    device: torch.device,
    *,
    input_size: Tuple[int, int],
    warmup: int,
    iters: int,
    use_amp: bool,
    repeat_idx: int,
) -> Dict[str, Any]:
    """Latency benchmark with batch=1 on fixed dummy input. Repeats do not affect predictions."""
    model = model.to(device)
    model.eval()

    C, H, W = 3, input_size[0], input_size[1]
    x = torch.zeros(1, C, H, W, device=device)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    proc = psutil.Process(os.getpid())
    rss_peak = 0

    # Warm-up
    for _ in range(max(0, warmup)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x)
        else:
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x)
        else:
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        rss = int(proc.memory_info().rss)
        if rss > rss_peak:
            rss_peak = rss

    t1 = time.perf_counter()

    total_s = max(1e-12, (t1 - t0))
    n = max(1, iters)
    ms_per_img = (total_s / n) * 1000.0
    fps = n / total_s

    vram_peak = None
    vram_reserved_peak = None
    if device.type == "cuda" and torch.cuda.is_available():
        vram_peak = int(torch.cuda.max_memory_allocated(device))
        vram_reserved_peak = int(torch.cuda.max_memory_reserved(device))

    return {
        "repeat_idx": int(repeat_idx),
        "warmup": int(warmup),
        "iters": int(iters),
        "ms_per_img": float(ms_per_img),
        "fps": float(fps),
        "peak_rss_bytes": int(rss_peak),
        "peak_vram_alloc_bytes": vram_peak,
        "peak_vram_reserved_bytes": vram_reserved_peak,
    }


def aggregate_rows(
    rows: List[Dict[str, Any]],
    *,
    group_keys: List[str],
    value_keys: List[str],
) -> List[Dict[str, Any]]:
    from collections import defaultdict

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g = tuple(r.get(k) for k in group_keys)
        groups[g].append(r)

    out: List[Dict[str, Any]] = []
    for g, items in groups.items():
        agg: Dict[str, Any] = {k: v for k, v in zip(group_keys, g)}
        agg["row_type"] = "aggregate"
        agg["n_items"] = len(items)

        for vk in value_keys:
            vals = [x.get(vk) for x in items if isinstance(x.get(vk), (int, float))]
            if not vals:
                agg[f"{vk}_mean"] = None
                agg[f"{vk}_std"] = None
                continue
            m = float(sum(float(v) for v in vals) / len(vals))
            var = float(sum((float(v) - m) ** 2 for v in vals) / len(vals))
            agg[f"{vk}_mean"] = m
            agg[f"{vk}_std"] = math.sqrt(var)

        out.append(agg)

    return out


# -------------------------
# Models: build + load checkpoint
# -------------------------

def build_model(entry: ModelEntry) -> torch.nn.Module:
    fam = entry.family.lower()

    if fam == "ultralight":
        seg_params = entry.seg_params or dict(SEG_PARAMS)
        return UltraLightFCN(in_channels=entry.in_channels, num_classes=entry.num_classes, params=seg_params)

    if fam == "smp":
        if entry.arch is None or entry.encoder_name is None:
            raise RuntimeError(f"SMP entry requires arch and encoder_name: {entry}")
        cls = getattr(smp, entry.arch)
        return cls(
            encoder_name=entry.encoder_name,
            encoder_weights=entry.encoder_weights,
            in_channels=entry.in_channels,
            classes=entry.num_classes,
            activation=None,
        )

    raise RuntimeError(f"Unknown model family: {entry.family}")


def load_checkpoint(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")
    return ckpt


def extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    # Support common formats
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    # Otherwise assume it's already a state dict
    return ckpt


# -------------------------
# Manifests -> canonical entries
# -------------------------


def _infer_smp_meta(model_name: str) -> Tuple[str, str]:
    """Infer (arch, encoder_name) for SMP models from a compact model_name."""
    name = (model_name or "").lower().strip()

    # Expected patterns in your reports:
    #   dlv3p_resnet50, dlv3p_mobilenetv2, unet_resnet34
    if name.startswith("dlv3p_"):
        arch = "DeepLabV3Plus"
        enc = name.replace("dlv3p_", "")
        if enc == "mobilenetv2":
            enc = "mobilenet_v2"  # SMP naming
        return arch, enc

    if name.startswith("unet_"):
        arch = "Unet"
        enc = name.replace("unet_", "")
        return arch, enc

    # Fallback: treat as DeepLabV3Plus with raw encoder token
    return "DeepLabV3Plus", name


def load_manifest(path: str) -> List[ModelEntry]:
    """Load a roster source file and convert it into a list of ModelEntry.

    Supported inputs:
      - Generic manifest: list[dict] or {'models':[...]} with checkpoint_path.
      - Phase 6 test report JSON: has key 'runs' and each run has 'ckpt_last_path'.
      - Phase SOTA test report JSON: has key 'per_seed' and each run has 'last_ckpt'.
    """
    obj = read_json(path)

    out: List[ModelEntry] = []

    # -----------------------------
    # (B) Phase 6 test report format
    # -----------------------------
    if isinstance(obj, dict) and isinstance(obj.get("runs"), list) and obj.get("phase") == 6:
        cand = None
        try:
            cand = obj.get("winner_source", {}).get("candidate_id", None)
        except Exception:
            cand = None

        for r in obj["runs"]:
            if not isinstance(r, dict):
                continue
            ckpt = r.get("ckpt_last_path")
            if not ckpt:
                continue
            seed = r.get("seed", None)
            model_id = f"ultralight_phase6"
            if cand is not None:
                model_id += f"_cand{cand}"
            if seed is not None:
                model_id += f"_seed{seed}"

            out.append(
                ModelEntry(
                    model_id=model_id,
                    seed=int(seed) if seed is not None else None,
                    checkpoint_path=str(ckpt),
                    family="ultralight",
                    arch=None,
                    encoder_name=None,
                    encoder_weights=None,
                    in_channels=3,
                    num_classes=1,
                    seg_params=None,  # will use SEG_PARAMS from config
                )
            )
        return out

    # --------------------------------
    # (C) Phase SOTA test report format
    # --------------------------------
    if isinstance(obj, dict) and isinstance(obj.get("per_seed"), list):
        for r in obj["per_seed"]:
            if not isinstance(r, dict):
                continue
            if str(r.get("status", "ok")).lower() != "ok":
                continue

            ckpt = r.get("last_ckpt")
            if not ckpt:
                continue

            model_name = str(r.get("model_name", "sota_model"))
            regime = str(r.get("regime", ""))
            seed = r.get("seed", None)

            arch, enc = _infer_smp_meta(model_name)
            model_id = f"{model_name}::{regime}"
            if seed is not None:
                model_id += f"::seed_{seed}"

            out.append(
                ModelEntry(
                    model_id=model_id,
                    seed=int(seed) if seed is not None else None,
                    checkpoint_path=str(ckpt),
                    family="smp",
                    arch=arch,
                    encoder_name=enc,
                    encoder_weights="imagenet",
                    in_channels=3,
                    num_classes=1,
                    seg_params=None,
                )
            )
        return out

    # -----------------------------
    # (A) Generic manifest format
    # -----------------------------
    items = obj.get("models", obj) if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise RuntimeError(f"Roster must be list or {{'models':[...]}} or a known report format: {path}")

    for it in items:
        if not isinstance(it, dict):
            continue

        ckpt = it.get("checkpoint_path") or it.get("ckpt_path") or it.get("last_ckpt") or it.get("path")
        if not ckpt:
            continue

        out.append(
            ModelEntry(
                model_id=str(it.get("model_id", it.get("model_name", Path(str(ckpt)).parent.name))),
                seed=int(it["seed"]) if "seed" in it and it["seed"] is not None else None,
                checkpoint_path=str(ckpt),
                family=str(it.get("family", "ultralight")),
                arch=it.get("arch"),
                encoder_name=it.get("encoder_name", it.get("encoder")),
                encoder_weights=it.get("encoder_weights", "imagenet"),
                in_channels=int(it.get("in_channels", 3)),
                num_classes=int(it.get("num_classes", it.get("classes", 1))),
                seg_params=it.get("seg_params"),
            )
        )

    return out


def build_roster_from_sources(roster_paths: Tuple[str, ...]) -> List[ModelEntry]:
    entries: List[ModelEntry] = []
    for mp in roster_paths:
        entries.extend(load_manifest(mp))

    # De-dup by checkpoint path (keep first occurrence)
    seen = set()
    uniq: List[ModelEntry] = []
    for e in entries:
        if e.checkpoint_path in seen:
            continue
        seen.add(e.checkpoint_path)
        uniq.append(e)

    return uniq


# -------------------------
# Dataset loader
# -------------------------

def build_test_loader(cfg: BenchmarkConfig) -> DataLoader:
    test_dir = os.path.join(cfg.data_root, cfg.test_split)

    # return_extra=True ensures names are available for subgroup tagging
    ds = SolarPanelDataset(test_dir, mode="test", files=None, return_extra=bool(cfg.subgroups))

    return DataLoader(
        ds,
        batch_size=int(cfg.eval_batch),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory and torch.cuda.is_available()),
        drop_last=False,
    )


# -------------------------
# Main pipeline
# -------------------------

def run_phase7(cfg: BenchmarkConfig) -> Path:
    set_eval_determinism()

    # Devices to benchmark
    devices: List[torch.device] = []
    if cfg.measure_cpu or (not cfg.measure_gpu):
        devices.append(torch.device("cpu"))
    if cfg.measure_gpu and torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    # Output folder
    run_dir = Path(cfg.out_root) / now_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PHASE 1/6] Writing config + audit to: {run_dir}")
    write_json(str(run_dir / "config.json"), asdict(cfg))

    print("[PHASE 2/6] Loading manifests (model roster)")
    entries = build_roster_from_sources(cfg.roster_paths)
    if not entries:
        raise RuntimeError("No model entries found in manifests.")
    print(f"  -> found {len(entries)} unique checkpoints")

    print("[PHASE 3/6] Building TEST DataLoader (locked-box)")
    test_loader = build_test_loader(cfg)
    print(f"  -> TEST batches: {len(test_loader)}  (batch_size={cfg.eval_batch}, workers={cfg.num_workers})")

    print("[PHASE 4/6] Benchmarking models (quality + efficiency + timing)")
    per_image_dir = run_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    # Master report object
    master: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "locked_box_test": True,
        "phase": "phase7_test_benchmark",
        "config": asdict(cfg),
        "versions": get_versions(),
        "devices": {str(d): device_info(d) for d in devices},
        "resolution_groups": {
            "PV01": {"gsd_m": 0.1, "size_px": [256, 256]},
            "PV03": {"gsd_m": 0.3, "size_px": [256, 256]},
            "PV08": {"gsd_m": 0.8, "size_px": [256, 256]},
        },
        "models": [],
        "results": {},
        "artifacts": {},
    }

    timing_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []

    for mi, entry in enumerate(entries, start=1):
        model_key = f"{entry.model_id}::seed{entry.seed}" if entry.seed is not None else f"{entry.model_id}::seedNA"
        print(f"  [model {mi:>2d}/{len(entries)}] {model_key}")
        print(f"      ckpt: {entry.checkpoint_path}")

        # Build + load checkpoint
        model = build_model(entry)
        ckpt = load_checkpoint(entry.checkpoint_path)
        state = extract_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)

        # Efficiency: params + split
        params_total = count_params(model.parameters())
        if entry.family.lower() == "smp":
            params_enc, params_dec = split_params_smp(model)
        else:
            params_enc, params_dec = split_params_ultralight(model)

        # FLOPs/MACs on CPU for consistency
        flops_info = {"tool": None, "flops": None, "macs": None}
        if cfg.compute_flops_macs:
            flops_info = compute_flops_macs(model, torch.device("cpu"), input_size=(cfg.input_size, cfg.input_size))

        # Quality recompute (deterministic)
        per_image_artifact_path = None
        per_image_summary = None
        quality_summary = None
        if cfg.recompute_quality:
            print("      [quality] recomputing on TEST (CPU, deterministic)")
            per_image = compute_per_image_on_test(
                model,
                test_loader,
                device=torch.device("cpu"),
                thr=cfg.threshold,
                include_iou=cfg.per_image_include_iou,
                subgroups=cfg.subgroups,
            )
            per_image_summary = per_image["summary"]
            quality_summary = per_image_summary  # already contains overall + subgroup summaries

            # Save per-image vectors as compressed NPZ (better than CSV for big arrays)
            if cfg.compute_per_image_metrics:
                safe_id = entry.model_id.replace("/", "_").replace("::", "__")
                seed_tag = f"seed{entry.seed}" if entry.seed is not None else "seedNA"
                per_image_artifact_path = str(per_image_dir / f"{safe_id}_{seed_tag}_per_image.npz")

                npz_payload: Dict[str, Any] = {
                    "dice": per_image["dice"],
                    "precision": per_image["precision"],
                    "recall": per_image["recall"],
                }
                if cfg.per_image_include_iou and "iou" in per_image:
                    npz_payload["iou"] = per_image["iou"]
                if cfg.subgroups and "group" in per_image:
                    npz_payload["group"] = per_image["group"]

                np.savez_compressed(per_image_artifact_path, **npz_payload)

        # Composite score uses overall Dice@0.5 mean
        dice_mean = None
        if quality_summary is not None and "overall" in quality_summary:
            dice_mean = float(quality_summary["overall"]["dice_hard05"]["mean"])  # type: ignore
        score = composite_score(dice_mean, params_total) if (dice_mean is not None and params_total > 0) else None

        # Record quality to CSV rows (overall + subgroups)
        if quality_summary is not None:
            for subset_key, subset_val in quality_summary.items():
                if not isinstance(subset_val, dict) or "n" not in subset_val:
                    continue
                row = {
                    "row_type": "quality_summary",
                    "model_id": entry.model_id,
                    "seed": entry.seed,
                    "family": entry.family,
                    "arch": entry.arch,
                    "encoder": entry.encoder_name,
                    "subset": subset_key,
                    "n_images": subset_val.get("n"),
                    "dice_mean": subset_val.get("dice_hard05", {}).get("mean"),
                    "dice_std": subset_val.get("dice_hard05", {}).get("std"),
                    "iou_mean": subset_val.get("iou_hard05", {}).get("mean") if subset_val.get("iou_hard05") else None,
                    "iou_std": subset_val.get("iou_hard05", {}).get("std") if subset_val.get("iou_hard05") else None,
                    "precision_mean": subset_val.get("precision_hard05", {}).get("mean"),
                    "precision_std": subset_val.get("precision_hard05", {}).get("std"),
                    "recall_mean": subset_val.get("recall_hard05", {}).get("mean"),
                    "recall_std": subset_val.get("recall_hard05", {}).get("std"),
                }
                quality_rows.append(row)

        # Timing benchmark per device (repeats)
        timing_per_device: Dict[str, Any] = {}
        for dev in devices:
            print(f"      [timing] device={dev} repeats={cfg.repeats} warmup={cfg.warmup} iters={cfg.iters}")
            per_repeats: List[Dict[str, Any]] = []
            for r in range(int(cfg.repeats)):
                t = time_inference(
                    model,
                    dev,
                    input_size=(cfg.input_size, cfg.input_size),
                    warmup=int(cfg.warmup),
                    iters=int(cfg.iters),
                    use_amp=bool(cfg.use_amp_timing),
                    repeat_idx=r,
                )
                per_repeats.append(t)

                timing_rows.append(
                    {
                        "row_type": "per_repeat",
                        "model_id": entry.model_id,
                        "seed": entry.seed,
                        "family": entry.family,
                        "arch": entry.arch,
                        "encoder": entry.encoder_name,
                        "device": str(dev),
                        "repeat_idx": r,
                        "ckpt_path": entry.checkpoint_path,
                        "params_total": params_total,
                        "params_encoder": params_enc,
                        "params_decoder": params_dec,
                        "flops": flops_info.get("flops"),
                        "macs": flops_info.get("macs"),
                        "dice_overall_mean": dice_mean,
                        "composite_score": score,
                        "ms_per_img": t.get("ms_per_img"),
                        "fps": t.get("fps"),
                        "peak_rss_bytes": t.get("peak_rss_bytes"),
                        "peak_vram_alloc_bytes": t.get("peak_vram_alloc_bytes"),
                        "peak_vram_reserved_bytes": t.get("peak_vram_reserved_bytes"),
                    }
                )

            agg = aggregate_rows(
                [{"model_id": entry.model_id, "seed": entry.seed, "device": str(dev), **x} for x in per_repeats],
                group_keys=["model_id", "seed", "device"],
                value_keys=["ms_per_img", "fps", "peak_rss_bytes", "peak_vram_alloc_bytes", "peak_vram_reserved_bytes"],
            )
            timing_per_device[str(dev)] = {"per_repeat": per_repeats, "aggregate": agg[0] if agg else None}

        # Master JSON entry
        master["models"].append(
            {
                "model_key": model_key,
                "model_id": entry.model_id,
                "seed": entry.seed,
                "family": entry.family,
                "arch": entry.arch,
                "encoder": entry.encoder_name,
                "encoder_weights": entry.encoder_weights,
                "ckpt_path": entry.checkpoint_path,
            }
        )
        master["results"][model_key] = {
            "quality": quality_summary,  # recomputed summaries
            "efficiency": {
                "params_total": params_total,
                "params_encoder": params_enc,
                "params_decoder": params_dec,
                "flops": flops_info.get("flops"),
                "macs": flops_info.get("macs"),
                "flops_tool": flops_info.get("tool"),
                "input_size": [cfg.input_size, cfg.input_size],
                "batch": 1,
            },
            "composite_score": score,
            "per_image": {"artifact_path": per_image_artifact_path} if cfg.compute_per_image_metrics else None,
            "deployment": {
                "timing": timing_per_device,
                "timing_repeats": int(cfg.repeats),
                "warmup": int(cfg.warmup),
                "iters": int(cfg.iters),
                "use_amp": bool(cfg.use_amp_timing),
            },
            "state_dict": {
                "missing": list(missing) if isinstance(missing, (list, tuple)) else str(missing),
                "unexpected": list(unexpected) if isinstance(unexpected, (list, tuple)) else str(unexpected),
            },
        }

    print("[PHASE 5/6] Writing CSV artifacts")
    timing_csv = str(run_dir / "phase7_timing_per_repeat.csv")
    save_csv_rows(timing_csv, timing_rows)

    timing_agg = aggregate_rows(
        timing_rows,
        group_keys=["model_id", "seed", "device"],
        value_keys=["ms_per_img", "fps", "peak_rss_bytes", "peak_vram_alloc_bytes", "peak_vram_reserved_bytes"],
    )
    timing_agg_csv = str(run_dir / "phase7_timing_aggregate.csv")
    save_csv_rows(timing_agg_csv, timing_agg)

    quality_csv = str(run_dir / "phase7_quality_summary.csv")
    save_csv_rows(quality_csv, quality_rows)

    master["artifacts"]["timing_per_repeat_csv"] = timing_csv
    master["artifacts"]["timing_aggregate_csv"] = timing_agg_csv
    master["artifacts"]["quality_summary_csv"] = quality_csv

    print("[PHASE 6/6] Writing master JSON report")
    master_path = run_dir / "phase7_master_report.json"
    write_json(str(master_path), master)

    print(f"[ok] wrote master report: {master_path}")
    return master_path


if __name__ == "__main__":
    # Select one config explicitly.
    # - Desktop: CFG_DESKTOP
    # - Jetson:  CFG_JETSON
    run_phase7(CFG_DESKTOP)
