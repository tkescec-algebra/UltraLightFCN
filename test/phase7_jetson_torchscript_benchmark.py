"""
Phase 7 (Jetson/Desktop) — TEST benchmarking + deployment efficiency for TorchScript (.ts) models.

This variant preserves the benchmark methodology while isolating each model in its own
child process. The purpose is robustness on memory-constrained Jetson devices:
  - if one model is OOM-killed by the OS, the overall benchmark continues,
  - the failed model is explicitly logged as failed/OOM,
  - successful models are measured exactly as before.

Methodology is unchanged for successful models:
  - same TEST split,
  - same preprocessing,
  - same deterministic quality recompute,
  - same timing procedure,
  - same CSV/JSON reporting schema for successful runs.

Hidden internal CLI is used only for parent/child orchestration.
"""

import argparse
import csv
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import cv2
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset


# --------------------------
# Desktop-aligned dataset + transforms (no albumentations dependency)
# --------------------------

def _resize_longest_side(arr: np.ndarray, max_size: int, interpolation: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) == max_size:
        return arr
    scale = float(max_size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation)


def _desktop_test_geo_transform(image: np.ndarray, mask: np.ndarray, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    img = _resize_longest_side(image, image_size, cv2.INTER_LINEAR)
    msk = _resize_longest_side(mask, image_size, cv2.INTER_NEAREST)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    msk = cv2.resize(msk, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    if msk.ndim != 2:
        msk = np.squeeze(msk)
    msk = (msk > 0.5).astype(np.float32)
    return img, msk


def _imagenet_normalize_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(image)).float()


class SolarPanelDataset(Dataset):
    IMG_EXTS = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        files=None,
        mask_suffix: str = "_label",
        mask_ext: str = ".png",
        return_extra: bool = False,
        image_size: int = 256,
    ):
        assert mode in ("train", "valid", "test"), f"Invalid mode: {mode}"
        if mode != "test":
            raise RuntimeError(
                "This Jetson benchmark dataset implementation is intentionally locked to mode='test' "
                "to mirror the desktop test benchmark without introducing train/valid augmentation dependencies."
            )
        self.data_dir = data_dir
        self.mode = mode
        self.mask_suffix = mask_suffix
        self.mask_ext = mask_ext
        self.return_extra = bool(return_extra)
        self.image_size = int(image_size)

        mask_tail = f"{self.mask_suffix}{self.mask_ext}".lower()
        if files is not None:
            imgs = [
                f for f in list(files)
                if f.lower().endswith(self.IMG_EXTS) and (not f.lower().endswith(mask_tail))
            ]
        else:
            imgs = [
                f for f in os.listdir(self.data_dir)
                if f.lower().endswith(self.IMG_EXTS) and (not f.lower().endswith(mask_tail))
            ]

        imgs = sorted(imgs)
        if len(imgs) == 0:
            raise RuntimeError(f"No images found in {self.data_dir}")
        if any(f.lower().endswith(mask_tail) for f in imgs):
            raise RuntimeError("Mask leakage: mask files ended up in the image list.")
        self.images = imgs

    def __len__(self):
        return len(self.images)

    def _resolve_mask_path(self, img_name: str) -> str:
        stem, _ = os.path.splitext(img_name)
        mask_name = f"{stem}{self.mask_suffix}{self.mask_ext}"
        mask_path = os.path.join(self.data_dir, mask_name)
        if not os.path.isfile(mask_path):
            raise RuntimeError(
                f"Mask not found for image '{img_name}'. Expected '{mask_name}' in: {self.data_dir}"
            )
        return mask_path

    def __getitem__(self, idx: int):
        name = self.images[idx]
        img_path = os.path.join(self.data_dir, name)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_path = self._resolve_mask_path(name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        mask = (mask.astype(np.float32) > 0).astype(np.float32)

        img_aug, mask_aug = _desktop_test_geo_transform(img, mask, self.image_size)
        img_tensor = _imagenet_normalize_to_tensor(img_aug)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_aug)).unsqueeze(0).float()

        if self.return_extra:
            return img_tensor, mask_tensor, name
        return img_tensor, mask_tensor


# -------------------------
# Configuration
# -------------------------

@dataclass(frozen=True)
class BenchmarkConfig:
    ts_root: str
    data_root: str = "dataset"
    test_split: str = "test"
    input_size: int = 256
    threshold: float = 0.5
    recompute_quality: bool = True
    compute_per_image_metrics: bool = True
    per_image_include_iou: bool = True
    eval_batch: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    subgroups: bool = True
    compute_flops_macs: bool = False
    measure_cpu: bool = True
    measure_gpu: bool = True
    use_amp_timing: bool = False
    warmup: int = 30
    iters: int = 200
    repeats: int = 10
    out_root: str = "bench_phase7_jetson_ts"


CFG_JETSON_TS = BenchmarkConfig(
    ts_root="tools/export_torchscript_10",
    data_root="dataset",
    test_split="test",
    eval_batch=16,
    num_workers=0,
    measure_cpu=True,
    measure_gpu=True,
    pin_memory=False,
    use_amp_timing=False,
    warmup=30,
    iters=200,
    repeats=10,
    out_root="bench_phase7_jetson_ts",
)


# -------------------------
# Helpers
# -------------------------

def set_eval_determinism() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_versions() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "smp": None,
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


def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
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


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# TorchScript roster
# -------------------------

@dataclass
class TSEntry:
    name: str
    ts_path: str
    meta_path: Optional[str]


def load_ts_roster(ts_root: str) -> List[TSEntry]:
    root = Path(ts_root)
    if not root.exists():
        raise RuntimeError("ts_root does not exist: %s" % str(root))

    ts_files = sorted(root.glob("*.ts"))
    if not ts_files:
        raise RuntimeError("No .ts files found under: %s" % str(root))

    out: List[TSEntry] = []
    for ts in ts_files:
        meta = root / ("%s.meta.json" % ts.stem)
        out.append(TSEntry(name=ts.stem, ts_path=str(ts), meta_path=str(meta) if meta.exists() else None))
    return out


def parse_meta(meta_path: Optional[str]) -> Dict[str, Any]:
    if not meta_path:
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _pick_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return default


def normalize_ts_metadata(name: str, ts_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(meta or {})
    model_id = _pick_first(meta, ["model_id", "name", "model_name", "source_model_id"], default=name)
    seed = _pick_first(meta, ["seed", "random_seed"])
    family = _pick_first(meta, ["family", "model_family"])
    arch = _pick_first(meta, ["arch", "architecture"])
    encoder = _pick_first(meta, ["encoder", "encoder_name"])
    encoder_weights = _pick_first(meta, ["encoder_weights"])
    ckpt_path = _pick_first(meta, ["ckpt_path", "checkpoint_path", "last_ckpt", "source_ckpt_path", "source_checkpoint_path"])

    base = str(model_id or name)
    m_seed = re.search(r'(?:^|[_:])seed[_-]?(\d+)(?:$|[_:])', base)
    if seed is None:
        m_seed_name = re.search(r'(?:^|[_:])seed[_-]?(\d+)(?:$|[_:])', str(name))
        m_seed_path = re.search(r'(?:^|[/\\])seed[_-]?(\d+)(?:$|[/\\])', str(ts_path))
        m = m_seed or m_seed_name or m_seed_path
        if m:
            try:
                seed = int(m.group(1))
            except Exception:
                seed = None

    low = base.lower()
    if family is None:
        family = "smp" if (low.startswith("dlv3p_") or low.startswith("unet_") or arch is not None or encoder is not None) else "ultralight"

    if arch is None and low.startswith("dlv3p_"):
        arch = "DeepLabV3Plus"
    elif arch is None and low.startswith("unet_"):
        arch = "Unet"

    if encoder is None and low.startswith("dlv3p_"):
        enc = low.replace("dlv3p_", "")
        encoder = "mobilenet_v2" if enc == "mobilenetv2" else enc
    elif encoder is None and low.startswith("unet_"):
        encoder = low.replace("unet_", "")

    if encoder_weights is None and family == "smp" and encoder is not None:
        encoder_weights = "imagenet"

    try:
        seed = int(seed) if seed is not None else None
    except Exception:
        seed = None

    return {
        "model_id": str(model_id),
        "seed": seed,
        "family": family,
        "arch": arch,
        "encoder": encoder,
        "encoder_weights": encoder_weights,
        "ckpt_path": ckpt_path,
        "ts_path": ts_path,
        "meta_path": _pick_first(meta, ["meta_path"], default=None),
    }


def desktop_compatible_config(cfg: BenchmarkConfig) -> Dict[str, Any]:
    return {
        "roster_paths": [cfg.ts_root],
        "data_root": cfg.data_root,
        "test_split": cfg.test_split,
        "input_size": cfg.input_size,
        "threshold": cfg.threshold,
        "recompute_quality": cfg.recompute_quality,
        "compute_per_image_metrics": cfg.compute_per_image_metrics,
        "per_image_include_iou": cfg.per_image_include_iou,
        "eval_batch": cfg.eval_batch,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "subgroups": cfg.subgroups,
        "compute_flops_macs": cfg.compute_flops_macs,
        "measure_cpu": cfg.measure_cpu,
        "measure_gpu": cfg.measure_gpu,
        "use_amp_timing": cfg.use_amp_timing,
        "warmup": cfg.warmup,
        "iters": cfg.iters,
        "repeats": cfg.repeats,
        "out_root": cfg.out_root,
        "ts_root": cfg.ts_root,
    }


# -------------------------
# Dataset loader
# -------------------------

def build_test_loader(cfg: BenchmarkConfig) -> DataLoader:
    test_dir = os.path.join(cfg.data_root, cfg.test_split)
    ds = SolarPanelDataset(
        test_dir,
        mode="test",
        files=None,
        return_extra=bool(cfg.subgroups),
        image_size=int(cfg.input_size),
    )
    return DataLoader(
        ds,
        batch_size=int(cfg.eval_batch),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory and torch.cuda.is_available()),
        drop_last=False,
    )


# -------------------------
# Quality utilities
# -------------------------

def infer_resolution_group(name: str) -> str:
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


def per_image_metrics_hard(logits: torch.Tensor, mask: torch.Tensor, *, thr: float, eps: float = 1e-6):
    prob = sigmoid(logits)
    pred = (prob > thr).float()
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    b = pred.shape[0]
    p = pred.view(b, -1)
    g = mask.view(b, -1)
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


def quality_from_vectors(dice: np.ndarray, iou: Optional[np.ndarray], precision: np.ndarray, recall: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dice_hard05": summarize_array(dice),
        "precision_hard05": summarize_array(precision),
        "recall_hard05": summarize_array(recall),
    }
    if iou is not None:
        out["iou_hard05"] = summarize_array(iou)
    return out


@torch.no_grad()
def compute_per_image_on_test(model, test_loader: DataLoader, *, device: torch.device, thr: float, include_iou: bool, subgroups: bool) -> Dict[str, Any]:
    model.eval()
    try:
        model.to(device)
    except Exception:
        pass

    dice_all: List[float] = []
    iou_all: List[float] = []
    prec_all: List[float] = []
    rec_all: List[float] = []
    group_all: List[str] = []

    total_batches = len(test_loader)
    for bi, batch in enumerate(test_loader, start=1):
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0]
            y = batch[1]
            names = batch[-1] if len(batch) >= 3 else None
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

        if bi == 1 or bi == total_batches or (bi % max(1, min(50, total_batches // 10)) == 0):
            print("      [quality] batches %4d/%d  (accumulated images=%d)" % (bi, total_batches, len(dice_all)), flush=True)

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


def count_params_ts(model) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()))


def _amp_autocast_ctx(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return None
    if not hasattr(torch, "cuda") or not hasattr(torch.cuda, "amp") or not hasattr(torch.cuda.amp, "autocast"):
        return None
    return torch.cuda.amp.autocast()


@torch.no_grad()
def time_inference(model, device: torch.device, *, input_size: Tuple[int, int], warmup: int, iters: int, use_amp: bool, repeat_idx: int) -> Dict[str, Any]:
    model.eval()
    c, h, w = 3, input_size[0], input_size[1]
    x = torch.zeros(1, c, h, w, device=device)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass

    proc = psutil.Process(os.getpid())
    rss_peak = 0

    for _ in range(max(0, warmup)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ctx = _amp_autocast_ctx(use_amp, device)
        if ctx is None:
            _ = model(x)
        else:
            with ctx:
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        ctx = _amp_autocast_ctx(use_amp, device)
        if ctx is None:
            _ = model(x)
        else:
            with ctx:
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
        if hasattr(torch.cuda, "max_memory_allocated"):
            try:
                vram_peak = int(torch.cuda.max_memory_allocated(device))
            except Exception:
                vram_peak = None
        if hasattr(torch.cuda, "max_memory_reserved"):
            try:
                vram_reserved_peak = int(torch.cuda.max_memory_reserved(device))
            except Exception:
                vram_reserved_peak = None

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


def aggregate_rows(rows: List[Dict[str, Any]], *, group_keys: List[str], value_keys: List[str]) -> List[Dict[str, Any]]:
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


def _model_key_from_ident(ident: Dict[str, Any]) -> str:
    if ident.get("seed") is not None:
        return f"{ident['model_id']}::seed{ident['seed']}"
    return f"{ident['model_id']}::seedNA"


def _safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def benchmark_single_model(entry: TSEntry, cfg: BenchmarkConfig, run_dir: Path) -> Dict[str, Any]:
    set_eval_determinism()

    devices: List[torch.device] = []
    if cfg.measure_cpu or (not cfg.measure_gpu):
        devices.append(torch.device("cpu"))
    if cfg.measure_gpu and torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    meta = parse_meta(entry.meta_path)
    ident = normalize_ts_metadata(entry.name, entry.ts_path, meta)
    ident["meta_path"] = entry.meta_path
    model_key = _model_key_from_ident(ident)

    print(f"  [child] model={model_key}", flush=True)
    print(f"      ts:   {entry.ts_path}", flush=True)
    if entry.meta_path:
        print(f"      meta: {entry.meta_path}", flush=True)

    test_loader = build_test_loader(cfg)
    per_image_dir = run_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    model_cpu = torch.jit.load(entry.ts_path, map_location="cpu")
    model_cpu.eval()
    params_total = count_params_ts(model_cpu)

    per_image_artifact_path = None
    quality_summary = None
    if cfg.recompute_quality:
        print("      [quality] recomputing on TEST (CPU, deterministic)", flush=True)
        per_image = compute_per_image_on_test(
            model_cpu,
            test_loader,
            device=torch.device("cpu"),
            thr=cfg.threshold,
            include_iou=cfg.per_image_include_iou,
            subgroups=cfg.subgroups,
        )
        quality_summary = per_image["summary"]

        if cfg.compute_per_image_metrics:
            per_image_artifact_path = str(per_image_dir / f"{_safe_id(model_key)}_per_image.npz")
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

    dice_mean = None
    if quality_summary is not None and "overall" in quality_summary:
        dice_mean = float(quality_summary["overall"]["dice_hard05"]["mean"])
    score = composite_score(dice_mean, params_total) if (dice_mean is not None and params_total > 0) else None

    quality_rows: List[Dict[str, Any]] = []
    if quality_summary is not None:
        for subset_key, subset_val in quality_summary.items():
            if not isinstance(subset_val, dict) or "n" not in subset_val:
                continue
            quality_rows.append(
                {
                    "row_type": "quality_summary",
                    "model_id": ident["model_id"],
                    "seed": ident["seed"],
                    "family": ident["family"],
                    "arch": ident["arch"],
                    "encoder": ident["encoder"],
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
            )

    timing_rows: List[Dict[str, Any]] = []
    timing_per_device: Dict[str, Any] = {}
    for dev in devices:
        print(f"      [timing] device={dev} repeats={cfg.repeats} warmup={cfg.warmup} iters={cfg.iters}", flush=True)
        if str(dev) == "cpu":
            model_dev = model_cpu
        else:
            model_dev = torch.jit.load(entry.ts_path, map_location=str(dev))
            model_dev.eval()

        per_repeats: List[Dict[str, Any]] = []
        for r in range(int(cfg.repeats)):
            t = time_inference(
                model_dev,
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
                    "model_id": ident["model_id"],
                    "seed": ident["seed"],
                    "family": ident["family"],
                    "arch": ident["arch"],
                    "encoder": ident["encoder"],
                    "device": str(dev),
                    "repeat_idx": r,
                    "ckpt_path": ident["ckpt_path"],
                    "params_total": params_total,
                    "params_encoder": None,
                    "params_decoder": None,
                    "flops": None,
                    "macs": None,
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
            [{"model_id": ident["model_id"], "seed": ident["seed"], "device": str(dev), **x} for x in per_repeats],
            group_keys=["model_id", "seed", "device"],
            value_keys=["ms_per_img", "fps", "peak_rss_bytes", "peak_vram_alloc_bytes", "peak_vram_reserved_bytes"],
        )
        timing_per_device[str(dev)] = {"per_repeat": per_repeats, "aggregate": agg[0] if agg else None}
        del model_dev
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_record = {
        "model_key": model_key,
        "model_id": ident["model_id"],
        "seed": ident["seed"],
        "family": ident["family"],
        "arch": ident["arch"],
        "encoder": ident["encoder"],
        "encoder_weights": ident["encoder_weights"],
        "ckpt_path": ident["ckpt_path"],
    }

    result_record = {
        "status": "ok",
        "quality": quality_summary,
        "efficiency": {
            "params_total": params_total,
            "params_encoder": None,
            "params_decoder": None,
            "flops": None,
            "macs": None,
            "flops_tool": None,
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
        "state_dict": {"missing": [], "unexpected": []},
    }

    return {
        "entry": {"name": entry.name, "ts_path": entry.ts_path, "meta_path": entry.meta_path},
        "ident": ident,
        "model": model_record,
        "result": result_record,
        "timing_rows": timing_rows,
        "quality_rows": quality_rows,
    }


def infer_failure_status(returncode: int) -> Tuple[str, Optional[str]]:
    if returncode == 0:
        return "ok", None
    if returncode < 0:
        sig = -returncode
        if sig == signal.SIGKILL:
            return "oom_killed", "terminated by SIGKILL"
        return "signal_error", f"terminated by signal {sig}"
    if returncode == 137:
        return "oom_killed", "exit code 137"
    return "runtime_error", f"exit code {returncode}"


def make_failure_result(entry: TSEntry, cfg: BenchmarkConfig, returncode: int, stdout_path: str, stderr_path: str) -> Dict[str, Any]:
    meta = parse_meta(entry.meta_path)
    ident = normalize_ts_metadata(entry.name, entry.ts_path, meta)
    ident["meta_path"] = entry.meta_path
    model_key = _model_key_from_ident(ident)
    status, msg = infer_failure_status(returncode)

    model_record = {
        "model_key": model_key,
        "model_id": ident["model_id"],
        "seed": ident["seed"],
        "family": ident["family"],
        "arch": ident["arch"],
        "encoder": ident["encoder"],
        "encoder_weights": ident["encoder_weights"],
        "ckpt_path": ident["ckpt_path"],
    }
    result_record = {
        "status": status,
        "failure": {
            "return_code": int(returncode),
            "message": msg,
            "stdout_log": stdout_path,
            "stderr_log": stderr_path,
        },
        "quality": None,
        "efficiency": None,
        "composite_score": None,
        "per_image": None,
        "deployment": None,
        "state_dict": {"missing": [], "unexpected": []},
    }
    return {"ident": ident, "model": model_record, "result": result_record, "timing_rows": [], "quality_rows": []}


def run_parent(cfg: BenchmarkConfig) -> Path:
    set_eval_determinism()

    devices: List[torch.device] = []
    if cfg.measure_cpu or (not cfg.measure_gpu):
        devices.append(torch.device("cpu"))
    if cfg.measure_gpu and torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    run_dir = Path(cfg.out_root) / now_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    workers_dir = run_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)

    print("[PHASE 1/6] Writing config + audit to: %s" % str(run_dir), flush=True)
    write_json(str(run_dir / "config.json"), asdict(cfg))

    print("[PHASE 2/6] Loading TorchScript roster", flush=True)
    entries = load_ts_roster(cfg.ts_root)
    print("  -> found %d .ts models" % len(entries), flush=True)

    print("[PHASE 3/6] Parent preflight for TEST benchmark", flush=True)
    preflight_loader = build_test_loader(cfg)
    print("  -> TEST batches: %d  (batch_size=%d, workers=%d)" % (len(preflight_loader), cfg.eval_batch, cfg.num_workers), flush=True)
    del preflight_loader

    master: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "locked_box_test": True,
        "phase": "phase7_test_benchmark",
        "config": desktop_compatible_config(cfg),
        "versions": get_versions(),
        "devices": {str(d): device_info(d) for d in devices},
        "resolution_groups": {
            "PV01": {"gsd_m": 0.1, "size_px": [cfg.input_size, cfg.input_size]},
            "PV03": {"gsd_m": 0.3, "size_px": [cfg.input_size, cfg.input_size]},
            "PV08": {"gsd_m": 0.8, "size_px": [cfg.input_size, cfg.input_size]},
        },
        "models": [],
        "results": {},
        "artifacts": {},
    }

    timing_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    status_jsonl = str(run_dir / "phase7_model_status.jsonl")

    print("[PHASE 4/6] Benchmarking models in isolated child processes", flush=True)
    for mi, e in enumerate(entries, start=1):
        meta = parse_meta(e.meta_path)
        ident = normalize_ts_metadata(e.name, e.ts_path, meta)
        model_key = _model_key_from_ident(ident)
        safe = _safe_id(model_key)
        child_json = str(workers_dir / f"{safe}.result.json")
        stdout_path = str(workers_dir / f"{safe}.stdout.log")
        stderr_path = str(workers_dir / f"{safe}.stderr.log")

        print("  [model %2d/%d] %s" % (mi, len(entries), model_key), flush=True)
        print("      ts:   %s" % e.ts_path, flush=True)
        if e.meta_path:
            print("      meta: %s" % e.meta_path, flush=True)

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--ts-path", e.ts_path,
            "--run-dir", str(run_dir),
            "--result-json", child_json,
        ]
        if e.meta_path:
            cmd.extend(["--meta-path", e.meta_path])
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        start_t = time.time()
        with open(stdout_path, "w", encoding="utf-8") as f_out, open(stderr_path, "w", encoding="utf-8") as f_err:
            proc = subprocess.run(cmd, stdout=f_out, stderr=f_err, env=env)
        elapsed_s = float(time.time() - start_t)

        if proc.returncode == 0 and os.path.isfile(child_json):
            payload = read_json(child_json)
        else:
            payload = make_failure_result(e, cfg, proc.returncode, stdout_path, stderr_path)

        model_record = payload["model"]
        result_record = payload["result"]
        status_value = result_record.get("status", "unknown")

        master["models"].append(model_record)
        master["results"][model_record["model_key"]] = result_record
        timing_rows.extend(payload.get("timing_rows", []))
        quality_rows.extend(payload.get("quality_rows", []))

        status_row = {
            "row_type": "model_status",
            "model_id": model_record.get("model_id"),
            "seed": model_record.get("seed"),
            "family": model_record.get("family"),
            "arch": model_record.get("arch"),
            "encoder": model_record.get("encoder"),
            "ckpt_path": model_record.get("ckpt_path"),
            "ts_path": e.ts_path,
            "meta_path": e.meta_path,
            "status": status_value,
            "return_code": result_record.get("failure", {}).get("return_code"),
            "message": result_record.get("failure", {}).get("message"),
            "elapsed_s": elapsed_s,
            "stdout_log": result_record.get("failure", {}).get("stdout_log", stdout_path),
            "stderr_log": result_record.get("failure", {}).get("stderr_log", stderr_path),
        }
        status_rows.append(status_row)
        append_jsonl(status_jsonl, status_row)

        if status_value == "ok":
            print("      [status] OK", flush=True)
        else:
            print("      [status] %s (%s)" % (status_value, status_row.get("message")), flush=True)

    print("[PHASE 5/6] Writing CSV artifacts", flush=True)
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

    status_csv = str(run_dir / "phase7_model_status.csv")
    save_csv_rows(status_csv, status_rows)

    master["artifacts"]["timing_per_repeat_csv"] = timing_csv
    master["artifacts"]["timing_aggregate_csv"] = timing_agg_csv
    master["artifacts"]["quality_summary_csv"] = quality_csv
    master["artifacts"]["model_status_csv"] = status_csv
    master["artifacts"]["model_status_jsonl"] = status_jsonl

    print("[PHASE 6/6] Writing master JSON report", flush=True)
    master_path = run_dir / "phase7_master_report.json"
    write_json(str(master_path), master)
    print("[ok] wrote master report: %s" % str(master_path), flush=True)
    return master_path


def run_worker(ts_path: str, meta_path: Optional[str], run_dir: str, result_json: str) -> int:
    cfg = CFG_JETSON_TS
    entry = TSEntry(name=Path(ts_path).stem, ts_path=ts_path, meta_path=meta_path)
    payload = benchmark_single_model(entry, cfg, Path(run_dir))
    write_json(result_json, payload)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--worker", action="store_true", help="internal: benchmark a single model in child process")
    p.add_argument("--ts-path", type=str, default=None)
    p.add_argument("--meta-path", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--result-json", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        if not args.ts_path or not args.run_dir or not args.result_json:
            raise SystemExit("--worker requires --ts-path, --run-dir and --result-json")
        raise SystemExit(run_worker(args.ts_path, args.meta_path, args.run_dir, args.result_json))
    run_parent(CFG_JETSON_TS)
