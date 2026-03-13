# export_torchscript_phase7_torch10.py
# Unified TorchScript exporter for Phase-7:
#   - UltraLightFCN (custom model)
#   - SOTA models (segmentation_models_pytorch: DeepLabV3Plus / Unet variants)
#
# Designed for: Python 3.6 + torch 1.10.x (desktop export),
# to match Jetson Nano runtime constraints (JetPack 4.x).
#
# Usage (inside docker):
#   python tools/export_torchscript_phase7_torch10.py
#   # optionally override rosters:
#   ULTRAFCN_ROSTERS="path1.json,path2.json" python tools/export_torchscript_phase7_torch10.py

import json
import os
from datetime import datetime
from pathlib import Path

import torch

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None  # allow running even if SMP isn't installed (UltraLightFCN-only)

from models.UltraLightFCN_base import UltraLightFCN


# -------------------------
# Phase-7 canonical settings (deployment benchmark)
# -------------------------
INPUT_SIZE = 256
IN_CHANNELS = 3
NUM_CLASSES = 1

# Default rosters (relative to repo root /work)
ROSTERS_DEFAULT = (
    "train/seg_phase6/final_retrain90/phase6_test_report.json",
    "train/seg_sota/stage2_final90_test/phaseSOTA_test_report.json",
)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent  # /work
EXPORT_DIR = THIS_DIR / "export_torchscript_10"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# --- SOTA fine-tuning filter ---
# Default: export only SOTA "minft". If set to "1", include "fullft" too.
SOTA_INCLUDE_FULLFT = os.environ.get("SOTA_INCLUDE_FULLFT", "0").strip() == "1"

# ---- UltraLightFCN default params (must match UltraLightFCN_base.py defaults) ----
DEFAULT_ULTRA_PARAMS = {
    # Encoder
    "enc_channels": [16, 16, 32, 32, 64],
    "enc_kernel_sizes": [3, 3, 3, 3, 3],
    "enc_strides": [1, 2, 2, 1, 1],
    "dilations": [2, 4],
    # Decoder
    "dec_channels": [32, 16, 16],
    "dec_kernel_sizes": [3, 3],
    "dec_strides": [1, 1],
    "upscale": [2, 2],
    # Context
    "mini_aspp": True,
    "mini_aspp_gpool": False,
    # Attention
    "use_sa": True,
    "sa_windowed": True,
    "sa_window_size": 8,
    "sa_shifted": True,
    "sa_heads": 4,
    "sa_dropout": 0.0,
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sanitize_name(s):
    out = []
    for ch in str(s):
        out.append(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_")
    return "".join(out)


def find_checkpoint_paths(obj):
    # Recursively collect ALL strings that look like checkpoint paths (*.pth or *.pt).
    hits = []

    def rec(x):
        if isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                rec(v)
        elif isinstance(x, str):
            s = x.strip().replace("\\", "/")
            if s.endswith(".pth") or s.endswith(".pt"):
                hits.append(s)

    rec(obj)

    # unique preserve order
    out, seen = [], set()
    for h in hits:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out


def resolve_ckpt_path(path_str, roster_path):
    """
    Resolve checkpoint paths from roster JSON to absolute paths.

    Handles common cases:
    - 'train/...' or '/train/...' (repo-relative)  -> /work/train/...
    - paths that contain '.../train/...' anywhere  -> strip before 'train/' and anchor to /work
    - same for 'dataset/'
    - otherwise resolve relative to roster JSON directory
    """
    s = str(path_str).strip()
    norm = s.replace("\\", "/").strip()

    p = Path(norm)
    if p.is_absolute() and p.exists():
        return p

    if norm.startswith("./"):
        norm = norm[2:]
    if norm.startswith("/"):
        norm = norm.lstrip("/")

    i = norm.find("train/")
    if i != -1:
        return (REPO_ROOT / norm[i:]).resolve()

    j = norm.find("dataset/")
    if j != -1:
        return (REPO_ROOT / norm[j:]).resolve()

    return (Path(roster_path).resolve().parent / Path(s)).resolve()


def strip_prefix_if_present(sd, prefix):
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k.replace(prefix, "", 1): v for k, v in sd.items()}
    return sd


def load_checkpoint_any(ckpt_path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a dict")

    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    else:
        if all(isinstance(k, str) for k in ckpt.keys()):
            sd = ckpt
        else:
            raise RuntimeError("Missing state_dict keys")

    sd = strip_prefix_if_present(sd, "module.")
    return ckpt, sd


def classify_arch_hint(ckpt_str):
    s = ckpt_str.replace("\\", "/").lower()
    if "seg_sota" in s:
        if "dlv3p_resnet50" in s:
            return "sota_dlv3p_resnet50"
        if "dlv3p_mobilenetv2" in s:
            return "sota_dlv3p_mobilenetv2"
        if "unet_resnet34" in s:
            return "sota_unet_resnet34"
        return "sota_unknown"
    return "ultra"


# -------------------------
# SOTA export (segmentation_models_pytorch)
# -------------------------
def build_sota_from_hint(hint):
    if smp is None:
        raise RuntimeError("segmentation_models_pytorch is not installed in this environment")

    # activation=None -> raw logits
    if hint == "sota_dlv3p_resnet50":
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
        )
    if hint == "sota_dlv3p_mobilenetv2":
        return smp.DeepLabV3Plus(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
        )
    if hint == "sota_unet_resnet34":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
        )

    raise RuntimeError("unknown SOTA architecture hint: {}".format(hint))


def export_sota_one(ckpt_path, ckpt_str, hint):
    model = build_sota_from_hint(hint)
    ckpt, sd = load_checkpoint_any(ckpt_path)

    # Some training code saves with an extra "model." prefix.
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        sd2 = strip_prefix_if_present(sd, "model.")
        model.load_state_dict(sd2, strict=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    x = torch.randn(1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        ts = torch.jit.trace(model, x, strict=False)
        ts = torch.jit.freeze(ts)

    with torch.no_grad():
        y_e = model(x)
        y_t = ts(x)
        max_abs = (y_t - y_e).abs().max().item()

    name = "SOTA__" + sanitize_name(str(Path(ckpt_path).parent.as_posix()))
    name = name[-180:]
    ts_path = EXPORT_DIR / (name + ".ts")
    meta_path = EXPORT_DIR / (name + ".meta.json")

    ts.save(str(ts_path))
    meta = {
        "kind": "sota",
        "arch_hint": hint,
        "name": name,
        "ts": ts_path.name,
        "meta": meta_path.name,
        "ckpt_path": str(ckpt_path),
        "ckpt_str_in_roster": str(ckpt_str),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "torch_version": torch.__version__,
        "input": {"shape": [1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE]},
        "trace_check": {"max_abs_ts_minus_eager": float(max_abs)},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


# -------------------------
# UltraLightFCN export (custom)
# -------------------------
def export_ultra_one(ckpt_path, ckpt_str):
    ckpt, sd = load_checkpoint_any(ckpt_path)

    params = None
    if isinstance(ckpt, dict):
        params = ckpt.get("params", None)

    merged_params = dict(DEFAULT_ULTRA_PARAMS)
    if isinstance(params, dict):
        merged_params.update(params)

    # Auto-detect MiniASPP gpool from checkpoint weights to avoid shape mismatch.
    k = "mini_aspp.fuse.0.weight"
    if isinstance(sd, dict) and k in sd:
        try:
            fuse_in = int(sd[k].shape[1])
            if fuse_in == 128:
                merged_params["mini_aspp_gpool"] = True
            elif fuse_in == 96:
                merged_params["mini_aspp_gpool"] = False
        except Exception:
            pass

    model = UltraLightFCN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, params=merged_params)

    # strip a couple of common wrappers
    sd = strip_prefix_if_present(sd, "module.")
    sd = strip_prefix_if_present(sd, "model.")

    missing, unexpected = model.load_state_dict(sd, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    x = torch.randn(1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        ts = torch.jit.script(model)
        # ts = torch.jit.trace(model, x, strict=False)
        # ts = torch.jit.freeze(ts)

    with torch.no_grad():
        y_e = model(x)
        y_t = ts(x)
        max_abs = (y_t - y_e).abs().max().item()

    name = "UltraLightFCN__" + sanitize_name(Path(ckpt_path).parent.as_posix())
    name = name[-180:]

    ts_path = EXPORT_DIR / (name + ".ts")
    meta_path = EXPORT_DIR / (name + ".meta.json")

    ts.save(str(ts_path))
    meta = {
        "kind": "ultra",
        "name": name,
        "ts": ts_path.name,
        "meta": meta_path.name,
        "ckpt_path": str(ckpt_path),
        "ckpt_str_in_roster": str(ckpt_str),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "torch_version": torch.__version__,
        "input": {"shape": [1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE]},
        "state_dict_load": {
            "missing": list(missing) if missing is not None else [],
            "unexpected": list(unexpected) if unexpected is not None else [],
        },
        "trace_check": {"max_abs_ts_minus_eager": float(max_abs)},
        "params_in_ckpt": params if isinstance(params, dict) else None,
        "merged_params_used": merged_params,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if (len(meta["state_dict_load"]["missing"]) + len(meta["state_dict_load"]["unexpected"])) > 200:
        try:
            ts_path.unlink()
            meta_path.unlink()
        except Exception:
            pass
        raise RuntimeError("looks incompatible (missing+unexpected too large)")

    return meta


def main():
    env_rosters = os.environ.get("ULTRAFCN_ROSTERS", "").strip()
    if env_rosters:
        roster_paths = [s.strip() for s in env_rosters.split(",") if s.strip()]
    else:
        roster_paths = [str((REPO_ROOT / p).resolve()) for p in ROSTERS_DEFAULT]

    print("[rosters]", roster_paths)

    candidates = []  # list of (ckpt_str, abs_path, arch_hint)
    for rp in roster_paths:
        obj = load_json(rp)
        ckpts = find_checkpoint_paths(obj)
        for ckpt_str in ckpts:
            abs_path = resolve_ckpt_path(ckpt_str, rp)
            hint = classify_arch_hint(ckpt_str)
            # --- Filter: Ultra always included; SOTA default only minft ---
            s = ckpt_str.replace("\\", "/").lower()
            if hint.startswith("sota_"):
                is_minft = "/minft/" in s
                is_fullft = "/fullft/" in s

                if is_fullft and not SOTA_INCLUDE_FULLFT:
                    continue          # skip fullft by default
                if (not is_minft) and (not is_fullft):
                    continue          # if neither minft nor fullft, skip (keeps export clean)
            # --- end filter ---
            candidates.append((ckpt_str, abs_path, hint))

    # unique by abs path preserve order
    uniq, seen = [], set()
    for ckpt_str, p, hint in candidates:
        k = str(p)
        if k not in seen:
            uniq.append((ckpt_str, p, hint))
            seen.add(k)

    print("[found] {} unique checkpoint candidates".format(len(uniq)))

    exported, skipped = [], 0

    for ckpt_str, p, hint in uniq:
        try:
            print("\n[+] exporting", p)
            if hint.startswith("sota_"):
                if hint == "sota_unknown":
                    raise RuntimeError("SOTA checkpoint without known arch keyword in path")
                meta = export_sota_one(p, ckpt_str, hint)
            else:
                meta = export_ultra_one(p, ckpt_str)

            print("    ->", (EXPORT_DIR / meta["ts"]))
            print("    check max|ts-eager| =", meta["trace_check"]["max_abs_ts_minus_eager"])
            exported.append(meta)
        except Exception as e:
            print("    [skip]", p, "(", e, ")")
            skipped += 1

    index = {
        "export_dir": str(EXPORT_DIR),
        "torch_version": torch.__version__,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "count_exported": len(exported),
        "count_skipped": int(skipped),
        "items": exported,
    }
    index_path = EXPORT_DIR / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDone. Exported {} models. Skipped {}.".format(len(exported), skipped))
    print("Wrote {}".format(index_path))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
