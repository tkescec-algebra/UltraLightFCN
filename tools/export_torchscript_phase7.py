import json
from pathlib import Path
import torch

# --- IMPORTANT: adjust these imports to your repo structure ---
# For ours:
from utils.config import SEG_PARAMS
from models.UltraLightFCN_base import UltraLightFCN

# For SOTA (SMP):
import segmentation_models_pytorch as smp


# ---- EDIT THIS: same 4 final checkpoints used in Phase-7 ----
ROSTER = [
    {
        "name": "ours_ultralightfcn",
        "kind": "ours",
        "ckpt": r"/ABS/PATH/TO/phase6_seed13_LAST.pt",  # or whatever Phase-7 uses
    },
    {
        "name": "dlv3p_resnet50",
        "kind": "smp",
        "arch": "dlv3p",
        "encoder": "resnet50",
        "ckpt": r"/ABS/PATH/TO/dlv3p_resnet50_seed13_LAST.pt",
    },
    {
        "name": "dlv3p_mobilenetv2",
        "kind": "smp",
        "arch": "dlv3p",
        "encoder": "mobilenet_v2",
        "ckpt": r"/ABS/PATH/TO/dlv3p_mobilenetv2_seed13_LAST.pt",
    },
    {
        "name": "unet_resnet34",
        "kind": "smp",
        "arch": "unet",
        "encoder": "resnet34",
        "ckpt": r"/ABS/PATH/TO/unet_resnet34_seed13_LAST.pt",
    },
]

EXPORT_DIR = Path("export_torchscript")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Phase-7 fixed settings
IMAGE_SIZE = 256
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def build_model(entry):
    if entry["kind"] == "ours":
        model = UltraLightFCN(**SEG_PARAMS)
        return model

    if entry["kind"] == "smp":
        if entry["arch"] == "dlv3p":
            return smp.DeepLabV3Plus(
                encoder_name=entry["encoder"],
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        if entry["arch"] == "unet":
            return smp.Unet(
                encoder_name=entry["encoder"],
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
    raise ValueError(f"Unknown model entry: {entry}")


def load_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Support common formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    # If trained with DataParallel etc., you may need key cleanup here
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")


def main():
    device = torch.device("cpu")
    example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    for e in ROSTER:
        name = e["name"]
        print(f"\n=== Export: {name} ===")
        model = build_model(e).to(device)
        load_ckpt(model, e["ckpt"])
        model.eval()

        # Trace is usually safest for deployment
        with torch.no_grad():
            traced = torch.jit.trace(model, example, strict=False)
            traced = torch.jit.freeze(traced)

            out_ts = traced(example)
            out_eager = model(example)

            # quick numeric sanity (not a "tuning", just safety)
            max_abs = (out_ts - out_eager).abs().max().item()
            print(f"[check] max|ts-eager| = {max_abs:.6g}")

        ts_path = EXPORT_DIR / f"{name}.ts"
        traced.save(str(ts_path))

        meta = {
            "name": name,
            "image_size": IMAGE_SIZE,
            "mean": MEAN,
            "std": STD,
            "threshold": 0.5,
            "logits_output": True,
        }
        meta_path = EXPORT_DIR / f"{name}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        print(f"[ok] saved: {ts_path}")
        print(f"[ok] saved: {meta_path}")

    print("\nAll done. Copy 'export_torchscript/' to Jetson.")


if __name__ == "__main__":
    main()
