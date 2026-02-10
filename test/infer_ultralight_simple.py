#!/usr/bin/env python
import os
from typing import Tuple, Optional, List, Union

import numpy as np
import cv2
from PIL import Image

import torch
from torch import nn

from models.UltraLightFCN_base import UltraLightFCN
from utils.helpers import get_edge_detector
# ⚠️ Adjust this import to match your project structure
from utils.transforms import get_transforms
import segmentation_models_pytorch as smp


# =======================
#         CONSTANTS
# =======================

# Put one or more image paths here
IMAGE_PATHS: Union[str, List[str]] = [
    "../dataset/test/PV03-PV03_Rooftop-PV03_337371_1202234_r0_c0.jpg",
    "../dataset/test/PV03-PV03_Rooftop-PV03_321944_1198644_r2_c1.jpg",
    "../dataset/test/PV08-PV08_Rooftop-PV08_322077_1196795_r1_c3.jpg",
    "../dataset/test/PV01-PV01_Rooftop_Brick-PV01_325472_1204667.jpg",
    "../dataset/test/PV03-PV03_Ground_Cropland-PV03_318181_1207703_r2_c3.jpg",
    "../dataset/test/PV03-PV03_Rooftop-PV03_337371_1202234_r2_c0.jpg",
]

# Path to the trained model checkpoint (.pth) you want to use
# CHECKPOINT_PATH = (
#     "../train/sota_outputs/train_models/unet_efficientnet-b0/seed_62/unet_efficientnet-b0(rgb)-seed_62.pth"
# )

CHECKPOINT_PATH = (
    "../train/ablation_outputs/finetune/train_models/baseline/seed_52/UltraLightFCN_base(rgb)-seed_52-finetune.pth"
)

# Variant name (must match how the model was trained)
VARIANT_NAME = "baseline"  # "baseline", "no_sa", "mini_aspp_off", ...

MODEL_TYPE = "ultralight"      # "ultralight" ili "unet_effb0"

# Number of input channels:
#   3 = RGB
#   4 = RGB + edge (must also set EDGE_DETECTOR to the correct name)
CHANNELS = 3

# If you trained with 4 channels (RGB+edge), set this to the same edge detector name as in training.
EDGE_DETECTOR: Optional[str] = None  # or "sobel", "canny", ...

# Threshold for binary mask (0–1).
THRESHOLD = 0.65

# Where to save the output masks
OUT_DIR = "./pred_masks"


# =======================
#      GLOBAL CONFIG
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANTS = {
    "baseline":             dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=16),
    "mini_aspp_off":        dict(mini_aspp=False, mini_aspp_gpool=False, use_sa=True,  sa_window_size=16),
    "no_sa":                dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=False, sa_window_size=16),
    "mini_aspp_no_gpool":   dict(mini_aspp=True,  mini_aspp_gpool=False, use_sa=True,  sa_window_size=16),
    "sa_ws_8":              dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=8),
    "sa_ws_32":             dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=32),
    "loss_03_07":           dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=16),
    "loss_05_05":           dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=16),
}


# =======================
#   MODEL PARAM HELPERS
# =======================
def build_ultralight_params(variant_cfg: dict) -> dict:
    """Build the parameter dictionary for UltraLightFCN (same as in ablation)."""
    return {
        # Encoder configuration
        'enc_channels':     [16, 16, 32, 32, 64],
        'enc_kernel_sizes': [3,  3,  3,  3,  3],
        'enc_strides':      [1,  2,  2,  1,  1],
        'dilations':        [2,  4],

        # Decoder configuration
        'dec_channels':     [32, 16, 16],
        'dec_kernel_sizes': [3,  3],
        'dec_strides':      [1,  1],
        'upscale':          [2,  2],

        # Context / Self-attention configuration
        'mini_aspp':        variant_cfg['mini_aspp'],
        'mini_aspp_gpool':  variant_cfg['mini_aspp_gpool'],
        'use_sa':           variant_cfg['use_sa'],
        'sa_windowed':      True,
        'sa_window_size':   variant_cfg['sa_window_size'],
        'sa_shifted':       True,
        'sa_heads':         4,
        'sa_dropout':       0.0,
    }


# =======================
#   IMAGE PREPROCESSING
# =======================
def preprocess_image_with_dataset_transforms(
    img_path: str,
    channels: int = 3,
    edge_detector: Optional[str] = None,
    mode: str = "test",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess a single image using the SAME pipeline as SolarPanelDataset.
    """
    assert mode in ("train", "val", "test")

    # Load image (BGR -> RGB)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # H x W x 3
    orig_h, orig_w = img.shape[:2]
    orig_size = (orig_w, orig_h)

    # Dummy mask so geo_tf can run; we ignore it afterwards
    dummy_mask = np.zeros((orig_h, orig_w), dtype=np.float32)

    # Decide whether to use edge channel
    use_edge = (channels == 4 and edge_detector is not None)

    # Get same transforms as in SolarPanelDataset
    geo_tf, photo_tf, to_tensor = get_transforms(
        mode=mode,
        edge_transforms=use_edge
    )

    # Compute edge if needed
    if use_edge:
        edge_fn = get_edge_detector(edge_detector)
        edge = edge_fn(img).astype(np.float32) / 255.0
        aug = geo_tf(image=img, mask=dummy_mask, edge=edge)
        img_aug = aug["image"]
        edge_aug = aug["edge"]
    else:
        edge_aug = None
        aug = geo_tf(image=img, mask=dummy_mask)
        img_aug = aug["image"]

    # Photometric transforms + normalization (same as dataset)
    img_photometric = photo_tf(image=img_aug)["image"]

    # To tensor: (3,H,W)
    img_tensor = to_tensor(image=img_photometric)["image"]

    # If we have edge, convert to tensor and concatenate to RGB channels
    if use_edge and edge_aug is not None:
        edge_tensor = torch.from_numpy(edge_aug).unsqueeze(0)  # (1,H,W)
        input_tensor = torch.cat([img_tensor, edge_tensor], dim=0)  # (4,H,W)
    else:
        input_tensor = img_tensor

    return input_tensor.float().unsqueeze(0), orig_size  # (1,C,H,W), (W,H)


# =======================
#   MASK POSTPROCESSING
# =======================
def postprocess_mask(
    prob_mask: np.ndarray,
    orig_size: Tuple[int, int],
    thr: float = 0.5,
) -> Tuple[Image.Image, Image.Image]:
    """
    Convert probability mask to:
    - grayscale probability image (0-255)
    - binary mask (0 or 255) using a threshold

    Then resize both back to original image resolution.
    """
    # prob_mask is H x W in [0,1]
    prob_uint8 = (prob_mask * 255.0).clip(0, 255).astype(np.uint8)
    bin_uint8 = (prob_mask >= thr).astype(np.uint8) * 255

    prob_img = Image.fromarray(prob_uint8)
    bin_img = Image.fromarray(bin_uint8)

    # Resize to original (W, H)
    prob_img = prob_img.resize(orig_size, resample=Image.BILINEAR)
    bin_img = bin_img.resize(orig_size, resample=Image.NEAREST)

    return prob_img, bin_img


# =======================
#    MODEL LOADING
# =======================
def load_model(ckpt_path: str,
               model_type: str = MODEL_TYPE,
               channels: int = 3) -> nn.Module:
    """
    Build and load a model depending on model_type.

    model_type:
      - "ultralight": UltraLightFCN (your custom model)
      - "unet_effb0": U-Net with EfficientNet-B0 encoder (SOTA baseline)
    """
    if model_type == "ultralight":
        variant_name = VARIANT_NAME
        if variant_name not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'.")

        params = build_ultralight_params(VARIANTS[variant_name])
        model = UltraLightFCN(in_channels=channels, num_classes=1, params=params)

    elif model_type == "unet_effb0":
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=channels,
            classes=1,
        )

    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)

    model.to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


# =======================
#       INFERENCE
# =======================
def predict_mask_for_image(
    image_path: str,
    model: nn.Module,
    channels: int = 3,
    thr: float = 0.5,
    out_dir: Optional[str] = None,
    edge_detector: Optional[str] = None,
):
    """
    Run inference for a single image, save probability and binary masks as PNGs.
    Uses the SAME transforms as SolarPanelDataset (mode='test').

    NOTE: model is passed in so we don't reload it for every image.
    """
    os.makedirs(out_dir or ".", exist_ok=True)

    # Preprocess image using dataset-like transforms
    image_tensor, orig_size = preprocess_image_with_dataset_transforms(
        img_path=image_path,
        channels=channels,
        edge_detector=edge_detector,
        mode="test",
    )
    image_tensor = image_tensor.to(DEVICE)
    if DEVICE.type == "cuda":
        image_tensor = image_tensor.to(memory_format=torch.channels_last)

    # Forward pass
    with torch.no_grad():
        logits = model(image_tensor)  # [1, 1, H, W] or [1, H, W]
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)

        probs = torch.sigmoid(logits)  # [1, 1, H, W]
        prob_mask = probs[0, 0].cpu().numpy()

    # Postprocess mask
    prob_img, bin_img = postprocess_mask(prob_mask, orig_size, thr=thr)

    # Output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = out_dir or os.path.dirname(image_path)
    prob_path = os.path.join(out_dir, f"{base_name}_prob.png")
    bin_path = os.path.join(out_dir, f"{base_name}_mask_thr{thr:.2f}.png")

    prob_img.save(prob_path)
    bin_img.save(bin_path)

    print(f"Saved probability mask to: {prob_path}")
    print(f"Saved binary mask to:      {bin_path}")


# =======================
#         MAIN
# =======================
if __name__ == "__main__":
    # Load model once
    model = load_model(
        ckpt_path=CHECKPOINT_PATH,
        model_type=MODEL_TYPE,
        channels=CHANNELS,
    )

    # Normalize IMAGE_PATHS to a list
    if isinstance(IMAGE_PATHS, str):
        image_paths = [IMAGE_PATHS]
    else:
        image_paths = IMAGE_PATHS

    # Run inference for each image
    for img_path in image_paths:
        print(f"\nProcessing image: {img_path}")
        predict_mask_for_image(
            image_path=img_path,
            model=model,
            channels=CHANNELS,
            thr=THRESHOLD,
            out_dir=OUT_DIR,
            edge_detector=EDGE_DETECTOR,
        )
