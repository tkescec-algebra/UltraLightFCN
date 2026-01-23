import csv
import gc
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Sequence, Callable

import cv2
import numpy as np
import optuna
import torch
import random
import segmentation_models_pytorch as smp

from utils.config import ENCODER_PREFIXES
from utils.edge_detectors import sobel_detector, canny_detector, prewitt_detector, laplacian_detector
from utils.loss_functions import BCEDiceLoss, BCEDiceTverskyLoss, BCEDiceFocalLoss

# Factory function to get the loss function based on the name
def get_loss_function(loss_name='BCEDiceLoss', **kwargs):
    """
    Factory function to get the loss function based on the name.
    """
    if loss_name == 'BCEDiceLoss':
        return BCEDiceLoss(
            pos_weight=kwargs.get('pos_weight', None),
            bce_weight=kwargs.get('bce_weight', 0.5),
            dice_weight=kwargs.get('dice_weight', 0.5)
        )
    elif loss_name == 'BCELoss':
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=kwargs.get('pos_weight', None)
        )
    elif loss_name == 'DiceLoss':
        return smp.losses.DiceLoss(
            mode="binary",
            from_logits=True,
            smooth=1e-6
        )
    elif loss_name == 'TverskyLoss':
        return smp.losses.TverskyLoss(
            mode="binary",
            from_logits=True,
            alpha=kwargs.get('alpha', 0.3),
            beta=kwargs.get('beta', 0.7),
            gamma=kwargs.get('gamma', 1.0),
            smooth=1e-6
        )
    elif loss_name == 'FocalLoss':
        return smp.losses.FocalLoss(
            mode="binary",
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2),
        )
    elif loss_name == 'BCEDiceTverskyLoss':
        return BCEDiceTverskyLoss(
            pos_weight=kwargs.get('pos_weight', None),
            bce_weight=kwargs.get('bce_weight', 0.5),
            dice_weight=kwargs.get('dice_weight', 0.3),
            tversky_weight=kwargs.get('tversky_weight', 0.2),
            alpha=kwargs.get('alpha', 0.3),
            beta=kwargs.get('beta', 0.7),
        )

    elif loss_name == 'BCEDiceFocalLoss':
        return BCEDiceFocalLoss(
            pos_weight=kwargs.get('pos_weight', None),
            bce_weight=kwargs.get('bce_weight', 0.4),
            dice_weight=kwargs.get('dice_weight', 0.3),
            focal_weight=kwargs.get('focal_weight', 0.3),
            alpha_focal=kwargs.get('alpha_focal', 0.25),
            gamma_focal=kwargs.get('gamma_focal', 2.0),
        )
    else:
        raise ValueError(f"Loss function {loss_name} not supported")

# Factory function to get the model based on the name
def get_model(model_name='base'):
    """
    Factory function to get the model based on the name.
    """
    if model_name == 'base':
        from models.UltraLightFCN_base import UltraLightFCN
        return UltraLightFCN
    # elif model_name == 'se':
    #     from test_models.UltraLightFCN_SE_model import UltraLightFCN_SE
    #     return UltraLightFCN_SE
    # elif model_name == 'cbam':
    #     from test_models.UltraLightFCN_CBAM_model import UltraLightFCN_CBAM
    #     return UltraLightFCN_CBAM
    elif model_name == 'dlv3p':
        return smp.DeepLabV3Plus
    elif model_name == 'unet':
        return smp.Unet
    else:
        raise ValueError(f"Model {model_name} not supported")

# Function to clear CUDA cache and collect garbage
def clear_cuda_cache(*args, **kwargs):
    """
    Clear CUDA cache + run GC.

    Works both as:
      - Optuna callback: clear_cuda_cache(study, trial)
      - Direct call: clear_cuda_cache()
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Callback to save the best trial's state_dict
def save_best_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    trial.user_attrs["state_dict"] = trial.user_attrs.get("state_dict", None)
    if study.best_trial.number == trial.number:
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(
            trial.user_attrs["state_dict"],
            f"checkpoints/{study.study_name}-best_trial{trial.number}.pth"
        )
        print(f"[ckpt] Saved the weight of the best trail in checkpoints/{study.study_name}-best_trial{trial.number}.pth")

# Function to infer subset from filename
def infer_subset_from_filename(name: str) -> str:
    m = re.compile(r"^(PV\d{2})[-_]").match(name)
    return m.group(1) if m else "OTHER"

# Function to make a reduced file list based on custom policy
def make_reduced_file_list(
    data_dir: str,
    max_total: Optional[int] = 5000,
    seed: int = 42,
    keep_all_pv01: bool = True,
    balance_subsets: Tuple[str, str] = ("PV03", "PV08"),
) -> List[str]:
    """
    Deterministically select a reduced set of filenames from data_dir with a custom policy:

    Policy:
      1) Keep all PV01 files (if keep_all_pv01=True).
      2) Fill the remaining budget by sampling equally from PV03 and PV08 (balance_subsets).
      3) OTHER subsets are ignored by default (can be added later if needed).

    Notes:
      - If max_total is None: returns all files (no reduction), but still shuffles deterministically.
      - If PV01 count exceeds max_total and keep_all_pv01=True, we keep PV01 only up to max_total
        (otherwise it is impossible to satisfy max_total).

    Returns:
      List[str]: selected filenames (shuffled deterministically).
    """
    exts = (".png", ".jpg", ".jpeg")
    all_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(exts) and (not f.lower().endswith("_label.png"))
    ]
    if not all_files:
        return []

    rng = random.Random(seed)

    # Bucket files by subset prefix (uses your existing helper)
    buckets = {}
    for f in all_files:
        subset = infer_subset_from_filename(f)
        buckets.setdefault(subset, []).append(f)

    # Shuffle each bucket deterministically
    for subset_files in buckets.values():
        rng.shuffle(subset_files)

    pv01_files = buckets.get("PV01", [])
    a_files = buckets.get(balance_subsets[0], [])
    b_files = buckets.get(balance_subsets[1], [])

    # If no max_total -> return everything (but deterministic ordering)
    if max_total is None:
        selected = list(all_files)
        rng.shuffle(selected)
        return selected

    selected: List[str] = []

    # 1) Keep all PV01 (or as many as possible if PV01 > max_total)
    if keep_all_pv01:
        if len(pv01_files) >= max_total:
            # Can't keep all PV01 and still respect max_total; keep first max_total deterministically
            selected = pv01_files[:max_total]
            rng.shuffle(selected)
            return selected
        selected.extend(pv01_files)
    else:
        # If you ever want to sample PV01 too, you'd implement that here.
        pass

    remaining = max_total - len(selected)
    if remaining <= 0:
        rng.shuffle(selected)
        return selected

    # 2) Split remaining budget equally between PV03 and PV08
    half = remaining // 2
    rest = remaining - 2 * half  # remainder 0 or 1

    take_a = min(half + rest, len(a_files))  # give remainder to the first subset
    take_b = min(half, len(b_files))

    selected.extend(a_files[:take_a])
    selected.extend(b_files[:take_b])

    # 3) If one subset doesn't have enough files, top-up from the other
    shortfall = remaining - (take_a + take_b)
    if shortfall > 0:
        # Try to fill from whichever still has capacity
        a_left = a_files[take_a:]
        b_left = b_files[take_b:]
        topup_pool = a_left + b_left
        # Already shuffled (bucket shuffle), but for safety:
        rng.shuffle(topup_pool)
        selected.extend(topup_pool[:shortfall])

    # Final deterministic shuffle (so the loader sees a mixed sequence)
    rng.shuffle(selected)

    # Final hard safety
    if any(f.lower().endswith("_label.png") for f in selected):
        raise RuntimeError("Mask leakage: *_label.png present in reduced file list.")

    return selected

# Function to estimate pos_weight from masks
def estimate_pos_weight_from_masks(
    train_dir: str,
    max_images: int = 500,
    seed: int = 13,
) -> float | None:
    """
    Estimate pos_weight for BCEWithLogitsLoss as neg_pixels / pos_pixels.
    Uses a fixed random subset of masks for speed. Returns None if no positives found.
    """
    rng = np.random.default_rng(seed)

    all_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    # Keep only image files (exclude label files); assumes label name contains "_label"
    image_files = [f for f in all_files if "_label" not in f]

    if len(image_files) == 0:
        return None

    # Deterministic subset
    if len(image_files) > max_images:
        idx = rng.choice(len(image_files), size=max_images, replace=False)
        image_files = [image_files[i] for i in idx]
    else:
        image_files = sorted(image_files)

    pos = 0
    neg = 0

    for img_name in image_files:
        base, ext = os.path.splitext(img_name)
        mask_path = os.path.join(train_dir, f"{base}_label.png")  # adjust if your mask ext differs

        if not os.path.exists(mask_path):
            continue

        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        # Assume masks are 0/255 or 0/1; binarize
        m_bin = (m > 0).astype(np.uint8)
        pos += int(m_bin.sum())
        neg += int(m_bin.size - m_bin.sum())

    if pos == 0:
        return None

    return float(neg / pos)

# Function to calculate steps per epoch
def steps_per_epoch(n_items: int, batch_size: int, drop_last: bool) -> int:
    """Number of optimizer updates per epoch (iteration-based scheduler).

    - drop_last=True  -> floor(n / bs)
    - drop_last=False -> ceil(n / bs) via integer math
    - returns 0 only when n_items <= 0
    - never returns 0 when n_items > 0 (robust for schedulers)
    """
    if n_items <= 0:
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if drop_last:
        return max(1, n_items // batch_size)
    return max(1, (n_items + batch_size - 1) // batch_size)

# Function to split model parameters into encoder and decoder stacks
def split_encoder_decoder_params(model: torch.nn.Module):
    """Split params into (encoder_stack, decoder_stack).

    Encoder stack includes:
      - backbone blocks (block1, dsconv2, dsconv3, dilconv4, dilconv5)
      - mini_aspp
      - sa
    """

    enc_params: List[torch.nn.Parameter] = []
    dec_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if name.startswith(ENCODER_PREFIXES):
            enc_params.append(p)
        else:
            dec_params.append(p)
    return enc_params, dec_params

# Functions to save JSON
def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# Functions to save CSV rows
def save_csv_rows(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

# Function to build loss from parameters
def build_loss_from_params(
    params: Dict[str, Any],
    get_loss_fn: Callable[..., Any],
):
    """
    Reconstruct Phase-4 loss exactly from candidate params (used in Phase-5 and Phase-6).
    """
    loss_name = params["loss"]

    if loss_name == "BCEDiceLoss":
        bce_w = float(params["bce_w"])
        dice_w = 1.0 - bce_w
        return loss_name, get_loss_fn("BCEDiceLoss", bce_weight=bce_w, dice_weight=dice_w)

    if loss_name == "BCEDiceFocalLoss":
        bce_w = float(params["bce_w"])
        dice_w = float(params["dice_w"])
        focal_w = 1.0 - (bce_w + dice_w)
        if focal_w <= 0:
            raise ValueError("Invalid loss weights: focal_w <= 0 (check candidate params).")

        alpha = float(params["alpha_focal"])
        gamma = float(params["gamma_focal"])
        return loss_name, get_loss_fn(
            "BCEDiceFocalLoss",
            bce_weight=bce_w,
            dice_weight=dice_w,
            focal_weight=focal_w,
            alpha_focal=alpha,
            gamma_focal=gamma,
        )

    raise ValueError(f"Unsupported loss in candidate params: {loss_name}")