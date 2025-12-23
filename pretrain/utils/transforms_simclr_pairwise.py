"""
SimCLR-faithful pairwise augmentation policies for Phase1 heatmap.

Key idea:
- For each heatmap cell (t1, t2) we define ONE augmentation distribution (policy).
- Both SimCLR views are sampled independently from the SAME policy
  (the dataset calls transform(img) twice).
- The policy is BASE (always-on crop) + a restricted set of allowed ops {t1, t2},
  applied with the SAME probabilities/strengths as used in HPO.

This avoids the methodological pitfall of "view1 always gets T1 and view2 always gets T2",
which is not the original SimCLR formulation (Chen et al., 2020).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Set

from torchvision import transforms


AugName = Literal["identity", "color", "gray", "blur", "hflip", "vflip", "rotate"]


@dataclass(frozen=True)
class BaseAugCfg:
    """
    Base configuration for building pairwise SimCLR policies.

    IMPORTANT:
    - Defaults are aligned with HPO transform settings:
      RandomResizedCrop scale=(0.4, 1.0)
      RandomRotation(10)
      ColorJitter strength 0.4 / hue 0.1 and RandomApply p=0.8
      RandomGrayscale p=0.1
      GaussianBlur k=3 sigma=(0.1,0.8) with RandomApply p=0.5
      H/V flip p=0.5
    """
    # Crop (always-on)
    crop_scale_min: float = 0.4

    # Rotation (used if "rotate" allowed)
    rot_deg: int = 10

    # Color jitter (used if "color" allowed)
    cj_strength: float = 0.4
    cj_hue: float = 0.1
    p_color: float = 0.8

    # Grayscale (used if "gray" allowed)
    p_gray: float = 0.1

    # Blur (used if "blur" allowed)
    blur_k: int = 3
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 0.8
    p_blur: float = 0.5

    # Flips (used if allowed)
    p_hflip: float = 0.5
    p_vflip: float = 0.5


def _canonical_pair(t1: AugName, t2: AugName) -> tuple[AugName, AugName]:
    """
    Canonicalize pair order to avoid duplicate heatmap entries:
    (color, blur) == (blur, color)
    """
    return tuple(sorted((t1, t2)))  # type: ignore[return-value]


def build_simclr_pairwise_transforms(
    image_size: int,
    cfg: BaseAugCfg,
    t1: AugName,
    t2: AugName,
    RGBOnlyColorJitter,
):
    """
    Build a SimCLR-faithful augmentation policy for Phase1 heatmap.

    The returned transform is called twice per sample by SimCLRSolarPanelDataset:
      view1 = tf(img)
      view2 = tf(img)

    Therefore, each view is an independent stochastic draw from the same policy.

    Policy structure:
      ToPIL
      RandomResizedCrop (always on)
      + only those ops that are allowed by {t1, t2} (excluding "identity")
        with HPO-matched probabilities/strengths
      ToTensor
    """
    t1, t2 = _canonical_pair(t1, t2)

    allowed: Set[AugName] = {t1, t2}
    allowed.discard("identity")  # "identity" means no extra ops beyond BASE

    ops = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(cfg.crop_scale_min, 1.0)),
    ]

    # -------------------------
    # Geometry (HPO-aligned)
    # -------------------------
    if "hflip" in allowed:
        ops.append(transforms.RandomHorizontalFlip(p=cfg.p_hflip))

    if "vflip" in allowed:
        ops.append(transforms.RandomVerticalFlip(p=cfg.p_vflip))

    if "rotate" in allowed:
        ops.append(transforms.RandomRotation(cfg.rot_deg))

    # -------------------------
    # Photometric (HPO-aligned)
    # -------------------------
    if "color" in allowed:
        ops.append(
            transforms.RandomApply(
                [RGBOnlyColorJitter(cfg.cj_strength, cfg.cj_strength, cfg.cj_strength, cfg.cj_hue)],
                p=cfg.p_color,
            )
        )

    if "gray" in allowed:
        ops.append(transforms.RandomGrayscale(p=cfg.p_gray))

    if "blur" in allowed:
        ops.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=cfg.blur_k, sigma=(cfg.blur_sigma_min, cfg.blur_sigma_max))],
                p=cfg.p_blur,
            )
        )

    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)
