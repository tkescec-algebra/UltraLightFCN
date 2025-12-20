from dataclasses import dataclass
from typing import Literal

import random
from torchvision import transforms
import torchvision.transforms.functional as TF


AugName = Literal["identity", "color", "blur", "hflip", "vflip", "rotate"]


class IdentityTransform:
    """Pickle-friendly identity transform (Windows DataLoader compatibility)."""
    def __call__(self, x):
        return x


class ForcedRandomRotation:
    """
    Always applies a non-trivial rotation (avoids ~0 deg).
    Makes 'rotate' truly forced-ON for ablation experiments.
    """
    def __init__(self, deg: int, min_abs_deg: float = 1.0):
        self.deg = float(deg)
        self.min_abs_deg = float(min_abs_deg)

    def __call__(self, img):
        angle = random.uniform(-self.deg, self.deg)
        if abs(angle) < self.min_abs_deg:
            angle = self.min_abs_deg if angle >= 0 else -self.min_abs_deg
        return TF.rotate(img, angle)


@dataclass(frozen=True)
class BaseAugCfg:
    # Crop (always on)
    crop_scale_min: float = 0.2

    # Color jitter params (used if "color" selected)
    cj_strength: float = 0.4
    cj_hue: float = 0.04

    # Blur params (used if "blur" selected)
    blur_k: int = 5
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0

    # Rotation params (used if "rotate" selected)
    rot_deg: int = 15
    rot_min_abs_deg: float = 1.0


def _op(name: AugName, cfg: BaseAugCfg, RGBOnlyColorJitter):
    """Return an op that is forced ON when selected."""
    if name == "identity":
        return IdentityTransform()

    if name == "color":
        return RGBOnlyColorJitter(cfg.cj_strength, cfg.cj_strength, cfg.cj_strength, cfg.cj_hue)

    if name == "blur":
        return transforms.GaussianBlur(cfg.blur_k, sigma=(cfg.blur_sigma_min, cfg.blur_sigma_max))

    if name == "hflip":
        return transforms.RandomHorizontalFlip(p=1.0)

    if name == "vflip":
        return transforms.RandomVerticalFlip(p=1.0)

    if name == "rotate":
        return ForcedRandomRotation(cfg.rot_deg, min_abs_deg=cfg.rot_min_abs_deg)

    raise ValueError(f"Unknown augmentation op: {name}")


def build_simclr_pairwise_transforms(image_size: int, cfg: BaseAugCfg, t1: AugName, t2: AugName, RGBOnlyColorJitter):
    """
    Pairwise composition for heatmap:
      ToPIL -> RandomResizedCrop(always) -> T1(forced) -> T2(forced) -> ToTensor
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(cfg.crop_scale_min, 1.0)),
        _op(t1, cfg, RGBOnlyColorJitter),
        _op(t2, cfg, RGBOnlyColorJitter),
        transforms.ToTensor(),
    ])
