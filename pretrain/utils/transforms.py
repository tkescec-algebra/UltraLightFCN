from torchvision.transforms import ColorJitter
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image
from dataclasses import dataclass
from typing import Tuple

"""
SimCLR-style augmentations for pretraining.
"""
# Applying ColorJitter only to the RGB channels of an image
class RGBOnlyColorJitter:
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img):
        # 1) osigurajte PIL input
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)

        # 2) splitajte na numpy array
        arr = np.array(img)  # shape (H, W, C)
        rgb, rest = arr[:, :, :3], arr[:, :, 3:] if arr.shape[2] > 3 else None

        # 3) jitter na RGB
        rgb_j = self.jitter(Image.fromarray(rgb))

        # 4) spojite natrag
        rgb_j_arr = np.array(rgb_j)
        if rest is not None:
            combined = np.concatenate([rgb_j_arr, rest], axis=2)
        else:
            combined = rgb_j_arr

        # 5) vratite PIL image
        return Image.fromarray(combined.astype(np.uint8))

# Configuration dataclass for SimCLR augmentations
@dataclass(frozen=True)
class SimCLRAugConfig:
    image_size: int = 256

    # crop
    use_crop: bool = True
    crop_scale: Tuple[float, float] = (0.4, 1.0)

    # flips
    p_hflip: float = 0.5
    p_vflip: float = 0.5

    # rotation
    rot_deg: float = 10.0

    # color jitter (applied with RandomApply)
    p_jitter: float = 0.8
    jitter_b: float = 0.4
    jitter_c: float = 0.4
    jitter_s: float = 0.4
    jitter_h: float = 0.1

    # grayscale
    p_gray: float = 0.1

    # blur
    p_blur: float = 0.5
    blur_kernel: int = 3
    blur_sigma: Tuple[float, float] = (0.1, 0.8)

# Function to build SimCLR augmentation transforms based on the config
def build_simclr_transforms(cfg: SimCLRAugConfig):
    t = [transforms.ToPILImage()]

    if cfg.use_crop:
        t.append(transforms.RandomResizedCrop(cfg.image_size, scale=cfg.crop_scale))
    else:
        # "no-crop" baseline: keep content, remove crop randomness
        t += [transforms.Resize(cfg.image_size), transforms.CenterCrop(cfg.image_size)]

    if cfg.p_hflip > 0:
        t.append(transforms.RandomHorizontalFlip(p=cfg.p_hflip))
    if cfg.p_vflip > 0:
        t.append(transforms.RandomVerticalFlip(p=cfg.p_vflip))
    if cfg.rot_deg and cfg.rot_deg > 0:
        t.append(transforms.RandomRotation(cfg.rot_deg))

    if cfg.p_jitter > 0:
        t.append(transforms.RandomApply([
            RGBOnlyColorJitter(cfg.jitter_b, cfg.jitter_c, cfg.jitter_s, cfg.jitter_h)
        ], p=cfg.p_jitter))

    if cfg.p_gray > 0:
        t.append(transforms.RandomGrayscale(p=cfg.p_gray))

    if cfg.p_blur > 0:
        t.append(transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=cfg.blur_kernel, sigma=cfg.blur_sigma)
        ], p=cfg.p_blur))

    t.append(transforms.ToTensor())
    return transforms.Compose(t)

# Convenience function to get default SimCLR transforms
def get_simclr_transforms(image_size=256):
    return build_simclr_transforms(SimCLRAugConfig(image_size=image_size))