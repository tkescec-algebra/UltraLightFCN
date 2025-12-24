import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ColorJitter
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image



def get_transforms(
    mode: str = "train",          # "train" | "valid" | "test"
    image_size: int = 256,
):
    """
    Albumentations transforms for RGB images + binary masks.

    Goals:
    - Fixed spatial size across splits (robust batching + stable eval)
    - Train: moderate geo + photo augs
    - Valid/Test: deterministic (no random aug), only resize/pad + normalization
    - Safe mask handling (NEAREST for masks, fill_mask=0)

    Returns:
        geo_tf, photo_tf, to_tensor
    """
    assert mode in ("train", "valid", "test"), f"Invalid mode: {mode}"

    # Deterministic size policy for all splits.
    # Using LongestMaxSize + PadIfNeeded tends to preserve aspect ratio better than Resize.
    size_tf = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            position="center",
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),
            fill_mask=0,
            p=1.0,
        ),
    ]

    # --- Geometric transforms (image + mask) ---
    if mode == "train":
        geo_tf = A.Compose(
            size_tf + [
                A.HorizontalFlip(p=0.5),

                # Enable only if vertical orientation is not meaningful in your data.
                A.VerticalFlip(p=0.2),

                A.RandomRotate90(p=0.25),

                A.Affine(
                    translate_percent=(-0.05, 0.05),
                    scale=(0.90, 1.10),
                    rotate=(-25, 25),
                    shear=(-5, 5),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(0, 0, 0),
                    fill_mask=0,
                    p=0.5,
                ),

                # Keep distortion mild to avoid unrealistic panel boundaries.
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=(-0.15, 0.15),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(0, 0, 0),
                    fill_mask=0,
                    p=0.10,
                ),
            ]
        )
    else:
        geo_tf = A.Compose(size_tf)

    # --- Photometric transforms + normalization (image only) ---
    if mode == "train":
        photo_tf = A.Compose([
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=0.20),
            A.RandomBrightnessContrast(p=0.40),

            # New API: std_range is relative; keep it mild
            A.GaussNoise(std_range=(0.10, 0.25), mean_range=(0.0, 0.0), per_channel=True, p=0.20),

            # Optional: very mild blur; remove if it hurts boundary quality
            A.GaussianBlur(blur_limit=(3, 3), p=0.05),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ])
    else:
        photo_tf = A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ])

    to_tensor = ToTensorV2(transpose_mask=True)

    return geo_tf, photo_tf, to_tensor

def get_simclr_transforms(image_size=256):
    """
    Transformacije za SimCLR kontrastno učenje.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),

        transforms.RandomApply([
            RGBOnlyColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),

        transforms.RandomGrayscale(p=0.1),

        # Random blur
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
        ], p=0.5),

        transforms.ToTensor()
    ])

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