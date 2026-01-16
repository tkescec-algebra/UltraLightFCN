import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

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

def get_transforms(
    mode: str = "train",          # "train" | "valid" | "test"
    image_size: int = 256,
):
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