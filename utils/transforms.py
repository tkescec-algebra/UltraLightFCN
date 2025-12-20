import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ColorJitter
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image



def get_transforms(
    mode: str = "train"  # "train" | "valid" | "test"
):
    """
    Returns Albumentations transforms for RGB images + binary masks (no edge channel).

    Args:
        mode (str): "train", "valid", or "test"
            - "train": geometric + photometric augmentations + normalization
            - "valid": no geometric aug, only normalization
            - "test":  no geometric aug, only normalization

    Returns:
        geo_tf: Albumentations.Compose for geometric transforms (applied to image + mask)
        photo_tf: Albumentations.Compose for photometric transforms + normalization (image only)
        to_tensor: ToTensorV2 transform (converts image/mask to torch tensors)
    """
    assert mode in ("train", "valid", "test")

    # We only need to keep the mask aligned with the image during geometric transforms
    additional = {"mask": "mask"}

    # 1) Geometric transforms (image + mask)
    if mode == "train":
        geo_tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-25, 25),
                p=0.5
            ),
            A.GridDistortion(p=0.2),
        ], additional_targets=additional)
    else:
        # For validation/test: no geometric augmentations
        geo_tf = A.Compose([], additional_targets=additional)

    # 2) Photometric transforms + normalization (image only)
    if mode == "train":
        photo_tf = A.Compose([
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(p=0.3),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
    else:
        # For validation/test: only normalization
        photo_tf = A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    # 3) Convert to torch tensors (deterministic)
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