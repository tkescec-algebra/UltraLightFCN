import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_transforms, get_simclr_transforms

# Basic dataset for solar panel images
class SolarPanelDataset(Dataset):
    """
    Dataset for solar panel images (RGB only, 3 channels).

    Arguments:
        data_dir (str): directory with JPG images and corresponding labels (_label.png).
        mode (str): "train", "val", or "test" for different transformations.
    """
    def __init__(
        self,
        data_dir: str,
        mode: str = "train"  # "train" | "valid" | "test"
    ):
        assert mode in ("train", "valid", "test")
        self.data_dir = data_dir
        self.mode = mode

        # We don't use edge transforms anymore
        self.geo_tf, self.photo_tf, self.to_tensor = get_transforms(
            mode=self.mode,
            edge_transforms=False
        )

        self.images = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        # --- load image and mask ---
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.data_dir, name)),
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            os.path.join(self.data_dir, name.replace(".jpg", "_label.png")),
            cv2.IMREAD_GRAYSCALE
        ).astype(np.float32)

        # Normalize mask to {0,1}
        mask = (mask > 0).astype(np.float32)

        # --- apply geometric transforms ---
        aug = self.geo_tf(image=img, mask=mask)
        img_aug, mask = aug["image"], aug["mask"]

        # --- save "original" for visualization (before normalization) ---
        original = img_aug.copy()

        # --- photometric + normalize on RGB ---
        img_photometric = self.photo_tf(image=img_aug)["image"]

        # --- convert to tensors ---
        img_tensor = self.to_tensor(image=img_photometric)["image"]  # (3, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)            # (1, H, W)

        # --- return input, mask, original and name ---
        return img_tensor.float(), mask_tensor.float(), original, name

# Dataset for SimCLR contrastive learning on solar panel images.
class SimCLRSolarPanelDataset(Dataset):
    """
    RGB-only SimCLR dataset.
    Returns two augmented views (Tensor CxHxW) and the filename.

    - Supports .png/.jpg/.jpeg
    - Can operate on a preselected list of filenames (files=...)
    """
    def __init__(self, data_dir: str, image_size: int = 256, transform=None, files=None):
        self.data_dir = data_dir
        self.simclr_tf = transform if transform is not None else get_simclr_transforms(image_size)

        if files is not None:
            self.images = list(files)
        else:
            exts = (".png", ".jpg", ".jpeg")
            self.images = [f for f in os.listdir(data_dir) if f.lower().endswith(exts)]

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.data_dir, name)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        view1 = self.simclr_tf(img)
        view2 = self.simclr_tf(img)

        return view1.float(), view2.float(), name

