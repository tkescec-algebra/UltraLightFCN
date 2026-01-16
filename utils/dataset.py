import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pretrain.utils.transforms import get_simclr_transforms
from utils.transforms import get_transforms

# Segmentation dataset for solar panel images + binary masks.
class SolarPanelDataset(Dataset):
    """
    Segmentation dataset for solar panel images + binary masks.

    - Supports .png/.jpg/.jpeg
    - Excludes mask files (*_label.png) from the image list (prevents leakage)
    - Deterministic ordering via sorted(...)
    - Optional preselected list of filenames (files=...), like SimCLR dataset
    - Optional return_extra: returns (img, mask, original, name) when True,
      otherwise returns (img, mask) for faster training/Optuna.
    """

    IMG_EXTS = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",              # "train" | "valid" | "test"
        files=None,                       # optional iterable of filenames (relative to data_dir)
        mask_suffix: str = "_label",      # mask naming: <stem> + mask_suffix + mask_ext
        mask_ext: str = ".png",
        return_extra: bool = False,
    ):
        assert mode in ("train", "valid", "test"), f"Invalid mode: {mode}"
        self.data_dir = data_dir
        self.mode = mode
        self.mask_suffix = mask_suffix
        self.mask_ext = mask_ext
        self.return_extra = return_extra

        # Albumentations-style transforms:
        # geo_tf: joint (image+mask), photo_tf: image-only, to_tensor: tensor conversion/normalization
        self.geo_tf, self.photo_tf, self.to_tensor = get_transforms(mode=self.mode)

        # Build image list (exclude masks)
        mask_tail = f"{self.mask_suffix}{self.mask_ext}".lower()

        if files is not None:
            imgs = [
                f for f in list(files)
                if f.lower().endswith(self.IMG_EXTS) and (not f.lower().endswith(mask_tail))
            ]
        else:
            imgs = [
                f for f in os.listdir(data_dir)
                if f.lower().endswith(self.IMG_EXTS) and (not f.lower().endswith(mask_tail))
            ]

        imgs = sorted(imgs)
        if len(imgs) == 0:
            raise RuntimeError(f"No images found in {data_dir}")

        # Safety: ensure we didn't accidentally include masks
        if any(f.lower().endswith(mask_tail) for f in imgs):
            raise RuntimeError("Mask leakage: mask files ended up in the image list.")

        self.images = imgs

    def __len__(self):
        return len(self.images)

    def _resolve_mask_path(self, img_name: str) -> str:
        """
        Default mask naming: <stem>_label.png
        Example: foo.jpg -> foo_label.png
        """
        stem, _ = os.path.splitext(img_name)
        mask_name = f"{stem}{self.mask_suffix}{self.mask_ext}"
        mask_path = os.path.join(self.data_dir, mask_name)
        if not os.path.isfile(mask_path):
            raise RuntimeError(
                f"Mask not found for image '{img_name}'. Expected '{mask_name}' in: {self.data_dir}"
            )
        return mask_path

    def __getitem__(self, idx: int):
        name = self.images[idx]
        img_path = os.path.join(self.data_dir, name)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_path = self._resolve_mask_path(name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        mask = (mask.astype(np.float32) > 0).astype(np.float32)

        # Joint geometric transforms (must keep image/mask aligned)
        aug = self.geo_tf(image=img, mask=mask)
        img_aug, mask_aug = aug["image"], aug["mask"]

        # Optional original (after geo, before photometric/normalization)
        original = img_aug.copy() if self.return_extra else None

        # Photometric + normalization on image only
        img_photo = self.photo_tf(image=img_aug)["image"]
        img_tensor = self.to_tensor(image=img_photo)["image"].float()

        # Mask to tensor (1, H, W)
        mask_tensor = torch.from_numpy(mask_aug).unsqueeze(0).float()

        if self.return_extra:
            return img_tensor, mask_tensor, original, name
        return img_tensor, mask_tensor


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

        exts = (".png", ".jpg", ".jpeg")

        if files is not None:
            self.images = [
                f for f in list(files)
                if f.lower().endswith(exts) and (not f.lower().endswith("_label.png"))
            ]
        else:
            self.images = [
                f for f in os.listdir(data_dir)
                if f.lower().endswith(exts) and (not f.lower().endswith("_label.png"))
            ]

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {data_dir}")

        if any(f.lower().endswith("_label.png") for f in self.images):
            raise RuntimeError("Mask leakage: *_label.png found in SimCLRSolarPanelDataset input list.")

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

