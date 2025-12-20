import os
import cv2
import torch
from torch.utils.data import Dataset


class UMAPSolarPanelEvalDataset(Dataset):
    """
    Returns (image_tensor, has_panel_label, name).
    Mask naming:
      <stem>.jpg -> <stem>_label.png
    If mask_dir is None, masks are searched in img_dir.
    """
    def __init__(self, img_dir: str, transform, mask_dir: str | None = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir if mask_dir is not None else img_dir
        self.transform = transform

        self.images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.img_dir, name)

        stem = name[:-4]
        mask_name = f"{stem}_label.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        has_panel = 1 if (mask is not None and mask.max() > 0) else 0

        x = self.transform(img).float()
        return x, torch.tensor(has_panel, dtype=torch.long), name
