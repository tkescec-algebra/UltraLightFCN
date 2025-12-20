import torch
from typing import Tuple

def calculate_iou(pred_logits: torch.Tensor,
                  mask: torch.Tensor,
                  thr: float = None,
                  smooth: float = 1e-6) -> float:
    """
    Jaccard IoU per-image, batch-averaged.
    Ako je thr!=None: diskretni IoU nakon thresholda,
    inače: 'soft' IoU (sigmoid output).
    """
    pred = torch.sigmoid(pred_logits)
    if thr is not None:
        pred = (pred > thr).float()

    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.float()

    B = pred.shape[0]
    p = pred.view(B, -1)
    g = mask.view(B, -1)

    inter = (p * g).sum(dim=1)
    union = p.sum(dim=1) + g.sum(dim=1) - inter

    iou_per_image = (inter + smooth) / (union + smooth)
    return iou_per_image.mean().item()


def calculate_dice(pred_logits: torch.Tensor,
                   mask: torch.Tensor,
                   thr: float = None,
                   smooth: float = 1e-6) -> float:
    """
    Dice (F1) per-image, batch-averaged.
    Ako je thr!=None: diskretni Dice,
    inače: 'soft' Dice.
    """
    pred = torch.sigmoid(pred_logits)
    if thr is not None:
        pred = (pred > thr).float()

    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.float()

    B = pred.shape[0]
    p = pred.view(B, -1)
    g = mask.view(B, -1)

    inter = (p * g).sum(dim=1)
    total = p.sum(dim=1) + g.sum(dim=1)

    dice_per_image = (2 * inter + smooth) / (total + smooth)

    return dice_per_image.mean().item()



def calculate_precision_recall(pred_logits: torch.Tensor,
                               mask: torch.Tensor,
                               thr: float = 0.5,
                               smooth: float = 1e-6) -> Tuple[float, float]:
    """
    Precision & Recall per-image, batch-averaged.
    Koristi hard threshold thr.
    """
    pred = (torch.sigmoid(pred_logits) > thr).float()
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.float()

    B = pred.shape[0]
    p = pred.view(B, -1)
    g = mask.view(B, -1)

    tp = (p * g).sum(dim=1)
    fp = (p * (1 - g)).sum(dim=1)
    fn = ((1 - p) * g).sum(dim=1)

    precision_per_image = (tp + smooth) / (tp + fp + smooth)
    recall_per_image    = (tp + smooth) / (tp + fn + smooth)

    return precision_per_image.mean().item(), recall_per_image.mean().item()