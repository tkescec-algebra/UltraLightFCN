import torch.nn.functional as F
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        # BCEWithLogits s pos_weight ako je predan
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(
            mode="binary", from_logits=True, smooth=1e-6
        )
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce_loss  = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class BCEDiceTverskyLoss(nn.Module):
    def __init__(
        self,
        pos_weight=None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.3,
        tversky_weight: float = 0.2,
        smooth: float = 1e-6,
        alpha: float = 0.3,
        beta: float = 0.7,
    ):
        super().__init__()
        # BCEWithLogits s pos_weight ako je predan
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce = nn.BCEWithLogitsLoss()

        # Dice i Tversky
        self.dice = smp.losses.DiceLoss(
            mode="binary", from_logits=True, smooth=smooth
        )
        self.tversky = smp.losses.TverskyLoss(
            mode="binary", from_logits=True, alpha=alpha, beta=beta
        )

        # Težine za svaki loss
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight

    def forward(self, inputs, targets):
        """
        inputs: logits iz mreže (N,1,H,W)
        targets: maska (N,1,H,W) ili (N,H,W) s 0/1
        """
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        tversky_loss = self.tversky(inputs, targets)

        total = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.tversky_weight * tversky_loss
        )
        return total

class BCEDiceFocalLoss(nn.Module):
    def __init__(
        self,
        pos_weight=None,
        bce_weight: float = 0.4,
        dice_weight: float = 0.3,
        focal_weight: float = 0.3,
        smooth: float = 1e-6,
        alpha_focal: float = 0.25,
        gamma_focal: float = 2.0,
    ):
        super().__init__()
        # 1) BCEWithLogitsLoss s pos_weight
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce = nn.BCEWithLogitsLoss()

        # 2) DiceLoss
        self.dice = smp.losses.DiceLoss(
            mode="binary", from_logits=True, smooth=smooth
        )

        # 3) FocalLoss
        self.focal = smp.losses.FocalLoss(
            mode="binary",
            alpha=alpha_focal,
            gamma=gamma_focal
        )

        # Težine
        self.bce_weight   = bce_weight
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        """
        logits: (N,1,H,W) sirovi izlazi mreže
        targets: (N,1,H,W) ili (N,H,W) 0/1 maske
        """
        loss_bce   = self.bce(logits, targets)
        loss_dice  = self.dice(logits, targets)
        loss_focal = self.focal(logits, targets)

        total = (
            self.bce_weight   * loss_bce
          + self.dice_weight  * loss_dice
          + self.focal_weight * loss_focal
        )
        return total

class NTXentLoss(nn.Module):
    """
    NT-Xent loss (contrastive) that dynamically builds its mask
    to match the incoming batch size.
    """
    def __init__(self, temperature=0.5, device='cpu'):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def _mask_correlated(self, N):
        # Build a 2N x 2N mask with zeros on the diagonal and zero for positive pairs
        # True where negative samples, False otherwise
        mask = torch.ones(2*N, 2*N, dtype=torch.bool, device=self.device)
        mask.fill_diagonal_(False)

        idx = torch.arange(N, device=self.device)
        mask[idx, idx + N] = False
        mask[idx + N, idx] = False

        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (N, D) normalized embeddings for two views
        """
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)               # (2N, D)
        sim = torch.matmul(z, z.T) / self.temperature  # (2N, 2N)

        # subtract max for numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        logits_full = sim - sim_max

        # rebuild mask for this batch
        mask = self._mask_correlated(N)
        # extract negatives
        negatives = logits_full[mask].view(2*N, -1)   # (2N, 2N-2)

        # extract positive similarities: diagonals offset by +N and -N
        positives  = torch.cat([
            torch.diag(logits_full,   N),
            torch.diag(logits_full,  -N)
        ], dim=0).unsqueeze(1)                         # (2N, 1)

        # logits: positive first column, then negatives
        logits = torch.cat([positives , negatives], dim=1)    # (2N, 1 + 2N-2)

        # labels: 0 for the positive sample column
        labels = torch.zeros(2*N, dtype=torch.long).to(self.device)

        return self.criterion(logits, labels)