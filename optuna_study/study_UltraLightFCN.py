# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" # for deterministic behavior and memory-efficient attention

import torch, optuna
from skimage.filters.rank import threshold

from tqdm import tqdm
from torch.amp import autocast
from torch import GradScaler
from torch.utils.data import DataLoader

from models.UltraLightFCN_base import UltraLightFCN
from utils.dataset import SolarPanelDataset
from utils.helpers import get_loss_function, clear_cuda_cache, get_model, save_best_callback

from utils.metrics import calculate_dice
from utils.repro import set_global_seed, GLOBAL_SEED, seed_worker

# -------------------------------------------------------------------------
# 0) GLOBAL: reproducibility and environment logging
# -------------------------------------------------------------------------
set_global_seed(42, deterministic=False)      # 1️⃣ set random seed for reproducibility

# -------------------------------------------------------------------------
# 1) Constants
# -------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT   = "datasets/reduced_dataset"
CHANNELS    = 3                       # 4 RGB + edge
MODEL       = 'UltraLightFCN'
EPOCHS      = 30                      # Number of epochs for training
WD          = 1e-5                    # Default weight decay

# -------------------------------------------------------------------------
# 2) One Optuna trial
# -------------------------------------------------------------------------
def objective(trial):
    # --- 2.1 Define search space ----------------------------------------
    mini_aspp = trial.suggest_categorical('mini_aspp', [True, False])
    mini_aspp_gpool = trial.suggest_categorical('mini_aspp_gpool', [True, False]) if mini_aspp else False
    use_sa = trial.suggest_categorical('use_sa', [True, False])
    sa_window_size = trial.suggest_categorical('sa_window_size', [8, 16]) if use_sa else 8
    seg_lr = trial.suggest_float("seg_lr", 0.001, 0.009, log=True)
    bs = trial.suggest_categorical("batch_size", [8, 16])
    dice_thr = trial.suggest_float("dice_thr", 0.3, 0.7, step=0.1)
    edge_detector = trial.suggest_categorical("edge_detector", ['prewitt', 'laplacian', 'sobel', 'canny']) if CHANNELS == 4 else None
    loss_name = trial.suggest_categorical(
        "loss", [
            # 'BCELoss', 'DiceLoss', 'FocalLoss', 'BCEDiceFocalLoss', 'BCEDiceLoss',
            'DiceLoss', 'BCEDiceFocalLoss', 'BCEDiceLoss'
        ])

    extra = {}
    if loss_name in {"BCEDiceLoss", "BCEDiceFocalLoss"}:
        w1 = trial.suggest_float("w1", 0.2, 0.8, step=0.1)
        w2 = trial.suggest_float("w2", 0.1, 0.8, step=0.1)

        if loss_name == "BCEDiceLoss":
            s = w1 + w2
            extra |= dict(bce_weight=w1 / s, dice_weight=w2 / s)

        elif loss_name == "BCEDiceFocalLoss":
            w3 = trial.suggest_float("w3", 0.1, 0.8, step=0.1)
            s = w1 + w2 + w3

            alpha = trial.suggest_float("alpha", 0.05, 0.95, step=0.05)
            gamma = trial.suggest_float("gamma", 1.0, 5.0, step=0.5)

            extra |= dict(
                bce_weight=w1 / s,
                dice_weight=w2 / s,
                focal_weight=w3 / s,
                alpha_focal=alpha,
                gamma_focal=gamma
            )

    elif loss_name == "FocalLoss":
        alpha = trial.suggest_float("alpha", 0.05, 0.95, step=0.05)
        gamma = trial.suggest_float("gamma", 1.0, 5.0, step=0.5)
        extra |= dict(alpha=alpha, gamma=gamma)

    model_params = {
        # Encoder
        'enc_channels': [16, 16, 32, 32, 64],
        'enc_kernel_sizes': [3, 3, 3, 3, 3],
        'enc_strides': [1, 2, 2, 1, 1],
        'dilations': [2, 4],
        # Decoder
        'dec_channels': [32, 16, 16],
        'dec_kernel_sizes': [3, 3],
        'dec_strides': [1, 1],
        'upscale': [2, 2],
        # Context
        'mini_aspp': mini_aspp,
        'mini_aspp_gpool': mini_aspp_gpool,
        # Attention
        'use_sa': use_sa,
        'sa_windowed': True,
        'sa_window_size': sa_window_size,
        'sa_shifted': True,
        'sa_heads': 4,
        'sa_dropout': 0.1,
    }


    # --- 2.2 Data loaders -----------------------------------------------
    # Generators with global seed for reproducibility
    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)

    # Segmentation dataset and dataloader
    train_ds = SolarPanelDataset(
        f"{DATA_ROOT}/train",
        mode="train",
        edge_detector=edge_detector,
        channels=CHANNELS
    )
    val_ds = SolarPanelDataset(
        f"{DATA_ROOT}/valid",
        mode="val",
        edge_detector=edge_detector,
        channels=CHANNELS
    )

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        pin_memory=True, drop_last=True,
        # num_workers=2, persistent_workers=True,
        # worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        pin_memory=True, drop_last=False,
        # num_workers=2, persistent_workers=True,
        # worker_init_fn=seed_worker, generator=g
    )

    # --- 2.3 Segmentation  ---------------------------------------------
    seg_model     = UltraLightFCN(in_channels=CHANNELS, num_classes=1, params=model_params).to(DEVICE)
    seg_opt       = torch.optim.AdamW(seg_model.parameters(), lr=seg_lr, weight_decay=WD)
    seg_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        seg_opt, mode="max",
        factor=0.5, patience=2,
        threshold=1e-4, min_lr=1e-6
    )
    seg_crit      = get_loss_function(loss_name, **extra)

    # --- 2.4 Training parameters ---------------------------------------
    # GradScaler for mixed precision training
    scaler = GradScaler()
    # Variable to track the best Dice score
    best_dice = 0.0

    for epoch in range(EPOCHS):
        # ---- Training ----
        seg_model.train()
        # train_loss_sum = 0.0
        for images, masks, *_ in tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{EPOCHS}", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            seg_opt.zero_grad()
            with autocast(device_type='cuda'):
                outputs = seg_model(images)
                loss    = seg_crit(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(seg_opt)
            scaler.update()
        #     train_loss_sum += loss.item()
        # avg_train_loss = train_loss_sum / len(train_loader)

        # ---- Validation ----
        seg_model.eval()
        # val_loss_sum = 0.0
        val_dice_sum = 0.0
        with torch.no_grad():
            for images, masks, *_ in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                with autocast(device_type='cuda'):
                    outputs  = seg_model(images)
                    # val_loss  = seg_crit(outputs, masks)
                # val_loss_sum += val_loss.item()
                val_dice_sum += calculate_dice(outputs, masks, thr=dice_thr)
        # avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_dice = val_dice_sum / len(val_loader)

        # ---- Optuna pruning on Dice ----
        if trial is not None:
            trial.report(avg_val_dice, epoch + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()


        # ---- Scheduler step & EarlyStop ----
        seg_scheduler.step(avg_val_dice)
        best_dice = max(best_dice, avg_val_dice)

        # (optional) saving model state_dict
        # trial.set_user_attr("state_dict", seg_model.state_dict())

    return best_dice

# -------------------------------------------------------------------------

def main():
    training_type = "RGB+edge" if CHANNELS == 4 else "RGB"

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{MODEL}-({training_type})",
        storage="sqlite:///UltraLightFCN_study.db",
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5,  # minimum number of epochs before evaluation
            max_resource=EPOCHS,  # max epochs to train
            reduction_factor=3  # how many trials are "discarded" per round
        )
    )

    study.optimize(
        objective,
        n_trials=100,
        timeout=24*60*60,
        callbacks=[clear_cuda_cache]
    )

    print("\n📈  Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"   {k:16s}: {v}")
    print("   Val Dice:", study.best_value)

if __name__ == '__main__':
    main()