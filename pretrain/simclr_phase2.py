import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # optional: determinism + mem-efficient attention

import math
import torch
import wandb
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel
from utils.dataset import SimCLRSolarPanelDataset
from utils.loss_functions import NTXentLoss
from utils.repro import set_global_seed, GLOBAL_SEED, seed_worker

# ====================================
# 0) Constants / Configuration
# ====================================

DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT         = "/workspace/UltraLightFCN_snakemake/dataset" # prilagodi po potrebi (putanja do skupa podataka na RunPod-u)
MODEL_NAME        = "UltraLightFCN"
EDGE_DETECTOR     = None      # 'prewitt' | 'laplacian' | 'sobel' | 'canny' | None
CHANNELS          = 3         # 3 = RGB, 4 = RGB+edge
IMAGE_SIZE        = 256

# --- Batch & epochs (no grad accumulation) ---
SIMCLR_BS         = 1024
DROP_LAST         = True

# --- Hyperparameters  ---
SIMCLR_LR          = 0.0024554497897962607
SIMCLR_TEMPERATURE = 0.060705096152236224
SIMCLR_WD          = 1.584361249525883e-06

# --- Scheduler (step-based cosine with warmup) ---
WARMUP_PCT        = 0.10
LR_MIN            = 1e-5
MAX_GRAD_NORM     = 1.0

# --- Epoch control (always use a fixed total epoch count) ---
TOTAL_EPOCHS      = 200


# ====================================
# Helpers
# ====================================

def steps_per_epoch(n_samples: int, batch_size: int, drop_last: bool) -> int:
    """Number of batches per epoch from the dataloader."""
    return max(1, n_samples // batch_size) if drop_last else max(1, math.ceil(n_samples / batch_size))


# ====================================
# Main
# ====================================

def main(seed=GLOBAL_SEED, deterministic=False, strict=False):
    # Postavi reproducibilnost/performanse ZA OVAJ RUN
    set_global_seed(seed, deterministic=deterministic, strict=strict)
    training_type = "edge" if CHANNELS == 4 else "rgb"

    # W&B init (offline je već u tvom primjeru; ostavljam isto)
    run_name = f"{MODEL_NAME}({training_type})-simclr-seed_{seed}" + ("-DET" if deterministic else "")
    wandb.init(
        mode="offline",
        name=run_name,
        project="UltraLightFCN_SimCLR-pretrening",
        entity="tomislav-kescec-algebra",
        config={
            "model": MODEL_NAME,
            "channels": CHANNELS,
            "edge_detector": EDGE_DETECTOR,
            "image_size": IMAGE_SIZE,
            "simclr_bs": SIMCLR_BS,
            "simclr_lr": SIMCLR_LR,
            "simclr_wd": SIMCLR_WD,
            "simclr_temperature": SIMCLR_TEMPERATURE,
            "scheduler": "cosine(step-based)+warmup",
            "warmup_pct": WARMUP_PCT,
            "lr_min": LR_MIN,
            "total_epochs": TOTAL_EPOCHS,
            # NOVO:
            "seed": seed,
            "deterministic": deterministic,
            "strict": strict,
        },
        reinit=True,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("simclr/*", step_metric="global_step")

    # Dataset & Loader
    g = torch.Generator(); g.manual_seed(seed)
    train_ds = SimCLRSolarPanelDataset(
        f"{DATA_ROOT}/train", edge_detector=EDGE_DETECTOR, channels=CHANNELS, image_size=IMAGE_SIZE
    )
    n_full = len(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=SIMCLR_BS,
        shuffle=True,
        pin_memory=True,
        drop_last=DROP_LAST,
        num_workers=12,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Accounting
    simclr_epochs = int(TOTAL_EPOCHS)
    spe_batches   = steps_per_epoch(n_full, SIMCLR_BS, DROP_LAST)
    total_updates = simclr_epochs * spe_batches
    warmup_steps  = max(1, int(WARMUP_PCT * total_updates))

    wandb.config.update({
        "n_train_full": n_full,
        "steps_per_epoch_batches": spe_batches,
        "total_updates": total_updates,
        "warmup_steps": warmup_steps,
    }, allow_val_change=True)

    print(
        f"[Info] N={n_full}, BS={SIMCLR_BS}, epochs={simclr_epochs}, batches/epoch={spe_batches}, "
        f"total_updates={total_updates}, warmup={warmup_steps}, LR={SIMCLR_LR:.3g}, WD={SIMCLR_WD:.2e}"
    )

    # Model & Optimizer
    encoder      = UltraLightEncoder(in_channels=CHANNELS, params={
        # Encoder
        'enc_channels': [16, 16, 32, 32, 64],
        'enc_kernel_sizes': [3, 3, 3, 3, 3],
        'enc_strides': [1, 2, 2, 1, 1],
        'dilations': [2, 4],
        # Decoder (unused in SSL forward but part of module)
        'dec_channels': [32, 16, 16],
        'dec_kernel_sizes': [3, 3],
        'dec_strides': [1, 1],
        'upscale': [2, 2],
        # Context
        'mini_aspp': True,
        'mini_aspp_gpool': True,
        # Attention (disabled here; lightweight encoder)
        'use_sa': False,
        'sa_windowed': True,
        'sa_window_size': 16,
        'sa_shifted': True,
        'sa_heads': 4,
        'sa_dropout': 0.0,
    }).to(DEVICE)

    proj_head    = ProjectionHead(in_dim=encoder.out_channels, hidden_dim=128, out_dim=64).to(DEVICE)
    simclr_model = SimCLRModel(encoder, proj_head).to(DEVICE)

    if DEVICE.type == 'cuda':
        simclr_model = simclr_model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(simclr_model.parameters(), lr=SIMCLR_LR, weight_decay=SIMCLR_WD)

    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=total_updates,
        lr_min=LR_MIN,
        warmup_lr_init=1e-6,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    criterion = NTXentLoss(temperature=SIMCLR_TEMPERATURE, device=DEVICE)
    scaler = GradScaler()

    # Train
    global_update = 0
    for epoch in range(simclr_epochs):
        simclr_model.train()
        running_loss = 0.0

        for xi, xj, *_ in tqdm(train_loader, desc=f"SimCLR Pretrain Epoch {epoch+1}/{simclr_epochs}", leave=False):
            xi = xi.to(DEVICE, non_blocking=True)
            xj = xj.to(DEVICE, non_blocking=True)
            if DEVICE.type == 'cuda':
                xi = xi.to(memory_format=torch.channels_last)
                xj = xj.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda' if DEVICE.type == 'cuda' else 'cpu'):
                zi = simclr_model(xi)
                zj = simclr_model(xj)
                loss = criterion(zi, zj)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(simclr_model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            global_update += 1
            scheduler.step_update(num_updates=global_update)

            running_loss += loss.item()

        avg_loss = running_loss / max(1, spe_batches)

        wandb.log({
            "global_step": global_update,
            "simclr/epoch": epoch + 1,
            "simclr/loss": avg_loss,
            "simclr/lr": optimizer.param_groups[0]['lr'],
            "simclr/global_updates": global_update,
        })

        print(
            f"[BS={SIMCLR_BS} | LR={SIMCLR_LR:.5g} | WD={SIMCLR_WD:.1e}] "
            f"Epoch {epoch+1}/{simclr_epochs}, NT-Xent: {avg_loss:.4f}, updates_so_far={global_update}"
        )

    # Save encoder only
    os.makedirs("checkpoints/simclr", exist_ok=True)
    det_tag = "_DET" if deterministic else ""
    final_path = os.path.join("checkpoints", "simclr", f"{MODEL_NAME}({training_type})-simclr_encoder_final_seed{seed}{det_tag}.pth")
    torch.save({"encoder": simclr_model.encoder.state_dict()}, final_path)
    print(f"\u2713 Saved final encoder to: {final_path}")

    wandb.finish()


# ====================================
# Launch multiple runs
# ====================================

if __name__ == "__main__":
    # 1) Brzi multi-seed (nedeterministički) — za odabir najboljeg encodera
    fast_seeds = [13, 37, 73]  # 3–5 seedova je dovoljno
    for s in fast_seeds:
        print(f"[FAST] Running SIMCLR pretraining with seed: {s}")
        main(seed=s, deterministic=False, strict=False)

    # 2) Referentni deterministički run (za točnu reprodukciju)
    ref_seed = GLOBAL_SEED  # npr. 42
    print(f"[REF-DET] Running SIMCLR pretraining with seed: {ref_seed} (deterministic)")
    main(seed=ref_seed, deterministic=True, strict=False)   # stavi strict=True ako želiš maksimalnu strogoću