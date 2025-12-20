import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # enforce deterministic cuBLAS behavior & enable mem-efficient attention kernels

import csv
import numpy as np
from collections import defaultdict, OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import wandb

from ptflops import get_model_complexity_info

from models.UltraLightFCN_base import UltraLightFCN
from utils.dataset import SolarPanelDataset
from utils.helpers import get_loss_function
from utils.metrics import calculate_dice, calculate_iou, calculate_precision_recall
from utils.repro import set_global_seed, seed_worker

torch.multiprocessing.set_sharing_strategy("file_system")

# =======================
#        GLOBAL CONFIG
# =======================
# Device selection: use GPU if available, otherwise fall back to CPU
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset root folder and split names (expected directory structure: DATA_ROOT/train, /valid, /test)
DATA_ROOT      = "/workspace/UltraLightFCN_snakemake/dataset"  # adjust as needed
TRAIN_SPLIT    = "train"
VAL_SPLIT      = "valid"
TEST_SPLIT     = "test"

# Model and pretraining config
MODEL_NAME     = "UltraLightFCN_base"
# Path to pretrained SimCLR encoder checkpoint (only encoder part will be loaded)
PRETRAINED_ENCODER_CKPT = "/workspace/UltraLightFCN_snakemake/pretrain/checkpoints/simclr/UltraLightFCN(rgb)-simclr_encoder_final_seed37.pth"

# Encoder training mode:
#   - "frozen": encoder weights are frozen, only decoder is trained
#   - "finetune": encoder is finetuned with a smaller LR (ENC_LR_MULT)
ENCODER_MODE   = "finetune"    # "frozen" or "finetune"
ENC_LR_MULT    = 0.1

# Edge detector and input channels:
#   EDGE_DETECTOR: optional edge preprocessing (must match training setup)
#   CHANNELS: 3 for RGB, 4 for RGB+edge
EDGE_DETECTOR  = None        # keep consistent with training
CHANNELS       = 3           # 3=RGB, 4=RGB+edge

# Training hyperparameters for the ablation study
EPOCHS         = 30
BATCH_SZ       = 8
WEIGHT_DECAY   = 1e-5
BASE_LR        = 2.409541000688119e-03
NUM_WORKERS    = 4

# Variants to evaluate in the ablation study.
# Each entry defines a set of architectural and loss-related hyperparameters.
VARIANTS = OrderedDict({
    # Baseline configuration (from Optuna or main training)
    "baseline":             dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=16,
                                 loss_name="BCEDiceLoss", w1=0.4, w2=0.6),

    # Ablations: remove mini-ASPP entirely (no context module)
    "mini_aspp_off":        dict(mini_aspp=False, mini_aspp_gpool=False, use_sa=True,  sa_window_size=16,
                                 loss_name="BCEDiceLoss", w1=0.4, w2=0.6),

    # Ablation: completely remove self-attention
    "no_sa":                dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=False, sa_window_size=16,
                                 loss_name="BCEDiceLoss", w1=0.4, w2=0.6),

    # Ablation: keep mini-ASPP but disable global pooling branch
    "mini_aspp_no_gpool":   dict(mini_aspp=True,  mini_aspp_gpool=False, use_sa=True,  sa_window_size=16,
                                 loss_name="BCEDiceLoss", w1=0.4, w2=0.6),

    # Ablation: smaller attention window (more local)
    "sa_ws_8":              dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=8,
                                 loss_name="BCEDiceLoss", w1=0.4, w2=0.6),

    # Ablation: larger attention window (more global)
    "sa_ws_32":             dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=32,
                                 loss_name="BCEDiceLoss", w1=0.4, w2=0.6),

    # Ablation: change BCE vs Dice weighting to 0.3 / 0.7
    "loss_03_07":           dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=16,
                                 loss_name="BCEDiceLoss", w1=0.3, w2=0.7),

    # Ablation: equal BCE/Dice weighting (0.5 / 0.5)
    "loss_05_05":           dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=16,
                                 loss_name="BCEDiceLoss", w1=0.5, w2=0.5),
})

# List of seeds for multi-run robustness (report mean ± std over these seeds)
SEEDS = [42, 52, 62]   # at least 3 seeds recommended

# Thresholds used for validation-time threshold sweep (to select best segmentation threshold)
THRESHOLDS = np.linspace(0.05, 0.95, 19)  # 0.05 step

# Weights & Biases (experiment tracking) configuration
WB_PROJECT = "UltraLightFCN-ablation"
WB_ENTITY  = "tomislav-kescec-algebra"
WB_MODE    = "offline"  # use "online" for live syncing to W&B

# Output directory for all CSVs and checkpoints related to this ablation study
OUT_DIR = os.path.join("ablation_outputs", ENCODER_MODE)
os.makedirs(OUT_DIR, exist_ok=True)


# =======================
#   MODEL PARAM HELPERS
# =======================
def build_ultralight_params(variant_cfg: dict):
    """
    Build the parameter dictionary for UltraLightFCN.

    The encoder/decoder topology is fixed here, while variant_cfg
    injects ablation-specific options for context/attention modules.
    """
    return {
        # Encoder configuration
        'enc_channels':     [16, 16, 32, 32, 64],
        'enc_kernel_sizes': [3,  3,  3,  3,  3],
        'enc_strides':      [1,  2,  2,  1,  1],
        'dilations':        [2,  4],

        # Decoder configuration
        'dec_channels':     [32, 16, 16],
        'dec_kernel_sizes': [3,  3],
        'dec_strides':      [1,  1],
        'upscale':          [2,  2],

        # Context / Self-attention configuration (driven by VARIANTS)
        'mini_aspp':        variant_cfg['mini_aspp'],
        'mini_aspp_gpool':  variant_cfg['mini_aspp_gpool'],
        'use_sa':           variant_cfg['use_sa'],
        'sa_windowed':      True,
        'sa_window_size':   variant_cfg['sa_window_size'],
        'sa_shifted':       True,
        'sa_heads':         4,
        'sa_dropout':       0.0,
    }


# Encoder layer name prefixes – used to separate encoder/decoder params for different LR / freezing
ENC_PREFIXES = ("block1", "dsconv2", "dsconv3", "dilconv4", "dilconv5")

def split_encoder_decoder_params(model):
    """
    Split model parameters into encoder and decoder groups based on layer name prefixes.

    Returns:
        enc_params: list of encoder parameters
        dec_params: list of decoder parameters
    """
    enc_params, dec_params = [], []
    for name, p in model.named_parameters():
        if name.startswith(ENC_PREFIXES):
            enc_params.append(p)
        else:
            dec_params.append(p)
    return enc_params, dec_params

def freeze_encoder(model):
    """
    Freeze encoder modules:
        - put them in eval() mode
        - disable gradient computation (requires_grad = False)
    This is used when ENCODER_MODE == "frozen".
    """
    for m in [model.block1, model.dsconv2, model.dsconv3, model.dilconv4, model.dilconv5]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

def set_encoder_eval(model):
    """
    Force encoder submodules to eval() each epoch,
    ensuring batch-norm/dropout etc. are not updated when encoder is frozen.
    """
    for m in [model.block1, model.dsconv2, model.dsconv3, model.dilconv4, model.dilconv5]:
        m.eval()

def load_simclr_encoder(model, ckpt_path):
    """
    Load SimCLR-pretrained encoder weights into the model.

    - Expects ckpt that may contain:
        - a full dict with key "encoder", or
        - a flat state_dict with encoder.* keys
    - Strips prefixes like "module." and "encoder." and only loads keys
      that match encoder layer names and shapes.

    This function is robust to missing keys (strict=False).
    """
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print(f"[INFO] No valid encoder checkpoint: {ckpt_path}")
        return

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "encoder" in sd:
        sd = sd["encoder"]

    target = model.state_dict()
    new_sd = {}
    loaded = 0
    for k, v in sd.items():
        k2 = k
        # Remove possible DataParallel and encoder prefixes
        if k2.startswith("module."):
            k2 = k2[7:]
        if k2.startswith("encoder."):
            k2 = k2[len("encoder."):]
        # Only load encoder layers that exist in the target model with matching shape
        if k2.startswith(ENC_PREFIXES) and k2 in target and target[k2].shape == v.shape:
            new_sd[k2] = v
            loaded += 1

    model.load_state_dict(new_sd, strict=False)
    print(f"[ENC] loaded keys: {loaded}")


# =======================
#      DATA HELPERS
# =======================
def make_loader(split, batch_size, shuffle, seed):
    """
    Create a DataLoader for a given split (train/val/test).

    Args:
        split: one of TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
        batch_size: batch size for this loader
        shuffle: whether to shuffle samples (True for train, False for val/test)
        seed: base seed for the Generator and worker_init_fn

    Returns:
        PyTorch DataLoader with deterministic behavior across runs.
    """
    ds = SolarPanelDataset(
        data_dir=f"{DATA_ROOT}/{split}",
        mode="train" if split == TRAIN_SPLIT else ("val" if split == VAL_SPLIT else "test"),
        edge_detector=EDGE_DETECTOR,
        channels=CHANNELS
    )
    # Per-loader random generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,  # ensures deterministic worker seeding
        generator=g,
        persistent_workers=True,
        prefetch_factor=2
    )


# =======================
#     TRAIN ONE VARIANT
# =======================
def train_one_variant(seed, variant_name, variant_cfg):
    """
    Train UltraLightFCN for a given variant and seed, tune the threshold on validation,
    and return the best checkpoint path, chosen threshold, and best validation Dice.

    Flow:
        1) set global seeds
        2) create W&B run
        3) build model + load SimCLR encoder
        4) train with BCE+Dice or BCE+Dice+Focal loss
        5) keep best weights by val Dice@0.6
        6) sweep thresholds on validation to find best threshold
    """
    set_global_seed(seed, deterministic=True)

    # --- W&B run for train+val phase ---
    run = wandb.init(
        mode=WB_MODE,
        project=WB_PROJECT,
        entity=WB_ENTITY,
        name=f"{variant_name}-seed_{seed}-{ENCODER_MODE}",
        config={
            "phase": "ablation-train+val",
            "variant": variant_name,
            "seed": seed,
            "model": MODEL_NAME,
            "encoder_mode": ENCODER_MODE,
            "edge_detector": EDGE_DETECTOR,
            "channels": CHANNELS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SZ,
            "weight_decay": WEIGHT_DECAY,
            "base_lr": BASE_LR,
            "dataset_root": DATA_ROOT,
            "splits": {"train": TRAIN_SPLIT, "val": VAL_SPLIT},
            "variant_cfg": variant_cfg,
        }
    )

    # Data loaders for this seed/variant
    train_loader = make_loader(TRAIN_SPLIT, BATCH_SZ, shuffle=True,  seed=seed)
    val_loader   = make_loader(VAL_SPLIT,   BATCH_SZ, shuffle=False, seed=seed)

    # Build model for this variant
    params = build_ultralight_params(variant_cfg)
    model = UltraLightFCN(in_channels=CHANNELS, num_classes=1, params=params).to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Optionally load SimCLR encoder weights (same as in main training)
    load_simclr_encoder(model, PRETRAINED_ENCODER_CKPT)

    # Log MACs and number of parameters (for 256x256 inputs) to W&B config
    macs, nparams = get_model_complexity_info(
        model,
        (CHANNELS, 256, 256),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    wandb.config.update(
        {"MACs": macs, "Parameters": nparams},
        allow_val_change=True
    )

    # Optimizer:
    #   - if encoder is frozen: train only decoder params
    #   - otherwise: encoder gets reduced LR, decoder full LR
    enc_params, dec_params = split_encoder_decoder_params(model)
    if ENCODER_MODE == "frozen":
        freeze_encoder(model)
        optimizer = torch.optim.AdamW(dec_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": BASE_LR * ENC_LR_MULT},
                {"params": dec_params, "lr": BASE_LR}
            ],
            lr=BASE_LR,
            weight_decay=WEIGHT_DECAY
        )

    # LR scheduler: ReduceLROnPlateau on validation Dice
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6
    )

    # Loss function: configured per-variant (either BCEDiceLoss or BCEDiceFocalLoss)
    if variant_cfg["loss_name"] == "BCEDiceLoss":
        criterion = get_loss_function(
            "BCEDiceLoss",
            bce_weight=variant_cfg["w1"],
            dice_weight=variant_cfg["w2"]
        )
    else:
        criterion = get_loss_function(
            "BCEDiceFocalLoss",
            bce_weight=0.5,
            dice_weight=0.1,
            focal_weight=0.3,
            alpha_focal=0.7,
            gamma_focal=3.0
        )

    # Mixed-precision grad scaler (if CUDA is available)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Tracking best checkpoint according to val Dice@0.6
    best_dice, best_epoch = -1.0, -1
    ckpt_dir = os.path.join(OUT_DIR, "train_models", variant_name, f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{MODEL_NAME}({ 'rgb' if CHANNELS==3 else 'edge' })-seed_{seed}-{ENCODER_MODE}.pth"
    )

    # --- Training loop over EPOCHS ---
    for epoch in range(EPOCHS):
        model.train()
        if ENCODER_MODE == "frozen":
            # Ensure encoder stays in eval mode (no BN stats updates etc.)
            set_encoder_eval(model)

        train_loss_sum = 0.0

        # Single training epoch
        for images, masks, *rest in tqdm(
            train_loader,
            desc=f"[{variant_name}|seed{seed}|{ENCODER_MODE}] Train {epoch+1}/{EPOCHS}"
        ):
            images = images.to(DEVICE, non_blocking=True)
            if DEVICE.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            masks  = masks.to(DEVICE,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # Mixed precision for forward + loss
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                logits = model(images)
                loss   = criterion(logits, masks)

            # Backprop with AMP scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(1, len(train_loader))

        # --- Validation epoch (batch-averaged metrics, fixed thr=0.6 for consistency) ---
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            dice = iou = prec = rec = 0.0

            for images, masks, *rest in tqdm(
                val_loader,
                desc=f"[{variant_name}|seed{seed}|{ENCODER_MODE}] Val"
            ):
                images = images.to(DEVICE, non_blocking=True)
                masks  = masks.to(DEVICE,  non_blocking=True)

                with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    logits = model(images)
                    vloss  = criterion(logits, masks)

                val_loss_sum += float(vloss.item())

                # Metrics with a fixed threshold 0.6 (used for scheduler + model selection)
                dice += float(calculate_dice(logits, masks, thr=0.6))
                iou  += float(calculate_iou(logits, masks, thr=0.6))
                p, r = calculate_precision_recall(logits, masks, thr=0.6)
                prec += float(p)
                rec  += float(r)

            val_loss = val_loss_sum / max(1, len(val_loader))
            dice    /= max(1, len(val_loader))
            iou     /= max(1, len(val_loader))
            prec    /= max(1, len(val_loader))
            rec     /= max(1, len(val_loader))

        # Step LR scheduler using validation Dice
        scheduler.step(dice)

        # Log training/validation stats to W&B
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice@0.6": dice,
            "val_iou@0.6": iou,
            "val_precision@0.6": prec,
            "val_recall@0.6": rec,
            "lr_group0": optimizer.param_groups[0]["lr"],
            "lr_group1": optimizer.param_groups[1]["lr"]
                        if len(optimizer.param_groups) > 1
                        else optimizer.param_groups[0]["lr"]
        })

        # Save checkpoint if this epoch gives the best val Dice so far
        if dice > best_dice:
            best_dice, best_epoch = dice, epoch+1
            torch.save(model.state_dict(), ckpt_path)
            wandb.log({
                "best_val_dice@0.6": best_dice,
                "best_epoch": best_epoch,
                "best_ckpt": ckpt_path
            })

    # --- Threshold sweep on validation with the best checkpoint ---
    # Reload best weights (by val Dice@0.6) and evaluate Dice over a grid of thresholds.
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model.to(DEVICE)
    model.eval()

    thr2dice = {}
    with torch.no_grad():
        for thr in THRESHOLDS:
            dsum = 0.0
            for images, masks, *rest in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks  = masks.to(DEVICE,  non_blocking=True)
                logits = model(images)
                dsum  += float(calculate_dice(logits, masks, thr=float(thr)))

            thr2dice[float(thr)] = dsum / max(1, len(val_loader))

    # Pick the threshold that yields the highest mean validation Dice
    best_thr = max(thr2dice.items(), key=lambda kv: kv[1])[0]

    # Log threshold sweep as a W&B line plot
    wandb.log({
        "val/threshold_sweep": wandb.plot.line_series(
            xs=[list(thr2dice.keys())],
            ys=[list(thr2dice.values())],
            keys=[f"{variant_name}-seed{seed}"],
            title="Val Dice vs threshold",
            xname="threshold"
        )
    })
    wandb.log({
        "val/best_threshold": best_thr,
        "val/best_threshold_dice": thr2dice[best_thr]
    })
    wandb.finish()

    return ckpt_path, best_thr, best_dice


# =======================
#     TEST & STATS
# =======================
def bootstrap_ci(values, alpha=0.05, n_boot=1000, rng_seed=123):
    """
    Compute a non-parametric bootstrap confidence interval for the mean.

    Args:
        values: list/array of per-sample metric values
        alpha: 1 - confidence level (e.g. 0.05 → 95% CI)
        n_boot: number of bootstrap resamples
        rng_seed: seed for the bootstrap RNG

    Returns:
        mean, lower_bound, upper_bound (for the specified CI level)
    """
    rng = np.random.default_rng(rng_seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(values[idx].mean())
    lo = np.percentile(stats, 100*(alpha/2))
    hi = np.percentile(stats, 100*(1-alpha/2))
    return float(values.mean()), float(lo), float(hi)

def paired_wilcoxon(x, y):
    """
    Paired Wilcoxon signed-rank test on two paired samples x, y.

    - If scipy is available:
        uses scipy.stats.wilcoxon with two-sided alternative.
    - Otherwise:
        falls back to a simple sign test with an approximate two-sided
        binomial p-value.

    Returns:
        statistic (Wilcoxon or sign test stat),
        p-value,
        method string ("wilcoxon" or "sign_test")
    """
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(x, y, zero_method='pratt', alternative='two-sided', mode='auto')
        stat = float(getattr(res, "statistic", res[0]))
        p    = float(getattr(res, "pvalue",    res[1]))
        return stat, p, "wilcoxon"
    except Exception:
        # Fallback: sign test using a binomial distribution approximation.
        diffs = np.array(x) - np.array(y)
        npos = int((diffs > 0).sum())
        nneg = int((diffs < 0).sum())
        n = npos + nneg

        from math import comb
        # Two-sided binomial p-value approximation
        p_two = 2 * sum(comb(n, k) * (0.5**n) for k in range(0, min(npos, nneg)+1))
        return float(npos - nneg), float(min(1.0, p_two)), "sign_test"

def test_and_stats(variant_name, seed, ckpt_path, thr):
    """
    Run test split evaluation for a given variant/seed using the chosen threshold.

    Steps:
        1) reload model with best weights
        2) run over TEST_SPLIT
        3) compute per-image Dice/IoU/precision/recall
        4) compute bootstrap CI for each metric
        5) log results to W&B
        6) save per-image metrics as CSV and return summary dict

    Returns:
        dict with:
            - variant, seed, threshold
            - mean metrics and their 95% CI
            - path to per-image CSV
            - raw per-image metric arrays (for Wilcoxon)
    """
    set_global_seed(seed, deterministic=True)

    # W&B run for the test phase (separate run from training)
    run = wandb.init(
        mode=WB_MODE,
        project=WB_PROJECT,
        entity=WB_ENTITY,
        name=f"{variant_name}-seed_{seed}-{ENCODER_MODE}-TEST",
        config={
            "phase": "ablation-test",
            "variant": variant_name,
            "encoder_mode": ENCODER_MODE,
            "seed": seed,
            "test_split": TEST_SPLIT,
            "chosen_threshold": thr
        }
    )

    # Data & model for this test run
    test_loader = make_loader(TEST_SPLIT, BATCH_SZ, shuffle=False, seed=seed)
    params = build_ultralight_params(VARIANTS[variant_name])
    model = UltraLightFCN(in_channels=CHANNELS, num_classes=1, params=params).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model.eval()

    # Per-image metric containers
    per_image_ids, per_dice, per_iou, per_prec, per_rec = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[{variant_name}|seed{seed}] Test"):
            images, masks, *extras = batch
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE,  non_blocking=True)
            logits = model(images)

            # Compute metrics one sample at a time (utils internally apply threshold)
            for i in range(images.size(0)):
                di    = float(calculate_dice(logits[i:i+1], masks[i:i+1], thr=float(thr)))
                io    = float(calculate_iou(logits[i:i+1],  masks[i:i+1], thr=float(thr)))
                pi, ri = calculate_precision_recall(logits[i:i+1], masks[i:i+1], thr=float(thr))

                per_dice.append(di)
                per_iou.append(float(io))
                per_prec.append(float(pi))
                per_rec.append(float(ri))

                # Try to recover an image ID from dataset extras if available,
                # otherwise fall back to a synthetic index-based ID.
                img_id = None
                if len(extras) > 0:
                    maybe = extras[-1]
                    if isinstance(maybe, (list, tuple)) and len(maybe) == images.size(0):
                        img_id = str(maybe[i])
                    elif isinstance(maybe, str):
                        img_id = maybe
                if not img_id:
                    img_id = f"idx_{len(per_image_ids):06d}"

                per_image_ids.append(img_id)

    # Bootstrap CIs for each metric
    dice_mean, dice_lo, dice_hi = bootstrap_ci(per_dice)
    iou_mean,  iou_lo,  iou_hi  = bootstrap_ci(per_iou)
    p_mean,    p_lo,    p_hi    = bootstrap_ci(per_prec)
    r_mean,    r_lo,    r_hi    = bootstrap_ci(per_rec)

    # Log aggregate test metrics to W&B
    wandb.log({
        "test/dice_mean": dice_mean, "test/dice_lo95": dice_lo, "test/dice_hi95": dice_hi,
        "test/iou_mean":  iou_mean,  "test/iou_lo95":  iou_lo,  "test/iou_hi95":  iou_hi,
        "test/prec_mean": p_mean,    "test/prec_lo95": p_lo,    "test/prec_hi95": p_hi,
        "test/rec_mean":  r_mean,    "test/rec_lo95":  r_lo,    "test/rec_hi95":  r_hi,
    })

    # Save per-image metrics to CSV for this variant/seed
    csv_dir = os.path.join(OUT_DIR, "test_metrics", variant_name)
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"seed_{seed}_per_image.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "dice", "iou", "precision", "recall"])
        for rid, di, io, pp, rr in zip(per_image_ids, per_dice, per_iou, per_prec, per_rec):
            w.writerow([rid, f"{di:.6f}", f"{io:.6f}", f"{pp:.6f}", f"{rr:.6f}"])

    # Log per-image metrics CSV as a W&B artifact
    # art = wandb.Artifact(f"{variant_name}_seed_{seed}_{ENCODER_MODE}_test_metrics", type="evaluation")
    # art.add_file(csv_path)
    # wandb.log_artifact(art)
    wandb.finish()

    return {
        "variant": variant_name,
        "seed": seed,
        "threshold": float(thr),
        "dice_mean": dice_mean, "dice_lo95": dice_lo, "dice_hi95": dice_hi,
        "iou_mean":  iou_mean,  "iou_lo95":  iou_lo,  "iou_hi95":  iou_hi,
        "prec_mean": p_mean,    "prec_lo95": p_lo,    "prec_hi95": p_hi,
        "rec_mean":  r_mean,    "rec_lo95":  r_lo,    "rec_hi95":  r_hi,
        "per_image_csv": csv_path,
        "per_image_arrays": {
            "ids":  per_image_ids,
            "dice": per_dice,
            "iou":  per_iou,
            "prec": per_prec,
            "rec":  per_rec
        }
    }


# =======================
#        MAIN FLOW
# =======================
def main():
    """
    Main ablation study driver.

    For each (variant, seed) pair:
        - train the model and pick best threshold on validation
        - evaluate on test and collect per-image metrics

    Then:
        - aggregate results across seeds per variant and save ablation_table.csv
        - run paired Wilcoxon tests vs baseline and save ablation_summary.csv
        - upload both tables as a W&B artifact
    """
    # Per (variant, seed) summary results from test_and_stats()
    all_results = []
    # Per (seed, variant) per-image metric arrays (for Wilcoxon vs baseline)
    per_seed_arrays = defaultdict(dict)

    for variant_name, variant_cfg in VARIANTS.items():
        for seed in SEEDS:
            # 1) Train + choose threshold on validation
            ckpt_path, best_thr, best_val_dice = train_one_variant(seed, variant_name, variant_cfg)

            # 2) Test with chosen threshold + compute bootstrap CIs
            res = test_and_stats(variant_name, seed, ckpt_path, best_thr)
            all_results.append(res)
            per_seed_arrays[seed][variant_name] = res["per_image_arrays"]

    # =======================
    #    AGGREGATE BY SEEDS & TABLE OUTPUT
    # =======================
    # Group all_results by variant name and compute mean/std of metrics across seeds.
    # This produces ablation_table.csv which is "paper-ready":
    #   variant, n_seeds, mean ± std for Dice/IoU/Precision/Recall.
    variant2runs = defaultdict(list)
    for res in all_results:
        variant2runs[res["variant"]].append(res)

    table_path = os.path.join(OUT_DIR, "ablation_table.csv")
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant",
            "n_seeds",
            "dice_mean", "dice_std",
            "iou_mean",  "iou_std",
            "prec_mean", "prec_std",
            "rec_mean",  "rec_std"
        ])

        for vname, runs in variant2runs.items():
            n = len(runs)

            dice_arr = np.array([r["dice_mean"] for r in runs], dtype=float)
            iou_arr  = np.array([r["iou_mean"]  for r in runs], dtype=float)
            p_arr    = np.array([r["prec_mean"] for r in runs], dtype=float)
            r_arr    = np.array([r["rec_mean"]  for r in runs], dtype=float)

            # std(ddof=1) → sample standard deviation.
            # If only one seed is available, use 0.0 to avoid NumPy warnings.
            dice_std = float(dice_arr.std(ddof=1)) if n > 1 else 0.0
            iou_std  = float(iou_arr.std(ddof=1))  if n > 1 else 0.0
            p_std    = float(p_arr.std(ddof=1))    if n > 1 else 0.0
            r_std    = float(r_arr.std(ddof=1))    if n > 1 else 0.0

            w.writerow([
                vname,
                n,
                float(dice_arr.mean()), dice_std,
                float(iou_arr.mean()),  iou_std,
                float(p_arr.mean()),    p_std,
                float(r_arr.mean()),    r_std,
            ])

    print(f"[INFO] Seed-level aggregation table written to: {table_path}")

    # =======================
    #    PAIRED WILCOXON vs BASELINE (by seed, pooled report)
    # =======================
    # For each seed, compare each ablation variant against the baseline
    # using a paired test on per-image Dice scores. This yields a CSV with
    # (seed, variant, test, statistic, p_value, n_images).
    baseline_name = "baseline"
    wilcoxon_rows = []

    for seed in SEEDS:
        if baseline_name not in per_seed_arrays[seed]:
            print(f"[WARN] Missing baseline for seed {seed}, skipping Wilcoxon.")
            continue

        base = per_seed_arrays[seed][baseline_name]
        base_ids = base["ids"]
        # Map from image_id to index for baseline metrics (for alignment)
        base_map = {img_id: i for i, img_id in enumerate(base_ids)}

        for variant_name in VARIANTS:
            if variant_name == baseline_name:
                continue
            if variant_name not in per_seed_arrays[seed]:
                continue

            var = per_seed_arrays[seed][variant_name]

            # Align Dice values by image_id (robust against ordering differences)
            xs_dice, ys_dice = [], []
            for img_id, di in zip(var["ids"], var["dice"]):
                if img_id in base_map:
                    xs_dice.append(di)
                    ys_dice.append(base["dice"][base_map[img_id]])

            # Only run the test if we have enough paired samples
            if len(xs_dice) >= 10:
                stat, p, method = paired_wilcoxon(xs_dice, ys_dice)
                wilcoxon_rows.append([seed, variant_name, method, stat, p, len(xs_dice)])

    # Save Wilcoxon/sign-test summary to CSV
    sum_path = os.path.join(OUT_DIR, "ablation_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "variant_vs_baseline", "test", "statistic", "p_value", "n_images"])
        for row in wilcoxon_rows:
            w.writerow(row)

    # Upload both summary tables (stats + seed-aggregated metrics) as a W&B artifact
    # run = wandb.init(
    #     mode=WB_MODE,
    #     project=WB_PROJECT,
    #     entity=WB_ENTITY,
    #     name=f"ablation-summary-{ENCODER_MODE}",
    #     config={"seeds": SEEDS, "variants": list(VARIANTS.keys())}
    # )
    # art = wandb.Artifact("ablation_results", type="results")
    # art.add_file(sum_path)   # Wilcoxon / sign-test results
    # art.add_file(table_path) # Seed-aggregated metrics table (for the paper)
    # wandb.log_artifact(art)
    # wandb.finish()

    print(f"[DONE] Ablation summary written to: {sum_path}")


if __name__ == "__main__":
    # Entry point for running the full ablation study.
    # This will train+test all (variant, seed) combinations and produce CSVs + W&B artifacts.
    main()
