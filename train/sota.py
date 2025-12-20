import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # determinism + mem-efficient attention

import csv
import numpy as np
from collections import defaultdict, OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import wandb

from ptflops import get_model_complexity_info

from utils.dataset import SolarPanelDataset
from utils.helpers import get_loss_function, get_model
from utils.metrics import calculate_dice, calculate_iou, calculate_precision_recall
from utils.repro import set_global_seed, seed_worker

torch.multiprocessing.set_sharing_strategy("file_system")

# =======================
#   GLOBAL CONFIG
# =======================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT   = "/workspace/UltraLightFCN_snakemake/dataset"
TRAIN_SPLIT = "train"
VAL_SPLIT   = "valid"
TEST_SPLIT  = "test"

EDGE_DETECTOR = None
CHANNELS      = 3          # 3=RGB, 4=RGB+edge

EPOCHS        = 30
BATCH_SZ      = 8
WEIGHT_DECAY  = 1e-5
NUM_WORKERS   = 4

# same threshold grid as in UltraLightFCN
THRESHOLDS = np.linspace(0.05, 0.95, 19)

WB_PROJECT = "solar-segmentation-baselines"
WB_ENTITY  = "tomislav-kescec-algebra"
WB_MODE    = "offline"

OUT_DIR = "sota_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 52, 62]

# =======================
#   SOTA MODELS
# =======================
SOTA_MODELS = OrderedDict({
    "dlv3p_resnet50": dict(
        model_key="dlv3p",
        encoder_name="resnet50",
        encoder_weights="imagenet",
        base_lr=0.0003007384563229953,
        bce_weight=0.4,
        dice_weight=0.6,
    ),

    "dlv3p_mobilenetv2": dict(
        model_key="dlv3p",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        base_lr=0.00032996241171076834,
        bce_weight=0.3,
        dice_weight=0.7,
    ),

    "unet_resnet34": dict(
        model_key="unet",
        encoder_name="resnet34",
        encoder_weights="imagenet",
        base_lr=0.0003072508498316878,
        bce_weight=0.4,
        dice_weight=0.6,
    ),

    "unet_efficientnet-b0": dict(
        model_key="unet",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        base_lr=0.00034419144358219245,
        bce_weight=0.6,
        dice_weight=0.4,
    ),
})


# =======================
#   DATA HELPERS
# =======================
def make_loader(split, batch_size, shuffle, seed):
    ds = SolarPanelDataset(
        data_dir=f"{DATA_ROOT}/{split}",
        mode="train" if split == TRAIN_SPLIT else ("val" if split == VAL_SPLIT else "test"),
        edge_detector=EDGE_DETECTOR,
        channels=CHANNELS
    )

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
        prefetch_factor=2
    )


# =======================
#   TRAIN ONE MODEL
# =======================
def build_model_from_cfg(cfg):
    ModelCls = get_model(cfg["model_key"])

    # SMP modeli
    if cfg.get("encoder_name", None) is not None:
        model = ModelCls(
            encoder_name=cfg["encoder_name"],
            encoder_weights=cfg["encoder_weights"],
            in_channels=CHANNELS,
            classes=1
        )
    else:
        # npr. torchvision FCN/DeepLab – prilagodi po potrebi
        model = ModelCls(num_classes=1)

    return model


def train_one_model(seed, model_name, cfg):
    """
    Treniraj jedan SOTA model za zadani seed,
    napravi threshold sweep na validation i vrati:
        ckpt_path, best_thr, best_val_dice
    """
    set_global_seed(seed, deterministic=True)

    run = wandb.init(
        mode=WB_MODE,
        project=WB_PROJECT,
        entity=WB_ENTITY,
        name=f"{model_name}-seed_{seed}",
        config={
            "phase": "sota-train+val",
            "model_name": model_name,
            "model_key": cfg["model_key"],
            "encoder_name": cfg.get("encoder_name", None),
            "encoder_weights": cfg.get("encoder_weights", None),
            "seed": seed,
            "epochs": EPOCHS,
            "batch_size": BATCH_SZ,
            "weight_decay": WEIGHT_DECAY,
            "base_lr": cfg["base_lr"],
            "dataset_root": DATA_ROOT,
            "splits": {"train": TRAIN_SPLIT, "val": VAL_SPLIT},
        }
    )

    train_loader = make_loader(TRAIN_SPLIT, BATCH_SZ, shuffle=True, seed=seed)
    val_loader   = make_loader(VAL_SPLIT,   BATCH_SZ, shuffle=False, seed=seed)

    model = build_model_from_cfg(cfg).to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # MACs/params
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["base_lr"],
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6
    )

    criterion = get_loss_function(
        "BCEDiceLoss",
        bce_weight=cfg["bce_weight"],
        dice_weight=cfg["dice_weight"]
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_dice, best_epoch = -1.0, -1
    ckpt_dir = os.path.join(OUT_DIR, "train_models", model_name, f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{model_name}({'rgb' if CHANNELS==3 else 'edge'})-seed_{seed}.pth"
    )

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for images, masks, *_ in tqdm(
            train_loader,
            desc=f"[{model_name}|seed{seed}] Train {epoch+1}/{EPOCHS}"
        ):
            images = images.to(DEVICE, non_blocking=True)
            if DEVICE.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(1, len(train_loader))

        # --- VAL ---
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            dice = iou = prec = rec = 0.0

            for images, masks, *_ in tqdm(
                val_loader,
                desc=f"[{model_name}|seed{seed}] Val"
            ):
                images = images.to(DEVICE, non_blocking=True)
                masks  = masks.to(DEVICE,  non_blocking=True)

                with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    logits = model(images)
                    vloss  = criterion(logits, masks)

                val_loss_sum += float(vloss.item())

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

        scheduler.step(dice)

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice@0.6": dice,
            "val_iou@0.6": iou,
            "val_precision@0.6": prec,
            "val_recall@0.6": rec,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if dice > best_dice:
            best_dice, best_epoch = dice, epoch+1
            torch.save(model.state_dict(), ckpt_path)
            wandb.log({
                "best_val_dice@0.6": best_dice,
                "best_epoch": best_epoch,
                "best_ckpt": ckpt_path
            })

    # --- threshold sweep na validation ---
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model.to(DEVICE).eval()

    thr2dice = {}
    with torch.no_grad():
        for thr in THRESHOLDS:
            dsum = 0.0
            for images, masks, *_ in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks  = masks.to(DEVICE,  non_blocking=True)
                logits = model(images)
                dsum  += float(calculate_dice(logits, masks, thr=float(thr)))
            thr2dice[float(thr)] = dsum / max(1, len(val_loader))

    best_thr = max(thr2dice.items(), key=lambda kv: kv[1])[0]

    wandb.log({
        "val/threshold_sweep": wandb.plot.line_series(
            xs=[list(thr2dice.keys())],
            ys=[list(thr2dice.values())],
            keys=[f"{model_name}-seed{seed}"],
            title=f"{model_name}: Val Dice vs threshold",
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
#   TEST & STATS
# =======================
def bootstrap_ci(values, alpha=0.05, n_boot=1000, rng_seed=123):
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
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(x, y, zero_method="pratt", alternative="two-sided", mode="auto")
        stat = float(getattr(res, "statistic", res[0]))
        p    = float(getattr(res, "pvalue",    res[1]))
        return stat, p, "wilcoxon"
    except Exception:
        diffs = np.array(x) - np.array(y)
        npos = int((diffs > 0).sum())
        nneg = int((diffs < 0).sum())
        n = npos + nneg
        from math import comb
        p_two = 2 * sum(comb(n, k) * (0.5**n) for k in range(0, min(npos, nneg)+1))
        return float(npos - nneg), float(min(1.0, p_two)), "sign_test"


def test_and_stats(model_name, seed, ckpt_path, thr):
    set_global_seed(seed, deterministic=True)

    run = wandb.init(
        mode=WB_MODE,
        project=WB_PROJECT,
        entity=WB_ENTITY,
        name=f"{model_name}-seed_{seed}-TEST",
        config={
            "phase": "sota-test",
            "model_name": model_name,
            "seed": seed,
            "test_split": TEST_SPLIT,
            "chosen_threshold": thr
        }
    )

    test_loader = make_loader(TEST_SPLIT, BATCH_SZ, shuffle=False, seed=seed)
    cfg = SOTA_MODELS[model_name]
    model = build_model_from_cfg(cfg).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model.eval()

    per_image_ids, per_dice, per_iou, per_prec, per_rec = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[{model_name}|seed{seed}] Test"):
            images, masks, *extras = batch
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE,  non_blocking=True)
            logits = model(images)

            for i in range(images.size(0)):
                di    = float(calculate_dice(logits[i:i+1], masks[i:i+1], thr=float(thr)))
                io    = float(calculate_iou(logits[i:i+1],  masks[i:i+1], thr=float(thr)))
                pi, ri = calculate_precision_recall(logits[i:i+1], masks[i:i+1], thr=float(thr))

                per_dice.append(di)
                per_iou.append(io)
                per_prec.append(pi)
                per_rec.append(ri)

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

    dice_mean, dice_lo, dice_hi = bootstrap_ci(per_dice)
    iou_mean,  iou_lo,  iou_hi  = bootstrap_ci(per_iou)
    p_mean,    p_lo,    p_hi    = bootstrap_ci(per_prec)
    r_mean,    r_lo,    r_hi    = bootstrap_ci(per_rec)

    wandb.log({
        "test/dice_mean": dice_mean, "test/dice_lo95": dice_lo, "test/dice_hi95": dice_hi,
        "test/iou_mean":  iou_mean,  "test/iou_lo95":  iou_lo,  "test/iou_hi95":  iou_hi,
        "test/prec_mean": p_mean,    "test/prec_lo95": p_lo,    "test/prec_hi95": p_hi,
        "test/rec_mean":  r_mean,    "test/rec_lo95":  r_lo,    "test/rec_hi95":  r_hi,
    })

    csv_dir = os.path.join(OUT_DIR, "test_metrics", model_name)
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"seed_{seed}_per_image.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "dice", "iou", "precision", "recall"])
        for rid, di, io, pp, rr in zip(per_image_ids, per_dice, per_iou, per_prec, per_rec):
            w.writerow([rid, f"{di:.6f}", f"{io:.6f}", f"{pp:.6f}", f"{rr:.6f}"])

    wandb.finish()

    return {
        "model_name": model_name,
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
#   MAIN FLOW
# =======================
def main():
    all_results = []
    per_seed_arrays = defaultdict(dict)

    for model_name, cfg in SOTA_MODELS.items():
        for seed in SEEDS:
            ckpt_path, best_thr, best_val_dice = train_one_model(seed, model_name, cfg)
            res = test_and_stats(model_name, seed, ckpt_path, best_thr)
            all_results.append(res)
            per_seed_arrays[seed][model_name] = res["per_image_arrays"]

    # agregacija po modelu
    model2runs = defaultdict(list)
    for res in all_results:
        model2runs[res["model_name"]].append(res)

    table_path = os.path.join(OUT_DIR, "sota_table.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model_name",
            "n_seeds",
            "dice_mean", "dice_std",
            "iou_mean",  "iou_std",
            "prec_mean", "prec_std",
            "rec_mean",  "rec_std"
        ])

        for mname, runs in model2runs.items():
            n = len(runs)
            dice_arr = np.array([r["dice_mean"] for r in runs], dtype=float)
            iou_arr  = np.array([r["iou_mean"]  for r in runs], dtype=float)
            p_arr    = np.array([r["prec_mean"] for r in runs], dtype=float)
            r_arr    = np.array([r["rec_mean"]  for r in runs], dtype=float)

            dice_std = float(dice_arr.std(ddof=1)) if n > 1 else 0.0
            iou_std  = float(iou_arr.std(ddof=1))  if n > 1 else 0.0
            p_std    = float(p_arr.std(ddof=1))    if n > 1 else 0.0
            r_std    = float(r_arr.std(ddof=1))    if n > 1 else 0.0

            w.writerow([
                mname,
                n,
                float(dice_arr.mean()), dice_std,
                float(iou_arr.mean()),  iou_std,
                float(p_arr.mean()),    p_std,
                float(r_arr.mean()),    r_std,
            ])

    print(f"[INFO] SOTA aggregation table written to: {table_path}")

    # (opcionalno) Wilcoxon vs prvog modela u listi
    baseline_name = list(SOTA_MODELS.keys())[0]
    wilcoxon_rows = []

    for seed in SEEDS:
        if baseline_name not in per_seed_arrays[seed]:
            continue

        base = per_seed_arrays[seed][baseline_name]
        base_ids = base["ids"]
        base_map = {img_id: i for i, img_id in enumerate(base_ids)}

        for model_name in SOTA_MODELS:
            if model_name == baseline_name:
                continue
            if model_name not in per_seed_arrays[seed]:
                continue

            var = per_seed_arrays[seed][model_name]

            xs_dice, ys_dice = [], []
            for img_id, di in zip(var["ids"], var["dice"]):
                if img_id in base_map:
                    xs_dice.append(di)
                    ys_dice.append(base["dice"][base_map[img_id]])

            if len(xs_dice) >= 10:
                stat, p, method = paired_wilcoxon(xs_dice, ys_dice)
                wilcoxon_rows.append([seed, f"{model_name}_vs_{baseline_name}", method, stat, p, len(xs_dice)])

    sum_path = os.path.join(OUT_DIR, "sota_wilcoxon.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "model_vs_baseline", "test", "statistic", "p_value", "n_images"])
        for row in wilcoxon_rows:
            w.writerow(row)

    print(f"[DONE] SOTA Wilcoxon summary written to: {sum_path}")


if __name__ == "__main__":
    main()
