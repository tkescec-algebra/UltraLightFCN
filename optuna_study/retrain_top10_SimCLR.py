from __future__ import annotations

import os
import math
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from torchvision import transforms
from PIL import Image

from utils.repro import set_global_seed, GLOBAL_SEED
from utils.dataset import SimCLRSolarPanelDataset
from utils.loss_functions import NTXentLoss
from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel


# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STORAGE = "sqlite:///UltraLightFCN_study.db"
STUDY_NAME = "UltraLightFCN_SimCLR_pretrain_RGB"

DATA_ROOT = "/workspace/UltraLightFCN/dataset/train"  # png images + *_label.png masks exist here
TRAIN_LIST_PATH = "runs/simclr_hpo/pretrain_train_files.txt"
VAL_LIST_PATH   = "runs/simclr_hpo/pretrain_val_files.txt"

SIMCLR_BS = 256
EPOCHS = 40
DROP_LAST = True

OUT_DIR = Path("checkpoints/top10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# kNN settings
KNN_K = 20
KNN_T = 0.07
MIN_POS_PIXELS = 1  # label=1 if mask has >= this many positive pixels

# Reproducibility (match your pipeline)
set_global_seed(GLOBAL_SEED, deterministic=False)
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))


# -----------------------------
# Helpers
# -----------------------------
def _load_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def steps_per_epoch(n: int, bs: int, drop_last: bool) -> int:
    if n <= 0:
        return 0
    if drop_last:
        return max(1, n // bs)
    return max(1, math.ceil(n / bs))


def build_encoder_and_model(proj_hidden_dim: int, proj_out_dim: int) -> Tuple[UltraLightEncoder, SimCLRModel]:
    # Must match your study encoder params
    model_params = {
        'enc_channels': [16, 16, 32, 32, 64],
        'enc_kernel_sizes': [3, 3, 3, 3, 3],
        'enc_strides': [1, 2, 2, 1, 1],
        'dilations': [2, 4],
        'mini_aspp': True,
        'mini_aspp_gpool': True,
        'use_sa': False,
        'sa_windowed': True,
        'sa_window_size': 16,
        'sa_shifted': True,
        'sa_heads': 4,
        'sa_dropout': 0.1,
    }

    encoder = UltraLightEncoder(in_channels=3, params=model_params).to(DEVICE)
    proj_head = ProjectionHead(encoder.out_channels, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim).to(DEVICE)
    model = SimCLRModel(encoder, proj_head).to(DEVICE)
    return encoder, model


# -----------------------------
# kNN Dataset (binary presence label from masks)
# -----------------------------
class KNNPresenceDataset(Dataset):
    """
    (image_tensor, binary_label, filename)
    binary_label derived from mask: *_label.png
    """
    def __init__(self, data_root: str, files: List[str], image_size: int = 256, min_pos_pixels: int = 1):
        self.data_root = data_root
        self.files = files
        self.min_pos_pixels = min_pos_pixels
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def _mask_path(self, img_name: str) -> str:
        stem, ext = os.path.splitext(img_name)
        return os.path.join(self.data_root, f"{stem}_label{ext}")

    def __getitem__(self, idx: int):
        name = self.files[idx]
        img_path = os.path.join(self.data_root, name)
        mask_path = self._mask_path(name)

        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)

        m = Image.open(mask_path).convert("L")
        m_np = np.array(m, dtype=np.uint8)
        pos = int((m_np > 0).sum())
        y = 1 if pos >= self.min_pos_pixels else 0

        return x, y, name


@torch.no_grad()
def compute_embeddings(encoder: torch.nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses encoder features (NOT projection head):
      h = normalize(GAP(encoder_deep_feature))
    """
    encoder.eval()
    feats_list, y_list = [], []
    gap = torch.nn.AdaptiveAvgPool2d(1)

    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        deep, _ = encoder(x)
        h = gap(deep).flatten(1)
        h = F.normalize(h, dim=1)

        feats_list.append(h)
        y_list.append(y)

    return torch.cat(feats_list, dim=0), torch.cat(y_list, dim=0)


@torch.no_grad()
def weighted_knn_binary(
    bank_feats: torch.Tensor,
    bank_labels: torch.Tensor,
    query_feats: torch.Tensor,
    k: int = 20,
    t_knn: float = 0.07
) -> torch.Tensor:
    sims = query_feats @ bank_feats.T
    topk_sims, topk_idx = torch.topk(sims, k=k, dim=1)
    topk_labels = bank_labels[topk_idx]
    w = torch.exp(topk_sims / t_knn)
    wsum = w.sum(dim=1).clamp_min(1e-12)
    pos = (w * (topk_labels == 1).float()).sum(dim=1)
    return pos / wsum


def metrics_from_probs(probs: torch.Tensor, y_true: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (probs >= thr).long()

    tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
    fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true == 0)).sum().item())

    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc}


@torch.no_grad()
def evaluate_val_ratio(model: SimCLRModel, val_loader: DataLoader, crit: NTXentLoss) -> float:
    """
    Computes mean val_ratio over validation:
      val_ratio = NTXentLoss / ln(2B - 1)
    """
    model.eval()
    running = 0.0
    seen = 0

    for xi, xj, *_ in tqdm(val_loader, desc="Val proxy", leave=False):
        B = xi.size(0)
        if B < 2:
            continue

        xi = xi.to(DEVICE, non_blocking=True)
        xj = xj.to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
            zi = model(xi)
            zj = model(xj)
            loss = crit(zi, zj).item()

        baseline = math.log(2 * B - 1)
        ratio = loss / baseline

        running += ratio * B
        seen += B

    return float(running / max(1, seen))


def retrain_and_knn_one(
    trial_number: int,
    optuna_value: float,
    params: Dict[str, Any],
    simclr_train_files: List[str],
    simclr_val_files: List[str],
) -> Dict[str, Any]:
    """
    Retrains one candidate with its hyperparameters, saves encoder checkpoint,
    and immediately runs kNN evaluation using binary labels derived from masks.
    Returns a dict row for CSV logging.
    """
    simclr_lr = float(params["simclr_lr"])
    wd = float(params["weight_decay"])
    temperature = float(params["simclr_temperature"])
    warmup_ratio = float(params["warmup_ratio"])
    proj_hidden_dim = int(params["proj_hidden_dim"])
    proj_out_dim = int(params["proj_out_dim"])
    max_grad_norm = float(params["max_grad_norm"])

    # --- SimCLR datasets/loaders (fixed split lists) ---
    simclr_train_ds = SimCLRSolarPanelDataset(DATA_ROOT, image_size=256, files=simclr_train_files)
    simclr_val_ds   = SimCLRSolarPanelDataset(DATA_ROOT, image_size=256, files=simclr_val_files)

    n_train = len(simclr_train_ds)
    drop_last_trial = DROP_LAST
    if DROP_LAST and n_train < SIMCLR_BS:
        drop_last_trial = False

    train_loader = DataLoader(
        simclr_train_ds,
        batch_size=SIMCLR_BS,
        shuffle=True,
        pin_memory=True,
        drop_last=drop_last_trial,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        simclr_val_ds,
        batch_size=SIMCLR_BS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # --- Scheduling in update-steps ---
    spe = steps_per_epoch(n_train, SIMCLR_BS, drop_last_trial)
    total_steps = EPOCHS * spe
    if total_steps <= 0:
        raise RuntimeError(f"total_steps<=0 for retrain trial {trial_number}")

    warmup_steps = max(1, int(warmup_ratio * total_steps))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    # --- Model/opt/sched/loss ---
    encoder, model = build_encoder_and_model(proj_hidden_dim, proj_out_dim)

    opt = torch.optim.AdamW(model.parameters(), lr=simclr_lr, weight_decay=wd)
    scheduler = CosineLRScheduler(
        optimizer=opt,
        t_initial=total_steps,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    crit = NTXentLoss(temperature=temperature, device=DEVICE)

    # --- Train loop ---
    global_step = 0
    val_ratio_last = None

    for epoch in range(EPOCHS):
        model.train()
        seen = 0

        for xi, xj, *_ in tqdm(train_loader, desc=f"Trial{trial_number} Train {epoch+1}/{EPOCHS}", leave=False):
            B = xi.size(0)
            if B < 2:
                continue

            xi = xi.to(DEVICE, non_blocking=True)
            xj = xj.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
                zi = model(xi)
                zj = model(xj)
                loss = crit(zi, zj)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(opt)
            scaler.update()

            global_step += 1
            scheduler.step_update(num_updates=global_step)

            seen += B

        if seen == 0:
            raise RuntimeError(f"Empty epoch in retrain trial {trial_number}")

        # Optional: compute val proxy each epoch (useful sanity)
        val_ratio_last = evaluate_val_ratio(model, val_loader, crit)
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[Retrain trial {trial_number}] Epoch {epoch+1}/{EPOCHS} | "
            f"lr_now={lr_now:.3e} | val_ratio={val_ratio_last:.4f}"
        )

    # --- Save encoder checkpoint ---
    ckpt_path = str(OUT_DIR / f"trial{trial_number}_encoder.pth")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "trial_number": int(trial_number),
            "optuna_proxy_value": float(optuna_value),
            "params": dict(params),
            "final_val_ratio_retrain": float(val_ratio_last) if val_ratio_last is not None else None,
        },
        ckpt_path,
    )
    print(f"[ckpt] Saved encoder: {ckpt_path}")

    # --- kNN evaluation (binary presence) ---
    train_files = simclr_train_files
    val_files = simclr_val_files

    bank_ds = KNNPresenceDataset(DATA_ROOT, train_files, image_size=256, min_pos_pixels=MIN_POS_PIXELS)
    qry_ds  = KNNPresenceDataset(DATA_ROOT, val_files, image_size=256, min_pos_pixels=MIN_POS_PIXELS)

    bank_loader = DataLoader(bank_ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    qry_loader  = DataLoader(qry_ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    bank_feats, bank_y = compute_embeddings(encoder, bank_loader)
    qry_feats, qry_y   = compute_embeddings(encoder, qry_loader)

    probs = weighted_knn_binary(bank_feats, bank_y, qry_feats, k=KNN_K, t_knn=KNN_T)
    m = metrics_from_probs(probs, qry_y, thr=0.5)

    print(
        f"[kNN trial {trial_number}] F1={m['f1']:.4f} "
        f"P={m['precision']:.4f} R={m['recall']:.4f} Acc={m['accuracy']:.4f}"
    )

    row = {
        "trial_number": int(trial_number),
        "optuna_rank_proxy_value": float(optuna_value),
        "ckpt_path": ckpt_path,
        "knn_f1": float(m["f1"]),
        "knn_precision": float(m["precision"]),
        "knn_recall": float(m["recall"]),
        "knn_accuracy": float(m["accuracy"]),
        "final_val_ratio_retrain": float(val_ratio_last) if val_ratio_last is not None else None,
        "simclr_lr": simclr_lr,
        "weight_decay": wd,
        "simclr_temperature": temperature,
        "warmup_ratio": warmup_ratio,
        "proj_hidden_dim": proj_hidden_dim,
        "proj_out_dim": proj_out_dim,
        "max_grad_norm": max_grad_norm,
    }
    return row


def main():
    # Load top-10 trials from Optuna DB
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    complete = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
    complete.sort(key=lambda t: t.value)  # minimize

    topk = 10
    top_trials = complete[:topk]
    print(f"Loaded {len(complete)} complete trials. Running retrain+kNN for top-{topk}.\n")

    # Fixed split lists
    pretrain_train_files = _load_list(TRAIN_LIST_PATH)
    pretrain_val_files = _load_list(VAL_LIST_PATH)

    # CSV logging
    csv_path = OUT_DIR / "knn_results.csv"
    fieldnames = [
        "trial_number", "optuna_rank_proxy_value", "ckpt_path",
        "knn_f1", "knn_precision", "knn_recall", "knn_accuracy",
        "final_val_ratio_retrain",
        "simclr_lr", "weight_decay", "simclr_temperature", "warmup_ratio",
        "proj_hidden_dim", "proj_out_dim", "max_grad_norm"
    ]

    rows: List[Dict[str, Any]] = []
    for rank, t in enumerate(top_trials, start=1):
        print(f"\n=== Candidate #{rank}/{topk}: trial {t.number} (proxy={t.value:.6f}) ===")
        row = retrain_and_knn_one(
            trial_number=t.number,
            optuna_value=float(t.value),
            params=t.params,
            simclr_train_files=pretrain_train_files,
            simclr_val_files=pretrain_val_files,
        )
        rows.append(row)

        # Write incremental results (safe if interrupted)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # Print current best by kNN-F1
        best = max(rows, key=lambda r: r["knn_f1"])
        print(f"[Current best] trial {best['trial_number']} with kNN-F1={best['knn_f1']:.4f}")

    # Final ranking
    rows_sorted = sorted(rows, key=lambda r: r["knn_f1"], reverse=True)
    print("\n=== Final ranking by kNN-F1 ===")
    for i, r in enumerate(rows_sorted, start=1):
        print(f"#{i:2d} trial {r['trial_number']:3d}  kNN-F1={r['knn_f1']:.4f}  proxy={r['optuna_rank_proxy_value']:.6f}")


if __name__ == "__main__":
    main()
