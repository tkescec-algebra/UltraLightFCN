import os

# ---- Windows / numba / OpenMP stability ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import umap

from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel
from utils.umap_eval_dataset import UMAPSolarPanelEvalDataset


# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "../dataset"
IMG_DIR   = os.path.join(DATA_ROOT, "test")
MASK_DIR  = None

CSV_PATH = "pairwise_results/simclr_pairwise_results.csv"
CKPT_DIR = "pairwise_results/checkpoints"
OUT_DIR  = "pairwise_results/figures/umap"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "UltraLightFCN"
IMAGE_SIZE = 256
CHANNELS   = 3

MAX_SAMPLES = 5000
BATCH_SIZE  = 256

# Keep fixed for reproducibility
GLOBAL_SEED = 42
NUM_WORKERS = 0   # Windows-safe

# =========================
# UMAP PARAMS (MATCH PRETRAIN!)
# =========================
UMAP_K        = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC   = "cosine"
UMAP_SEED     = 42


# =========================
# Transforms (NO augmentations)
# =========================
def build_umap_eval_transform(image_size=256):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# =========================
# SimCLR model builder
# =========================
def build_simclr_model():
    encoder = UltraLightEncoder(
        in_channels=CHANNELS,
        params={
            "enc_channels": [16, 16, 32, 32, 64],
            "enc_kernel_sizes": [3, 3, 3, 3, 3],
            "enc_strides": [1, 2, 2, 1, 1],
            "dilations": [2, 4],
            "mini_aspp": True,
            "mini_aspp_gpool": True,
            "use_sa": False,
        },
    ).to(DEVICE)

    proj = ProjectionHead(
        in_dim=encoder.out_channels,
        hidden_dim=128,
        out_dim=64,
    ).to(DEVICE)

    model = SimCLRModel(encoder, proj).to(DEVICE)

    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    return model


# =========================
# Embedding extraction (encoder only)
# =========================
@torch.inference_mode()
def extract_encoder_embeddings_and_labels(model, loader, max_samples: int):
    model.eval()
    feats, labels = [], []
    seen = 0

    for x, y, *_ in loader:
        x = x.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            x = x.to(memory_format=torch.channels_last)

        feat_map, _ = model.encoder(x)
        h = feat_map.mean(dim=(2, 3))
        h = F.normalize(h, dim=1)

        feats.append(h.cpu().numpy())
        labels.append(y.numpy())

        seen += x.size(0)
        if seen >= max_samples:
            break

    emb = np.concatenate(feats, axis=0)[:max_samples]
    lab = np.concatenate(labels, axis=0)[:max_samples]
    return emb, lab


# =========================
# UMAP plotting (STABLE)
# =========================
def umap_plot(emb, lab, out_png):
    reducer = umap.UMAP(
        n_neighbors=UMAP_K,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_SEED,
        n_jobs=1,
        low_memory=True,
    )

    # CRITICAL: dtype + contiguous
    X = np.ascontiguousarray(emb, dtype=np.float32)
    coords = reducer.fit_transform(X)

    plt.figure(figsize=(7, 6))
    for val, name in [(0, "No panel"), (1, "Panel")]:
        idx = (lab == val)
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=4,
            alpha=0.9,
            label=name,
            rasterized=True,
        )

    plt.legend()
    plt.title("UMAP of SimCLR encoder embeddings")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    print("[Info] DEVICE:", DEVICE)

    # --------- Load ranking CSV ----------
    df = pd.read_csv(CSV_PATH)

    # Ranking logic (same as before)
    df["r_loss"]  = df["loss"].rank(ascending=True)
    df["r_align"] = df["alignment"].rank(ascending=True)
    df["r_uni"]   = df["uniformity"].rank(ascending=True)

    target = df["ssim"].median()
    df["r_ssim"] = (df["ssim"] - target).abs().rank(ascending=True)

    df["rank_score"] = (df["r_loss"] + df["r_align"] + df["r_uni"] + df["r_ssim"]) / 4.0
    df = df.sort_values("rank_score", ascending=True)

    topk = df.head(3)
    print("[Info] Top-k pairs:")
    print(topk[["t1", "t2", "rank_score"]])

    # --------- Dataset & loader ----------
    tf = build_umap_eval_transform(IMAGE_SIZE)
    ds = UMAPSolarPanelEvalDataset(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        transform=tf,
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------- Process top-k pairs ----------
    for _, row in topk.iterrows():
        t1, t2 = row["t1"], row["t2"]
        seed   = int(row["seed"])

        ckpt_path = os.path.join(
            CKPT_DIR,
            f"simclr_{MODEL_NAME}_seed{seed}_{t1}+{t2}_final.pth",
        )

        if not os.path.exists(ckpt_path):
            print(f"[WARN] Missing checkpoint: {ckpt_path}")
            continue

        model = build_simclr_model()
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
        model.proj_head.load_state_dict(ckpt["proj_head_state_dict"], strict=True)
        model.to(DEVICE)

        emb, lab = extract_encoder_embeddings_and_labels(
            model, loader, MAX_SAMPLES
        )

        out_png = os.path.join(OUT_DIR, f"umap_pair_{t1}_{t2}.png")
        umap_plot(emb, lab, out_png)

        print(f"✓ Saved UMAP: {out_png} | pair={t1}+{t2} | N={emb.shape[0]}")


if __name__ == "__main__":
    main()
