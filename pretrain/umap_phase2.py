# ============================================================
# UMAP visualization for SimCLR-pretrained encoders
# Shared UMAP space across multiple seeds (paper-safe)
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import umap

from models.UltraLightFCN_SimCLR import UltraLightEncoder
from utils.umap_eval_dataset import UMAPSolarPanelEvalDataset


# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "../dataset"
IMG_DIR   = os.path.join(DATA_ROOT, "test")   # TEST split (recommended)
MASK_DIR  = None

MODEL_NAME = "UltraLightFCN"
CHANNELS   = 3
IMAGE_SIZE = 256

SEEDS = [13, 37, 73]
MAX_SAMPLES = 5000
BATCH_SIZE  = 256
NUM_WORKERS = 0

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "simclr")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "umap_pretrain")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# UMAP PARAMS (FIXED!)
# =========================
UMAP_K        = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC   = "cosine"
UMAP_SEED     = 42


# =========================
# Transforms (NO augmentations)
# =========================
def build_eval_transform(image_size=256):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# =========================
# Encoder builder (same as pretraining)
# =========================
def build_encoder():
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

    if DEVICE.type == "cuda":
        encoder = encoder.to(memory_format=torch.channels_last)

    return encoder


# =========================
# Load pretrained encoder
# =========================
def load_pretrained_encoder(encoder, seed):
    path = os.path.join(
        CKPT_DIR,
        f"{MODEL_NAME}(rgb)-simclr_encoder_final_seed{seed}.pth"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("encoder", ckpt)

    encoder.load_state_dict(state, strict=False)
    encoder.to(DEVICE)
    return path


# =========================
# Extract embeddings
# =========================
@torch.inference_mode()
def extract_embeddings(encoder, loader, max_samples):
    encoder.eval()
    feats, labels = [], []
    seen = 0

    for x, y, *_ in loader:
        x = x.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            x = x.to(memory_format=torch.channels_last)

        feat_map, _ = encoder(x)
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
# Plot helpers
# =========================
def plot_seed_panel(ax, coords, labels, title):
    for val, name in [(0, "No panel"), (1, "Panel")]:
        idx = (labels == val)
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=4,
            alpha=0.9,
            label=name,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


# =========================
# MAIN
# =========================
def main():
    print("[Info] DEVICE:", DEVICE)
    print("[Info] Using TEST split for UMAP")

    tf = build_eval_transform(IMAGE_SIZE)
    ds = UMAPSolarPanelEvalDataset(IMG_DIR, transform=tf, mask_dir=MASK_DIR)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    seed_to_emb, seed_to_lab = {}, {}

    # --------- Extract embeddings ----------
    for seed in SEEDS:
        encoder = build_encoder()
        ckpt = load_pretrained_encoder(encoder, seed)

        emb, lab = extract_embeddings(encoder, loader, MAX_SAMPLES)
        seed_to_emb[seed] = emb
        seed_to_lab[seed] = lab

        print(f"✓ seed={seed} | emb={emb.shape} | panels={(lab==1).sum()} | {os.path.basename(ckpt)}")

    # --------- Shared-space UMAP ----------
    reducer = umap.UMAP(
        n_neighbors=UMAP_K,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_SEED,
        n_jobs=1,
        low_memory=True,
    )

    E_all = np.concatenate([seed_to_emb[s] for s in SEEDS], axis=0)
    E_all = np.ascontiguousarray(E_all, dtype=np.float32)

    coords_all = reducer.fit_transform(E_all)

    coords_by_seed = {}
    start = 0
    for s in SEEDS:
        n = seed_to_emb[s].shape[0]
        coords_by_seed[s] = coords_all[start:start + n]
        start += n

    # --------- Plot shared-space ----------
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(5 * len(SEEDS), 4))
    if len(SEEDS) == 1:
        axes = [axes]

    for ax, s in zip(axes, SEEDS):
        plot_seed_panel(ax, coords_by_seed[s], seed_to_lab[s], f"Pretrain UMAP (seed {s})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("UMAP of SimCLR-pretrained encoder embeddings (shared space)")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    out_path = os.path.join(OUT_DIR, "umap_pretrain_3seeds_sharedspace.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print("✓ Saved:", out_path)

    # --------- Individual UMAPs (optional) ----------
    for s in SEEDS:
        reducer_ind = umap.UMAP(
            n_neighbors=UMAP_K,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=UMAP_SEED,
            n_jobs=1,
            low_memory=True,
        )

        X = np.ascontiguousarray(seed_to_emb[s], dtype=np.float32)
        coords = reducer_ind.fit_transform(X)

        plt.figure(figsize=(6, 5))
        for val, name in [(0, "No panel"), (1, "Panel")]:
            idx = (seed_to_lab[s] == val)
            plt.scatter(coords[idx, 0], coords[idx, 1], s=4, alpha=0.9, label=name)

        plt.legend()
        plt.title(f"UMAP of pretrain encoder embeddings (seed {s})")
        plt.tight_layout()

        out_ind = os.path.join(OUT_DIR, f"umap_pretrain_seed{s}.png")
        plt.savefig(out_ind, dpi=220)
        plt.close()
        print("✓ Saved:", out_ind)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
