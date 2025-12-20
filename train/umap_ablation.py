import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import umap

from models.UltraLightFCN_base import UltraLightFCN
from utils.dataset import SolarPanelDataset
from utils.repro import set_global_seed, seed_worker


# =======================
# CONFIG (match your ablation)
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT   = "../dataset"
SPLIT       = "test"        # "test" recommended for paper; can use "valid" for analysis
MODE        = "test"        # SolarPanelDataset mode: train/val/test

MODEL_NAME  = "UltraLightFCN_base"
EDGE_DETECTOR = None
CHANNELS    = 3

ENCODER_MODE = "finetune"     # "frozen" or "finetune"
OUT_DIR     = os.path.join("ablation_outputs", ENCODER_MODE, "umap")
os.makedirs(OUT_DIR, exist_ok=True)

# Windows: set NUM_WORKERS=0
NUM_WORKERS = 0
BATCH_SIZE  = 8
MAX_SAMPLES = 2000          # keep moderate (variants * seeds * samples)

VARIANTS = [
    "baseline",
    "mini_aspp_off",
    "no_sa",
    "mini_aspp_no_gpool",
    "sa_ws_8",
    "sa_ws_32",
    "loss_03_07",
    "loss_05_05",
]

SEEDS = [42, 52, 62]

# UMAP params
UMAP_K = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_SEED = 42

# Optional: also produce 3-seed shared-space plot for a chosen variant (like pretrain)
MAKE_VARIANT_ACROSS_SEEDS = True
ACROSS_SEEDS_VARIANT = "baseline"


# =======================
# MODEL PARAMS (same function as ablation)
# =======================
def build_ultralight_params(variant_cfg: dict):
    return {
        'enc_channels':     [16, 16, 32, 32, 64],
        'enc_kernel_sizes': [3,  3,  3,  3,  3],
        'enc_strides':      [1,  2,  2,  1,  1],
        'dilations':        [2,  4],

        'dec_channels':     [32, 16, 16],
        'dec_kernel_sizes': [3,  3],
        'dec_strides':      [1,  1],
        'upscale':          [2,  2],

        'mini_aspp':        variant_cfg['mini_aspp'],
        'mini_aspp_gpool':  variant_cfg['mini_aspp_gpool'],
        'use_sa':           variant_cfg['use_sa'],
        'sa_windowed':      True,
        'sa_window_size':   variant_cfg['sa_window_size'],
        'sa_shifted':       True,
        'sa_heads':         4,
        'sa_dropout':       0.0,
    }


VARIANT_CFGS = {
    "baseline":           dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=16),
    "mini_aspp_off":      dict(mini_aspp=False, mini_aspp_gpool=False, use_sa=True,  sa_window_size=16),
    "no_sa":              dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=False, sa_window_size=16),
    "mini_aspp_no_gpool": dict(mini_aspp=True,  mini_aspp_gpool=False, use_sa=True,  sa_window_size=16),
    "sa_ws_8":            dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=8),
    "sa_ws_32":           dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=32),
    "loss_03_07":         dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=16),
    "loss_05_05":         dict(mini_aspp=True,  mini_aspp_gpool=True,  use_sa=True,  sa_window_size=16),
}


# =======================
# DATA
# =======================
def make_loader(seed: int):
    ds = SolarPanelDataset(
        data_dir=f"{DATA_ROOT}/{SPLIT}",
        mode=MODE,
        edge_detector=EDGE_DETECTOR,
        channels=CHANNELS,
    )

    # With shuffle=False, generator is not strictly required,
    # but kept for full determinism if you ever toggle shuffle.
    g = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,              # IMPORTANT: fixed order
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    return ds, loader


# =======================
# EMBEDDING EXTRACTION via forward hook
# =======================
class FeatureHook:
    def __init__(self):
        self.feat = None

    def __call__(self, module, inp, out):
        self.feat = out


@torch.inference_mode()
def extract_embeddings(model, loader, max_samples: int):
    """
    Captures feature map from the last encoder block (model.dilconv5) and computes:
      h = GAP(feature_map) -> normalize
    Label is derived from mask coverage:
      y_bin = 1 if mask has any positives else 0
    Also returns coverage (fraction of mask positives) for optional analysis.
    """
    model.eval()

    hook = FeatureHook()
    if not hasattr(model, "dilconv5"):
        raise AttributeError("Model has no attribute 'dilconv5'. Update hook point or expose encoder features.")
    handle = model.dilconv5.register_forward_hook(hook)

    feats = []
    ybin = []
    ycov = []
    seen = 0

    for images, masks, *_ in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        if DEVICE.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        _ = model(images)  # forward populates hook.feat

        feat_map = hook.feat
        if feat_map is None:
            handle.remove()
            raise RuntimeError("Hook did not capture features. Check that model.dilconv5 exists and is used.")

        # GAP: (B,C,H,W) -> (B,C)
        h = feat_map.mean(dim=(2, 3))
        h = F.normalize(h, dim=1)

        # Coverage label (mask presence)
        cov = masks.float().mean(dim=(1, 2, 3))  # (B,)
        y_c = cov.detach().cpu().numpy()
        y_b = (cov > 0).long().detach().cpu().numpy()

        feats.append(h.detach().cpu().numpy())
        ycov.append(y_c)
        ybin.append(y_b)

        seen += images.size(0)
        if seen >= max_samples:
            break

    handle.remove()

    emb = np.concatenate(feats, axis=0)[:max_samples]
    lab_bin = np.concatenate(ybin, axis=0)[:max_samples]
    lab_cov = np.concatenate(ycov, axis=0)[:max_samples]
    return emb, lab_bin, lab_cov


# =======================
# CKPT PATH helper
# =======================
def ckpt_path_for(variant_name: str, seed: int):
    ckpt_dir = os.path.join("ablation_outputs", ENCODER_MODE, "train_models", variant_name, f"seed_{seed}")
    fname = f"{MODEL_NAME}({'rgb' if CHANNELS==3 else 'edge'})-seed_{seed}-{ENCODER_MODE}.pth"
    return os.path.join(ckpt_dir, fname)


def load_model_for_variant_seed(variant_name: str, seed: int):
    path = ckpt_path_for(variant_name, seed)
    if not os.path.exists(path):
        return None, path

    params = build_ultralight_params(VARIANT_CFGS[variant_name])
    model = UltraLightFCN(in_channels=CHANNELS, num_classes=1, params=params).to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE)
    return model, path


# =======================
# PLOTTING
# =======================
def plot_grid(coords_by_variant, lab_bin, title, out_path, ncols=4):
    variants = list(coords_by_variant.keys())
    n = len(variants)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.8*nrows))
    axes = np.array(axes).reshape(-1)

    for i, v in enumerate(variants):
        ax = axes[i]
        coords = coords_by_variant[v]

        for val, name in [(0, "No panel"), (1, "Panel")]:
            idx = (lab_bin == val)
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                s=4, alpha=0.9, label=name, rasterized=True
            )

        ax.set_title(v)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_3seeds_variant_shared(coords_by_seed, lab_bin, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, seed in zip(axes, sorted(coords_by_seed.keys())):
        coords = coords_by_seed[seed]
        for val, name in [(0, "No panel"), (1, "Panel")]:
            idx = (lab_bin == val)
            ax.scatter(coords[idx, 0], coords[idx, 1], s=4, alpha=0.9, label=name, rasterized=True)
        ax.set_title(f"seed {seed}")
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# =======================
# Main
# =======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[Info] OUT_DIR:", os.path.abspath(OUT_DIR))
    print("[Info] DEVICE:", DEVICE)
    print("[Info] SPLIT:", SPLIT, "| MODE:", MODE, "| ENCODER_MODE:", ENCODER_MODE)

    # One dataset/loader reused for all models to ensure identical sample order
    ds, loader = make_loader(seed=123)
    print("[Info] Dataset size:", len(ds))

    # -----------------------
    # A) Per-seed: variants in shared UMAP space (your original intention)
    # -----------------------
    for seed in SEEDS:
        set_global_seed(seed, deterministic=True)

        emb_by_variant = {}
        lab_by_variant = {}
        ref_lab_bin = None

        # 1) Extract embeddings for each variant
        for vname in VARIANTS:
            model, path = load_model_for_variant_seed(vname, seed)
            if model is None:
                print(f"[WARN] Missing ckpt: {path}")
                continue

            emb, lab_bin, lab_cov = extract_embeddings(model, loader, max_samples=MAX_SAMPLES)
            emb_by_variant[vname] = emb
            lab_by_variant[vname] = lab_bin

            if ref_lab_bin is None:
                ref_lab_bin = lab_bin
            else:
                if not np.array_equal(ref_lab_bin, lab_bin):
                    print(f"[ERROR] Label/order mismatch across variants for seed={seed}. "
                          f"Skipping this seed to avoid invalid visualization.")
                    emb_by_variant = {}
                    break

            print(f"✓ seed={seed} | {vname:16s} | emb={emb.shape} | panels={(lab_bin==1).sum()} | ckpt={os.path.basename(path)}")

        if len(emb_by_variant) < 2:
            print(f"[WARN] Not enough valid variants for seed {seed}, skipping plot.")
            continue

        # 2) One shared UMAP fit on concatenated embeddings
        reducer = umap.UMAP(
            n_neighbors=UMAP_K,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=UMAP_SEED,
        )

        variants_sorted = [v for v in VARIANTS if v in emb_by_variant]

        E_all = np.concatenate([emb_by_variant[v] for v in variants_sorted], axis=0)
        coords_all = reducer.fit_transform(E_all)

        # 3) Split coords back per variant (ROBUST slicing)
        sizes = [emb_by_variant[v].shape[0] for v in variants_sorted]
        coords_by_variant = {}
        start = 0
        for v, n_v in zip(variants_sorted, sizes):
            coords_by_variant[v] = coords_all[start:start + n_v]
            start += n_v

        # 4) Plot grid
        out_path = os.path.join(OUT_DIR, f"umap_variants_sharedspace_seed{seed}_{ENCODER_MODE}.png")
        title = f"UMAP (shared space) | seed={seed} | encoder_mode={ENCODER_MODE} | split={SPLIT}"
        plot_grid(coords_by_variant, ref_lab_bin, title, out_path, ncols=4)
        print("✓ Saved:", out_path)

    # -----------------------
    # B) Optional: one variant across 3 seeds in a shared UMAP space (like pretrain)
    # -----------------------
    if MAKE_VARIANT_ACROSS_SEEDS:
        vname = ACROSS_SEEDS_VARIANT
        print(f"[Info] Building across-seeds shared UMAP for variant='{vname}'")

        emb_by_seed = {}
        lab_by_seed = {}
        ref_lab_bin = None

        for seed in SEEDS:
            set_global_seed(seed, deterministic=True)

            model, path = load_model_for_variant_seed(vname, seed)
            if model is None:
                print(f"[WARN] Missing ckpt: {path}")
                continue

            emb, lab_bin, _ = extract_embeddings(model, loader, max_samples=MAX_SAMPLES)
            emb_by_seed[seed] = emb
            lab_by_seed[seed] = lab_bin

            if ref_lab_bin is None:
                ref_lab_bin = lab_bin
            else:
                if not np.array_equal(ref_lab_bin, lab_bin):
                    print("[ERROR] Label/order mismatch across seeds. Skipping across-seeds plot.")
                    emb_by_seed = {}
                    break

            print(f"✓ variant={vname} | seed={seed} | emb={emb.shape}")

        if len(emb_by_seed) >= 2:
            reducer = umap.UMAP(
                n_neighbors=UMAP_K,
                min_dist=UMAP_MIN_DIST,
                metric=UMAP_METRIC,
                random_state=UMAP_SEED,
            )

            seeds_sorted = sorted(emb_by_seed.keys())
            E_all = np.concatenate([emb_by_seed[s] for s in seeds_sorted], axis=0)
            coords_all = reducer.fit_transform(E_all)

            # Robust slicing per seed
            sizes = [emb_by_seed[s].shape[0] for s in seeds_sorted]
            coords_by_seed = {}
            start = 0
            for s, n_s in zip(seeds_sorted, sizes):
                coords_by_seed[s] = coords_all[start:start + n_s]
                start += n_s

            out_path = os.path.join(OUT_DIR, f"umap_{vname}_3seeds_sharedspace_{ENCODER_MODE}.png")
            title = f"UMAP (shared space) | variant={vname} | encoder_mode={ENCODER_MODE} | split={SPLIT}"
            plot_3seeds_variant_shared(coords_by_seed, ref_lab_bin, title, out_path)
            print("✓ Saved:", out_path)
        else:
            print("[WARN] Not enough seeds available for across-seeds plot.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
