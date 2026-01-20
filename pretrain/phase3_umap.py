"""phase3_umap.py

UMAP visualization of Phase 3 SimCLR representations (paper-grade, Phase-3 aligned).

Alignment with Phase 3
- Uses ONLY Phase 3 LAST checkpoint (no checkpoint selection).
- Uses deterministic, persisted file lists (Phase-3 style).
- Uses canonical deterministic preprocessing for embedding extraction (no SimCLR augmentations).

Outputs
- Two spaces: pooled (encoder->GAP) and proj_z (projection head output, L2-normalized).
- Optional PCA -> UMAP(metric='cosine') with small seed/param sweep.
- Paper-trail artifacts: file list, configs, coords, plots, kNN purity summary.

"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt

# Project imports (keep consistent with Phase 3)
from utils.repro import set_global_seed

# If your repo uses different module paths, adjust later.
from models.UltraLightFCN_SimCLR import UltraLightEncoder, ProjectionHead, SimCLRModel


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class UMAPConfig:
    # Paths
    train_dir: Path = Path("../dataset/train")
    phase3_ckpt_path: Path = Path("checkpoints/simclr_phase3/phase3_seed13_last.pth")

    # Output
    out_dir: Path = Path("runs/umap_phase3")
    run_tag: str = "phase3_umap"

    # Sampling
    embed_all_train: bool = True
    sample_n: int = 10_000
    sample_seed: int = 13

    # Data loading
    image_size: int = 256
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True

    # Embedding extraction
    l2_normalize_pooled: bool = True  # helps cosine UMAP comparability
    amp: bool = True

    # UMAP / PCA
    use_pca: bool = True
    pca_dim: int = 50
    umap_metric: str = "cosine"
    umap_n_neighbors: Tuple[int, ...] = (30,)
    umap_min_dist: Tuple[float, ...] = (0.1,)
    umap_seeds: Tuple[int, ...] = (13, 17, 23)

    # kNN purity
    knn_k: int = 15

    # Which label views to plot
    plot_by_pv_group: bool = True
    plot_by_has_panel: bool = True
    plot_by_area_bins: bool = True

    # Plot styling / variants (paper-grade readability)
    point_size: float = 6.0
    alpha_all: float = 0.25

    # For area_bin overlay plots: draw background (0) faintly, positives on top
    alpha_bg_zero: float = 0.05
    alpha_fg_pos: float = 0.45

    # Additional plots to reduce overplotting and improve interpretability
    plot_area_ratio_continuous: bool = True     # color by continuous area_ratio (positives-only)
    plot_area_bin_pos_only: bool = True         # plot only area_bin > 0
    plot_area_bin_overlay: bool = True          # overlay: 0 faint + positives colored

    # Per-PV subgroup plots (helps diagnose confounding)
    plot_area_ratio_per_pv_group: bool = True
    pv_groups_for_area_ratio: Tuple[str, ...] = ("PV01", "PV03", "PV08", "OTHER")
    min_points_per_group: int = 300


CFG = UMAPConfig()


# -------------------------
# Utilities
# -------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def ensure_deps() -> None:
    """Fail fast with clear error if optional deps are missing."""
    try:
        import umap  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: umap-learn. Install with: pip install umap-learn") from e

    try:
        import sklearn  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: scikit-learn. Install with: pip install scikit-learn") from e


def infer_pv_group(filename: str) -> str:
    """Infer PV subset label from filename prefix."""
    base = Path(filename).name
    up = base.upper()
    for key in ("PV01", "PV03", "PV08"):
        if up.startswith(key):
            return key
    return "OTHER"


def list_images_no_masks_sorted(folder: Path) -> List[str]:
    """Deterministic sorted list excluding *_label.png masks."""
    files: List[str] = []
    for fp in folder.iterdir():
        if not fp.is_file():
            continue
        name = fp.name
        lname = name.lower()
        if not lname.endswith(IMG_EXTS):
            continue
        if lname.endswith("_label.png"):
            continue
        files.append(name)
    files.sort()
    return files


def mask_path_for_image(train_dir: Path, image_filename: str) -> Path:
    """Assume mask file ends with _label.png."""
    stem = Path(image_filename).stem
    return train_dir / f"{stem}_label.png"


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_phase3_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 3 checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unexpected checkpoint format (not a dict)")
    return ckpt


def infer_proj_dims_from_state_dict(proj_sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """Infer (in_dim, hidden_dim, out_dim) from ProjectionHead state_dict."""
    w0 = None
    w2 = None
    for k, v in proj_sd.items():
        if k.endswith("net.0.weight"):
            w0 = v
        if k.endswith("net.2.weight"):
            w2 = v
    if w0 is None or w2 is None:
        raise RuntimeError("Cannot infer projection dims from proj_head_state_dict")
    hidden_dim, in_dim = int(w0.shape[0]), int(w0.shape[1])
    out_dim, hidden2 = int(w2.shape[0]), int(w2.shape[1])
    if hidden2 != hidden_dim:
        raise RuntimeError("Projection head dims mismatch in state dict")
    return in_dim, hidden_dim, out_dim


def area_ratio_to_bin(area_ratio: Optional[float]) -> str:
    if area_ratio is None or (isinstance(area_ratio, float) and math.isnan(area_ratio)):
        return "NA"
    if area_ratio <= 0.0:
        return "0"
    if area_ratio <= 0.01:
        return "(0-1%]"
    if area_ratio <= 0.05:
        return "(1-5%]"
    if area_ratio <= 0.20:
        return "(5-20%]"
    return ">20%"


# -------------------------
# Dataset for canonical embedding extraction
# -------------------------
class CanonicalEmbedDataset(Dataset):
    def __init__(self, data_dir: Path, files: List[str], image_size: int):
        self.data_dir = data_dir
        self.files = list(files)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def _load_mask_stats(self, img_filename: str) -> Tuple[Optional[int], Optional[float]]:
        mp = mask_path_for_image(self.data_dir, img_filename)
        if not mp.exists():
            return None, None
        m = Image.open(mp).convert("L")
        m = m.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        arr = np.asarray(m)
        has_panel = int(np.any(arr > 0))
        area_ratio = float(np.mean(arr > 0))
        return has_panel, area_ratio

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        x = self._load_image(self.data_dir / fn)
        pv = infer_pv_group(fn)
        has_panel, area_ratio = self._load_mask_stats(fn)
        return x, fn, pv, has_panel, area_ratio


# -------------------------
# Embedding extraction
# -------------------------
@torch.no_grad()
def extract_embeddings(
    encoder: UltraLightEncoder,
    proj_head: ProjectionHead,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    l2_normalize_pooled: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return pooled (N,64), z (N,proj_out_dim), and metadata DF."""

    encoder.eval()
    proj_head.eval()

    pooled_list: List[np.ndarray] = []
    z_list: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []

    gap = torch.nn.AdaptiveAvgPool2d(1)
    use_amp = bool(amp and device.type == "cuda")

    for xb, fn, pv, has_panel, area_ratio in loader:
        xb = xb.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            feat, _ = encoder(xb)
            pooled = gap(feat).view(xb.size(0), -1)
            if l2_normalize_pooled:
                pooled_n = torch.nn.functional.normalize(pooled, dim=1)
            else:
                pooled_n = pooled

            z = proj_head(pooled)
            z = torch.nn.functional.normalize(z, dim=1)

        pooled_list.append(pooled_n.detach().cpu().numpy())
        z_list.append(z.detach().cpu().numpy())

        for i in range(len(fn)):
            ar = None if area_ratio[i] is None else float(area_ratio[i])
            hp = None if has_panel[i] is None else int(has_panel[i])
            meta_rows.append(
                {
                    "filename": str(fn[i]),
                    "pv_group": str(pv[i]),
                    "has_panel": hp,
                    "area_ratio": ar,
                    "area_bin": area_ratio_to_bin(ar),
                }
            )

    pooled_arr = np.concatenate(pooled_list, axis=0) if pooled_list else np.zeros((0, 64), dtype=np.float32)
    z_arr = np.concatenate(z_list, axis=0) if z_list else np.zeros((0, 64), dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    return pooled_arr, z_arr, meta_df


# -------------------------
# UMAP + plotting
# -------------------------
def run_umap(
    X: np.ndarray,
    use_pca: bool,
    pca_dim: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ensure_deps()

    from sklearn.decomposition import PCA
    import umap

    X_in = X
    pca_meta: Dict[str, Any] = {"use_pca": False, "pca_dim": None}

    if use_pca:
        d = int(min(pca_dim, X_in.shape[1]))
        pca = PCA(n_components=d, random_state=seed)
        X_in = pca.fit_transform(X_in)
        pca_meta = {
            "use_pca": True,
            "pca_dim": d,
            "pca_explained_var_sum": float(np.sum(pca.explained_variance_ratio_)),
        }

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(seed),
    )

    coords = reducer.fit_transform(X_in)

    meta = {
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": str(metric),
        "seed": int(seed),
        **pca_meta,
    }
    return coords.astype(np.float32), meta


def _make_palette(categories: List[str], cmap_name: str = "tab10") -> Dict[str, Any]:
    cmap = plt.get_cmap(cmap_name)
    pal: Dict[str, Any] = {}
    for i, c in enumerate(categories):
        pal[c] = cmap(i % cmap.N)
    return pal


def plot_umap_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    out_path: Path,
    categorical: bool,
    *,
    palette: Optional[Dict[str, Any]] = None,
    order: Optional[List[str]] = None,
    alpha: float = 0.25,
    point_size: float = 6.0,
    legend_title: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    if categorical:
        cats = df[color_col].fillna("NA").astype(str)
        uniq = order if order is not None else sorted(cats.unique().tolist())
        if palette is None:
            palette = _make_palette(uniq, cmap_name="tab10")

        for c in uniq:
            m = (cats == c).to_numpy()
            if m.sum() == 0:
                continue
            ax.scatter(
                df.loc[m, x_col],
                df.loc[m, y_col],
                s=point_size,
                alpha=alpha,
                c=[palette.get(c)],
                label=c,
                linewidths=0,
            )

        ax.legend(
            title=(legend_title if legend_title is not None else color_col),
            loc="best",
            frameon=True,
            fontsize=8,
        )

    else:
        vals = df[color_col].to_numpy(dtype=np.float32)
        sc = ax.scatter(df[x_col], df[y_col], c=vals, s=point_size, alpha=alpha, linewidths=0)
        fig.colorbar(sc, ax=ax, label=color_col)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_area_bin_suite(
    df_plot: pd.DataFrame,
    space_name: str,
    nn: int,
    md: float,
    us: int,
    out_dir: Path,
    tag: str,
) -> None:
    # Stable order for bins
    bin_order = ["0", "(0-1%]", "(1-5%]", "(5-20%]", ">20%", "NA"]
    palette = _make_palette(bin_order, cmap_name="tab10")

    # Overlay: draw 0 as faint grey, positives on top in palette colors
    if CFG.plot_area_bin_overlay:
        df0 = df_plot[df_plot["area_bin"].astype(str) == "0"].copy()
        dfp = df_plot[df_plot["area_bin"].astype(str).isin(["(0-1%]", "(1-5%]", "(5-20%]", ">20%"])]

        fig, ax = plt.subplots(figsize=(10, 8))

        if len(df0) > 0:
            ax.scatter(
                df0["umap_x"],
                df0["umap_y"],
                s=CFG.point_size,
                alpha=CFG.alpha_bg_zero,
                c=["#808080"],
                label="0",
                linewidths=0,
            )

        for c in ["(0-1%]", "(1-5%]", "(5-20%]", ">20%"]:
            m = (dfp["area_bin"].astype(str) == c)
            if m.sum() == 0:
                continue
            ax.scatter(
                dfp.loc[m, "umap_x"],
                dfp.loc[m, "umap_y"],
                s=CFG.point_size,
                alpha=CFG.alpha_fg_pos,
                c=[palette[c]],
                label=c,
                linewidths=0,
            )

        ax.legend(title="area_bin", loc="best", frameon=True, fontsize=8)
        ax.set_title(f"UMAP({space_name}) area_bin overlay | nn={nn} md={md} seed={us} | N={len(df_plot)}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.tight_layout()

        p = out_dir / f"plot_umap_{tag}_area_bin_overlay"
        fig.savefig(p.with_suffix(".png"), dpi=200)
        fig.savefig(p.with_suffix(".pdf"))
        plt.close(fig)

    # Positives-only bins
    if CFG.plot_area_bin_pos_only:
        df_pos = df_plot[df_plot["area_bin"].astype(str).isin(["(0-1%]", "(1-5%]", "(5-20%]", ">20%"])]
        if len(df_pos) > 0:
            plot_umap_scatter(
                df=df_pos,
                x_col="umap_x",
                y_col="umap_y",
                color_col="area_bin",
                title=f"UMAP({space_name}) positives only (area bins) | nn={nn} md={md} seed={us} | N={len(df_pos)}",
                out_path=out_dir / f"plot_umap_{tag}_area_bin_pos_only",
                categorical=True,
                palette=palette,
                order=["(0-1%]", "(1-5%]", "(5-20%]", ">20%"],
                alpha=CFG.alpha_all,
                point_size=CFG.point_size,
                legend_title="area_bin",
            )

    # Continuous area_ratio (best on positives)
    if CFG.plot_area_ratio_continuous:
        df_pos = df_plot[df_plot["has_panel"].astype(float) == 1].copy()
        df_pos = df_pos[df_pos["area_ratio"].notna()].copy()
        if len(df_pos) > 0:
            plot_umap_scatter(
                df=df_pos,
                x_col="umap_x",
                y_col="umap_y",
                color_col="area_ratio",
                title=f"UMAP({space_name}) positives only (area_ratio) | nn={nn} md={md} seed={us} | N={len(df_pos)}",
                out_path=out_dir / f"plot_umap_{tag}_area_ratio_pos_only",
                categorical=False,
                alpha=CFG.alpha_all,
                point_size=CFG.point_size,
            )


def knn_purity(X: np.ndarray, labels: List[Any], k: int, metric: str = "cosine") -> float:
    """Neighborhood agreement: average fraction of kNN sharing the same label."""
    ensure_deps()
    from sklearn.neighbors import NearestNeighbors

    lab = np.array(labels, dtype=object)
    valid = np.array([l is not None and str(l) != "NA" for l in lab], dtype=bool)
    if valid.sum() < max(10, k + 1):
        return float("nan")

    Xv = X[valid]
    lv = lab[valid]

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(Xv)), metric=metric)
    nbrs.fit(Xv)
    inds = nbrs.kneighbors(return_distance=False)

    pur: List[float] = []
    for i in range(len(Xv)):
        neigh = inds[i, 1:]
        pur.append(float(np.mean(lv[neigh] == lv[i])))

    return float(np.mean(pur))


def plot_area_ratio_per_pv_group(
    df_plot: pd.DataFrame,
    space_name: str,
    nn: int,
    md: float,
    us: int,
    out_dir: Path,
    tag: str,
) -> None:
    if not CFG.plot_area_ratio_per_pv_group:
        return

    for g in CFG.pv_groups_for_area_ratio:
        df_g = df_plot[(df_plot["pv_group"].astype(str) == str(g)) & (df_plot["has_panel"].astype(float) == 1)].copy()
        df_g = df_g[df_g["area_ratio"].notna()].copy()
        if len(df_g) < CFG.min_points_per_group:
            continue

        plot_umap_scatter(
            df=df_g,
            x_col="umap_x",
            y_col="umap_y",
            color_col="area_ratio",
            title=f"UMAP({space_name}) {g} positives (area_ratio) | nn={nn} md={md} seed={us} | N={len(df_g)}",
            out_path=out_dir / f"plot_umap_{tag}_area_ratio_{g}_pos_only",
            categorical=False,
            alpha=CFG.alpha_all,
            point_size=CFG.point_size,
        )


# -------------------------
# Main
# -------------------------
def main() -> None:
    set_global_seed(CFG.sample_seed, deterministic=False, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = CFG.out_dir / CFG.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------- file list (Phase-3 style) ---------
    all_files = list_images_no_masks_sorted(CFG.train_dir)
    if len(all_files) == 0:
        raise RuntimeError(f"No images found in: {CFG.train_dir}")

    if CFG.embed_all_train:
        files = all_files
        sample_note = "all_train"
    else:
        rng = random.Random(CFG.sample_seed)
        buckets: Dict[str, List[str]] = {}
        for f in all_files:
            buckets.setdefault(infer_pv_group(f), []).append(f)
        for b in buckets.values():
            rng.shuffle(b)

        total = len(all_files)
        files = []
        for _, b in buckets.items():
            n_g = max(1, int(round(CFG.sample_n * (len(b) / total))))
            files.extend(b[:n_g])
        rng.shuffle(files)
        files = files[: CFG.sample_n]
        sample_note = f"sample{len(files)}_seed{CFG.sample_seed}"

    file_list_path = out_dir / f"umap_files_{sample_note}.txt"
    file_list_path.write_text("\n".join(files), encoding="utf-8")

    # --------- load checkpoint (Phase 3 LAST) ---------
    ckpt = load_phase3_checkpoint(CFG.phase3_ckpt_path)
    meta = ckpt.get("meta", {})

    encoder_params = None
    try:
        cfg_meta = meta.get("config", {}) if isinstance(meta, dict) else {}
        encoder_params = cfg_meta.get("encoder_params", None)
    except Exception:
        encoder_params = None

    simclr_hp = meta.get("simclr_hp", {}) if isinstance(meta, dict) else {}

    if "proj_hidden_dim" in simclr_hp and "proj_out_dim" in simclr_hp:
        proj_hidden_dim = int(simclr_hp["proj_hidden_dim"])
        proj_out_dim = int(simclr_hp["proj_out_dim"])
        in_dim = 64
    else:
        proj_sd = ckpt.get("proj_head_state_dict", None)
        if proj_sd is None:
            raise RuntimeError("Checkpoint missing proj_head_state_dict; cannot build projection head")
        in_dim, proj_hidden_dim, proj_out_dim = infer_proj_dims_from_state_dict(proj_sd)

    encoder = UltraLightEncoder(in_channels=3, params=encoder_params).to(device)
    if int(getattr(encoder, "out_channels", 64)) != 64:
        raise RuntimeError(f"Expected encoder.out_channels=64 for SimCLR recipe, got {getattr(encoder, 'out_channels', None)}")

    proj_head = ProjectionHead(in_dim=in_dim, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim).to(device)

    # Load weights
    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    elif "model_state_dict" in ckpt:
        # fallback: load through a full SimCLRModel
        tmp_model = SimCLRModel(encoder=encoder, proj_head=proj_head).to(device)
        tmp_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        raise RuntimeError("Checkpoint missing encoder_state_dict and model_state_dict")

    if "proj_head_state_dict" in ckpt:
        proj_head.load_state_dict(ckpt["proj_head_state_dict"], strict=True)

    # --------- dataset/loader ---------
    ds = CanonicalEmbedDataset(CFG.train_dir, files=files, image_size=CFG.image_size)
    g = torch.Generator().manual_seed(CFG.sample_seed)

    loader = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        drop_last=False,
        persistent_workers=(CFG.num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    # --------- extract embeddings ---------
    t0 = time.time()
    pooled, z, meta_df = extract_embeddings(
        encoder=encoder,
        proj_head=proj_head,
        loader=loader,
        device=device,
        amp=CFG.amp,
        l2_normalize_pooled=CFG.l2_normalize_pooled,
    )
    elapsed = time.time() - t0

    npz_path = out_dir / f"embeddings_{sample_note}.npz"
    np.savez_compressed(npz_path, pooled=pooled, z=z)

    meta_csv = out_dir / f"metadata_{sample_note}.csv"
    meta_df.to_csv(meta_csv, index=False)

    # Save config + provenance (default=str handles Path on Windows)
    prov = {
        "script": "phase3_umap.py",
        "sample_note": sample_note,
        "n_points": int(len(meta_df)),
        "phase3_ckpt_path": CFG.phase3_ckpt_path,
        "train_dir": CFG.train_dir,
        "device": str(device),
        "elapsed_sec_extract": float(elapsed),
        "encoder_out_channels": int(getattr(encoder, "out_channels", 64)),
        "proj_hidden_dim": int(proj_hidden_dim),
        "proj_out_dim": int(proj_out_dim),
        "cfg": asdict(CFG),
        "ckpt_meta_keys": list(meta.keys()) if isinstance(meta, dict) else [],
    }
    (out_dir / f"umap_provenance_{sample_note}.json").write_text(
        json.dumps(prov, indent=2, default=str),
        encoding="utf-8",
    )

    # --------- run UMAPs (two spaces) ---------
    results_rows: List[Dict[str, Any]] = []

    spaces = [
        ("pooled", pooled),
        ("proj_z", z),
    ]

    # Stable palettes
    pv_order = ["PV01", "PV03", "PV08", "OTHER", "NA"]
    pv_palette = _make_palette(pv_order, cmap_name="tab10")

    hp_order = ["0", "1", "NA"]
    hp_palette = _make_palette(hp_order, cmap_name="tab10")

    for space_name, X in spaces:
        pv_purity = knn_purity(X, meta_df["pv_group"].tolist(), k=CFG.knn_k, metric=CFG.umap_metric)
        hp_purity = knn_purity(X, meta_df["has_panel"].tolist(), k=CFG.knn_k, metric=CFG.umap_metric)

        for nn in CFG.umap_n_neighbors:
            for md in CFG.umap_min_dist:
                for us in CFG.umap_seeds:
                    coords, um_meta = run_umap(
                        X=X,
                        use_pca=CFG.use_pca,
                        pca_dim=CFG.pca_dim,
                        n_neighbors=nn,
                        min_dist=md,
                        metric=CFG.umap_metric,
                        seed=us,
                    )

                    df_plot = meta_df.copy()
                    df_plot["umap_x"] = coords[:, 0]
                    df_plot["umap_y"] = coords[:, 1]

                    tag = f"{space_name}_nn{nn}_md{md}_seed{us}_{sample_note}"

                    coords_csv = out_dir / f"umap_coords_{tag}.csv"
                    df_plot.to_csv(coords_csv, index=False)

                    # PV group
                    if CFG.plot_by_pv_group:
                        plot_umap_scatter(
                            df=df_plot,
                            x_col="umap_x",
                            y_col="umap_y",
                            color_col="pv_group",
                            title=f"UMAP({space_name}) colored by PV group | nn={nn} md={md} seed={us} | N={len(df_plot)}",
                            out_path=out_dir / f"plot_umap_{tag}_pv_group",
                            categorical=True,
                            palette=pv_palette,
                            order=pv_order,
                            alpha=CFG.alpha_all,
                            point_size=CFG.point_size,
                            legend_title="pv_group",
                        )

                    # has_panel
                    if CFG.plot_by_has_panel:
                        df_plot["has_panel_plot"] = df_plot["has_panel"].apply(lambda x: "NA" if pd.isna(x) else str(int(x)))
                        plot_umap_scatter(
                            df=df_plot,
                            x_col="umap_x",
                            y_col="umap_y",
                            color_col="has_panel_plot",
                            title=f"UMAP({space_name}) colored by has_panel | nn={nn} md={md} seed={us} | N={len(df_plot)}",
                            out_path=out_dir / f"plot_umap_{tag}_has_panel",
                            categorical=True,
                            palette=hp_palette,
                            order=hp_order,
                            alpha=CFG.alpha_all,
                            point_size=CFG.point_size,
                            legend_title="has_panel",
                        )

                    # area_bin suite (overlay + pos-only bins + continuous area_ratio)
                    if CFG.plot_by_area_bins:
                        plot_area_bin_suite(
                            df_plot=df_plot,
                            space_name=space_name,
                            nn=nn,
                            md=md,
                            us=us,
                            out_dir=out_dir,
                            tag=tag,
                        )

                    # area_ratio per PV group (positives-only)
                    plot_area_ratio_per_pv_group(
                        df_plot=df_plot,
                        space_name=space_name,
                        nn=nn,
                        md=md,
                        us=us,
                        out_dir=out_dir,
                        tag=tag,
                    )

                    # Save UMAP config metadata
                    umap_cfg = {
                        "space": space_name,
                        **um_meta,
                        "sample_note": sample_note,
                        "n_points": int(len(df_plot)),
                        "knn_k": int(CFG.knn_k),
                        "knn_purity_pv_group": float(pv_purity),
                        "knn_purity_has_panel": float(hp_purity),
                    }
                    (out_dir / f"umap_config_{tag}.json").write_text(
                        json.dumps(umap_cfg, indent=2, default=str),
                        encoding="utf-8",
                    )

                    results_rows.append({"tag": tag, **umap_cfg})

    summary_df = pd.DataFrame(results_rows)
    summary_path = out_dir / f"umap_summary_{sample_note}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Saved embeddings: {npz_path}")
    print(f"[OK] Saved metadata:   {meta_csv}")
    print(f"[OK] Saved summary:    {summary_path}")
    print(f"[OK] Output dir:       {out_dir.resolve()}")


if __name__ == "__main__":
    main()
