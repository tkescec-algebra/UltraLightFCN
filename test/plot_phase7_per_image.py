"""
Phase-7 per-image plots from *_per_image.npz (TEST locked; reporting only).

Config-driven:
  - PHASE7_PER_IMAGE_DIR
  - PHASE7_PER_IMAGE_OUTDIR
  - PHASE7_TAIL_DICE_THRESH
  - PHASE7_WORST_N
  - PHASE7_INCLUDE_FULLFT (optional; used only for labeling logic)
  - PHASE7_MODEL_ORDER / PHASE7_MODEL_LABELS (optional)

Expected npz arrays:
  - dice, iou, precision, recall  (float arrays length N)
  - group                         (string array length N: PV01/PV03/PV08)

Outputs:
  - ecdf_dice_overall
  - box_dice_by_group
  - tailrate_dice_by_group
  - per_image_merged.parquet (optional; handy for later analysis)
  - worst_images_overall.csv
  - worst_images_by_model.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.config as config


# -----------------------------
# Model id parsing (from filename)
# -----------------------------

def parse_from_filename(stem: str) -> Tuple[str, str, int]:
    """
    Returns (base_model, ft_regime, seed) inferred from filename stem.

    Supports examples:
      ultralight_phase6_cand54_seed71_seed71_per_image
      dlv3p_resnet50__minft__seed_13_per_image
      dlv3p_resnet50::minft::seed_13_per_image
      unet_resnet34__seed37_per_image  (ft_regime="")
    """
    s = stem

    # Seed: try seed_13, seed13, seed_071, etc.
    m = re.search(r"(?:^|[^a-zA-Z])seed[_\-]?(\d+)(?:[^0-9]|$)", s)
    seed = int(m.group(1)) if m else -1

    # Normalize separators
    s_norm = s.replace("::", "__").replace("-", "_")

    # ft regime
    ft_regime = ""
    if "__minft__" in s_norm:
        ft_regime = "minft"
    elif "__fullft__" in s_norm:
        ft_regime = "fullft"

    # base_model heuristic:
    if s_norm.startswith("ultralight_phase6"):
        base_model = "ultralight_phase6"
    else:
        # take first token up to regime/seed marker
        # e.g. dlv3p_resnet50__minft__seed_13_per_image -> dlv3p_resnet50
        base_model = s_norm.split("__")[0]
        # fallback: cut at _seed
        base_model = re.split(r"_seed[_\-]?\d+", base_model)[0]

    return base_model, ft_regime, seed


# -----------------------------
# Labels & colors
# -----------------------------

DEFAULT_ORDER = getattr(config, "PHASE7_MODEL_ORDER", [
    "ultralight_phase6",
    "dlv3p_resnet50",
    "dlv3p_mobilenetv2",
    "unet_resnet34",
])

DEFAULT_LABELS: Dict[str, str] = dict(getattr(config, "PHASE7_MODEL_LABELS", {
    "ultralight_phase6": "UltraLightFCN (ours)",
    "dlv3p_resnet50": "DeepLabV3+ R50",
    "dlv3p_mobilenetv2": "DeepLabV3+ MNetV2",
    "unet_resnet34": "U-Net R34",
}))

INCLUDE_FULLFT = bool(getattr(config, "PHASE7_INCLUDE_FULLFT", False))
SHOW_REGIME = INCLUDE_FULLFT  # same policy as Phase-7 plots

SUBSETS = ["overall", "PV01", "PV03", "PV08"]


def label_for(base_model: str, ft_regime: str) -> str:
    s = DEFAULT_LABELS.get(base_model, base_model)
    if SHOW_REGIME and ft_regime in ("minft", "fullft"):
        s = f"{s} ({ft_regime.upper()})"
    return s


def build_color_map(model_keys: List[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
    # lock colors based on model order (matplotlib default cycle)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(cycle) < 8:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    # Sort keys by DEFAULT_ORDER then regime (minft before fullft)
    def base_idx(bm: str) -> int:
        return DEFAULT_ORDER.index(bm) if bm in DEFAULT_ORDER else 999

    def ksort(k: Tuple[str, str]) -> Tuple[int, int, str]:
        bm, fr = k
        rr = 0 if fr in ("", "minft") else 1
        return (base_idx(bm), rr, bm)

    keys_sorted = sorted(model_keys, key=ksort)

    cmap: Dict[Tuple[str, str], str] = {}
    for i, k in enumerate(keys_sorted):
        cmap[k] = cycle[i % len(cycle)]
    return cmap


# -----------------------------
# IO
# -----------------------------

def load_all_npz(root_dir: Path) -> pd.DataFrame:
    npz_files = sorted(root_dir.rglob("*_per_image.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_per_image.npz found under: {root_dir}")

    rows = []
    for fp in npz_files:
        data = np.load(fp, allow_pickle=True)

        # required keys
        for k in ("dice", "iou", "precision", "recall", "group"):
            if k not in data:
                raise KeyError(f"Missing key '{k}' in {fp.name}")

        dice = data["dice"].astype(np.float32)
        iou = data["iou"].astype(np.float32)
        prec = data["precision"].astype(np.float32)
        rec = data["recall"].astype(np.float32)

        group = data["group"]
        # group may be bytes/object -> normalize to str
        group = np.array([g.decode("utf-8") if isinstance(g, (bytes, bytearray)) else str(g) for g in group])

        base_model, ft_regime, seed = parse_from_filename(fp.stem)

        n = len(dice)
        for i in range(n):
            rows.append({
                "file": str(fp),
                "base_model": base_model,
                "ft_regime": ft_regime,
                "seed": seed,
                "subset": group[i],     # PV01/PV03/PV08
                "dice": float(dice[i]),
                "iou": float(iou[i]),
                "precision": float(prec[i]),
                "recall": float(rec[i]),
            })

    df = pd.DataFrame(rows)

    # Add "overall" view by copying rows with subset="overall"
    overall = df.copy()
    overall["subset"] = "overall"
    df = pd.concat([df, overall], ignore_index=True)

    return df


# -----------------------------
# Plot helpers
# -----------------------------

def save_fig(fig: plt.Figure, outpath: Path, dpi: int = 300) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".pdf"))
    fig.savefig(outpath.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / float(len(x))
    return x, y


# -----------------------------
# Plots
# -----------------------------

def plot_ecdf_dice_overall(df: pd.DataFrame, outdir: Path, color_map: Dict[Tuple[str, str], str]) -> None:
    dfo = df[df["subset"] == "overall"].copy()
    keys = sorted({(r.base_model, r.ft_regime) for r in dfo.itertuples(index=False)})

    fig, ax = plt.subplots()

    for key in keys:
        bm, fr = key
        vals = dfo[(dfo["base_model"] == bm) & (dfo["ft_regime"] == fr)]["dice"].to_numpy(dtype=float)
        x, y = ecdf(vals)
        ax.plot(x, y, label=label_for(bm, fr), color=color_map.get(key, None))

    ax.set_xlabel("Hard Dice@0.5 (per-image)")
    ax.set_ylabel("ECDF")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", axis="both", alpha=0.3)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

    save_fig(fig, outdir / "ecdf_dice_overall")


def plot_box_dice_by_group(df: pd.DataFrame, outdir: Path, color_map: Dict[Tuple[str, str], str]) -> None:
    # 2x2 panels: overall + PV01 + PV03 + PV08
    titles = {"overall": "Overall", "PV01": "PV01 (0.1 m)", "PV03": "PV03 (0.3 m)", "PV08": "PV08 (0.8 m)"}

    keys = sorted({(r.base_model, r.ft_regime) for r in df.itertuples(index=False)})

    # lock order by DEFAULT_ORDER for readability
    def base_idx(bm: str) -> int:
        return DEFAULT_ORDER.index(bm) if bm in DEFAULT_ORDER else 999

    def ksort(k: Tuple[str, str]) -> Tuple[int, int, str]:
        bm, fr = k
        rr = 0 if fr in ("", "minft") else 1
        return (base_idx(bm), rr, bm)

    keys = sorted(keys, key=ksort)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for ax, subset in zip(axes, SUBSETS):
        sub = df[df["subset"] == subset].copy()

        data = []
        labels = []
        colors = []
        for key in keys:
            bm, fr = key
            vals = sub[(sub["base_model"] == bm) & (sub["ft_regime"] == fr)]["dice"].to_numpy(dtype=float)
            data.append(vals)
            labels.append(label_for(bm, fr))
            colors.append(color_map.get(key, None))

        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)

        ax.set_title(titles.get(subset, subset))
        ax.set_ylabel("Hard Dice@0.5")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticklabels(labels, rotation=20, ha="right")

    fig.suptitle("Per-image Hard Dice@0.5 distributions (TEST)")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    save_fig(fig, outdir / "box_dice_by_group")


def plot_tailrate_by_group(df: pd.DataFrame, outdir: Path, color_map: Dict[Tuple[str, str], str], thresh: float) -> None:
    # Tail rate = % images with dice < thresh
    keys = sorted({(r.base_model, r.ft_regime) for r in df.itertuples(index=False)})

    def base_idx(bm: str) -> int:
        return DEFAULT_ORDER.index(bm) if bm in DEFAULT_ORDER else 999

    def ksort(k: Tuple[str, str]) -> Tuple[int, int, str]:
        bm, fr = k
        rr = 0 if fr in ("", "minft") else 1
        return (base_idx(bm), rr, bm)

    keys = sorted(keys, key=ksort)

    subsets = ["PV01", "PV03", "PV08"]  # tail rate is most meaningful per subgroup
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(subsets))
    width = 0.8 / max(1, len(keys))  # group bars per model

    for i, key in enumerate(keys):
        bm, fr = key
        rates = []
        for s in subsets:
            vals = df[(df["subset"] == s) & (df["base_model"] == bm) & (df["ft_regime"] == fr)]["dice"].to_numpy(dtype=float)
            rate = float(np.mean(vals < thresh)) if len(vals) else float("nan")
            rates.append(rate)
        ax.bar(x + i * width, rates, width=width, color=color_map.get(key, None), label=label_for(bm, fr))

    ax.set_xticks(x + width * (len(keys) - 1) / 2.0)
    ax.set_xticklabels(subsets)
    ax.set_ylabel(f"Fraction of images with Dice@0.5 < {thresh:.2f}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Tail failure rate by resolution subgroup (TEST)")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

    save_fig(fig, outdir / f"tailrate_dice_lt_{str(thresh).replace('.','p')}_by_group")


# -----------------------------
# Exports
# -----------------------------

def export_worst_cases(df: pd.DataFrame, outdir: Path, worst_n: int) -> None:
    # overall worst across all models (for sanity)
    dfo = df[df["subset"] == "overall"].copy()
    worst_all = dfo.sort_values("dice", ascending=True).head(worst_n)
    worst_all.to_csv(outdir / "worst_images_overall.csv", index=False)

    # worst per model (overall)
    rows = []
    for (bm, fr), g in dfo.groupby(["base_model", "ft_regime"]):
        w = g.sort_values("dice", ascending=True).head(worst_n)
        w = w.assign(model_label=label_for(bm, fr))
        rows.append(w)
    worst_by_model = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    worst_by_model.to_csv(outdir / "worst_images_by_model.csv", index=False)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    root = Path(getattr(config, "PHASE7_PER_IMAGE_DIR"))
    outdir = Path(getattr(config, "PHASE7_PER_IMAGE_OUTDIR"))
    thresh = float(getattr(config, "PHASE7_TAIL_DICE_THRESH", 0.10))
    worst_n = int(getattr(config, "PHASE7_WORST_N", 50))

    df = load_all_npz(root)

    # If include_fullft=False, we can drop fullft entries if they exist
    if not INCLUDE_FULLFT:
        df = df[df["ft_regime"].isin(["", "minft"])].copy()

    # Build consistent colors per (base_model, ft_regime) present
    keys = sorted({(r.base_model, r.ft_regime) for r in df.itertuples(index=False)})
    cmap = build_color_map(keys)

    # Save merged data for later (fast reload)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(outdir / "per_image_merged.parquet", index=False)
    except Exception:
        # parquet optional; CSV fallback
        df.to_csv(outdir / "per_image_merged.csv", index=False)

    plot_ecdf_dice_overall(df, outdir, cmap)
    plot_box_dice_by_group(df, outdir, cmap)
    plot_tailrate_by_group(df, outdir, cmap, thresh)
    export_worst_cases(df, outdir, worst_n)

    print(f"[OK] Per-image plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()