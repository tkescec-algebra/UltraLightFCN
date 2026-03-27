"""
make_fig3_simclr.py

Generates:
  - A) augmentation sensitivity heatmap
  - B) pretraining convergence (train loss + LR)
  - C) alignment vs uniformity scatter (colored by Δ NT-Xent late)
  - D) ablation ranking by impact score
  - Figure2: 2x2 grid with A/B/C/D

Outputs:
  - Individual plots: PNG + PDF
  - 2x2 figure: PNG + PDF

Notes:
  - Uses recursive glob for train_metrics.csv files inside subfolders.
  - Plot C: x-axis forced to end at 0.5, denser major+minor grid.
"""

from dataclasses import dataclass
import os
import glob
import re
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------
# Config dataclass (as requested)
# -----------------------------
@dataclass(frozen=True)
class Fig3SimCLRConfig:
    # Inputs
    heatmap_csv: str = "runs/heatmaps/heatmap_matrix.csv"
    summary_csv: str = "runs/heatmaps/summary_metrics.csv"

    # NOTE: recursive glob across subfolders
    train_glob: str = "runs/aug_sensitivity/**/train_metrics.csv"

    # Outputs
    outdir: str = "./fig3_simclr_outputs"
    dpi: int = 300

    # Titles
    fig3_suptitle: str = "Figure 2. SimCLR pretraining methodology"


CFG = Fig3SimCLRConfig()


# -----------------------------
# Label mapping (paper-friendly)
# -----------------------------
LABEL_MAP: Dict[str, str] = {
    "baseline": "baseline",
    "no_crop": "crop OFF",
    "no_jitter": "jitter OFF",
    "no_rotation": "rotation OFF",
    "no_gray": "grayscale OFF",
    "no_blur": "blur OFF",
    "no_vflip": "vflip OFF",
    "no_hflip": "hflip OFF",
}


def pretty_label(cfg: str) -> str:
    return LABEL_MAP.get(cfg, cfg)


# -----------------------------
# Helpers
# -----------------------------
def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, outdir: str, stem: str, dpi: int = 300) -> None:
    png_path = os.path.join(outdir, f"{stem}.png")
    pdf_path = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")


def strip_seed(run_id: str) -> str:
    # Examples: baseline_seed13 -> baseline
    return re.sub(r"_seed\d+$", "", str(run_id))


def require_cols(df: pd.DataFrame, cols: set, df_name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing columns: {missing}")


def load_train_metrics(train_glob: str) -> pd.DataFrame:
    # IMPORTANT: recursive=True to allow ** pattern
    paths = sorted(glob.glob(train_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No files match train_glob (recursive): {train_glob}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)

        needed = {"seed", "epoch", "lr", "train_loss"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing columns: {missing}")

        # Helpful metadata for debugging / filtering
        df["source_file"] = os.path.basename(p)
        df["source_dir"] = os.path.basename(os.path.dirname(p))  # e.g., baseline_seed13
        df["source_path"] = p

        dfs.append(df)

    tm = pd.concat(dfs, ignore_index=True)

    # Drop exact duplicates (e.g., copied logs)
    tm = tm.drop_duplicates(subset=["source_path", "seed", "epoch", "lr", "train_loss"])
    return tm


def aggregate_epoch_mean_std(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = df.groupby("epoch")[value_col]
    epochs = g.mean().index.to_numpy()
    mean = g.mean().to_numpy()
    std = g.std(ddof=1).fillna(0.0).to_numpy()
    return epochs, mean, std


# -----------------------------
# Plot A: Augmentation sensitivity heatmap (Δ vs baseline)
# -----------------------------
def plot_aug_sensitivity_heatmap(
    heatmap_df: pd.DataFrame,
    title: str = "Augmentation sensitivity (Δ vs baseline)"
) -> plt.Figure:
    df = heatmap_df.copy()

    col_map = {
        "delta_late_mean_loss": "Δ NT-Xent (late)",
        "delta_alignment": "Δ Alignment",
        "delta_uniformity": "Δ Uniformity",
    }
    df = df.rename(columns=col_map)

    data = df.values.astype(float)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(data, aspect="auto")

    ax.set_title(title)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels([pretty_label(str(i)) for i in df.index])
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=25, ha="right")

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Δ vs baseline")

    fig.tight_layout()
    return fig


# -----------------------------
# Plot B: Pretraining convergence (NT-Xent train loss + LR)
# -----------------------------
def plot_pretrain_convergence(
    train_metrics: pd.DataFrame,
    title: str = "SimCLR pretraining convergence"
) -> plt.Figure:
    epochs, loss_mean, loss_std = aggregate_epoch_mean_std(train_metrics, "train_loss")
    _, lr_mean, _ = aggregate_epoch_mean_std(train_metrics, "lr")

    fig, ax1 = plt.subplots(figsize=(7.8, 4.6))

    ax1.plot(epochs, loss_mean, label="Train loss (mean)")
    ax1.fill_between(
        epochs, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2, label="±1 std (across seeds)"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NT-Xent train loss")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(epochs, lr_mean, linestyle="--", label="LR (mean)")
    ax2.set_ylabel("Learning rate")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    return fig


# -----------------------------
# Plot C: Alignment–Uniformity scatter colored by Δ NT-Xent (late)
# -----------------------------
def plot_alignment_uniformity_scatter_colored(
    summary_df: pd.DataFrame,
    title: str = "Alignment vs Uniformity (augmentation ablations)",
    color_col: str = "delta_late_mean_loss",  # Δ NT-Xent (late) vs baseline
) -> plt.Figure:
    require_cols(summary_df, {"run_id", "alignment", "uniformity"}, "summary_metrics.csv")

    if color_col not in summary_df.columns:
        raise ValueError(
            f"summary_metrics.csv is missing '{color_col}'. "
            f"Available columns: {list(summary_df.columns)}"
        )

    df = summary_df.copy()
    df["cfg"] = df["run_id"].apply(strip_seed)

    # Aggregate across seeds per cfg
    g = df.groupby("cfg").agg(
        alignment_mean=("alignment", "mean"),
        uniformity_mean=("uniformity", "mean"),
        color_mean=(color_col, "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7.8, 5.1))

    sc = ax.scatter(
        g["alignment_mean"],
        g["uniformity_mean"],
        c=g["color_mean"],
        s=85,
        zorder=3,
    )

    # Dense grid: major + minor
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.25, zorder=0)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # X axis forced to end at 0.5
    x = g["alignment_mean"].to_numpy(dtype=float)
    y = g["uniformity_mean"].to_numpy(dtype=float)
    x_range = max(np.ptp(x), 1e-6)
    y_range = max(np.ptp(y), 1e-6)

    x_pad_left = 0.05 * x_range
    y_pad = 0.08 * y_range
    ax.set_xlim(max(0.0, x.min() - x_pad_left), 0.5)
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

    # Slightly denser major ticks as well
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7))

    # Label offsets (more separation from points)
    base_offset = (10, 7)
    custom_offsets = {
        "baseline": (-50, 0),
        "no_hflip": (10, 0),
        "no_vflip": (10, 0),
        "no_blur": (-50, 0),
        "no_gray": (10, 0),
        "no_rotation": (-70, 0),
        "no_jitter": (10, 0),
        "no_crop": (10, 0),
    }

    for _, r in g.iterrows():
        cfg = r["cfg"]
        dx, dy = custom_offsets.get(cfg, base_offset)
        ax.annotate(
            pretty_label(cfg),
            (r["alignment_mean"], r["uniformity_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            ha="left",
            va="center",
            zorder=4,
        )

    # Highlight baseline
    base = g[g["cfg"] == "baseline"]
    if len(base) == 1:
        bx = float(base["alignment_mean"].iloc[0])
        by = float(base["uniformity_mean"].iloc[0])
        ax.scatter([bx], [by], s=230, marker="*", edgecolors="k", linewidths=0.7, label="baseline", zorder=5)
        ax.legend(loc="best")

    ax.set_xlabel("Alignment (lower is better)")
    ax.set_ylabel("Uniformity (more negative is better)")
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Δ NT-Xent (late) vs baseline")

    fig.tight_layout()
    return fig


# -----------------------------
# Plot D: Augmentation ranking by impact_score (bar)
# -----------------------------
def plot_impact_ranking(
    summary_df: pd.DataFrame,
    title: str = "Augmentation ablation ranking (impact score)"
) -> plt.Figure:
    require_cols(summary_df, {"run_id", "impact_score"}, "summary_metrics.csv")

    df = summary_df.copy()
    df["cfg"] = df["run_id"].apply(strip_seed)

    g = df.groupby("cfg").agg(
        impact_mean=("impact_score", "mean"),
        impact_std=("impact_score", "std"),
    ).reset_index()

    g = g.sort_values("impact_mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    y = np.arange(len(g))

    ax.barh(y, g["impact_mean"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_label(x) for x in g["cfg"].to_numpy()])
    ax.invert_yaxis()

    ax.set_xlabel("Impact score (mean across seeds)")
    ax.set_title(title)

    ax.errorbar(
        g["impact_mean"].to_numpy(),
        y,
        xerr=np.nan_to_num(g["impact_std"].to_numpy(), nan=0.0),
        fmt="none",
        capsize=3,
    )

    fig.tight_layout()
    return fig


# -----------------------------
# Combined Figure 2 (2x2 grid)
# -----------------------------
def make_figure2_grid(
    heatmap_df: pd.DataFrame,
    train_metrics: pd.DataFrame,
    summary_df: pd.DataFrame,
    suptitle: str,
    scatter_color_col: str = "delta_late_mean_loss",
) -> plt.Figure:
    require_cols(summary_df, {"run_id", "alignment", "uniformity"}, "summary_metrics.csv")
    if scatter_color_col not in summary_df.columns:
        raise ValueError(
            f"summary_metrics.csv is missing '{scatter_color_col}'. "
            f"Available columns: {list(summary_df.columns)}"
        )

    fig = plt.figure(figsize=(14, 9))

    # More space between plots (as requested)
    gs = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.50)

    # Panel A: heatmap
    axA = fig.add_subplot(gs[0, 0])
    dfA = heatmap_df.copy().rename(columns={
        "delta_late_mean_loss": "Δ NT-Xent (late)",
        "delta_alignment": "Δ Alignment",
        "delta_uniformity": "Δ Uniformity",
    })
    dataA = dfA.values.astype(float)
    im = axA.imshow(dataA, aspect="auto")
    axA.set_title("(a) Augmentation sensitivity (Δ vs baseline)")
    axA.set_yticks(np.arange(dfA.shape[0]))
    axA.set_yticklabels([pretty_label(str(i)) for i in dfA.index])
    axA.set_xticks(np.arange(dfA.shape[1]))
    axA.set_xticklabels(dfA.columns, rotation=25, ha="right")
    for i in range(dfA.shape[0]):
        for j in range(dfA.shape[1]):
            axA.text(j, i, f"{dataA[i, j]:.3f}", ha="center", va="center", fontsize=8)
    cbarA = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cbarA.set_label("Δ vs baseline")

    # Panel B: convergence
    axB = fig.add_subplot(gs[0, 1])
    epochs, loss_mean, loss_std = aggregate_epoch_mean_std(train_metrics, "train_loss")
    _, lr_mean, _ = aggregate_epoch_mean_std(train_metrics, "lr")

    axB.plot(epochs, loss_mean, label="Train loss (mean)")
    axB.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2, label="±1 std")
    axB.set_xlabel("Epoch")
    axB.set_ylabel("NT-Xent train loss")
    axB.set_title("(b) Pretraining convergence (loss + LR)")

    axB2 = axB.twinx()
    axB2.plot(epochs, lr_mean, linestyle="--", label="LR (mean)")
    axB2.set_ylabel("Learning rate")

    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axB2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

    # Panel C: colored scatter with dense grid + x up to 0.5
    axC = fig.add_subplot(gs[1, 0])
    dfC = summary_df.copy()
    dfC["cfg"] = dfC["run_id"].apply(strip_seed)
    gC = dfC.groupby("cfg").agg(
        alignment_mean=("alignment", "mean"),
        uniformity_mean=("uniformity", "mean"),
        color_mean=(scatter_color_col, "mean"),
    ).reset_index()

    sc = axC.scatter(
        gC["alignment_mean"],
        gC["uniformity_mean"],
        c=gC["color_mean"],
        s=80,
        zorder=3,
    )

    axC.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    axC.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.25, zorder=0)
    axC.minorticks_on()
    axC.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    axC.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    axC.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
    axC.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7))

    x = gC["alignment_mean"].to_numpy(dtype=float)
    y = gC["uniformity_mean"].to_numpy(dtype=float)
    x_range = max(np.ptp(x), 1e-6)
    y_range = max(np.ptp(y), 1e-6)
    x_pad_left = 0.05 * x_range
    y_pad = 0.08 * y_range
    axC.set_xlim(max(0.0, x.min() - x_pad_left), 0.5)
    axC.set_ylim(y.min() - y_pad, y.max() + y_pad)

    base_offset = (9, 6)
    custom_offsets = {
        "baseline": (-45, 0),
        "no_hflip": (10, 0),
        "no_vflip": (10, 0),
        "no_blur": (-45, 0),
        "no_gray": (10, 0),
        "no_rotation": (-65, 0),
        "no_jitter": (10, 0),
        "no_crop": (10, 0),
    }
    for _, r in gC.iterrows():
        cfg = r["cfg"]
        dx, dy = custom_offsets.get(cfg, base_offset)
        axC.annotate(
            pretty_label(cfg),
            (r["alignment_mean"], r["uniformity_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="center",
            zorder=4,
        )

    baseC = gC[gC["cfg"] == "baseline"]
    if len(baseC) == 1:
        bx = float(baseC["alignment_mean"].iloc[0])
        by = float(baseC["uniformity_mean"].iloc[0])
        axC.scatter([bx], [by], s=180, marker="*", edgecolors="k", linewidths=0.6, label="baseline", zorder=5)
        axC.legend(loc="best", fontsize=9)

    axC.set_xlabel("Alignment (lower is better)")
    axC.set_ylabel("Uniformity (more negative is better)")
    axC.set_title("(c) Alignment–Uniformity (colored by Δ NT-Xent late)")

    cbarC = fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.04)
    cbarC.set_label("Δ NT-Xent (late) vs baseline")

    # Panel D: impact ranking
    axD = fig.add_subplot(gs[1, 1])
    require_cols(summary_df, {"run_id", "impact_score"}, "summary_metrics.csv")
    dfD = summary_df.copy()
    dfD["cfg"] = dfD["run_id"].apply(strip_seed)
    gD = dfD.groupby("cfg").agg(
        impact_mean=("impact_score", "mean"),
        impact_std=("impact_score", "std"),
    ).reset_index().sort_values("impact_mean", ascending=False)

    y = np.arange(len(gD))
    axD.barh(y, gD["impact_mean"].to_numpy())
    axD.set_yticks(y)
    axD.set_yticklabels([pretty_label(x) for x in gD["cfg"].to_numpy()])
    axD.invert_yaxis()
    axD.set_xlabel("Impact score (mean across seeds)")
    axD.set_title("(d) Ablation ranking (impact score)")
    axD.errorbar(
        gD["impact_mean"].to_numpy(),
        y,
        xerr=np.nan_to_num(gD["impact_std"].to_numpy(), nan=0.0),
        fmt="none",
        capsize=3,
    )

    # fig.suptitle(suptitle, y=0.98, fontsize=14)

    # Use a slightly larger top margin; keep our spacing (avoid squeezing)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# -----------------------------
# Run
# -----------------------------
def run(cfg: Fig3SimCLRConfig) -> None:
    outdir = ensure_outdir(cfg.outdir)

    heatmap_df = pd.read_csv(cfg.heatmap_csv, index_col=0)
    summary_df = pd.read_csv(cfg.summary_csv)
    train_metrics = load_train_metrics(cfg.train_glob)

    # Individual plots
    figA = plot_aug_sensitivity_heatmap(heatmap_df)
    save_fig(figA, outdir, "A_aug_sensitivity_heatmap", dpi=cfg.dpi)
    plt.close(figA)

    figB = plot_pretrain_convergence(train_metrics)
    save_fig(figB, outdir, "B_pretrain_convergence_loss_lr", dpi=cfg.dpi)
    plt.close(figB)

    figC = plot_alignment_uniformity_scatter_colored(summary_df)
    save_fig(figC, outdir, "C_alignment_uniformity_scatter_colored_delta_ntxent", dpi=cfg.dpi)
    plt.close(figC)

    figD = plot_impact_ranking(summary_df)
    save_fig(figD, outdir, "D_impact_ranking_barh", dpi=cfg.dpi)
    plt.close(figD)

    # Combined 2x2 figure
    fig3 = make_figure2_grid(
        heatmap_df=heatmap_df,
        train_metrics=train_metrics,
        summary_df=summary_df,
        suptitle=cfg.fig3_suptitle,
        scatter_color_col="delta_late_mean_loss",
    )
    save_fig(fig3, outdir, "Figure3_SimCLR", dpi=cfg.dpi)
    plt.close(fig3)

    print("\nDone.")


if __name__ == "__main__":
    run(CFG)
