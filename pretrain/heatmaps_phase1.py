import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "pairwise_results/simclr_pairwise_results.csv"
OUT_DIR = "pairwise_results/figures/heatmaps"

OPS = ["identity", "color", "blur", "hflip", "vflip", "rotate"]


def plot_heatmap_simclr_style(df, value_col, title, out_png,
                              cmap="plasma",  # SimCLR-ish (plasma/inferno)
                              fmt="{:.1f}",
                              vmin=None, vmax=None,
                              cbar_label=None,
                              lower_is_better=True):
    pivot = df.pivot(index="t1", columns="t2", values=value_col)
    pivot = pivot.reindex(index=OPS, columns=OPS)
    data = pivot.values.astype(float)

    # Figure / axes
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=200)

    im = ax.imshow(
        data,
        cmap=cmap,
        origin="upper",      # SimCLR figure reads top-to-bottom (Crop on top)
        aspect="equal",      # square cells
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    # Ticks / labels
    ax.set_xticks(np.arange(len(OPS)))
    ax.set_yticks(np.arange(len(OPS)))
    ax.set_xticklabels(OPS, rotation=35, ha="right")
    ax.set_yticklabels(OPS)

    ax.set_xlabel("2nd transformation")
    ax.set_ylabel("1st transformation")
    ax.set_title(title, pad=10)

    # White gridlines between cells (SimCLR style)
    ax.set_xticks(np.arange(-.5, len(OPS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(OPS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate values in each cell
    # Choose text color based on brightness for readability
    # (simple heuristic using normalized value)
    if vmin is None:
        vmin_ = np.nanmin(data)
    else:
        vmin_ = vmin
    if vmax is None:
        vmax_ = np.nanmax(data)
    else:
        vmax_ = vmax

    denom = (vmax_ - vmin_) if (vmax_ - vmin_) != 0 else 1.0
    norm = (data - vmin_) / denom

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            txt_color = "black" if norm[i, j] > 0.6 else "white"
            ax.text(j, i, fmt.format(val), ha="center", va="center", color=txt_color, fontsize=9)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    plot_heatmap_simclr_style(df, "loss",
        "Pairwise Augmentations (NT-Xent loss, lower better)",
        f"{OUT_DIR}/heatmap_loss_simclr.png",
        fmt="{:.3f}", cmap="plasma")

    plot_heatmap_simclr_style(df, "alignment",
        "Pairwise Augmentations (Alignment, lower better)",
        f"{OUT_DIR}/heatmap_alignment_simclr.png",
        fmt="{:.3f}", cmap="plasma")

    plot_heatmap_simclr_style(df, "uniformity",
        "Pairwise Augmentations (Uniformity, lower better)",
        f"{OUT_DIR}/heatmap_uniformity_simclr.png",
        fmt="{:.3f}", cmap="plasma")

    plot_heatmap_simclr_style(df, "ssim",
        "Pairwise Augmentations (SSIM, higher better)",
        f"{OUT_DIR}/heatmap_ssim_simclr.png",
        fmt="{:.3f}", cmap="plasma")


if __name__ == "__main__":
    main()
