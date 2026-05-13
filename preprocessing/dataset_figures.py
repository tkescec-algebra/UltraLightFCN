from dataclasses import dataclass
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# -----------------------------
# Config dataclass
# -----------------------------
@dataclass(frozen=True)
class DatasetFigureConfig:
    dataset_root: Path = Path("../dataset")
    outdir: Path = Path(".")
    save_stem: str = "figure_2_dataset_description"
    dpi: int = 300
    seed: int = 42

    # Figure/layout style aligned with make_fig3_simclr
    figure_size: tuple = (11, 7)
    title_fontsize: int = 10
    label_fontsize: int = 10
    tick_fontsize: int = 10
    legend_fontsize: int = 9
    annotation_fontsize: int = 8
    title_pad: int = 10
    panel_wspace: float = 0.18
    panel_hspace: float = 0.32
    tight_layout_rect: tuple = (0,0,1,1)  # left, bottom, right, top


CFG = DatasetFigureConfig()

SUBSETS = ["PV01", "PV03", "PV08"]
SPLITS = ["train", "valid", "test"]
SUBSET_COLORS: Dict[str, str] = {
    "PV01": "tab:blue",
    "PV03": "tab:orange",
    "PV08": "tab:green",
}


# -----------------------------
# Helpers
# -----------------------------
def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, outdir: Path, stem: str, dpi: int = 300) -> None:
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a mask as a binary numpy array {0, 1}."""
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    return (mask_np > 0).astype(np.uint8)


def mask_coverage(mask_np: np.ndarray) -> float:
    """Return the fraction of positive pixels in a mask."""
    return float((mask_np > 0).mean())


def get_subset_from_name(filename: str) -> str:
    """Extract the subset label from a filename."""
    upper = filename.upper()
    for subset in SUBSETS:
        if subset in upper:
            return subset
    return "OTHER"


def thousands_formatter(x, pos):
    del pos
    return f"{int(x):,}"


def collect_dataset_records(dataset_root: Path) -> List[dict]:
    """Scan the dataset root and collect paired image/mask records."""
    records: List[dict] = []
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[warning] Missing split folder: {split_dir}")
            continue

        all_files = sorted(
            p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts
        )
        image_files = [p for p in all_files if "_label" not in p.stem]

        for img_path in image_files:
            mask_name = f"{img_path.stem}_label{img_path.suffix}"
            mask_path = split_dir / mask_name

            if not mask_path.exists():
                print(f"[warning] Missing mask for image: {img_path.name}")
                continue

            mask_np = load_mask(mask_path)
            coverage = mask_coverage(mask_np)
            positive_pixels = int(mask_np.sum())
            is_positive = (coverage >= 0.005) or (positive_pixels >= 64)

            records.append(
                {
                    "split": split,
                    "subset": get_subset_from_name(img_path.name),
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "coverage": coverage,
                    "is_positive": is_positive,
                }
            )

    return records


def aggregate_dataset_stats(records: List[dict]) -> dict:
    """Aggregate counts and coverage statistics used by the figure panels."""
    pos_neg_by_subset = {subset: {"positive": 0, "negative": 0} for subset in SUBSETS}
    split_counts = {split: {subset: 0 for subset in SUBSETS} for split in SPLITS}
    coverage_by_subset_pos = {subset: [] for subset in SUBSETS}

    for record in records:
        subset = record["subset"]
        split = record["split"]

        if subset not in SUBSETS or split not in SPLITS:
            continue

        if record["is_positive"]:
            pos_neg_by_subset[subset]["positive"] += 1
            coverage_by_subset_pos[subset].append(record["coverage"] * 100.0)
        else:
            pos_neg_by_subset[subset]["negative"] += 1

        split_counts[split][subset] += 1

    return {
        "pos_neg_by_subset": pos_neg_by_subset,
        "split_counts": split_counts,
        "coverage_by_subset_pos": coverage_by_subset_pos,
    }


def style_axis(ax: plt.Axes, cfg: DatasetFigureConfig, *, y_grid_only: bool = True, which: str = "major") -> None:
    """Apply a consistent visual style to an axis."""
    if y_grid_only:
        ax.grid(axis="y", linestyle="--", alpha=0.35, which=which)
    else:
        ax.grid(True, linestyle="--", alpha=0.35, which=which)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=cfg.tick_fontsize)


def set_axis_text_bold(ax: plt.Axes) -> None:
    """Make all visible text elements on an axis bold."""
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    for text in ax.texts:
        text.set_fontweight("bold")

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")
        title = legend.get_title()
        if title is not None:
            title.set_fontweight("bold")


def add_bar_labels(ax: plt.Axes, bar_groups: List, cfg: DatasetFigureConfig, scale: float = 1.08) -> None:
    """Add numeric labels above bar containers."""
    for bars in bar_groups:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height * scale,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=cfg.annotation_fontsize,
                fontweight="bold",
            )


# -----------------------------
# Panel builders
# -----------------------------
def plot_panel_a_tile_distribution(ax: plt.Axes, stats: dict, cfg: DatasetFigureConfig) -> None:
    x = np.arange(len(SUBSETS))
    width = 0.34
    positives = [stats["pos_neg_by_subset"][subset]["positive"] for subset in SUBSETS]
    negatives = [stats["pos_neg_by_subset"][subset]["negative"] for subset in SUBSETS]

    bars_pos = ax.bar(x - width / 2, positives, width=width, label="Positive")
    bars_neg = ax.bar(x + width / 2, negatives, width=width, label="Negative")

    ax.set_title(
        "(a) Tile distribution by subset",
        fontsize=cfg.title_fontsize,
        pad=cfg.title_pad,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS, fontweight="bold")
    ax.set_ylabel("Number of tiles (log scale)", fontsize=cfg.label_fontsize, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(10, 30000)
    style_axis(ax, cfg, which="both")
    ax.legend(frameon=False, loc="upper right", prop={"size": cfg.legend_fontsize, "weight": "bold"})
    add_bar_labels(ax, [bars_pos, bars_neg], cfg)


def plot_panel_b_split_distribution(ax: plt.Axes, stats: dict, cfg: DatasetFigureConfig) -> None:
    x = np.arange(len(SPLITS))
    width = 0.24
    pv01_vals = [stats["split_counts"][split]["PV01"] for split in SPLITS]
    pv03_vals = [stats["split_counts"][split]["PV03"] for split in SPLITS]
    pv08_vals = [stats["split_counts"][split]["PV08"] for split in SPLITS]

    bars_pv01 = ax.bar(x - width, pv01_vals, width=width, label="PV01")
    bars_pv03 = ax.bar(x, pv03_vals, width=width, label="PV03")
    bars_pv08 = ax.bar(x + width, pv08_vals, width=width, label="PV08")

    ax.set_title(
        "(b) Split distribution",
        fontsize=cfg.title_fontsize,
        pad=cfg.title_pad,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS, fontweight="bold")
    ax.set_ylabel("Number of tiles (log scale)", fontsize=cfg.label_fontsize, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(10, 30000)
    style_axis(ax, cfg, which="both")
    ax.legend(frameon=False, loc="upper right", prop={"size": cfg.legend_fontsize, "weight": "bold"})
    add_bar_labels(ax, [bars_pv01, bars_pv03, bars_pv08], cfg)


def plot_panel_c_mask_histogram(ax: plt.Axes, stats: dict, cfg: DatasetFigureConfig) -> None:
    bins = np.linspace(0, 100, 21)
    yfmt = FuncFormatter(thousands_formatter)
    subsets_draw_order = ["PV03", "PV08", "PV01"]
    mean_values = {}

    for subset in subsets_draw_order:
        values = stats["coverage_by_subset_pos"][subset]
        if values:
            ax.hist(
                values,
                bins=bins,
                alpha=0.95,
                color=SUBSET_COLORS[subset],
                label=subset,
            )
            mean_values[subset] = float(np.mean(values))

    for subset in subsets_draw_order:
        if subset in mean_values:
            ax.axvline(
                mean_values[subset],
                linestyle="--",
                linewidth=1.8,
                color=SUBSET_COLORS[subset],
                label=f"{subset} mean: {mean_values[subset]:.1f}%",
            )

    ax.set_title(
        "(c) Mask coverage distribution (positive tiles only)",
        fontsize=cfg.title_fontsize,
        pad=cfg.title_pad,
        fontweight="bold",
    )
    ax.set_xlabel("Mask coverage (%)", fontsize=cfg.label_fontsize, fontweight="bold")
    ax.set_ylabel("Number of tiles", fontsize=cfg.label_fontsize, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.yaxis.set_major_formatter(yfmt)
    style_axis(ax, cfg)

    handles, labels = ax.get_legend_handles_labels()
    desired_order = [
        "PV01",
        "PV03",
        "PV08",
        *(f"{subset} mean: {mean_values[subset]:.1f}%" for subset in SUBSETS if subset in mean_values),
    ]

    ordered_handles = []
    ordered_labels = []
    for target_label in desired_order:
        for handle, label in zip(handles, labels):
            if label == target_label:
                ordered_handles.append(handle)
                ordered_labels.append(label)
                break

    ax.legend(
        ordered_handles,
        ordered_labels,
        frameon=False,
        prop={"size": cfg.annotation_fontsize, "weight": "bold"},
        bbox_to_anchor=(0.95, 0.98)
    )

def plot_panel_d_mask_boxplot(ax: plt.Axes, stats: dict, cfg: DatasetFigureConfig) -> None:
    boxplot_data = [stats["coverage_by_subset_pos"][subset] for subset in SUBSETS]
    means = [np.mean(values) if values else np.nan for values in boxplot_data]

    ax.boxplot(
        boxplot_data,
        labels=SUBSETS,
        patch_artist=False,
        showfliers=False,
    )
    ax.plot([1, 2, 3], means, marker="o", linestyle="None", label="Mean")

    ax.set_title(
        "(d) Mask coverage by subset (positive tiles only)",
        fontsize=cfg.title_fontsize,
        pad=cfg.title_pad,
        fontweight="bold",
    )
    ax.set_xlabel("Subset", fontsize=cfg.label_fontsize, fontweight="bold")
    ax.set_ylabel("Mask coverage (%)", fontsize=cfg.label_fontsize, fontweight="bold")
    ax.set_ylim(0, 100)
    style_axis(ax, cfg)
    ax.legend(frameon=False, prop={"size": cfg.legend_fontsize, "weight": "bold"})


# -----------------------------
# Combined figure
# -----------------------------
def make_figure_2_grid(records: List[dict], cfg: DatasetFigureConfig) -> plt.Figure:
    stats = aggregate_dataset_stats(records)

    fig = plt.figure(figsize=cfg.figure_size)
    gs = fig.add_gridspec(2, 2, wspace=cfg.panel_wspace, hspace=cfg.panel_hspace)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_panel_a_tile_distribution(ax_a, stats, cfg)
    plot_panel_b_split_distribution(ax_b, stats, cfg)
    plot_panel_c_mask_histogram(ax_c, stats, cfg)
    plot_panel_d_mask_boxplot(ax_d, stats, cfg)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        set_axis_text_bold(ax)

    # fig.tight_layout(rect=cfg.tight_layout_rect)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.94,
                        wspace=0.22, hspace=0.28)
    return fig


# -----------------------------
# Summary
# -----------------------------
def print_summary(records: List[dict]) -> None:
    print("\n=== DATASET SUMMARY ===")
    print(f"Total tiles: {len(records)}")

    for subset in SUBSETS:
        subset_records = [record for record in records if record["subset"] == subset]
        positives = sum(record["is_positive"] for record in subset_records)
        negatives = len(subset_records) - positives
        mean_coverage = np.mean([record["coverage"] for record in subset_records]) if subset_records else 0.0
        print(
            f"{subset}: total={len(subset_records)}, positive={positives}, "
            f"negative={negatives}, mean_coverage={mean_coverage:.4f}"
        )

    print("\n=== SPLIT SUMMARY ===")
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        print(f"{split}: {len(split_records)} tiles")


# -----------------------------
# Run
# -----------------------------
def run(cfg: DatasetFigureConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    outdir = ensure_outdir(cfg.outdir)
    records = collect_dataset_records(cfg.dataset_root)

    print_summary(records)

    fig = make_figure_2_grid(records, cfg)
    save_fig(fig, outdir, cfg.save_stem, dpi=cfg.dpi)
    plt.show()
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    run(CFG)
