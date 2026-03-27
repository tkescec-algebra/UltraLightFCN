import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# =========================================================
# CONFIG
# =========================================================
# Adjust this path to your final dataset root produced by preprocessing.
# Expected structure:
# dataset/
#   train/
#       images/
#       masks/
#   valid/
#       images/
#       masks/
#   test/
#       images/
#       masks/
DATASET_ROOT = Path("../dataset")

# Number of example tiles per subset in panel (d)
EXAMPLES_PER_SUBSET = 1

# Reproducibility
random.seed(42)
np.random.seed(42)


# =========================================================
# HELPERS
# =========================================================
def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask as binary numpy array {0,1}."""
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    return (mask_np > 0).astype(np.uint8)


def load_image(img_path: Path) -> np.ndarray:
    """Load RGB image as numpy array."""
    img = Image.open(img_path).convert("RGB")
    return np.array(img)


def mask_coverage(mask_np: np.ndarray) -> float:
    """Fraction of positive pixels in a tile."""
    return float((mask_np > 0).mean())


def get_subset_from_name(filename: str) -> str:
    """
    Extract subset label from filename.
    Works for names containing PV01, PV03, PV08.
    """
    upper = filename.upper()
    for sub in ["PV01", "PV03", "PV08"]:
        if sub in upper:
            return sub
    return "OTHER"


def collect_dataset_records(dataset_root: Path):
    """
    Scan dataset root and collect records for train/valid/test.

    Expected structure:
    dataset/
        train/
            image1.png
            image1_label.png
            image2.png
            image2_label.png
        valid/
            ...
        test/
            ...

    Mask files are identified by '_label' before the file extension.
    """
    records = []
    valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    for split in ["train", "valid", "test"]:
        split_dir = dataset_root / split

        if not split_dir.exists():
            print(f"[WARNING] Missing split folder: {split_dir}")
            continue

        all_files = sorted([p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

        # Keep only original images, not masks
        image_files = [p for p in all_files if "_label" not in p.stem]

        for img_path in image_files:
            mask_name = f"{img_path.stem}_label{img_path.suffix}"
            mask_path = split_dir / mask_name

            if not mask_path.exists():
                print(f"[WARNING] Missing mask for image: {img_path.name}")
                continue

            subset = get_subset_from_name(img_path.name)
            mask_np = load_mask(mask_path)
            cov = mask_coverage(mask_np)
            positive_pixels = int(mask_np.sum())
            is_pos = (cov >= 0.005) or (positive_pixels >= 64)

            records.append({
                "split": split,
                "subset": subset,
                "image_path": img_path,
                "mask_path": mask_path,
                "coverage": cov,
                "is_positive": is_pos
            })

    return records


def build_overlay(img_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Create red overlay for mask on image.
    """
    overlay = img_np.copy().astype(np.float32)
    red_layer = np.zeros_like(overlay)
    red_layer[..., 0] = 255  # red channel

    mask_3c = np.stack([mask_np] * 3, axis=-1).astype(bool)
    overlay[mask_3c] = (1 - alpha) * overlay[mask_3c] + alpha * red_layer[mask_3c]
    return overlay.astype(np.uint8)


def choose_example_records(records, subsets=("PV01", "PV03", "PV08"), n_per_subset=1):
    """
    Pick positive examples for panel (d), one per subset by default.
    Falls back to any example from that subset if no positive exists.
    """
    selected = []

    for subset in subsets:
        subset_records = [r for r in records if r["subset"] == subset]
        pos_records = [r for r in subset_records if r["is_positive"]]

        if len(pos_records) >= n_per_subset:
            chosen = random.sample(pos_records, n_per_subset)
        elif len(pos_records) > 0:
            chosen = pos_records
        elif len(subset_records) > 0:
            chosen = random.sample(subset_records, min(n_per_subset, len(subset_records)))
        else:
            chosen = []

        selected.extend(chosen)

    return selected


# =========================================================
# PLOTTING
# =========================================================
def plot_figure_2(records, save_base="figure_2_dataset_description"):
    subsets = ["PV01", "PV03", "PV08"]
    splits = ["train", "valid", "test"]

    # -------------------------
    # Aggregations
    # -------------------------
    pos_neg_by_subset = {sub: {"positive": 0, "negative": 0} for sub in subsets}
    split_counts = {split: {sub: 0 for sub in subsets} for split in splits}
    coverage_by_subset_pos = {sub: [] for sub in subsets}

    for r in records:
        subset = r["subset"]
        split = r["split"]

        if subset not in subsets:
            continue
        if split not in splits:
            continue

        if r["is_positive"]:
            pos_neg_by_subset[subset]["positive"] += 1
            coverage_by_subset_pos[subset].append(r["coverage"] * 100.0)  # percent
        else:
            pos_neg_by_subset[subset]["negative"] += 1

        split_counts[split][subset] += 1

    # -------------------------
    # Figure setup: 2x2 layout
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    def thousands_formatter(x, pos):
        return f"{int(x):,}"

    yfmt = FuncFormatter(thousands_formatter)

    # =====================================================
    # (a) Positive / Negative by subset with log y-scale
    # =====================================================
    x = np.arange(len(subsets))
    positives = [pos_neg_by_subset[sub]["positive"] for sub in subsets]
    negatives = [pos_neg_by_subset[sub]["negative"] for sub in subsets]
    width = 0.34

    bars1 = ax1.bar(x - width / 2, positives, width=width, label="Positive")
    bars2 = ax1.bar(x + width / 2, negatives, width=width, label="Negative")

    ax1.set_title("(a) Tile distribution by subset", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subsets, fontsize=10)
    ax1.set_ylabel("Number of tiles (log scale)", fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, which="both")
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="upper right", fontsize=9)

    # Optional: define visible limits
    ax1.set_ylim(10, 30000)

    # Add values above bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.08,
                f"{int(h):,}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # =====================================================
    # (b) Split distribution by subset with log y-scale
    # =====================================================
    x2 = np.arange(len(splits))
    width2 = 0.24

    pv01_vals = [split_counts[sp]["PV01"] for sp in splits]
    pv03_vals = [split_counts[sp]["PV03"] for sp in splits]
    pv08_vals = [split_counts[sp]["PV08"] for sp in splits]

    bars_pv01 = ax2.bar(x2 - width2, pv01_vals, width=width2, label="PV01")
    bars_pv03 = ax2.bar(x2, pv03_vals, width=width2, label="PV03")
    bars_pv08 = ax2.bar(x2 + width2, pv08_vals, width=width2, label="PV08")

    ax2.set_title("(b) Split distribution", fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(splits, fontsize=10)
    ax2.set_ylabel("Number of tiles (log scale)", fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(axis="y", linestyle="--", alpha=0.35, which="both")
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)

    # Optional limits; adjust if needed after rendering
    ax2.set_ylim(10, 30000)

    # Add values above bars
    for bars in [bars_pv01, bars_pv03, bars_pv08]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.08,
                f"{int(h):,}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # =====================================================
    # (c) Mask coverage distribution (positive tiles only)
    # =====================================================
    bins = np.linspace(0, 100, 21)  # 20 bins, in percent

    # Drawing order chosen for better visibility in the histogram
    subsets_c = ["PV03", "PV08", "PV01"]

    # Consistent colors by subset
    subset_colors = {
        "PV01": "tab:blue",
        "PV03": "tab:orange",
        "PV08": "tab:green",
    }

    mean_values = {}

    # Draw histograms
    for sub in subsets_c:
        vals = coverage_by_subset_pos[sub]
        if len(vals) > 0:
            ax3.hist(
                vals,
                bins=bins,
                alpha=0.95,
                color=subset_colors[sub],
                label=sub
            )
            mean_values[sub] = float(np.mean(vals))

    # Draw mean lines using the same color as the corresponding subset
    for sub in subsets_c:
        if sub in mean_values:
            ax3.axvline(
                mean_values[sub],
                linestyle="--",
                linewidth=1.8,
                color=subset_colors[sub],
                label=f"{sub} mean: {mean_values[sub]:.1f}%"
            )

    ax3.set_title("(c) Mask coverage distribution (positive tiles only)", fontsize=12)
    ax3.set_xlabel("Mask coverage (%)", fontsize=10)
    ax3.set_ylabel("Number of tiles", fontsize=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.95)
    ax3.set_axisbelow(True)
    ax3.set_xlim(0, 100)
    ax3.yaxis.set_major_formatter(yfmt)

    # Reorder legend independently from drawing order
    handles, labels = ax3.get_legend_handles_labels()

    desired_order = [
        "PV01",
        "PV03",
        "PV08",
        f"PV01 mean: {mean_values['PV01']:.1f}%",
        f"PV03 mean: {mean_values['PV03']:.1f}%",
        f"PV08 mean: {mean_values['PV08']:.1f}%"
    ]

    ordered_handles = []
    ordered_labels = []

    for target_label in desired_order:
        for h, l in zip(handles, labels):
            if l == target_label:
                ordered_handles.append(h)
                ordered_labels.append(l)
                break

    ax3.legend(ordered_handles, ordered_labels, frameon=False, fontsize=8)

    # =====================================================
    # (d) Boxplot of mask coverage by subset (positive tiles only)
    # =====================================================
    boxplot_data = [coverage_by_subset_pos[sub] for sub in subsets]

    bp = ax4.boxplot(
        boxplot_data,
        labels=subsets,
        patch_artist=False,
        showfliers=False
    )

    # Mean markers
    means = [
        np.mean(coverage_by_subset_pos[sub]) if len(coverage_by_subset_pos[sub]) > 0 else np.nan
        for sub in subsets
    ]
    ax4.plot([1, 2, 3], means, marker="o", linestyle="None", label="Mean")

    ax4.set_title("(d) Mask coverage by subset (positive tiles only)", fontsize=12)
    ax4.set_xlabel("Subset", fontsize=10)
    ax4.set_ylabel("Mask coverage (%)", fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.grid(axis="y", linestyle="--", alpha=0.35)
    ax4.set_axisbelow(True)
    ax4.legend(frameon=False, fontsize=9)

    # -------------------------
    # Final layout and saving
    # -------------------------
    # fig.suptitle("Figure 2. Dataset description", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = f"{save_base}.png"
    pdf_path = f"{save_base}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print(f"[INFO] Figure saved to: {png_path}")
    print(f"[INFO] Figure saved to: {pdf_path}")

# =========================================================
# OPTIONAL: PRINT NUMERICAL SUMMARY
# =========================================================
def print_summary(records):
    subsets = ["PV01", "PV03", "PV08"]
    splits = ["train", "valid", "test"]

    print("\n=== DATASET SUMMARY ===")
    print(f"Total tiles: {len(records)}")

    for sub in subsets:
        sub_records = [r for r in records if r["subset"] == sub]
        pos = sum(r["is_positive"] for r in sub_records)
        neg = len(sub_records) - pos
        mean_cov = np.mean([r["coverage"] for r in sub_records]) if sub_records else 0.0
        print(f"{sub}: total={len(sub_records)}, positive={pos}, negative={neg}, mean_coverage={mean_cov:.4f}")

    print("\n=== SPLIT SUMMARY ===")
    for split in splits:
        split_records = [r for r in records if r["split"] == split]
        print(f"{split}: {len(split_records)} tiles")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    records = collect_dataset_records(DATASET_ROOT)
    print_summary(records)
    plot_figure_2(records, save_base="figure_2_dataset_description")