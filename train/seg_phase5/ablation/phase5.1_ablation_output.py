### Caption for the figure:
# Best and final validation hard@0.5 scores for the evaluated encoder variants. Each point represents one random seed.
# “Best” denotes the validation hard@0.5 score at the epoch where the moving average of the last k validation soft Dice scores (avg_last_k_soft) achieved its maximum,
# whereas “final” denotes the validation hard@0.5 score at the last training epoch. Black overlays indicate the group mean ± standard deviation across seeds. The y-axis is broken to improve readability.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# Configuration
# =========================================================
CSV_PATH = "phase5.1_ablation_results.csv"   # change if needed
OUTPUT_DIR = "ablation_figures"
DPI = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication-friendly labels
VARIANT_LABELS = {
    "ablation1_pretrained_freeze": "Freeze",
    "ablation2_scratch_finetune": "Finetune",
}

# Broken y-axis ranges
Y_BOTTOM = (0.7065, 0.7175)
Y_TOP = (0.8925, 0.9045)

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(CSV_PATH)

required_columns = [
    "variant",
    "seed",
    "best_epoch",
    "best_val_soft",
    "best_val_hard05",
    "final_val_soft",
    "final_val_hard05",
]
missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

variant_order = list(df["variant"].drop_duplicates())

def pretty_variant_name(variant_name):
    return VARIANT_LABELS.get(variant_name, variant_name.replace("_", " "))

# =========================================================
# Helpers
# =========================================================
def deterministic_offsets(n, spread=0.035):
    """Create fixed offsets so points do not overlap too much."""
    if n == 1:
        return np.array([0.0])
    return np.linspace(-spread, spread, n)

def add_mean_sd_overlay(ax, x, values, width=0.09, marker="o", color="black"):
    """Draw group mean and standard deviation."""
    mean_val = np.mean(values)
    sd_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

    ax.errorbar(
        x,
        mean_val,
        yerr=sd_val,
        fmt=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
        capsize=4,
        elinewidth=1.6,
        markersize=7,
        linewidth=1.6,
        zorder=4,
    )
    ax.hlines(
        mean_val,
        x - width,
        x + width,
        linewidth=2.0,
        zorder=4,
        color=color,
    )

def mean_sd_table(dataframe, metrics, group_col="variant"):
    """Create summary table with mean, SD, and formatted mean ± SD."""
    grouped = dataframe.groupby(group_col)[metrics]
    mean_df = grouped.mean()
    std_df = grouped.std(ddof=1)

    out = pd.DataFrame(index=mean_df.index)
    for metric in metrics:
        out[f"{metric}_mean"] = mean_df[metric]
        out[f"{metric}_sd"] = std_df[metric]
        out[f"{metric}_mean±sd"] = [
            f"{mean_df.loc[idx, metric]:.4f} ± {std_df.loc[idx, metric]:.4f}"
            for idx in mean_df.index
        ]
    return out.reset_index()

def style_axis(ax):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

# =========================================================
# Summary table
# =========================================================
metrics = [
    "best_val_soft",
    "best_val_hard05",
    "final_val_soft",
    "final_val_hard05",
    "best_epoch",
]

summary_df = mean_sd_table(df, metrics, group_col="variant")
summary_df["variant"] = summary_df["variant"].map(pretty_variant_name)

summary_csv_path = os.path.join(OUTPUT_DIR, "summary_table.csv")
summary_tex_path = os.path.join(OUTPUT_DIR, "summary_table.tex")

summary_df.to_csv(summary_csv_path, index=False)
with open(summary_tex_path, "w", encoding="utf-8") as f:
    f.write(summary_df.to_latex(index=False, escape=False))

# =========================================================
# Figure: Best vs final validation performance with broken y-axis
# =========================================================
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(8.6, 6.6),
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05}
)

metric_best = "best_val_hard05"
metric_final = "final_val_hard05"
delta = 0.16

best_color = "tab:blue"
final_color = "tab:orange"

for i, variant in enumerate(variant_order):
    subset = df[df["variant"] == variant].sort_values("seed")

    y_best = subset[metric_best].values
    y_final = subset[metric_final].values

    offsets_best = deterministic_offsets(len(y_best), spread=0.03)
    offsets_final = deterministic_offsets(len(y_final), spread=0.03)

    x_best = np.full(len(y_best), i - delta) + offsets_best
    x_final = np.full(len(y_final), i + delta) + offsets_final

    for ax in [ax_top, ax_bottom]:
        # Individual seed points
        ax.scatter(
            x_best,
            y_best,
            s=62,
            alpha=0.85,
            marker="o",
            zorder=3,
            color=best_color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.scatter(
            x_final,
            y_final,
            s=62,
            alpha=0.85,
            marker="s",
            zorder=3,
            color=final_color,
            edgecolor="black",
            linewidth=0.5,
        )

        # Mean ± SD overlays in black
        add_mean_sd_overlay(ax, i - delta, y_best, width=0.09, marker="o", color="black")
        add_mean_sd_overlay(ax, i + delta, y_final, width=0.09, marker="s", color="black")

# Set broken ranges
ax_bottom.set_ylim(*Y_BOTTOM)
ax_top.set_ylim(*Y_TOP)

# Axis styling
style_axis(ax_top)
style_axis(ax_bottom)

# Hide spines between axes
ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)

# Ticks
ax_top.tick_params(labeltop=False, bottom=False)
ax_bottom.xaxis.tick_bottom()

# Diagonal break marks
d = 0.008
kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.2)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Labels and title
ax_bottom.set_xticks(range(len(variant_order)))
ax_bottom.set_xticklabels([pretty_variant_name(v) for v in variant_order], rotation=0)

ax_top.set_title("Best and final validation hard@0.5 scores")
ax_bottom.set_xlabel("Encoder variant")
fig.supylabel("Validation hard@0.5 score")

# Legend
legend_elements = [
    Line2D(
        [0], [0],
        marker="o", linestyle="None", markersize=7,
        markerfacecolor=best_color, markeredgecolor="black",
        label="Best validation score (per-seed runs)"
    ),
    Line2D(
        [0], [0],
        marker="s", linestyle="None", markersize=7,
        markerfacecolor=final_color, markeredgecolor="black",
        label="Final validation score (per-seed runs)"
    ),
    Line2D(
        [0], [0],
        marker="o", linestyle="-", markersize=7,
        color="black", markerfacecolor="black", markeredgecolor="black",
        label="Group mean ± SD across seeds"
    ),
]

ax_top.legend(handles=legend_elements, frameon=True, loc="upper left")

plt.tight_layout()

figure_png_path = os.path.join(OUTPUT_DIR, "figure_best_vs_final_hard05.png")
figure_pdf_path = os.path.join(OUTPUT_DIR, "figure_best_vs_final_hard05.pdf")

plt.savefig(figure_png_path, dpi=DPI, bbox_inches="tight")
plt.savefig(figure_pdf_path, bbox_inches="tight")
plt.close()

# =========================================================
# Console summary
# =========================================================
print("Saved files:")
print(summary_csv_path)
print(summary_tex_path)
print(figure_png_path)
print(figure_pdf_path)

print("\nDone.")

