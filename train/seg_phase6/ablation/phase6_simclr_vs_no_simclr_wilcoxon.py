"""Paired SimCLR vs no-SimCLR Wilcoxon analysis for Phase 6 TEST results.

This script compares final Phase 6 held-out TEST metrics across paired seeds.
All configurable values live in the dataclass below; there are no CLI options.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
from matplotlib.ticker import FormatStrFormatter

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class WilcoxonAblationConfig:
    simclr_csv: str = "../final_retrain90/phase6_test_per_seed.csv"
    no_simclr_csv: str = "../final_retrain90_fixed_recipe_no_simclr/phase6_test_per_seed_fixed_recipe_no_simclr.csv"
    out_dir: str = "analysis/simclr_no_simclr_wilcoxon"
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 42
    make_slope_plot: bool = True
    make_beeswarm_boxplot: bool = True
    beeswarm_jitter_seed: int = 42
    box_width: float = 0.35
    expected_n_pairs: int = 20
    dpi: int = 300


METRICS = (
    "test_hard_dice@0.5",
    "test_hard_iou@0.5",
    "test_precision@0.5",
    "test_recall@0.5",
)

PANEL_TITLES = {
    "test_hard_dice@0.5": "Mask overlap",
    "test_hard_iou@0.5": "Mask overlap",
    "test_precision@0.5": "Positive prediction precision",
    "test_recall@0.5": "Positive region recall",
}

METRIC_YLABELS = {
    "test_hard_dice@0.5": "Hard Dice@0.5",
    "test_hard_iou@0.5": "Hard IoU@0.5",
    "test_precision@0.5": "Precision@0.5",
    "test_recall@0.5": "Recall@0.5",
}

PANEL_LABELS = ("(A)", "(B)", "(C)", "(D)")

MAIN_BEESWARM_METRICS = (
    "test_hard_dice@0.5",
    "test_precision@0.5",
    "test_recall@0.5",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_input_path(path_text: str, condition: str) -> Path:
    raw_path = Path(path_text)
    root = repo_root()

    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw_path,
                root / raw_path,
            ]
        )
        if condition == "simclr":
            candidates.append(root / "train" / "seg_phase6" / "final_retrain90" / raw_path)
        elif condition == "no_simclr":
            candidates.append(
                root
                / "train"
                / "seg_phase6"
                / "final_retrain90_fixed_recipe_no_simclr"
                / raw_path
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {condition} CSV '{path_text}'. Checked:\n  - {checked}")


def resolve_output_dir(path_text: str) -> Path:
    output_dir = Path(path_text)
    if not output_dir.is_absolute():
        output_dir = repo_root() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def find_seed_column(df: pd.DataFrame, label: str) -> str:
    normalized = {str(column).strip().lower(): column for column in df.columns}
    if "seed" not in normalized:
        raise ValueError(f"{label} must contain a seed column named 'seed'.")
    return str(normalized["seed"])


def validate_required_columns(df: pd.DataFrame, label: str, metrics: Iterable[str]) -> str:
    seed_column = find_seed_column(df, label)
    missing = [metric for metric in metrics if metric not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required metric columns: {', '.join(missing)}")
    return seed_column


def validate_no_duplicate_seeds(df: pd.DataFrame, seed_column: str, label: str) -> None:
    duplicated = df.loc[df[seed_column].duplicated(), seed_column].tolist()
    if duplicated:
        raise ValueError(f"{label} contains duplicate seeds: {duplicated}")


def numeric_metric_series(series: pd.Series, label: str, metric: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise ValueError(f"{label} metric '{metric}' contains non-numeric values.") from exc
    if numeric.isna().any():
        raise ValueError(f"{label} metric '{metric}' contains missing values.")
    return numeric.astype(float)


def load_paired_data(config: WilcoxonAblationConfig) -> pd.DataFrame:
    simclr_path = resolve_input_path(config.simclr_csv, "simclr")
    no_simclr_path = resolve_input_path(config.no_simclr_csv, "no_simclr")

    simclr_df = pd.read_csv(simclr_path)
    no_simclr_df = pd.read_csv(no_simclr_path)

    simclr_seed = validate_required_columns(simclr_df, "SimCLR CSV", METRICS)
    no_simclr_seed = validate_required_columns(no_simclr_df, "no-SimCLR CSV", METRICS)
    validate_no_duplicate_seeds(simclr_df, simclr_seed, "SimCLR CSV")
    validate_no_duplicate_seeds(no_simclr_df, no_simclr_seed, "no-SimCLR CSV")

    simclr_seeds = set(simclr_df[simclr_seed].tolist())
    no_simclr_seeds = set(no_simclr_df[no_simclr_seed].tolist())
    if simclr_seeds != no_simclr_seeds:
        simclr_only = sorted(simclr_seeds - no_simclr_seeds)
        no_simclr_only = sorted(no_simclr_seeds - simclr_seeds)
        raise ValueError(
            "Seed sets do not match exactly. "
            f"Only in SimCLR: {simclr_only}; only in no-SimCLR: {no_simclr_only}"
        )

    simclr_view = simclr_df[[simclr_seed, *METRICS]].rename(columns={simclr_seed: "seed"})
    no_simclr_view = no_simclr_df[[no_simclr_seed, *METRICS]].rename(columns={no_simclr_seed: "seed"})

    for metric in METRICS:
        simclr_view[metric] = numeric_metric_series(simclr_view[metric], "SimCLR CSV", metric)
        no_simclr_view[metric] = numeric_metric_series(no_simclr_view[metric], "no-SimCLR CSV", metric)

    paired = pd.merge(
        simclr_view,
        no_simclr_view,
        on="seed",
        how="inner",
        suffixes=("_simclr", "_no_simclr"),
        validate="one_to_one",
    )

    if len(paired) != config.expected_n_pairs:
        raise ValueError(
            f"Expected {config.expected_n_pairs} paired seeds, but merge produced {len(paired)}."
        )

    numeric_seed = pd.to_numeric(paired["seed"], errors="coerce")
    if numeric_seed.notna().all():
        paired = paired.assign(_sort_seed=numeric_seed).sort_values("_sort_seed")
        paired = paired.drop(columns="_sort_seed")
    else:
        paired = paired.sort_values("seed")
    return paired.reset_index(drop=True)


def bootstrap_mean_ci(differences: np.ndarray, config: WilcoxonAblationConfig) -> tuple[float, float]:
    rng = np.random.default_rng(config.bootstrap_seed)
    n = differences.size
    samples = rng.choice(differences, size=(config.bootstrap_resamples, n), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def build_paired_differences(paired: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"seed": paired["seed"]})
    for metric in METRICS:
        simclr_col = f"{metric}_simclr"
        no_simclr_col = f"{metric}_no_simclr"
        output[f"simclr_{metric}"] = paired[simclr_col]
        output[f"no_simclr_{metric}"] = paired[no_simclr_col]
        output[f"diff_{metric}"] = paired[simclr_col] - paired[no_simclr_col]
    return output


def summarize_metric(
    paired: pd.DataFrame, metric: str, config: WilcoxonAblationConfig
) -> dict[str, float | int | str]:
    simclr = paired[f"{metric}_simclr"].to_numpy(dtype=float)
    no_simclr = paired[f"{metric}_no_simclr"].to_numpy(dtype=float)
    differences = simclr - no_simclr
    ci_low, ci_high = bootstrap_mean_ci(differences, config)

    one_sided = wilcoxon(differences, alternative="greater")
    two_sided = wilcoxon(differences, alternative="two-sided")

    return {
        "metric": metric,
        "n_pairs": int(differences.size),
        "simclr_mean": float(np.mean(simclr)),
        "simclr_std": float(np.std(simclr, ddof=1)),
        "no_simclr_mean": float(np.mean(no_simclr)),
        "no_simclr_std": float(np.std(no_simclr, ddof=1)),
        "mean_diff": float(np.mean(differences)),
        "std_diff": float(np.std(differences, ddof=1)),
        "median_diff": float(np.median(differences)),
        "n_simclr_better": int(np.sum(differences > 0)),
        "n_equal": int(np.sum(differences == 0)),
        "n_no_simclr_better": int(np.sum(differences < 0)),
        "wilcoxon_statistic": float(one_sided.statistic),
        "wilcoxon_p_one_sided": float(one_sided.pvalue),
        "wilcoxon_p_two_sided": float(two_sided.pvalue),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
    }


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "font.size": 10,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_axis_text_bold(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def format_pvalue(value: float) -> str:
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def save_main_difference_plot(
    paired_diffs: pd.DataFrame,
    results_df: pd.DataFrame,
    out_dir: Path,
    config: WilcoxonAblationConfig,
) -> list[Path]:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes_flat = axes.ravel()

    for index, (ax, metric) in enumerate(zip(axes_flat, METRICS)):
        differences = paired_diffs[f"diff_{metric}"].to_numpy(dtype=float)
        x = np.arange(1, len(differences) + 1)
        row = results_df.loc[results_df["metric"] == metric].iloc[0]

        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
        ax.axhspan(
            row["bootstrap_ci95_low"],
            row["bootstrap_ci95_high"],
            color="#9ecae1",
            alpha=0.28,
            linewidth=0,
            label="95% bootstrap CI",
            zorder=0,
        )
        ax.axhline(
            row["mean_diff"],
            color="#1f4e79",
            linewidth=1.8,
            label="Mean paired difference",
            zorder=2,
        )
        ax.scatter(
            x,
            differences,
            s=34,
            color="#222222",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )

        annotation = (
            f"n = {int(row['n_pairs'])}\n"
            f"mean diff = {row['mean_diff']:.4f}\n"
            f"95% CI [{row['bootstrap_ci95_low']:.4f}, {row['bootstrap_ci95_high']:.4f}]\n"
            f"one-sided p = {format_pvalue(row['wilcoxon_p_one_sided'])}\n"
        )
        ax.text(
            0.98,
            0.04,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb"},
        )

        ax.set_title(f"{PANEL_TITLES[metric]}", loc="left", fontweight="bold")
        ax.set_xlabel("Paired seed index", fontweight="bold")
        ax.set_ylabel(f"Δ {METRIC_YLABELS[metric]}", fontweight="bold")
        ax.set_xlim(0.25, len(differences) + 0.75)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
        make_axis_text_bold(ax)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    # fig.suptitle(
    #     "Seed-paired held-out test-set differences (Δ = SimCLR − no-SimCLR)",
    #     y=1.06,
    #     fontweight="bold",
    # )

    png_path = out_dir / "fig_simclr_no_simclr_paired_differences.png"
    pdf_path = out_dir / "fig_simclr_no_simclr_paired_differences.pdf"
    fig.savefig(png_path, dpi=config.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def save_slope_plot(
    paired: pd.DataFrame,
    out_dir: Path,
    config: WilcoxonAblationConfig,
) -> list[Path]:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7), constrained_layout=True)
    axes_flat = axes.ravel()

    for index, (ax, metric) in enumerate(zip(axes_flat, METRICS)):
        simclr = paired[f"{metric}_simclr"].to_numpy(dtype=float)
        no_simclr = paired[f"{metric}_no_simclr"].to_numpy(dtype=float)
        differences = simclr - no_simclr

        for no_value, simclr_value, diff in zip(no_simclr, simclr, differences):
            color = "#1f4e79" if diff > 0 else "#9b2226" if diff < 0 else "#777777"
            ax.plot([0, 1], [no_value, simclr_value], color=color, alpha=0.42, linewidth=1.0)
            ax.scatter([0, 1], [no_value, simclr_value], color=color, s=16, alpha=0.75)

        ax.plot(
            [0, 1],
            [float(np.mean(no_simclr)), float(np.mean(simclr))],
            color="#000000",
            linewidth=2.4,
            marker="o",
            markersize=5,
            label="Mean",
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["no-SimCLR", "SimCLR"], fontweight="bold")
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(f"{PANEL_LABELS[index]} {PANEL_TITLES[metric]}", loc="left", fontweight="bold")
        ax.set_ylabel(METRIC_YLABELS[metric], fontweight="bold")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
        make_axis_text_bold(ax)
        if index == 0:
            legend = ax.legend(frameon=False, loc="best")
            for text in legend.get_texts():
                text.set_fontweight("bold")

    # fig.suptitle(
    #     "Seed-paired held-out test-set metrics by initialization condition",
    #     y=1.03,
    #     fontweight="bold",
    # )
    png_path = out_dir / "fig_simclr_no_simclr_paired_slope.png"
    pdf_path = out_dir / "fig_simclr_no_simclr_paired_slope.pdf"
    fig.savefig(png_path, dpi=config.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def save_beeswarm_boxplot_main_metrics(
    paired: pd.DataFrame,
    results_df: pd.DataFrame,
    out_dir: Path,
    config: WilcoxonAblationConfig,
) -> list[Path]:
    configure_plot_style()
    rng = np.random.default_rng(config.beeswarm_jitter_seed)
    fig, axes = plt.subplots(1, 3, figsize=(8, 4), constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.05,
        h_pad=0.06,
        wspace=0.00,
        hspace=0.00,
    )
    title_size = 7
    label_size = 7
    tick_size = 7
    annotation_size = 7

    for index, (ax, metric) in enumerate(zip(axes, MAIN_BEESWARM_METRICS)):
        no_simclr = paired[f"{metric}_no_simclr"].to_numpy(dtype=float)
        simclr = paired[f"{metric}_simclr"].to_numpy(dtype=float)
        row = results_df.loc[results_df["metric"] == metric].iloc[0]
        positions = [0.0, 1.0]

        box = ax.boxplot(
            [no_simclr, simclr],
            positions=positions,
            widths=config.box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.6},
            boxprops={"facecolor": "#eef3f7", "edgecolor": "#4f6070", "linewidth": 1.2},
            whiskerprops={"color": "#4f6070", "linewidth": 1.1},
            capprops={"color": "#4f6070", "linewidth": 1.1},
        )
        box["boxes"][1].set_facecolor("#eaf2ea")

        for x_pos, values, color in (
            (positions[0], no_simclr, "#8b1e24"),
            (positions[1], simclr, "#1f4e79"),
        ):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), x_pos) + jitter,
                values,
                s=16,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.82,
                zorder=3,
            )
            ax.scatter(
                x_pos,
                float(np.mean(values)),
                s=36,
                marker="D",
                color="#111111",
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )

        annotation = (
            f"p = {format_pvalue(row['wilcoxon_p_one_sided'])}\n"
            f"mean $\\Delta$ = {row['mean_diff']:+.4f}\n"
            f"95% CI [{row['bootstrap_ci95_low']:.4f}, {row['bootstrap_ci95_high']:.4f}]"

        )
        ax.text(
            0.70,
            0.80,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=annotation_size,
            fontweight="bold",
        )

        ax.set_title(
            f"{PANEL_LABELS[index]} {PANEL_TITLES[metric]}",
            loc="left",
            fontweight="bold",
            fontsize=title_size,
        )
        ax.set_ylabel(METRIC_YLABELS[metric], fontweight="bold", fontsize=label_size)
        ax.set_xticks(positions)
        ax.set_xticklabels(["Non-pretrained", "SimCLR-initialized"], fontweight="bold", fontsize=tick_size)
        ax.tick_params(axis="y", labelsize=tick_size)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlim(-0.45, 1.45)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
        make_axis_text_bold(ax)

    png_path = out_dir / "fig_simclr_no_simclr_beeswarm_boxplot_main_metrics.png"
    pdf_path = out_dir / "fig_simclr_no_simclr_beeswarm_boxplot_main_metrics.pdf"
    fig.savefig(png_path, dpi=config.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    config = WilcoxonAblationConfig()
    out_dir = resolve_output_dir(config.out_dir)

    paired = load_paired_data(config)
    paired_diffs = build_paired_differences(paired)
    results = [summarize_metric(paired, metric, config) for metric in METRICS]
    results_df = pd.DataFrame(results)

    results_path = out_dir / "simclr_no_simclr_wilcoxon_results.csv"
    paired_diffs_path = out_dir / "simclr_no_simclr_paired_differences.csv"
    results_df.to_csv(results_path, index=False)
    paired_diffs.to_csv(paired_diffs_path, index=False)

    plot_paths = save_main_difference_plot(paired_diffs, results_df, out_dir, config)
    beeswarm_paths: list[Path] = []
    if config.make_beeswarm_boxplot:
        beeswarm_paths = save_beeswarm_boxplot_main_metrics(paired, results_df, out_dir, config)
    slope_paths: list[Path] = []
    if config.make_slope_plot:
        slope_paths = save_slope_plot(paired, out_dir, config)

    print(f"Merged paired seeds: {len(paired)}")
    print(f"Saved results table: {results_path}")
    print(f"Saved paired differences: {paired_diffs_path}")
    print("Saved figures:")
    for path in [*plot_paths, *beeswarm_paths, *slope_paths]:
        print(f"  {path}")
    print("\nOne-sided paired Wilcoxon p-values, alternative SimCLR > no-SimCLR:")
    for _, row in results_df.iterrows():
        print(f"  {row['metric']}: p={format_pvalue(float(row['wilcoxon_p_one_sided']))}")


if __name__ == "__main__":
    main()
