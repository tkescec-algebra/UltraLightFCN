"""Internal simulation planning for the SimCLR vs no-SimCLR ablation.

This script uses the currently available paired per-seed TEST results to decide
whether extending the ablation to 10 seeds is likely to be worth the compute.
It is not manuscript evidence. Manuscript-level evidence must come from the
actual extended 10-seed experiment, with the directional hypothesis defined
before evaluating the final results.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SIMCLR_CSV = "phase6_test_per_seed.csv"
DEFAULT_NOSIMCLR_CSV = "phase6_test_per_seed_fixed_recipe_no_simclr.csv"
DEFAULT_METRIC = "test_hard_dice@0.5"
DEFAULT_OUTPUT_DIR = "wilcoxon_simulation_outputs"
SEED_COLUMN_CANDIDATES = ("seed", "random_seed", "run_seed")


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float | None
    pvalue: float | None
    n_nonzero: int
    warning: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Internal simulation-based planning analysis for the paired "
            "SimCLR vs no-SimCLR ablation."
        )
    )
    parser.add_argument("--simclr-csv", default=DEFAULT_SIMCLR_CSV)
    parser.add_argument("--nosimclr-csv", default=DEFAULT_NOSIMCLR_CSV)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--target-n", type=int, default=10)
    parser.add_argument("--num-simulations", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_input_path(path_arg: str, script_dir: Path) -> Path:
    """Resolve CLI paths while supporting the existing phase-6 folder layout."""
    raw_path = Path(path_arg)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw_path,
                script_dir / raw_path,
                script_dir / "final_retrain90" / raw_path,
                script_dir / "final_retrain90_fixed_recipe_no_simclr" / raw_path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    candidate_text = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find input CSV '{path_arg}'. Checked:\n  - {candidate_text}"
    )


def resolve_output_dir(path_arg: str, script_dir: Path) -> Path:
    output_dir = Path(path_arg)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def find_seed_column(df: pd.DataFrame, label: str) -> str:
    normalized = {column.strip().lower(): column for column in df.columns}
    for candidate in SEED_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"No seed column found in {label}. Expected one of: "
        f"{', '.join(SEED_COLUMN_CANDIDATES)}"
    )


def validate_metric(df: pd.DataFrame, metric: str, label: str) -> None:
    if metric not in df.columns:
        columns = ", ".join(str(column) for column in df.columns)
        raise ValueError(
            f"Metric '{metric}' not found in {label}. Available columns: {columns}"
        )


def load_and_match_pairs(
    simclr_csv: Path, nosimclr_csv: Path, metric: str
) -> pd.DataFrame:
    simclr_df = pd.read_csv(simclr_csv)
    nosimclr_df = pd.read_csv(nosimclr_csv)

    simclr_seed_col = find_seed_column(simclr_df, "SimCLR CSV")
    nosimclr_seed_col = find_seed_column(nosimclr_df, "no-SimCLR CSV")
    validate_metric(simclr_df, metric, "SimCLR CSV")
    validate_metric(nosimclr_df, metric, "no-SimCLR CSV")

    simclr_view = simclr_df[[simclr_seed_col, metric]].rename(
        columns={simclr_seed_col: "seed", metric: "simclr"}
    )
    nosimclr_view = nosimclr_df[[nosimclr_seed_col, metric]].rename(
        columns={nosimclr_seed_col: "seed", metric: "nosimclr"}
    )

    simclr_seeds = set(simclr_view["seed"].tolist())
    nosimclr_seeds = set(nosimclr_view["seed"].tolist())
    simclr_only = sorted(simclr_seeds - nosimclr_seeds)
    nosimclr_only = sorted(nosimclr_seeds - simclr_seeds)
    if simclr_only:
        print(f"WARNING: Seeds only in SimCLR CSV and excluded: {simclr_only}")
    if nosimclr_only:
        print(f"WARNING: Seeds only in no-SimCLR CSV and excluded: {nosimclr_only}")

    matched = pd.merge(simclr_view, nosimclr_view, on="seed", how="inner")
    if matched.empty:
        raise ValueError("No matching seeds found between the two CSV files.")

    matched["simclr"] = pd.to_numeric(matched["simclr"], errors="raise")
    matched["nosimclr"] = pd.to_numeric(matched["nosimclr"], errors="raise")
    matched["difference"] = matched["simclr"] - matched["nosimclr"]
    numeric_seed = pd.to_numeric(matched["seed"], errors="coerce")
    if numeric_seed.notna().all():
        matched = matched.assign(_sort_seed=numeric_seed).sort_values("_sort_seed")
        matched = matched.drop(columns="_sort_seed")
    else:
        matched = matched.sort_values("seed")
    return matched


def rank_absolute_values(values: np.ndarray) -> np.ndarray:
    """Return average ranks for absolute values, using 1-based ranks."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def wilcoxon_signed_rank_exact(
    differences: np.ndarray, alternative: str = "greater"
) -> WilcoxonResult:
    """Exact paired Wilcoxon signed-rank test for small samples.

    Zero differences are dropped, matching the common "wilcox" convention.
    The exact null distribution is computed by dynamic programming over signed
    ranks, so SciPy is not required.
    """
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    finite_differences = np.asarray(differences, dtype=float)
    finite_differences = finite_differences[np.isfinite(finite_differences)]
    nonzero_differences = finite_differences[finite_differences != 0.0]
    n_nonzero = int(nonzero_differences.size)
    if n_nonzero == 0:
        return WilcoxonResult(
            statistic=None,
            pvalue=None,
            n_nonzero=0,
            warning="Wilcoxon test cannot be computed because all paired differences are zero.",
        )

    abs_values = np.abs(nonzero_differences)
    ranks = rank_absolute_values(abs_values)
    w_plus = float(np.sum(ranks[nonzero_differences > 0]))

    # Multiplying by 2 converts integer and half-integer average ranks to ints.
    scaled_ranks = [int(round(rank * 2.0)) for rank in ranks]
    observed = int(round(w_plus * 2.0))
    counts: Counter[int] = Counter({0: 1})
    for rank in scaled_ranks:
        updated = Counter(counts)
        for subtotal, count in counts.items():
            updated[subtotal + rank] += count
        counts = updated

    total_assignments = 2**n_nonzero
    lower_count = sum(count for subtotal, count in counts.items() if subtotal <= observed)
    upper_count = sum(count for subtotal, count in counts.items() if subtotal >= observed)

    if alternative == "greater":
        pvalue = upper_count / total_assignments
    elif alternative == "less":
        pvalue = lower_count / total_assignments
    else:
        pvalue = min(1.0, 2.0 * min(lower_count, upper_count) / total_assignments)

    return WilcoxonResult(statistic=w_plus, pvalue=float(pvalue), n_nonzero=n_nonzero)


def summarize_series(values: pd.Series | np.ndarray) -> dict[str, float]:
    series = pd.Series(values, dtype=float)
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def simulate_planning_analysis(
    observed_differences: np.ndarray,
    target_n: int,
    num_simulations: int,
    alpha: float,
    random_seed: int,
) -> dict[str, Any]:
    if target_n <= 0:
        raise ValueError("--target-n must be positive.")
    if num_simulations <= 0:
        raise ValueError("--num-simulations must be positive.")

    rng = np.random.default_rng(random_seed)
    simulated_mean_differences = np.empty(num_simulations, dtype=float)
    simulated_pvalues = np.empty(num_simulations, dtype=float)
    invalid_tests = 0

    for index in range(num_simulations):
        simulated_differences = rng.choice(
            observed_differences, size=target_n, replace=True
        )
        simulated_mean_differences[index] = float(np.mean(simulated_differences))
        result = wilcoxon_signed_rank_exact(simulated_differences, alternative="greater")
        if result.pvalue is None:
            simulated_pvalues[index] = np.nan
            invalid_tests += 1
        else:
            simulated_pvalues[index] = result.pvalue

    valid_pvalues = simulated_pvalues[np.isfinite(simulated_pvalues)]
    probability_significant = (
        float(np.mean(valid_pvalues < alpha)) if valid_pvalues.size else float("nan")
    )

    return {
        "mean_differences": simulated_mean_differences,
        "pvalues": simulated_pvalues,
        "invalid_tests": invalid_tests,
        "estimated_probability_p_lt_alpha": probability_significant,
    }


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    return plt


def bold_ticks(axis: Any) -> None:
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")


def save_observed_value_boxplot(
    matched: pd.DataFrame, metric: str, output_dir: Path, plt: Any
) -> Path:
    path = output_dir / "observed_simclr_vs_no_simclr_boxplot.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [matched["simclr"], matched["nosimclr"]],
        tick_labels=["SimCLR", "No SimCLR"],
        patch_artist=True,
        boxprops={"facecolor": "#8ecae6", "linewidth": 1.5},
        medianprops={"color": "#d00000", "linewidth": 2.0},
    )
    ax.scatter(np.ones(len(matched)), matched["simclr"], color="#023047", zorder=3)
    ax.scatter(np.ones(len(matched)) * 2, matched["nosimclr"], color="#023047", zorder=3)
    ax.set_title(f"Observed SimCLR vs No-SimCLR Values\n{metric} (n={len(matched)})")
    ax.set_xlabel("Condition")
    ax.set_ylabel(metric)
    bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_difference_boxplot(
    matched: pd.DataFrame, metric: str, output_dir: Path, plt: Any
) -> Path:
    path = output_dir / "observed_paired_differences_boxplot.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [matched["difference"]],
        tick_labels=["SimCLR - No SimCLR"],
        patch_artist=True,
        boxprops={"facecolor": "#ffb703", "linewidth": 1.5},
        medianprops={"color": "#d00000", "linewidth": 2.0},
    )
    ax.scatter(np.ones(len(matched)), matched["difference"], color="#023047", zorder=3)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_title(f"Observed Paired Differences\n{metric} (n={len(matched)})")
    ax.set_xlabel("Difference")
    ax.set_ylabel(f"{metric} difference")
    bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_simulated_mean_histogram(
    mean_differences: np.ndarray,
    metric: str,
    observed_n: int,
    target_n: int,
    output_dir: Path,
    plt: Any,
) -> Path:
    path = output_dir / "simulated_10_seed_mean_differences_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mean_differences, bins=40, color="#219ebc", edgecolor="black", alpha=0.85)
    ax.axvline(float(np.mean(mean_differences)), color="#d00000", linewidth=2.0)
    ax.set_title(
        f"Simulated Target-{target_n} Mean Differences\n"
        f"{metric} (observed n={observed_n})"
    )
    ax.set_xlabel(f"Mean paired difference in simulated {target_n}-seed experiment")
    ax.set_ylabel("Simulation count")
    bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_pvalue_histogram(
    pvalues: np.ndarray,
    alpha: float,
    target_n: int,
    output_dir: Path,
    plt: Any,
) -> Path:
    path = output_dir / "simulated_10_seed_wilcoxon_pvalues_histogram.png"
    valid_pvalues = pvalues[np.isfinite(pvalues)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid_pvalues, bins=np.linspace(0, 1, 41), color="#90be6d", edgecolor="black")
    ax.axvline(alpha, color="#d00000", linestyle="--", linewidth=2.0)
    ax.set_title(f"Simulated One-Sided Wilcoxon P-Values\nTarget n={target_n}")
    ax.set_xlabel("One-sided Wilcoxon p-value (SimCLR > no-SimCLR)")
    ax.set_ylabel("Simulation count")
    ax.annotate(
        f"alpha = {alpha:g}",
        xy=(alpha, 0.95),
        xycoords=("data", "axes fraction"),
        xytext=(8, -18),
        textcoords="offset points",
        fontweight="bold",
        color="#d00000",
    )
    bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_json_summary(path: Path, summary: dict[str, Any]) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and math.isnan(value):
            return None
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(summary, indent=2, default=default), encoding="utf-8")


def print_observed_results(
    matched: pd.DataFrame,
    metric: str,
    difference_summary: dict[str, float],
    one_sided: WilcoxonResult,
    two_sided: WilcoxonResult,
) -> None:
    print("\nINTERNAL PLANNING SIMULATION - NOT MANUSCRIPT EVIDENCE")
    print("This analysis is based only on the currently available observed seeds.")
    print(
        "The primary manuscript-level evidence must come from the actual extended "
        "10-seed experiment."
    )
    print(
        "A one-sided final test is justified only if the directional hypothesis is "
        "defined before evaluating the final 10-seed results."
    )
    print(f"\nPrimary metric: {metric}")
    print("\nMatched observed per-seed TEST values:")
    print(matched.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    print("\nObserved paired difference summary (SimCLR - no-SimCLR):")
    for key, value in difference_summary.items():
        print(f"  {key}: {value:.10f}")

    print("\nObserved paired Wilcoxon tests using currently available seeds only:")
    if one_sided.warning:
        print(f"  WARNING: {one_sided.warning}")
    else:
        print(
            "  One-sided alternative SimCLR > no-SimCLR: "
            f"W+={one_sided.statistic:.6f}, p={one_sided.pvalue:.6g}, "
            f"nonzero n={one_sided.n_nonzero}"
        )

    if two_sided.warning:
        print(f"  WARNING: {two_sided.warning}")
    else:
        print(
            "  Two-sided reference: "
            f"W+={two_sided.statistic:.6f}, p={two_sided.pvalue:.6g}, "
            f"nonzero n={two_sided.n_nonzero}"
        )


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    simclr_csv = resolve_input_path(args.simclr_csv, script_dir)
    nosimclr_csv = resolve_input_path(args.nosimclr_csv, script_dir)
    output_dir = resolve_output_dir(args.output_dir, script_dir)

    matched = load_and_match_pairs(simclr_csv, nosimclr_csv, args.metric)
    observed_differences = matched["difference"].to_numpy(dtype=float)
    difference_summary = summarize_series(observed_differences)
    one_sided = wilcoxon_signed_rank_exact(observed_differences, alternative="greater")
    two_sided = wilcoxon_signed_rank_exact(observed_differences, alternative="two-sided")

    print_observed_results(
        matched=matched,
        metric=args.metric,
        difference_summary=difference_summary,
        one_sided=one_sided,
        two_sided=two_sided,
    )

    print(
        "\nRunning simulation-based planning analysis for extending to "
        f"{args.target_n} seeds with {args.num_simulations} simulations..."
    )
    print(
        "Simulation assumption: future paired differences are sampled with "
        "replacement from the currently observed paired differences."
    )
    print(
        "Because the assumption is based on only the observed seeds, the estimated "
        "power/probability of significance is uncertain."
    )

    simulation = simulate_planning_analysis(
        observed_differences=observed_differences,
        target_n=args.target_n,
        num_simulations=args.num_simulations,
        alpha=args.alpha,
        random_seed=args.random_seed,
    )
    estimated_probability = simulation["estimated_probability_p_lt_alpha"]
    print(
        f"\nEstimated probability of one-sided Wilcoxon p < {args.alpha:g} "
        f"under this assumed effect distribution: {estimated_probability:.6f}"
    )
    if simulation["invalid_tests"]:
        print(
            "WARNING: "
            f"{simulation['invalid_tests']} simulated tests were degenerate and excluded."
        )

    observed_csv = output_dir / "observed_paired_values_and_differences.csv"
    matched.to_csv(observed_csv, index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        plt = configure_matplotlib()
        plot_paths = [
            save_observed_value_boxplot(matched, args.metric, output_dir, plt),
            save_difference_boxplot(matched, args.metric, output_dir, plt),
            save_simulated_mean_histogram(
                simulation["mean_differences"],
                args.metric,
                len(matched),
                args.target_n,
                output_dir,
                plt,
            ),
            save_pvalue_histogram(
                simulation["pvalues"], args.alpha, args.target_n, output_dir, plt
            ),
        ]

    json_summary = {
        "analysis_note": (
            "Internal simulation-based planning analysis only; not manuscript evidence."
        ),
        "methodological_notes": [
            "Simulation uses only currently available observed seeds, so estimated power is uncertain.",
            "Primary manuscript-level evidence must come from the actual extended 10-seed experiment.",
            "A one-sided test is justified only if the directional hypothesis is defined before final 10-seed evaluation.",
        ],
        "input_files": {
            "simclr_csv": str(simclr_csv),
            "nosimclr_csv": str(nosimclr_csv),
        },
        "metric": args.metric,
        "observed_n": int(len(matched)),
        "target_n": int(args.target_n),
        "observed_mean_values": {
            "simclr": float(matched["simclr"].mean()),
            "nosimclr": float(matched["nosimclr"].mean()),
        },
        "observed_paired_differences_summary": difference_summary,
        "observed_wilcoxon": {
            "one_sided_simclr_greater": {
                "statistic_w_plus": one_sided.statistic,
                "p_value": one_sided.pvalue,
                "n_nonzero": one_sided.n_nonzero,
                "warning": one_sided.warning,
            },
            "two_sided_reference": {
                "statistic_w_plus": two_sided.statistic,
                "p_value": two_sided.pvalue,
                "n_nonzero": two_sided.n_nonzero,
                "warning": two_sided.warning,
            },
        },
        "simulation_settings": {
            "num_simulations": int(args.num_simulations),
            "alpha": float(args.alpha),
            "random_seed": int(args.random_seed),
            "sampling_model": "Empirical bootstrap of observed paired differences.",
            "invalid_tests": int(simulation["invalid_tests"]),
        },
        "estimated_probability_p_lt_alpha": estimated_probability,
        "simulation_summaries": {
            "simulated_mean_differences": summarize_series(
                simulation["mean_differences"]
            ),
            "simulated_one_sided_p_values": summarize_series(
                simulation["pvalues"][np.isfinite(simulation["pvalues"])]
            )
            if np.isfinite(simulation["pvalues"]).any()
            else None,
        },
        "outputs": {
            "observed_csv": str(observed_csv),
            "plots": [str(path) for path in plot_paths],
        },
    }
    json_path = output_dir / "wilcoxon_simulation_summary.json"
    write_json_summary(json_path, json_summary)

    print(f"\nSaved observed paired values CSV: {observed_csv}")
    print(f"Saved JSON summary: {json_path}")
    print("Saved plots:")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
