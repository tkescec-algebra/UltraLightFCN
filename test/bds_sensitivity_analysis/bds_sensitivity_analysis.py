from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Hardcoded configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(".")

DESKTOP_INPUT = BASE_DIR / "bds_sensitivity_input_desktop.csv"
JETSON_INPUT = BASE_DIR / "bds_sensitivity_input_jetson.csv"

OUTDIR = BASE_DIR / "bds_sensitivity_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

COMBINED_FIG_PNG = OUTDIR / "bds_sensitivity_scores.png"
COMBINED_FIG_PDF = OUTDIR / "bds_sensitivity_scores.pdf"

DESKTOP_OUT_SUBDIR = OUTDIR / "desktop"
JETSON_OUT_SUBDIR = OUTDIR / "jetson"
DESKTOP_OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
JETSON_OUT_SUBDIR.mkdir(parents=True, exist_ok=True)

COLORMAP = "viridis"
NUMBER_FORMAT = ".3f"

MODEL_ORDER = [
    "ULFCN (ours)",
    'DLV3+ R50',
    "DLV3+ MNetV2",
    "U-Net R34",
]

WEIGHT_SCHEMES: List[Dict] = [
    {"scheme_id": "W1", "label": "Primary",        "w_D": 0.55, "w_L": 0.25, "w_M": 0.15, "w_P": 0.05},
    {"scheme_id": "W2", "label": "Latency-up",     "w_D": 0.50, "w_L": 0.30, "w_M": 0.15, "w_P": 0.05},
    {"scheme_id": "W3", "label": "Accuracy-up",    "w_D": 0.60, "w_L": 0.20, "w_M": 0.15, "w_P": 0.05},
    {"scheme_id": "W4", "label": "Memory-up",      "w_D": 0.55, "w_L": 0.20, "w_M": 0.20, "w_P": 0.05},
    {"scheme_id": "W5", "label": "Balanced-alt",   "w_D": 0.50, "w_L": 0.25, "w_M": 0.20, "w_P": 0.05},
    {"scheme_id": "W6", "label": "Accuracy-heavy", "w_D": 0.65, "w_L": 0.15, "w_M": 0.15, "w_P": 0.05},
    {"scheme_id": "W7", "label": "Latency-heavy",  "w_D": 0.45, "w_L": 0.35, "w_M": 0.15, "w_P": 0.05},
    {"scheme_id": "W8", "label": "Memory-heavy",   "w_D": 0.50, "w_L": 0.20, "w_M": 0.25, "w_P": 0.05},
]

WEIGHT_ORDER = [ws["scheme_id"] for ws in WEIGHT_SCHEMES]
REQUIRED_COLUMNS = ["model", "D", "L_cpu_ms", "R_cpu_bytes", "L_gpu_ms", "V_gpu_bytes", "P_params"]


# ---------------------------------------------------------------------
# Validation and score computation
# ---------------------------------------------------------------------
def validate_input(df: pd.DataFrame, source_name: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name}: missing required columns: {missing}")

    for c in REQUIRED_COLUMNS[1:]:
        if (df[c] <= 0).any():
            raise ValueError(f"{source_name}: column '{c}' must contain strictly positive values.")

    if "display_name" not in df.columns:
        raise ValueError(f"{source_name}: expected a 'display_name' column for stable model labeling.")


def compute_bds_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dmax = df["D"].max()
    lmin_cpu = df["L_cpu_ms"].min()
    rmin_cpu = df["R_cpu_bytes"].min()
    lmin_gpu = df["L_gpu_ms"].min()
    vmin_gpu = df["V_gpu_bytes"].min()
    pmin = df["P_params"].min()

    weight_df = pd.DataFrame(WEIGHT_SCHEMES)
    score_rows = []

    for _, row in df.iterrows():
        model_label = row["display_name"]
        for ws in WEIGHT_SCHEMES:
            cpu_bds = (
                (row["D"] / dmax) ** ws["w_D"] *
                (lmin_cpu / row["L_cpu_ms"]) ** ws["w_L"] *
                (rmin_cpu / row["R_cpu_bytes"]) ** ws["w_M"] *
                (pmin / row["P_params"]) ** ws["w_P"]
            )

            gpu_bds = (
                (row["D"] / dmax) ** ws["w_D"] *
                (lmin_gpu / row["L_gpu_ms"]) ** ws["w_L"] *
                (vmin_gpu / row["V_gpu_bytes"]) ** ws["w_M"] *
                (pmin / row["P_params"]) ** ws["w_P"]
            )

            score_rows.append(
                {
                    "model": row["model"],
                    "display_name": model_label,
                    "scheme_id": ws["scheme_id"],
                    "scheme_label": ws["label"],
                    "CPU_BDS": cpu_bds,
                    "GPU_BDS": gpu_bds,
                }
            )

    long_df = pd.DataFrame(score_rows)
    long_df["CPU_rank"] = long_df.groupby("scheme_id")["CPU_BDS"].rank(ascending=False, method="min").astype(int)
    long_df["GPU_rank"] = long_df.groupby("scheme_id")["GPU_BDS"].rank(ascending=False, method="min").astype(int)

    refs_df = pd.DataFrame(
        {
            "D_max": [dmax],
            "L_min_CPU_ms": [lmin_cpu],
            "R_min_CPU_bytes": [rmin_cpu],
            "L_min_GPU_ms": [lmin_gpu],
            "V_min_GPU_bytes": [vmin_gpu],
            "P_min_params": [pmin],
        }
    )

    return weight_df, long_df, refs_df


def wide_score_table(long_df: pd.DataFrame, score_key: str) -> pd.DataFrame:
    table = long_df.pivot(index="display_name", columns="scheme_id", values=score_key)

    existing_models = [m for m in MODEL_ORDER if m in table.index]
    remaining_models = [m for m in table.index if m not in existing_models]
    table = table.loc[existing_models + remaining_models]

    existing_weights = [w for w in WEIGHT_ORDER if w in table.columns]
    remaining_weights = [w for w in table.columns if w not in existing_weights]
    table = table[existing_weights + remaining_weights]

    # transpose so y-axis = weights, x-axis = models
    return table.T


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def luminance_from_rgba(rgba: Tuple[float, float, float, float]) -> float:
    r, g, b, _ = rgba
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color_for_value(value: float, norm: mpl.colors.Normalize, cmap_obj) -> str:
    rgba = cmap_obj(norm(value))
    lum = luminance_from_rgba(rgba)
    return "white" if lum < 0.45 else "black"


def draw_heatmap(
    ax,
    matrix: pd.DataFrame,
    title: str,
    panel_label: str,
    vmin: float,
    vmax: float,
) -> None:
    cmap_obj = plt.get_cmap(COLORMAP)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        matrix.values.astype(float),
        aspect="auto",
        cmap=cmap_obj,
        norm=norm,
    )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticklabels(matrix.index, fontsize=10)

    # ax.set_xlabel("Models")
    ax.set_ylabel("Weighting schemes")
    ax.set_title(f"{panel_label} {title}", loc="center", fontweight="bold", pad=10, fontsize=10)

    # cell borders
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    best_value = np.nanmax(matrix.values.astype(float))

    for i in range(matrix.shape[0]):
        row_values = matrix.values[i, :].astype(float)
        row_best = np.nanmax(row_values)

        for j in range(matrix.shape[1]):
            value = float(matrix.values[i, j])
            is_best_in_row = np.isclose(value, row_best)

            ax.text(
                j,
                i,
                format(value, NUMBER_FORMAT),
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color_for_value(value, norm, cmap_obj),
                fontweight="bold" if is_best_in_row else "normal",
            )

    # one colorbar per panel, placed to the right of each plot
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("BDS score")


def save_combined_figure(
    desktop_cpu: pd.DataFrame,
    desktop_gpu: pd.DataFrame,
    jetson_cpu: pd.DataFrame,
    jetson_gpu: pd.DataFrame,
) -> None:
    all_vals = np.concatenate(
        [
            desktop_cpu.values.astype(float).ravel(),
            desktop_gpu.values.astype(float).ravel(),
            jetson_cpu.values.astype(float).ravel(),
            jetson_gpu.values.astype(float).ravel(),
        ]
    )
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        raise ValueError("No finite score values found for plotting.")

    vmin = float(finite_vals.min())
    vmax = float(finite_vals.max())

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.07, top=0.95, wspace=0.35, hspace=0.42)

    draw_heatmap(axes[0, 0], desktop_cpu, "Desktop CPU-BDS", "(a)", vmin, vmax)
    draw_heatmap(axes[0, 1], desktop_gpu, "Desktop GPU-BDS", "(b)", vmin, vmax)
    draw_heatmap(axes[1, 0], jetson_cpu, "Jetson CPU-BDS", "(c)", vmin, vmax)
    draw_heatmap(axes[1, 1], jetson_gpu, "Jetson GPU-BDS", "(d)", vmin, vmax)

    fig.savefig(COMBINED_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(COMBINED_FIG_PDF, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------
def export_environment_outputs(
    env_name: str,
    weight_df: pd.DataFrame,
    long_df: pd.DataFrame,
    refs_df: pd.DataFrame,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)

    weight_df.to_csv(outdir / "weight_schemes.csv", index=False)
    long_df.to_csv(outdir / "bds_scores_long.csv", index=False)
    refs_df.to_csv(outdir / "benchmark_reference_values.csv", index=False)

    winner_rows = []
    for scheme_id, grp in long_df.groupby("scheme_id"):
        cpu_top = grp.sort_values(["CPU_BDS", "display_name"], ascending=[False, True]).iloc[0]
        gpu_top = grp.sort_values(["GPU_BDS", "display_name"], ascending=[False, True]).iloc[0]
        winner_rows.append(
            {
                "scheme_id": scheme_id,
                "scheme_label": grp["scheme_label"].iloc[0],
                "cpu_winner": cpu_top["display_name"],
                "cpu_rank1_score": cpu_top["CPU_BDS"],
                "gpu_winner": gpu_top["display_name"],
                "gpu_rank1_score": gpu_top["GPU_BDS"],
            }
        )
    pd.DataFrame(winner_rows).sort_values("scheme_id").to_csv(outdir / "winner_summary.csv", index=False)

    cpu_score_table = wide_score_table(long_df, "CPU_BDS")
    gpu_score_table = wide_score_table(long_df, "GPU_BDS")

    cpu_score_table.to_csv(outdir / "cpu_bds_scores_wide_transposed.csv")
    gpu_score_table.to_csv(outdir / "gpu_bds_scores_wide_transposed.csv")

    print(f"[{env_name}] outputs saved to: {outdir.resolve()}")
    return cpu_score_table, gpu_score_table


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    desktop_df = pd.read_csv(DESKTOP_INPUT)
    jetson_df = pd.read_csv(JETSON_INPUT)

    validate_input(desktop_df, "desktop")
    validate_input(jetson_df, "jetson")

    desktop_weight_df, desktop_long_df, desktop_refs_df = compute_bds_tables(desktop_df)
    jetson_weight_df, jetson_long_df, jetson_refs_df = compute_bds_tables(jetson_df)

    desktop_cpu, desktop_gpu = export_environment_outputs(
        env_name="desktop",
        weight_df=desktop_weight_df,
        long_df=desktop_long_df,
        refs_df=desktop_refs_df,
        outdir=DESKTOP_OUT_SUBDIR,
    )
    jetson_cpu, jetson_gpu = export_environment_outputs(
        env_name="jetson",
        weight_df=jetson_weight_df,
        long_df=jetson_long_df,
        refs_df=jetson_refs_df,
        outdir=JETSON_OUT_SUBDIR,
    )

    save_combined_figure(
        desktop_cpu=desktop_cpu,
        desktop_gpu=desktop_gpu,
        jetson_cpu=jetson_cpu,
        jetson_gpu=jetson_gpu,
    )

    print(f"Combined PNG saved to: {COMBINED_FIG_PNG.resolve()}")
    print(f"Combined PDF saved to: {COMBINED_FIG_PDF.resolve()}")


if __name__ == "__main__":
    main()
