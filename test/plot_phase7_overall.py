#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase-7 paper-ready plots (TEST locked; reporting only).

Config-driven (no CLI args). Reads paths from config.py:
  - PHASE7_MASTER_REPORT (required)
  - PHASE7_PLOTS_OUTDIR (required)
  - PHASE7_INCLUDE_FULLFT (optional, default False)
  - PHASE7_MODEL_ORDER / PHASE7_MODEL_LABELS (optional)

Uses master_report.json -> artifacts paths to find:
  - quality_summary_csv
  - timing_aggregate_csv
  - timing_per_repeat_csv

Outputs (PDF+PNG):
  - composite_score_overall
  - pareto_dice_vs_cpu_latency
  - pareto_dice_vs_gpu_latency
  - pareto_dice_vs_params
  - pareto_dice_vs_flops
  - overview_pareto_2x2
  - dice_panels_overall_pv01_pv03_pv08
  - plot_data_pareto_agg.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import utils.config as config


# -----------------------------
# Parsing model_id
# -----------------------------

def parse_model_parts(model_id: str) -> Tuple[str, str]:
    """
    Returns (base_model, ft_regime)

    Examples:
      ultralight_phase6_cand54_seed13 -> (ultralight_phase6, "")
      dlv3p_resnet50::minft::seed_13  -> (dlv3p_resnet50, "minft")
      dlv3p_resnet50::fullft::seed_13 -> (dlv3p_resnet50, "fullft")
    """
    if str(model_id).startswith("ultralight_phase6"):
        return ("ultralight_phase6", "")
    parts = str(model_id).split("::")
    base = parts[0] if len(parts) >= 1 else str(model_id)
    regime = ""
    if len(parts) >= 2 and parts[1] in ("minft", "fullft"):
        regime = parts[1]
    return (base, regime)


# -----------------------------
# Metrics
# -----------------------------

def composite_score(dice: float, params_total: float) -> float:
    denom = math.log10(1.0 + float(params_total))
    return float(dice) / denom if denom > 0 else float("nan")


# -----------------------------
# Plot helpers
# -----------------------------

def save_fig(fig: plt.Figure, outpath: Path, dpi: int = 300, use_tight_layout: bool = True) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: tight_layout can override subplots_adjust()
    if use_tight_layout:
        fig.tight_layout()

    fig.savefig(outpath.with_suffix(".pdf"))
    fig.savefig(outpath.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def label_for(base_model: str, ft_regime: str, labels: Dict[str, str], show_regime: bool) -> str:
    s = labels.get(base_model, base_model)
    if show_regime and ft_regime in ("minft", "fullft"):
        s = f"{s} ({ft_regime.upper()})"
    return s


def fmt_params(params_total: float) -> str:
    """Format parameter counts for display (e.g., '12.3M')."""
    try:
        v = float(params_total)
    except Exception:
        return "?"
    if not np.isfinite(v) or v <= 0:
        return "?"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:.0f}"


# -----------------------------
# Resolve artifact paths from master json
# -----------------------------

def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/")


def _is_abs_path(p: str) -> bool:
    pp = Path(p)
    return pp.is_absolute() or (len(p) >= 2 and p[1] == ":")


def resolve_phase7_paths(master_json_path: Path) -> Tuple[Path, Path, Path]:
    """
    Robust resolver for artifact paths stored in master_report.json.

    Handles:
      1) Absolute path -> use as-is
      2) Project-root relative path (e.g., 'bench_phase7/2026.../file.csv')
      3) Relative to master json directory
    """
    with master_json_path.open("r", encoding="utf-8") as f:
        master = json.load(f)

    artifacts = master.get("artifacts", {})
    q_raw = _norm_path(artifacts["quality_summary_csv"])
    ta_raw = _norm_path(artifacts["timing_aggregate_csv"])
    tr_raw = _norm_path(artifacts["timing_per_repeat_csv"])

    master_dir = master_json_path.parent.resolve()

    # Infer project root from location of bench_phase7
    if master_dir.name in ("bench_phase7", "bench_phase7_jetson_ts"):
        project_root = master_dir.parent
    elif master_dir.parent.name in ("bench_phase7", "bench_phase7_jetson_ts"):
        project_root = master_dir.parent.parent
    else:
        project_root = master_dir.parent

    def resolve_one(raw: str) -> Path:
        if _is_abs_path(raw):
            return Path(raw).resolve()
        if raw.startswith("bench_phase7/") or raw.startswith("bench_phase7\\") or raw.startswith("bench_phase7_jetson_ts/") or raw.startswith("bench_phase7_jetson_ts\\"):
            return (project_root / raw).resolve()
        return (master_dir / raw).resolve()

    return resolve_one(q_raw), resolve_one(ta_raw), resolve_one(tr_raw)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    master_path = Path(getattr(config, "PHASE7_MASTER_REPORT"))
    outdir = Path(getattr(config, "PHASE7_PLOTS_OUTDIR"))

    include_fullft = bool(getattr(config, "PHASE7_INCLUDE_FULLFT", False))
    show_regime = include_fullft  # show regime in labels if fullft is included
    model_order = list(getattr(
        config,
        "PHASE7_MODEL_ORDER",
        ["ultralight_phase6", "dlv3p_resnet50", "dlv3p_mobilenetv2", "unet_resnet34"],
    ))
    model_labels: Dict[str, str] = dict(getattr(config, "PHASE7_MODEL_LABELS", {}))

    if not master_path.exists():
        raise FileNotFoundError(f"PHASE7_MASTER_REPORT not found: {master_path}")

    q_path, ta_path, tr_path = resolve_phase7_paths(master_path)

    if not q_path.exists():
        raise FileNotFoundError(f"quality_summary_csv not found: {q_path}")
    if not ta_path.exists():
        raise FileNotFoundError(f"timing_aggregate_csv not found: {ta_path}")
    if not tr_path.exists():
        raise FileNotFoundError(f"timing_per_repeat_csv not found: {tr_path}")

    q = pd.read_csv(q_path)
    ta = pd.read_csv(ta_path)
    tr = pd.read_csv(tr_path)

    # Add base_model / ft_regime
    q[["base_model", "ft_regime"]] = q["model_id"].apply(lambda s: pd.Series(parse_model_parts(s)))
    ta[["base_model", "ft_regime"]] = ta["model_id"].apply(lambda s: pd.Series(parse_model_parts(s)))
    tr[["base_model", "ft_regime"]] = tr["model_id"].apply(lambda s: pd.Series(parse_model_parts(s)))

    # Paper-safe filter: ours + SOTA MINFT only (unless include_fullft)
    keep_regimes = {"", "minft", "fullft"} if include_fullft else {"", "minft"}
    q = q[q["ft_regime"].isin(keep_regimes)].copy()
    ta = ta[ta["ft_regime"].isin(keep_regimes)].copy()
    tr = tr[tr["ft_regime"].isin(keep_regimes)].copy()

    # -----------------------------
    # Composite score (overall) mean±std across seeds
    # -----------------------------
    q_overall = q[q["subset"] == "overall"].copy()

    # params_total: first per (base_model, ft_regime, seed) from timing_per_repeat (device=cpu)
    # Also try to pick up FLOPs from timing_per_repeat if available.
    flops_candidates = [
        "flops_total",
        "total_flops",
        "flops",
        "gflops",
        "macs_total",
        "macs",
        "total_macs",
    ]
    flops_col = next((c for c in flops_candidates if c in tr.columns), None)

    tr_cpu_first = (
        tr[tr["device"] == "cpu"]
        .sort_values(["base_model", "ft_regime", "seed", "repeat_idx"])
        .groupby(["base_model", "ft_regime", "seed"], as_index=False)
        .first()
    )

    keep_cols = ["base_model", "ft_regime", "seed", "params_total"]
    if flops_col is not None:
        keep_cols.append(flops_col)

    tr_params = tr_cpu_first[keep_cols].copy()

    comp = q_overall.merge(tr_params, on=["base_model", "ft_regime", "seed"], how="left")
    comp["score"] = comp.apply(lambda r: composite_score(r["dice_mean"], r["params_total"]), axis=1)

    comp_agg = (
        comp.groupby(["base_model", "ft_regime"], as_index=False)
        .agg(score_mean=("score", "mean"), score_std=("score", "std"))
    )

    # stable ordering: by model_order then ft regime
    def base_idx(bm: str) -> int:
        return model_order.index(bm) if bm in model_order else 999

    comp_agg["_bi"] = comp_agg["base_model"].apply(base_idx)
    comp_agg["_rr"] = comp_agg["ft_regime"].map({"": 0, "minft": 0, "fullft": 1}).fillna(2).astype(int)
    comp_agg = comp_agg.sort_values(["_bi", "_rr"]).drop(columns=["_bi", "_rr"])

    # -----------------------------
    # Pareto data (needed for consistent colors)
    # -----------------------------
    ta_cpu = (
        ta[ta["device"] == "cpu"][["base_model", "ft_regime", "seed", "ms_per_img_mean"]]
        .rename(columns={"ms_per_img_mean": "cpu_ms"})
    )
    ta_gpu = (
        ta[ta["device"] == "cuda"][["base_model", "ft_regime", "seed", "ms_per_img_mean"]]
        .rename(columns={"ms_per_img_mean": "gpu_ms"})
    )

    dice_seed = q_overall[["base_model", "ft_regime", "seed", "dice_mean"]].rename(columns={"dice_mean": "dice"})
    params_seed = tr_params.copy()

    merged = (
        dice_seed
        .merge(params_seed, on=["base_model", "ft_regime", "seed"], how="left")
        .merge(ta_cpu, on=["base_model", "ft_regime", "seed"], how="left")
        .merge(ta_gpu, on=["base_model", "ft_regime", "seed"], how="left")
    )

    pareto = (
        merged.groupby(["base_model", "ft_regime"], as_index=False)
        .agg(
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            params_total=("params_total", "mean"),
            cpu_ms_mean=("cpu_ms", "mean"),
            gpu_ms_mean=("gpu_ms", "mean"),
            **({"flops_mean": (flops_col, "mean")} if flops_col is not None else {}),
        )
    )

    pareto["_bi"] = pareto["base_model"].apply(base_idx)
    pareto["_rr"] = pareto["ft_regime"].map({"": 0, "minft": 0, "fullft": 1}).fillna(2).astype(int)
    pareto = pareto.sort_values(["_bi", "_rr"]).drop(columns=["_bi", "_rr"])

    # marker size scaling by params_total
    p = pareto["params_total"].to_numpy(dtype=float)
    p_min, p_max = np.nanmin(p), np.nanmax(p)
    sizes = (50 + 250 * (p - p_min) / (p_max - p_min)) if p_max > p_min else np.full_like(p, 120.0)

    # Consistent colors across all plots (based on Pareto entry order)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(color_cycle) < 8:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    entries = [(r.base_model, r.ft_regime) for r in pareto.itertuples(index=False)]
    color_map: Dict[Tuple[str, str], str] = {}
    for i, key in enumerate(entries):
        color_map[key] = color_cycle[i % len(color_cycle)]

    # Params lookup for legends (to explain marker size)
    params_by_key: Dict[Tuple[str, str], float] = {}
    for r in pareto.itertuples(index=False):
        params_by_key[(r.base_model, r.ft_regime)] = float(r.params_total) if np.isfinite(r.params_total) else float("nan")

    def legend_label(bm: str, fr: str) -> str:
        base = label_for(bm, fr, model_labels, show_regime)
        ptxt = fmt_params(params_by_key.get((bm, fr), float("nan")))
        return f"{base} — {ptxt} params"

    # -----------------------------
    # Composite score plot (single; unchanged filename)
    # -----------------------------
    fig, ax = plt.subplots()
    x = np.arange(len(comp_agg))
    y = comp_agg["score_mean"].to_numpy()
    yerr = comp_agg["score_std"].to_numpy()
    colors = [color_map.get((r.base_model, r.ft_regime), None) for r in comp_agg.itertuples(index=False)]

    ax.bar(x, y, yerr=yerr, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label_for(r.base_model, r.ft_regime, model_labels, show_regime) for r in comp_agg.itertuples(index=False)],
        rotation=20, ha="right",
    )
    ax.set_ylabel("Composite Score = Dice@0.5 / log10(1 + params_total)")
    ax.set_title("Composite Score (TEST overall, mean±std across seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, outdir / "composite_score_overall")

    # -----------------------------
    # Pareto plots (single + overview 2x2)
    # -----------------------------
    def draw_pareto(ax: plt.Axes, xcol: str, xlabel: str, xlog: bool = False, show_ylabel: bool = True) -> None:
        """Draw scatter on a given axis without creating a legend."""
        for i, r in enumerate(pareto.itertuples(index=False)):
            key = (r.base_model, r.ft_regime)
            ax.scatter(
                [getattr(r, xcol)],
                [r.dice_mean],
                s=[sizes[i]],
                color=color_map.get(key, None),
            )

        ax.set_xlabel(xlabel)
        if show_ylabel:
            ax.set_ylabel("Hard Dice@0.5 (TEST overall)")

        ax.set_ylim(0.8, 1.0)  # <--- start at 0.7 (as requested)

        if xlog:
            ax.set_xscale("log")

        ax.grid(True, which="both", axis="both", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def scatter_pareto(xcol: str, xlabel: str, outname: str, xlog: bool = False) -> None:
        """Single Pareto plot (filename unchanged) with legend matching the 2x2 style."""
        fig, ax = plt.subplots(figsize=(10, 7))

        for i, r in enumerate(pareto.itertuples(index=False)):
            key = (r.base_model, r.ft_regime)
            ax.scatter(
                [getattr(r, xcol)],
                [r.dice_mean],
                s=[sizes[i]],
                color=color_map.get(key, None),
                label=legend_label(r.base_model, r.ft_regime),
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Hard Dice@0.5 (TEST overall)")
        ax.set_ylim(0.8, 1.0)  # same as 2x2
        ax.set_title("Pareto view (marker size ~ params_total)")

        if xlog:
            ax.set_xscale("log")

        ax.grid(True, which="both", axis="both", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend: same style as 2x2, placed below
        ax.legend(
            ncol=2,
            frameon=True,
            fontsize=8,
            title="Marker size ∝ params_total",
            title_fontsize=8,
            handletextpad=0.6,
            columnspacing=1.0,
            borderpad=1,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            labelspacing=2
        )

        # Leave room at bottom for legend
        fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])

        save_fig(fig, outdir / outname)

    scatter_pareto("cpu_ms_mean", "CPU latency (ms/img) — mean across seeds", "pareto_dice_vs_cpu_latency")
    scatter_pareto("gpu_ms_mean", "GPU latency (ms/img) — mean across seeds", "pareto_dice_vs_gpu_latency")
    scatter_pareto("params_total", "Total parameters (log scale) — mean across seeds", "pareto_dice_vs_params", xlog=True)

    # FLOPs single plot (only if FLOPs column exists)
    if flops_col is not None:
        xlabel = "FLOPs (log scale) — mean across seeds"
        scatter_pareto("flops_mean", xlabel, "pareto_dice_vs_flops", xlog=True)
    else:
        print("[WARN] No FLOPs column found in timing_per_repeat_csv. Skipping pareto_dice_vs_flops and overview_pareto_2x2 FLOPs panel.")

    # NEW: 2x2 overview of Pareto plots (CPU, GPU, Params, FLOPs)
    def plot_overview_pareto_2x2() -> None:
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        axs = axs.flatten()

        draw_pareto(axs[0], "cpu_ms_mean", "CPU latency (ms/img)", xlog=False, show_ylabel=True)
        axs[0].set_title("Pareto: Dice vs CPU latency")

        draw_pareto(axs[1], "gpu_ms_mean", "GPU latency (ms/img)", xlog=False, show_ylabel=False)
        axs[1].set_title("Pareto: Dice vs GPU latency")

        draw_pareto(axs[2], "params_total", "Total parameters (log)", xlog=True, show_ylabel=True)
        axs[2].set_title("Pareto: Dice vs Params")

        if flops_col is not None:
            draw_pareto(axs[3], "flops_mean", "FLOPs (log)", xlog=True, show_ylabel=False)
            axs[3].set_title("Pareto: Dice vs FLOPs")
        else:
            axs[3].set_axis_off()
            axs[3].set_title("Pareto: Dice vs FLOPs (missing)")

        # Build legend handles with SAME marker sizes as in plots
        legend_handles = []
        legend_labels = []

        for i, (bm, fr) in enumerate(entries):
            key = (bm, fr)
            # scatter "proxy" artist for legend
            h = plt.scatter(
                [], [],  # no data; just a handle
                s=sizes[i],  # <-- same size as in plots
                color=color_map.get(key, "gray"),
            )
            legend_handles.append(h)
            legend_labels.append(legend_label(bm, fr))

        # --- KEY FIX: reserve bottom space for legend ---
        # Increase this if your legend is taller (0.22 -> 0.26)
        fig.subplots_adjust(bottom=0.20, top=0.93, left=0.07, right=0.98, wspace=0.12, hspace=0.32)

        # Place legend inside the reserved bottom margin (below subplots)
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=2,
            frameon=True,
            fontsize=8,
            title="Marker size ∝ params_total",
            title_fontsize=8,
            handletextpad=0.6,
            columnspacing=1.0,
            borderpad=1,
            labelspacing=2,  # ako želiš razmak između redaka
        )

        fig.suptitle("Phase-7 overview: Pareto views", y=0.98)

        save_fig(fig, outdir / "overview_pareto_2x2", use_tight_layout=False)  # disable tight_layout to preserve custom spacing

    plot_overview_pareto_2x2()

    # Export helper CSV
    pareto.to_csv(outdir / "plot_data_pareto_agg.csv", index=False)

    # -----------------------------
    # Dice panels: overall + PV01 + PV03 + PV08 (2x2), consistent colors + grid
    # -----------------------------
    def plot_dice_panels(qdf: pd.DataFrame) -> None:
        subsets = ["overall", "PV01", "PV03", "PV08"]
        titles = {
            "overall": "Overall",
            "PV01": "PV01 (0.1 m)",
            "PV03": "PV03 (0.3 m)",
            "PV08": "PV08 (0.8 m)",
        }

        # Y-axis control (starts from ymin, not 0)
        auto_ylim = True
        pad_low = 0.04
        pad_high = 0.02
        fixed_ymin, fixed_ymax = 0.80, 1.00  # used only if auto_ylim=False

        major_step = 0.05
        minor_step = 0.01
        bar_width = 0.78
        capsize = 4

        def round_down_to_step(x: float, step: float) -> float:
            return float(np.floor(x / step) * step)

        def round_up_to_step(x: float, step: float) -> float:
            return float(np.ceil(x / step) * step)

        if auto_ylim:
            qtmp = (
                qdf.groupby(["subset", "base_model", "ft_regime"], as_index=False)
                   .agg(dice_mean=("dice_mean", "mean"), dice_std=("dice_mean", "std"))
            )
            qtmp["dice_std"] = qtmp["dice_std"].fillna(0.0)
            global_min = float((qtmp["dice_mean"] - qtmp["dice_std"]).min())
            global_max = float((qtmp["dice_mean"] + qtmp["dice_std"]).max())

            ymin = round_down_to_step(max(0.0, global_min - pad_low), 0.05)
            ymax = round_up_to_step(min(1.0, global_max + pad_high), 0.05)
            if ymax - ymin < 0.10:
                ymin = max(0.0, ymin - 0.05)
                ymax = min(1.0, ymax + 0.05)
        else:
            ymin, ymax = fixed_ymin, fixed_ymax

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
        axes = axes.flatten()

        for ax, subset in zip(axes, subsets):
            qs = qdf[qdf["subset"] == subset].copy()

            qs_agg = (
                qs.groupby(["base_model", "ft_regime"], as_index=False)
                  .agg(dice_mean=("dice_mean", "mean"), dice_std=("dice_mean", "std"))
            )
            qs_agg["dice_std"] = qs_agg["dice_std"].fillna(0.0)

            ordered_rows = []
            for bm, fr in entries:
                hit = qs_agg[(qs_agg["base_model"] == bm) & (qs_agg["ft_regime"] == fr)]
                if len(hit) == 1:
                    ordered_rows.append(hit.iloc[0])

            if not ordered_rows:
                ax.set_title(titles.get(subset, subset))
                ax.set_axis_off()
                continue

            qs2 = pd.DataFrame(ordered_rows)
            x = np.arange(len(qs2))
            y = qs2["dice_mean"].to_numpy()
            yerr = qs2["dice_std"].to_numpy()
            colors = [color_map[(r.base_model, r.ft_regime)] for r in qs2.itertuples(index=False)]

            ax.bar(
                x,
                y,
                yerr=yerr,
                capsize=capsize,
                width=bar_width,
                color=colors,
                edgecolor="none",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [label_for(r.base_model, r.ft_regime, model_labels, show_regime) for r in qs2.itertuples(index=False)],
                rotation=20,
                ha="right",
            )

            ax.set_ylim(ymin, ymax)
            ax.set_yticks(np.arange(ymin, ymax + 1e-9, major_step))
            ax.set_yticks(np.arange(ymin, ymax + 1e-9, minor_step), minor=True)
            ax.grid(True, which="major", axis="y", alpha=0.30)
            ax.grid(True, which="minor", axis="y", alpha=0.12)
            ax.tick_params(axis="y", labelleft=True)

            ax.set_ylabel("Hard Dice@0.5")
            ax.set_title(titles.get(subset, subset))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("Hard Dice@0.5 on TEST (mean±std across seeds)")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
        save_fig(fig, outdir / "dice_panels_overall_pv01_pv03_pv08")

    plot_dice_panels(q)

    print(f"[OK] Saved plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
