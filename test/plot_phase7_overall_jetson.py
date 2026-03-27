#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase-7 paper-ready plots (TEST locked; reporting only).

Refactored version of plot_phase7_overall_jetson.py.

What it generates (PDF + PNG):
  - composite_score_overall
  - pareto_dice_vs_cpu_latency
  - pareto_dice_vs_gpu_latency
  - pareto_dice_vs_params
  - pareto_dice_vs_flops                (only if FLOPs column exists)
  - overview_pareto_2x2                 (only if FLOPs column exists; otherwise last panel is blank)
  - dice_panels_overall_pv01_pv03_pv08
  - figure5_jetson_main                  (new main paper figure)
  - plot_data_pareto_agg.csv

Config-driven (no CLI args). Reads from utils.config:
  - PHASE7_MASTER_REPORT                (required)
  - PHASE7_PLOTS_OUTDIR                 (required)
  - PHASE7_INCLUDE_FULLFT               (optional, default False)
  - PHASE7_MODEL_ORDER                  (optional)
  - PHASE7_MODEL_LABELS                 (optional)

Figure 5 layout:
  Row 1: Composite | Overall hard Dice
  Row 2: PV01 | PV03 | PV08 hard Dice
  Row 3: Pareto GPU | Pareto CPU | Pareto Params
  Bottom: One shared Pareto legend

Design choices:
  - one shared legend for the three Pareto panels in Figure 5
  - log scale only for Params on x-axis
  - linear x-axis for GPU and CPU latency
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import utils.config as config


# -----------------------------
# Parsing model_id / ckpt_path for Jetson export
# -----------------------------

def _normalize_text(s: object) -> str:
    return str(s).strip().lower().replace("\\", "/")


def detect_base_model(model_id: object, ckpt_path: object = "") -> str:
    text = f"{_normalize_text(model_id)} || {_normalize_text(ckpt_path)}"

    if "dlv3p_resnet50" in text:
        return "dlv3p_resnet50"
    if "dlv3p_mobilenetv2" in text:
        return "dlv3p_mobilenetv2"
    if "unet_resnet34" in text:
        return "unet_resnet34"

    if (
        "ultralightfcn" in text
        or "seg_phase6" in text
        or "phase6" in text
        or "trial_54" in text
    ):
        return "ultralight_phase6"

    return _normalize_text(model_id)


def detect_ft_regime(model_id: object, ckpt_path: object = "") -> str:
    text = f"{_normalize_text(model_id)} || {_normalize_text(ckpt_path)}"
    if "fullft" in text:
        return "fullft"
    if "minft" in text:
        return "minft"
    return ""


def parse_model_parts(model_id: object, ckpt_path: object = "") -> Tuple[str, str]:
    return detect_base_model(model_id, ckpt_path), detect_ft_regime(model_id, ckpt_path)


# -----------------------------
# Metrics
# -----------------------------

def composite_score(dice: float, params_total: float) -> float:
    denom = math.log10(float(params_total))
    return float(dice) / denom if denom > 0 else float("nan")


# -----------------------------
# Generic helpers
# -----------------------------

def save_fig(fig: plt.Figure, outpath: Path, dpi: int = 300, use_tight_layout: bool = True) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(outpath.with_suffix(".pdf"))
    fig.savefig(outpath.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def label_for(base_model: str, ft_regime: str, labels: Dict[str, str], show_regime: bool) -> str:
    text = labels.get(base_model, base_model)
    if show_regime and ft_regime in ("minft", "fullft"):
        text = f"{text} ({ft_regime.upper()})"
    return text


def fmt_params(params_total: float) -> str:
    try:
        value = float(params_total)
    except Exception:
        return "?"
    if not np.isfinite(value) or value <= 0:
        return "?"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def round_down_to_step(x: float, step: float) -> float:
    return float(np.floor(x / step) * step)


def round_up_to_step(x: float, step: float) -> float:
    return float(np.ceil(x / step) * step)


# -----------------------------
# Resolve artifact paths from master json
# -----------------------------

def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/")


def _is_abs_path(p: str) -> bool:
    pp = Path(p)
    return pp.is_absolute() or (len(p) >= 2 and p[1] == ":")


def resolve_phase7_paths(master_json_path: Path) -> Tuple[Path, Path, Path]:
    with master_json_path.open("r", encoding="utf-8") as f:
        master = json.load(f)

    artifacts = master.get("artifacts", {})
    q_raw = _norm_path(artifacts["quality_summary_csv"])
    ta_raw = _norm_path(artifacts["timing_aggregate_csv"])
    tr_raw = _norm_path(artifacts["timing_per_repeat_csv"])

    master_dir = master_json_path.parent.resolve()

    if master_dir.name in ("bench_phase7", "bench_phase7_jetson_ts"):
        project_root = master_dir.parent
        bench_root = master_dir
    elif master_dir.parent.name in ("bench_phase7", "bench_phase7_jetson_ts"):
        project_root = master_dir.parent.parent
        bench_root = master_dir.parent
    else:
        project_root = master_dir.parent
        bench_root = master_dir.parent

    def _candidate_paths(raw: str):
        raw_path = Path(raw)
        name_only = raw_path.name
        candidates = []

        if _is_abs_path(raw):
            candidates.append(Path(raw).resolve())
            return candidates

        candidates.append((master_dir / name_only).resolve())
        candidates.append((project_root / raw).resolve())
        candidates.append((master_dir / raw).resolve())

        raw_posix = raw.replace("\\", "/")
        for prefix in ("bench_phase7/", "bench_phase7_jetson_ts/"):
            if raw_posix.startswith(prefix):
                candidates.append((project_root / raw_posix[len(prefix):]).resolve())
                candidates.append((bench_root / raw_posix[len(prefix):]).resolve())

        candidates.append((Path("test") / name_only).resolve())

        seen = set()
        unique = []
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                unique.append(c)
        return unique

    def resolve_one(raw: str) -> Path:
        for cand in _candidate_paths(raw):
            if cand.exists():
                return cand
        return _candidate_paths(raw)[0]

    return resolve_one(q_raw), resolve_one(ta_raw), resolve_one(tr_raw)


def load_desktop_params_from_phase7(include_fullft: bool) -> pd.DataFrame:
    desktop_master = Path(getattr(config, "PHASE7_MASTER_REPORT"))
    if not desktop_master.exists():
        print(f"[WARN] Desktop PHASE7_MASTER_REPORT not found: {desktop_master}")
        return pd.DataFrame(columns=["base_model", "ft_regime", "seed", "params_total"])

    _, _, tr_path = resolve_phase7_paths(desktop_master)
    if not tr_path.exists():
        print(f"[WARN] Desktop timing_per_repeat_csv not found: {tr_path}")
        return pd.DataFrame(columns=["base_model", "ft_regime", "seed", "params_total"])

    tr = pd.read_csv(tr_path)
    tr["ckpt_path"] = tr.get("ckpt_path", "")
    parts = tr.apply(
        lambda r: pd.Series(parse_model_parts(r.get("model_id", ""), r.get("ckpt_path", ""))),
        axis=1,
    )
    parts.columns = ["base_model", "ft_regime"]
    tr[["base_model", "ft_regime"]] = parts

    keep_regimes = {"", "minft", "fullft"} if include_fullft else {"", "minft"}
    tr = tr[tr["ft_regime"].isin(keep_regimes)].copy()

    tr_cpu_first = (
        tr[tr["device"] == "cpu"]
        .sort_values(["base_model", "ft_regime", "seed", "repeat_idx"])
        .groupby(["base_model", "ft_regime", "seed"], as_index=False)
        .first()
    )

    if "params_total" not in tr_cpu_first.columns:
        print("[WARN] Desktop export has no params_total column")
        return pd.DataFrame(columns=["base_model", "ft_regime", "seed", "params_total"])

    out = tr_cpu_first[["base_model", "ft_regime", "seed", "params_total"]].copy()
    out["params_total"] = pd.to_numeric(out["params_total"], errors="coerce")
    out = out[np.isfinite(out["params_total"]) & (out["params_total"] > 0)].copy()
    return out


# -----------------------------
# Plot context + settings
# -----------------------------

@dataclass
class PlotSettings:
    include_fullft: bool
    show_regime: bool
    model_order: List[str]
    model_labels: Dict[str, str]
    outdir: Path


@dataclass
class PlotContext:
    settings: PlotSettings
    entries: List[Tuple[str, str]]
    color_map: Dict[Tuple[str, str], str]
    params_by_key: Dict[Tuple[str, str], float]
    sizes_by_key: Dict[Tuple[str, str], float]
    comp_agg: pd.DataFrame
    pareto: pd.DataFrame
    q_filtered: pd.DataFrame
    flops_col: Optional[str]
    dice_ylim: Tuple[float, float]

    def ordered_labels(self, df: pd.DataFrame) -> List[str]:
        return [
            label_for(r.base_model, r.ft_regime, self.settings.model_labels, self.settings.show_regime)
            for r in df.itertuples(index=False)
        ]

    def legend_label(self, bm: str, fr: str) -> str:
        base = label_for(bm, fr, self.settings.model_labels, self.settings.show_regime)
        ptxt = fmt_params(self.params_by_key.get((bm, fr), float("nan")))
        return f"{base} — {ptxt} params"


# -----------------------------
# Data preparation
# -----------------------------

def load_phase7_tables(settings: PlotSettings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master_path = Path(getattr(config, "PHASE7_MASTER_REPORT_JETSON"))
    if not master_path.exists():
        raise FileNotFoundError(f"PHASE7_MASTER_REPORT_JETSON not found: {master_path}")

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

    q["ckpt_path"] = q.get("ckpt_path", "")
    ta["ckpt_path"] = ta.get("ckpt_path", "")
    tr["ckpt_path"] = tr.get("ckpt_path", "")

    for df in (q, ta, tr):
        parts = df.apply(
            lambda r: pd.Series(parse_model_parts(r.get("model_id", ""), r.get("ckpt_path", ""))),
            axis=1,
        )
        parts.columns = ["base_model", "ft_regime"]
        df[["base_model", "ft_regime"]] = parts

    keep_regimes = {"", "minft", "fullft"} if settings.include_fullft else {"", "minft"}
    q = q[q["ft_regime"].isin(keep_regimes)].copy()
    ta = ta[ta["ft_regime"].isin(keep_regimes)].copy()
    tr = tr[tr["ft_regime"].isin(keep_regimes)].copy()

    return q, ta, tr


def detect_flops_column(tr: pd.DataFrame) -> Optional[str]:
    flops_candidates = [
        "flops_total",
        "total_flops",
        "flops",
        "gflops",
        "macs_total",
        "macs",
        "total_macs",
    ]
    return next((c for c in flops_candidates if c in tr.columns), None)


def aggregate_composite(
    q: pd.DataFrame,
    tr: pd.DataFrame,
    model_order: Sequence[str],
    include_fullft: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str], pd.DataFrame]:
    q_overall = q[q["subset"] == "overall"].copy()
    flops_col = detect_flops_column(tr)

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
    tr_params["params_total"] = pd.to_numeric(tr_params["params_total"], errors="coerce")

    # Jetson timing exports may miss or corrupt params_total. Fill from desktop Phase-7 report when available.
    desktop_params = load_desktop_params_from_phase7(include_fullft)
    if not desktop_params.empty:
        tr_params = tr_params.merge(
            desktop_params.rename(columns={"params_total": "params_total_desktop"}),
            on=["base_model", "ft_regime", "seed"],
            how="outer",
        )
        tr_params["params_total"] = tr_params["params_total"].where(
            np.isfinite(tr_params["params_total"]) & (tr_params["params_total"] > 0),
            tr_params["params_total_desktop"],
        )
        if flops_col is not None and flops_col not in tr_params.columns:
            tr_params[flops_col] = np.nan
        tr_params = tr_params.drop(columns=[c for c in ["params_total_desktop"] if c in tr_params.columns])

    tr_params = tr_params[np.isfinite(tr_params["params_total"]) & (tr_params["params_total"] > 0)].copy()

    comp = q_overall.merge(tr_params, on=["base_model", "ft_regime", "seed"], how="left")
    comp["params_total"] = pd.to_numeric(comp["params_total"], errors="coerce")
    comp = comp[np.isfinite(comp["params_total"]) & (comp["params_total"] > 0)].copy()
    comp["score"] = comp.apply(lambda r: composite_score(r["dice_mean"], r["params_total"]), axis=1)

    comp_agg = (
        comp.groupby(["base_model", "ft_regime"], as_index=False)
        .agg(score_mean=("score", "mean"), score_std=("score", "std"))
    )

    def base_idx(bm: str) -> int:
        return model_order.index(bm) if bm in model_order else 999

    comp_agg["_bi"] = comp_agg["base_model"].apply(base_idx)
    comp_agg["_rr"] = comp_agg["ft_regime"].map({"": 0, "minft": 0, "fullft": 1}).fillna(2).astype(int)
    comp_agg = comp_agg.sort_values(["_bi", "_rr"]).drop(columns=["_bi", "_rr"])

    return q_overall, comp_agg, flops_col, tr_params


def aggregate_pareto(
    q_overall: pd.DataFrame,
    ta: pd.DataFrame,
    tr_params: pd.DataFrame,
    flops_col: Optional[str],
    model_order: Sequence[str],
) -> pd.DataFrame:
    ta_cpu = (
        ta[ta["device"] == "cpu"][["base_model", "ft_regime", "seed", "ms_per_img_mean"]]
        .rename(columns={"ms_per_img_mean": "cpu_ms"})
    )
    ta_gpu = (
        ta[ta["device"] == "cuda"][["base_model", "ft_regime", "seed", "ms_per_img_mean"]]
        .rename(columns={"ms_per_img_mean": "gpu_ms"})
    )

    dice_seed = q_overall[["base_model", "ft_regime", "seed", "dice_mean"]].rename(columns={"dice_mean": "dice"})

    merged = (
        dice_seed
        .merge(tr_params, on=["base_model", "ft_regime", "seed"], how="left")
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

    def base_idx(bm: str) -> int:
        return model_order.index(bm) if bm in model_order else 999

    pareto["_bi"] = pareto["base_model"].apply(base_idx)
    pareto["_rr"] = pareto["ft_regime"].map({"": 0, "minft": 0, "fullft": 1}).fillna(2).astype(int)
    pareto = pareto.sort_values(["_bi", "_rr"]).drop(columns=["_bi", "_rr"])
    return pareto


def build_style_context(pareto: pd.DataFrame) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], str], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    params = pareto["params_total"].to_numpy(dtype=float)
    p_min, p_max = np.nanmin(params), np.nanmax(params)
    sizes = (50 + 250 * (params - p_min) / (p_max - p_min)) if p_max > p_min else np.full_like(params, 120.0)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(color_cycle) < 8:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    entries = [(r.base_model, r.ft_regime) for r in pareto.itertuples(index=False)]
    color_map: Dict[Tuple[str, str], str] = {}
    params_by_key: Dict[Tuple[str, str], float] = {}
    sizes_by_key: Dict[Tuple[str, str], float] = {}

    for i, row in enumerate(pareto.itertuples(index=False)):
        key = (row.base_model, row.ft_regime)
        color_map[key] = color_cycle[i % len(color_cycle)]
        params_by_key[key] = float(row.params_total) if np.isfinite(row.params_total) else float("nan")
        sizes_by_key[key] = float(sizes[i])

    return entries, color_map, params_by_key, sizes_by_key


def compute_dice_ylim(q: pd.DataFrame) -> Tuple[float, float]:
    qtmp = (
        q.groupby(["subset", "base_model", "ft_regime"], as_index=False)
        .agg(dice_mean=("dice_mean", "mean"), dice_std=("dice_mean", "std"))
    )
    qtmp["dice_std"] = qtmp["dice_std"].fillna(0.0)
    global_min = float((qtmp["dice_mean"] - qtmp["dice_std"]).min())
    global_max = float((qtmp["dice_mean"] + qtmp["dice_std"]).max())

    ymin = round_down_to_step(max(0.0, global_min - 0.04), 0.05)
    ymax = round_up_to_step(min(1.0, global_max + 0.02), 0.05)
    if ymax - ymin < 0.10:
        ymin = max(0.0, ymin - 0.05)
        ymax = min(1.0, ymax + 0.05)
    return ymin, ymax


def build_context(settings: PlotSettings) -> PlotContext:
    q, ta, tr = load_phase7_tables(settings)
    q_overall, comp_agg, flops_col, tr_params = aggregate_composite(q, tr, settings.model_order, settings.include_fullft)
    pareto = aggregate_pareto(q_overall, ta, tr_params, flops_col, settings.model_order)
    entries, color_map, params_by_key, sizes_by_key = build_style_context(pareto)
    dice_ylim = compute_dice_ylim(q)

    return PlotContext(
        settings=settings,
        entries=entries,
        color_map=color_map,
        params_by_key=params_by_key,
        sizes_by_key=sizes_by_key,
        comp_agg=comp_agg,
        pareto=pareto,
        q_filtered=q,
        flops_col=flops_col,
        dice_ylim=dice_ylim,
    )


# -----------------------------
# Aggregation helpers for panel drawing
# -----------------------------

def aggregate_dice_subset(q: pd.DataFrame, subset: str, entries: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    qs = q[q["subset"] == subset].copy()
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

    return pd.DataFrame(ordered_rows) if ordered_rows else pd.DataFrame(columns=qs_agg.columns)


# -----------------------------
# Panel drawing functions
# -----------------------------

def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_composite_panel(ax: plt.Axes, ctx: PlotContext, title: str = "Composite Score") -> None:
    df = ctx.comp_agg
    x = np.arange(len(df))
    y = df["score_mean"].to_numpy()
    yerr = df["score_std"].fillna(0.0).to_numpy()
    colors = [ctx.color_map.get((r.base_model, r.ft_regime), None) for r in df.itertuples(index=False)]

    ax.bar(x, y, yerr=yerr, capsize=4, color=colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(ctx.ordered_labels(df), rotation=20, ha="right")
    ax.set_ylabel("Composite Score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    style_axis(ax)


def draw_dice_subset_panel(
    ax: plt.Axes,
    ctx: PlotContext,
    subset: str,
    title: str,
    show_ylabel: bool = True,
) -> None:
    qs2 = aggregate_dice_subset(ctx.q_filtered, subset, ctx.entries)
    if qs2.empty:
        ax.set_title(title)
        ax.set_axis_off()
        return

    x = np.arange(len(qs2))
    y = qs2["dice_mean"].to_numpy()
    yerr = qs2["dice_std"].fillna(0.0).to_numpy()
    colors = [ctx.color_map[(r.base_model, r.ft_regime)] for r in qs2.itertuples(index=False)]

    ax.bar(x, y, yerr=yerr, capsize=4, width=0.78, color=colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(ctx.ordered_labels(qs2), rotation=20, ha="right")

    ymin, ymax = ctx.dice_ylim
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + 1e-9, 0.05))
    ax.set_yticks(np.arange(ymin, ymax + 1e-9, 0.01), minor=True)
    ax.grid(True, which="major", axis="y", alpha=0.30)
    ax.grid(True, which="minor", axis="y", alpha=0.12)
    ax.tick_params(axis="y", labelleft=True)
    if show_ylabel:
        ax.set_ylabel("Hard Dice@0.5")
    ax.set_title(title)
    style_axis(ax)


def draw_pareto_panel(
    ax: plt.Axes,
    ctx: PlotContext,
    xcol: str,
    xlabel: str,
    title: str,
    xlog: bool = False,
    show_ylabel: bool = True,
) -> None:
    xvals = pd.to_numeric(ctx.pareto[xcol], errors="coerce")
    valid = np.isfinite(xvals)
    if xlog:
        valid &= xvals > 0

    if not bool(valid.any()):
        ax.set_axis_off()
        ax.set_title(f"{title} (missing)")
        return

    for row in ctx.pareto.itertuples(index=False):
        xv = pd.to_numeric(pd.Series([getattr(row, xcol)]), errors="coerce").iloc[0]
        if pd.isna(xv) or (xlog and float(xv) <= 0):
            continue
        key = (row.base_model, row.ft_regime)
        ax.scatter(
            [float(xv)],
            [row.dice_mean],
            s=[ctx.sizes_by_key[key]],
            color=ctx.color_map.get(key, None),
        )

    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("Hard Dice@0.5 (TEST overall)")
    ax.set_title(title)
    ax.set_ylim(0.8, 1.0)
    if xlog:
        ax.set_xscale("log")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    style_axis(ax)


def build_shared_pareto_legend(ctx: PlotContext):
    handles = []
    labels = []
    for bm, fr in ctx.entries:
        key = (bm, fr)
        h = plt.scatter([], [], s=ctx.sizes_by_key[key], color=ctx.color_map.get(key, "gray"))
        handles.append(h)
        labels.append(ctx.legend_label(bm, fr))
    return handles, labels


# -----------------------------
# Export single-panel plots
# -----------------------------

def save_composite_plot(ctx: PlotContext) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    draw_composite_panel(ax, ctx, title="Composite Score (TEST overall, mean±std across seeds)")
    ax.set_ylabel("Composite Score = Dice@0.5 / log10(params_total)")
    save_fig(fig, ctx.settings.outdir / "composite_score_overall")


def save_single_pareto_plot(ctx: PlotContext, xcol: str, xlabel: str, outname: str, xlog: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_pareto_panel(ax, ctx, xcol=xcol, xlabel=xlabel, title="Pareto view (marker size ∝ params_total)", xlog=xlog, show_ylabel=True)
    handles, labels = build_shared_pareto_legend(ctx)
    ax.legend(
        handles=handles,
        labels=labels,
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
        labelspacing=2,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])
    save_fig(fig, ctx.settings.outdir / outname, use_tight_layout=False)


def save_overview_pareto_2x2(ctx: PlotContext) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    axs = axs.flatten()

    draw_pareto_panel(axs[0], ctx, "cpu_ms_mean", "CPU latency (ms/img)", "Pareto: Dice vs CPU latency", xlog=False, show_ylabel=True)
    draw_pareto_panel(axs[1], ctx, "gpu_ms_mean", "GPU latency (ms/img)", "Pareto: Dice vs GPU latency", xlog=False, show_ylabel=False)
    draw_pareto_panel(axs[2], ctx, "params_total", "Total parameters (log)", "Pareto: Dice vs Params", xlog=True, show_ylabel=True)

    if ctx.flops_col is not None and "flops_mean" in ctx.pareto.columns:
        draw_pareto_panel(axs[3], ctx, "flops_mean", "FLOPs (log)", "Pareto: Dice vs FLOPs", xlog=True, show_ylabel=False)
    else:
        axs[3].set_axis_off()
        axs[3].set_title("Pareto: Dice vs FLOPs (missing)")

    handles, labels = build_shared_pareto_legend(ctx)
    fig.subplots_adjust(bottom=0.20, top=0.93, left=0.07, right=0.98, wspace=0.12, hspace=0.32)
    fig.legend(
        handles=handles,
        labels=labels,
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
        labelspacing=2,
    )
    fig.suptitle("Phase-7 overview: Pareto views", y=0.98)
    save_fig(fig, ctx.settings.outdir / "overview_pareto_2x2", use_tight_layout=False)


def save_dice_panels_2x2(ctx: PlotContext) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    axes = axes.flatten()
    subset_titles = {
        "overall": "Overall",
        "PV01": "PV01 (0.1 m)",
        "PV03": "PV03 (0.3 m)",
        "PV08": "PV08 (0.8 m)",
    }
    for ax, subset in zip(axes, ["overall", "PV01", "PV03", "PV08"]):
        draw_dice_subset_panel(ax, ctx, subset=subset, title=subset_titles[subset], show_ylabel=True)
    fig.suptitle("Hard Dice@0.5 on TEST (mean±std across seeds)")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    save_fig(fig, ctx.settings.outdir / "dice_panels_overall_pv01_pv03_pv08", use_tight_layout=False)


# -----------------------------
# Final Figure 5 export
# -----------------------------

def save_figure5_jetson_main(ctx: PlotContext) -> None:
    fig = plt.figure(figsize=(17, 14))
    gs = GridSpec(
        nrows=4,
        ncols=6,
        figure=fig,
        height_ratios=[1.15, 1.0, 1.0, 0.08],
        hspace=0.65,
        wspace=0.75,
    )

    # Row 1
    ax_comp = fig.add_subplot(gs[0, 0:3])
    ax_overall = fig.add_subplot(gs[0, 3:6])
    draw_composite_panel(ax_comp, ctx, title="(a) Composite efficiency score")
    ax_comp.set_ylabel("Composite Score = Dice@0.5 / log10(params_total)")
    draw_dice_subset_panel(ax_overall, ctx, subset="overall", title="(b) Overall segmentation performance", show_ylabel=True)

    # Row 2
    ax_pv01 = fig.add_subplot(gs[1, 0:2])
    ax_pv03 = fig.add_subplot(gs[1, 2:4])
    ax_pv08 = fig.add_subplot(gs[1, 4:6])
    draw_dice_subset_panel(ax_pv01, ctx, subset="PV01", title="(c) PV01 segmentation performance", show_ylabel=True)
    draw_dice_subset_panel(ax_pv03, ctx, subset="PV03", title="(d) PV03 segmentation performance", show_ylabel=True)
    draw_dice_subset_panel(ax_pv08, ctx, subset="PV08", title="(e) PV08 segmentation performance", show_ylabel=True)

    # Row 3
    ax_gpu = fig.add_subplot(gs[2, 0:2])
    ax_cpu = fig.add_subplot(gs[2, 2:4])
    ax_params = fig.add_subplot(gs[2, 4:6])
    draw_pareto_panel(ax_gpu, ctx, "gpu_ms_mean", "GPU latency (ms/img)", "(f) GPU latency trade-off", xlog=False, show_ylabel=True)
    draw_pareto_panel(ax_cpu, ctx, "cpu_ms_mean", "CPU latency (ms/img)", "(g) CPU latency trade-off", xlog=False, show_ylabel=True)
    draw_pareto_panel(ax_params, ctx, "params_total", "Total parameters (log scale)", "(h) Model size trade-off", xlog=True, show_ylabel=True)

    # Bottom legend row
    ax_leg = fig.add_subplot(gs[3, :])
    ax_leg.axis("off")
    handles, labels = build_shared_pareto_legend(ctx)
    ax_leg.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=2,
        frameon=True,
        fontsize=10,
        title="Marker size ∝ params_total",
        title_fontsize=10,
        handletextpad=0.7,
        columnspacing=1.4,
        borderpad=0.9,
        labelspacing=1.2,
    )

    # fig.suptitle("Figure 5. Edge benchmark on Jetson: segmentation accuracy and efficiency trade-offs", y=0.992)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.94, bottom=0.035)
    save_fig(fig, ctx.settings.outdir / "figure5_jetson_benchmark", use_tight_layout=False)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    settings = PlotSettings(
        include_fullft=bool(getattr(config, "PHASE7_INCLUDE_FULLFT", False)),
        show_regime=bool(getattr(config, "PHASE7_INCLUDE_FULLFT", False)),
        model_order=list(getattr(
            config,
            "PHASE7_MODEL_ORDER",
            ["ultralight_phase6", "dlv3p_resnet50", "dlv3p_mobilenetv2", "unet_resnet34"],
        )),
        model_labels=dict(getattr(config, "PHASE7_MODEL_LABELS", {})),
        outdir=Path(getattr(config, "PHASE7_PLOTS_OUTDIR_JETSON")),
    )

    ctx = build_context(settings)

    save_composite_plot(ctx)
    save_single_pareto_plot(ctx, "cpu_ms_mean", "CPU latency (ms/img) — mean across seeds", "pareto_dice_vs_cpu_latency", xlog=False)
    save_single_pareto_plot(ctx, "gpu_ms_mean", "GPU latency (ms/img) — mean across seeds", "pareto_dice_vs_gpu_latency", xlog=False)
    save_single_pareto_plot(ctx, "params_total", "Total parameters (log scale) — mean across seeds", "pareto_dice_vs_params", xlog=True)

    if ctx.flops_col is not None and "flops_mean" in ctx.pareto.columns:
        flops_vals = pd.to_numeric(ctx.pareto["flops_mean"], errors="coerce")
        if np.isfinite(flops_vals).any() and (flops_vals > 0).any():
            save_single_pareto_plot(ctx, "flops_mean", "FLOPs (log scale) — mean across seeds", "pareto_dice_vs_flops", xlog=True)
        else:
            print("[WARN] FLOPs/MACs present but non-positive/invalid for log scale -> skipping pareto_dice_vs_flops")
    else:
        print("[WARN] No FLOPs column found in timing_per_repeat_csv. Skipping pareto_dice_vs_flops.")

    save_overview_pareto_2x2(ctx)
    save_dice_panels_2x2(ctx)
    save_figure5_jetson_main(ctx)

    ctx.pareto.to_csv(settings.outdir / "plot_data_pareto_agg.csv", index=False)
    print(f"[OK] Saved plots to: {settings.outdir.resolve()}")


if __name__ == "__main__":
    main()
