"""
make_supp_optuna_simclr.py

Supplementary Optuna plots for SimCLR study (from Optuna SQLite DB),
implemented WITHOUT optuna dependency.

This refactored version:
  - Uses a stable non-interactive backend (Agg) for safer PDF export on Windows.
  - Avoids heavy/unstable parallelism (n_jobs=1).
  - Replaces the old S2C "Parallel coordinates" with a more interpretable and lighter:
      S2C: Key-parameter landscape scatter (lr vs temperature), colored by objective.
  - Makes S2D "Slice grid" safer by rasterizing scatter points in PDF (rasterized=True).
  - Optionally allows disabling PDF export for the heavy S2D grid (default: False -> it DOES export PDF).

Outputs:
  - S2A_opt_history.(png|pdf)
  - S2B_param_importance.(png|pdf)
  - S2C_keyparam_scatter_lr_temp.(png|pdf)
  - S2D_slice_grid.(png|pdf)  (can disable PDF via config)
  - SuppFigS2_Optuna_SimCLR_2x2.(png|pdf)

If the study_name is wrong, the script prints available study names and fails fast.
"""

# IMPORTANT: set backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")

from dataclasses import dataclass
import os
import json
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class SuppOptunaSimCLRConfig:
    # Path to Optuna sqlite DB
    db_path: str = "UltraLightFCN_study.db"  # <- change if needed

    # Study name in the DB
    study_name: str = "UltraLightFCN_SimCLR_pretrain_RGB"

    # Outputs
    outdir: str = "./supp_optuna_simclr_outputs"
    dpi: int = 300

    # Plot controls
    random_state: int = 0
    n_perm_repeats: int = 15

    # Stability controls
    sklearn_n_jobs: int = 1  # keep 1 for Windows stability

    # If Windows still crashes on PDF for S2D, set this to True
    disable_slice_pdf: bool = False

    # Figure title
    suptitle: str = "Supplementary Figure S2. Optuna tuning for SimCLR"


CFG = SuppOptunaSimCLRConfig()


# -----------------------------
# Helpers
# -----------------------------
def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, outdir: str, stem: str, dpi: int = 300, save_pdf: bool = True) -> None:
    png_path = os.path.join(outdir, f"{stem}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {png_path}")

    if save_pdf:
        pdf_path = os.path.join(outdir, f"{stem}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[saved] {pdf_path}")


def _decode_param_value(param_value: float, distribution_json: str):
    """Decode categorical index to actual choice; keep floats as-is."""
    dist = json.loads(distribution_json)
    name = dist.get("name", "")
    attrs = dist.get("attributes", {})

    if name == "CategoricalDistribution":
        choices = attrs.get("choices", [])
        idx = int(round(float(param_value)))
        if 0 <= idx < len(choices):
            return choices[idx]
        return param_value  # fallback

    # FloatDistribution / IntDistribution: param_value already numeric
    return float(param_value)


def _is_log_param(param_name: str) -> bool:
    pn = param_name.lower()
    return ("lr" in pn) or ("learning_rate" in pn) or ("weight_decay" in pn) or (pn in {"wd"})


def _pick_param(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the candidate params found: {candidates}. Available: {list(df.columns)}")


# -----------------------------
# Load data from Optuna SQLite
# -----------------------------
def load_study_frames(db_path: str, study_name: str) -> Tuple[pd.DataFrame, str]:
    con = sqlite3.connect(db_path)

    studies = pd.read_sql("SELECT study_id, study_name FROM studies;", con)
    if study_name not in set(studies["study_name"]):
        avail = sorted(studies["study_name"].tolist())
        con.close()
        raise ValueError(f"Study '{study_name}' not found. Available study_name values: {avail}")

    study_id = int(studies.loc[studies["study_name"] == study_name, "study_id"].iloc[0])

    direction = pd.read_sql(
        f"SELECT direction FROM study_directions WHERE study_id={study_id};",
        con,
    )["direction"].iloc[0]

    # Trials with objective value (single-objective assumed)
    trials = pd.read_sql(
        f"""
        SELECT t.trial_id, t.number, t.state, v.value
        FROM trials t
        JOIN trial_values v ON v.trial_id=t.trial_id
        WHERE t.study_id={study_id}
        ORDER BY t.number;
        """,
        con,
    )

    # Params (with distributions)
    params = pd.read_sql(
        f"""
        SELECT t.trial_id, t.number, t.state, p.param_name, p.param_value, p.distribution_json
        FROM trials t
        JOIN trial_params p ON p.trial_id=t.trial_id
        WHERE t.study_id={study_id}
        ORDER BY t.number;
        """,
        con,
    )

    con.close()

    # Keep COMPLETE only
    trials = trials[trials["state"] == "COMPLETE"].copy()
    params = params[params["state"] == "COMPLETE"].copy()

    # Decode categorical params
    params["decoded_value"] = params.apply(
        lambda r: _decode_param_value(r["param_value"], r["distribution_json"]), axis=1
    )

    # Pivot params to wide format
    wide = params.pivot_table(
        index=["trial_id", "number"],
        columns="param_name",
        values="decoded_value",
        aggfunc="first",
    ).reset_index()

    # Merge objective
    df = pd.merge(trials[["trial_id", "number", "value"]], wide, on=["trial_id", "number"], how="inner")
    df = df.sort_values("number").reset_index(drop=True)

    return df, direction


# -----------------------------
# S2A: Optimization history
# -----------------------------
def plot_optimization_history(df: pd.DataFrame, direction: str, title: str = "S2A) Optimization history") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    x = df["number"].to_numpy()
    y = df["value"].to_numpy(dtype=float)

    ax.plot(x, y, marker="o", linestyle="none", markersize=3.5, alpha=0.7, label="trial value")

    if direction.upper() == "MINIMIZE":
        best = np.minimum.accumulate(y)
    else:
        best = np.maximum.accumulate(y)

    ax.plot(x, best, linewidth=2.0, label="best-so-far")

    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective value")
    ax.set_title(title)
    ax.legend(loc="best")

    fig.tight_layout()
    return fig


# -----------------------------
# S2B: Hyperparameter importance (Permutation Importance)
# -----------------------------
def plot_param_importance(
    df: pd.DataFrame,
    cfg: SuppOptunaSimCLRConfig,
    title: str = "S2B) Hyperparameter importance (permutation)"
) -> plt.Figure:
    y = df["value"].to_numpy(dtype=float)
    feature_cols = [c for c in df.columns if c not in {"trial_id", "number", "value"}]

    X_raw = df[feature_cols].copy()
    X = pd.get_dummies(X_raw, drop_first=False)

    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=cfg.random_state,
        n_jobs=cfg.sklearn_n_jobs,
    )
    rf.fit(X, y)

    pim = permutation_importance(
        rf, X, y,
        n_repeats=cfg.n_perm_repeats,
        random_state=cfg.random_state,
        n_jobs=cfg.sklearn_n_jobs,
    )

    imp = pd.Series(pim.importances_mean, index=X.columns)

    # Aggregate back to original param names
    agg: Dict[str, float] = {}
    for col, val in imp.items():
        base = col
        for p in feature_cols:
            if col == p or col.startswith(p + "_"):
                base = p
                break
        agg[base] = agg.get(base, 0.0) + float(val)

    imp_param = pd.Series(agg).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    y_pos = np.arange(len(imp_param))

    ax.barh(y_pos, imp_param.values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_param.index.tolist())
    ax.invert_yaxis()

    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.set_xlabel("Importance (sum of permutation importances)")
    ax.set_title(title)

    fig.tight_layout()
    return fig


# -----------------------------
# S2C (NEW): Key-parameter landscape (lr vs temperature) colored by objective
# -----------------------------
def plot_keyparam_scatter(
    df: pd.DataFrame,
    direction: str,
    title: str = "S2C) Key-parameter landscape (colored by objective)"
) -> plt.Figure:
    lr_name = _pick_param(df, ["simclr_lr", "lr", "learning_rate"])
    t_name = _pick_param(df, ["simclr_temperature", "temperature", "temp"])

    x = df[lr_name].astype(float).to_numpy()
    y = df[t_name].astype(float).to_numpy()
    z = df["value"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sc = ax.scatter(x, y, c=z, s=36, alpha=0.85, rasterized=True)

    ax.set_xscale("log")
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.20)
    ax.minorticks_on()

    ax.set_xlabel(lr_name)
    ax.set_ylabel(t_name)
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Objective value")

    # Optional: mark the best trial
    best_idx = int(np.argmin(z)) if direction.upper() == "MINIMIZE" else int(np.argmax(z))
    ax.scatter([x[best_idx]], [y[best_idx]], s=120, marker="*", edgecolors="k", linewidths=0.7, zorder=5)

    fig.tight_layout()
    return fig


# -----------------------------
# S2D: Slice plot grid (objective vs each param) - safer via rasterization
# -----------------------------
def plot_slice_grid(
    df: pd.DataFrame,
    title: str = "S2D) Slice plots (objective vs hyperparameters)"
) -> plt.Figure:
    param_cols = [c for c in df.columns if c not in {"trial_id", "number", "value"}]
    n = len(param_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(12.5, 3.8 * nrows))
    gs = fig.add_gridspec(nrows, ncols, wspace=0.28, hspace=0.35)

    y = df["value"].to_numpy(dtype=float)

    for idx, p in enumerate(param_cols):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])

        x = df[p].copy()
        if not np.issubdtype(x.dtype, np.number):
            x = pd.Categorical(x).codes.astype(float)
        else:
            x = x.astype(float)

        # Rasterize scatter points (greatly reduces PDF complexity/crash risk)
        ax.scatter(x.to_numpy(), y, s=18, alpha=0.7, rasterized=True)

        ax.set_title(p)
        ax.set_ylabel("Objective" if c == 0 else "")
        ax.set_xlabel(p)

        ax.grid(True, which="major", linestyle="--", alpha=0.30)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", alpha=0.15)

        if _is_log_param(p):
            if (x > 0).all():
                ax.set_xscale("log")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    # fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# -----------------------------
# Combined 2x2 Supplementary Figure S2
# -----------------------------
def make_supp_grid(df: pd.DataFrame, direction: str, cfg: SuppOptunaSimCLRConfig) -> plt.Figure:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.40)

    # S2A
    axA = fig.add_subplot(gs[0, 0])
    x = df["number"].to_numpy()
    y = df["value"].to_numpy(dtype=float)
    axA.plot(x, y, marker="o", linestyle="none", markersize=3.2, alpha=0.7, label="trial value")
    best = np.minimum.accumulate(y) if direction.upper() == "MINIMIZE" else np.maximum.accumulate(y)
    axA.plot(x, best, linewidth=2.0, label="best-so-far")
    axA.set_title("(a) Optimization history")
    axA.set_xlabel("Trial")
    axA.set_ylabel("Objective")
    axA.grid(True, linestyle="--", alpha=0.35)
    axA.legend(loc="best", fontsize=9)

    # S2B
    axB = fig.add_subplot(gs[0, 1])
    feature_cols = [c for c in df.columns if c not in {"trial_id", "number", "value"}]
    X_raw = df[feature_cols].copy()
    X = pd.get_dummies(X_raw, drop_first=False)

    rf = RandomForestRegressor(n_estimators=400, random_state=cfg.random_state, n_jobs=cfg.sklearn_n_jobs)
    rf.fit(X, y)

    pim = permutation_importance(
        rf, X, y,
        n_repeats=cfg.n_perm_repeats,
        random_state=cfg.random_state,
        n_jobs=cfg.sklearn_n_jobs,
    )
    imp = pd.Series(pim.importances_mean, index=X.columns)

    agg: Dict[str, float] = {}
    for col, val in imp.items():
        base = col
        for p in feature_cols:
            if col == p or col.startswith(p + "_"):
                base = p
                break
        agg[base] = agg.get(base, 0.0) + float(val)

    imp_param = pd.Series(agg).sort_values(ascending=False)
    y_pos = np.arange(len(imp_param))
    axB.barh(y_pos, imp_param.values)
    axB.set_yticks(y_pos)
    axB.set_yticklabels(imp_param.index.tolist())
    axB.invert_yaxis()
    axB.set_title("(b) Param importance")
    axB.set_xlabel("Permutation importance (sum)")
    axB.grid(True, axis="x", linestyle="--", alpha=0.35)

    # S2C (NEW)
    axC = fig.add_subplot(gs[1, 0])
    lr_name = _pick_param(df, ["simclr_lr", "lr", "learning_rate"])
    t_name = _pick_param(df, ["simclr_temperature", "temperature", "temp"])
    xx = df[lr_name].astype(float).to_numpy()
    yy = df[t_name].astype(float).to_numpy()
    zz = df["value"].astype(float).to_numpy()

    sc = axC.scatter(xx, yy, c=zz, s=28, alpha=0.85, rasterized=True)
    axC.set_xscale("log")
    axC.grid(True, which="major", linestyle="--", alpha=0.35)
    axC.grid(True, which="minor", linestyle=":", alpha=0.20)
    axC.minorticks_on()
    axC.set_title("(c) lr vs temperature (colored by objective)")
    axC.set_xlabel(lr_name)
    axC.set_ylabel(t_name)
    cbarC = fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.04)
    cbarC.set_label("Objective")

    # S2D (lightweight in 2x2): objective vs trial index (distribution view)
    axD = fig.add_subplot(gs[1, 1])
    axD.hist(zz, bins=min(30, max(10, len(zz) // 5)))
    axD.set_title("(d) Objective distribution")
    axD.set_xlabel("Objective value")
    axD.set_ylabel("Count")
    axD.grid(True, linestyle="--", alpha=0.30)

    # fig.suptitle(cfg.suptitle, y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# -----------------------------
# Run
# -----------------------------
def run(cfg: SuppOptunaSimCLRConfig) -> None:
    outdir = ensure_outdir(cfg.outdir)

    df, direction = load_study_frames(cfg.db_path, cfg.study_name)

    # S2A
    figA = plot_optimization_history(df, direction)
    save_fig(figA, outdir, "S2A_opt_history", dpi=cfg.dpi, save_pdf=True)
    plt.close(figA)

    # S2B
    figB = plot_param_importance(df, cfg)
    save_fig(figB, outdir, "S2B_param_importance", dpi=cfg.dpi, save_pdf=True)
    plt.close(figB)

    # S2C (NEW)
    figC = plot_keyparam_scatter(df, direction)
    save_fig(figC, outdir, "S2C_keyparam_scatter_lr_temp", dpi=cfg.dpi, save_pdf=True)
    plt.close(figC)

    # S2D (Slice grid) - can be heavy; optionally disable PDF
    figD = plot_slice_grid(df)
    save_fig(figD, outdir, "S2D_slice_grid", dpi=cfg.dpi, save_pdf=(not cfg.disable_slice_pdf))
    plt.close(figD)

    # Combined 2x2 (S2)
    figS2 = make_supp_grid(df, direction, cfg)
    save_fig(figS2, outdir, "Figure_S2_Optuna_SimCLR", dpi=cfg.dpi, save_pdf=True)
    plt.close(figS2)

    print("\nDone.")


if __name__ == "__main__":
    run(CFG)
