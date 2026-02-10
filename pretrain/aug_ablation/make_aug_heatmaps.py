from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG (Phase3-style, like aug_run.py)
# =============================================================================
@dataclass(frozen=True)
class HeatmapConfig:
    runs_root: Path = Path("runs/aug_sensitivity")
    plan_json: Path = Path("aug_run_plan.json")

    out_dir_name: str = "../heatmaps"

    late_frac: float = 0.20
    min_late_points: int = 10
    only_done_runs: bool = True


CFG = HeatmapConfig()


# =============================================================================
# Helpers
# =============================================================================
def safe_read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_run_dirs(root: Path) -> List[Path]:
    run_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    run_dirs.sort(key=lambda p: p.name)
    return run_dirs


def read_train_steps_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expected columns: seed,epoch,step,lr,train_loss
    for col in ["step", "train_loss"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")
    df = df.sort_values("step").reset_index(drop=True)
    return df


def late_mean(series: np.ndarray, frac: float, min_points: int) -> float:
    n = len(series)
    if n == 0:
        return float("nan")
    k = max(min_points, int(math.ceil(frac * n)))
    k = min(k, n)
    return float(np.nanmean(series[-k:]))


def strip_seed(name: str) -> str:
    # removes trailing _seed<number>, e.g. baseline_seed13 -> baseline
    return re.sub(r"_seed\d+$", "", name)


def auc_mean_over_steps(steps: np.ndarray, values: np.ndarray) -> float:
    """
    Normalized AUC (average value over step axis).
    Returns AUC / (max_step - min_step).
    """
    if len(values) < 2:
        return float("nan")
    x = steps.astype(np.float64)
    y = values.astype(np.float64)
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return float("nan")
    dx = float(x[-1] - x[0])
    if dx <= 0:
        return float("nan")
    area = float(np.trapezoid(y, x))
    return area / dx


def choose_baseline_run_id(plan: Optional[List[Dict[str, Any]]], rows: pd.DataFrame) -> str:
    """
    Prefer baseline using plan:
      - run_id contains 'baseline' OR base == 'baseline'
    Fallback: first run_id in rows.
    """
    if plan:
        candidates: List[str] = []
        for r in plan:
            rid = str(r.get("run_id", ""))
            base = str(r.get("base", ""))
            if "baseline" in rid.lower() or base.lower() == "baseline":
                candidates.append(rid)
        for rid in candidates:
            if rid in set(rows["run_id"].tolist()):
                return rid

    for rid in rows["run_id"].tolist():
        if "baseline" in rid.lower():
            return rid
    return str(rows["run_id"].iloc[0])


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: Path,
):
    # Symmetric color range around 0 for diverging visualization
    max_abs = float(np.nanmax(np.abs(matrix))) if matrix.size else 0.0
    if not np.isfinite(max_abs) or max_abs == 0.0:
        max_abs = 1e-6

    fig, ax = plt.subplots(
        figsize=(max(8, 1.2 * len(col_labels)), max(4, 0.35 * len(row_labels)))
    )
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=+max_abs,
    )
    ax.set_title(title)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Δ vs baseline", rotation=90)

    # Annotate values for readability
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    out_dir = CFG.runs_root / CFG.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional plan (helps baseline detection + run ordering, if needed)
    plan = None
    if CFG.plan_json.exists():
        try:
            tmp = safe_read_json(CFG.plan_json)
            if isinstance(tmp, list):
                plan = tmp
        except Exception:
            plan = None

    run_dirs = find_run_dirs(CFG.runs_root)

    records = []
    for rd in run_dirs:
        if CFG.only_done_runs and not (rd / "DONE").exists():
            continue

        manifest_path = rd / "manifest.json"
        steps_path = rd / "train_steps.csv"
        rep_path = rd / "rep_metrics.json"

        if not manifest_path.exists() or not steps_path.exists() or not rep_path.exists():
            continue

        manifest = safe_read_json(manifest_path)
        rep = safe_read_json(rep_path)
        df_steps = read_train_steps_csv(steps_path)

        steps = df_steps["step"].to_numpy()
        loss = df_steps["train_loss"].to_numpy()

        rec = {
            "run_id": rd.name,
            "group": manifest.get("group", ""),
            "base": manifest.get("base", ""),
            "seed": manifest.get("seed", None),
            "max_steps": manifest.get("max_steps", None),

            "final_loss": float(loss[-1]) if len(loss) else float("nan"),
            "late_mean_loss": late_mean(loss, CFG.late_frac, CFG.min_late_points),
            "auc_mean_loss": auc_mean_over_steps(steps, loss),

            "alignment": float(rep.get("alignment", float("nan"))),
            "uniformity": float(rep.get("uniformity", float("nan"))),
        }

        # Optional metric (auto-included if present)
        if "view_ssim" in rep:
            rec["view_ssim"] = float(rep["view_ssim"])

        # Paper trail: store aug_config as JSON string
        aug_cfg = manifest.get("aug_config", {})
        rec["aug_config_json"] = json.dumps(aug_cfg, sort_keys=True)

        records.append(rec)

    if not records:
        raise RuntimeError(f"No completed runs found in: {CFG.runs_root}")

    rows = pd.DataFrame.from_records(records)

    # Determine baseline
    baseline_id = choose_baseline_run_id(plan, rows)
    if baseline_id not in set(rows["run_id"]):
        raise RuntimeError(f"Baseline run_id '{baseline_id}' not found among discovered runs")

    base_row = rows.loc[rows["run_id"] == baseline_id].iloc[0]

    # Deltas vs baseline
    rows["delta_late_mean_loss"] = rows["late_mean_loss"] - float(base_row["late_mean_loss"])
    rows["delta_alignment"] = rows["alignment"] - float(base_row["alignment"])
    rows["delta_uniformity"] = rows["uniformity"] - float(base_row["uniformity"])
    if "view_ssim" in rows.columns:
        rows["delta_view_ssim"] = rows["view_ssim"] - float(base_row["view_ssim"])

    # Impact score for sorting (baseline fixed first)
    rows["impact_score"] = (
        rows["delta_late_mean_loss"].abs()
        + rows["delta_alignment"].abs()
        + rows["delta_uniformity"].abs()
    )

    # Sort: baseline first, others by impact_score (desc)
    rows_sorted = rows.copy()
    rows_sorted["is_baseline"] = (rows_sorted["run_id"] == baseline_id).astype(int)
    rows_sorted = rows_sorted.sort_values(
        by=["is_baseline", "impact_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    rows = rows_sorted.drop(columns=["is_baseline"])

    # Save summary table (paper trail)
    summary_csv = out_dir / "summary_metrics.csv"
    rows.to_csv(summary_csv, index=False)

    # Heatmap columns
    heat_cols = ["delta_late_mean_loss", "delta_alignment", "delta_uniformity"]
    if "delta_view_ssim" in rows.columns:
        heat_cols.append("delta_view_ssim")

    # Display labels for nicer x-axis
    heat_col_labels = {
        "delta_late_mean_loss": "Δ NT-Xent (late)",
        "delta_alignment": "Δ Alignment",
        "delta_uniformity": "Δ Uniformity",
        "delta_view_ssim": "Δ View SSIM",
    }
    col_labels = [heat_col_labels.get(c, c) for c in heat_cols]

    matrix = rows[heat_cols].to_numpy(dtype=np.float64)
    row_labels = [strip_seed(rid) for rid in rows["run_id"].tolist()]
    baseline_label = strip_seed(baseline_id)

    # Save matrix csv (paper trail)
    mat_csv = out_dir / "heatmap_matrix.csv"
    pd.DataFrame(matrix, index=row_labels, columns=heat_cols).to_csv(mat_csv)

    # Plot heatmap
    plot_heatmap(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"Augmentation sensitivity (Δ vs baseline: {baseline_label})",
        out_path=out_dir / "aug_sensitivity_heatmap_deltas",
    )

    print(f"[OK] baseline={baseline_id}")
    print(f"[OK] saved: {summary_csv}")
    print(f"[OK] saved: {mat_csv}")
    print(f"[OK] saved: {out_dir / 'aug_sensitivity_heatmap_deltas.png'}")
    print(f"[OK] saved: {out_dir / 'aug_sensitivity_heatmap_deltas.pdf'}")


if __name__ == "__main__":
    main()
