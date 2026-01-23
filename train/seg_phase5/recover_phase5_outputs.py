import os
import json
import pandas as pd
import torch
from statistics import mean, pstdev
from datetime import datetime
from collections import defaultdict

OUT_ROOT = "topk"

SUMMARY_CSV = os.path.join(OUT_ROOT, "phase5_topk_results.csv")
WINNER_JSON = os.path.join(OUT_ROOT, "phase5_winner.json")

def load_params_from_last(ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        return None
    obj = torch.load(ckpt_path, map_location="cpu")
    return obj.get("params", None)

def summarize_one_run(run_dir: str):
    epoch_csv = os.path.join(run_dir, "epoch_log.csv")
    ckpt_last = os.path.join(run_dir, "last.pth")

    if not os.path.isfile(epoch_csv):
        return None

    df = pd.read_csv(epoch_csv)
    needed = {"epoch", "val_soft", "val_hard05", "avg_last_k_soft"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {epoch_csv}")

    best_idx = df["avg_last_k_soft"].idxmax()
    best = df.loc[best_idx]
    final = df.iloc[-1]

    # trial_X i seed_Y iz putanje
    p = run_dir.replace("\\", "/")
    trial = int(p.split("/trial_")[1].split("/")[0])
    seed = int(p.split("/seed_")[1].split("/")[0])

    params = load_params_from_last(ckpt_last)

    return {
        "candidate_id": trial,
        "seed": seed,
        "optuna_value_phase4": None,   # ako nemaš pristup optuna study-u, ostaje prazno
        "loss_name": None,             # isto
        "best_avg_last_k_soft": float(best["avg_last_k_soft"]),
        "best_epoch": int(best["epoch"]),
        "best_val_soft": float(best["val_soft"]),
        "best_val_hard05": float(best["val_hard05"]),
        "final_val_soft": float(final["val_soft"]),
        "final_val_hard05": float(final["val_hard05"]),
        "ckpt_last_path": ckpt_last if os.path.isfile(ckpt_last) else None,
        "params": params,              # bitno za winner.json
    }

def pick_winner(all_rows):
    per_cand = defaultdict(list)
    for r in all_rows:
        per_cand[int(r["candidate_id"])].append(r)

    scored = []
    for cand_id, runs in per_cand.items():
        scores = [float(x["best_avg_last_k_soft"]) for x in runs]
        mu = mean(scores)
        sd = pstdev(scores) if len(scores) > 1 else 0.0
        best_val_soft_max = max(float(x["best_val_soft"]) for x in runs)

        best_run = max(runs, key=lambda x: float(x["best_avg_last_k_soft"]))

        scored.append({
            "candidate_id": cand_id,
            "mean_best_avg_last_k_soft": float(mu),
            "std_best_avg_last_k_soft": float(sd),
            "best_val_soft_max": float(best_val_soft_max),
            "best_run": {
                "seed": int(best_run["seed"]),
                "best_avg_last_k_soft": float(best_run["best_avg_last_k_soft"]),
                "best_epoch": int(best_run["best_epoch"]),
                "ckpt_last_path": str(best_run["ckpt_last_path"]),
            },
            "params": dict(best_run["params"]) if isinstance(best_run.get("params"), dict) else None,
        })

    scored.sort(key=lambda x: (
        -float(x["mean_best_avg_last_k_soft"]),
        float(x["std_best_avg_last_k_soft"]),
        -float(x["best_val_soft_max"]),
    ))
    return scored[0], scored

def main():
    # Nađi sve seed foldere
    run_dirs = []
    for trial_name in os.listdir(OUT_ROOT):
        if not trial_name.startswith("trial_"):
            continue
        trial_dir = os.path.join(OUT_ROOT, trial_name)
        if not os.path.isdir(trial_dir):
            continue
        for seed_name in os.listdir(trial_dir):
            if seed_name.startswith("seed_"):
                run_dirs.append(os.path.join(trial_dir, seed_name))

    all_rows = []
    for rd in sorted(run_dirs):
        row = summarize_one_run(rd)
        if row is not None:
            all_rows.append(row)

    if not all_rows:
        raise SystemExit(f"Nema runova (epoch_log.csv) pod: {OUT_ROOT}")

    # 1) Summary CSV (filtriraj 'params' da CSV ne pukne)
    keys = [
        "candidate_id","seed","optuna_value_phase4","loss_name",
        "best_avg_last_k_soft","best_epoch","best_val_soft","best_val_hard05",
        "final_val_soft","final_val_hard05","ckpt_last_path"
    ]
    rows_sorted = sorted(all_rows, key=lambda r: float(r["best_avg_last_k_soft"]), reverse=True)
    df = pd.DataFrame([{k: r.get(k) for k in keys} for r in rows_sorted])
    df.to_csv(SUMMARY_CSV, index=False)
    print("Wrote:", SUMMARY_CSV)

    # 2) Winner JSON
    winner, ranking = pick_winner(all_rows)

    winner_obj = {
        "phase": 5,
        "selection_metric": "mean(best_avg_last_k_soft) over seeds; tie-break: min std, then max(best_val_soft)",
        "timestamp": datetime.now().isoformat(),
        # ovo su metapolja koja je originalni kod punio iz cfg; ostavljamo minimalno potrebno:
        "study_source": None,
        "data": None,
        "init": None,
        "training": None,
        "winner": winner,
        "ranking_topk": ranking,
    }

    with open(WINNER_JSON, "w", encoding="utf-8") as f:
        json.dump(winner_obj, f, ensure_ascii=False, indent=2)
    print("Wrote:", WINNER_JSON)

if __name__ == "__main__":
    main()
