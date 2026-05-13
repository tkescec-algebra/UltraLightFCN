import csv
import os
from statistics import mean, pstdev

from phase6_seg_final_retrain90_test_fixed_recipe_no_simclr import (
    Phase6FixedRecipeNoSimCLRConfig,
    _load_phase5_winner_recipe,
    run_one_seed,
    save_json,
    clear_cuda_cache,
    ABLATION_NAME,
    PRETRAINING_NAME,
    INIT_TYPE,
    SIMCLR_PRETRAINING_LOADED,
    USES_SIMCLR_PHASE5_WINNER_PARAMS,
    TEST_POLICY_NAME,
)


def mean_std(xs):
    if len(xs) == 1:
        return {"mean": float(xs[0]), "std": 0.0}
    return {"mean": float(mean(xs)), "std": float(pstdev(xs))}


def read_existing_rows(csv_path):
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path, rows, fieldnames):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = Phase6FixedRecipeNoSimCLRConfig()

    target_seed = 71

    candidate_id, params = _load_phase5_winner_recipe(cfg.phase5_winner_json)

    print(f"Rerunning only seed {target_seed}...")
    new_row = run_one_seed(
        cfg,
        candidate_id=candidate_id,
        params=params,
        seed=target_seed,
    )
    clear_cuda_cache()

    per_seed_csv = os.path.join(cfg.out_root, cfg.per_seed_test_csv_name)

    fieldnames = [
        "seed",
        "candidate_id",
        "loss_name",
        "test_loss",
        "test_soft_dice",
        "test_hard_dice@0.5",
        "test_soft_iou",
        "test_hard_iou@0.5",
        "test_precision@0.5",
        "test_recall@0.5",
        "ckpt_last_path",
    ]

    existing_rows = read_existing_rows(per_seed_csv)

    # Remove old seed 71 row if it exists.
    updated_rows = [
        row for row in existing_rows
        if int(row["seed"]) != target_seed
    ]

    # Add new seed 71 result.
    updated_rows.append(new_row)

    # Keep stable seed order.
    updated_rows = sorted(updated_rows, key=lambda r: int(r["seed"]))

    write_rows(per_seed_csv, updated_rows, fieldnames)

    report = {
        "phase": 6,
        "pretraining": PRETRAINING_NAME,
        "encoder_init": INIT_TYPE,
        "ablation": ABLATION_NAME,
        "simclr_pretraining_loaded": SIMCLR_PRETRAINING_LOADED,
        "uses_simclr_phase5_winner_params": USES_SIMCLR_PHASE5_WINNER_PARAMS,
        "phase5_recipe_source": cfg.phase5_winner_json,
        "test_policy": TEST_POLICY_NAME,
        "data": {
            "train": "train+valid (90%)",
            "test": cfg.test_split,
            "test_used_during_training": False,
        },
        "training": {
            "epochs": cfg.epochs,
            "seeds": [int(row["seed"]) for row in updated_rows],
            "save": "LAST only",
        },
        "winner_source": {
            "phase5_winner_json": cfg.phase5_winner_json,
            "candidate_id": candidate_id,
        },
        "init": {
            "type": INIT_TYPE,
            "pretraining": PRETRAINING_NAME,
            "encoder_init": INIT_TYPE,
            "simclr_pretraining_loaded": SIMCLR_PRETRAINING_LOADED,
        },
        "metrics_test": {
            "soft_dice": mean_std([float(r["test_soft_dice"]) for r in updated_rows]),
            "hard_dice@0.5": mean_std([float(r["test_hard_dice@0.5"]) for r in updated_rows]),
            "soft_iou": mean_std([float(r["test_soft_iou"]) for r in updated_rows]),
            "hard_iou@0.5": mean_std([float(r["test_hard_iou@0.5"]) for r in updated_rows]),
            "precision@0.5": mean_std([float(r["test_precision@0.5"]) for r in updated_rows]),
            "recall@0.5": mean_std([float(r["test_recall@0.5"]) for r in updated_rows]),
            "loss": mean_std([float(r["test_loss"]) for r in updated_rows]),
        },
        "runs": updated_rows,
        "artifacts": {
            "per_seed_csv": per_seed_csv,
        },
    }

    report_path = os.path.join(cfg.out_root, cfg.report_json_name)
    save_json(report_path, report)

    print(f"Updated per-seed CSV: {per_seed_csv}")
    print(f"Updated report JSON: {report_path}")
    print(f"New seed {target_seed} TEST result:")
    print(new_row)


if __name__ == "__main__":
    main()