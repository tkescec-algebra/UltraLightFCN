# Reproduction Guide

This guide documents the current phase-based workflow from source files only. It does not imply one-command reproducibility, and it does not require rerunning expensive phases when released artifacts are available.

## Phase 0-7 Overview

| Phase | Script | Input | Output | Required for main reproduction? | Expensive? |
| --- | --- | --- | --- | --- | --- |
| 0 | `preprocessing/image_preprocessing.py` | Raw recursive BMP pairs in `../data` | `../temp`, `../dataset/train`, `../dataset/valid`, `../dataset/test` | Yes, unless prepared `dataset/` is provided | Yes |
| 1 | `optuna_study/phase1_simclr_study_UltraLightFCN.py` | `../dataset/train` images | `optuna_study/UltraLightFCN_study.db`, `optuna_study/runs/simclr_hpo/*.txt` | Yes, unless Optuna study/results are provided | Yes |
| 2 | `pretrain/phase2_simclr_retrain_top10_downstream.py` | Phase 1 study, `../dataset/train`, `../dataset/valid` | `pretrain/checkpoints/simclr_topk_retrain_downstream/phase2_topk_results.csv`, candidate encoder checkpoints | Yes, unless Phase 2 CSV/checkpoints are provided | Yes |
| 3 | `pretrain/phase3_simclr_full_pretrain.py` | Phase 2 CSV, `../dataset/train` | `pretrain/checkpoints/simclr_phase3/phase3_seed13_last.pth`, metrics JSON/CSV | Yes, unless Phase 3 checkpoint is provided | Yes |
| 4 | `optuna_study/phase4_seg_study_UltraLightFCN.py` | Phase 3 checkpoint, `../dataset/train`, `../dataset/valid` | Segmentation trials in `optuna_study/UltraLightFCN_study.db`, `optuna_study/runs/hpo_subsets/*.txt` | Yes, unless segmentation Optuna study is provided | Yes |
| 5 | `train/phase5_seg_retrain_topk.py` | Phase 4 study, Phase 3 checkpoint, TRAIN/VALID | `train/seg_phase5/topk_retrain/phase5_topk_results.csv`, `phase5_winner.json` | Yes, unless winner JSON is provided | Yes |
| 6 | `train/phase6_seg_final_retrain90_test.py` | Phase 5 winner JSON, Phase 3 checkpoint, TRAIN+VALID, TEST | `train/seg_phase6/final_retrain90/phase6_test_per_seed.csv`, `phase6_test_report.json`, LAST checkpoints | Yes for final TEST results | Yes |
| 7a | `test/phase7_test_benchmark.py` | Phase 6 and SOTA final reports/checkpoints, TEST | `test/bench_phase7/<timestamp>/...` | Yes for desktop benchmark claims | Yes |
| 7b | `tools/export_torchscript_phase7_torch10.py` | Phase 6 and SOTA final reports/checkpoints | `tools/export_torchscript_10/*.ts`, `index.json` | Yes for TorchScript/Jetson benchmarking | Yes |
| 7c | `test/phase7_jetson_torchscript_benchmark.py` | `tools/export_torchscript_10`, TEST | `test/bench_phase7_jetson_ts/<timestamp>/...` | Yes for Jetson/TorchScript claims | Yes |

## Main Reproduction Path

1. Prepare the dataset with Phase 0 or obtain a compatible prepared `dataset/` folder.
2. Run SimCLR HPO with Phase 1.
3. Run Phase 2 top-K SimCLR retraining and downstream validation selection.
4. Run Phase 3 final SimCLR pretraining on TRAIN only.
5. Run Phase 4 segmentation HPO using the Phase 3 LAST checkpoint.
6. Run Phase 5 top-K confirmation to produce the segmentation winner JSON.
7. Run Phase 6 final TRAIN+VALID retrain and locked TEST evaluation.
8. Run Phase 7 desktop benchmark, TorchScript export, and TorchScript/Jetson benchmark as needed for deployment claims.

Do not use TEST for selection, tuning, or decisions. Phase 6 and Phase 7 consume TEST for final reporting only.

## Artifact-Assisted Reproduction

Users may avoid rerunning all HPO/training phases if released artifacts are provided. The minimum shortcut artifacts depend on the desired entry point:

- To start at Phase 3: provide `pretrain/checkpoints/simclr_topk_retrain_downstream/phase2_topk_results.csv`.
- To start at Phase 4 or Phase 5: provide `pretrain/checkpoints/simclr_phase3/phase3_seed13_last.pth`.
- To start at Phase 6: provide `train/seg_phase5/topk_retrain/phase5_winner.json` and its referenced Phase 3 checkpoint.
- To start at Phase 7 desktop benchmarking: provide Phase 6 final report/checkpoints and SOTA final reports/checkpoints if comparing against SOTA.
- To start at Jetson/TorchScript benchmarking: provide exported `.ts` files and metadata under `tools/export_torchscript_10`.

Artifact-assisted reproduction should state exactly which artifacts were supplied externally.

## Required Output Dependencies

- Phase 2 uses the Phase 1 Optuna study named `UltraLightFCN_SimCLR_pretrain_RGB`.
- Phase 3 uses `pretrain/checkpoints/simclr_topk_retrain_downstream/phase2_topk_results.csv`.
- Phase 4 uses `pretrain/checkpoints/simclr_phase3/phase3_seed13_last.pth`.
- Phase 5 uses the Phase 4 study named `UltraLightFCN_seg_softdice` and writes `train/seg_phase5/topk_retrain/phase5_winner.json`.
- Phase 6 uses the Phase 5 winner JSON and the Phase 3 checkpoint referenced inside it.
- Phase 7 consumes Phase 6 and SOTA reports/checkpoints through roster JSON paths.

## TRAIN/VALID/TEST Separation

The source comments and configs enforce a staged split policy:

- Phase 1/3 SimCLR pretraining uses image-only training data and does not use TEST.
- Phase 2 uses TRAIN for SimCLR retraining and official VALID for downstream-aware selection.
- Phase 4 and Phase 5 use TRAIN/VALID only.
- Phase 6 trains on TRAIN+VALID and evaluates TEST as a locked-box final report.
- Phase 7 benchmarks already trained final checkpoints and recomputes TEST metrics for reporting.

## Seeds And Determinism

`utils/repro.py` defines `GLOBAL_SEED = 42`, `set_global_seed`, and `seed_worker`. Several phases use seeded `DataLoader` generators. Some HPO/final scripts set `deterministic=False` for speed while preserving seeded loaders; strict bit-level determinism is not guaranteed unless a specific phase is configured that way.

Known phase seeds include Phase 3 seed 13, Phase 5 seeds `(13, 37, 71)`, and the current Phase 6 main `Phase6Config.seeds` set of 20 seeds: `(13, 37, 71, 101, 131, 151, 181, 211, 241, 271, 307, 353, 409, 457, 521, 601, 701, 809, 907, 997)`.

## Known Path Assumptions

- Most scripts assume they are run from their own folder because paths use `../dataset`, `../pretrain`, or `../optuna_study`.
- `utils/config.py` hardcodes Phase 7 plotting report paths with specific timestamps.
- TorchScript export writes `tools/export_torchscript_10`; `test/smoke_ts.py` references `export_torchscript/...`, and `tools/check_all_torchscript_models.py` references `/work/tools/export_torchscript_10`.
