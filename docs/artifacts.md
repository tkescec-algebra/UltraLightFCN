# Artifact Guide

This document describes generated outputs and release-artifact candidates. It does not authorize cleanup.

## What Belongs In Git

Source code, lightweight configuration encoded in source, documentation, and deliberate small analysis scripts belong in Git. Generated datasets, checkpoints, Optuna databases, run logs, benchmark outputs, and exports generally do not.

## What Is Generated Locally

Generated artifacts include:

- Raw/local data and prepared splits.
- Preprocessing temp caches.
- Optuna SQLite databases and subset lists.
- SimCLR and segmentation checkpoints.
- Training logs and per-epoch CSVs.
- Phase reports and summaries.
- Benchmark timing/quality reports.
- Per-image metric `.npz` files.
- TorchScript `.ts` exports and export metadata.
- Plot outputs and paper figures.

## External Release Artifacts

For artifact-assisted reproduction, consider publishing:

- Prepared dataset instructions or dataset access details, if redistribution is allowed.
- Phase 1/4 Optuna DBs or summarized trial exports.
- Phase 2 top-K CSV and selected candidate metadata.
- Phase 3 final SimCLR checkpoint.
- Phase 5 winner JSON.
- Phase 6 final reports and LAST checkpoints.
- SOTA final reports and LAST checkpoints if benchmark comparisons are claimed.
- TorchScript exports and metadata for deployment benchmarking.
- Final benchmark reports used for paper plots.

## Generated Folder Inventory

| Path | Description |
| --- | --- |
| `data/` | Local raw input data. Ignored. |
| `dataset/` | Prepared split dataset. Ignored. |
| `temp/` | Normalized PNG preprocessing cache. Ignored. |
| `optuna_study/*.db` | Optuna SQLite storage, including `UltraLightFCN_study.db`. Ignored by `optuna_study/.gitignore`. |
| `optuna_study/runs/` | HPO split lists and run artifacts. Ignored by `optuna_study/.gitignore`. |
| `pretrain/checkpoints/` | Phase 2/3 SimCLR checkpoints and metrics. Ignored by `pretrain/.gitignore`. |
| `train/seg_phase5/` | Phase 5 top-K outputs and Phase 5.1 ablation outputs. Ignored by `train/.gitignore` except tracked scripts under this tree. |
| `train/seg_phase6/` | Phase 6 final outputs, no-SimCLR outputs, statistics, and simulations. Ignored by `train/.gitignore` except tracked scripts under this tree. |
| `train/seg_sota/` | SOTA comparator outputs. Ignored by `train/.gitignore`. |
| `train/seg_sota_extension/` | SOTA extension outputs. Ignored by `train/.gitignore`. |
| `train/seg_experimental_ablation/` | Architecture-ablation outputs. Ignored by `train/.gitignore`. |
| `test/bench_phase7/` | Desktop Phase 7 benchmark outputs and plots. Ignored by `test/.gitignore`. |
| `test/bench_phase7_jetson_ts/` | TorchScript/Jetson benchmark outputs and plots. Ignored by `test/.gitignore`. |
| `test/bds_sensitivity_analysis/bds_sensitivity_out/` | BDS sensitivity outputs. Ignored by `test/.gitignore`. |
| `tools/export_torchscript_10/` | TorchScript exports and `index.json`. Ignored by `tools/.gitignore`. |

## Tracked Generated-Looking Outputs Requiring Human Decision

Current tracked files matching generated-output patterns include:

- `pretrain/aug_ablation/train.pid`.
- `pretrain/aug_ablation/aug_run_plan.json`.
- `test/bds_sensitivity_analysis/bds_sensitivity_input_desktop.csv`.
- `test/bds_sensitivity_analysis/bds_sensitivity_input_jetson.csv`.
- `train/seg_phase5/ablation/phase5.1_ablation_output.py`.
- `train/seg_phase6/ablation/phase6_simclr_vs_no_simclr_wilcoxon.py`.
- `train/seg_phase6/wilcoxon_simulation.py`.

Architecture-ablation and BDS outputs are present in ignored generated folders. Any decision to remove, keep, release, or reclassify these artifacts should be made by a human maintainer.

## Policy

This document describes artifacts; it does not approve cleanup. Do not delete, move, rename, archive, regenerate, or untrack artifacts unless explicitly requested.
