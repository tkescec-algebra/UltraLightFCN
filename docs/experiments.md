# Experiments Guide

## Main UltraLightFCN Path

The main paper-style reproduction path is Phase 0 through Phase 7:

- Preprocess raw BMP pairs into prepared splits.
- Run SimCLR HPO and downstream-aware SimCLR selection.
- Run final SimCLR pretraining.
- Run segmentation HPO and top-K confirmation.
- Run final TRAIN+VALID retraining and locked-box TEST evaluation.
- Benchmark final checkpoints on desktop and TorchScript/Jetson paths.

This path is required for main UltraLightFCN reproduction unless equivalent released artifacts are supplied.

## SOTA Comparator Path

SOTA comparator scripts are present:

- `train/phaseSOTA_stage1_dev80.py`
- `train/phaseSOTA_stage2_final90_test.py`
- `utils/sota_registry.py`

The registry defines:

- `dlv3p_resnet50`
- `dlv3p_mobilenetv2`
- `unet_resnet34`

Regimes are `minft` and `fullft`. These comparators are optional for reproducing UltraLightFCN training itself, but required for reproducing comparison tables/benchmark claims that include SOTA models.

## SOTA Extension Path

SOTA extension scripts are present:

- `train/phaseSOTA_extension_stage1_dev80.py`
- `train/phaseSOTA_extension_stage2_final90_test.py`
- `utils/sota_registry_extension.py`

The extension registry defines:

- `unet_mobilenetv2`
- `dlv3p_efficientnetb0`
- `unet_efficientnetb0`

Source comments/config indicate Stage 2 is restricted to `minft` only. This branch is optional unless a manuscript or report explicitly includes the extension results.

## no-SimCLR Ablation Path

no-SimCLR scripts are present:

- `optuna_study/phase4_seg_study_UltraLightFCN_no_simclr.py`
- `train/phase5_seg_retrain_topk_no_simclr.py`
- `train/phase6_seg_final_retrain90_test_no_simclr.py`
- `train/phase6_seg_final_retrain90_test_fixed_recipe_no_simclr.py`
- `train/rerun_phase6_no_simclr_seed71.py`
- `utils/no_simclr_guard.py`

There are two Phase 6 variants:

- A no-SimCLR branch using its own no-SimCLR Phase 5 winner.
- A fixed-recipe no-SimCLR branch using the main Phase 5 recipe without loading the Phase 3 checkpoint.

Which variant is public-facing requires a human interpretation decision.

## Architecture Ablation Path

Architecture ablation scripts and registries are present:

- `train/phase5_ultralight_arch_ablation_dev80.py`
- `train/phase6_ultralight_arch_ablation_final90_test.py`
- `models/UltraLightFCN_experimental_variants.py`
- `utils/ultralight_variant_registry.py`

Discovered variants include:

- `baseline`
- `no_mini_aspp`
- `no_shifted_sa`
- `no_mini_aspp_no_sa`
- `no_shallow_skip`
- `no_dilation`
- `decoder_narrow`
- `decoder_wide`

This branch is optional for main reproduction and requires careful handling because outputs exist under generated artifact folders.

## SimCLR Augmentation Ablation Path

SimCLR augmentation ablation code is present under `pretrain/aug_ablation/`:

- `aug_run.py`
- `aug_run_plan.json`
- `make_aug_heatmaps.py`
- `make_fig3_simclr.py`

This branch is optional for main reproduction and appears figure/diagnostic oriented.

## UMAP / Statistical / BDS Analysis Paths

Present analysis paths:

- UMAP: `pretrain/phase3_umap.py` and `pretrain/utils/umap_eval_dataset.py`.
- Phase 6 SimCLR/no-SimCLR statistics: `train/seg_phase6/ablation/phase6_simclr_vs_no_simclr_wilcoxon.py`.
- Wilcoxon simulation: `train/seg_phase6/wilcoxon_simulation.py`.
- BDS sensitivity: `test/bds_sensitivity_analysis/bds_sensitivity_analysis.py`.
- Phase 7 plotting: `test/plot_phase7_overall.py`, `test/plot_phase7_overall_jetson.py`, and training-loss plotting under `train/`.

These are optional reporting/analysis branches, not required for training the main model.

## Optional vs Required

Required for main UltraLightFCN reproduction:

- Phase 0-6 for model selection and final TEST report.
- Phase 7 for benchmark/deployment claims.

Optional for main model reproduction:

- SOTA comparators and SOTA extension.
- no-SimCLR ablations.
- Architecture ablations.
- SimCLR augmentation ablation.
- UMAP, statistical, plotting, and BDS sensitivity analyses.

## Open Questions

- Which no-SimCLR Phase 6 variant should be treated as the canonical public ablation?
- Which generated CSV/JSON/figure artifacts are paper-facing release artifacts versus local outputs?
- Whether SOTA extension results are part of the main manuscript or an appendix/supplement.
- Whether tracked generated-looking files such as `pretrain/aug_ablation/train.pid` should remain tracked.
