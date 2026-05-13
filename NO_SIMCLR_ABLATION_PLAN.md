# NO_SIMCLR_ABLATION_PLAN

## Scientific rationale
This ablation isolates the effect of SimCLR encoder initialization on `UltraLightFCN`. The architecture, dataset, augmentation pipeline, downstream HPO/search logic, top-K confirmation protocol, final train+valid retraining budget, and locked-box TEST evaluation are kept aligned with the main pipeline. The only intended scientific change is encoder initialization:

- Main branch: initialize the UltraLightFCN encoder from the Phase-3 SimCLR checkpoint.
- No-SimCLR ablation: initialize the full UltraLightFCN model from random weights.
- Main branch: searches `enc_lr_mult` because the encoder is pretrained and may need more conservative fine-tuning.
- No-SimCLR ablation: fixes `enc_lr_mult = 1.0` because encoder and decoder are both randomly initialized.

## What is controlled
- Same `UltraLightFCN` architecture via `SEG_PARAMS`.
- Same dataset splits in `dataset/train`, `dataset/valid`, `dataset/test`.
- Same `SolarPanelDataset` loading behavior.
- Same segmentation transforms from `utils/transforms.py`.
- Same metrics from `utils/metrics.py`.
- Same loss search space and reconstruction logic.
- Same seeds and deterministic worker seeding policy.
- Same Phase-4 objective: validation soft Dice with `avg_last_k`.
- Same Phase-5 tie-break logic.
- Same Phase-6 60-epoch train+valid budget.
- Same locked-box TEST rule and metric aggregation.
- Same encoder/decoder optimizer grouping for code-path compatibility, but equal LR in the no-SimCLR branch.

## What changes relative to the main UltraLightFCN pipeline
- `optuna_study/phase4_seg_study_UltraLightFCN_no_simclr.py`
  - Uses a distinct Optuna study name and SQLite file.
  - Reuses the same HPO subset lists for fair comparison.
  - Never loads `load_pretrained_encoder_into_ultralight`.
  - Removes the pretrained-encoder LR multiplier search and fixes `enc_lr_mult = 1.0`.
- `train/phase5_seg_retrain_topk_no_simclr.py`
  - Draws candidates from the no-SimCLR Phase-4 study.
  - Uses a distinct output root and winner JSON.
  - Trains from random initialization only.
  - Defaults `enc_lr_mult` to `1.0` so encoder and decoder train at the same base LR.
- `train/phase6_seg_final_retrain90_test_no_simclr.py`
  - Uses the no-SimCLR Phase-5 winner.
  - Uses a distinct output root and report filenames.
  - Trains from random initialization only before the single final TEST evaluation.
  - Defaults `enc_lr_mult` to `1.0` for final train90 as well.

## Expected scripts to run
1. Phase 4 HPO:
   - `python optuna_study/phase4_seg_study_UltraLightFCN_no_simclr.py`
2. Phase 5 top-K confirmation:
   - `python train/phase5_seg_retrain_topk_no_simclr.py`
3. Phase 6 final train90 + TEST:
   - `python train/phase6_seg_final_retrain90_test_no_simclr.py`

## Expected artifacts
- Phase 4:
  - Optuna study `UltraLightFCN_seg_softdice_no_simclr`
  - SQLite database `optuna_study/UltraLightFCN_study_no_simclr.db`
- Phase 5:
  - `train/seg_phase5/topk_retrain_no_simclr/phase5_topk_results_no_simclr.csv`
  - `train/seg_phase5/topk_retrain_no_simclr/phase5_winner_no_simclr.json`
  - per-candidate/per-seed LAST checkpoints and epoch logs
- Phase 6:
  - `train/seg_phase6/final_retrain90_no_simclr/phase6_test_per_seed_no_simclr.csv`
  - `train/seg_phase6/final_retrain90_no_simclr/phase6_test_report_no_simclr.json`
  - per-seed LAST checkpoints and epoch logs

## Locked-box TEST rule
- The TEST split must not be used for model selection, HPO, tie-breaking, threshold tuning, or early stopping.
- Phase 6 performs one final TEST evaluation only after training is complete.
- The no-SimCLR branch preserves that same rule.

## Manuscript interpretation
This ablation should be interpreted as:

- A controlled comparison between `UltraLightFCN + SimCLR initialization` and `UltraLightFCN + random initialization`.
- Not a change in architecture, data, augmentations, losses, or evaluation policy.
- Evidence for whether the SimCLR checkpoint improves the downstream selection-and-retraining pipeline under otherwise matched conditions.
- It avoids confounding SimCLR removal with an artificially reduced encoder learning rate inherited from pretrained fine-tuning.

## Learning-rate policy
- Main SimCLR branch:
  - keeps the `enc_lr_mult` search because the encoder starts from a Phase-3 SimCLR checkpoint.
- No-SimCLR branch:
  - fixes `enc_lr_mult = 1.0` and trains encoder and decoder jointly from scratch under the same base LR.
- Rationale:
  - this prevents the random-initialized encoder from being unfairly slowed by a pretrained-encoder learning-rate policy.

## Safety guard
The no-SimCLR scripts use `utils/no_simclr_guard.py` to fail fast if a Phase-3 checkpoint path or pretrained-init metadata is introduced accidentally.
