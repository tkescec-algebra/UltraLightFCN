from __future__ import annotations

from typing import Any, Dict


NO_SIMCLR_PRETRAINING = "none"
NO_SIMCLR_ENCODER_INIT = "random"
NO_SIMCLR_ABLATION = "no_simclr"
NO_SIMCLR_TEST_POLICY = "locked_box_single_final_evaluation"


def build_no_simclr_metadata(*, phase: int) -> Dict[str, Any]:
    """Canonical metadata for the no-SimCLR ablation branch."""
    meta: Dict[str, Any] = {
        "phase": int(phase),
        "pretraining": NO_SIMCLR_PRETRAINING,
        "encoder_init": NO_SIMCLR_ENCODER_INIT,
        "ablation": NO_SIMCLR_ABLATION,
    }
    if int(phase) == 6:
        meta["test_policy"] = NO_SIMCLR_TEST_POLICY
    return meta


def assert_no_pretrained_checkpoint(
    *,
    branch_name: str,
    phase3_last_ckpt: str | None = None,
    init_metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Fail fast if the no-SimCLR branch is accidentally configured to use a Phase-3 checkpoint
    or if upstream metadata indicates pretrained initialization.
    """
    if phase3_last_ckpt not in (None, ""):
        raise RuntimeError(
            f"{branch_name}: no-SimCLR ablation must not use a Phase-3 checkpoint "
            f"(received phase3_last_ckpt={phase3_last_ckpt!r})."
        )

    if init_metadata is not None:
        pretraining = init_metadata.get("pretraining")
        encoder_init = init_metadata.get("encoder_init")
        if pretraining not in (None, NO_SIMCLR_PRETRAINING):
            raise RuntimeError(
                f"{branch_name}: expected pretraining={NO_SIMCLR_PRETRAINING!r}, got {pretraining!r}."
            )
        if encoder_init not in (None, NO_SIMCLR_ENCODER_INIT):
            raise RuntimeError(
                f"{branch_name}: expected encoder_init={NO_SIMCLR_ENCODER_INIT!r}, got {encoder_init!r}."
            )
