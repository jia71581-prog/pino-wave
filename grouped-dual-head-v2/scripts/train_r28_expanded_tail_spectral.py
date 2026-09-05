#!/usr/bin/env python3
"""R28 expanded-coverage training on a never-opened train holdout."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
R26_PATH = SCRIPT_PATH.with_name("train_r26_tail_spectral_pilot.py")
SPEC = importlib.util.spec_from_file_location("r26_components", R26_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R26 components: {R26_PATH}")
r26 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r26
SPEC.loader.exec_module(r26)
r25 = r26.r25


def argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"missing required argument {name}") from error


def write_preregistration() -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output_dir = Path(argument_value("--output-dir")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "r28_expanded_tail_spectral_preregistration_v1",
        "status": "frozen_before_training",
        "role": "expanded_train_coverage_with_never_opened_group_disjoint_train_holdout",
        "model": {
            "local_branch": "r25_coarse_residual_unet",
            "spectral_branch": "two_global_complex_fourier_blocks_width16_modes24",
            "spectral_cap": 0.12,
            "total_correction_cap": float(argument_value("--correction-cap")),
            "zero_gated_identity_start": True,
        },
        "optimization": {
            "epochs": int(argument_value("--epochs")),
            "per_gpu_batch_size": int(argument_value("--batch-size")),
            "learning_rate": float(argument_value("--learning-rate")),
            "hinge_weight": float(argument_value("--hinge-weight")),
            "gradient_weight": float(argument_value("--gradient-weight")),
            "sampling": "repeat_all_marmousi_and_top_quartile_parent_error_records_once",
            "loss": "0.25_mean_plus_0.75_parent_difficulty_weighted_plus_1.0_batch_top50pct_CVaR_proxy",
            "configuration_frozen_before_new_holdout_evaluation": True,
        },
        "data_access": {
            "fit": "train_only_896_records",
            "holdout": "train_only_56_records_from_groups_never_opened_in_R25_R26_R27",
            "truth_role": "training_supervision_only_not_deployment_input",
            "validation_opened": False,
            "test_id_opened": False,
        },
        "success_gate": {
            "record_rel_l2_mean_lte": 0.05,
            "record_rel_l2_max_lte": 0.05,
            "both_required": True,
        },
        "evidence_boundary": "Passing R28 authorizes one frozen validation run but is not itself validation evidence.",
        "script_sha256": r25.sha256_file(SCRIPT_PATH),
        "r26_components_sha256": r25.sha256_file(R26_PATH),
        "r25_driver_sha256": r25.sha256_file(r26.R25_PATH),
    }
    r25.atomic_json(payload, output_dir / "r28_preregistration.json")


def write_terminal_sidecar() -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output_dir = Path(argument_value("--output-dir")).expanduser().resolve()
    base = json.loads((output_dir / "terminal.json").read_text(encoding="utf-8"))
    sidecar = {
        "schema": "r28_expanded_tail_spectral_terminal_v1",
        "status": base["status"],
        "best_epoch": base["best_epoch"],
        "best_metrics": base["best_metrics"],
        "checkpoint": base["checkpoint"],
        "checkpoint_sha256": base["checkpoint_sha256"],
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": r25.sha256_file(SCRIPT_PATH),
        "r26_components_sha256": r25.sha256_file(R26_PATH),
        "r25_driver_sha256": r25.sha256_file(r26.R25_PATH),
    }
    r25.atomic_json(sidecar, output_dir / "r28_terminal.json")


def main() -> None:
    write_preregistration()
    r25.FitFrameDataset = r26.TailAwareFitFrameDataset
    r25.CoarseResidualUNet = r26.TailSpectralResidualUNet
    r25.train_loss = r26.tail_risk_loss
    r25.main()
    write_terminal_sidecar()


if __name__ == "__main__":
    main()
