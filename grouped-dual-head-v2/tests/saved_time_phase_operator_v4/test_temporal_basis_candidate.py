from copy import deepcopy
import json
from pathlib import Path

import pytest

from saved_time_phase_operator_v4.continuation import (
    PilotSelection,
    build_temporal_delta_floor_pilot_config,
    build_temporal_gate_absorption_pilot_config,
    build_temporal_basis_pilot_config,
    temporal_basis_candidate_gate,
    temporal_delta_floor_candidate_gate,
)
from scripts.gate_saved_time_temporal_delta_floor_candidate import (
    _completed_optimizer_updates,
    _intervention_metadata,
)


def _selection(tmp_path: Path) -> PilotSelection:
    config = {
        "base_config": "base.yaml",
        "parent_identity": "v3.json",
        "artifact_dir": str(tmp_path / "parent"),
        "seed": 307,
        "epochs": 40,
        "macro_records": 12,
        "macros_per_update": 8,
        "microbatch_records": 3,
        "schedule_epoch_offset": 9,
        "time_policy": "appearance16",
        "travel_time_h5": "travel.h5",
        "variant_overrides": {"modes": 101, "coupled_2d_rank": 0},
        "checkpoint_transfer": {
            "allow_parent_manifest_mismatch": True,
            "parent_residual_already_active": True,
            "allow_spectral_mode_expansion": True,
        },
        "residual_recovery": {
            "activation_mode": "preserve",
            "stage_epoch_offset": 3,
            "decoder_only_epochs": 2,
        },
        "optimizer": {
            "dense_learning_rate": 5.0e-5,
            "geometry_learning_rate": 2.0e-5,
            "backbone_learning_rate": 2.0e-6,
            "schedule_epoch_offset": 3,
            "schedule_total_epochs": 40,
        },
        "loss": {"delta": 0.5, "spectrum": 0.1, "temporal_difference": 0.1},
        "validation": {
            "panel_records": 48,
            "frames_per_record": 32,
            "full_panel_records": 48,
            "final_frames_per_record": 401,
        },
        "gate": {
            "pilot_epochs": 3,
            "family_regression_tolerance": 0.03,
            "maximum_peak_cuda_gib": 23.0,
        },
    }
    metrics = {
        "aggregate_relative_l2": 0.40,
        "coarse_metrics": {"aggregate_relative_l2": 0.45},
        "family_relative_l2": {
            "uniform": 0.30,
            "layered": 0.40,
            "marmousi": 0.50,
        },
        "spectrum_relative_l2": {"high": 0.80},
    }
    return PilotSelection(
        name="corrected_parent",
        config_path=tmp_path / "parent.yaml",
        config=config,
        checkpoint=tmp_path / "epoch_0003.pt",
        checkpoint_identity=tmp_path / "run_identity.json",
        epoch=3,
        score=0.40,
        family_relative_l2=dict(metrics["family_relative_l2"]),
        metrics=metrics,
        loss_components={"delta": 0.90, "frame": 0.45},
    )


def test_build_temporal_basis_candidate_is_identity_bound_and_resets_warmup(
    tmp_path: Path,
):
    selection = _selection(tmp_path)

    generated = build_temporal_basis_pilot_config(
        selection,
        artifact_dir=tmp_path / "v31",
        temporal_basis_rank=96,
        effective_batch=96,
        pilot_epochs=3,
    )

    assert generated["parent_checkpoint"] == str(selection.checkpoint)
    assert generated["parent_checkpoint_identity"] == str(
        selection.checkpoint_identity
    )
    assert generated["artifact_dir"] == str((tmp_path / "v31").resolve())
    assert generated["macro_records"] == 12
    assert generated["macros_per_update"] == 8
    assert generated["microbatch_records"] == 4
    assert generated["variant_overrides"] == {
        "modes": 101,
        "coupled_2d_rank": 0,
        "temporal_basis_rank": 96,
    }
    transfer = generated["checkpoint_transfer"]
    assert transfer["parent_optimizer_state"] is False
    assert transfer["allow_new_temporal_basis_parameters"] is True
    assert "allow_spectral_mode_expansion" not in transfer
    assert generated["residual_recovery"] == {
        "activation_mode": "preserve",
        "stage_epoch_offset": 0,
        "decoder_only_epochs": 2,
    }
    assert generated["schedule_epoch_offset"] == 12
    assert generated["optimizer"]["schedule_epoch_offset"] == 6
    assert generated["seed"] == 307
    assert generated["gate"]["pilot_epochs"] == 3
    assert generated["time_policy"] == "appearance16"
    assert generated["validation"]["frames_per_record"] == 32


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("time_policy", "appearance4"), "appearance16"),
        (("validation", "panel_records", 12), "48-by-32"),
        (("variant_overrides", "temporal_basis_rank", 16), "rank zero"),
    ],
)
def test_build_temporal_basis_candidate_rejects_incomparable_parent(
    tmp_path: Path, mutation, message
):
    selection = _selection(tmp_path)
    config = deepcopy(selection.config)
    if len(mutation) == 2:
        config[mutation[0]] = mutation[1]
    else:
        config[mutation[0]][mutation[1]] = mutation[2]
    selection = PilotSelection(**{**selection.__dict__, "config": config})

    with pytest.raises(ValueError, match=message):
        build_temporal_basis_pilot_config(
            selection,
            artifact_dir=tmp_path / "v31",
            temporal_basis_rank=96,
        )


def test_build_temporal_delta_floor_candidate_continues_rank96_without_expansion(
    tmp_path: Path,
):
    selection = _selection(tmp_path)
    selection.config["variant_overrides"]["temporal_basis_rank"] = 96
    selection.config["checkpoint_transfer"][
        "allow_new_temporal_basis_parameters"
    ] = True
    selection.config["microbatch_records"] = 4

    generated = build_temporal_delta_floor_pilot_config(
        selection,
        artifact_dir=tmp_path / "v33",
        delta_energy_floor_fraction=0.1,
        dense_learning_rate_multiplier=2.0,
        pilot_epochs=3,
    )

    assert generated["parent_checkpoint"] == str(selection.checkpoint)
    assert generated["parent_checkpoint_identity"] == str(
        selection.checkpoint_identity
    )
    assert generated["artifact_dir"] == str((tmp_path / "v33").resolve())
    assert generated["microbatch_records"] == 5
    assert generated["variant_overrides"]["temporal_basis_rank"] == 96
    assert generated["loss"]["delta_energy_floor_fraction"] == pytest.approx(0.1)
    assert generated["optimizer"]["dense_learning_rate"] == pytest.approx(1.0e-4)
    assert generated["checkpoint_transfer"]["parent_optimizer_state"] is False
    assert "allow_new_temporal_basis_parameters" not in generated["checkpoint_transfer"]
    assert generated["residual_recovery"]["stage_epoch_offset"] == 0
    assert generated["residual_recovery"]["decoder_only_epochs"] == 3
    assert generated["schedule_epoch_offset"] == 12
    assert generated["optimizer"]["schedule_epoch_offset"] == 6
    assert generated["seed"] == 307


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("delta_energy_floor_fraction", 0.0, "delta energy floor"),
        ("delta_energy_floor_fraction", 1.1, "delta energy floor"),
        ("dense_learning_rate_multiplier", 0.0, "learning-rate multiplier"),
    ],
)
def test_build_temporal_delta_floor_candidate_rejects_invalid_intervention(
    tmp_path: Path, field, value, message
):
    selection = _selection(tmp_path)
    selection.config["variant_overrides"]["temporal_basis_rank"] = 96
    arguments = {
        "artifact_dir": tmp_path / "v33",
        "delta_energy_floor_fraction": 0.1,
        "dense_learning_rate_multiplier": 2.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        build_temporal_delta_floor_pilot_config(selection, **arguments)


def test_build_temporal_gate_absorption_candidate_is_fresh_and_function_bound(
    tmp_path: Path,
):
    selection = _selection(tmp_path)
    selection.config["variant_overrides"]["temporal_basis_rank"] = 96
    selection.config["microbatch_records"] = 4
    selection.config["loss"]["delta_energy_floor_fraction"] = 0.1
    selection.config["optimizer"]["dense_learning_rate"] = 1.0e-4

    generated = build_temporal_gate_absorption_pilot_config(
        selection,
        artifact_dir=tmp_path / "v35",
        pilot_epochs=3,
    )

    assert generated["parent_checkpoint"] == str(selection.checkpoint)
    assert generated["artifact_dir"] == str((tmp_path / "v35").resolve())
    assert generated["checkpoint_transfer"]["parent_optimizer_state"] is False
    assert generated["residual_recovery"]["absorb_temporal_basis_gate"] is True
    assert generated["residual_recovery"]["stage_epoch_offset"] == 0
    assert generated["residual_recovery"]["decoder_only_epochs"] == 3
    assert generated["loss"]["delta_energy_floor_fraction"] == pytest.approx(0.1)
    assert generated["optimizer"]["dense_learning_rate"] == pytest.approx(1.0e-4)
    assert generated["optimizer"]["temporal_basis_learning_rate"] == pytest.approx(
        1.0e-5
    )
    assert generated["schedule_epoch_offset"] == 12
    assert generated["optimizer"]["schedule_epoch_offset"] == 6
    assert generated["seed"] == 307


def test_gate_absorption_can_start_from_pre_floor_parent_at_calibrated_lr(
    tmp_path: Path,
):
    selection = _selection(tmp_path)
    selection.config["variant_overrides"]["temporal_basis_rank"] = 96

    generated = build_temporal_gate_absorption_pilot_config(
        selection,
        artifact_dir=tmp_path / "v35",
        delta_energy_floor_fraction=0.1,
        dense_learning_rate=1.0e-5,
        pilot_epochs=3,
    )

    assert generated["parent_checkpoint"] == str(selection.checkpoint)
    assert generated["loss"]["delta_energy_floor_fraction"] == pytest.approx(0.1)
    assert generated["optimizer"]["dense_learning_rate"] == pytest.approx(1.0e-5)
    assert generated["optimizer"]["temporal_basis_learning_rate"] == pytest.approx(
        1.0e-5
    )


def test_temporal_gate_records_gate_absorption_learning_rate_metadata():
    assert _intervention_metadata(
        {
            "dense_learning_rate": 1.0e-5,
            "gate_absorption": True,
        }
    ) == {
        "dense_learning_rate": pytest.approx(1.0e-5),
        "gate_absorption": True,
    }
    assert _intervention_metadata(
        {"dense_learning_rate_multiplier": 2.0}
    ) == {"dense_learning_rate_multiplier": pytest.approx(2.0)}


def _candidate_row():
    return {
        "event": "epoch",
        "epoch": 3,
        "global_step": 72,
        "physical_microbatch_records": 4,
        "peak_cuda_bytes": 22 * 1024**3,
        "ddp": {"global_macros_per_update": 8, "world_size": 4},
        "gradient_norms": {
            "temporal_basis_gate": 0.4,
            "temporal_basis_features": 0.2,
        },
        "last_update_loss_components": {"delta": 0.80, "frame": 0.40},
        "metrics": {
            "aggregate_relative_l2": 0.35,
            "coarse_metrics": {"aggregate_relative_l2": 0.45},
            "relative_improvement_vs_coarse": 0.02,
            "family_relative_l2": {
                "uniform": 0.28,
                "layered": 0.38,
                "marmousi": 0.49,
            },
            "spectrum_relative_l2": {"high": 0.79},
        },
    }


def test_temporal_basis_gate_accepts_complete_improving_candidate(tmp_path: Path):
    parent = _selection(tmp_path)

    report = temporal_basis_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=_candidate_row(),
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert report["passes"]
    assert all(report["checks"].values())
    assert report["candidate"]["effective_batch"] == 96


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (("metrics", "aggregate_relative_l2", 0.41), "aggregate_improved"),
        (("metrics", "coarse_metrics", "aggregate_relative_l2", 0.48), "same_coarse_panel"),
        (("metrics", "family_relative_l2", "marmousi", 0.54), "families_safe"),
        (("metrics", "spectrum_relative_l2", "high", 0.81), "high_band_safe"),
        (("last_update_loss_components", "delta", 0.91), "delta_improved"),
        (("metrics", "relative_improvement_vs_coarse", 0.0), "better_than_coarse"),
        (("gradient_norms", "temporal_basis_gate", 0.0), "temporal_gate_active"),
        (("gradient_norms", "temporal_basis_features", 0.0), "temporal_features_active"),
        (("physical_microbatch_records", 3), "physical_microbatch_4"),
        (("global_step", 71), "optimizer_updates_72"),
        (("peak_cuda_bytes", 23 * 1024**3), "cuda_peak_safe"),
    ],
)
def test_temporal_basis_gate_rejects_each_registered_failure(
    tmp_path: Path, mutation, failed_check
):
    candidate = _candidate_row()
    cursor = candidate
    for key in mutation[:-2]:
        cursor = cursor[key]
    cursor[mutation[-2]] = mutation[-1]
    parent = _selection(tmp_path)

    report = temporal_basis_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=candidate,
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert not report["passes"]
    assert not report["checks"][failed_check]


def test_temporal_delta_floor_gate_accepts_comparable_frame_improvement(
    tmp_path: Path,
):
    parent = _selection(tmp_path)

    report = temporal_delta_floor_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=_candidate_row(),
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert report["passes"]
    assert all(report["checks"].values())


def test_temporal_delta_floor_gate_separates_best_epoch_from_completed_updates(
    tmp_path: Path,
):
    parent = _selection(tmp_path)
    candidate = _candidate_row()
    candidate["epoch"] = 1
    candidate["global_step"] = 24

    report = temporal_delta_floor_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=candidate,
        completed_optimizer_updates=72,
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert report["passes"]
    assert report["candidate"]["selected_epoch_optimizer_updates"] == 24
    assert report["candidate"]["completed_optimizer_updates"] == 72


def test_temporal_delta_floor_gate_names_actual_physical_microbatch(tmp_path: Path):
    parent = _selection(tmp_path)
    candidate = _candidate_row()
    candidate["physical_microbatch_records"] = 5

    report = temporal_delta_floor_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=candidate,
        expected_physical_microbatch=5,
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert report["checks"]["physical_microbatch_5"]
    assert "physical_microbatch_4" not in report["checks"]


def test_temporal_delta_floor_gate_rejects_sub_noise_parent_improvement(
    tmp_path: Path,
):
    parent = _selection(tmp_path)
    candidate = _candidate_row()
    candidate["metrics"]["aggregate_relative_l2"] = 0.3998

    report = temporal_delta_floor_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=candidate,
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
        minimum_relative_improvement=1.0e-3,
    )

    assert not report["passes"]
    assert not report["checks"]["parent_relative_improvement_1e3"]


def test_temporal_delta_floor_gate_rejects_sub_noise_coarse_improvement(
    tmp_path: Path,
):
    parent = _selection(tmp_path)
    candidate = _candidate_row()
    candidate["metrics"]["relative_improvement_vs_coarse"] = 5.0e-4

    report = temporal_delta_floor_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=candidate,
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
        minimum_relative_improvement=1.0e-3,
    )

    assert not report["passes"]
    assert not report["checks"]["coarse_relative_improvement_1e3"]


def test_temporal_delta_floor_cli_counts_the_completed_run_not_the_best_epoch(
    tmp_path: Path,
):
    path = tmp_path / "metrics.jsonl"
    rows = (
        {"event": "epoch", "validation_scope": "pilot_fixed_panel", "global_step": 24},
        {"event": "epoch", "validation_scope": "pilot_fixed_panel", "global_step": 48},
        {"event": "epoch", "validation_scope": "pilot_fixed_panel", "global_step": 72},
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _completed_optimizer_updates(path) == 72


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (("last_update_loss_components", "frame", 0.46), "frame_improved"),
        (("last_update_loss_components", "delta", 1.0), "floored_delta_below_one"),
        (("metrics", "aggregate_relative_l2", 0.41), "aggregate_improved"),
        (("metrics", "coarse_metrics", "aggregate_relative_l2", 0.48), "same_coarse_panel"),
        (("metrics", "relative_improvement_vs_coarse", 0.0), "better_than_coarse"),
        (("gradient_norms", "temporal_basis_features", 0.0), "temporal_features_active"),
        (("global_step", 71), "optimizer_updates_72"),
    ],
)
def test_temporal_delta_floor_gate_rejects_registered_failures(
    tmp_path: Path, mutation, failed_check
):
    candidate = _candidate_row()
    cursor = candidate
    for key in mutation[:-2]:
        cursor = cursor[key]
    cursor[mutation[-2]] = mutation[-1]
    parent = _selection(tmp_path)

    report = temporal_delta_floor_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=candidate,
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert not report["passes"]
    assert not report["checks"][failed_check]
