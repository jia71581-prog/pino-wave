from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from saved_time_phase_operator_v4.continuation import (
    PilotSelection,
    build_update_density_pilot_config,
    update_density_candidate_gate,
)
from scripts.gate_saved_time_update_density_candidate import (
    _completed_optimizer_updates,
    main as gate_main,
)
from scripts.prepare_saved_time_update_density_candidate import main as prepare_main


def _selection(tmp_path: Path) -> PilotSelection:
    config = {
        "base_config": "base.yaml",
        "artifact_dir": str(tmp_path / "parent"),
        "epochs": 40,
        "macro_records": 12,
        "macros_per_update": 16,
        "microbatch_records": 8,
        "schedule_epoch_offset": 9,
        "time_policy": "appearance16",
        "travel_time_h5": "travel.h5",
        "seed": 331,
        "variant_overrides": {"modes": 101, "coupled_2d_rank": 16},
        "checkpoint_transfer": {
            "allow_parent_manifest_mismatch": True,
            "allow_spectral_mode_expansion": True,
            "allow_new_coupled_2d_parameters": True,
            "allow_new_temporal_basis_parameters": True,
            "parent_residual_already_active": True,
            "parent_optimizer_state": False,
        },
        "residual_recovery": {
            "activation_mode": "preserve",
            "stage_epoch_offset": 4,
            "decoder_only_epochs": 2,
        },
        "optimizer": {
            "dense_learning_rate": 1.0e-4,
            "geometry_learning_rate": 2.0e-5,
            "backbone_learning_rate": 2.0e-6,
            "schedule_epoch_offset": 4,
            "schedule_total_epochs": 40,
        },
        "loss": {
            "delta": 0.5,
            "spectrum": 0.1,
            "temporal_difference": 0.1,
        },
        "validation": {
            "panel_records": 48,
            "frames_per_record": 32,
            "full_panel_records": 48,
            "final_frames_per_record": 401,
        },
        "gate": {
            "pilot_epochs": 2,
            "family_regression_tolerance": 0.03,
            "maximum_peak_cuda_gib": 23.0,
        },
    }
    metrics = {
        "aggregate_relative_l2": 0.40,
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
        checkpoint=tmp_path / "epoch_0002.pt",
        checkpoint_identity=tmp_path / "run_identity.json",
        epoch=2,
        score=0.40,
        family_relative_l2=dict(metrics["family_relative_l2"]),
        metrics=metrics,
        loss_components={"delta": 0.90},
    )


def test_build_update_density_candidate_changes_only_update_contract(tmp_path: Path):
    selection = _selection(tmp_path)

    generated = build_update_density_pilot_config(
        selection,
        artifact_dir=tmp_path / "v28",
        effective_batch=96,
        pilot_epochs=3,
    )

    assert generated["parent_checkpoint"] == str(selection.checkpoint)
    assert generated["parent_checkpoint_identity"] == str(
        selection.checkpoint_identity
    )
    assert generated["artifact_dir"] == str((tmp_path / "v28").resolve())
    assert generated["macro_records"] == 12
    assert generated["macros_per_update"] == 8
    assert generated["microbatch_records"] == 3
    assert generated["gate"]["pilot_epochs"] == 3
    assert generated["gate"]["external_evidence_gate"] is True
    assert generated["time_policy"] == "appearance16"
    assert generated["travel_time_h5"] == "travel.h5"
    assert generated["loss"] == selection.config["loss"]
    assert generated["variant_overrides"] == selection.config["variant_overrides"]
    assert generated["schedule_epoch_offset"] == 11
    assert generated["optimizer"]["schedule_epoch_offset"] == 6
    assert generated["residual_recovery"]["stage_epoch_offset"] == 6
    transfer = generated["checkpoint_transfer"]
    assert transfer["parent_optimizer_state"] is False
    assert transfer["parent_residual_already_active"] is True
    assert "allow_new_coupled_2d_parameters" not in transfer
    assert "allow_new_temporal_basis_parameters" not in transfer
    assert "allow_spectral_mode_expansion" not in transfer


@pytest.mark.parametrize(
    ("effective_batch", "message"),
    [
        (0, "positive"),
        (100, "macro_records"),
        (48, "instantaneous"),
    ],
)
def test_build_update_density_candidate_rejects_invalid_effective_batch(
    tmp_path: Path,
    effective_batch: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        build_update_density_pilot_config(
            _selection(tmp_path),
            artifact_dir=tmp_path / "v28",
            effective_batch=effective_batch,
            pilot_epochs=3,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("time_policy", "appearance4"), "appearance16"),
        (("validation", "panel_records", 47), "48-by-32"),
        (("validation", "frames_per_record", 16), "48-by-32"),
    ],
)
def test_build_update_density_candidate_rejects_incomparable_parent(
    tmp_path: Path,
    mutation: tuple[object, ...],
    message: str,
):
    selection = _selection(tmp_path)
    config = deepcopy(selection.config)
    if len(mutation) == 2:
        config[mutation[0]] = mutation[1]
    else:
        config[mutation[0]][mutation[1]] = mutation[2]
    selection = PilotSelection(**{**selection.__dict__, "config": config})

    with pytest.raises(ValueError, match=message):
        build_update_density_pilot_config(
            selection,
            artifact_dir=tmp_path / "v28",
            effective_batch=96,
            pilot_epochs=3,
        )


def test_build_update_density_candidate_uses_safe_stage_four_microbatch_three(
    tmp_path: Path,
):
    selection = _selection(tmp_path)
    selection.config["microbatch_records"] = 6

    generated = build_update_density_pilot_config(
        selection,
        artifact_dir=tmp_path / "v28",
        effective_batch=96,
        pilot_epochs=3,
    )

    assert generated["microbatch_records"] == 3


def test_build_update_density_candidate_accepts_measured_unfrozen_microbatch_two(
    tmp_path: Path,
):
    generated = build_update_density_pilot_config(
        _selection(tmp_path),
        artifact_dir=tmp_path / "v37",
        effective_batch=96,
        pilot_epochs=3,
        physical_microbatch_records=2,
        balanced_families=True,
    )

    assert generated["microbatch_records"] == 2
    assert generated["macros_per_update"] == 8


@pytest.mark.parametrize("value", [0, -1, 13])
def test_build_update_density_candidate_rejects_invalid_physical_microbatch(
    tmp_path: Path, value: int
):
    with pytest.raises(ValueError, match="physical microbatch"):
        build_update_density_pilot_config(
            _selection(tmp_path),
            artifact_dir=tmp_path / "v37",
            effective_batch=96,
            pilot_epochs=3,
            physical_microbatch_records=value,
        )


def test_build_update_density_candidate_rejects_nonpositive_pilot_epochs(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="pilot epochs"):
        build_update_density_pilot_config(
            _selection(tmp_path),
            artifact_dir=tmp_path / "v28",
            effective_batch=96,
            pilot_epochs=0,
        )


def _candidate_row() -> dict[str, object]:
    return {
        "event": "epoch",
        "epoch": 3,
        "global_step": 72,
        "physical_microbatch_records": 3,
        "peak_cuda_bytes": 21 * 1024**3,
        "ddp": {"global_macros_per_update": 8, "world_size": 4},
        "gradient_norms": {"dense_decoder": 12.0, "medium_encoder": 0.5},
        "last_update_loss_components": {"delta": 0.80},
        "metrics": {
            "aggregate_relative_l2": 0.35,
            "relative_improvement_vs_coarse": 0.02,
            "family_relative_l2": {
                "uniform": 0.28,
                "layered": 0.38,
                "marmousi": 0.49,
            },
            "spectrum_relative_l2": {"high": 0.79},
        },
    }


def test_update_density_gate_accepts_complete_improving_candidate(tmp_path: Path):
    parent = _selection(tmp_path)

    report = update_density_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=_candidate_row(),
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert report["passes"]
    assert all(report["checks"].values())
    assert report["candidate"]["effective_batch"] == 96
    assert report["candidate"]["optimizer_updates"] == 72


def test_update_density_gate_separates_best_epoch_from_completed_updates(
    tmp_path: Path,
):
    parent = _selection(tmp_path)
    candidate = _candidate_row()
    candidate["epoch"] = 1
    candidate["global_step"] = 24

    report = update_density_candidate_gate(
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


def test_update_density_gate_names_the_measured_physical_microbatch(tmp_path: Path):
    parent = _selection(tmp_path)
    candidate = _candidate_row()
    candidate["physical_microbatch_records"] = 2

    report = update_density_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=candidate,
        expected_physical_microbatch=2,
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert report["checks"]["physical_microbatch_2"]
    assert "physical_microbatch_3" not in report["checks"]


def test_update_density_gate_requires_material_parent_and_coarse_improvement(
    tmp_path: Path,
):
    parent = _selection(tmp_path)
    candidate = _candidate_row()
    candidate["metrics"]["aggregate_relative_l2"] = 0.3998
    candidate["metrics"]["relative_improvement_vs_coarse"] = 5.0e-4

    report = update_density_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=candidate,
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
        minimum_relative_improvement=1.0e-3,
    )

    assert not report["passes"]
    assert not report["checks"]["parent_relative_improvement_1e3"]
    assert not report["checks"]["coarse_relative_improvement_1e3"]


def test_update_density_cli_counts_completed_run_progress(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    rows = tuple(
        {
            "event": "epoch",
            "validation_scope": "pilot_fixed_panel",
            "global_step": 24 * epoch,
        }
        for epoch in (1, 2, 3)
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _completed_optimizer_updates(path) == 72


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (("metrics", "aggregate_relative_l2", 0.41), "aggregate_improved"),
        (("metrics", "family_relative_l2", "marmousi", 0.54), "families_safe"),
        (("metrics", "spectrum_relative_l2", "high", 0.81), "high_band_safe"),
        (("last_update_loss_components", "delta", 0.91), "delta_improved"),
        (("metrics", "relative_improvement_vs_coarse", 0.0), "better_than_coarse"),
        (("ddp", "global_macros_per_update", 16), "macro_accumulation_8"),
        (("physical_microbatch_records", 2), "physical_microbatch_3"),
        (("global_step", 71), "optimizer_updates_72"),
        (("gradient_norms", "dense_decoder", float("nan")), "finite_gradients"),
        (("peak_cuda_bytes", 23 * 1024**3), "cuda_peak_safe"),
    ],
)
def test_update_density_gate_rejects_each_registered_failure(
    tmp_path: Path,
    mutation: tuple[object, ...],
    failed_check: str,
):
    candidate = _candidate_row()
    cursor = candidate
    for key in mutation[:-2]:
        cursor = cursor[key]
    cursor[mutation[-2]] = mutation[-1]
    parent = _selection(tmp_path)

    report = update_density_candidate_gate(
        parent_metrics=parent.metrics,
        parent_loss_components=parent.loss_components,
        candidate_row=candidate,
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert not report["passes"]
    assert not report["checks"][failed_check]


def test_update_density_gate_rejects_malformed_evidence(tmp_path: Path):
    parent = _selection(tmp_path)
    candidate = _candidate_row()
    candidate["gradient_norms"] = {}

    with pytest.raises(ValueError, match="gradient"):
        update_density_candidate_gate(
            parent_metrics=parent.metrics,
            parent_loss_components=parent.loss_components,
            candidate_row=candidate,
            family_tolerance=0.03,
            maximum_peak_cuda_gib=23.0,
        )


def _write_parent_evidence(
    tmp_path: Path,
    *,
    name: str,
    score: float,
) -> Path:
    selection = _selection(tmp_path / name)
    artifact = tmp_path / name / "artifact"
    pilot = artifact / "pilot"
    checkpoint = pilot / "checkpoints" / "epoch_0002.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config = deepcopy(selection.config)
    config["artifact_dir"] = str(artifact)
    config_path = tmp_path / name / f"{name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    identity = {
        "run_digest": f"digest-{name}",
        "config": config,
    }
    (pilot / "run_identity.json").write_text(json.dumps(identity))
    metrics = deepcopy(selection.metrics)
    metrics["aggregate_relative_l2"] = score
    row = {
        "event": "epoch",
        "epoch": 2,
        "validation_scope": "pilot_fixed_panel",
        "checkpoint": str(checkpoint),
        "last_update_loss_components": {"delta": 0.90},
        "metrics": {**metrics, "frame_count": 48 * 32},
    }
    (pilot / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    return config_path


def test_update_density_prepare_and_gate_clis_write_bound_evidence(tmp_path: Path):
    worse = _write_parent_evidence(tmp_path, name="worse", score=0.40)
    better = _write_parent_evidence(tmp_path, name="better", score=0.38)
    generated = tmp_path / "generated" / "v28.yaml"
    report = tmp_path / "v28" / "parent_selection.json"
    artifact = tmp_path / "v28" / "artifact"

    assert prepare_main(
        [
            "--candidate-config",
            str(worse),
            "--candidate-config",
            str(better),
            "--output-config",
            str(generated),
            "--artifact-dir",
            str(artifact),
            "--report",
            str(report),
            "--effective-batch",
            "96",
            "--pilot-epochs",
            "3",
        ]
    ) == 0

    generated_config = yaml.safe_load(generated.read_text())
    selection = json.loads(report.read_text())
    assert generated_config["macros_per_update"] == 8
    assert generated_config["microbatch_records"] == 3
    assert selection["schema"] == "saved_time_update_density_parent_selection_v1"
    assert selection["name"] == "better"
    assert selection["score"] == pytest.approx(0.38)
    assert selection["effective_batch"] == 96
    assert selection["expected_optimizer_updates"] == 72
    assert selection["generated_config"] == str(generated.resolve())

    candidate_checkpoint = artifact / "pilot" / "checkpoints" / "epoch_0003.pt"
    candidate_checkpoint.parent.mkdir(parents=True)
    candidate_checkpoint.write_bytes(b"candidate")
    candidate = _candidate_row()
    candidate.update(
        {
            "validation_scope": "pilot_fixed_panel",
            "checkpoint": str(candidate_checkpoint),
        }
    )
    candidate_metrics = artifact / "pilot" / "metrics.jsonl"
    candidate_metrics.write_text(json.dumps(candidate) + "\n")
    gate_path = artifact / "candidate_gate.json"

    assert gate_main(
        [
            "--selection-report",
            str(report),
            "--candidate-metrics",
            str(candidate_metrics),
            "--output",
            str(gate_path),
        ]
    ) == 0
    gate = json.loads(gate_path.read_text())
    assert gate["schema"] == "saved_time_update_density_evidence_gate_v1"
    assert gate["passes"] is True
    assert gate["candidate_epoch"] == 3
    assert gate["candidate_checkpoint"] == str(candidate_checkpoint)


def test_build_family_curriculum_candidate_is_independent_easy_to_hard(
    tmp_path: Path,
):
    selection = _selection(tmp_path)

    generated = build_update_density_pilot_config(
        selection,
        artifact_dir=tmp_path / "v29",
        effective_batch=96,
        pilot_epochs=3,
        family_curriculum=True,
    )

    assert generated["parent_checkpoint"] == str(selection.checkpoint)
    assert generated["macros_per_update"] == 8
    assert generated["microbatch_records"] == 3
    assert "schedule_epoch_offset" not in generated
    assert generated["time_appearance_offset"] == 11
    assert generated["family_curriculum"]["stages"] == [
        {"epochs": 1, "macro_pattern": ["uniform"]},
        {
            "epochs": 1,
            "macro_pattern": ["layered", "layered", "layered", "uniform"],
        },
        {
            "epochs": 1,
            "macro_pattern": [
                "marmousi",
                "marmousi",
                "marmousi",
                "uniform",
                "layered",
            ],
        },
    ]


def test_build_family_curriculum_candidate_requires_three_epochs(tmp_path: Path):
    with pytest.raises(ValueError, match="three epochs"):
        build_update_density_pilot_config(
            _selection(tmp_path),
            artifact_dir=tmp_path / "v29",
            effective_batch=96,
            pilot_epochs=2,
            family_curriculum=True,
        )


def test_build_balanced_family_candidate_uses_equal_repeating_macros(
    tmp_path: Path,
):
    generated = build_update_density_pilot_config(
        _selection(tmp_path),
        artifact_dir=tmp_path / "balanced",
        effective_batch=96,
        pilot_epochs=3,
        balanced_families=True,
        family_gradient_norms={
            "uniform": 30.0,
            "layered": 90.0,
            "marmousi": 10.0,
        },
    )

    assert "schedule_epoch_offset" not in generated
    assert generated["family_curriculum"] == {
        "stages": [
            {
                "epochs": 3,
                "macro_pattern": ["uniform", "layered", "marmousi"],
            }
        ]
    }
    assert generated["family_gradient_weights"] == pytest.approx(
        {"uniform": 0.6923076923, "layered": 0.2307692308, "marmousi": 2.0769230769}
    )


def test_balanced_and_easy_to_hard_family_schedules_are_mutually_exclusive(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_update_density_pilot_config(
            _selection(tmp_path),
            artifact_dir=tmp_path / "invalid",
            family_curriculum=True,
            balanced_families=True,
        )


def test_prepare_cli_records_family_curriculum_as_separate_intervention(
    tmp_path: Path,
):
    parent = _write_parent_evidence(tmp_path, name="parent", score=0.38)
    generated = tmp_path / "generated" / "v29.yaml"
    report = tmp_path / "v29" / "parent_selection.json"

    assert prepare_main(
        [
            "--candidate-config",
            str(parent),
            "--output-config",
            str(generated),
            "--artifact-dir",
            str(tmp_path / "v29" / "artifact"),
            "--report",
            str(report),
            "--effective-batch",
            "96",
            "--pilot-epochs",
            "3",
            "--family-curriculum",
        ]
    ) == 0

    config = yaml.safe_load(generated.read_text())
    evidence = json.loads(report.read_text())
    assert config["family_curriculum"]["stages"][0]["macro_pattern"] == [
        "uniform"
    ]
    assert evidence["intervention"] == "family_curriculum"


def test_prepare_cli_records_balanced_family_intervention(tmp_path: Path):
    parent = _write_parent_evidence(tmp_path, name="parent", score=0.38)
    generated = tmp_path / "generated" / "balanced.yaml"
    report = tmp_path / "balanced" / "parent_selection.json"
    gradient_report = tmp_path / "family_gradients.json"
    gradient_report.write_text(
        json.dumps(
            {
                "schema": "saved_time_family_gradient_conflict_v1",
                "gradient_report": {
                    "gradient_norm": {
                        "uniform": 30.0,
                        "layered": 90.0,
                        "marmousi": 10.0,
                    }
                },
            }
        )
    )

    assert prepare_main(
        [
            "--candidate-config",
            str(parent),
            "--output-config",
            str(generated),
            "--artifact-dir",
            str(tmp_path / "balanced" / "artifact"),
            "--report",
            str(report),
            "--balanced-families",
            "--family-gradient-report",
            str(gradient_report),
        ]
    ) == 0

    config = yaml.safe_load(generated.read_text())
    evidence = json.loads(report.read_text())
    assert config["family_curriculum"]["stages"][0]["macro_pattern"] == [
        "uniform",
        "layered",
        "marmousi",
    ]
    assert evidence["intervention"] == "balanced_families"
    assert evidence["family_gradient_report"] == str(gradient_report.resolve())
    assert evidence["family_gradient_weights"] == pytest.approx(
        config["family_gradient_weights"]
    )
