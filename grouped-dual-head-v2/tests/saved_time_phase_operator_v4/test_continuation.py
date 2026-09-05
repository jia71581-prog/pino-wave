import json
import importlib
from pathlib import Path

import pytest
import yaml

import saved_time_phase_operator_v4.continuation as continuation
from saved_time_phase_operator_v4.continuation import (
    PilotSelection,
    build_all_modes_pilot_config,
    build_family_expert_pilot_config,
    build_long_continuation_config,
    progressive_family_curriculum_stages,
    select_best_pilot_candidate,
)


def _candidate(tmp_path: Path, name: str, scores: tuple[float, ...], *, offset: int = 3):
    artifact = tmp_path / name
    pilot = artifact / "pilot"
    checkpoints = pilot / "checkpoints"
    checkpoints.mkdir(parents=True)
    identity = {
        "run_digest": f"digest-{name}",
        "manifest_digest": "manifest",
    }
    (pilot / "run_identity.json").write_text(json.dumps(identity))
    rows = []
    for epoch, score in enumerate(scores, start=1):
        checkpoint = checkpoints / f"epoch_{epoch:04d}.pt"
        checkpoint.write_bytes(b"checkpoint")
        rows.append(
            {
                "event": "epoch",
                "epoch": epoch,
                "validation_scope": "pilot_fixed_panel",
                "checkpoint": str(checkpoint),
                "metrics": {
                    "aggregate_relative_l2": score,
                    "frame_count": 48 * 32,
                    "family_relative_l2": {
                        "uniform": score - 0.01,
                        "layered": score,
                        "marmousi": score + 0.01,
                    },
                },
            }
        )
    (pilot / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    config = {
        "base_config": "base.yaml",
        "parent_checkpoint": "old.pt",
        "parent_checkpoint_identity": "old_identity.json",
        "parent_identity": "v3_identity.json",
        "artifact_dir": str(artifact),
        "seed": 331,
        "epochs": 40,
        "macro_records": 12,
        "macros_per_update": 8,
        "microbatch_records": 5,
        "schedule_epoch_offset": offset,
        "residual_recovery": {
            "stage_epoch_offset": offset,
            "activation_mode": "reset",
        },
        "optimizer": {"gradient_clip_mode": "global", "warmup_epochs": 2},
        "validation": {"all_records_every": 5},
        "gate": {"pilot_epochs": len(scores)},
    }
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_select_best_pilot_and_build_coverage_continuous_long_config(tmp_path):
    first = _candidate(tmp_path, "first", (0.48, 0.44, 0.45))
    second = _candidate(tmp_path, "second", (0.46, 0.39, 0.41))

    selection = select_best_pilot_candidate((first, second))

    assert selection.name == "second"
    assert selection.epoch == 2
    assert selection.score == pytest.approx(0.39)
    assert selection.family_relative_l2["marmousi"] == pytest.approx(0.40)
    assert selection.checkpoint.name == "epoch_0002.pt"

    output_artifact = tmp_path / "long"
    config = build_long_continuation_config(
        selection,
        artifact_dir=output_artifact,
        epochs=40,
    )

    assert config["parent_checkpoint"] == str(selection.checkpoint)
    assert config["parent_checkpoint_identity"] == str(selection.checkpoint_identity)
    assert config["artifact_dir"] == str(output_artifact)
    assert config["residual_recovery"]["stage_epoch_offset"] == 5
    assert config["schedule_epoch_offset"] == 5
    assert config["optimizer"]["gradient_clip_mode"] == "prefix"
    assert config["optimizer"]["schedule_epoch_offset"] == 2
    assert config["optimizer"]["schedule_total_epochs"] == 40
    assert config["checkpoint_transfer"]["parent_optimizer_state"] is True
    assert config["residual_recovery"]["activation_mode"] == "preserve"
    assert config["loss"]["hard_causality"] is True
    assert config["loss"]["hard_causality_lead_cycles"] == pytest.approx(1.0)
    assert config["epochs"] == 40


def test_long_continuation_can_select_a_safe_unfrozen_physical_microbatch(tmp_path):
    source = _candidate(tmp_path, "unfrozen", (0.42, 0.39), offset=3)
    selection = select_best_pilot_candidate((source,))

    config = build_long_continuation_config(
        selection,
        artifact_dir=tmp_path / "long-safe",
        epochs=40,
        physical_microbatch_records=2,
    )

    assert config["microbatch_records"] == 2
    assert config["macros_per_update"] == 8


@pytest.mark.parametrize("value", [0, -1, 13])
def test_long_continuation_rejects_invalid_physical_microbatch(tmp_path, value):
    source = _candidate(tmp_path, "invalid-microbatch", (0.42, 0.39), offset=3)
    selection = select_best_pilot_candidate((source,))

    with pytest.raises(ValueError, match="physical microbatch"):
        build_long_continuation_config(
            selection,
            artifact_dir=tmp_path / "long-invalid",
            epochs=40,
            physical_microbatch_records=value,
        )


def test_build_all_modes_pilot_forks_exact_best_epoch_without_optimizer_state(tmp_path):
    source = _candidate(tmp_path, "residual_wakeup", (0.46, 0.41, 0.43), offset=6)
    selection = select_best_pilot_candidate((source,))

    config = build_all_modes_pilot_config(
        selection,
        artifact_dir=tmp_path / "all-modes",
    )

    assert config["parent_checkpoint"] == str(selection.checkpoint)
    assert config["parent_checkpoint_identity"] == str(selection.checkpoint_identity)
    assert config["variant_overrides"]["modes"] == 101
    assert config["checkpoint_transfer"]["parent_residual_already_active"] is True
    assert config["checkpoint_transfer"]["parent_optimizer_state"] is False
    assert config["checkpoint_transfer"]["allow_spectral_mode_expansion"] is True
    assert config["residual_recovery"]["activation_mode"] == "preserve"
    assert config["residual_recovery"]["stage_epoch_offset"] == 0
    assert config["residual_recovery"]["decoder_only_epochs"] == 2
    assert config["schedule_epoch_offset"] == 8
    assert config["optimizer"]["schedule_epoch_offset"] == 2
    assert config["optimizer"]["dense_learning_rate"] == pytest.approx(1.0e-4)
    assert config["gate"]["pilot_epochs"] == 2


def test_long_continuation_consumes_coupled_2d_expansion_permission(tmp_path):
    source = _candidate(tmp_path, "coupled_2d", (0.40, 0.35), offset=11)
    selection = select_best_pilot_candidate((source,))
    selection.config["variant_overrides"] = {
        "modes": 101,
        "coupled_2d_rank": 16,
    }
    selection.config["checkpoint_transfer"] = {
        "allow_new_coupled_2d_parameters": True,
        "parent_optimizer_state": False,
    }

    config = build_long_continuation_config(
        selection,
        artifact_dir=tmp_path / "coupled-long",
        epochs=40,
    )

    assert config["variant_overrides"]["coupled_2d_rank"] == 16
    assert config["checkpoint_transfer"]["parent_optimizer_state"] is True
    assert "allow_new_coupled_2d_parameters" not in config["checkpoint_transfer"]


def test_long_continuation_consumes_temporal_basis_expansion_permission(tmp_path):
    source = _candidate(tmp_path, "temporal_basis", (0.40, 0.34), offset=11)
    selection = select_best_pilot_candidate((source,))
    selection.config["variant_overrides"] = {
        "modes": 101,
        "temporal_basis_rank": 96,
    }
    selection.config["checkpoint_transfer"] = {
        "allow_new_temporal_basis_parameters": True,
        "parent_optimizer_state": False,
    }

    config = build_long_continuation_config(
        selection,
        artifact_dir=tmp_path / "temporal-long",
        epochs=40,
    )

    assert config["variant_overrides"]["temporal_basis_rank"] == 96
    assert config["checkpoint_transfer"]["parent_optimizer_state"] is True
    assert "allow_new_temporal_basis_parameters" not in config["checkpoint_transfer"]


def test_build_family_expert_candidate_is_balanced_and_identity_bound(tmp_path):
    source = _candidate(tmp_path, "expert-parent", (0.46, 0.42, 0.43), offset=8)
    selection = select_best_pilot_candidate((source,))
    selection.config.update(
        {
            "time_policy": "appearance16",
            "validation": {
                "panel_records": 48,
                "frames_per_record": 32,
                "all_records_every": 5,
            },
            "variant_overrides": {
                "modes": 101,
                "temporal_basis_rank": 96,
            },
            "optimizer": {
                "dense_learning_rate": 1.0e-5,
                "geometry_learning_rate": 2.0e-6,
                "backbone_learning_rate": 1.0e-6,
                "schedule_epoch_offset": 8,
            },
        }
    )

    config = build_family_expert_pilot_config(
        selection,
        artifact_dir=tmp_path / "family-experts",
        physical_microbatch_records=3,
    )

    assert config["parent_checkpoint"] == str(selection.checkpoint)
    assert config["parent_checkpoint_identity"] == str(selection.checkpoint_identity)
    assert config["variant_overrides"]["family_expert_rank"] == 16
    assert config["checkpoint_transfer"]["allow_new_family_expert_parameters"] is True
    assert config["checkpoint_transfer"]["parent_optimizer_state"] is False
    assert config["macros_per_update"] == 16
    assert config["microbatch_records"] == 3
    assert config["family_curriculum"] == {
        "stages": [
            {
                "epochs": 1,
                "macro_pattern": ["uniform"],
            },
            {
                "epochs": 1,
                "macro_pattern": ["layered", "layered", "layered", "uniform"],
            },
            {
                "epochs": 1,
                "macro_pattern": [
                    "marmousi", "marmousi", "marmousi", "marmousi",
                    "layered", "uniform",
                ],
            },
        ]
    }
    assert config["family_experts"] == {
        "router_loss_weight": pytest.approx(0.01),
        "head_only_epochs": 1,
        "minimum_router_accuracy": pytest.approx(0.95),
        "minimum_route_probability": pytest.approx(0.10),
        "teacher_forced_routing": True,
        "dense_unfreeze_epoch": 3,
        "shared_unfreeze_epoch": 8,
        "geometry_unfreeze_epoch": 13,
        "backbone_unfreeze_epoch": 18,
    }
    assert config["optimizer"]["family_expert_learning_rate"] == pytest.approx(1.0e-4)
    assert config["gate"]["pilot_epochs"] == 3


def test_progressive_family_curriculum_scales_to_long_training():
    assert tuple(stage["epochs"] for stage in progressive_family_curriculum_stages(3)) == (
        1,
        1,
        1,
    )
    stages = progressive_family_curriculum_stages(40)
    assert tuple(stage["epochs"] for stage in stages) == (5, 10, 25)
    assert stages[0]["macro_pattern"] == ["uniform"]
    assert stages[-1]["macro_pattern"].count("marmousi") == 4


def test_family_expert_long_continuation_preserves_and_expands_curriculum(tmp_path):
    source = _candidate(tmp_path, "expert-long-parent", (0.46, 0.42, 0.40), offset=8)
    selected = select_best_pilot_candidate((source,))
    selected.config.update(
        {
            "family_curriculum": {
                "stages": progressive_family_curriculum_stages(3)
            },
            "family_experts": {
                "head_only_epochs": 1,
                "dense_unfreeze_epoch": 3,
                "shared_unfreeze_epoch": 8,
                "geometry_unfreeze_epoch": 13,
                "backbone_unfreeze_epoch": 18,
            },
        }
    )
    selected.config["optimizer"].update(
        {
            "gradient_clip_mode": "prefix_limits",
            "gradient_clip_prefix_limits": {
                "dense_decoder": 20.0,
                "source_encoder": 5.0,
                "default": 1.0,
            },
        }
    )

    config = build_long_continuation_config(
        selected,
        artifact_dir=tmp_path / "expert-long",
        epochs=40,
        physical_microbatch_records=4,
    )

    assert tuple(
        stage["epochs"] for stage in config["family_curriculum"]["stages"]
    ) == (5, 10, 25)
    assert config["family_experts"]["stage_epoch_offset"] == selected.epoch
    assert config["family_experts"]["shared_unfreeze_epoch"] == 8
    assert config["family_experts"]["geometry_unfreeze_epoch"] == 13
    assert config["family_experts"]["backbone_unfreeze_epoch"] == 18
    assert config["optimizer"]["gradient_clip_mode"] == "prefix_limits"
    assert config["optimizer"]["gradient_clip_prefix_limits"]["dense_decoder"] == 20.0
    assert config["microbatch_records"] == 4


def test_family_expert_long_continuation_materializes_missing_unfreeze_schedule(
    tmp_path,
):
    source = _candidate(tmp_path, "expert-partial-parent", (0.46, 0.42, 0.40))
    selected = select_best_pilot_candidate((source,))
    selected.config.update(
        {
            "family_curriculum": {
                "stages": progressive_family_curriculum_stages(3)
            },
            "family_experts": {
                "router_loss_weight": 0.01,
                "head_only_epochs": 1,
                "minimum_router_accuracy": 0.95,
                "minimum_route_probability": 0.10,
                "teacher_forced_routing": True,
            },
        }
    )

    config = build_long_continuation_config(
        selected,
        artifact_dir=tmp_path / "expert-partial-long",
        epochs=40,
        physical_microbatch_records=12,
    )

    assert config["family_experts"]["dense_unfreeze_epoch"] == 3
    assert config["family_experts"]["shared_unfreeze_epoch"] == 8
    assert config["family_experts"]["geometry_unfreeze_epoch"] == 13
    assert config["family_experts"]["backbone_unfreeze_epoch"] == 18
    assert config["family_experts"]["stage_epoch_offset"] == selected.epoch


def test_family_expert_candidate_rejects_parent_that_already_has_experts(tmp_path):
    source = _candidate(tmp_path, "existing-experts", (0.44, 0.41), offset=4)
    selection = select_best_pilot_candidate((source,))
    selection.config["time_policy"] = "appearance16"
    selection.config["validation"] = {"panel_records": 48, "frames_per_record": 32}
    selection.config["variant_overrides"] = {"family_expert_rank": 16}

    with pytest.raises(ValueError, match="must not already contain"):
        build_family_expert_pilot_config(
            selection, artifact_dir=tmp_path / "invalid-experts"
        )


def test_prepare_family_expert_cli_writes_provenance_bound_report(tmp_path):
    source = _candidate(tmp_path, "expert-cli-parent", (0.46, 0.40, 0.42), offset=5)
    parent_config = yaml.safe_load(source.read_text())
    parent_config.update(
        {
            "time_policy": "appearance16",
            "validation": {"panel_records": 48, "frames_per_record": 32},
            "variant_overrides": {"modes": 101, "temporal_basis_rank": 96},
            "optimizer": {
                "dense_learning_rate": 1.0e-5,
                "schedule_epoch_offset": 5,
            },
        }
    )
    source.write_text(yaml.safe_dump(parent_config))
    output = tmp_path / "generated" / "family-expert.yaml"
    report = tmp_path / "expert-artifact" / "parent_selection.json"
    command = importlib.import_module(
        "scripts.prepare_saved_time_family_expert_candidate"
    )

    assert command.main(
        [
            "--candidate-config",
            str(source),
            "--output-config",
            str(output),
            "--artifact-dir",
            str(tmp_path / "expert-artifact"),
            "--report",
            str(report),
            "--physical-microbatch-records",
            "3",
        ]
    ) == 0

    generated = yaml.safe_load(output.read_text())
    evidence = json.loads(report.read_text())
    assert generated["variant_overrides"]["family_expert_rank"] == 16
    assert evidence["schema"] == "saved_time_family_expert_parent_selection_v1"
    assert evidence["checkpoint"].endswith("epoch_0002.pt")
    assert evidence["checkpoint_identity"].endswith("pilot/run_identity.json")
    assert evidence["family_expert_rank"] == 16
    assert evidence["effective_batch"] == 192
    assert evidence["physical_microbatch_records"] == 3
    assert evidence["parent_metrics"]["aggregate_relative_l2"] == pytest.approx(0.40)


def test_prepare_all_modes_cli_writes_identity_bound_config_and_report(tmp_path):
    source = _candidate(tmp_path, "wakeup", (0.45, 0.40), offset=6)
    output = tmp_path / "generated" / "v20.yaml"
    report = tmp_path / "generated" / "v20_selection.json"
    artifact = tmp_path / "v20-artifact"
    command = importlib.import_module(
        "scripts.prepare_saved_time_all_modes_candidate"
    )

    result = command.main(
        [
            "--candidate-config",
            str(source),
            "--output-config",
            str(output),
            "--artifact-dir",
            str(artifact),
            "--report",
            str(report),
        ]
    )

    assert result == 0
    generated = yaml.safe_load(output.read_text())
    evidence = json.loads(report.read_text())
    assert generated["variant_overrides"]["modes"] == 101
    assert evidence["epoch"] == 2
    assert evidence["checkpoint"].endswith("epoch_0002.pt")
    assert evidence["generated_config"] == str(output.resolve())
    assert evidence["parent_metrics"]["aggregate_relative_l2"] == pytest.approx(0.40)
    assert evidence["parent_loss_components"] == {}


def test_gate_all_modes_cli_requires_parent_relative_accuracy_and_high_band(tmp_path):
    generated = tmp_path / "v20.yaml"
    generated.write_text(
        yaml.safe_dump(
            {
                "gate": {
                    "family_regression_tolerance": 0.03,
                    "maximum_peak_cuda_gib": 23.0,
                }
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "generated_config": str(generated),
                "parent_metrics": {
                    "aggregate_relative_l2": 0.40,
                    "family_relative_l2": {
                        "uniform": 0.30,
                        "layered": 0.40,
                        "marmousi": 0.50,
                    },
                    "spectrum_relative_l2": {"high": 0.80},
                },
                "parent_loss_components": {"delta": 0.95},
            }
        )
    )
    candidate = tmp_path / "metrics.jsonl"
    row = {
        "event": "epoch",
        "epoch": 2,
        "validation_scope": "pilot_fixed_panel",
        "checkpoint": str(tmp_path / "epoch_0002.pt"),
        "peak_cuda_bytes": 20 * 1024**3,
        "last_update_loss_components": {"delta": 0.85},
        "metrics": {
            "aggregate_relative_l2": 0.35,
            "relative_improvement_vs_coarse": 0.02,
            "family_relative_l2": {
                "uniform": 0.28,
                "layered": 0.38,
                "marmousi": 0.48,
            },
            "spectrum_relative_l2": {"high": 0.70},
        },
    }
    candidate.write_text(json.dumps(row) + "\n")
    output = tmp_path / "gate.json"
    command = importlib.import_module("scripts.gate_saved_time_all_modes_candidate")

    result = command.main(
        [
            "--selection-report",
            str(selection),
            "--candidate-metrics",
            str(candidate),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text())
    assert report["passes"]
    assert all(report["checks"].values())

    row["metrics"]["spectrum_relative_l2"]["high"] = 0.81
    candidate.write_text(json.dumps(row) + "\n")
    assert command.main(
        [
            "--selection-report",
            str(selection),
            "--candidate-metrics",
            str(candidate),
            "--output",
            str(output),
        ]
    ) == 2
    assert not json.loads(output.read_text())["checks"]["high_band_improved"]


def test_remote_all_modes_pipeline_gates_before_starting_long_training():
    pipeline = Path(
        "/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/"
        "run_remote_v20_all_modes_pipeline.sh"
    ).read_text()
    gate = "gate_saved_time_all_modes_candidate.py"
    prepare = "prepare_saved_time_long_continuation.py"
    long_log = "long_screen.log"

    assert pipeline.index(gate) < pipeline.index(prepare) < pipeline.index(long_log)
    fallback_prepare = "prepare_saved_time_local_differential_candidate.py"
    fallback_gate = "gate_saved_time_local_differential_candidate.py"

    assert pipeline.index(gate) < pipeline.index(fallback_prepare)
    assert pipeline.index(fallback_prepare) < pipeline.index(fallback_gate)
    assert "if [[ $gate_rc -eq 0 ]]" in pipeline
    assert "saved_time_v21_all_modes_long_bigbatch192_4gpu_remote_r1" in pipeline
    assert "saved_time_v23_local_differential_long_bigbatch192_4gpu_remote_r1" in pipeline


def test_candidate_selection_rejects_missing_or_noncomparable_evidence(tmp_path):
    missing = tmp_path / "missing.yaml"
    missing.write_text(yaml.safe_dump({"artifact_dir": str(tmp_path / "none")}))

    with pytest.raises(ValueError, match="no comparable"):
        select_best_pilot_candidate((missing,))

    malformed = _candidate(tmp_path, "malformed", (0.4,))
    metrics = Path(yaml.safe_load(malformed.read_text())["artifact_dir"]) / "pilot" / "metrics.jsonl"
    row = json.loads(metrics.read_text())
    row["metrics"]["frame_count"] = 12
    metrics.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="no comparable"):
        select_best_pilot_candidate((malformed,))


def test_candidate_selection_can_keep_a_better_registered_run_baseline(tmp_path):
    pilot = _candidate(tmp_path, "pilot_candidate", (0.44, 0.40))
    baseline = _candidate(tmp_path, "run_baseline", (0.39, 0.35))
    artifact = Path(yaml.safe_load(baseline.read_text())["artifact_dir"])
    (artifact / "pilot").rename(artifact / "run")
    metrics = artifact / "run" / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    for row in rows:
        row["validation_scope"] = "fixed_panel"
        row["checkpoint"] = row["checkpoint"].replace("/pilot/", "/run/")
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selection = select_best_pilot_candidate((pilot, baseline))

    assert selection.name == "run_baseline"
    assert selection.epoch == 2
    assert selection.score == pytest.approx(0.35)
    assert selection.checkpoint_identity.parent.name == "run"


def _add_coarse_metrics(config_path: Path, scores: tuple[float, ...]) -> None:
    artifact = Path(yaml.safe_load(config_path.read_text())["artifact_dir"])
    metrics_path = artifact / "pilot" / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(rows) == len(scores)
    for row, score in zip(rows, scores, strict=True):
        row["last_update_loss_components"] = {"delta": 0.9}
        row["metrics"]["spectrum_relative_l2"] = {
            "high": float(row["metrics"]["aggregate_relative_l2"]) + 0.10
        }
        row["metrics"]["coarse_metrics"] = {
            "aggregate_relative_l2": score,
            "frame_count": 48 * 32,
            "family_relative_l2": {
                "uniform": score - 0.01,
                "layered": score,
                "marmousi": score + 0.01,
            },
            "spectrum_relative_l2": {"high": score + 0.10},
        }
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_select_best_pilot_head_compares_coarse_and_corrected_outputs(tmp_path):
    hybrid = _candidate(tmp_path, "hybrid", (0.451969,))
    _add_coarse_metrics(hybrid, (0.451386,))
    corrected = _candidate(tmp_path, "corrected", (0.440000,))
    _add_coarse_metrics(corrected, (0.450000,))

    hybrid_selection = continuation.select_best_pilot_head((hybrid,))
    corrected_selection = continuation.select_best_pilot_head((hybrid, corrected))

    assert hybrid_selection.correction_scale == pytest.approx(0.0)
    assert hybrid_selection.score == pytest.approx(0.451386)
    assert hybrid_selection.metrics["aggregate_relative_l2"] == pytest.approx(
        0.451386
    )
    assert hybrid_selection.checkpoint.name == "epoch_0001.pt"
    assert hybrid_selection.checkpoint_identity.parent.name == "pilot"
    assert hybrid_selection.loss_components["delta"] == pytest.approx(0.9)
    assert corrected_selection.name == "corrected"
    assert corrected_selection.correction_scale == pytest.approx(1.0)
    assert corrected_selection.score == pytest.approx(0.44)


def test_third_continuation_reaches_inherited_epoch_120(tmp_path):
    checkpoint = tmp_path / "epoch_0040.pt"
    identity = tmp_path / "run_identity.json"
    checkpoint.write_bytes(b"checkpoint")
    identity.write_text(json.dumps({"run_digest": "digest-long2"}))
    selection = PilotSelection(
        name="long_winner_v2",
        config_path=tmp_path / "long_winner_v2.yaml",
        config={
            "epochs": 40,
            "schedule_epoch_offset": 80,
            "residual_recovery": {"stage_epoch_offset": 80},
            "optimizer": {
                "schedule_epoch_offset": 80,
                "schedule_total_epochs": 40,
            },
        },
        checkpoint=checkpoint,
        checkpoint_identity=identity,
        epoch=40,
        score=0.20,
        family_relative_l2={
            "uniform": 0.10,
            "layered": 0.20,
            "marmousi": 0.30,
        },
        metrics={},
        loss_components={},
    )

    config = build_long_continuation_config(
        selection,
        artifact_dir=tmp_path / "long3",
        epochs=40,
    )

    assert config["schedule_epoch_offset"] == 120
    assert config["residual_recovery"]["stage_epoch_offset"] == 120
    assert config["optimizer"]["schedule_epoch_offset"] == 120
    assert config["epochs"] == 40


def _full_time_parent_candidate(
    tmp_path: Path,
    name: str,
    *,
    corrected_score: float,
    coarse_score: float,
):
    artifact = tmp_path / name
    run = artifact / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "epoch_0040.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run / "run_identity.json").write_text(
        json.dumps({"run_digest": f"digest-{name}"})
    )

    def metrics(score):
        return {
            "aggregate_relative_l2": score,
            "frame_count": 48 * 401,
            "family_relative_l2": {
                "uniform": score - 0.01,
                "layered": score,
                "marmousi": score + 0.01,
            },
        }

    row = {
        "event": "epoch",
        "epoch": 40,
        "validation_scope": "fixed_full_time_panel",
        "checkpoint": str(checkpoint),
        "metrics": {
            **metrics(corrected_score),
            "coarse_metrics": metrics(coarse_score),
        },
    }
    (run / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump({"artifact_dir": str(artifact)}))
    return config_path


def test_select_best_full_time_parent_binds_config_identity_checkpoint_and_scale(
    tmp_path,
):
    long2 = _full_time_parent_candidate(
        tmp_path,
        "long2",
        corrected_score=0.09,
        coarse_score=0.11,
    )
    long3 = _full_time_parent_candidate(
        tmp_path,
        "long3",
        corrected_score=0.12,
        coarse_score=0.08,
    )

    selection = continuation.select_best_full_time_parent((long2, long3))

    assert selection.name == "long3"
    assert selection.config_path == long3.resolve()
    assert selection.correction_scale == pytest.approx(0.0)
    assert selection.score == pytest.approx(0.08)
    assert selection.family_relative_l2["marmousi"] == pytest.approx(0.09)
    assert selection.checkpoint.name == "epoch_0040.pt"
    assert selection.checkpoint_identity.parent.name == "run"


def test_instance_parent_cli_writes_one_bound_selection_record(tmp_path):
    long2 = _full_time_parent_candidate(
        tmp_path,
        "long2",
        corrected_score=0.10,
        coarse_score=0.11,
    )
    long3 = _full_time_parent_candidate(
        tmp_path,
        "long3",
        corrected_score=0.07,
        coarse_score=0.09,
    )
    output = tmp_path / "selection.json"
    cli = importlib.import_module("scripts.select_saved_time_instance_parent")

    result = cli.main(
        [
            "--candidate-config",
            str(long2),
            "--candidate-config",
            str(long3),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert result == 0
    assert report["name"] == "long3"
    assert report["config"] == str(long3.resolve())
    assert report["correction_scale"] == pytest.approx(1.0)
    assert report["score"] == pytest.approx(0.07)
    assert Path(report["checkpoint_identity"]).parent.name == "run"
