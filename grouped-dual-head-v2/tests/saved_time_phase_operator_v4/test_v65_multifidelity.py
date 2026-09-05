from pathlib import Path

import pytest
import yaml

from saved_time_phase_operator_v4.full_support import (
    audit_family_curriculum_epoch_schedule,
    build_family_curriculum_schedule,
)
from saved_time_phase_operator_v4.multifidelity import fixed_teacher_time_indices
from scripts.gate_saved_time_v65_multifidelity import (
    _parent_row_from_same_panel_report,
    v65_candidate_gate,
)
from scripts.train_saved_time_v4_full_support import (
    coverage_frames_per_appearance,
    microbatch_records_for_epoch,
    numerical_teacher_time_pool,
)


ROOT = Path(__file__).resolve().parents[2]
PILOT_CONFIG = (
    ROOT
    / "configs/saved_time_v4/generated/v65_lwc84_multifidelity_pilot_4gpu.yaml"
)
LONG_CONFIG = (
    ROOT
    / "configs/saved_time_v4/generated/v66_lwc84_multifidelity_long_4gpu.yaml"
)
PIPELINE = ROOT / "scripts/run_remote_v65_multifidelity.sh"
WATCHER = ROOT / "scripts/watch_remote_v65_pull.sh"


def test_v65_registers_solver_teacher_full_backbone_contract():
    config = yaml.safe_load(PILOT_CONFIG.read_text())

    assert config["epochs"] == config["gate"]["pilot_epochs"] == 4
    assert config["time_policy"] == "numerical_teacher_pool"
    assert config["training_frames_per_record"] == 16
    assert numerical_teacher_time_pool(config) == fixed_teacher_time_indices()
    assert coverage_frames_per_appearance(config) == 16
    teacher = config["numerical_teacher"]
    assert teacher["cache_h5"].endswith("/teacher_vds.h5")
    assert teacher["low_fidelity_weight"] == 0.1
    assert teacher["residual_weight"] == 0.05
    assert teacher["residual_energy_floor_fraction"] == 0.1
    assert config["macro_records"] * config["macros_per_update"] == 96
    assert config["microbatch_records"] == 3
    assert config["workers"] == 16
    assert config["optimizer"]["full_forward_checkpointing"] is True
    assert config["optimizer"]["training_loss_time_block"] == 16
    assert config["optimizer"]["dense_learning_rate"] == 1.0e-5
    assert config["optimizer"]["warmup_epochs"] == 2
    assert config["optimizer"]["gradient_clip_prefix_limits"]["dense_decoder"] == 5.0
    assert "band_limited_adapter" not in config
    assert microbatch_records_for_epoch(config, epoch=1) == 3
    assert microbatch_records_for_epoch(config, epoch=2) == 2
    assert microbatch_records_for_epoch(config, epoch=3) == 2


def test_v65_curriculum_reaches_all_2240_eligible_records():
    config = yaml.safe_load(PILOT_CONFIG.read_text())
    families = ("uniform",) * 420 + ("layered",) * 1120 + ("marmousi",) * 700
    schedule = build_family_curriculum_schedule(
        families,
        stages=config["family_curriculum"]["stages"],
        epochs=config["epochs"],
        macro_records=config["macro_records"],
        macros_per_update=config["macros_per_update"],
        seed=config["seed"],
        appearance_offset=config["time_appearance_offset"],
    )
    audits = tuple(
        audit_family_curriculum_epoch_schedule(
            schedule,
            epoch=epoch,
            record_families=families,
            macros_per_update=config["macros_per_update"],
        )
        for epoch in range(config["epochs"])
    )

    assert [row.record_count for row in audits] == [420, 1540, 2240, 2240]
    assert all(row.appearances == 2304 for row in audits)
    assert all(row.optimizer_updates == 24 for row in audits)


def test_v66_removes_solver_from_training_and_inference_path():
    config = yaml.safe_load(LONG_CONFIG.read_text())

    assert config["epochs"] == 40
    assert config["parent_checkpoint"].endswith("/pilot/best.pt")
    assert config["parent_checkpoint_identity"].endswith("/pilot/run_identity.json")
    assert "numerical_teacher" not in config
    assert config["time_policy"] == "appearance16"
    assert config["training_frames_per_record"] == 22
    assert config["optimizer"]["training_time_block"] == 4
    assert coverage_frames_per_appearance(config) == 22


def _epoch_row(
    aggregate,
    *,
    uniform,
    layered,
    marmousi,
    epoch=1,
    scope="pilot_fixed_panel",
    panel="fixed",
):
    return {
        "event": "epoch",
        "epoch": epoch,
        "global_step": 24 * epoch,
        "validation_scope": scope,
        "metrics": {
            "aggregate_relative_l2": aggregate,
            "family_relative_l2": {
                "uniform": uniform,
                "layered": layered,
                "marmousi": marmousi,
            },
            "source_relative_l2": {
                f"validation_{panel}_{index:05d}": aggregate
                for index in range(48 if scope == "pilot_fixed_panel" else 12)
            },
            "record_count": 48 if scope == "pilot_fixed_panel" else 12,
            "frame_count": (48 if scope == "pilot_fixed_panel" else 12) * 32,
        },
        "train_loss": 0.3,
        "physical_microbatch_records": 3,
        "peak_cuda_bytes": 22 * 1024**3,
        "gradient_norms": {"fusion": 1.0, "dense_decoder": 2.0},
        "ddp": {"world_size": 4, "global_macros_per_update": 8},
    }


def test_v65_gate_allows_long_exploration_only_after_safe_improvement():
    report = v65_candidate_gate(
        parent_row=_epoch_row(
            0.4691, uniform=0.3967, layered=0.4373, marmousi=0.5391
        ),
        candidate_row=_epoch_row(
            0.42, uniform=0.38, layered=0.41, marmousi=0.51, epoch=4
        ),
        smoke_row=_epoch_row(
            0.46,
            uniform=0.39,
            layered=0.43,
            marmousi=0.53,
            scope="smoke",
        ),
        smoke_terminal={"status": "complete", "global_step": 2},
        cache_summary={"status": "complete", "record_count": 2240, "time_count": 64},
    )

    assert report["passes"] is True
    assert all(report["checks"].values())
    assert report["accuracy_target"]["passes"] is False


def test_v65_gate_rejects_statistically_tiny_improvement_before_long_run():
    report = v65_candidate_gate(
        parent_row=_epoch_row(
            0.4892666, uniform=0.5025945, layered=0.4314280, marmousi=0.5588185
        ),
        candidate_row=_epoch_row(
            0.4879735,
            uniform=0.5012414,
            layered=0.4286467,
            marmousi=0.5594211,
            epoch=2,
        ),
        smoke_row=_epoch_row(
            0.48,
            uniform=0.50,
            layered=0.43,
            marmousi=0.55,
            scope="smoke",
        ),
        smoke_terminal={"status": "complete", "global_step": 2},
        cache_summary={"status": "complete", "record_count": 2240, "time_count": 64},
    )

    assert report["passes"] is False
    assert report["checks"]["minimum_relative_improvement"] is False
    assert report["candidate"]["relative_improvement"] == pytest.approx(
        0.0026429, abs=1.0e-7
    )


def test_v65_gate_rejects_cross_panel_parent_metrics():
    report = v65_candidate_gate(
        parent_row=_epoch_row(
            0.4691,
            uniform=0.3967,
            layered=0.4373,
            marmousi=0.5391,
            panel="seed372",
        ),
        candidate_row=_epoch_row(
            0.42,
            uniform=0.38,
            layered=0.41,
            marmousi=0.51,
            epoch=4,
            panel="seed401",
        ),
        smoke_row=_epoch_row(
            0.46,
            uniform=0.39,
            layered=0.43,
            marmousi=0.53,
            scope="smoke",
        ),
        smoke_terminal={"status": "complete", "global_step": 2},
        cache_summary={"status": "complete", "record_count": 2240, "time_count": 64},
    )

    assert report["passes"] is False
    assert report["checks"]["same_validation_panel"] is False


def test_v65_gate_accepts_identity_bound_same_panel_parent_report():
    metrics = _epoch_row(
        0.48, uniform=0.41, layered=0.44, marmousi=0.55
    )["metrics"]
    parent_report = {
        "schema": "saved_time_family_expert_same_panel_parent_v1",
        "status": "complete",
        "binding": {
            "validation_seed": 401,
            "validation_records": 48,
            "validation_frames_per_record": 32,
            "validation_indices": list(range(48)),
            "parent_checkpoint_sha256": "abc",
            "active_manifest_digest": "manifest",
        },
        "metrics": metrics,
    }

    row = _parent_row_from_same_panel_report(parent_report)

    assert row["epoch"] == 0
    assert row["validation_scope"] == "pilot_fixed_panel"
    assert row["metrics"] == metrics

    parent_report["binding"]["validation_records"] = 47
    with pytest.raises(ValueError, match="48 by 32"):
        _parent_row_from_same_panel_report(parent_report)


def test_remote_pipeline_binds_frozen_cpml_config_and_conditional_long_run():
    text = PIPELINE.read_text()

    assert "frozen_config.yaml" in text
    assert "build_lwc84_multifidelity_cache.py" in text
    assert "merge_lwc84_multifidelity_cache.py" in text
    assert "evaluate_saved_time_family_expert_parent.py" in text
    assert "parent_same_panel_seed401.json" in text
    assert "gate_saved_time_v65_multifidelity.py" in text
    assert "--parent-report" in text
    assert "v66_lwc84_multifidelity_long_4gpu.yaml" in text
    assert "setsid" not in text  # the outer local launcher owns detachment


def test_local_watcher_materializes_an_epoch_gate_summary_after_each_sync():
    text = WATCHER.read_text()

    assert "saved_time_phase_operator_v4.epoch_monitor" in text
    assert "pilot/metrics.jsonl" in text
    assert "parent_same_panel_seed401.json" in text
    assert "saved_time_v49_family_experts_structural_prior_pilot_r1/pilot/metrics.jsonl" not in text
    assert "monitor/epoch_summary.json" in text
