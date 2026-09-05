import hashlib
import json

import h5py
import pytest

from scripts.supervise_target10_instance_finetune import (
    _accepted_pretraining_parent,
    _cpadc_target_met,
    _external_evaluation_contract,
    _holdout_generation_command,
    _sealed_full_support_target_met,
    _traditional_runtime_reference,
    _validation_authorizes_test_id,
)


def test_full_support_target_requires_complete_480x401_protocol():
    report = {
        "status": "complete",
        "stored_times_only": True,
        "interpolated_targets": 0,
        "metrics": {
            "record_count": 480,
            "unique_time_index_count": 401,
            "aggregate_relative_l2": 0.049,
            "family_relative_l2": {
                "uniform": 0.04,
                "layered": 0.05,
                "marmousi": 0.049,
            },
        },
    }
    assert _sealed_full_support_target_met(report) is True
    assert (
        _sealed_full_support_target_met(report, expected_split="test_id")
        is False
    )
    report["metrics"]["record_count"] = 48
    assert _sealed_full_support_target_met(report) is False
    report["metrics"]["record_count"] = 480
    report["metrics"]["family_relative_l2"]["layered"] = 0.050001
    assert _sealed_full_support_target_met(report) is False


def test_cpadc_target_rechecks_absolute_metrics_not_just_boolean():
    terminal = {
        "status": "complete",
        "same_protocol_validation_passed": True,
        "promotion_gate": {
            "passed": True,
            "record_count": 480,
            "aggregate_adapted_future_relative_l2": 0.049,
            "mean_adapted_future_relative_l2": 0.049,
            "checks": {
                "full_validation_protocol": True,
                "future_truth_sealed": True,
                "cpu_coefficient_finetune": True,
                "absolute_mean_accuracy": True,
                "absolute_family_accuracy": True,
                "runtime_measurement_synchronized": True,
                "end_to_end_speedup_mean": True,
                "end_to_end_speedup_p95": True,
            },
            "mean_end_to_end_speedup_vs_traditional": 12.0,
            "p95_end_to_end_speedup_vs_traditional": 10.5,
            "families": {
                family: {
                    "mean_adapted_future_relative_l2": value,
                    "aggregate_adapted_future_relative_l2": value,
                }
                for family, value in {
                    "uniform": 0.048,
                    "layered": 0.049,
                    "marmousi": 0.05,
                }.items()
            },
        },
    }
    assert _cpadc_target_met(terminal) is True
    assert _validation_authorizes_test_id(terminal) is True
    assert _cpadc_target_met(terminal, expected_split="test_id") is False
    terminal["promotion_gate"]["families"]["marmousi"][
        "aggregate_adapted_future_relative_l2"
    ] = 0.050001
    assert _cpadc_target_met(terminal) is False
    assert _validation_authorizes_test_id(terminal) is False


def test_cpadc_target_rejects_accuracy_pass_without_ten_x_speedup():
    terminal = {
        "status": "complete",
        "same_protocol_validation_passed": True,
        "promotion_gate": {
            "passed": True,
            "record_count": 480,
            "aggregate_adapted_future_relative_l2": 0.049,
            "mean_adapted_future_relative_l2": 0.049,
            "mean_end_to_end_speedup_vs_traditional": 12.0,
            "p95_end_to_end_speedup_vs_traditional": 9.99,
            "checks": {
                "full_validation_protocol": True,
                "future_truth_sealed": True,
                "cpu_coefficient_finetune": True,
                "absolute_mean_accuracy": True,
                "absolute_family_accuracy": True,
                "runtime_measurement_synchronized": True,
                "end_to_end_speedup_mean": True,
                "end_to_end_speedup_p95": True,
            },
            "families": {
                family: {
                    "mean_adapted_future_relative_l2": 0.049,
                    "aggregate_adapted_future_relative_l2": 0.049,
                }
                for family in ("uniform", "layered", "marmousi")
            },
        },
    }
    assert _cpadc_target_met(terminal) is False


def test_traditional_runtime_reference_requires_comparable_protocol():
    report = {
        "status": "complete",
        "schema": "lwc84_traditional_runtime_reference_v1",
        "conservative_reference_runtime_s": 12.0,
        "protocol": {
            "solver_grid": [401, 401],
            "saved_grid": [201, 201],
            "saved_frames": 401,
            "propagation_time_s": 1.0,
            "single_instance": True,
            "same_gpu_as_deployment": True,
            "cuda_synchronized_timing": True,
            "includes_output_materialization": True,
            "excludes_disk_io": True,
        },
    }
    assert _traditional_runtime_reference(report) == 12.0
    report["protocol"]["single_instance"] = False
    with pytest.raises(ValueError, match="not deployment comparable"):
        _traditional_runtime_reference(report)


def test_plateau_uses_only_the_control_state_accepted_checkpoint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    identity = run / "run_identity.json"
    identity.write_text('{"run_digest":"digest"}')
    checkpoint = run / "checkpoints" / "epoch_0003.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"accepted")
    (run / "epoch_validation_control.json").write_text(
        '{"accepted_epoch":3,"checkpoint":"' + str(checkpoint) + '"}'
    )

    chosen, chosen_identity, report, reason = _accepted_pretraining_parent(
        tmp_path,
        {"status": "failed", "error": "epoch validation did not improve"},
    )

    assert chosen == checkpoint.resolve()
    assert chosen_identity == identity
    assert report is None
    assert reason == "plateau_accepted_epoch_3"


def test_new_holdout_generation_is_split_scoped_and_reuses_frozen_manifest(tmp_path):
    command = _holdout_generation_command(
        config=tmp_path / "holdout.yaml",
        output=tmp_path / "data",
        split="validation",
        confirm_token="RUN_FROZEN",
    )

    assert command[command.index("--splits") + 1] == "validation"
    assert command[command.index("--devices") + 1] == "0,1,2,3"
    assert command[command.index("--batch-size") + 1] == "128"
    assert "--frozen-manifest" in command


def test_external_contract_binds_training_and_evaluation_manifests(tmp_path):
    audit = tmp_path / "audit.json"
    pretruth = tmp_path / "pretruth.json"
    internal = tmp_path / "internal.json"
    dataset = tmp_path / "validation.h5"
    travel = tmp_path / "travel.h5"
    for path in (audit, pretruth, internal):
        path.write_text(json.dumps({"status": "passed"}))
    shard = tmp_path / "validation-00000.h5"
    shard.write_bytes(b"frozen-shard")
    sidecar = tmp_path / "validation-00000.h5.sha256"
    sidecar.write_text(hashlib.sha256(shard.read_bytes()).hexdigest() + "\n")
    with h5py.File(dataset, "w") as handle:
        handle.attrs["vds_source_shards"] = json.dumps([str(shard.resolve())])
    with h5py.File(travel, "w") as handle:
        handle.attrs["source_h5"] = str(dataset.resolve())
        handle.attrs["content_sha256"] = "d" * 64
        handle.attrs["schema"] = "source_family_adaptive_travel_v1"
    contract = _external_evaluation_contract(
        role="independent_test_id",
        training_manifest_digest="train",
        evaluation_manifest_digest="test",
        sample_hash_audit=audit,
        pretruth_overlap_audit=pretruth,
        internal_split_overlap_audit=internal,
        evaluation_dataset=dataset,
        travel_time_h5=travel,
    )

    assert contract["evaluation_only"] is True
    assert contract["training_forbidden"] is True
    assert contract["training_manifest_digest"] == "train"
    assert contract["evaluation_manifest_digest"] == "test"
    assert contract["pretruth_historical_overlap_audit"].endswith("pretruth.json")
    assert contract["internal_split_overlap_audit"].endswith("internal.json")
    assert contract["postgeneration_sample_sha256_audit_sha256"] == hashlib.sha256(
        audit.read_bytes()
    ).hexdigest()
    assert contract["travel_time_h5_sha256"] == hashlib.sha256(
        travel.read_bytes()
    ).hexdigest()
    assert contract["travel_time_content_sha256"] == "d" * 64
