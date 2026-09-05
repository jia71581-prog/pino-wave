import csv
import json
from pathlib import Path

import pytest

from scripts.promote_latest_model_evidence_bundle import (
    record_nonpromotion_evidence,
    stage_bundle,
    validate_promotion_inputs,
)


def _payloads(*, gate=True, count=240, hashes=("abc", "abc", "abc")):
    comparison = {
        "schema": "latest_vs_phase4b_fixed15_same_sample_comparison_v1",
        "status": "complete",
        "checkpoint_sha256": hashes[0],
        "replacement_decision": {"replacement_gate_passed": gate},
    }
    score = {
        "schema": "marmousi_fixed_frequency_source_position_scores_v1",
        "status": "complete",
        "source_generalization_variable": "position_only",
        "frequency_generalization_claim_permitted": False,
        "fixed_source_frequency_hz": 19.0,
        "checkpoint_sha256": hashes[1],
        "records": [{"record_id": f"r{i}"} for i in range(count)],
    }
    selection = {
        "checkpoint_sha256": hashes[2],
        "selection_source": "r5b_preregistered_fallback",
    }
    return comparison, score, selection


def test_complete_same_checkpoint_gate_permits_promotion():
    result = validate_promotion_inputs(*_payloads())
    assert result["promotion_permitted"] is True
    assert result["position_record_count"] == 240


def test_failed_accuracy_gate_does_not_permit_promotion():
    result = validate_promotion_inputs(*_payloads(gate=False))
    assert result["promotion_permitted"] is False


def test_diagnostic_r5d_is_testable_but_not_promotion_eligible():
    comparison, score, selection = _payloads()
    selection["selection_source"] = "r5d_train_gate_best"

    result = validate_promotion_inputs(comparison, score, selection)

    assert result["promotion_permitted"] is False
    assert result["promotion_block_reason"] == "selected_checkpoint_is_diagnostic_only"


def test_checkpoint_mismatch_rejects_mixed_model_bundle():
    with pytest.raises(ValueError, match="different checkpoints"):
        validate_promotion_inputs(*_payloads(hashes=("a", "b", "a")))


def test_incomplete_position_census_is_rejected():
    with pytest.raises(ValueError, match="240-record census"):
        validate_promotion_inputs(*_payloads(count=239))


def test_stage_bundle_replaces_every_model_dependent_directory(tmp_path: Path):
    current = tmp_path / "bundle"
    archive = tmp_path / "bundle_archive"
    comparison = tmp_path / "comparison"
    position = tmp_path / "position"
    staging = tmp_path / "staging"
    for name in (
        "01_complete_wavefields",
        "02_wavefield_snapshots",
        "03_receiver_waveforms",
        "04_numerical_dispersion",
        "05_relative_error",
        "06_network_hyperparameters",
        "07_protocols_and_figure_plan",
        "08_reproducibility_and_diagnostics",
        "09_pending_position_evaluation",
        "10_training_runs_read_only",
        "99_historical_experiment_index",
    ):
        (current / name).mkdir(parents=True)
    (current / "02_wavefield_snapshots/historical.png").write_bytes(b"old")
    (current / "03_receiver_waveforms/historical.png").write_bytes(b"old")
    (current / "04_numerical_dispersion/phase_velocity.pdf").write_bytes(b"keep")
    (current / "05_relative_error/cpadc_r7_all_960_records.csv").write_text("keep")
    (current / "05_relative_error/r5b_fixed_train_gate_all_48_records.csv").write_text("keep")
    (current / "06_network_hyperparameters/network_hyperparameters.json").write_text("keep")
    (current / "07_protocols_and_figure_plan/protocol.json").write_text("{}")
    (current / "08_reproducibility_and_diagnostics/audit.json").write_text("{}")
    (current / "10_training_runs_read_only/README.md").write_text("keep")

    for family in ("uniform", "layered", "marmousi"):
        family_dir = comparison / family
        family_dir.mkdir(parents=True)
        (family_dir / f"{family}_complete_wavefield.npz").write_bytes(b"npz")
        for stem in (
            f"{family}_wavefield_snapshots",
            f"{family}_receiver_waveforms",
            f"{family}_receiver_gather",
        ):
            (family_dir / f"{stem}.png").write_bytes(b"png")
            (family_dir / f"{stem}.pdf").write_bytes(b"pdf")
    (comparison / "comparison_report.json").write_text("{}")
    (comparison / "prediction_manifest.json").write_text("{}")

    position.mkdir()
    records = []
    for index in range(240):
        records.append(
            {
                "record_id": f"r{index}",
                "slice_rank": index // 8 + 1,
                "case_id": f"c{index % 8}",
                "role": (
                    "interpolation"
                    if index % 8 < 5
                    else "outside_train_position_range"
                ),
                "metrics": {
                    "record_relative_l2": 0.1 + index / 10_000,
                    "late_relative_l2": 0.2,
                    "spectrum_high_relative_l2": 0.3,
                    "receiver_relative_l2": 0.4,
                },
            }
        )
    (position / "score_summary.json").write_text(
        json.dumps({"checkpoint_sha256": "abc", "records": records})
    )
    (position / "checkpoint_selection.json").write_text("{}")
    figure_dir = position / "figures"
    figure_dir.mkdir()
    (figure_dir / "position.pdf").write_bytes(b"pdf")

    config = tmp_path / "selected.yaml"
    identity = tmp_path / "identity.json"
    config.write_text("epochs: 1\n")
    identity.write_text("{}")
    selection = {
        "config": str(config),
        "checkpoint_identity": str(identity),
        "checkpoint_sha256": "abc",
        "selection_source": "r5b_preregistered_fallback",
    }
    decision = {"promotion_permitted": True}

    stage_bundle(
        current_bundle=current,
        archive_bundle=archive,
        comparison_dir=comparison,
        position_dir=position,
        selection=selection,
        decision=decision,
        staging=staging,
    )

    assert not (staging / "02_wavefield_snapshots/historical.png").exists()
    assert not (staging / "03_receiver_waveforms/historical.png").exists()
    assert (staging / "02_wavefield_snapshots/latest_marmousi_wavefield_snapshots.pdf").is_file()
    assert (staging / "03_receiver_waveforms/latest_layered_receiver_gather.png").is_file()
    assert (staging / "05_relative_error/latest_fixed19hz_all_240_records.csv").is_file()
    assert (staging / "05_relative_error/cpadc_r7_all_960_records.csv").is_file()
    assert (staging / "05_relative_error/r5b_fixed_train_gate_all_48_records.csv").is_file()
    assert (staging / "06_network_hyperparameters/network_hyperparameters.json").is_file()
    assert (staging / "09_fixed19_position_evaluation/score_summary.json").is_file()
    assert (staging / "04_numerical_dispersion/phase_velocity.pdf").is_file()
    assert (staging / "99_historical_experiment_index/pre_latest_bundle_read_only").is_symlink()
    manifest_rows = list(
        csv.DictReader((staging / "FILE_MANIFEST.csv").open(encoding="utf8"))
    )
    assert manifest_rows
    assert all(None not in row for row in manifest_rows)


def test_nonpromotion_records_results_without_replacing_historical_assets(tmp_path: Path):
    bundle = tmp_path / "bundle"
    comparison = tmp_path / "comparison"
    position = tmp_path / "position"
    (bundle / "01_complete_wavefields").mkdir(parents=True)
    historical = bundle / "01_complete_wavefields/historical.npz"
    historical.write_bytes(b"keep")
    (bundle / "05_relative_error").mkdir()
    pending = bundle / "09_pending_position_evaluation"
    pending.mkdir()
    (bundle / "EVIDENCE_STATUS.csv").write_text(
        "artifact,status,evidence_scope,directory\n"
        "240-case fixed-19-Hz position boxplot,pending,queued,09_pending_position_evaluation\n",
        encoding="utf8",
    )
    (bundle / "README.md").write_text("# Bundle\n", encoding="utf8")
    (bundle / "README_zh.md").write_text("# 证据包\n", encoding="utf8")
    comparison.mkdir()
    (comparison / "comparison_report.json").write_text(
        json.dumps({"replacement_decision": {"replacement_gate_passed": False}})
    )
    (comparison / "prediction_manifest.json").write_text("{}")
    position.mkdir()
    records = []
    for index in range(240):
        records.append(
            {
                "record_id": f"r{index}",
                "slice_rank": index // 8 + 1,
                "case_id": f"c{index % 8}",
                "role": "interpolation" if index % 8 < 5 else "outside_train_position_range",
                "metrics": {
                    "record_relative_l2": 0.5,
                    "late_relative_l2": 0.6,
                    "spectrum_high_relative_l2": 0.7,
                    "receiver_relative_l2": 0.4,
                },
            }
        )
    (position / "score_summary.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": "abc",
                "overall": {"record_relative_l2": {"mean": 0.5}},
                "records": records,
            }
        )
    )
    for name in ("checkpoint_selection.json", "prediction_manifest.json", "reference_manifest.json"):
        (position / name).write_text("{}")
    figures = position / "figures"
    figures.mkdir()
    (figures / "position.pdf").write_bytes(b"pdf")

    record_nonpromotion_evidence(
        bundle=bundle,
        comparison_dir=comparison,
        position_dir=position,
        selection={"checkpoint_sha256": "abc"},
        decision={"promotion_permitted": False, "promotion_block_reason": "same_sample_accuracy_gate_failed"},
    )

    assert historical.read_bytes() == b"keep"
    assert (bundle / "05_relative_error/latest_fixed19hz_all_240_records.csv").is_file()
    assert (pending / "score_summary.json").is_file()
    assert (bundle / "11_latest_candidate_nonpromotion_test/comparison_report.json").is_file()
    rows = list(csv.DictReader((bundle / "EVIDENCE_STATUS.csv").open(encoding="utf8")))
    assert any(row["status"] == "complete; not promoted" for row in rows)
