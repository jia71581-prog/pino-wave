import json

import pytest

from saved_time_phase_operator_v4.epoch_monitor import (
    summarize_epoch_metrics,
    write_epoch_summary,
)


def _epoch(
    epoch,
    aggregate,
    *,
    uniform,
    layered,
    marmousi,
    elapsed=100.0,
    panel="fixed",
):
    return {
        "event": "epoch",
        "epoch": epoch,
        "elapsed_seconds": elapsed,
        "train_loss": 0.2,
        "checkpoint": f"checkpoints/epoch_{epoch:04d}.pt",
        "metrics": {
            "aggregate_relative_l2": aggregate,
            "family_relative_l2": {
                "uniform": uniform,
                "layered": layered,
                "marmousi": marmousi,
            },
            "source_relative_l2": {
                f"validation_{panel}_{index:05d}": aggregate for index in range(48)
            },
        },
    }


def test_summarize_waits_until_candidate_epoch_exists():
    parent = [_epoch(1, 0.47, uniform=0.40, layered=0.44, marmousi=0.54)]

    report = summarize_epoch_metrics([], parent)

    assert report == {"status": "waiting_for_epoch"}


def test_summarize_selects_best_epoch_and_applies_both_gates():
    parent = [
        _epoch(1, 0.47, uniform=0.40, layered=0.44, marmousi=0.54),
        _epoch(2, 0.48, uniform=0.41, layered=0.45, marmousi=0.55),
    ]
    candidate = [
        _epoch(1, 0.46, uniform=0.39, layered=0.43, marmousi=0.53, elapsed=900.0),
        _epoch(2, 0.09, uniform=0.08, layered=0.10, marmousi=0.11, elapsed=1800.0),
    ]

    report = summarize_epoch_metrics(candidate, parent)

    assert report["status"] == "epochs_available"
    assert report["completed_epochs"] == 2
    assert report["latest_epoch"] == 2
    assert report["best_epoch"] == 2
    assert report["best_aggregate_relative_l2"] == pytest.approx(0.09)
    assert report["parent_aggregate_relative_l2"] == pytest.approx(0.47)
    assert report["relative_improvement_vs_parent"] == pytest.approx((0.47 - 0.09) / 0.47)
    assert report["pilot_promotion_eligible"] is True
    assert report["final_target_met"] is True
    assert report["seconds_per_completed_epoch"] == pytest.approx(900.0)
    assert report["best_checkpoint"] == "checkpoints/epoch_0002.pt"


def test_summarize_rejects_family_regression_even_when_aggregate_improves():
    parent = [_epoch(1, 0.47, uniform=0.40, layered=0.44, marmousi=0.54)]
    candidate = [_epoch(1, 0.46, uniform=0.39, layered=0.48, marmousi=0.50)]

    report = summarize_epoch_metrics(candidate, parent, family_regression_tolerance=0.03)

    assert report["pilot_promotion_eligible"] is False
    assert report["family_regression_ok"] is False
    assert report["final_target_met"] is False


def test_summarize_rejects_cross_panel_comparison():
    parent = [
        _epoch(
            1,
            0.47,
            uniform=0.40,
            layered=0.44,
            marmousi=0.54,
            panel="seed372",
        )
    ]
    candidate = [
        _epoch(
            1,
            0.46,
            uniform=0.39,
            layered=0.43,
            marmousi=0.50,
            panel="seed401",
        )
    ]

    report = summarize_epoch_metrics(candidate, parent)

    assert report["same_validation_panel"] is False
    assert report["pilot_promotion_eligible"] is False


def test_summarize_rejects_nonfinite_or_incomplete_metrics():
    parent = [_epoch(1, 0.47, uniform=0.40, layered=0.44, marmousi=0.54)]
    bad = _epoch(1, float("nan"), uniform=0.39, layered=0.43, marmousi=0.50)

    with pytest.raises(ValueError, match="finite"):
        summarize_epoch_metrics([bad], parent)

    del bad["metrics"]["family_relative_l2"]["uniform"]
    bad["metrics"]["aggregate_relative_l2"] = 0.46
    with pytest.raises(ValueError, match="families"):
        summarize_epoch_metrics([bad], parent)


def test_write_epoch_summary_atomically_materializes_json(tmp_path):
    candidate_path = tmp_path / "candidate.jsonl"
    parent_path = tmp_path / "parent.jsonl"
    output_path = tmp_path / "monitor" / "epoch_summary.json"
    candidate_path.write_text(
        json.dumps(_epoch(1, 0.46, uniform=0.39, layered=0.43, marmousi=0.53))
        + "\n"
    )
    parent_path.write_text(
        json.dumps(_epoch(1, 0.47, uniform=0.40, layered=0.44, marmousi=0.54))
        + "\n"
    )

    report = write_epoch_summary(candidate_path, parent_path, output_path)

    assert json.loads(output_path.read_text()) == report
    assert report["pilot_promotion_eligible"] is True
    assert list(output_path.parent.glob("*.partial.*")) == []


def test_write_epoch_summary_reads_identity_bound_parent_report(tmp_path):
    candidate_path = tmp_path / "candidate.jsonl"
    parent_path = tmp_path / "same_panel_parent.json"
    output_path = tmp_path / "epoch_summary.json"
    candidate = _epoch(
        1, 0.46, uniform=0.39, layered=0.43, marmousi=0.53, panel="seed401"
    )
    parent_metrics = _epoch(
        0, 0.47, uniform=0.40, layered=0.44, marmousi=0.54, panel="seed401"
    )["metrics"]
    candidate_path.write_text(json.dumps(candidate) + "\n")
    parent_path.write_text(
        json.dumps(
            {
                "schema": "saved_time_family_expert_same_panel_parent_v1",
                "status": "complete",
                "metrics": parent_metrics,
            }
        )
        + "\n"
    )

    report = write_epoch_summary(candidate_path, parent_path, output_path)

    assert report["same_validation_panel"] is True
    assert report["parent_aggregate_relative_l2"] == pytest.approx(0.47)
