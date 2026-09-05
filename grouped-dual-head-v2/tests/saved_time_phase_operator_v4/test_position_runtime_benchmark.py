from __future__ import annotations

import pytest

from scripts.benchmark_saved_time_vs_lwc84_position_runtime import (
    LWC_SCHEMA,
    MODEL_SCHEMA,
    merge_reports,
    nearest_rank,
)


def _runtime(schema: str, method: str, seconds: float):
    rows = []
    for rank in (1, 15, 30):
        for position in range(8):
            rows.append(
                {
                    "record_id": f"r{rank:02d}_p{position}",
                    "slice_rank": rank,
                    "case_id": f"p{position}",
                    "role": "interpolation" if position < 5 else "outside_train_position_range",
                    "source_parameters": [float(position), 100.0, 19.0, 1.5 / 19.0, 1.0],
                    "output_shape": [1, 401, 201, 201],
                    "runtime_s": seconds,
                }
            )
    return {
        "schema": schema,
        "status": "complete",
        "method": method,
        "protocol_sha256": "a" * 64,
        "device": {"name": "test", "torch": "test"},
        "measurements": rows,
    }


def _score(error: float):
    summary = lambda value: {"mean": value}
    return {
        "status": "complete",
        "source_generalization_variable": "position_only",
        "frequency_generalization_claim_permitted": False,
        "protocol_sha256": "a" * 64,
        "overall": {
            "record_relative_l2": summary(error),
            "prediction_finite": summary(1.0),
            "prediction_nonzero": summary(1.0),
        },
        "by_position_role": {
            "interpolation": {"record_relative_l2": summary(error)},
            "outside_train_position_range": {"record_relative_l2": summary(error)},
        },
    }


def test_nearest_rank_is_deterministic() -> None:
    assert nearest_rank([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0


def test_merge_requires_both_accuracy_and_speed() -> None:
    result = merge_reports(
        _runtime(MODEL_SCHEMA, "saved_time_operator", 1.0),
        _runtime(LWC_SCHEMA, "LWC-84_CPML", 20.0),
        _score(0.04),
    )
    assert result["accuracy_gate"]["passed"] is True
    assert result["speed_gate"]["passed"] is True
    assert result["matched_accuracy_speed_claim_permitted"] is True

    failed = merge_reports(
        _runtime(MODEL_SCHEMA, "saved_time_operator", 1.0),
        _runtime(LWC_SCHEMA, "LWC-84_CPML", 20.0),
        _score(0.2),
    )
    assert failed["matched_accuracy_speed_claim_permitted"] is False
    assert "forbidden" in failed["claim_boundary"]


def test_merge_rejects_device_mismatch() -> None:
    model = _runtime(MODEL_SCHEMA, "saved_time_operator", 1.0)
    lwc = _runtime(LWC_SCHEMA, "LWC-84_CPML", 20.0)
    lwc["device"] = {"name": "other", "torch": "test"}
    with pytest.raises(ValueError, match="devices"):
        merge_reports(model, lwc, _score(0.04))
