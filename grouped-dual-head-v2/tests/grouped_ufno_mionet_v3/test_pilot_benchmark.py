from __future__ import annotations

import math

import pytest

from grouped_ufno_mionet_v3.training.pilot import validate_pilot_benchmark


def _passing(**updates):
    report = {
        "steps": 3,
        "loss_min": 0.5,
        "loss_max": 0.8,
        "missing_gradient_groups": [],
        "interpolated_target_fraction": 0.25,
        "peak_cuda_memory_bytes": 8 * 1024**3,
        "data_wait_fraction": 0.1,
        "mean_step_seconds": 0.4,
        "records_per_second": 30.0,
    }
    report.update(updates)
    return report


def test_pilot_benchmark_policy_accepts_complete_finite_cuda_path():
    validated = validate_pilot_benchmark(_passing(), device_total_bytes=24 * 1024**3)
    assert validated["records_per_second"] == 30.0


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"steps": 2}, "three"),
        ({"loss_max": math.inf}, "finite"),
        ({"missing_gradient_groups": ["dense"]}, "gradient"),
        ({"interpolated_target_fraction": 0.0}, "interpolated"),
        ({"peak_cuda_memory_bytes": 25 * 1024**3}, "memory"),
        ({"mean_step_seconds": 0.0}, "timing"),
        ({"records_per_second": 0.0}, "throughput"),
    ],
)
def test_pilot_benchmark_policy_rejects_incomplete_or_unsafe_runs(updates, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        validate_pilot_benchmark(_passing(**updates), device_total_bytes=24 * 1024**3)
