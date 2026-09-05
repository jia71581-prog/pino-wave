import math

import pytest
import torch

from saved_time_phase_operator_v4.metrics import exact_wavefield_metrics


def test_energy_floor_prevents_zero_frame_relative_error_explosion():
    target = torch.zeros(2, 4, 8, 8)
    target[0, 1, 2, 2] = 1.0
    target[1, 2, 3, 3] = 2.0
    prediction = torch.full_like(target, 0.01)

    result = exact_wavefield_metrics(
        prediction,
        target,
        families=("uniform", "layered"),
        energy_floor_fraction=0.01,
    )

    assert math.isfinite(result["aggregate_floored_relative_l2"])
    assert result["near_zero_frame_count"] == 6
    assert result["frame_count"] == 8
    assert set(result["family_floored_relative_l2"]) == {"uniform", "layered"}


def test_exact_metrics_are_zero_for_identical_nonzero_fields():
    target = torch.randn(3, 4, 9, 11)

    result = exact_wavefield_metrics(
        target.clone(),
        target,
        families=("uniform", "layered", "marmousi"),
        energy_floor_fraction=0.01,
    )

    assert result["aggregate_floored_relative_l2"] == pytest.approx(0.0)
    assert result["aggregate_unfloored_relative_l2"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["phase_correlation"] == pytest.approx(1.0, abs=1.0e-6)
    assert all(value == pytest.approx(0.0) for value in result["spectrum_relative_l2"].values())


def test_exact_metrics_report_four_time_bins():
    target = torch.ones(1, 4, 5, 5)
    prediction = target.clone()
    prediction[:, -1] *= 0.5

    result = exact_wavefield_metrics(
        prediction,
        target,
        families=("uniform",),
        energy_floor_fraction=0.01,
    )

    assert set(result["time_bin_floored_relative_l2"]) == {
        "pre_onset",
        "early",
        "middle",
        "late",
    }
    assert result["time_bin_floored_relative_l2"]["late"] == pytest.approx(0.5)
    assert result["time_bin_floored_relative_l2"]["early"] == pytest.approx(0.0)


def test_exact_metrics_reject_wrong_shapes_and_families():
    with pytest.raises(ValueError, match="matching"):
        exact_wavefield_metrics(torch.zeros(1, 4, 3, 3), torch.zeros(1, 3, 3, 3), families=("uniform",))
    with pytest.raises(ValueError, match="family"):
        exact_wavefield_metrics(torch.zeros(1, 4, 3, 3), torch.zeros(1, 4, 3, 3), families=())
