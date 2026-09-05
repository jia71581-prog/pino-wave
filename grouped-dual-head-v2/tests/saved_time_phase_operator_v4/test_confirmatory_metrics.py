from __future__ import annotations

import math

import pytest
import torch

from saved_time_phase_operator_v4.confirmatory_metrics import (
    complete_transient_metrics,
    propagation_windows,
    receiver_lag_phase_metrics_saved_time,
    spatial_spectrum_relative_l2,
)


def _case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time = torch.linspace(0.0, 1.0, 401)
    z = torch.linspace(0.0, 1.0, 11)
    x = torch.linspace(0.0, 1.0, 13)
    target = (
        torch.sin(2.0 * torch.pi * (7.0 * time[:, None, None] - x[None, None, :]))
        * torch.exp(-4.0 * (z[None, :, None] - 0.4).square())
    )
    target[:20] = 0.0
    return target.clone(), target, time


def test_401_frame_windows_cover_axis_once() -> None:
    windows = propagation_windows(401, 20)
    indices = []
    for window in windows.values():
        indices.extend(range(*window.indices(401)))
    assert indices == list(range(401))
    assert windows["late"].stop == 401


def test_identity_has_finite_zero_error_and_perfect_receiver_phase() -> None:
    prediction, target, time = _case()
    result = complete_transient_metrics(
        prediction,
        target,
        time,
        onset_index=20,
        receiver_z_index=2,
        receiver_x_indices=torch.arange(1, 12, 2),
        compute_komega=True,
        temporal_modes=8,
        komega_mode_block_size=3,
    )
    assert all(math.isfinite(value) for value in result.values())
    assert result["record_relative_l2"] == pytest.approx(0.0, abs=1e-7)
    assert result["late_error_slope_per_s"] == pytest.approx(0.0, abs=1e-7)
    assert result["receiver_phase_coherence"] == pytest.approx(1.0, abs=1e-6)
    assert result["komega_high_late"] == pytest.approx(0.0, abs=1e-6)


def test_late_perturbation_is_exposed_by_late_and_slope_metrics() -> None:
    prediction, target, time = _case()
    ramp = torch.linspace(0.0, 0.5, 121)[:, None, None]
    prediction[-121:] = prediction[-121:] * (1.0 + ramp)
    result = complete_transient_metrics(
        prediction,
        target,
        time,
        onset_index=20,
        receiver_z_index=2,
        receiver_x_indices=[1, 3, 5, 7, 9, 11],
        compute_komega=False,
    )
    assert result["late_relative_l2"] > result["early_relative_l2"]
    assert result["late_error_slope_per_s"] > 0.0
    assert result["late_max_per_time_relative_l2"] > 0.45


def test_saved_time_receiver_lag_recovers_physical_delay_on_nonuniform_axis() -> None:
    parameter = torch.linspace(0.0, 1.0, 401, dtype=torch.float64)
    time = 0.8 * parameter.square() + 0.2 * parameter
    delay = 0.04
    target = torch.exp(-((time - 0.45) / 0.025).square()).float()[None, None]
    prediction = torch.exp(-((time - delay - 0.45) / 0.025).square()).float()[None, None]
    metrics = receiver_lag_phase_metrics_saved_time(prediction, target, time)
    uniform_dt = float((time[-1] - time[0]) / 400)
    assert metrics["receiver_lag_abs_s"] == pytest.approx(delay, abs=1.5 * uniform_dt)
    assert 0.0 <= metrics["receiver_phase_coherence"] <= 1.0


def test_high_spatial_frequency_error_is_band_localized() -> None:
    prediction, target, _ = _case()
    checker = ((-1.0) ** (torch.arange(13)[None, :] + torch.arange(11)[:, None]))
    prediction = prediction + 0.1 * checker
    bands = spatial_spectrum_relative_l2(prediction, target)
    assert bands["high"] > bands["low"]


def test_spectrum_metric_penalizes_phase_not_only_amplitude() -> None:
    _, target, _ = _case()
    shifted = torch.roll(target, shifts=1, dims=-1)
    bands = spatial_spectrum_relative_l2(shifted, target)
    assert bands["low"] > 0.0
    assert bands["middle"] > 0.0


def test_nonfinite_prediction_is_rejected() -> None:
    prediction, target, time = _case()
    prediction[30, 2, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        complete_transient_metrics(
            prediction,
            target,
            time,
            onset_index=20,
            receiver_z_index=2,
            receiver_x_indices=[1, 2],
            compute_komega=False,
        )
