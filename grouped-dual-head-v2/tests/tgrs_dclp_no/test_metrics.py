from __future__ import annotations

import pytest
import torch

from tgrs_dclp_no.metrics import receiver_phase_metrics, wavefront_radius


def test_receiver_lag_recovers_known_two_sample_delay():
    time = torch.arange(64, dtype=torch.float64)
    truth = torch.sin(2.0 * torch.pi * time / 16.0)[None]
    prediction = torch.roll(truth, shifts=2, dims=-1)
    result = receiver_phase_metrics(prediction, truth, dt_s=0.0025)
    assert result["median_lag_samples"] == 2
    assert abs(result["median_lag_s"] - 0.005) < 1.0e-12
    assert result["mean_coherence"] > 0.99


def test_receiver_zero_lag_for_identical_traces():
    time = torch.arange(48, dtype=torch.float64)
    truth = torch.stack([torch.sin(2.0 * torch.pi * time / 12.0), torch.cos(2.0 * torch.pi * time / 9.0)])
    result = receiver_phase_metrics(truth.clone(), truth, dt_s=0.0025)
    assert result["median_lag_samples"] == 0
    assert result["mean_coherence"] == pytest.approx(1.0, abs=1.0e-9)


def test_receiver_rejects_mismatched_or_nonfinite():
    good = torch.zeros(2, 8)
    with pytest.raises(ValueError):
        receiver_phase_metrics(torch.zeros(2, 7), good, dt_s=0.0025)
    bad = good.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        receiver_phase_metrics(bad, good, dt_s=0.0025)


def test_wavefront_radius_recovers_a_circular_energy_ring():
    axis = torch.arange(101, dtype=torch.float64) * 10.0
    z, x = torch.meshgrid(axis, axis, indexing="ij")
    radius = torch.sqrt((x - 500.0) ** 2 + (z - 500.0) ** 2)
    field = torch.exp(-0.5 * ((radius - 300.0) / 15.0) ** 2)
    estimated = wavefront_radius(field, source_x_m=500.0, source_z_m=500.0, x_m=axis, z_m=axis)
    assert abs(estimated - 300.0) <= 10.0


def test_wavefront_radius_rejects_axis_mismatch():
    axis = torch.arange(10, dtype=torch.float64) * 10.0
    with pytest.raises(ValueError):
        wavefront_radius(torch.zeros(9, 10), source_x_m=0.0, source_z_m=0.0, x_m=axis, z_m=axis)
