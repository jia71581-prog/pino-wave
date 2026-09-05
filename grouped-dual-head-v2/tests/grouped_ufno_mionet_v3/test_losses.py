from __future__ import annotations

import math

import pytest
import torch

from grouped_ufno_mionet_v3.losses import (
    V3LossWeights,
    complex_spectrum_loss,
    compute_v3_losses,
    per_frame_relative_l2,
    query_point_loss,
    relative_head_consistency_loss,
    spatial_gradient_loss,
    spectral_phase_loss,
    time_difference_loss,
)


def test_per_frame_relative_l2_weights_each_time_equally():
    target = torch.tensor([[[[1.0]], [[10.0]]]])
    prediction = torch.tensor([[[[0.0]], [[9.0]]]])
    mean, per_frame = per_frame_relative_l2(prediction, target)
    torch.testing.assert_close(per_frame, torch.tensor([[1.0, 0.1]]))
    assert mean.item() == pytest.approx(0.55)


def test_relative_energy_floor_prevents_near_silent_frame_from_dominating():
    target = torch.tensor([[[[1.0e-6]], [[1.0]]]])
    prediction = torch.tensor([[[[1.0e-3]], [[1.0]]]])
    unstable, _ = per_frame_relative_l2(
        prediction, target, energy_floor_fraction=0.0
    )
    stable, per_frame = per_frame_relative_l2(
        prediction, target, energy_floor_fraction=0.05
    )
    assert unstable.item() > 100.0
    assert stable.item() < 0.02
    assert per_frame[0, 0].item() == pytest.approx(0.01998, rel=1.0e-4)


def test_relative_energy_floor_is_scale_invariant_for_frame_and_spectrum():
    target = torch.zeros(1, 2, 8, 8)
    target[:, 0, 2, 3] = 1.0e-4
    target[:, 1, 2, 3] = 1.0
    prediction = target.clone()
    prediction[:, 0, 2, 3] += 2.0e-3
    frame_a, _ = per_frame_relative_l2(
        prediction, target, energy_floor_fraction=0.05
    )
    spectrum_a = complex_spectrum_loss(
        prediction, target, energy_floor_fraction=0.05
    )
    frame_b, _ = per_frame_relative_l2(
        prediction * 1.0e3, target * 1.0e3, energy_floor_fraction=0.05
    )
    spectrum_b = complex_spectrum_loss(
        prediction * 1.0e3, target * 1.0e3, energy_floor_fraction=0.05
    )
    torch.testing.assert_close(frame_a, frame_b)
    torch.testing.assert_close(spectrum_a, spectrum_b)


def test_complex_spectrum_detects_real_and_imaginary_coefficient_error():
    y = torch.zeros(1, 1, 8, 8)
    y[0, 0, 2, 3] = 1.0
    shifted = torch.roll(y, shifts=1, dims=-1)
    assert complex_spectrum_loss(y, y).item() == pytest.approx(0.0, abs=1.0e-7)
    assert complex_spectrum_loss(shifted, y).item() > 0.1


def test_spectral_phase_masks_low_energy_and_rejects_empty_mask():
    x = torch.arange(16, dtype=torch.float32)
    zz, xx = torch.meshgrid(x, x, indexing="ij")
    target = torch.sin(2.0 * math.pi * xx / 8.0)[None, None]
    shifted = torch.cos(2.0 * math.pi * xx / 8.0)[None, None]
    same, count = spectral_phase_loss(target, target, energy_fraction=0.01)
    different, different_count = spectral_phase_loss(shifted, target, energy_fraction=0.01)
    assert same.item() == pytest.approx(0.0, abs=1.0e-7)
    assert count == different_count and count > 0
    assert different.item() > 0.5
    with pytest.raises(RuntimeError, match="empty phase mask"):
        spectral_phase_loss(torch.zeros_like(target), torch.zeros_like(target))


def test_spatial_and_temporal_derivative_losses_are_zero_only_when_matched():
    target = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(1, 3, 5, 7)
    assert spatial_gradient_loss(target, target).item() == 0.0
    assert time_difference_loss(target, target).item() == 0.0
    spatial_bad = target.clone()
    spatial_bad[..., 2:, :] *= 2
    temporal_bad = target.clone()
    temporal_bad[:, 2] += 10
    assert spatial_gradient_loss(spatial_bad, target).item() > 0
    assert time_difference_loss(temporal_bad, target).item() > 0


def test_query_point_loss_uses_inverse_probability_correction_and_relative_term():
    prediction = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[1.0, 2.0]])
    uniform = query_point_loss(prediction, target, torch.tensor([[0.5, 0.5]]))
    rare_error = query_point_loss(prediction, target, torch.tensor([[0.9, 0.1]]))
    assert rare_error.total > uniform.total
    assert rare_error.relative_l2.item() == pytest.approx(math.sqrt(4.0 / 5.0))


def test_head_consistency_is_relative_to_supervised_target_energy():
    target = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    dense = target.clone()
    query = target + 1.0
    loss = relative_head_consistency_loss(dense, query, target)
    assert loss.item() == pytest.approx(0.55)
    scaled = relative_head_consistency_loss(dense * 7.0, query * 7.0, target * 7.0)
    torch.testing.assert_close(loss, scaled)


def test_composite_loss_logs_unweighted_and_weighted_components():
    target_dense = torch.randn(2, 3, 9, 9)
    prediction_dense = target_dense + 0.05 * torch.randn_like(target_dense)
    target_query = target_dense[:, :, 4, 4]
    prediction_query = prediction_dense[:, :, 4, 4]
    weights = V3LossWeights(
        point=1.0,
        frame=2.0,
        complex_spectrum=0.2,
        spectral_phase=0.1,
        spatial_gradient=0.3,
        time_difference=0.4,
        consistency=0.5,
    )
    result = compute_v3_losses(
        prediction_query=prediction_query,
        target_query=target_query,
        query_probability=torch.full_like(target_query, 1.0 / target_query.numel()),
        prediction_dense=prediction_dense,
        target_dense=target_dense,
        dense_at_query=prediction_query,
        weights=weights,
    )
    expected = sum(result.weighted.values())
    torch.testing.assert_close(result.total, expected)
    assert set(result.unweighted) == {
        "point",
        "frame",
        "complex_spectrum",
        "spectral_phase",
        "spatial_gradient",
        "time_difference",
        "consistency",
    }
    assert result.phase_mask_count > 0
    assert torch.isfinite(result.total)
