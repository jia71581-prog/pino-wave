from __future__ import annotations

import torch
from torch import nn

from saved_time_phase_operator_v4.instance_adaptation.b2_v11_local_meta import (
    LocalMetaAdaptConfig,
    LocalResidualMetaOperator,
    adapt_local_meta,
    apply_temporal_polynomial,
    fit_temporal_polynomial,
    prefix_residual_context,
    query_features,
)
from scripts.train_b2_v11_local_meta import meta_episode_loss


def test_prefix_context_uses_only_observed_residuals():
    parent = torch.randn(1, 16, 1, 7, 6)
    observed = parent[:, :12].clone()
    observed[:, 8:] += torch.linspace(0.1, 0.4, 4)[None, :, None, None, None]
    first = prefix_residual_context(parent, observed, anchor_frames=8)
    changed_future = parent.clone()
    changed_future[:, 12:] += 100.0
    second = prefix_residual_context(changed_future, observed, anchor_frames=8)
    assert first.shape == (1, 4, 7, 6)
    assert torch.equal(first, second)


def test_query_feature_channel_contract():
    parent = torch.randn(2, 16, 1, 9, 8)
    conditioning = torch.randn(2, 20, 9, 8)
    context = torch.randn(2, 4, 9, 8)
    indices = torch.tensor([[10, 12, 15], [9, 13, 14]])
    features = query_features(
        parent,
        conditioning,
        context,
        indices,
        context_count=torch.tensor([10, 9]),
        time_s=torch.arange(16) * 0.0025,
        source_f0_hz=torch.tensor([10.0, 20.0]),
        source_t0_s=torch.tensor([0.15, 0.075]),
    )
    assert features.shape == (6, 31, 9, 8)
    assert torch.isfinite(features).all()


def test_zero_initialized_local_operator_is_exact_zero_direction():
    model = LocalResidualMetaOperator(physics_channels=20, base_width=16)
    features = torch.randn(3, 31, 17, 15)
    direction = model(features)
    assert torch.equal(direction, torch.zeros_like(direction))


def test_temporal_polynomial_closed_form_recovers_known_scale():
    generator = torch.Generator().manual_seed(20260901)
    directions = torch.randn(1, 5, 1, 4, 3, generator=generator)
    time_s = torch.arange(12, dtype=torch.float32) * 0.0025
    query_time = time_s[torch.tensor([2, 4, 6, 8, 10])]
    expected = torch.tensor([[0.8, -0.2, 0.1]])
    target = apply_temporal_polynomial(directions, expected, query_time, time_s)
    fitted = fit_temporal_polynomial(
        directions,
        target,
        query_time,
        time_s,
        ridge_fraction=1.0e-10,
    )
    assert torch.allclose(fitted, expected, atol=1.0e-5)


class _ConstantDirection(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            features.shape[0], 1, *features.shape[-2:],
            dtype=features.dtype,
            device=features.device,
        )


def test_online_meta_fit_changes_only_future_and_uses_no_future_truth():
    parent = torch.ones(1, 20, 1, 5, 4)
    observed = parent[:, :14].clone()
    observed[:, 8:] += 1.0
    candidate, report = adapt_local_meta(
        _ConstantDirection(),
        parent,
        torch.zeros(1, 20, 5, 4),
        observed,
        time_s=torch.arange(20) * 0.0025,
        source_f0_hz=torch.tensor([20.0]),
        source_t0_s=torch.tensor([0.075]),
        config=LocalMetaAdaptConfig(
            anchor_frames=8,
            validation_tail_frames=4,
            ridge_fraction=1.0e-10,
            trust_ratio=10.0,
        ),
    )
    assert report["accepted"] is True
    assert report["future_truth_used"] is False
    assert report["observed_gain"] > 0.999
    assert torch.equal(candidate[:, :14], parent[:, :14])
    assert torch.allclose(candidate[:, 14:], torch.full_like(candidate[:, 14:], 2.0), atol=1.0e-4)


def test_zero_direction_online_fit_rolls_back_parent_exactly():
    parent = torch.randn(1, 20, 1, 5, 4)
    observed = parent[:, :14].clone()
    observed[:, 8:] += 0.1
    model = LocalResidualMetaOperator(physics_channels=20, base_width=16)
    candidate, report = adapt_local_meta(
        model,
        parent,
        torch.zeros(1, 20, 5, 4),
        observed,
        time_s=torch.arange(20) * 0.0025,
        source_f0_hz=torch.tensor([20.0]),
        source_t0_s=torch.tensor([0.075]),
    )
    assert report["accepted"] is False
    assert torch.equal(candidate, parent)


def test_zero_initialized_meta_episode_has_finite_nonzero_gradient():
    model = LocalResidualMetaOperator(physics_channels=20, base_width=16)
    parent = torch.ones(1, 20, 1, 17, 15)
    target = parent.clone()
    target[:, 8:] += 0.1
    loss, report = meta_episode_loss(
        model,
        parent,
        torch.zeros(1, 20, 17, 15),
        target,
        observed_count=14,
        query_indices_tensor=torch.tensor([14, 17, 19]),
        time_s=torch.arange(20) * 0.0025,
        source_f0_hz=torch.tensor([20.0]),
        source_t0_s=torch.tensor([0.075]),
        validation_tail_frames=3,
        trust_ratio=0.05,
    )
    loss.backward()
    gradient = model.output.weight.grad
    assert torch.isfinite(loss)
    assert report["candidate_relative_l2"] > 0.0
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0
