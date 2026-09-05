from __future__ import annotations

import torch
from torch import nn

from saved_time_phase_operator_v4.instance_adaptation.adapters import (
    OnsetAdaptedV5,
    normalize_observed_snapshots,
    normalize_velocity_contrast,
)


class DummyParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((4_000_000,), 2.0))

    def predict_wavefield(self, batch: int, times: int, height: int, width: int) -> torch.Tensor:
        return torch.ones(batch, times, height, width) * self.weight[0]


def _inputs():
    return (
        torch.ones(1, 1, 8, 8),
        torch.tensor([[20.0, 10.0, 12.0, 0.011, 1.0]]),
        torch.randn(1, 2, 8, 8),
        torch.linspace(0.0, 0.09, 10),
    )


def test_zero_init_adapter_is_parent_identity():
    parent = DummyParent()
    wrapper = OnsetAdaptedV5(parent, latent_dim=8, lora_rank=2)
    velocity, source, observed, times = _inputs()
    base = parent.predict_wavefield(1, 10, 8, 8)
    adapted = wrapper.raw_wavefield(base, velocity, source, observed, times)
    assert torch.allclose(base, adapted, atol=1e-7, rtol=1e-6)


def test_adapter_uses_less_than_one_percent_trainable_parameters():
    parent = DummyParent()
    wrapper = OnsetAdaptedV5(parent, latent_dim=8, lora_rank=2)
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapper.parameters())
    assert trainable / total < 0.01


def test_onset_loss_has_gradient_before_hard_projection():
    parent = DummyParent()
    wrapper = OnsetAdaptedV5(parent, latent_dim=8, lora_rank=2)
    velocity, source, observed, times = _inputs()
    base = parent.predict_wavefield(1, 10, 8, 8)
    prediction = wrapper.raw_wavefield(base, velocity, source, observed, times)
    loss = torch.nn.functional.mse_loss(prediction[:, [2, 3]], observed)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in wrapper.adapter_parameters())


def test_zero_latent_delta_keeps_parent_identity():
    parent = DummyParent()
    wrapper = OnsetAdaptedV5(parent, latent_dim=8, lora_rank=2)
    velocity, source, observed, times = _inputs()
    base = parent.predict_wavefield(1, 10, 8, 8)
    # The deployment LoRA (latent_delta) is zero-initialized, so a freshly
    # wrapped parent still reproduces the parent field exactly.
    assert torch.count_nonzero(wrapper.latent_delta) == 0
    adapted = wrapper.raw_wavefield(base, velocity, source, observed, times)
    assert torch.allclose(base, adapted, atol=1e-7, rtol=1e-6)


def test_adapter_parameters_exclude_deployment_latent():
    parent = DummyParent()
    wrapper = OnsetAdaptedV5(parent, latent_dim=8, lora_rank=2)
    adapter_params = wrapper.adapter_parameters()
    assert all(p is not wrapper.latent_delta for p in adapter_params)
    assert all(p is not wrapper.residual_gate for p in adapter_params)
    assert wrapper.deployment_parameters() == (wrapper.latent_delta, wrapper.residual_gate)


def test_sampled_linear_features_reproduce_full_output_at_selected_points():
    wrapper = OnsetAdaptedV5(DummyParent(), latent_dim=8, lora_rank=2)
    velocity, source, observed, times = _inputs()
    latent = torch.randn(1, 8)
    indices = torch.arange(10 * 5).reshape(1, 10, 5) % 64
    with torch.no_grad():
        wrapper.residual.output.weight.normal_(std=0.02)
        wrapper.residual.output.bias.fill_(0.01)
        full = wrapper.residual(
            velocity,
            source,
            observed,
            latent,
            times,
            output_scale=torch.ones(1),
        )
        features = wrapper.residual.sampled_linear_features(
            velocity, source, observed, latent, times, indices
        )
        linear = torch.nn.functional.linear(
            features,
            wrapper.residual.output.weight.flatten(1),
            wrapper.residual.output.bias,
        )[..., 0]
        selected = torch.gather(full.flatten(-2), -1, indices)
    assert torch.allclose(selected, torch.tanh(linear), atol=1.0e-6, rtol=1.0e-6)


def test_constant_velocity_contrast_is_exactly_zero_across_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        velocity = torch.full(
            (1, 1, 201, 201), 3497.3173828125, device=device
        )
        normalized = normalize_velocity_contrast(velocity)
        assert torch.count_nonzero(normalized) == 0
        assert torch.isfinite(normalized).all()


def test_heterogeneous_velocity_contrast_remains_standardized():
    velocity = torch.linspace(1500.0, 4500.0, 64).reshape(1, 1, 8, 8)
    normalized = normalize_velocity_contrast(velocity)
    torch.testing.assert_close(
        normalized.mean(dim=(-2, -1)), torch.zeros(1, 1), atol=1.0e-6, rtol=0
    )
    torch.testing.assert_close(
        normalized.std(dim=(-2, -1)), torch.ones(1, 1), atol=1.0e-6, rtol=0
    )


def test_tiny_observed_snapshots_are_normalized_by_record_rms():
    observed = torch.randn(2, 2, 8, 8) * torch.tensor([1.0e-13, 1.0e-11])[:, None, None, None]
    normalized, scale = normalize_observed_snapshots(observed)
    torch.testing.assert_close(
        normalized.square().flatten(1).mean(dim=1).sqrt(),
        torch.ones(2),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert bool(torch.all(scale > 0.0))
    scaled, _ = normalize_observed_snapshots(observed * 1.0e-4)
    torch.testing.assert_close(scaled, normalized, atol=1.0e-6, rtol=1.0e-6)


def test_zero_observed_snapshots_remain_exactly_zero():
    normalized, scale = normalize_observed_snapshots(torch.zeros(1, 2, 8, 8))
    assert torch.count_nonzero(normalized) == 0
    assert torch.count_nonzero(scale) == 0
