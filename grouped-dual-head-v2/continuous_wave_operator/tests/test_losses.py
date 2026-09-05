from __future__ import annotations

import torch

from continuous_wave_operator.training.losses import (
    finite_difference_wave_residual,
    query_pressure_loss,
    spectral_trace_loss,
    wave_equation_residual,
)


def test_query_and_spectral_losses_are_sensitive_and_differentiable() -> None:
    prediction = torch.tensor([[[0.0, 1.0, -1.0, 0.5]]], requires_grad=True)
    target = prediction.detach().clone()
    probabilities = torch.tensor([[[0.5, 0.2, 0.2, 0.1]]])

    exact = query_pressure_loss(prediction, target, probabilities=probabilities)
    perturbed = query_pressure_loss(prediction, target + 0.2, probabilities=probabilities)
    phase_shifted = spectral_trace_loss(prediction, torch.roll(target, shifts=1, dims=-1))

    assert exact.item() == 0.0
    assert perturbed.item() > 0.0
    assert phase_shifted.item() > 0.0
    perturbed.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_plane_wave_has_near_zero_acoustic_residual() -> None:
    coordinates = torch.rand(2, 3, 7, 3, requires_grad=True)
    wave_number = 2.0
    velocity = 3.0
    pressure = torch.sin(wave_number * coordinates[..., 0] - velocity * wave_number * coordinates[..., 2])

    residual = wave_equation_residual(pressure, coordinates, torch.tensor(velocity))

    assert residual.abs().max() < 1.0e-4


def test_finite_difference_plane_wave_residual_is_small() -> None:
    coordinates = torch.rand(2, 1, 8, 3, dtype=torch.float64)
    coordinates[..., 0] = 100.0 + coordinates[..., 0] * 1000.0
    coordinates[..., 1] = 100.0 + coordinates[..., 1] * 1000.0
    coordinates[..., 2] = 0.1 + coordinates[..., 2] * 0.7
    wave_number = 0.01
    velocity = 3.0

    def plane_wave(query: torch.Tensor) -> torch.Tensor:
        return torch.sin(wave_number * query[..., 0] - velocity * wave_number * query[..., 2])

    residual = finite_difference_wave_residual(
        plane_wave,
        coordinates,
        torch.tensor(velocity, dtype=torch.float64),
        dx_m=0.5,
        dz_m=0.5,
        dt_s=0.01,
    )

    assert residual.abs().max() < 2.0e-4
