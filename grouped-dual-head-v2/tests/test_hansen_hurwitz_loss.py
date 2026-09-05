from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from fno_acoustic.query_losses import (
    HHFieldLoss,
    analytic_signal_on_physical_time,
    auxiliary_query_losses,
    hansen_hurwitz_field_loss,
    unweighted_field_loss,
)


def _traces_and_patches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    traces = torch.randn(2, 3, 160, generator=generator)
    patches = torch.randn(2, 2, 6, 8, 160, generator=generator)
    time_s = torch.linspace(0.0, 1.0, 160).square()
    return traces, patches, time_s


def test_hh_mse_equals_dense_mse_for_uniform_census():
    generator = torch.Generator().manual_seed(11)
    target = torch.randn(2, 12, 160, generator=generator)
    prediction = target + 0.1 * torch.randn(target.shape, generator=generator)
    q = torch.full((2, 12), 1.0 / 12.0)

    result = hansen_hurwitz_field_loss(prediction, target, q, population_size=12)

    assert torch.allclose(result.mse, (prediction - target).square().mean(), atol=1e-7)
    assert result.relative_kind == "consistent_ratio_surrogate"
    assert result.loss == result.mse + result.relative_l2


def test_hh_mse_is_monte_carlo_unbiased_for_nonuniform_sampling():
    generator = torch.Generator().manual_seed(17)
    population_size = 5
    draws = 30_000
    values = torch.tensor([0.2, 0.7, 1.1, 1.8, 2.4])
    q_population = torch.tensor([0.05, 0.10, 0.15, 0.25, 0.45])
    indices = torch.multinomial(q_population, draws, replacement=True, generator=generator)
    target = torch.zeros(1, draws, 160)
    prediction = values[indices].view(1, draws, 1).expand(-1, -1, 160)
    q = q_population[indices].view(1, draws)

    estimate = hansen_hurwitz_field_loss(
        prediction, target, q, population_size=population_size
    ).mse

    assert estimate.item() == pytest.approx(values.square().mean().item(), abs=0.02)


def test_late_time_weights_change_mse_and_must_be_positive_finite_full160():
    target = torch.zeros(1, 2, 160)
    prediction = torch.zeros_like(target)
    prediction[..., -1] = 2.0
    q = torch.full((1, 2), 0.5)
    weights = torch.ones(160)
    weights[-1] = 4.0

    weighted = hansen_hurwitz_field_loss(
        prediction, target, q, population_size=2, late_time_weights=weights
    )

    assert weighted.mse == pytest.approx(16.0 / 163.0)
    for invalid in (torch.ones(159), torch.cat((torch.ones(159), torch.tensor([0.0])))):
        with pytest.raises(ValueError, match="late_time_weights"):
            hansen_hurwitz_field_loss(
                prediction, target, q, population_size=2, late_time_weights=invalid
            )


def test_hh_and_unweighted_losses_are_finite_for_zero_target_and_differentiable():
    prediction = torch.zeros(1, 3, 160, requires_grad=True)
    target = torch.zeros_like(prediction)
    q = torch.full((1, 3), 1.0 / 3.0)

    hh = hansen_hurwitz_field_loss(prediction, target, q, population_size=3)
    plain = unweighted_field_loss(prediction, target)
    (hh.loss + plain.loss).backward()

    assert torch.isfinite(hh.loss) and torch.isfinite(plain.loss)
    assert hh.loss == 0 and plain.loss == 0
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    with pytest.raises(FrozenInstanceError):
        hh.mse = torch.tensor(1.0)


def test_hh_and_unweighted_relative_losses_use_norm_scale_epsilon():
    target = torch.full((1, 1, 160), 1.0e-6)
    prediction = target * 2.0
    q = torch.ones(1, 1)

    hh = hansen_hurwitz_field_loss(
        prediction, target, q, population_size=1, eps=1.0e-8
    )
    plain = unweighted_field_loss(prediction, target, eps=1.0e-8)

    assert hh.relative_l2.item() == pytest.approx(1.0, rel=1e-5)
    assert plain.relative_l2.item() == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hh_tiny_float32_q_is_finite_for_low_precision_fields(dtype):
    target = torch.ones(1, 1, 160, dtype=dtype)
    prediction = (target * 1.01).detach().requires_grad_()
    q = torch.full((1, 1), 1.0e-8, dtype=torch.float32)
    weights = torch.linspace(0.5, 2.0, 160, dtype=torch.float64)

    result = hansen_hurwitz_field_loss(
        prediction, target, q, population_size=1, late_time_weights=weights
    )
    result.loss.backward()

    assert result.loss.dtype == torch.float32
    assert torch.isfinite(result.loss)
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_hh_rejects_q_that_underflows_when_converted_to_compute_dtype():
    prediction = torch.ones(1, 1, 160, dtype=torch.float16)
    q = torch.tensor([[1.0e-8]], dtype=torch.float16)

    with pytest.raises(ValueError, match="positive"):
        hansen_hurwitz_field_loss(prediction, prediction, q, population_size=1)


@pytest.mark.parametrize(
    ("prediction", "target", "q", "population_size", "message"),
    [
        (torch.zeros(1, 2, 159), torch.zeros(1, 2, 159), torch.full((1, 2), 0.5), 2, "160"),
        (torch.zeros(1, 2, 160), torch.zeros(1, 3, 160), torch.full((1, 2), 0.5), 2, "shape"),
        (torch.zeros(1, 2, 160), torch.zeros(1, 2, 160), torch.ones(1, 3), 2, "probabilities"),
        (torch.zeros(1, 2, 160), torch.zeros(1, 2, 160), torch.tensor([[0.5, 0.0]]), 2, "positive"),
        (torch.zeros(1, 2, 160), torch.zeros(1, 2, 160), torch.tensor([[1.1, 0.5]]), 2, "probabilities"),
        (torch.zeros(1, 2, 160), torch.zeros(1, 2, 160), torch.full((1, 2), 0.5), 0, "population_size"),
    ],
)
def test_hh_rejects_invalid_contracts(prediction, target, q, population_size, message):
    with pytest.raises(ValueError, match=message):
        hansen_hurwitz_field_loss(prediction, target, q, population_size)


def test_auxiliary_losses_are_zero_for_identical_inputs():
    traces, patches, time_s = _traces_and_patches()
    traces.requires_grad_()

    losses = auxiliary_query_losses(traces, traces, patches, patches, time_s)
    losses.phase.backward()

    assert losses.receiver == 0
    assert losses.local_spectrum == 0
    assert losses.energy == 0
    assert losses.phase == 0
    assert traces.grad is not None and torch.isfinite(traces.grad).all()


def test_phase_loss_detects_known_shift_on_nonuniform_physical_time():
    uniform_parameter = torch.linspace(0.0, 1.0, 160)
    time_s = uniform_parameter.pow(1.7)
    phase = 2.0 * torch.pi * 5.0 * time_s
    target = torch.sin(phase).view(1, 1, 160)
    shifted = torch.sin(phase + 0.4).view(1, 1, 160)
    patch = torch.zeros(1, 1, 4, 4, 160)

    losses = auxiliary_query_losses(shifted, target, patch, patch, time_s)

    assert losses.phase > 0.05


def test_phase_loss_is_scale_invariant_for_tiny_amplitude():
    uniform_parameter = torch.linspace(0.0, 1.0, 160)
    time_s = uniform_parameter.pow(1.7)
    phase = 2.0 * torch.pi * 5.0 * time_s
    target = torch.sin(phase).view(1, 1, 160)
    shifted = torch.sin(phase + 0.4).view(1, 1, 160)
    patch = torch.zeros(1, 1, 4, 4, 160)

    base = auxiliary_query_losses(shifted, target, patch, patch, time_s).phase
    tiny = auxiliary_query_losses(
        shifted * 1.0e-6, target * 1.0e-6, patch, patch, time_s
    ).phase

    assert tiny > 0.05
    assert tiny.item() == pytest.approx(base.item(), rel=2e-4, abs=1e-6)


def test_zero_padded_hilbert_does_not_wrap_late_impulse_to_early_boundary():
    trace = torch.zeros(1, 1, 160)
    trace[..., 155] = 1.0
    time_s = torch.linspace(0.0, 1.0, 160)

    analytic = analytic_signal_on_physical_time(trace, time_s)

    assert analytic[..., 0].abs().item() < 0.05 * analytic[..., 154].abs().item()


@pytest.mark.parametrize("start", [8, 138])
def test_envelope_weighted_phase_detects_compact_early_and_late_transients(start):
    time_s = torch.linspace(0.0, 1.0, 160).pow(1.3)
    target = torch.zeros(1, 1, 160)
    prediction = torch.zeros_like(target)
    window = torch.hann_window(16, periodic=False)
    carrier = torch.linspace(0.0, 4.0 * torch.pi, 16)
    target[..., start : start + 16] = window * torch.cos(carrier)
    prediction[..., start : start + 16] = window * torch.cos(carrier + 0.4)
    patch = torch.zeros(1, 1, 4, 4, 160)

    phase = auxiliary_query_losses(
        prediction, target, patch, patch, time_s
    ).phase

    assert torch.isfinite(phase)
    assert phase > 0.05


def test_local_spectrum_detects_high_spatial_frequency_error_in_core():
    time_s = torch.linspace(0.0, 1.0, 160)
    receivers = torch.ones(1, 1, 160)
    target = torch.zeros(1, 1, 8, 8, 160)
    prediction = target.clone()
    checkerboard = (2 * ((torch.arange(6)[:, None] + torch.arange(6)[None, :]) % 2) - 1).float()
    prediction[:, :, 1:-1, 1:-1, :] = checkerboard[None, None, :, :, None]

    losses = auxiliary_query_losses(
        receivers, receivers, prediction, target, time_s, core_margin=1
    )

    assert torch.isfinite(losses.local_spectrum)
    assert losses.local_spectrum > 0


def test_local_spectrum_relative_stabilizer_is_scale_invariant():
    time_s = torch.linspace(0.0, 1.0, 160)
    receivers = torch.ones(1, 1, 160)
    checkerboard = (2 * ((torch.arange(6)[:, None] + torch.arange(6)[None, :]) % 2) - 1).float()
    target = checkerboard[None, None, :, :, None].expand(-1, -1, -1, -1, 160).clone()
    prediction = target * 1.2

    base = auxiliary_query_losses(
        receivers, receivers, prediction, target, time_s, core_margin=0
    ).local_spectrum
    tiny = auxiliary_query_losses(
        receivers, receivers, prediction * 1.0e-6, target * 1.0e-6, time_s, core_margin=0
    ).local_spectrum

    assert tiny.item() == pytest.approx(base.item(), rel=1e-5, abs=1e-7)


def test_zero_target_local_spectrum_tracks_hallucination_amplitude_and_gradient():
    time_s = torch.linspace(0.0, 1.0, 160)
    receivers = torch.ones(1, 1, 160)
    target = torch.zeros(1, 1, 6, 6, 160)
    checkerboard = (2 * ((torch.arange(6)[:, None] + torch.arange(6)[None, :]) % 2) - 1).float()
    losses = []

    for amplitude in (1.0, 1.0e-3, 1.0e-6):
        prediction = (
            checkerboard[None, None, :, :, None]
            .expand(-1, -1, -1, -1, 160)
            .mul(amplitude)
            .clone()
            .requires_grad_()
        )
        loss = auxiliary_query_losses(
            receivers, receivers, prediction, target, time_s, core_margin=0
        ).local_spectrum
        prediction_k = torch.fft.rfft2(prediction, dim=(2, 3))
        kx = torch.fft.fftfreq(6) / 0.5
        kz = torch.fft.rfftfreq(6) / 0.5
        high_k = torch.sqrt(kx[:, None].square() + kz[None, :].square()) >= 0.5
        expected_absolute_energy = prediction_k.abs().square()[:, :, high_k, :].mean()
        loss.backward()
        losses.append(loss.detach())
        assert loss.detach() == pytest.approx(expected_absolute_energy.detach().item())
        assert loss < 1.0e6
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()
        assert torch.count_nonzero(prediction.grad) > 0
        assert prediction.grad.abs().amax() < 1.0e6

    assert losses[0] > losses[1] > losses[2] > 0

    zero = torch.zeros_like(target, requires_grad=True)
    zero_loss = auxiliary_query_losses(
        receivers, receivers, zero, target, time_s, core_margin=0
    ).local_spectrum
    zero_loss.backward()
    assert zero_loss == 0
    assert zero.grad is not None and torch.isfinite(zero.grad).all()


def test_energy_loss_uses_nonuniform_physical_trapezoid_weights():
    time_s = torch.linspace(0.0, 1.0, 160).square()
    target = torch.linspace(0.1, 1.0, 160).view(1, 1, 160)
    prediction = target * 2.0**0.5
    patch = torch.zeros(1, 1, 4, 4, 160)

    losses = auxiliary_query_losses(prediction, target, patch, patch, time_s)

    assert losses.energy.item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), rel=1e-5)


def test_energy_log_ratio_is_scale_invariant_at_tiny_amplitude():
    time_s = torch.linspace(0.0, 1.0, 160).square()
    target = torch.linspace(0.1, 1.0, 160).view(1, 1, 160)
    prediction = target * 2.0**0.5
    patch = torch.zeros(1, 1, 4, 4, 160)

    base = auxiliary_query_losses(prediction, target, patch, patch, time_s).energy
    tiny = auxiliary_query_losses(
        prediction * 1.0e-6, target * 1.0e-6, patch, patch, time_s
    ).energy

    assert tiny.item() == pytest.approx(base.item(), rel=2e-4, abs=1e-6)


def test_zero_target_energy_tracks_hallucination_amplitude_and_gradient():
    time_s = torch.linspace(0.0, 1.0, 160).square()
    target = torch.zeros(1, 1, 160)
    patch = torch.zeros(1, 1, 4, 4, 160)
    losses = []

    for amplitude in (1.0, 1.0e-3, 1.0e-6):
        prediction = torch.full((1, 1, 160), amplitude, requires_grad=True)
        loss = auxiliary_query_losses(
            prediction, target, patch, patch, time_s
        ).energy
        loss.backward()
        losses.append(loss.detach())
        assert loss.detach() == pytest.approx(amplitude**2, rel=1e-5, abs=1e-15)
        assert loss < 1.0e6
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()
        assert torch.count_nonzero(prediction.grad) > 0
        assert prediction.grad.abs().amax() < 1.0e6

    assert losses[0] > losses[1] > losses[2] > 0

    zero = torch.zeros_like(target, requires_grad=True)
    zero_loss = auxiliary_query_losses(zero, target, patch, patch, time_s).energy
    zero_loss.backward()
    assert zero_loss == 0
    assert zero.grad is not None and torch.isfinite(zero.grad).all()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r, p, t: (r[..., :-1], r, p, p, t), "receiver"),
        (lambda r, p, t: (r, r, p[..., :-1], p, t), "patch"),
        (lambda r, p, t: (r, r, p, p, t[:-1]), "time_s"),
        (lambda r, p, t: (r, r, p, p, t.flip(0)), "increasing"),
    ],
)
def test_auxiliary_losses_reject_invalid_shapes_and_time(mutation, message):
    receivers, patches, time_s = _traces_and_patches()
    with pytest.raises(ValueError, match=message):
        auxiliary_query_losses(*mutation(receivers, patches, time_s))


def test_auxiliary_losses_reject_core_margin_that_removes_core():
    receivers, patches, time_s = _traces_and_patches()
    with pytest.raises(ValueError, match="core_margin"):
        auxiliary_query_losses(receivers, receivers, patches, patches, time_s, core_margin=3)


def test_single_pixel_core_has_finite_zero_high_k_loss():
    receivers = torch.ones(1, 1, 160)
    patches = torch.ones(1, 1, 3, 3, 160, requires_grad=True)
    time_s = torch.linspace(0.0, 1.0, 160)

    losses = auxiliary_query_losses(
        receivers, receivers, patches, patches, time_s, core_margin=1
    )
    losses.local_spectrum.backward()

    assert losses.local_spectrum == 0
    assert torch.isfinite(patches.grad).all()
