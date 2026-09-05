"""CPU contracts for the snapshot-only causal wave propagator."""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from saved_time_phase_operator_v4.snapshot_propagator import (
    SnapshotOnlyWavePropagator,
    estimate_snapshot_modal_symbol,
    transfer_pretrained_decoder_stack,
)


def _traveling_mode(*, batch: int = 2, frames: int = 8, z: int = 24, x: int = 28):
    zz = torch.arange(z, dtype=torch.float32).view(z, 1)
    xx = torch.arange(x, dtype=torch.float32).view(1, x)
    spatial = torch.cos(2.0 * math.pi * (2.0 * zz / z + 3.0 * xx / x))
    omega = 0.31
    sequence = torch.stack(
        [torch.cos(torch.tensor(omega * k)) * spatial for k in range(frames)], dim=0
    )
    return sequence[None].repeat(batch, 1, 1, 1), omega


def _model() -> SnapshotOnlyWavePropagator:
    torch.manual_seed(7)
    return SnapshotOnlyWavePropagator(
        minimum_history=4,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
    )


def test_inference_signature_exposes_only_snapshots_and_rollout_length():
    names = tuple(inspect.signature(SnapshotOnlyWavePropagator.forward).parameters)
    assert names == ("self", "wavefield_history", "steps")


def test_requires_four_consecutive_energetic_snapshots():
    model = _model()
    with pytest.raises(ValueError, match="at least 4"):
        model(torch.randn(1, 3, 16, 16), 2)
    with pytest.raises(ValueError, match="not energetic"):
        model(torch.zeros(1, 4, 16, 16), 2)


def test_modal_symbol_recovers_a_single_wave_frequency():
    history, omega = _traveling_mode()
    symbol = estimate_snapshot_modal_symbol(history)
    expected = math.cos(omega)
    energy = torch.fft.rfft2(history, norm="ortho")[:, 1:-1].abs().square().sum(dim=1)
    active = energy > 0.1 * energy.amax(dim=(-2, -1), keepdim=True)
    assert torch.allclose(
        symbol.cosine[active],
        torch.full_like(symbol.cosine[active], expected),
        atol=2.0e-4,
        rtol=2.0e-4,
    )
    assert symbol.context_features.shape == (history.shape[0], 2)
    assert torch.isfinite(symbol.context_features).all()


def test_zero_initialized_closure_is_exact_modal_baseline():
    history, _ = _traveling_mode()
    model = _model().eval()
    with torch.no_grad():
        learned = model(history, 6)
        baseline = model.modal_baseline(history, 6)
    assert torch.equal(learned, baseline)


def test_learned_modal_calibration_starts_damped_and_remains_stable():
    history, _ = _traveling_mode(batch=1)
    model = SnapshotOnlyWavePropagator(
        minimum_history=4,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        learned_modal_calibration=True,
        modal_radius=0.995,
        minimum_modal_radius=0.98,
        maximum_modal_radius=0.9999,
        initial_modal_radius=0.995,
    ).eval()
    symbol = model.infer_symbol(history)
    cosine, radius = model.modal_calibrator(symbol)
    assert torch.allclose(radius, torch.full_like(radius, 0.995), atol=1.0e-7)
    assert torch.allclose(cosine, symbol.cosine, atol=1.0e-7, rtol=1.0e-7)
    assert torch.all(radius < 1.0)
    with torch.no_grad():
        learned = model(history, 40)
        baseline = model.modal_baseline(history, 40)
    assert torch.equal(learned, baseline)
    assert torch.isfinite(learned).all()


def test_calibration_only_rollout_has_live_gradient_and_fixed_baseline():
    history, _ = _traveling_mode(batch=1)
    model = SnapshotOnlyWavePropagator(
        minimum_history=4,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        learned_modal_calibration=True,
    ).train()
    prediction = model.calibrated_modal_rollout(history, 4)
    target = prediction.detach() + 0.01 * torch.randn_like(prediction)
    (prediction - target).square().mean().backward()
    assert model.modal_calibrator.network[-1].weight.grad.abs().sum() > 0.0
    with torch.no_grad():
        fixed = model.modal_baseline(history, 4)
    assert fixed.grad_fn is None


def test_eval_forward_skips_exactly_zero_closure_after_modal_calibration():
    history, _ = _traveling_mode(batch=1)
    model = SnapshotOnlyWavePropagator(
        minimum_history=4,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        learned_modal_calibration=True,
    ).eval()
    with torch.no_grad():
        model.modal_calibrator.network[-1].bias.add_(0.1)
        forward = model(history, 5)
        calibrated = model.calibrated_modal_rollout(history, 5)
        fixed = model.modal_baseline(history, 5)
    assert torch.equal(forward, calibrated)
    assert not torch.equal(forward, fixed)


def test_modal_calibrator_output_receives_gradient_at_warm_start():
    history, _ = _traveling_mode(batch=1)
    model = SnapshotOnlyWavePropagator(
        minimum_history=4,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        learned_modal_calibration=True,
    ).train()
    prediction = model(history, 3)
    target = prediction.detach() + 0.01 * torch.randn_like(prediction)
    (prediction - target).square().mean().backward()
    final = model.modal_calibrator.network[-1]
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()
    assert final.weight.grad.abs().sum() > 0.0


def test_context_feature_calibrator_is_snapshot_conditioned_and_stable():
    history, _ = _traveling_mode(batch=1)
    model = SnapshotOnlyWavePropagator(
        minimum_history=4,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        learned_modal_calibration=True,
        modal_context_features=True,
    )
    symbol = model.infer_symbol(history)
    cosine, radius = model.modal_calibrator(symbol)
    assert model.modal_calibrator.network[0].in_channels == 7
    assert cosine.shape == symbol.cosine.shape
    assert torch.all((0.98 < radius) & (radius < 0.9999))


def test_causal_backtest_gate_uses_only_frames_inside_history():
    history, _ = _traveling_mode(batch=1)
    model = SnapshotOnlyWavePropagator(
        minimum_history=8,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        learned_modal_calibration=True,
        modal_context_features=True,
        causal_backtest_prefix=5,
    ).eval()
    decision = model.causal_backtest_decision(history)
    prediction = model(history, 5)
    assert decision.shape == (1,)
    assert decision.dtype == torch.bool
    assert prediction.shape == (1, 5, history.shape[-2], history.shape[-1])
    assert torch.isfinite(prediction).all()
    with pytest.raises(ValueError, match="causal_backtest_prefix"):
        SnapshotOnlyWavePropagator(
            minimum_history=8,
            learned_modal_calibration=True,
            causal_backtest_prefix=8,
        )


def test_prefix_invariance_and_long_rollout_are_finite():
    history, _ = _traveling_mode(batch=1)
    model = _model().eval()
    with torch.no_grad():
        short = model(history, 5)
        long = model(history, 40)
    assert torch.equal(short, long[:, :5])
    assert torch.isfinite(long).all()
    assert long.abs().max() < 4.0 * history.abs().max()


def test_observed_defect_memory_matches_causal_formula_and_preserves_baseline():
    history, _ = _traveling_mode(batch=1)
    history = history.clone()
    history[:, -2] += 0.02 * torch.randn_like(history[:, -2])
    history[:, -1] += 0.03 * torch.randn_like(history[:, -1])
    model = SnapshotOnlyWavePropagator(
        minimum_history=8,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        observed_defect_memory=True,
        defect_memory_decay=0.55,
        defect_trend_scale=0.75,
    ).eval()
    symbol = model.infer_symbol(history)
    defects = model._observed_modal_defects(
        history, cosine=symbol.cosine, radius=model.modal_radius
    )
    uncorrected = model.modal_baseline(history, 1)
    expected = uncorrected[:, 0] + defects[:, -1] + 0.75 * (
        defects[:, -1] - defects[:, -2]
    )
    with torch.no_grad():
        prediction = model(history, 12)
        short = model(history, 3)
    assert torch.allclose(prediction[:, 0], expected, atol=2.0e-6, rtol=2.0e-6)
    assert torch.equal(short, prediction[:, :3])
    assert not torch.equal(prediction[:, :1], uncorrected)
    assert torch.isfinite(prediction).all()


def test_stable_modal_defect_memory_matches_first_causal_defect_step():
    history, _ = _traveling_mode(batch=1)
    history = history.clone()
    torch.manual_seed(19)
    history[:, -3:] += 0.02 * torch.randn_like(history[:, -3:])
    model = SnapshotOnlyWavePropagator(
        minimum_history=8,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        observed_defect_memory=True,
        defect_memory_mode="stable_modal",
        defect_modal_radius=0.88,
    ).eval()
    symbol = model.infer_symbol(history)
    defects = model._observed_modal_defects(
        history, cosine=symbol.cosine, radius=model.modal_radius
    )
    defect_symbol = estimate_snapshot_modal_symbol(defects, minimum_frames=4)
    expected_defect_spectrum = model._modal_step(
        torch.fft.rfft2(defects[:, -2], norm="ortho"),
        torch.fft.rfft2(defects[:, -1], norm="ortho"),
        defect_symbol.cosine,
        model.defect_modal_radius,
    )
    expected_defect = torch.fft.irfft2(
        expected_defect_spectrum, s=history.shape[-2:], norm="ortho"
    )
    uncorrected = model.modal_baseline(history, 1)[:, 0]
    with torch.no_grad():
        prediction = model(history, 8)
        short = model(history, 3)
    assert torch.allclose(
        prediction[:, 0], uncorrected + expected_defect, atol=2.0e-6, rtol=2.0e-6
    )
    assert torch.equal(short, prediction[:, :3])
    assert torch.isfinite(prediction).all()


def test_stable_modal_defect_high_frequency_damping_starts_after_first_step():
    history, _ = _traveling_mode(batch=1)
    history = history.clone()
    torch.manual_seed(23)
    history[:, -3:] += 0.02 * torch.randn_like(history[:, -3:])
    base = SnapshotOnlyWavePropagator(
        minimum_history=8,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        observed_defect_memory=True,
        defect_memory_mode="stable_modal",
        defect_modal_radius=0.88,
    ).eval()
    multiscale = SnapshotOnlyWavePropagator(
        minimum_history=8,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        observed_defect_memory=True,
        defect_memory_mode="stable_modal",
        defect_modal_radius=0.88,
        defect_high_frequency_radius=0.82,
        defect_high_frequency_cutoff=0.75,
    ).eval()
    with torch.no_grad():
        base_prediction = base(history, 3)
        multiscale_prediction = multiscale(history, 3)
    assert torch.equal(base_prediction[:, 0], multiscale_prediction[:, 0])
    radius = multiscale._high_frequency_modal_radius(
        history.shape[-2:],
        base_radius=0.88,
        high_frequency_radius=0.82,
        cutoff=0.75,
        dtype=history.dtype,
        device=history.device,
    )
    assert radius[0, 0] == pytest.approx(0.88)
    assert radius[history.shape[-2] // 2, -1] == pytest.approx(0.82)
    assert not torch.equal(base_prediction[:, 1:], multiscale_prediction[:, 1:])
    assert torch.isfinite(multiscale_prediction).all()


def test_snapshot_local_wave_blend_preserves_first_step_and_prefix():
    history, _ = _traveling_mode(batch=1)
    base = SnapshotOnlyWavePropagator(
        minimum_history=8,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
    ).eval()
    blended = SnapshotOnlyWavePropagator(
        minimum_history=8,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        local_wave_blend_weight=0.125,
        local_wave_blend_ramp_steps=8,
        local_wave_substeps=8,
        local_wave_pool_size=15,
    ).eval()
    with torch.no_grad():
        base_prediction = base(history, 8)
        prediction = blended(history, 8)
        short = blended(history, 3)
        local = blended._snapshot_local_wave_rollout(history, steps=3)
    assert torch.equal(prediction[:, 0], base_prediction[:, 0])
    assert torch.equal(short, prediction[:, :3])
    assert local.shape == short.shape
    assert not torch.equal(prediction[:, 1:], base_prediction[:, 1:])
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(local).all()


def test_snapshot_local_wave_instance_backtest_is_bounded_and_causal():
    history, _ = _traveling_mode(batch=2)
    model = SnapshotOnlyWavePropagator(
        minimum_history=8,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        learned_modal_calibration=True,
        causal_backtest_prefix=5,
        local_wave_blend_weight=0.125,
        local_wave_instance_backtest=True,
    ).eval()
    with torch.no_grad():
        score = model._local_wave_backtest_score(history)
        weight = model._local_wave_backtest_weight(history)
        prediction = model(history, 4)
    assert score.shape == (history.shape[0],)
    assert weight.shape == (history.shape[0],)
    assert torch.equal(weight, score.clamp(0.0, 0.125))
    assert torch.all((0.0 <= weight) & (weight <= 0.125))
    assert prediction.shape == (2, 4, *history.shape[-2:])
    assert torch.isfinite(prediction).all()


def test_snapshot_local_wave_adaptive_boost_preserves_first_eight_steps():
    history, _ = _traveling_mode(batch=1)
    common = dict(
        minimum_history=8,
        memory_frames=4,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        modal_radius=0.995,
        learned_modal_calibration=True,
        causal_backtest_prefix=5,
        local_wave_blend_weight=0.125,
        local_wave_instance_backtest=True,
    )
    baseline = SnapshotOnlyWavePropagator(**common).eval()
    boosted = SnapshotOnlyWavePropagator(
        **common,
        local_wave_adaptive_boost=True,
        local_wave_adaptive_boost_threshold=0.1,
        local_wave_adaptive_boost_width=0.2,
        local_wave_adaptive_boost_maximum_weight=0.25,
        local_wave_adaptive_boost_start_step=8,
        local_wave_adaptive_boost_ramp_steps=4,
    ).eval()
    boosted.load_state_dict(baseline.state_dict(), strict=True)

    def fixed_score(value):
        return value.new_full((value.shape[0],), 0.3)

    baseline._local_wave_backtest_score = fixed_score
    boosted._local_wave_backtest_score = fixed_score
    with torch.no_grad():
        baseline_prediction = baseline(history, 16)
        boosted_prediction = boosted(history, 16)
    assert torch.equal(boosted_prediction[:, :8], baseline_prediction[:, :8])
    assert not torch.equal(boosted_prediction[:, 8:], baseline_prediction[:, 8:])
    assert torch.equal(boosted(history, 8), boosted_prediction[:, :8])
    assert torch.isfinite(boosted_prediction).all()


@pytest.mark.parametrize(
    ("decay", "trend", "mode", "radius", "high_radius", "cutoff", "message"),
    [
        (-0.1, 0.75, "polynomial", 0.88, None, 0.75, "defect_memory_decay"),
        (1.0, 0.75, "polynomial", 0.88, None, 0.75, "defect_memory_decay"),
        (
            0.55,
            float("inf"),
            "polynomial",
            0.88,
            None,
            0.75,
            "defect_trend_scale",
        ),
        (0.55, 0.75, "free_ar", 0.88, None, 0.75, "defect_memory_mode"),
        (0.55, 0.75, "stable_modal", 0.0, None, 0.75, "defect_modal_radius"),
        (
            0.55,
            0.75,
            "stable_modal",
            0.88,
            0.0,
            0.75,
            "defect_high_frequency_radius",
        ),
        (
            0.55,
            0.75,
            "stable_modal",
            0.88,
            0.82,
            0.0,
            "defect_high_frequency_cutoff",
        ),
    ],
)
def test_observed_defect_memory_configuration_is_bounded(
    decay, trend, mode, radius, high_radius, cutoff, message
):
    with pytest.raises(ValueError, match=message):
        SnapshotOnlyWavePropagator(
            defect_memory_decay=decay,
            defect_trend_scale=trend,
            defect_memory_mode=mode,
            defect_modal_radius=radius,
            defect_high_frequency_radius=high_radius,
            defect_high_frequency_cutoff=cutoff,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"local_wave_blend_weight": -0.1}, "local_wave_blend_weight"),
        ({"local_wave_blend_weight": 1.1}, "local_wave_blend_weight"),
        ({"local_wave_blend_ramp_steps": 0}, "local_wave_blend_ramp_steps"),
        ({"local_wave_substeps": 0}, "local_wave_substeps"),
        ({"local_wave_pool_size": 4}, "local_wave_pool_size"),
        ({"local_wave_regularization": 0.0}, "local_wave_regularization"),
        (
            {"local_wave_adaptive_boost_width": 0.0},
            "local_wave_adaptive_boost_width",
        ),
        (
            {
                "local_wave_blend_weight": 0.25,
                "local_wave_adaptive_boost": True,
                "local_wave_adaptive_boost_maximum_weight": 0.2,
            },
            "local_wave_adaptive_boost_maximum_weight",
        ),
        (
            {"local_wave_adaptive_boost_start_step": 0},
            "local_wave_adaptive_boost_start_step",
        ),
        (
            {"local_wave_adaptive_boost_ramp_steps": 0},
            "local_wave_adaptive_boost_ramp_steps",
        ),
        (
            {"local_wave_adaptive_boost": True},
            "adaptive boost requires",
        ),
        (
            {"local_wave_instance_backtest": True},
            "local wave instance backtest",
        ),
        (
            {
                "local_wave_minimum_coefficient": 1.0,
                "local_wave_maximum_coefficient": 0.5,
            },
            "local wave coefficients",
        ),
    ],
)
def test_snapshot_local_wave_configuration_is_bounded(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SnapshotOnlyWavePropagator(**kwargs)


def test_disabled_adaptive_boost_keeps_large_legacy_blend_configuration_valid():
    model = SnapshotOnlyWavePropagator(
        local_wave_blend_weight=0.5,
        local_wave_adaptive_boost=False,
    )
    assert model.local_wave_blend_weight == pytest.approx(0.5)


def test_neural_closure_projection_receives_gradient_at_warm_start():
    history, _ = _traveling_mode(batch=1)
    model = _model().train()
    prediction = model(history, 3)
    target = prediction.detach() + 0.01 * torch.randn_like(prediction)
    (prediction - target).square().mean().backward()
    assert model.project.weight.grad is not None
    assert torch.isfinite(model.project.weight.grad).all()
    assert model.project.weight.grad.abs().sum() > 0.0


def test_compatible_pretrained_stack_transfer_preserves_modal_warmstart():
    source = _model()
    target = _model()
    with torch.no_grad():
        for parameter in source.closure.parameters():
            parameter.add_(0.25)
    parent = {
        f"dense_decoder.stack.{name}": value.clone()
        for name, value in source.closure.state_dict().items()
    }
    report = transfer_pretrained_decoder_stack(target, parent)
    assert report["modal_warmstart_preserved"] is True
    assert report["transferred_tensors"] == len(source.closure.state_dict())
    for name, value in target.closure.state_dict().items():
        assert torch.equal(value, source.closure.state_dict()[name])
    assert torch.count_nonzero(target.project.weight) == 0
