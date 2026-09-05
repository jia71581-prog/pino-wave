"""CPU contracts for the parent-anchored block-32 corrector."""
import inspect

import numpy as np
import torch

from saved_time_phase_operator_v4.parent_anchored_block import (
    ParentAnchoredBlockCorrector,
    parent_anchored_block_loss,
    render_retained_rfft_frames,
)
from saved_time_phase_operator_v4.parent_anchored_relative_loss import (
    record_energy_squared_loss,
    unbiased_window_weights,
)
from scripts.build_transfer_dg_parent_block32_cache import public_conditioning
from scripts.train_transfer_dg_parent_anchored_block32 import (
    assert_group_disjoint_roles,
    training_window,
)


def _model(*, block_size=8, hard_free_surface=True):
    torch.manual_seed(7)
    return ParentAnchoredBlockCorrector(
        condition_channels=5,
        block_size=block_size,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=2,
        maximum_correction_ratio=0.4,
        boundary_blend_frames=2,
        activation_checkpointing=False,
        hard_free_surface=hard_free_surface,
    )


def _inputs(*, total=23, z=17, x=19):
    torch.manual_seed(11)
    parent = torch.randn(2, total, z, x)
    parent[..., 0, :] = 0.0
    condition = torch.randn(2, 5, z, x)
    return parent, condition


def test_selected_rfft_render_matches_full_irfft_for_odd_time_axis():
    torch.manual_seed(3)
    count = 23
    full = torch.randn(2, count, 7, 9)
    spectrum = torch.fft.rfft(full, dim=1, norm="ortho")
    retained = spectrum[:, :7]
    pairs = torch.stack((retained.real, retained.imag), dim=2)
    indices = torch.tensor([0, 1, 4, 11, 22])
    observed = render_retained_rfft_frames(
        pairs, time_count=count, frame_indices=indices
    )
    padded = torch.zeros_like(spectrum)
    padded[:, :7] = retained
    expected = torch.fft.irfft(padded, n=count, dim=1, norm="ortho")[:, indices]
    torch.testing.assert_close(observed, expected, rtol=2.0e-5, atol=2.0e-5)


def test_selected_rfft_render_matches_full_irfft_for_even_nyquist_axis():
    torch.manual_seed(5)
    count = 24
    full = torch.randn(1, count, 5, 6)
    spectrum = torch.fft.rfft(full, dim=1, norm="ortho")
    pairs = torch.stack((spectrum.real, spectrum.imag), dim=2)
    indices = (0, 3, 12, 23)
    observed = render_retained_rfft_frames(
        pairs, time_count=count, frame_indices=indices
    )
    expected = torch.fft.irfft(spectrum, n=count, dim=1, norm="ortho")[:, indices]
    torch.testing.assert_close(observed, expected, rtol=2.0e-5, atol=2.0e-5)


def test_zero_initialized_rollout_is_exact_parent_noop():
    model = _model().eval()
    parent, condition = _inputs()
    with torch.no_grad():
        prediction = model.rollout(parent, condition)
    assert torch.equal(prediction, parent)


def test_model_api_has_no_target_or_future_truth_argument():
    names = set(inspect.signature(ParentAnchoredBlockCorrector.rollout).parameters)
    assert not names.intersection({"target", "truth", "label", "future_truth"})


def test_block_boundary_prefix_is_invariant_to_later_parent_blocks():
    model = _model().eval()
    parent, condition = _inputs(total=26)
    with torch.no_grad():
        model.output_head[-1].weight.normal_(std=1.0e-3)
        complete = model.rollout(parent, condition)
        prefix = model.rollout(
            parent[:, :18].contiguous(), condition, total_frames=parent.shape[1]
        )
    torch.testing.assert_close(complete[:, :18], prefix, rtol=0.0, atol=1.0e-6)


def test_correction_is_bounded_and_free_surface_is_hard():
    model = _model().eval()
    parent, condition = _inputs(total=10)
    with torch.no_grad():
        model.output_head[-1].bias.fill_(100.0)
        prediction = model.rollout(parent, condition)
    assert torch.isfinite(prediction).all()
    assert torch.count_nonzero(prediction[..., 0, :]) == 0
    parent_scale = parent[:, :10].square().mean((1, 2, 3)).sqrt()
    correction_scale = (prediction - parent).square().mean((1, 2, 3)).sqrt()
    assert torch.all(correction_scale <= 0.41 * parent_scale)


def test_zero_init_loss_reaches_head_then_open_head_reaches_backbone():
    model = _model(hard_free_surface=False).train()
    parent, condition = _inputs(total=18)
    target = parent + 0.1 * torch.randn_like(parent)
    prediction = model.rollout(parent, condition)
    loss, metrics = parent_anchored_block_loss(
        prediction[:, 2:], target[:, 2:], parent[:, 2:]
    )
    loss.backward()
    head_gradient = model.output_head[-1].weight.grad
    assert head_gradient is not None and torch.count_nonzero(head_gradient) > 0
    assert torch.isfinite(head_gradient).all()
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not nonfinite, f"non-finite zero-init gradients: {nonfinite}"
    assert all(torch.isfinite(value) for value in metrics.values())

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.output_head[-1].weight.normal_(std=1.0e-3)
    prediction = model.rollout(parent, condition)
    parent_anchored_block_loss(
        prediction[:, 2:], target[:, 2:], parent[:, 2:]
    )[0].backward()
    gradient = model.input_projection[0].weight.grad
    assert gradient is not None and torch.count_nonzero(gradient) > 0
    assert torch.isfinite(gradient).all()


def test_optional_two_observation_history_only_changes_future_through_state():
    model = _model().eval()
    parent, condition = _inputs(total=18)
    observed = parent[:, :2].clone()
    observed[:, 1] += 0.25
    observed[..., 0, :] = 0.0
    with torch.no_grad():
        model.output_head[-1].weight.normal_(std=1.0e-3)
        prediction = model.rollout(parent, condition, initial_history=observed)
    assert torch.equal(prediction[:, :2], observed)
    assert not torch.equal(prediction[:, 2:], parent[:, 2:])


def test_public_conditioning_uses_only_static_medium_and_source_inputs():
    static = {
        "medium": np.zeros((16, 221, 241), dtype=np.float32),
        "source_map": np.ones((201, 201), dtype=np.float32),
        "travel": np.linspace(0.0, 1.0, 201 * 201, dtype=np.float32).reshape(201, 201),
        "parameters": np.asarray((1000.0, 500.0, 20.0, 0.1), dtype=np.float32),
    }
    condition = public_conditioning(static)
    assert condition.shape == (9, 201, 201)
    assert np.isfinite(condition).all()
    assert np.all(condition[3] == 1.0)


def test_training_window_uses_detached_generated_history_and_exact_truth_slice():
    torch.manual_seed(17)
    count, retained, z, x = 50, 10, 11, 13
    full = torch.randn(1, count, z, x)
    spectrum = torch.fft.rfft(full, dim=1, norm="ortho")[:, :retained]
    coefficients = torch.stack((spectrum.real, spectrum.imag), dim=2)
    parent = render_retained_rfft_frames(
        coefficients, time_count=count, frame_indices=torch.arange(count)
    )

    class FakeCache:
        time_count = count

        def truth(self, _position, start, stop, device):
            return (parent[:, start:stop] + 0.05).to(device)

    model = _model(block_size=8, hard_free_surface=False).train()
    condition = torch.randn(1, 5, z, x)
    prediction, target, anchor = training_window(
        model,
        coefficients,
        condition,
        FakeCache(),
        0,
        loss_start=10,
        rollout_blocks=2,
    )
    assert prediction.shape == target.shape == anchor.shape == (1, 16, z, x)
    torch.testing.assert_close(prediction, anchor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(target, anchor + 0.05, rtol=0.0, atol=2.0e-6)


def test_group_disjoint_audit_allows_multi_source_inside_one_role():
    assert_group_disjoint_roles(
        ["medium-a", "medium-a", "medium-b"],
        ["fit", "fit", "calibration"],
    )
    try:
        assert_group_disjoint_roles(
            ["medium-a", "medium-a"], ["fit", "confirmation"]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("cross-role group leakage was not rejected")


def test_unbiased_window_squared_relative_matches_complete_future_mean():
    torch.manual_seed(23)
    total, first, block, rollout_blocks = 18, 2, 4, 2
    target = torch.randn(1, total - first, 3, 5)
    prediction = target + 0.2 * torch.randn_like(target)
    parent = target + 0.3 * torch.randn_like(target)
    full_energy = float(target.double().square().sum())
    full_delta_energy = float((target[:, 1:] - target[:, :-1]).double().square().sum())
    estimates = []
    starts = list(range(first, total, block))
    for start in starts:
        stop = min(start + rollout_blocks * block, total)
        local = slice(start - first, stop - first)
        frame_weights, delta_weights = unbiased_window_weights(
            loss_start=start,
            sample_length=stop - start,
            time_count=total,
            block_size=block,
            rollout_blocks=rollout_blocks,
        )
        _, metrics = record_energy_squared_loss(
            prediction[:, local],
            target[:, local],
            parent[:, local],
            frame_weights=frame_weights,
            delta_weights=delta_weights,
            full_target_energy=full_energy,
            full_target_delta_energy=full_delta_energy,
            derivative_weight=0.0,
            spectral_weight=0.0,
            nonworse_weight=0.0,
            correction_weight=0.0,
        )
        estimates.append(metrics["record_relative_l2_squared"])
    observed = torch.stack(estimates).mean()
    expected = (prediction - target).double().square().sum() / full_energy
    torch.testing.assert_close(observed.double(), expected, rtol=1.0e-6, atol=1.0e-8)


def test_record_energy_loss_has_finite_zero_correction_gradient():
    torch.manual_seed(29)
    target = torch.randn(1, 8, 5, 7)
    parent = target + 0.3 * torch.randn_like(target)
    direction = torch.randn_like(target)
    gate = torch.nn.Parameter(torch.tensor(0.0))
    prediction = parent + gate * direction
    frame_weights = torch.ones(8)
    delta_weights = torch.ones(7)
    loss, metrics = record_energy_squared_loss(
        prediction,
        target,
        parent,
        frame_weights=frame_weights,
        delta_weights=delta_weights,
        full_target_energy=float(target.double().square().sum()),
        full_target_delta_energy=float(
            (target[:, 1:] - target[:, :-1]).double().square().sum()
        ),
    )
    loss.backward()
    assert gate.grad is not None and torch.isfinite(gate.grad)
    assert torch.count_nonzero(gate.grad) > 0
    assert all(torch.isfinite(value) for value in metrics.values())
