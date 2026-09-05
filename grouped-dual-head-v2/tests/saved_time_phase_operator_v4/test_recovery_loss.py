import pytest
import torch

from saved_time_phase_operator_v4.losses import (
    apply_hard_causality,
    frame_relative_l2,
    pml_interface_residual_loss,
    residual_recovery_loss,
    source_causality_onset_s,
    family_router_loss,
    family_route_targets,
    zero_initial_condition_loss,
)


def test_frame_relative_l2_weights_every_frame_equally():
    # One early frame with large energy, one late frame with tiny energy.
    target = torch.zeros(1, 2, 2, 2)
    target[0, 0] = 10.0  # early high-amplitude frame
    target[0, 1] = 0.1  # late low-amplitude frame
    prediction = target.clone()
    prediction[0, 1] = 0.0  # miss the whole late frame

    per_record = (prediction.flatten(1) - target.flatten(1)).norm() / target.flatten(1).norm()
    per_frame = frame_relative_l2(prediction, target)

    # Missing the low-energy late frame is nearly invisible per record, but the
    # per-frame objective still charges the full unit error for that frame,
    # so it averages ~0.5 across the two frames.
    assert per_record.item() < 0.02
    assert per_frame.item() == pytest.approx(0.5, rel=1.0e-4)


def test_frame_relative_l2_floor_bounds_near_zero_frames():
    target = torch.zeros(1, 2, 2, 2)
    target[0, 0] = 1.0
    target[0, 1] = 1.0e-6  # effectively a pre-onset frame
    prediction = torch.zeros_like(target)
    prediction[0, 0] = 1.0  # fit the energetic frame exactly

    floored = frame_relative_l2(prediction, target, energy_floor_fraction=0.1)
    unfloored = frame_relative_l2(prediction, target, energy_floor_fraction=0.0)

    # Without a floor the near-zero frame divides by ~0 and reads ~1.0; the
    # 10% peak floor caps that frame's contribution well below 1.0.
    assert unfloored.item() == pytest.approx(0.5, rel=1.0e-3)
    assert floored.item() < 0.11


def test_frame_relative_l2_rejects_invalid_floor():
    value = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="frame relative L2 energy floor"):
        frame_relative_l2(value, value, energy_floor_fraction=1.5)


def test_recovery_loss_per_frame_frame_matches_frame_relative_l2():
    target = torch.zeros(1, 2, 2, 2)
    target[0, 0] = 4.0
    target[0, 1] = 0.2
    coarse = torch.zeros_like(target)
    prediction = target.clone()
    prediction[0, 1] = 0.0

    result = residual_recovery_loss(
        prediction,
        coarse,
        target,
        time_indices=torch.tensor([[5, 6]]),
        delta_weight=0.0,
        temporal_weight=0.0,
        gradient_weight=0.0,
        spectrum_weight=0.0,
        per_frame_frame=True,
        frame_energy_floor_fraction=0.0,
    )

    assert result.frame.item() == pytest.approx(
        frame_relative_l2(prediction, target).item(), rel=1.0e-5
    )


def test_relative_l2_only_recovery_skips_disabled_components():
    target = torch.randn(2, 4, 5, 6)
    prediction = (target + 0.1 * torch.randn_like(target)).requires_grad_()
    coarse = torch.zeros_like(target)

    result = residual_recovery_loss(
        prediction,
        coarse,
        target,
        time_indices=torch.arange(4)[None, :].expand(2, 4),
        frame_weight=1.0,
        delta_weight=0.0,
        temporal_weight=0.0,
        gradient_weight=0.0,
        spectrum_weight=0.0,
        per_frame_frame=False,
    )
    result.total.backward()

    assert result.total.item() == pytest.approx(result.frame.item(), rel=1.0e-7)
    assert result.delta.item() == 0.0
    assert result.temporal.item() == 0.0
    assert result.gradient.item() == 0.0
    assert result.spectrum.item() == 0.0
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_pml_interface_loss_targets_three_saved_crop_sides_and_has_gradients():
    times = torch.arange(5, dtype=torch.float32)[None, :] * 0.0025
    velocity = torch.full((1, 25, 25), 1500.0)
    interior = torch.zeros(1, 5, 25, 25)
    interior[..., 12, 12] = 1.0
    boundary = torch.zeros_like(interior)
    boundary[..., 8:16, 0] = 1.0
    boundary.requires_grad_()

    interior_loss = pml_interface_residual_loss(
        interior, velocity, times, dz_m=10.0, dx_m=10.0, boundary_band_cells=4
    )
    boundary_loss = pml_interface_residual_loss(
        boundary, velocity, times, dz_m=10.0, dx_m=10.0, boundary_band_cells=4
    )
    boundary_loss.backward()

    assert float(interior_loss) == 0.0
    assert float(boundary_loss) > 0.0
    assert boundary.grad is not None and torch.isfinite(boundary.grad).all()


def test_zero_initial_condition_penalizes_pressure_and_first_step_change():
    reference = torch.ones(2)
    exact = torch.zeros(2, 2, 5, 6)
    pressure = exact.clone()
    pressure[:, 0] = 1.0
    pressure.requires_grad_()

    exact_loss = zero_initial_condition_loss(exact, reference_rms=reference)
    pressure_loss = zero_initial_condition_loss(
        pressure, reference_rms=reference, first_step_weight=1.0
    )
    pressure_loss.backward()

    assert float(exact_loss) == 0.0
    assert float(pressure_loss) > 0.0
    assert pressure.grad is not None and torch.isfinite(pressure.grad).all()


def test_recovery_loss_adds_weighted_pml_interface_term():
    torch.manual_seed(917)
    prediction = torch.randn(1, 5, 17, 19, requires_grad=True)
    target = torch.randn_like(prediction)
    coarse = torch.zeros_like(prediction)
    velocity = torch.full((1, 17, 19), 1500.0)
    times = torch.arange(5, dtype=torch.float32)[None, :] * 0.0025
    common = dict(
        time_indices=torch.arange(5)[None, :],
        delta_weight=0.0,
        temporal_weight=0.0,
        gradient_weight=0.0,
        spectrum_weight=0.0,
        velocity_mps=velocity,
        time_values_s=times,
        pde_dz_m=10.0,
        pde_dx_m=10.0,
    )
    base = residual_recovery_loss(prediction, coarse, target, **common)
    weight = 0.03
    constrained = residual_recovery_loss(
        prediction,
        coarse,
        target,
        pml_weight=weight,
        pml_boundary_band_cells=3,
        **common,
    )

    assert constrained.pml is not None and float(constrained.pml) > 0.0
    assert constrained.total.item() == pytest.approx(
        base.total.item() + weight * float(constrained.pml), rel=1.0e-5
    )


def test_residual_recovery_loss_is_zero_for_an_exact_prediction():
    target = torch.arange(5, dtype=torch.float32)[None, :, None, None].expand(1, 5, 3, 4)
    coarse = target - 0.5
    indices = torch.arange(5)[None]

    result = residual_recovery_loss(
        target,
        coarse,
        target,
        time_indices=indices,
        delta_weight=0.5,
        temporal_weight=0.1,
        gradient_weight=0.0,
        spectrum_weight=0.0,
    )

    assert result.total.item() == pytest.approx(0.0, abs=1.0e-7)
    assert result.delta.item() == pytest.approx(0.0, abs=1.0e-7)
    assert result.temporal.item() == pytest.approx(0.0, abs=1.0e-7)


def test_delta_term_penalizes_an_inactive_residual_branch():
    target = torch.ones(1, 4, 3, 3)
    coarse = torch.zeros_like(target)
    indices = torch.tensor([[10, 11, 30, 31]])

    result = residual_recovery_loss(
        coarse,
        coarse,
        target,
        time_indices=indices,
        delta_weight=0.5,
        temporal_weight=0.1,
        gradient_weight=0.0,
        spectrum_weight=0.0,
    )

    assert result.frame.item() == pytest.approx(1.0)
    assert result.delta.item() == pytest.approx(1.0)
    assert result.total.item() >= 1.5


def test_delta_energy_floor_prevents_tiny_true_residuals_from_dominating():
    target = torch.ones(1, 2, 2, 2)
    coarse = target - 0.01
    prediction = coarse.clone()

    result = residual_recovery_loss(
        prediction,
        coarse,
        target,
        time_indices=torch.tensor([[2, 3]]),
        delta_weight=1.0,
        temporal_weight=0.0,
        gradient_weight=0.0,
        spectrum_weight=0.0,
        delta_energy_floor_fraction=0.1,
    )

    # The inactive correction is 1% of target energy.  A 10% floor therefore
    # gives a 0.1 penalty instead of the legacy unit penalty.
    assert result.delta.item() == pytest.approx(0.1, rel=1.0e-5)


def test_delta_energy_floor_rejects_invalid_fraction():
    value = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="delta energy floor"):
        residual_recovery_loss(
            value,
            value,
            value,
            time_indices=torch.tensor([[0]]),
            delta_weight=1.0,
            temporal_weight=0.0,
            gradient_weight=0.0,
            spectrum_weight=0.0,
            delta_energy_floor_fraction=1.1,
        )


def test_recovery_loss_can_optimize_only_the_scattering_correction():
    target = torch.ones(1, 2, 2, 2)
    coarse = target - 0.04
    prediction = coarse.clone()

    result = residual_recovery_loss(
        prediction,
        coarse,
        target,
        time_indices=torch.tensor([[2, 3]]),
        frame_weight=0.0,
        delta_weight=1.0,
        temporal_weight=0.0,
        gradient_weight=0.0,
        spectrum_weight=0.0,
        delta_energy_floor_fraction=0.0,
    )

    assert result.delta.item() == pytest.approx(1.0, rel=1.0e-5)
    assert result.total.item() == pytest.approx(result.delta.item(), rel=1.0e-6)


def test_recovery_loss_rejects_invalid_frame_weight():
    value = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="recovery loss weights"):
        residual_recovery_loss(
            value,
            value,
            value,
            time_indices=torch.tensor([[0]]),
            frame_weight=-1.0,
            delta_weight=1.0,
            temporal_weight=0.0,
            gradient_weight=0.0,
            spectrum_weight=0.0,
        )


def test_target_delta_reference_does_not_cancel_a_trainable_coarse_branch():
    prediction = torch.zeros(1, 2, 2, 2, requires_grad=True)
    target = torch.ones_like(prediction)

    result = residual_recovery_loss(
        prediction,
        prediction,
        target,
        time_indices=torch.tensor([[2, 3]]),
        frame_weight=0.0,
        delta_weight=1.0,
        delta_reference="target",
        temporal_weight=0.0,
        gradient_weight=0.0,
        spectrum_weight=0.0,
    )
    result.total.backward()

    assert prediction.grad is not None
    assert prediction.grad.norm() > 0.0


def test_recovery_loss_rejects_unknown_delta_reference():
    value = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="delta reference"):
        residual_recovery_loss(
            value,
            value,
            value,
            time_indices=torch.tensor([[0]]),
            delta_weight=1.0,
            delta_reference="unknown",
            temporal_weight=0.0,
            gradient_weight=0.0,
            spectrum_weight=0.0,
        )


def test_global_target_delta_matches_pooled_scattering_relative_l2():
    target = torch.zeros(2, 1, 1, 2)
    target[1] = 2.0
    prediction = torch.zeros_like(target)
    prediction[0] = 1.0

    result = residual_recovery_loss(
        prediction,
        torch.zeros_like(prediction),
        target,
        time_indices=torch.tensor([[0], [0]]),
        frame_weight=0.0,
        delta_weight=1.0,
        delta_reference="target",
        delta_reduction="global",
        temporal_weight=0.0,
        gradient_weight=0.0,
        spectrum_weight=0.0,
    )
    expected = (prediction - target).norm() / target.norm()

    assert result.delta.item() == pytest.approx(expected.item(), rel=1.0e-6)


def test_recovery_loss_rejects_unknown_delta_reduction():
    value = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="delta reduction"):
        residual_recovery_loss(
            value,
            value,
            value,
            time_indices=torch.tensor([[0]]),
            delta_weight=1.0,
            delta_reduction="unknown",
            temporal_weight=0.0,
            gradient_weight=0.0,
            spectrum_weight=0.0,
        )


def test_global_squared_delta_streams_across_single_record_pieces():
    target = torch.zeros(2, 1, 1, 2)
    target[0] = 1.0
    target[1] = 2.0
    prediction = torch.zeros_like(target)
    prediction[0, 0, 0, 0] = 0.5
    denominator = float(target.double().square().sum())
    streamed = 0.0
    for row in range(2):
        result = residual_recovery_loss(
            prediction[row : row + 1],
            torch.zeros_like(prediction[row : row + 1]),
            target[row : row + 1],
            time_indices=torch.tensor([[0]]),
            frame_weight=0.0,
            delta_weight=1.0,
            delta_reference="target",
            delta_reduction="global_squared",
            delta_global_target_square=denominator,
            delta_piece_weight=0.5,
            temporal_weight=0.0,
            gradient_weight=0.0,
            spectrum_weight=0.0,
        )
        streamed += 0.5 * result.delta.item()
    expected = float((prediction.double() - target.double()).square().sum()) / denominator

    assert streamed == pytest.approx(expected, rel=1.0e-6)


def test_hard_causality_zeros_only_pre_onset_frames():
    field = torch.ones(2, 4, 2, 2)
    times = torch.tensor([[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]])
    onset = torch.tensor([0.15, 0.05])

    masked = apply_hard_causality(field, times, onset)

    assert torch.count_nonzero(masked[0, :2]) == 0
    assert torch.count_nonzero(masked[0, 2:]) == 8
    assert torch.count_nonzero(masked[1, :1]) == 0
    assert torch.count_nonzero(masked[1, 1:]) == 12


def test_source_causality_onset_uses_cycles_before_ricker_peak():
    source = torch.tensor(
        [
            [0.2, 0.1, 20.0, 0.075, 1.0],
            [0.4, 0.1, 10.0, 0.150, 1.0],
        ]
    )

    onset = source_causality_onset_s(source, lead_cycles=1.0)

    assert torch.allclose(onset, torch.tensor([0.025, 0.050]))
    with pytest.raises(ValueError, match="lead cycles"):
        source_causality_onset_s(source, lead_cycles=-1.0)
    invalid = source.clone()
    invalid[0, 2] = 0.0
    with pytest.raises(ValueError, match="frequency"):
        source_causality_onset_s(invalid, lead_cycles=1.0)


def test_router_loss_uses_one_label_per_medium_and_reports_probabilities():
    logits = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]])
    value, report = family_router_loss(
        logits,
        ("uniform", "uniform", "layered", "layered", "marmousi", "marmousi"),
        record_to_medium=torch.tensor([0, 0, 1, 1, 2, 2]),
    )

    assert value.item() < 1.0e-3
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["entropy"] < 0.01
    assert report["route_probability_uniform"] == pytest.approx(1.0 / 3.0, rel=1.0e-3)
    assert report["route_probability_layered"] == pytest.approx(1.0 / 3.0, rel=1.0e-3)
    assert report["route_probability_marmousi"] == pytest.approx(1.0 / 3.0, rel=1.0e-3)


def test_router_loss_rejects_conflicting_labels_for_one_medium():
    with pytest.raises(ValueError, match="sharing a medium"):
        family_router_loss(
            torch.zeros(1, 3),
            ("uniform", "layered"),
            record_to_medium=torch.tensor([0, 0]),
        )


def test_family_route_targets_collapse_record_labels_to_unique_media():
    targets = family_route_targets(
        ("uniform", "uniform", "marmousi", "layered", "layered"),
        medium_count=3,
        record_to_medium=torch.tensor([0, 0, 2, 1, 1]),
        device=torch.device("cpu"),
    )

    assert torch.equal(targets, torch.tensor([0, 1, 2]))


def test_recovery_loss_pde_weight_zero_is_backward_compatible():
    torch.manual_seed(0)
    target = torch.randn(2, 6, 8, 8)
    coarse = target + 0.1 * torch.randn_like(target)
    pred = target + 0.2 * torch.randn_like(target)
    indices = torch.arange(6)[None].expand(2, 6).contiguous()
    kw = dict(time_indices=indices, delta_weight=0.5, temporal_weight=0.1,
              gradient_weight=0.1, spectrum_weight=0.1)
    base = residual_recovery_loss(pred, coarse, target, **kw)
    # explicit pde_weight=0.0 (no velocity/time) must be bit-identical + pde==0
    withz = residual_recovery_loss(pred, coarse, target, pde_weight=0.0, **kw)
    assert withz.total.item() == base.total.item()
    assert float(withz.pde) == 0.0


def test_recovery_loss_pde_term_adds_when_enabled():
    torch.manual_seed(1)
    R, T, Z, X = 2, 6, 12, 12
    target = torch.randn(R, T, Z, X)
    coarse = target.clone()
    pred = target + 0.3 * torch.randn_like(target)  # non-solution field
    indices = torch.arange(T)[None].expand(R, T).contiguous()
    vel = torch.full((R, Z, X), 1500.0)
    dt = 0.02
    time_values = (torch.arange(T).float() * dt)[None].expand(R, T).contiguous()
    kw = dict(time_indices=indices, delta_weight=0.5, temporal_weight=0.1,
              gradient_weight=0.1, spectrum_weight=0.1)
    base = residual_recovery_loss(pred, coarse, target, **kw)
    pw = 0.05
    withp = residual_recovery_loss(
        pred, coarse, target, pde_weight=pw, velocity_mps=vel,
        time_values_s=time_values, pde_dz_m=100.0, pde_dx_m=100.0, **kw)
    assert float(withp.pde) > 0.0                                   # random field violates the PDE
    assert withp.total.item() == pytest.approx(base.total.item() + pw * float(withp.pde), rel=1e-5)


def test_recovery_loss_pde_weight_requires_physics_inputs():
    target = torch.randn(1, 4, 6, 6)
    indices = torch.arange(4)[None]
    with pytest.raises(ValueError):
        residual_recovery_loss(
            target, target, target, time_indices=indices, delta_weight=0.5,
            temporal_weight=0.1, gradient_weight=0.0, spectrum_weight=0.0,
            pde_weight=0.1)  # missing velocity/time/spacing
