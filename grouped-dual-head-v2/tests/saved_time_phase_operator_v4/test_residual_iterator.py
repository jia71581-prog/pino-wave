from __future__ import annotations

import pytest
import torch

from saved_time_phase_operator_v4.instance_adaptation.bridge import _laplacian8
from saved_time_phase_operator_v4.instance_adaptation.residual_iterator import (
    MultiScaleResidualCorrector,
    _uniform_saved_dt,
    align_lwc84_residual,
    refine_with_residual_iterator,
    residual_iterator_training_loss,
    unroll_residual_iterator,
)
from saved_time_phase_operator_v4.instance_adaptation.losses import lwc84_residual
from scripts.train_residual_iterator import _weighted_epoch_indices


def _consistent_case():
    torch.manual_seed(4)
    dt = 0.0025
    velocity = torch.full((1, 1, 17, 17), 1500.0)
    p_previous = torch.randn(1, 17, 17) * 1.0e-4
    p_current = p_previous.clone()
    frames = [p_previous, p_current]

    def operator(field):
        return velocity[:, 0].square() * _laplacian8(field, dx_m=10.0, dz_m=10.0)

    for _ in range(6):
        acceleration = operator(p_current)
        p_next = (
            2.0 * p_current - p_previous + dt**2 * acceleration
            + dt**4 * operator(acceleration) / 12.0
        )
        frames.append(p_next)
        p_previous, p_current = p_current, p_next
    truth = torch.stack(frames, dim=1)
    time_s = torch.arange(truth.shape[1], dtype=torch.float32) * dt
    return truth, velocity, time_s


def test_zero_initialized_corrector_is_exact_noop():
    truth, velocity, time_s = _consistent_case()
    residual = lwc84_residual(
        truth, velocity, dt=0.0025, dx=10.0, dz=10.0,
        observed_indices=(0, 1),
    )
    aligned = align_lwc84_residual(residual, truth, (0, 1))
    corrector = MultiScaleResidualCorrector(width=8)
    correction = corrector(truth, truth, velocity, aligned, time_s)
    assert torch.count_nonzero(correction) == 0
    result = refine_with_residual_iterator(
        corrector, truth, velocity, time_s, (0, 1), iterations=2,
    )
    assert torch.equal(result.field, truth)
    assert result.accepted_steps == 0
    assert result.stopped_reason == "no_contracting_step"


def test_residual_alignment_preserves_absolute_center_indices():
    truth, velocity, _ = _consistent_case()
    residual = lwc84_residual(
        truth, velocity, dt=0.0025, dx=10.0, dz=10.0,
        observed_indices=(0, 1),
    )
    aligned = align_lwc84_residual(residual, truth, (0, 1))
    assert torch.equal(aligned[:, 2 : 2 + residual.shape[1], 4:-4, 4:-4], residual)
    assert torch.count_nonzero(aligned[:, :2]) == 0


def test_line_search_accepts_contracting_or_rolls_back_harmful_updates():
    truth, velocity, time_s = _consistent_case()
    starter = truth.clone()
    noise = torch.randn_like(starter) * 1.0e-4
    noise[:, :2] = 0.0
    starter = starter + noise

    class OracleCorrection(torch.nn.Module):
        def forward(self, frozen, current, velocity_mps, residual, times):
            del frozen, velocity_mps, residual, times
            return truth - current

    improved = refine_with_residual_iterator(
        OracleCorrection(), starter, velocity, time_s, (0, 1), iterations=2,
        step_candidates=(0.0, 1.0), minimum_relative_improvement=0.0,
        minimum_energy_ratio=0.1, maximum_energy_ratio=10.0,
    )
    assert improved.accepted_steps >= 1
    assert improved.residual_history[0][-1] < improved.residual_history[0][0]

    class HarmfulCorrection(torch.nn.Module):
        def forward(self, frozen, current, velocity_mps, residual, times):
            del frozen, velocity_mps, residual, times
            return current * 100.0

    rejected = refine_with_residual_iterator(
        HarmfulCorrection(), starter, velocity, time_s, (0, 1), iterations=2,
    )
    assert rejected.accepted_steps == 0
    assert torch.equal(rejected.field, starter)


def test_unrolled_training_objective_is_finite_and_differentiable():
    truth, velocity, time_s = _consistent_case()
    starter = truth + torch.randn_like(truth) * 2.0e-5
    corrector = MultiScaleResidualCorrector(width=8)
    states, residuals = unroll_residual_iterator(
        corrector, starter, velocity, time_s, (0, 1), iterations=2,
    )
    terms = residual_iterator_training_loss(states, residuals, truth)
    assert all(torch.isfinite(value) for value in terms.values())
    terms["total"].backward()
    assert corrector.output.weight.grad is not None
    assert torch.isfinite(corrector.output.weight.grad).all()


def test_unroll_accepts_vectorized_records_with_shifted_uniform_time_axes():
    truth, velocity, time_s = _consistent_case()
    starter = torch.cat((truth, truth * 1.1), dim=0)
    velocity = velocity.expand(2, -1, -1, -1).clone()
    batched_time = torch.stack((time_s, time_s + 0.5), dim=0)
    corrector = MultiScaleResidualCorrector(width=8)
    states, residuals = unroll_residual_iterator(
        corrector, starter, velocity, batched_time, (-1, 0),
        iterations=1, frozen_time_indices=(),
    )
    assert states[-1].shape == starter.shape
    assert residuals[-1].shape[0] == 2


def test_uniform_saved_dt_accepts_float32_quantization_but_rejects_real_gap():
    late_times = 0.9 + torch.arange(16, dtype=torch.float32) * 0.0025
    assert _uniform_saved_dt(late_times) == pytest.approx(0.0025, rel=5.0e-5)

    skipped = late_times.clone()
    skipped[8:] += 0.001
    with pytest.raises(ValueError, match="uniform saved-time spacing"):
        _uniform_saved_dt(skipped)


def test_family_sample_weights_scale_batch_one_equivalently():
    truth, velocity, time_s = _consistent_case()
    starter = torch.cat(
        (
            truth + torch.randn_like(truth) * 2.0e-5,
            truth + torch.randn_like(truth) * 8.0e-5,
        ),
        dim=0,
    )
    batched_truth = truth.expand(2, -1, -1, -1).clone()
    batched_velocity = velocity.expand(2, -1, -1, -1).clone()
    batched_time = time_s.expand(2, -1).clone()
    corrector = MultiScaleResidualCorrector(width=8)
    states, residuals = unroll_residual_iterator(
        corrector,
        starter,
        batched_velocity,
        batched_time,
        (0, 1),
        iterations=1,
    )
    weighted = residual_iterator_training_loss(
        states,
        residuals,
        batched_truth,
        sample_weights=torch.tensor([2.0, 0.0]),
    )
    first_only = residual_iterator_training_loss(
        [value[:1] for value in states],
        [value[:1] for value in residuals],
        truth,
    )
    for name in weighted:
        assert float(weighted[name]) == pytest.approx(float(first_only[name]), rel=1.0e-5)


def test_family_weighted_schedule_preserves_coverage_and_targets_hard_families():
    families = ("uniform", "layered", "marmousi")
    cached = [
        {"sample_index": index, "medium_type": family}
        for index, family in enumerate(families * 2)
    ]
    weights = {"uniform": 0.5, "layered": 1.25, "marmousi": 1.25}
    counts = {family: 0 for family in families}
    for epoch in range(64):
        schedule = _weighted_epoch_indices(
            cached, family_weights=weights, seed=372, epoch=epoch
        )
        drawn = [str(cached[index]["medium_type"]) for index in schedule]
        assert len(drawn) == len(cached)
        assert set(drawn) == set(families)
        for family in drawn:
            counts[family] += 1
    assert counts["layered"] > counts["uniform"]
    assert counts["marmousi"] > counts["uniform"]
