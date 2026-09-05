from __future__ import annotations

import torch

from scripts.train_pa_cora_p1b_curriculum import (
    curriculum_slot_weights,
    robust_multiphase_loss,
)


def test_curriculum_is_onset_first_then_three_phase():
    assert curriculum_slot_weights(1) == (1.0, 0.0, 0.0)
    assert curriculum_slot_weights(2) == (1.0, 0.0, 0.0)
    assert curriculum_slot_weights(3) == (0.7, 0.3, 0.0)
    assert curriculum_slot_weights(5) == (0.7, 0.3, 0.0)
    assert curriculum_slot_weights(6) == (0.6, 0.25, 0.15)
    assert curriculum_slot_weights(10) == (0.6, 0.25, 0.15)


def test_robust_multiphase_loss_is_zero_for_exact_prediction():
    target = torch.randn(2, 56, 1, 5, 4)
    onset_norm = target.flatten(1).norm(dim=1)
    loss, report = robust_multiphase_loss(target, target, onset_norm)
    assert torch.equal(loss, torch.zeros_like(loss))
    assert report["target_floor_active_fraction"] == 0.0


def test_onset_floor_controls_near_zero_window_gradient():
    target = torch.full((1, 56, 1, 5, 4), 1.0e-5)
    prediction = torch.zeros_like(target, requires_grad=True)
    onset_norm = torch.tensor([100.0])
    loss, report = robust_multiphase_loss(
        prediction,
        target,
        onset_norm,
        denominator_floor_fraction=0.25,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert report["target_floor_active_fraction"] == 1.0
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.norm()) < 1.0


def test_robust_loss_has_finite_gradient_on_normal_window():
    target = torch.randn(2, 56, 1, 7, 6)
    prediction = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    onset_norm = target.flatten(1).norm(dim=1)
    loss, report = robust_multiphase_loss(prediction, target, onset_norm)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert float(report["relative_l2_robust"]) > 0.0
