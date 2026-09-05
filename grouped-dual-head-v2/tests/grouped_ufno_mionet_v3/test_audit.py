from __future__ import annotations

import pytest
import torch

from grouped_ufno_mionet_v3.training.audit import (
    assert_report_families,
    audit_required_gradients,
    energy_ratio,
    optimal_amplitude_scale,
    radial_centroid_displacement_m,
    zero_prediction_relative_l2,
)


def test_radial_centroid_reports_wavefront_displacement_in_metres():
    x = torch.linspace(0.0, 100.0, 11)
    z = torch.linspace(0.0, 100.0, 11)
    target = torch.zeros(1, 1, 11, 11)
    prediction = torch.zeros_like(target)
    target[0, 0, 5, 7] = 1.0
    prediction[0, 0, 5, 8] = 1.0
    displacement, valid = radial_centroid_displacement_m(
        prediction,
        target,
        source_xy_m=torch.tensor([[50.0, 50.0]]),
        x_m=x,
        z_m=z,
    )
    assert valid.tolist() == [[True]]
    assert displacement.item() == pytest.approx(10.0)


def test_energy_amplitude_and_zero_baselines_are_explicit():
    target = torch.tensor([1.0, -2.0, 3.0])
    prediction = 0.5 * target
    assert energy_ratio(prediction, target).item() == pytest.approx(0.25)
    assert optimal_amplitude_scale(prediction, target).item() == pytest.approx(2.0)
    assert zero_prediction_relative_l2(target).item() == pytest.approx(1.0)


def test_gradient_audit_requires_every_parameter_in_every_group():
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    (first.square()).backward()
    with pytest.raises(RuntimeError, match="branch_b"):
        audit_required_gradients({"branch_a": (first,), "branch_b": (second,)})
    (second.square()).backward()
    report = audit_required_gradients({"branch_a": (first,), "branch_b": (second,)})
    assert report == {"branch_a": 1, "branch_b": 1}


def test_reports_accept_exactly_three_allowed_families_and_reject_anomaly():
    assert_report_families(["uniform", "layered", "marmousi"])
    with pytest.raises(ValueError, match="anomaly"):
        assert_report_families(["uniform", "anomaly"])
    with pytest.raises(ValueError, match="missing"):
        assert_report_families(["uniform", "layered"])
