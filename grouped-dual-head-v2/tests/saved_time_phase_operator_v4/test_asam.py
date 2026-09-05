import ast
from pathlib import Path

import pytest
import torch

from saved_time_phase_operator_v4.asam import (
    asam_perturb,
    asam_rho_for_update,
    asam_validation_decision,
    build_asam_adamw,
    freeze_for_asam,
)


def test_asam_refinement_uses_valid_one_based_validation_panel():
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "refine_saved_time_v4_asam.py"
    ).read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validation_panel_indices"
    ]
    assert len(calls) == 1
    epoch = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "epoch")
    assert isinstance(epoch, ast.Constant)
    assert epoch.value == 1


def test_asam_perturbs_and_exactly_restores_parameters():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    parameter.grad = torch.tensor([0.2, -0.1, 0.4])
    original = parameter.detach().clone()

    state = asam_perturb((parameter,), rho=0.05, eta=0.01)

    assert not torch.equal(parameter, original)
    assert state.gradient_norm > 0
    assert state.perturbation_norm > 0
    state.restore()
    torch.testing.assert_close(parameter, original, rtol=0.0, atol=0.0)


def test_asam_rejects_missing_or_nonfinite_gradients():
    missing = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="gradient"):
        asam_perturb((missing,), rho=0.05, eta=0.01)
    missing.grad = torch.tensor([float("nan"), 0.0])
    with pytest.raises(FloatingPointError, match="gradient"):
        asam_perturb((missing,), rho=0.05, eta=0.01)


@pytest.mark.parametrize("rho,eta", [(0.0, 0.01), (0.05, 0.0)])
def test_asam_requires_positive_hyperparameters(rho, eta):
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.ones(2)
    with pytest.raises(ValueError, match="positive"):
        asam_perturb((parameter,), rho=rho, eta=eta)


def test_asam_can_perturb_only_matrix_parameters():
    weight = torch.nn.Parameter(torch.ones(2, 2))
    bias = torch.nn.Parameter(torch.ones(2))
    weight.grad = torch.full_like(weight, 0.25)
    bias.grad = torch.full_like(bias, 0.5)
    original_weight = weight.detach().clone()
    original_bias = bias.detach().clone()

    state = asam_perturb(
        (weight, bias), rho=0.05, eta=0.01, minimum_parameter_ndim=2
    )

    assert not torch.equal(weight, original_weight)
    assert torch.equal(bias, original_bias)
    assert state.parameters == (weight,)
    state.restore()
    torch.testing.assert_close(weight, original_weight, rtol=0.0, atol=0.0)


def test_asam_rho_schedule_warms_up_then_decays_to_floor():
    values = [
        asam_rho_for_update(
            update,
            total_updates=6,
            maximum_rho=0.05,
            minimum_rho=0.005,
            warmup_updates=2,
        )
        for update in range(1, 7)
    ]

    assert values[0] == pytest.approx(0.0275)
    assert values[1] == pytest.approx(0.05)
    assert values[-1] == pytest.approx(0.005)
    assert values[1:] == sorted(values[1:], reverse=True)


def test_single_update_asam_schedule_exercises_maximum_radius():
    assert asam_rho_for_update(
        1,
        total_updates=1,
        maximum_rho=0.05,
        minimum_rho=0.005,
    ) == pytest.approx(0.05)


def test_asam_adamw_excludes_vectors_and_biases_from_weight_decay():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 1, bias=False),
    )

    optimizer = build_asam_adamw(
        model.named_parameters(),
        learning_rate=2.0e-5,
        weight_decay=1.0e-6,
        betas=(0.9, 0.99),
    )

    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["decay"]["weight_decay"] == pytest.approx(1.0e-6)
    assert groups["no_decay"]["weight_decay"] == 0.0
    assert len(groups["decay"]["params"]) == 2
    assert len(groups["no_decay"]["params"]) == 3
    assert {group["betas"] for group in optimizer.param_groups} == {(0.9, 0.99)}


def test_asam_adamw_uses_longest_prefix_learning_rate():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense_decoder = torch.nn.Linear(3, 3)
            self.local_field = torch.nn.Module()
            self.local_field.temporal_latent = torch.nn.Linear(3, 2)

    model = Model()
    optimizer = build_asam_adamw(
        model.named_parameters(),
        learning_rate=1.25e-7,
        learning_rates_by_prefix={"local_field.temporal_latent": 1.25e-6},
        weight_decay=1.0e-6,
    )

    rates = {
        str(group["group_name"]): float(group["initial_lr"])
        for group in optimizer.param_groups
    }
    assert rates["default_decay"] == pytest.approx(1.25e-7)
    assert rates["default_no_decay"] == pytest.approx(1.25e-7)
    assert rates["local_field.temporal_latent_decay"] == pytest.approx(1.25e-6)
    assert rates["local_field.temporal_latent_no_decay"] == pytest.approx(1.25e-6)


def test_asam_validation_requires_strict_improvement_and_family_safety():
    accepted_family = {"uniform": 0.2, "layered": 0.3, "marmousi": 0.5}
    safe = asam_validation_decision(
        accepted_score=0.36,
        candidate_score=0.35,
        accepted_family=accepted_family,
        candidate_family={"uniform": 0.19, "layered": 0.29, "marmousi": 0.49},
        family_regression_tolerance=0.01,
    )
    unsafe = asam_validation_decision(
        accepted_score=0.36,
        candidate_score=0.35,
        accepted_family=accepted_family,
        candidate_family={"uniform": 0.19, "layered": 0.31, "marmousi": 0.49},
        family_regression_tolerance=0.01,
    )

    assert safe.accepted is True
    assert safe.learning_rate_multiplier == 1.0
    assert safe.rho_multiplier == 1.0
    assert unsafe.accepted is False
    assert unsafe.score_improved is True
    assert unsafe.family_safe is False
    assert unsafe.learning_rate_multiplier == pytest.approx(0.5)
    assert unsafe.rho_multiplier == pytest.approx(0.5)


def test_asam_stage_selects_explicit_nonoverlapping_prefixes():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 3)
            self.dense_decoder = torch.nn.Linear(3, 2)
            self.local_field = torch.nn.Linear(2, 1)

    model = Model()
    selected = freeze_for_asam(
        model, trainable_prefixes=("dense_decoder", "local_field")
    )

    assert {id(parameter) for parameter in selected} == {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(("dense_decoder.", "local_field."))
    }
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in selected)


def test_asam_stage_rejects_missing_or_overlapping_prefixes():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="no parameters"):
        freeze_for_asam(model, trainable_prefixes=("dense_decoder",))
    with pytest.raises(ValueError, match="overlap"):
        freeze_for_asam(model, trainable_prefixes=("0", "0.weight"))
