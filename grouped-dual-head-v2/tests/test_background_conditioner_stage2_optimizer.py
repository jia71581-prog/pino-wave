import math

import numpy as np
import pytest
import torch

from saved_time_phase_operator_v4.multifidelity import fixed_teacher_time_indices
from scripts.diagnose_helmholtz_g3_heldout import (
    build_background_conditioner_stage2_optimizer,
    fourier_training_design_report,
)


class _ToyDirectSpectralPropagator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = torch.nn.Parameter(torch.zeros(4, 4))
        self.expert_gate = torch.nn.Linear(4, 4)
        self.contrast_expert_gate = torch.nn.Linear(32, 4, bias=False)


class _ToyConditioner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.propagation_coupled_gates = torch.nn.Parameter(torch.zeros(2))
        self.direct_spectral_propagator = _ToyDirectSpectralPropagator()
        self.input_projection = torch.nn.Linear(3, 4)
        self.propagation = torch.nn.Linear(4, 4)
        self.output_projection = torch.nn.Linear(4, 2)


class _ToySynthesis(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(4, 4)
        self.late_basis_head = torch.nn.Linear(4, 3)
        self.late_mix = torch.nn.Parameter(torch.zeros(2, 3))


def test_stage2_optimizer_separates_warmed_output_from_core():
    conditioner = _ToyConditioner()
    optimizer, report = build_background_conditioner_stage2_optimizer(
        conditioner,
        output_learning_rate=1.0e-5,
        core_learning_rate=1.0e-6,
    )

    assert [group["lr"] for group in optimizer.param_groups] == [1.0e-5, 1.0e-6]
    assert report["output_parameter_count"] == sum(
        parameter.numel() for parameter in conditioner.output_projection.parameters()
    )
    assert report["core_parameter_count"] == sum(
        parameter.numel()
        for name, parameter in conditioner.named_parameters()
        if not name.startswith("output_projection.")
    )
    assert report["late_parameter_count"] == 0
    assert optimizer.defaults["eps"] == pytest.approx(1.0e-8)


def test_stage2_optimizer_supports_small_gradient_epsilon():
    conditioner = _ToyConditioner()
    optimizer, report = build_background_conditioner_stage2_optimizer(
        conditioner,
        output_learning_rate=1.0e-6,
        core_learning_rate=1.0e-6,
        adam_epsilon=1.0e-12,
    )

    assert optimizer.defaults["eps"] == pytest.approx(1.0e-12)
    assert report["adam_epsilon"] == pytest.approx(1.0e-12)


def test_stage2_optimizer_can_isolate_coupled_propagation_gates():
    conditioner = _ToyConditioner()
    optimizer, report = build_background_conditioner_stage2_optimizer(
        conditioner,
        output_learning_rate=1.0e-6,
        core_learning_rate=2.0e-7,
        coupled_learning_rate=3.0e-4,
    )

    assert [group["lr"] for group in optimizer.param_groups] == [
        1.0e-6,
        2.0e-7,
        3.0e-4,
    ]
    assert report["coupled_parameter_count"] == 2
    assert report["core_parameter_count"] == sum(
        parameter.numel()
        for name, parameter in conditioner.named_parameters()
        if not name.startswith("output_projection.")
        and name != "propagation_coupled_gates"
    )


def test_stage2_optimizer_can_isolate_direct_spectral_experts():
    conditioner = _ToyConditioner()
    optimizer, report = build_background_conditioner_stage2_optimizer(
        conditioner,
        output_learning_rate=1.0e-6,
        core_learning_rate=2.0e-7,
        expert_learning_rate=4.0e-5,
    )

    assert [group["lr"] for group in optimizer.param_groups] == [
        1.0e-6,
        2.0e-7,
        4.0e-5,
    ]
    assert report["expert_parameter_count"] == sum(
        parameter.numel()
        for parameter in conditioner.direct_spectral_propagator.parameters()
    )


def test_stage2_optimizer_can_isolate_contrast_expert_router():
    conditioner = _ToyConditioner()
    optimizer, report = build_background_conditioner_stage2_optimizer(
        conditioner,
        output_learning_rate=1.0e-6,
        core_learning_rate=2.0e-7,
        expert_learning_rate=4.0e-5,
        expert_router_learning_rate=3.0e-2,
    )

    assert [group["lr"] for group in optimizer.param_groups] == [
        1.0e-6,
        2.0e-7,
        4.0e-5,
        3.0e-2,
    ]
    assert report["expert_router_parameter_count"] == sum(
        parameter.numel()
        for parameter in conditioner.direct_spectral_propagator.contrast_expert_gate.parameters()
    )
    assert report["expert_parameter_count"] == sum(
        parameter.numel()
        for name, parameter in conditioner.direct_spectral_propagator.named_parameters()
        if not name.startswith("contrast_expert_gate.")
    )


def test_stage2_optimizer_adds_only_late_synthesis_parameters():
    conditioner = _ToyConditioner()
    synthesis = _ToySynthesis()
    for parameter in synthesis.parameters():
        parameter.requires_grad_(False)
    for name, parameter in synthesis.named_parameters():
        if name.startswith("late_"):
            parameter.requires_grad_(True)
    optimizer, report = build_background_conditioner_stage2_optimizer(
        conditioner,
        output_learning_rate=1.0e-5,
        core_learning_rate=1.0e-6,
        late_head=synthesis,
        late_learning_rate=2.0e-5,
    )

    assert [group["lr"] for group in optimizer.param_groups] == [
        1.0e-5,
        1.0e-6,
        2.0e-5,
    ]
    assert report["late_parameter_count"] == sum(
        parameter.numel()
        for name, parameter in synthesis.named_parameters()
        if name.startswith("late_")
    )


@pytest.mark.parametrize("bad_learning_rate", [0.0, -1.0, math.inf, math.nan])
def test_stage2_optimizer_rejects_invalid_learning_rates(bad_learning_rate):
    with pytest.raises(ValueError, match="learning rate must be positive"):
        build_background_conditioner_stage2_optimizer(
            _ToyConditioner(),
            output_learning_rate=bad_learning_rate,
            core_learning_rate=1.0e-6,
        )

    with pytest.raises(ValueError, match="coupled learning rate must be positive"):
        build_background_conditioner_stage2_optimizer(
            _ToyConditioner(),
            output_learning_rate=1.0e-6,
            core_learning_rate=1.0e-6,
            coupled_learning_rate=bad_learning_rate,
        )

    with pytest.raises(ValueError, match="expert learning rate must be positive"):
        build_background_conditioner_stage2_optimizer(
            _ToyConditioner(),
            output_learning_rate=1.0e-6,
            core_learning_rate=1.0e-6,
            expert_learning_rate=bad_learning_rate,
        )

    with pytest.raises(ValueError, match="expert router learning rate must be positive"):
        build_background_conditioner_stage2_optimizer(
            _ToyConditioner(),
            output_learning_rate=1.0e-6,
            core_learning_rate=1.0e-6,
            expert_router_learning_rate=bad_learning_rate,
        )


@pytest.mark.parametrize("bad_epsilon", [0.0, -1.0, math.inf, math.nan])
def test_stage2_optimizer_rejects_invalid_adam_epsilon(bad_epsilon):
    with pytest.raises(ValueError, match="Adam epsilon must be positive"):
        build_background_conditioner_stage2_optimizer(
            _ToyConditioner(),
            output_learning_rate=1.0e-6,
            core_learning_rate=1.0e-6,
            adam_epsilon=bad_epsilon,
        )


def test_fourier_uniform_training_grid_is_full_rank_and_well_conditioned():
    axis = np.linspace(0.0, 1.0, 401, dtype=np.float64)
    indices = fixed_teacher_time_indices(stored_time_count=401, count=64)

    report = fourier_training_design_report(axis, indices, frequencies=32)

    assert report["frame_count"] == 64
    assert report["effective_coefficient_count"] == 63
    assert report["rank"] == 63
    assert report["condition_number"] < 2.0
