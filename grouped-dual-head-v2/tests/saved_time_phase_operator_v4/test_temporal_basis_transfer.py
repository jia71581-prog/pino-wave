from types import SimpleNamespace

import pytest
import torch

from saved_time_phase_operator_v4.probe import ProbeVariant
from saved_time_phase_operator_v4.decoder import QueryInvariantTemporalBasis
from scripts.train_saved_time_v4_full_support import (
    absorb_temporal_basis_gate_if_requested,
    probe_variant_for_config,
    temporal_basis_gradient_norms,
    temporal_basis_missing_prefixes,
)


def _identity(**variant):
    values = {
        "depth": 8,
        "use_local_phase": True,
        "spectral_rank": 112,
        "modes": 101,
        "coupled_axes": False,
        "local_differential_residual": False,
        "coupled_2d_rank": 0,
    }
    values.update(variant)
    return {"variant_config": values}


def test_temporal_basis_variant_defaults_to_zero_and_accepts_rank_96():
    parent_identity = _identity()

    parent = probe_variant_for_config({}, parent_identity)
    candidate = probe_variant_for_config(
        {"variant_overrides": {"temporal_basis_rank": 96}}, parent_identity
    )

    assert parent.temporal_basis_rank == 0
    assert candidate.temporal_basis_rank == 96


@pytest.mark.parametrize("value", [-1, 129, True, 4.5])
def test_temporal_basis_variant_rejects_invalid_rank(value):
    with pytest.raises(ValueError, match="temporal_basis_rank"):
        probe_variant_for_config(
            {"variant_overrides": {"temporal_basis_rank": value}}, _identity()
        )


def test_temporal_basis_checkpoint_expansion_authorizes_only_its_module():
    parent = ProbeVariant(depth=8, use_local_phase=True, modes=101)
    candidate = ProbeVariant(
        depth=8,
        use_local_phase=True,
        modes=101,
        temporal_basis_rank=96,
    )
    config = {
        "checkpoint_transfer": {
            "allow_new_temporal_basis_parameters": True,
            "parent_optimizer_state": False,
        }
    }

    assert temporal_basis_missing_prefixes(
        config,
        parent_variant=parent,
        candidate_variant=candidate,
    ) == ("dense_decoder.temporal_basis.",)


@pytest.mark.parametrize(
    ("config", "parent", "candidate", "message"),
    [
        ({}, ProbeVariant(8, True, modes=101), ProbeVariant(8, True, modes=101, temporal_basis_rank=96), "requires"),
        (
            {"checkpoint_transfer": {"allow_new_temporal_basis_parameters": True, "parent_optimizer_state": True}},
            ProbeVariant(8, True, modes=101),
            ProbeVariant(8, True, modes=101, temporal_basis_rank=96),
            "optimizer",
        ),
        (
            {"checkpoint_transfer": {"allow_new_temporal_basis_parameters": True}},
            ProbeVariant(8, True, modes=32),
            ProbeVariant(8, True, modes=101, temporal_basis_rank=96),
            "staged",
        ),
        (
            {"checkpoint_transfer": {"allow_new_temporal_basis_parameters": True}},
            ProbeVariant(8, True, modes=101, temporal_basis_rank=16),
            ProbeVariant(8, True, modes=101, temporal_basis_rank=96),
            "only supports rank 0",
        ),
    ],
)
def test_temporal_basis_checkpoint_expansion_rejects_unsafe_transfer(
    config, parent, candidate, message
):
    with pytest.raises(ValueError, match=message):
        temporal_basis_missing_prefixes(
            config,
            parent_variant=parent,
            candidate_variant=candidate,
        )


def test_temporal_basis_gradient_telemetry_separates_gate_and_features():
    gate = torch.nn.Parameter(torch.tensor(0.0))
    weight = torch.nn.Parameter(torch.ones(2, 3))
    branch = SimpleNamespace(
        gate=gate,
        feature_parameters=lambda: (weight,),
    )
    model = SimpleNamespace(
        dense_decoder=SimpleNamespace(temporal_basis=branch)
    )
    gate.grad = torch.tensor(0.4)

    first = temporal_basis_gradient_norms(model)

    assert first == {
        "temporal_basis_gate": pytest.approx(0.4),
        "temporal_basis_features": 0.0,
    }
    weight.grad = torch.full_like(weight, 0.5)
    second = temporal_basis_gradient_norms(model)
    assert second["temporal_basis_features"] == pytest.approx(
        float(weight.grad.norm())
    )


def test_temporal_basis_gradient_audit_skips_a_frozen_expert_stage():
    gate = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
    weight = torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
    branch = SimpleNamespace(
        gate=gate,
        feature_parameters=lambda: (weight,),
    )
    model = SimpleNamespace(
        dense_decoder=SimpleNamespace(temporal_basis=branch)
    )

    assert temporal_basis_gradient_norms(model) == {}


def test_requested_gate_absorption_is_checkpoint_safe_and_audited():
    branch = QueryInvariantTemporalBasis(width=4, rank=6)
    branch.gate.data.fill_(3.0e-4)
    model = SimpleNamespace(dense_decoder=SimpleNamespace(temporal_basis=branch))

    report = absorb_temporal_basis_gate_if_requested(
        model,
        {
            "residual_recovery": {"absorb_temporal_basis_gate": True},
            "checkpoint_transfer": {"parent_optimizer_state": False},
        },
    )

    assert report["pre_gate"] == pytest.approx(3.0e-4)
    assert report["post_gate"] == pytest.approx(1.0)
    assert branch.gate.item() == pytest.approx(1.0)


def test_gate_absorption_rejects_optimizer_restore_and_missing_branch():
    branch = QueryInvariantTemporalBasis(width=4, rank=6)
    model = SimpleNamespace(dense_decoder=SimpleNamespace(temporal_basis=branch))
    with pytest.raises(ValueError, match="optimizer"):
        absorb_temporal_basis_gate_if_requested(
            model,
            {
                "residual_recovery": {"absorb_temporal_basis_gate": True},
                "checkpoint_transfer": {"parent_optimizer_state": True},
            },
        )
    with pytest.raises(ValueError, match="enabled temporal basis"):
        absorb_temporal_basis_gate_if_requested(
            SimpleNamespace(dense_decoder=SimpleNamespace(temporal_basis=None)),
            {"residual_recovery": {"absorb_temporal_basis_gate": True}},
        )
