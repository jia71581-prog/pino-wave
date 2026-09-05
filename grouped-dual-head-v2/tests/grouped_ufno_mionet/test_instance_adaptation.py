import pytest
import torch
from torch import nn
import h5py

from grouped_ufno_mionet.instance_adaptation.causal import (
    WavefieldAccessAudit,
    hard_project_observations,
    onset_window,
)


def test_onset_window_uses_first_two_saved_frames_at_or_after_t0():
    window = onset_window(torch.arange(0.0, 0.030, 0.0025), source_t0_s=0.0101)
    assert window.observed_indices == (5, 6)
    assert torch.equal(window.future_indices, torch.tensor([7, 8, 9, 10, 11]))


def test_onset_window_rejects_a_source_without_two_saved_observations():
    with pytest.raises(ValueError, match="two onset-aligned frames"):
        onset_window(torch.tensor([0.0, 0.0025]), source_t0_s=0.0025)


def test_hard_projection_replaces_only_observed_frames():
    prediction = torch.zeros(1, 5, 2, 2)
    observed = torch.full((1, 2, 2, 2), 3.0)
    out = hard_project_observations(prediction, observed, (1, 2))
    assert torch.equal(out[:, 1:3], observed)
    assert torch.count_nonzero(out[:, (0, 3, 4)]) == 0


def test_deployment_audit_rejects_future_wavefield_access():
    with pytest.raises(RuntimeError, match="future wavefield"):
        WavefieldAccessAudit((5, 6)).record((5, 7))


from grouped_ufno_mionet.instance_adaptation.adapters import LowRankLinearDelta, SpatialTemporalFiLM


def test_low_rank_delta_is_exact_identity_at_initialization():
    base = nn.Linear(3, 2, bias=False)
    x = torch.randn(4, 3)
    assert torch.equal(LowRankLinearDelta(base, rank=2)(x), base(x))


def test_film_changes_a_dense_basis_after_delta_update():
    film = SpatialTemporalFiLM(rank=3)
    basis = torch.ones(1, 2, 3, 2, 2)
    assert torch.equal(film(basis), basis)
    film.scale_delta.data.fill_(0.5)
    assert not torch.equal(film(basis), basis)


from grouped_ufno_mionet.instance_adaptation.data import OnsetDeploymentDataset
from grouped_ufno_mionet.model.operator import GroupedSingleSourceUFNOMIONetOperator
from grouped_ufno_mionet.instance_adaptation.model import OnsetAdaptedOperator
from grouped_ufno_mionet.instance_adaptation.trainer import accept_or_rollback
from grouped_ufno_mionet.instance_adaptation.losses import lwc84_residual


def test_deployment_dataset_reads_exactly_two_onset_frames(tmp_path):
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as h5:
        h5["velocity_mps"] = torch.ones(1, 2, 2).numpy()
        h5["wavefield"] = torch.arange(20.0).reshape(1, 5, 2, 2).numpy()
        h5["time_s"] = torch.arange(5).numpy() * 0.0025
        h5["source_t0_s"] = [0.005]
    example = OnsetDeploymentDataset(path)[0]
    assert example.observed_indices == (2, 3)
    assert example.observed_wavefield.shape == (2, 2, 2)
    assert example.accessed_wavefield_indices == (2, 3)


def test_adapted_operator_hard_projects_two_observed_frames():
    base = GroupedSingleSourceUFNOMIONetOperator(width=16, rank=8)
    model = OnsetAdaptedOperator(base, adapter_rank=2)
    output = model(
        torch.full((1, 1, 33, 33), 2000.0),
        torch.tensor([[1000.0, 1000.0, 10.0, 0.005, 1.0]]),
        torch.arange(5) * 0.0025,
        torch.full((1, 2, 33, 33), 7.0),
        (2, 3),
    )
    assert output.shape == (1, 5, 33, 33)
    assert torch.equal(output[:, 2:4], torch.full((1, 2, 33, 33), 7.0))
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert sum(parameter.numel() for parameter in model.trainable_parameters()) < sum(parameter.numel() for parameter in base.parameters())


def test_acceptance_gate_rolls_back_an_energy_explosion():
    accepted, chosen = accept_or_rollback(
        torch.ones(1), torch.full((1,), 9.0), baseline_residual=1.0, candidate_residual=0.5, energy_ratio=9.0
    )
    assert not accepted and torch.equal(chosen, torch.ones(1))


def test_lwc84_residual_of_constant_future_field_is_zero():
    field = torch.ones(1, 6, 5, 5)
    velocity = torch.full((1, 1, 5, 5), 2000.0)
    assert lwc84_residual(field, velocity, dt=0.0025, dx=10.0, dz=10.0, observed_indices=(0, 1)).item() == 0.0
