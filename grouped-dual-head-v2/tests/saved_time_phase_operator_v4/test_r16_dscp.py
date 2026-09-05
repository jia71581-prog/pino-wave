from __future__ import annotations

import inspect

import h5py
import numpy as np
import pytest
import torch

from saved_time_phase_operator_v4.instance_adaptation.contracts import onset_indices
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (
    CONDITION_MAXIMUM,
    MODEL_MACS_201,
    R16DSCP,
    RidgePointwiseBaseline,
    analytic_model_macs,
    analytic_temporal_macs,
    basis_condition,
    c1_causal_mask,
    deployment_features,
    frozen_loss,
    predictor_parameter_count,
    velocity_route,
)
from scripts.train_r16_dscp import SealedTruthError, read_two_observed_frames


def _basis() -> torch.Tensor:
    generator = torch.Generator().manual_seed(372)
    families = []
    for _ in range(3):
        matrix = torch.randn(401, 16, generator=generator, dtype=torch.float64)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        families.append(q.float())
    return torch.stack(families)


def test_exact_parameter_and_mac_contracts() -> None:
    model = R16DSCP(_basis(), torch.ones(3, 16))
    ridge = RidgePointwiseBaseline()
    assert predictor_parameter_count(model) == 1202
    assert predictor_parameter_count(ridge) == 480
    assert analytic_model_macs() == MODEL_MACS_201 == 45_451_125
    assert analytic_temporal_macs() == 259_212_816
    assert torch.count_nonzero(model.pointwise_out.weight) == 0
    assert torch.count_nonzero(model.pointwise_out.bias) == 0


def test_velocity_only_router_and_gray_abstention() -> None:
    uniform = torch.full((1, 1, 201, 201), 2000.0)
    layered = uniform.clone()
    layered[:, :, 100:] = 2500.0
    marmousi = uniform.clone()
    checker = (torch.arange(201)[:, None] + torch.arange(201)[None, :]) % 2 == 0
    marmousi[0, 0, checker] = 2600.0
    gray = uniform.clone()
    gray[:, :, :, ::5] = 2200.0
    decisions = [velocity_route(value)[0] for value in (uniform, layered, marmousi, gray)]
    assert [decision.name for decision in decisions] == [
        "uniform",
        "layered",
        "marmousi",
        "abstain",
    ]
    assert decisions[-1].abstain
    assert decisions[-1].one_hot.equal(torch.zeros(3))


def test_c1_mask_onset_and_basis_condition() -> None:
    mask = c1_causal_mask(401, 17)
    assert torch.count_nonzero(mask[:18]) == 0
    assert mask[18] > 0
    assert mask[21] == 1
    assert mask[22] == 1
    left_derivative = mask[17] - mask[16]
    assert left_derivative == 0
    assert basis_condition(_basis()[0], 17) <= CONDITION_MAXIMUM
    time_s = torch.linspace(0.0, 1.0, 401, dtype=torch.float64)
    assert onset_indices(time_s, t0_s=0.1, f0_hz=20.0) == (20, 21)


def test_observation_hdf5_read_is_exactly_k0_k1(tmp_path) -> None:
    path = tmp_path / "observations.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("wavefield", data=np.arange(1 * 6 * 2 * 2).reshape(1, 6, 2, 2))
    accesses: list[int] = []
    observed = read_two_observed_frames(path, 0, 2, 3, split="train", access_log=accesses)
    assert observed.shape == (2, 2, 2)
    assert accesses == [2, 3]
    with pytest.raises(SealedTruthError):
        read_two_observed_frames(path, 0, 2, 3, split="validation")


def test_inference_signatures_exclude_truth_family_and_ids() -> None:
    forbidden = ("truth", "target", "family", "medium_type", "split", "sample", "group", "oracle")
    for method in (R16DSCP.forward, R16DSCP.predict_coefficients, deployment_features):
        names = inspect.signature(method).parameters
        assert not [name for name in names if any(token in name.lower() for token in forbidden)]


def test_future_truth_mutation_cannot_change_deployment_head() -> None:
    model = R16DSCP(_basis(), torch.ones(3, 16))
    features = torch.randn(1, 29, 19, 23, generator=torch.Generator().manual_seed(372))
    future_truth = torch.randn(1, 7, 19, 23)
    before = model.coefficient_head(features).detach().clone()
    future_truth.add_(torch.randn_like(future_truth) * 1.0e6)
    after = model.coefficient_head(features).detach().clone()
    assert torch.equal(before, after)


def test_zero_state_is_exact_parent_with_causal_top_row_contract() -> None:
    model = R16DSCP(_basis(), torch.ones(3, 16))
    velocity = torch.full((1, 1, 201, 201), 2000.0)
    source_map = torch.zeros_like(velocity)
    travel = torch.ones_like(velocity)
    x_m = torch.linspace(0.0, 2000.0, 201)
    z_m = torch.linspace(0.0, 2000.0, 201)
    parent = torch.zeros(1, 401, 201, 201)
    observed0 = parent[:, 20, None].clone()
    observed1 = parent[:, 21, None].clone()
    corrected = model(
        velocity,
        source_map,
        travel,
        x_m,
        z_m,
        observed0,
        observed1,
        parent,
        torch.tensor([20]),
        torch.tensor([21]),
    )
    assert torch.equal(corrected, parent)
    # The structural correction contract is independent of learned weights.
    raw = torch.ones(401, 201, 201)
    masked = raw * c1_causal_mask(401, 21)[:, None, None]
    masked[:, 0, :] = 0
    assert torch.count_nonzero(masked[:22]) == 0
    assert torch.count_nonzero(masked[:, 0]) == 0


def test_frozen_loss_weights_and_finite() -> None:
    truth = torch.ones(1, 9, 3, 3)
    prediction = truth + 0.1
    coefficients = torch.full((1, 16, 3, 3), 0.2)
    terms = frozen_loss(prediction, truth, coefficients)
    reconstructed = (
        terms["frame_relative_l2_squared"]
        + 0.25 * terms["late_third_relative_l2_squared"]
        + 0.10 * terms["temporal_difference"]
        + 1.0e-4 * terms["normalized_coefficient_energy"]
    )
    assert torch.isfinite(terms["total"])
    assert torch.allclose(terms["total"], reconstructed)
