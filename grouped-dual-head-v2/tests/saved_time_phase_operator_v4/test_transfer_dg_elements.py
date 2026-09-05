from __future__ import annotations

import torch

from saved_time_phase_operator_v4.transfer_dg_elements import (
    PassiveComplexDtN,
    TransferDGLocalElementOperator,
    cartesian_element_origins,
    extract_element_normal_flux_traces,
    extract_element_traces,
    orthonormal_cosine_trace_basis,
    project_trace_modes,
    reconstruct_trace_modes,
)


def test_201_grid_partitions_into_100_cpml_aligned_elements():
    origins = cartesian_element_origins(201, 201, element_intervals=20)
    assert origins.shape == (100, 2)
    assert origins[0].tolist() == [0, 0]
    assert origins[-1].tolist() == [180, 180]


def test_full_cosine_trace_basis_is_orthonormal_and_reconstructs():
    basis = orthonormal_cosine_trace_basis(21, 21, dtype=torch.float64)
    torch.testing.assert_close(
        basis.T @ basis, torch.eye(21, dtype=torch.float64), atol=1.0e-12, rtol=0
    )
    trace = torch.randn(2, 3, 4, 21, dtype=torch.float64)
    recovered = reconstruct_trace_modes(project_trace_modes(trace, basis), basis)
    torch.testing.assert_close(recovered, trace, atol=1.0e-11, rtol=0)


def test_oriented_traces_and_outward_fluxes_have_expected_signs():
    z = torch.arange(21, dtype=torch.float64)[:, None]
    x = torch.arange(21, dtype=torch.float64)[None, :]
    field = 2.0 * x + 3.0 * z
    origins = torch.tensor([[0, 0]])
    trace = extract_element_traces(field, origins, element_intervals=20)
    flux = extract_element_normal_flux_traces(
        torch.full_like(field, 2.0),
        torch.full_like(field, 3.0),
        origins,
        element_intervals=20,
    )
    assert trace.shape == (1, 4, 21)
    torch.testing.assert_close(flux[0, 0], torch.full_like(flux[0, 0], -3.0))
    torch.testing.assert_close(flux[0, 1], torch.full_like(flux[0, 1], 2.0))
    torch.testing.assert_close(flux[0, 2], torch.full_like(flux[0, 2], 3.0))
    torch.testing.assert_close(flux[0, 3], torch.full_like(flux[0, 3], -2.0))


def test_passive_complex_dtn_has_symmetric_parts_and_nonnegative_dissipation():
    torch.manual_seed(307)
    model = PassiveComplexDtN(context_dim=7, trace_dofs=12, rank=4)
    context = torch.randn(5, 7)
    trace = torch.randn(5, 2, 12)
    symmetric, dissipative, source = model.matrices_and_source(context)
    output = model(context, trace)

    torch.testing.assert_close(symmetric, symmetric.transpose(1, 2))
    torch.testing.assert_close(dissipative, dissipative.transpose(1, 2))
    assert torch.linalg.eigvalsh(dissipative).min() > 0.0
    assert torch.all(model.dissipation_quadratic(context, trace) >= 0.0)
    assert source.shape == output.shape == (5, 2, 12)


def test_local_element_operator_preserves_trace_layout_and_gradients():
    torch.manual_seed(311)
    model = TransferDGLocalElementOperator(
        input_channels=4, trace_modes=6, context_dim=32, dtn_rank=4
    )
    features = torch.randn(4, 4, 21, 21)
    frequency = torch.randn(4, 4)
    trace = torch.randn(4, 2, 4, 6)
    output = model(features, frequency, trace)
    output.square().mean().backward()

    assert output.shape == trace.shape
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
