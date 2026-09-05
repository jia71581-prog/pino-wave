from __future__ import annotations

import torch

from saved_time_phase_operator_v4.dg_skeleton_oracle import (
    FixedSkeletonFluxProjector,
    skeleton_hat_basis,
)


def test_skeleton_basis_is_zero_on_free_surface_and_cpml_boundaries():
    basis, centers = skeleton_hat_basis(
        41, 41, element_intervals=10, cpml_margin=10
    )
    assert centers.shape == (2, 2)
    assert torch.count_nonzero(basis[:, 0]) == 0
    assert torch.count_nonzero(basis[:, :, :10]) == 0
    assert torch.count_nonzero(basis[:, :, 30:]) == 0
    assert torch.count_nonzero(basis[:, 30:]) == 0


def test_fixed_projector_exactly_recovers_its_bilinear_trial_space():
    projector = FixedSkeletonFluxProjector(
        height=41,
        width=41,
        element_intervals=10,
        cpml_margin=10,
        dx_m=2.0,
        dz_m=2.0,
        device="cpu",
        dtype=torch.float64,
    )
    torch.manual_seed(401)
    coefficients = torch.randn(3, projector.basis.shape[0], dtype=torch.float64)
    field = torch.einsum("bd,dzx->bzx", coefficients, projector.basis)
    recovered = projector.project_field_flux(field)

    assert projector.report["matrix_rank"] == projector.basis.shape[0]
    torch.testing.assert_close(recovered, field, atol=1.0e-10, rtol=1.0e-10)
