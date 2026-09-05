"""CPU math tests for the audited fixed wave-operator core (wave_operators.py).

Pins the physical/numerical correctness B2-H depends on, GPU-free:
  1. 8th-order Laplacian reproduces the analytic Laplacian of a smooth plane wave
     to high order (-k^2 eigenvalue) on the interior.
  2. stencil coefficients match the GT dataset provenance exactly.
  3. operator linearity (superposition) -- required for the source-linearity test later.
  4. discrete variable-c energy is non-negative and rises under a synthetic source kick.
  5. free-surface odd reflection keeps the top row consistent (p(z=0)=0 handled).
"""
import math

import torch

from saved_time_phase_operator_v4.wave_operators import (
    _D2_COEFFS,
    discrete_energy_variable_c,
    laplacian_8th,
    wave_acceleration,
)


def test_stencil_matches_dataset_provenance():
    # dataset attr second_derivative_coefficients (verified from the h5)
    ref = [-0.0017857142857142857, 0.025396825396825397, -0.2, 1.6,
           -2.8472222222222223, 1.6, -0.2, 0.025396825396825397, -0.0017857142857142857]
    for a, b in zip(_D2_COEFFS, ref):
        assert abs(a - b) < 1e-15
    assert abs(sum(_D2_COEFFS)) < 1e-14  # consistency: constant -> zero 2nd derivative


def test_laplacian_plane_wave_eigenvalue():
    # p = sin(kx x) sin(kz z); lap p = -(kx^2+kz^2) p. 8th order => tiny error on interior.
    Z = X = 64
    dx = dz = 10.0
    kx = 2 * math.pi / (16 * dx)   # ~16-cell wavelength: well resolved
    kz = 2 * math.pi / (20 * dz)
    zz = torch.arange(Z, dtype=torch.float64).view(Z, 1) * dz
    xx = torch.arange(X, dtype=torch.float64).view(1, X) * dx
    p = torch.sin(kx * xx) * torch.sin(kz * zz)
    lap = laplacian_8th(p, dx_m=dx, dz_m=dz, free_surface_top=False)
    expected = -(kx * kx + kz * kz) * p
    m = slice(6, -6)
    rel = (lap[m, m] - expected[m, m]).norm() / expected[m, m].norm()
    assert rel < 5e-3, rel   # 8th-order accuracy on a well-resolved wave


def test_operator_linearity():
    torch.manual_seed(0)
    Z = X = 40
    c = 1500.0 + 200.0 * torch.rand(Z, X, dtype=torch.float64)
    p1 = torch.randn(Z, X, dtype=torch.float64)
    p2 = torch.randn(Z, X, dtype=torch.float64)
    a, b = 0.7, -1.3
    lhs = wave_acceleration(a * p1 + b * p2, c, dx_m=10.0, dz_m=10.0)
    rhs = a * wave_acceleration(p1, c, dx_m=10.0, dz_m=10.0) + b * wave_acceleration(p2, c, dx_m=10.0, dz_m=10.0)
    assert (lhs - rhs).abs().max() < 1e-9


def test_energy_nonneg_and_rises_under_source_kick():
    torch.manual_seed(1)
    Z = X = 48
    c = torch.full((Z, X), 1500.0, dtype=torch.float64)
    dt = 0.0025
    # frame 0: at rest; frame 1: a localized displacement (source kick) -> v>0, E>0
    p0 = torch.zeros(Z, X, dtype=torch.float64)
    p1 = torch.zeros(Z, X, dtype=torch.float64)
    p1[Z // 2, X // 2] = 1.0
    p2 = 2 * p1 - p0 + dt * dt * wave_acceleration(p1, c, dx_m=10.0, dz_m=10.0)
    e_rest = discrete_energy_variable_c(p0, p0, p0, c, dt_s=dt)
    e_kick = discrete_energy_variable_c(p0, p1, p2, c, dt_s=dt)
    assert e_rest["total"] == 0.0
    assert e_kick["total"] > 0.0
    assert e_kick["kinetic"] >= 0.0 and e_kick["potential"] >= -1e-12


def test_free_surface_top_odd_reflection_finite():
    # a field nonzero near the top must give a finite Laplacian under free-surface handling
    Z = X = 32
    p = torch.zeros(Z, X, dtype=torch.float64)
    p[1, X // 2] = 1.0
    lap = laplacian_8th(p, dx_m=10.0, dz_m=10.0, free_surface_top=True)
    assert torch.isfinite(lap).all()
