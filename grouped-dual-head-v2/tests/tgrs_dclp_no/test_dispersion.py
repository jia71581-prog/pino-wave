from __future__ import annotations

import math

import pytest

from tgrs_dclp_no.dispersion import STENCILS, phase_velocity_ratio, spatial_symbol


def test_all_schemes_converge_to_unit_phase_velocity():
    for scheme in ("fd2", "fd4", "lwc84"):
        ratio = phase_velocity_ratio(
            scheme=scheme,
            points_per_wavelength=1000.0,
            angle_deg=31.0,
            courant_axis=0.05,
        )
        assert abs(ratio - 1.0) < 1.0e-4


def test_higher_order_space_is_more_accurate_at_eight_points_per_wavelength():
    values = {
        scheme: abs(
            phase_velocity_ratio(
                scheme=scheme,
                points_per_wavelength=8.0,
                angle_deg=45.0,
                courant_axis=0.10,
            )
            - 1.0
        )
        for scheme in ("fd2", "fd4", "lwc84")
    }
    assert values["fd4"] < values["fd2"]
    assert values["lwc84"] < values["fd4"]


def test_spatial_symbol_matches_continuum_second_derivative_at_small_theta():
    # S(theta) -> -theta**2 as theta -> 0 for every scheme.
    theta = 1.0e-3
    for scheme in STENCILS:
        assert spatial_symbol(scheme, theta) == pytest.approx(-(theta**2), rel=1.0e-3, abs=1.0e-9)


def test_symbol_is_even_in_theta():
    for scheme in STENCILS:
        assert spatial_symbol(scheme, 0.7) == pytest.approx(spatial_symbol(scheme, -0.7))


def test_invalid_sampling_is_rejected():
    with pytest.raises(ValueError):
        phase_velocity_ratio(scheme="lwc84", points_per_wavelength=2.0, angle_deg=0.0, courant_axis=0.1)
    with pytest.raises(ValueError):
        phase_velocity_ratio(scheme="lwc84", points_per_wavelength=10.0, angle_deg=0.0, courant_axis=0.0)
