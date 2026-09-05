from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.fwi_conditioning import (
    gaussian_smooth_2d,
    huber_tv_slowness_squared,
    illumination_gain,
    parse_stage_schedule,
    ray_illumination,
    select_l2_guard,
)


def test_parse_stage_schedule_broadcasts_one_value_and_rejects_wrong_count() -> None:
    assert parse_stage_schedule("2", stages=4, option="sigma") == [2.0, 2.0, 2.0, 2.0]

    with pytest.raises(ValueError, match="4 values"):
        parse_stage_schedule("1,2", stages=4, option="sigma")


def test_gaussian_gradient_smoothing_preserves_constant_and_reduces_impulse() -> None:
    constant = torch.ones((9, 9))
    impulse = torch.zeros((9, 9))
    impulse[4, 4] = 1.0

    assert torch.allclose(gaussian_smooth_2d(constant, sigma_cells=1.5), constant)
    assert gaussian_smooth_2d(impulse, sigma_cells=1.5)[4, 4].item() < 1.0


def test_ray_illumination_gain_is_bounded_and_deemphasises_well_covered_cells() -> None:
    illumination = ray_illumination(
        9,
        9,
        source_ij=np.asarray([[0, 4]], dtype=np.int64),
        receiver_ij=np.asarray([[8, 4]], dtype=np.int64),
    )
    gain = illumination_gain(illumination, max_gain=3.0)

    assert float(gain.min()) >= 1.0 / 3.0
    assert float(gain.max()) <= 3.0
    assert gain[4, 4].item() < gain[0, 0].item()


def test_huber_tv_on_slowness_is_zero_for_constant_velocity_and_positive_at_an_edge() -> None:
    constant = torch.full((8, 8), 3000.0)
    edge = constant.clone()
    edge[:, 4:] = 4000.0

    assert huber_tv_slowness_squared(
        constant,
        reference_velocity_mps=3000.0,
        delta=1.0e-3,
    ).item() == pytest.approx(0.0)
    assert (
        huber_tv_slowness_squared(
            edge,
            reference_velocity_mps=3000.0,
            delta=1.0e-3,
        ).item()
        > 0.0
    )


def test_select_l2_guard_falls_back_only_when_asvgd_exceeds_tolerance() -> None:
    assert select_l2_guard(l2_nmse=0.1, candidate_nmse=0.101, tolerance=0.02).selected == "asvgd"
    assert select_l2_guard(l2_nmse=0.1, candidate_nmse=0.103, tolerance=0.02).selected == "l2"
