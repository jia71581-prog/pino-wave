from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fno_acoustic.data_generation.grid import AcousticGrid
from fno_acoustic.data_generation.velocity_models_lwc84 import (
    generate_anomaly_velocity,
    generate_layered_velocity,
    generate_uniform_velocity,
    interpolate_marmousi_normalized_extent,
    load_marmousi_crop,
)


GRID = AcousticGrid(
    nx=401, nz=401, dx_m=5.0, dz_m=5.0, lx_m=2000.0, lz_m=2000.0, centering="node"
)


def test_uniform_distribution_and_ood_exclusion() -> None:
    velocity, metadata = generate_uniform_velocity(GRID, seed=19)
    assert velocity.shape == (401, 401) and velocity.dtype == np.float32
    assert 1800.0 <= float(velocity[0, 0]) <= 5800.0
    assert not 3950.0 <= float(velocity[0, 0]) <= 4050.0
    assert metadata["velocity_mps"] == pytest.approx(float(velocity[0, 0]))


def test_layered_models_are_reproducible_and_obey_geometry() -> None:
    first, metadata = generate_layered_velocity(GRID, seed=24)
    second, metadata_second = generate_layered_velocity(GRID, seed=24)
    assert np.array_equal(first, second)
    assert metadata == metadata_second
    assert 2 <= metadata["n_layers"] <= 5
    assert metadata["minimum_realized_thickness_m"] >= 150.0
    speeds = np.asarray(metadata["layer_velocity_mps"])
    assert np.all((1600.0 <= speeds) & (speeds <= 6000.0))
    assert np.all(np.abs(np.diff(speeds)) >= 200.0)
    # Even seeds are the deterministic >=50% horizontal stratum.
    assert metadata["interface_style"] == "horizontal"


def test_layered_models_are_faster_above_slower_below() -> None:
    for seed in range(32):
        velocity, metadata = generate_layered_velocity(GRID, seed=seed)
        speeds = np.asarray(metadata["layer_velocity_mps"], dtype=np.float64)
        assert np.all(np.diff(speeds) < 0.0)
        assert float(velocity[0].mean()) > float(velocity[-1].mean())


def test_anomaly_models_record_ellipses_and_are_finite() -> None:
    velocity, metadata = generate_anomaly_velocity(GRID, seed=17)
    assert velocity.shape == (401, 401)
    assert np.isfinite(velocity).all() and float(velocity.min()) > 0.0
    assert 1 <= metadata["anomaly_count"] <= 3
    assert len(metadata["anomalies"]) == metadata["anomaly_count"]
    for anomaly in metadata["anomalies"]:
        assert 100.0 <= anomaly["semi_axis_x_m"] <= 450.0
        assert 100.0 <= anomaly["semi_axis_z_m"] <= 450.0
        assert 0.05 <= abs(anomaly["relative_contrast"]) <= 0.35


def test_marmousi_crop_uses_real_coordinates_and_hash(tmp_path: Path) -> None:
    zz, xx = np.meshgrid(np.arange(451), np.arange(501), indexing="ij")
    source = (1500.0 + 2.0 * xx + 3.0 * zz).astype(np.float32)
    path = tmp_path / "marmousi_fixture.bin"
    torch.save(torch.from_numpy(source), path)
    crop, metadata = load_marmousi_crop(
        path,
        grid=GRID,
        source_dx_m=5.0,
        source_dz_m=5.0,
        source_unit="m/s",
        crop_x0_m=100.0,
        crop_z0_m=50.0,
        interpolation="linear",
    )
    assert crop.shape == (401, 401)
    assert crop[0, 0] == pytest.approx(source[10, 20])
    assert crop[-1, -1] == pytest.approx(source[410, 420])
    assert metadata["crop_bbox_m"] == [100.0, 2100.0, 50.0, 2050.0]
    assert len(metadata["source_sha256"]) == 64
    assert len(metadata["crop_sha256"]) == 64


def test_marmousi_crop_refuses_tiling_or_out_of_bounds(tmp_path: Path) -> None:
    path = tmp_path / "small.bin"
    torch.save(torch.full((116, 227), 2000.0), path)
    with pytest.raises(ValueError, match="outside"):
        load_marmousi_crop(
            path,
            grid=GRID,
            source_dx_m=5.0,
            source_dz_m=5.0,
            source_unit="m/s",
            crop_x0_m=0.0,
            crop_z0_m=0.0,
        )


def test_normalized_extent_interpolation_is_deterministic_and_records_stretch() -> None:
    source = np.asarray([[1500.0, 2000.0, 2500.0], [3000.0, 3500.0, 4000.0]], dtype=np.float32)
    first, metadata = interpolate_marmousi_normalized_extent(
        source, target_shape=(9, 13), target_dx_m=5.0, target_dz_m=5.0
    )
    second, _ = interpolate_marmousi_normalized_extent(
        source, target_shape=(9, 13), target_dx_m=5.0, target_dz_m=5.0
    )
    assert first.shape == (9, 13) and first.dtype == np.float32
    assert np.array_equal(first, second)
    assert first[0, 0] == source[0, 0]
    assert first[-1, -1] == source[-1, -1]
    assert float(first.min()) >= float(source.min())
    assert float(first.max()) <= float(source.max())
    assert metadata["coordinate_transform"] == "normalized_extent_stretch"
    assert metadata["target_physical_extent_m"] == [60.0, 40.0]
    assert len(metadata["derived_array_sha256"]) == 64

    with pytest.raises(ValueError, match="at least 2"):
        interpolate_marmousi_normalized_extent(source, target_shape=(1, 13))
