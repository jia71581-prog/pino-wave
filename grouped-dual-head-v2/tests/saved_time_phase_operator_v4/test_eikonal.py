import h5py
import numpy as np
import pytest

from saved_time_phase_operator_v4.eikonal import (
    EikonalTravelCache,
    grid_eikonal_travel_time,
)


def test_grid_eikonal_matches_constant_velocity_radial_solution():
    size = 31
    spacing = 10.0
    velocity = np.full((size, size), 2000.0, dtype=np.float64)
    source = (size // 2, size // 2)

    travel = grid_eikonal_travel_time(
        velocity,
        source_indices=[source],
        dx_m=spacing,
        dz_m=spacing,
    )[0]

    zz, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    exact = spacing * np.hypot(xx - source[1], zz - source[0]) / 2000.0
    informative = exact > 0.0
    relative = np.abs(travel[informative] - exact[informative]) / exact[informative]
    assert travel[source] == 0.0
    assert float(np.percentile(relative, 95)) < 0.015
    assert float(relative.max()) < 0.025


def test_grid_eikonal_supports_multiple_sources_on_one_medium():
    velocity = np.full((15, 17), 1800.0, dtype=np.float32)
    sources = [(2, 3), (12, 14)]

    travel = grid_eikonal_travel_time(
        velocity,
        source_indices=sources,
        dx_m=12.0,
        dz_m=10.0,
    )

    assert travel.shape == (2, 15, 17)
    assert travel.dtype == np.float32
    assert travel[0, sources[0][0], sources[0][1]] == 0.0
    assert travel[1, sources[1][0], sources[1][1]] == 0.0


def test_grid_eikonal_rejects_nonpositive_velocity():
    velocity = np.ones((5, 5), dtype=np.float32)
    velocity[1, 1] = 0.0

    try:
        grid_eikonal_travel_time(
            velocity,
            source_indices=[(2, 2)],
            dx_m=1.0,
            dz_m=1.0,
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive velocity must be rejected")


def test_eikonal_cache_source_alias_requires_explicit_opt_in(tmp_path):
    original_source = tmp_path / "dataset_v1.h5"
    retained_source = tmp_path / "dataset_v1.invalidated.h5"
    cache_path = tmp_path / "travel.h5"
    original_source.touch()
    retained_source.touch()
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["source_h5"] = str(original_source)
        handle.create_dataset("sample_id", data=np.asarray([b"sample_0"]))
        handle.create_dataset(
            "travel_time_s", data=np.ones((1, 3, 4), dtype=np.float32)
        )

    with pytest.raises(ValueError, match="does not match"):
        EikonalTravelCache(cache_path, source_h5=retained_source)

    cache = EikonalTravelCache(
        cache_path,
        source_h5=retained_source,
        allow_source_path_mismatch=True,
    )
    assert cache.source_path_mismatch_allowed
    assert cache.read(["sample_0"]).shape == (1, 3, 4)
