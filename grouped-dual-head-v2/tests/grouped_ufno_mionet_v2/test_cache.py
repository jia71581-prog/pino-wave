from pathlib import Path

import h5py
import numpy as np
import pytest

from grouped_ufno_mionet_v2.data.cache import StructuredCacheDataset, select_dense_times
from scripts.build_grouped_v2_cache import mixture_probabilities
from scripts.fit_grouped_v2_normalization import robust_scales


@pytest.fixture()
def tiny_v2_cache(tmp_path: Path) -> Path:
    path = tmp_path / "cache.h5"
    n, nt, nz, nx, nf, nr, nq = 2, 21, 9, 9, 16, 8, 32
    with h5py.File(path, "w") as h5:
        h5.attrs.update(schema="grouped_dual_head_v2", split="train", complete=True,
                        source_dataset_sha256="data", normalization_sha256="norm")
        h5["velocity_mps"] = np.full((n, nz, nx), 2000, np.float32)
        h5["source_parameters"] = np.tile(np.array([10, 20, 10, .005, 1], np.float32), (n, 1))
        source_map = np.zeros((n, nz, nx), np.float32); source_map[:, 2, 3] = 1
        h5["source_map"] = source_map
        h5["dense_time_indices"] = np.tile(np.arange(nf), (n, 1))
        h5["dense_target"] = np.zeros((n, nf, nz, nx), np.float32)
        h5["receiver_zx_indices"] = np.tile(np.stack((np.ones(nr, int), np.arange(nr)), -1), (n, 1, 1))
        h5["receiver_target"] = np.zeros((n, nr, nt), np.float32)
        h5["query_coords"] = np.zeros((n, nq, 3), np.float32)
        h5["query_target"] = np.zeros((n, nq), np.float32)
        h5["sample_probability"] = np.full((n, nq), 1 / nq, np.float32)
        h5["time_s"] = np.arange(nt, dtype=np.float32) * .0025
        h5["sample_id"] = np.asarray(["a", "b"], dtype=h5py.string_dtype())
        h5["group_id"] = np.asarray(["g", "g"], dtype=h5py.string_dtype())
        h5["medium_type"] = np.asarray(["uniform", "uniform"], dtype=h5py.string_dtype())
    return path


def test_cache_exposes_structured_frames_traces_and_probabilities(tiny_v2_cache):
    item = StructuredCacheDataset(tiny_v2_cache, expected_split="train")[0]
    assert item.dense_target.shape == (16, 9, 9)
    assert item.receiver_target.shape == (8, 21)
    assert item.query_coords.shape == (32, 3)
    assert item.source_map.shape == (1, 9, 9)
    assert item.sample_probability.min() > 0
    assert item.metadata["split"] == "train"


def test_cache_rejects_wrong_split_and_incomplete_file(tiny_v2_cache):
    with pytest.raises(ValueError, match="split"):
        StructuredCacheDataset(tiny_v2_cache, expected_split="validation")
    with h5py.File(tiny_v2_cache, "r+") as h5:
        h5.attrs["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        StructuredCacheDataset(tiny_v2_cache, expected_split="train")


def test_dense_times_include_two_onset_frames_and_end_time():
    time_s = np.arange(21) * .0025
    indices = select_dense_times(time_s, source_t0_s=.0101, count=8)
    assert tuple(indices[:2]) == (5, 6)
    assert indices[-1] == 20
    assert len(indices) == 8


def test_robust_scales_ignore_zeros_and_are_positive():
    velocity = np.array([1500, 2000, 2500, 3000, 3500], np.float32)
    pressure = np.array([0, 0, -1e-8, 2e-8], np.float32)
    center, scale, pressure_scale = robust_scales(velocity, pressure, pressure_percentile=99.0)
    assert center == 2500.0
    assert scale > 0
    assert pressure_scale >= 1e-8


def test_mixture_probabilities_keep_uniform_floor_and_normalize():
    probability = mixture_probabilities(np.array([0.0, 0.0, 10.0, 0.0]), uniform_floor=0.2)
    np.testing.assert_allclose(probability.sum(), 1.0)
    assert probability.min() >= 0.2 / 4
    assert probability[2] > probability[0]
