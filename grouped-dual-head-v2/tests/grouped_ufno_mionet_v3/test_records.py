from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from grouped_ufno_mionet_v3.data.index import build_manifest
from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset, select_balanced_exact_times
from grouped_ufno_mionet_v3.data.batch import pack_v3_groups


@pytest.fixture()
def tiny_wavefield_v3(tmp_path: Path) -> Path:
    path = tmp_path / "wavefield.h5"
    n, nt, nz, nx = 4, 3, 3, 4
    time_s = np.asarray([0.0, 0.5, 1.0], np.float64)
    base = np.arange(nz * nx, dtype=np.float32).reshape(nz, nx)
    wavefield = np.stack(
        [np.stack([base + 100 * record + 10 * frame for frame in range(nt)]) for record in range(n)]
    )
    velocity = np.stack(
        [np.full((nz, nx), value, np.float32) for value in (1800, 1800, 2400, 9000)]
    )
    source_map = np.zeros((n, nz, nx), np.float32)
    source_map[:, 1, 2] = 1.0
    with h5py.File(path, "w") as h5:
        h5.attrs.update(manifest_sha256="source-manifest", config_sha256="source-config")
        h5["medium_type"] = np.asarray(
            ["uniform", "uniform", "marmousi", "anomaly"], dtype=h5py.string_dtype()
        )
        h5["split"] = np.asarray(
            ["train", "train", "validation", "train"], dtype=h5py.string_dtype()
        )
        h5["sample_id"] = np.asarray(["a", "b", "c", "d"], dtype=h5py.string_dtype())
        h5["group_id"] = np.asarray(["same", "same", "other", "bad"], dtype=h5py.string_dtype())
        h5["sample_sha256"] = np.asarray(["sa", "sb", "sc", "sd"], dtype=h5py.string_dtype())
        h5["split_id"] = np.asarray([0, 0, 1, 0], np.uint8)
        h5["time_s"] = time_s
        h5["x_m"] = np.linspace(0, 30, nx)
        h5["z_m"] = np.linspace(0, 20, nz)
        h5["velocity_mps"] = velocity
        h5["wavefield"] = wavefield
        h5["source_map"] = source_map
        h5["source_x_m"] = np.asarray([20, 21, 22, 23], np.float64)
        h5["source_z_m"] = np.asarray([10, 11, 12, 13], np.float64)
        h5["source_f0_hz"] = np.asarray([10, 11, 12, 13], np.float64)
        h5["source_t0_s"] = np.asarray([0.1, 0.2, 0.3, 0.4], np.float64)
        h5["source_amplitude"] = np.asarray([2, 3, 4, 5], np.float64)
    return path


def test_dataset_filters_family_and_preserves_one_source_per_record(tiny_wavefield_v3: Path):
    manifest = build_manifest(tiny_wavefield_v3)
    dataset = V3WavefieldDataset(tiny_wavefield_v3, manifest, split="train")
    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    assert first.medium_type == second.medium_type == "uniform"
    assert first.group_id == second.group_id == "same"
    assert first.source_parameters.tolist() == pytest.approx([20, 10, 10, 0.1, 2])
    assert second.source_parameters.tolist() == pytest.approx([21, 11, 11, 0.2, 3])
    assert first.source_map.shape == (1, 3, 4)
    assert first.source_map.sum().item() == pytest.approx(1.0)


def test_dataset_combines_requested_splits_without_reintroducing_anomaly(
    tiny_wavefield_v3: Path,
):
    manifest = build_manifest(tiny_wavefield_v3)
    dataset = V3WavefieldDataset(
        tiny_wavefield_v3,
        manifest,
        split=("train", "validation"),
    )

    assert len(dataset) == 3
    assert tuple(record.sample_id for record in dataset.records) == ("a", "b", "c")
    assert tuple(dataset[index].split for index in range(len(dataset))) == (
        "train",
        "train",
        "validation",
    )
    assert all(record.medium_type != "anomaly" for record in dataset.records)


def test_dataset_rejects_duplicate_split_names(tiny_wavefield_v3: Path):
    manifest = build_manifest(tiny_wavefield_v3)
    with pytest.raises(ValueError, match="unique"):
        V3WavefieldDataset(
            tiny_wavefield_v3,
            manifest,
            split=("train", "train"),
        )


def test_exact_and_continuous_time_reads_use_adjacent_physical_frames(tiny_wavefield_v3: Path):
    manifest = build_manifest(tiny_wavefield_v3)
    dataset = V3WavefieldDataset(tiny_wavefield_v3, manifest, split="train")
    target = dataset.read_wavefield(0, torch.tensor([0.0, 0.125, 0.5, 0.75, 1.0]))
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.testing.assert_close(target.values[0], base)
    torch.testing.assert_close(target.values[1], base + 2.5)
    torch.testing.assert_close(target.values[2], base + 10.0)
    torch.testing.assert_close(target.values[3], base + 15.0)
    torch.testing.assert_close(target.values[4], base + 20.0)
    assert target.left_index.tolist() == [0, 0, 1, 1, 2]
    assert target.right_index.tolist() == [0, 1, 1, 2, 2]
    assert target.alpha.tolist() == pytest.approx([0.0, 0.25, 0.0, 0.5, 0.0])
    assert target.exact.tolist() == [True, False, True, False, True]


def test_continuous_read_rejects_extrapolation(tiny_wavefield_v3: Path):
    dataset = V3WavefieldDataset(tiny_wavefield_v3, build_manifest(tiny_wavefield_v3), split="train")
    with pytest.raises(ValueError, match="outside"):
        dataset.read_wavefield(0, [-0.01])
    with pytest.raises(ValueError, match="outside"):
        dataset.read_wavefield(0, [1.01])


def test_group_packing_reuses_identical_medium_without_combining_sources(tiny_wavefield_v3: Path):
    dataset = V3WavefieldDataset(tiny_wavefield_v3, build_manifest(tiny_wavefield_v3), split="train")
    batch = pack_v3_groups([dataset[0], dataset[1]])
    assert batch.velocity_mps.shape == (1, 1, 3, 4)
    assert batch.record_to_medium.tolist() == [0, 0]
    assert batch.source_parameters.shape == (2, 5)
    assert batch.source_parameters[:, 4].tolist() == [2.0, 3.0]
    assert batch.sample_id == ("a", "b")


def test_exact_time_selection_is_deterministic_and_phase_balanced():
    time_s = np.linspace(0.0, 1.0, 101)
    first = select_balanced_exact_times(time_s, source_t0_s=0.2, count=12)
    second = select_balanced_exact_times(time_s, source_t0_s=0.2, count=12)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 12
    assert first[:4].max() < first[4:8].min() < first[8:].min()
    assert first[0] == 20
    assert first[-1] == 100
