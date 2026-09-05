from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from continuous_wave_operator.data.dataset import ContinuousWaveDataset
from continuous_wave_operator.data.query_sampling import QuerySampler
from fno_acoustic.data_generation.hdf5_lwc84 import LWC84ShardWriter, build_lwc84_dataset_vds


def _attrs() -> dict[str, object]:
    return {
        "dt_requested_s": 0.01,
        "dt_used_s": 0.01,
        "snapshot_stride": 1,
        "config_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "marmousi_sha256": "c" * 64,
        "git_commit": "test",
        "software_environment": "pytest",
    }


def _write_shard(path: Path, split: str, offset: float) -> Path:
    time_s = np.linspace(0.0, 0.03, 4)
    x_m = np.linspace(0.0, 2000.0, 201)
    z_m = np.linspace(0.0, 2000.0, 201)
    writer = LWC84ShardWriter(
        path,
        sample_count=1,
        split=split,
        time_s=time_s,
        x_m=x_m,
        z_m=z_m,
        attrs=_attrs(),
    )
    source_map = np.zeros((201, 201), dtype=np.float32)
    source_map[10, 20] = 1.0
    writer.write_sample(
        0,
        velocity_mps=np.full((201, 201), 2500.0 + offset, dtype=np.float32),
        wavefield=np.arange(4 * 201 * 201, dtype=np.float32).reshape(4, 201, 201) + offset,
        source_map=source_map,
        source_wavelet=np.zeros(4, dtype=np.float32),
        source_x_m=200.0,
        source_z_m=100.0,
        source_f0_hz=15.0,
        source_t0_s=0.1,
        source_amplitude=1.0,
        vmin_mps=2500.0 + offset,
        vmax_mps=2500.0 + offset,
        cfl_2d=0.1,
        lwc_qmax=0.1,
        dt_used_s=0.01,
        crop_x0_m=np.nan,
        crop_z0_m=np.nan,
        qc_max_abs=1.0,
        qc_final_energy_ratio=0.1,
        medium_type="uniform",
        sample_id=f"{split}-0",
        group_id=f"{split}:uniform:0",
        qc_status="passed",
        seed=1,
    )
    return writer.commit()


def test_dataset_reads_chunk_aligned_continuous_queries(tmp_path: Path) -> None:
    train = _write_shard(tmp_path / "train.h5", "train", 0.0)
    validation = _write_shard(tmp_path / "validation.h5", "validation", 1.0)
    vds = build_lwc84_dataset_vds(tmp_path / "dataset.h5", [validation, train])

    dataset = ContinuousWaveDataset(vds, split="train")
    batch = dataset.sample_queries(
        [0], QuerySampler(seed=2026, time_frames=4, points_per_frame=32)
    )

    assert len(dataset) == 1
    assert batch.velocity_mps.shape == (1, 1, 201, 201)
    assert batch.source_map.shape == (1, 1, 1, 201, 201)
    assert batch.query_coords.shape == (1, 1, 128, 3)
    assert batch.target_pressure.shape == (1, 1, 128)
    with pytest.raises(ValueError, match="split"):
        ContinuousWaveDataset(vds, split="missing")


def test_query_sampler_tracks_the_same_receivers_across_time() -> None:
    sampler = QuerySampler(seed=9, time_frames=4, points_per_frame=6)

    _, z_indices, x_indices = sampler.sample(nt=8, nz=20, nx=30)

    assert np.all(z_indices == z_indices[0])
    assert np.all(x_indices == x_indices[0])


def test_query_sampler_uses_time_and_spatial_importance_probabilities() -> None:
    sampler = QuerySampler(seed=4, time_frames=1, points_per_frame=16)
    time_probability = np.zeros(8, dtype=np.float64)
    time_probability[-1] = 1.0
    spatial_probability = np.zeros((2, 2), dtype=np.float64)
    spatial_probability[-1, -1] = 1.0

    time, z_indices, x_indices = sampler.sample(
        nt=8,
        nz=20,
        nx=30,
        time_probabilities=time_probability,
        spatial_probabilities=spatial_probability,
    )

    assert time.tolist() == [7]
    assert np.all(z_indices >= 10)
    assert np.all(x_indices >= 15)


def test_dataset_can_read_queries_with_multiple_cpu_processes(tmp_path: Path) -> None:
    train = _write_shard(tmp_path / "train.h5", "train", 0.0)
    vds = build_lwc84_dataset_vds(tmp_path / "dataset.h5", [train])
    dataset = ContinuousWaveDataset(vds, split="train")

    dataset.enable_parallel_loading(2)
    batch = dataset.sample_queries(
        [0], QuerySampler(seed=17, time_frames=2, points_per_frame=8)
    )
    dataset.close()

    assert batch.query_coords.shape == (1, 1, 16, 3)
    assert torch.isfinite(batch.target_pressure).all()
