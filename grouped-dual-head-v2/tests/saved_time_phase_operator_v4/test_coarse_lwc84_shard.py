from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from scripts import evaluate_coarse_lwc84_201_shard as shard_cli
from scripts.evaluate_coarse_lwc84_201_shard import build_parser, run_shard


EXPECTED_COUNTS = {"uniform": 2, "layered": 2, "marmousi": 2}


def _write_wavefield_fixture(path: Path) -> None:
    grid_size = 41
    x_m = np.arange(grid_size, dtype=np.float64) * 5.0
    z_m = np.arange(grid_size, dtype=np.float64) * 5.0
    time_s = np.asarray([0.0, 0.01, 0.02], dtype=np.float64)
    families = np.asarray(
        ["uniform", "layered", "marmousi"] * 2, dtype="S16"
    )
    velocities = np.stack(
        [
            np.full((grid_size, grid_size), 1800.0 + 50.0 * index, dtype=np.float32)
            for index in range(6)
        ]
    )
    source_x_m = np.asarray(
        [83.25, 93.25, 103.25, 113.25, 123.25, 133.25], dtype=np.float64
    )
    source_z_m = np.asarray(
        [42.75, 47.75, 52.75, 57.75, 62.75, 67.75], dtype=np.float64
    )
    frequency_hz = np.full(6, 80.0, dtype=np.float64)
    source_t0_s = np.full(6, 1.5 / 80.0, dtype=np.float64)
    amplitude = np.ones(6, dtype=np.float64)
    solver = LWC84CPMLSolver(
        grid=AcousticGrid(
            nx=grid_size,
            nz=grid_size,
            dx_m=5.0,
            dz_m=5.0,
            lx_m=float(x_m[-1]),
            lz_m=float(z_m[-1]),
            centering="node",
        ),
        boundaries=BoundaryConfig(npml=8),
        dt_s=2.0e-4,
        output_times_s=time_s,
        c_ref_mps=3000.0,
        device="cpu",
        dtype=torch.float32,
        output_restriction_factor=1,
    )
    wavefield = solver.simulate(
        velocities,
        source_x_m=source_x_m,
        source_z_m=source_z_m,
        source_f0_hz=frequency_hz,
        source_t0_s=source_t0_s,
        source_amplitude=amplitude,
    ).wavefield
    with h5py.File(path, "w") as handle:
        handle.create_dataset("medium_type", data=families)
        handle.create_dataset(
            "split", data=np.asarray(["validation"] * 6, dtype="S16")
        )
        handle.create_dataset(
            "sample_id", data=np.asarray([f"s{index}" for index in range(6)], dtype="S8")
        )
        handle.create_dataset(
            "group_id", data=np.asarray([f"g{index}" for index in range(6)], dtype="S8")
        )
        handle.create_dataset(
            "sample_sha256", data=np.asarray([f"h{index}" for index in range(6)], dtype="S8")
        )
        handle.create_dataset("velocity_mps", data=velocities)
        handle.create_dataset("wavefield", data=wavefield)
        handle.create_dataset("source_x_m", data=source_x_m)
        handle.create_dataset("source_z_m", data=source_z_m)
        handle.create_dataset("source_f0_hz", data=frequency_hz)
        handle.create_dataset("source_t0_s", data=source_t0_s)
        handle.create_dataset("source_amplitude", data=amplitude)
        handle.create_dataset("time_s", data=time_s)
        handle.create_dataset("x_m", data=x_m)
        handle.create_dataset("z_m", data=z_m)
        handle.attrs["manifest_sha256"] = "fixture-manifest"
        handle.attrs["config_sha256"] = "fixture-config"


def _arguments(source: Path, output: Path) -> dict[str, object]:
    return {
        "source_h5": source,
        "output_dir": output,
        "shard_index": 0,
        "shard_count": 2,
        "solver_batch_size": 3,
        "device": "cpu",
        "internal_dt_s": 2.0e-4,
        "npml": 8,
        "c_ref_mps": 3000.0,
        "metric_block_size": 2,
        "expected_family_counts": EXPECTED_COUNTS,
    }


def test_shard_cli_defaults_match_registered_remote_run() -> None:
    args = build_parser().parse_args(
        [
            "--source-h5",
            "source.h5",
            "--output-dir",
            "out",
            "--shard-index",
            "0",
        ]
    )
    assert args.shard_count == 4
    assert args.solver_batch_size == 120
    assert args.internal_dt_s == 2.5e-4
    assert args.npml == 20
    assert args.metric_block_size == 20


def test_unindexed_cuda_device_maps_to_visible_device_zero() -> None:
    assert shard_cli._canonicalize_device("cuda") == torch.device("cuda:0")


def test_shard_worker_seals_every_prediction_before_any_truth(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny.h5"
    output = tmp_path / "attempt" / "shards" / "shard_00"
    _write_wavefield_fixture(source)
    summary = run_shard(**_arguments(source, output))
    assert summary["record_count"] == 3
    assert (
        summary["last_prediction_sealed_monotonic_ns"]
        < summary["first_truth_opened_monotonic_ns"]
    )
    assert len(list((output / "predictions").glob("*.pt"))) == 3
    rows = json.loads((output / "records.json").read_text())
    assert all(
        row["truth_opened_after_all_shard_predictions_sealed"] for row in rows
    )
    assert max(row["relative_l2"] for row in rows) < 1.0e-7


def test_completed_shard_is_idempotently_reused(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "tiny.h5"
    output = tmp_path / "shard_00"
    _write_wavefield_fixture(source)
    arguments = _arguments(source, output)
    first = run_shard(**arguments)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("solver reran")

    monkeypatch.setattr(LWC84CPMLSolver, "simulate", fail_if_called)
    second = run_shard(**arguments)
    assert second == first
    assert second["status"] == "complete"
