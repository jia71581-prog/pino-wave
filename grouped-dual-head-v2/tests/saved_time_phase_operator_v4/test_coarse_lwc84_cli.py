from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from scripts.evaluate_coarse_lwc84_201 import build_parser, run


def test_coarse_cli_defaults_match_registered_design() -> None:
    args = build_parser().parse_args(
        ["--source-h5", "input.h5", "--output-dir", "out"]
    )
    assert args.device == "cuda"
    assert args.internal_dt_s == 2.5e-4
    assert args.npml == 20
    assert args.seed == 17
    assert args.per_family == 1
    assert args.metric_block_size == 20


def test_coarse_cli_help_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_coarse_lwc84_201.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--source-h5" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--device" in completed.stdout


def _make_three_family_fixture(
    path: Path,
    *,
    grid_size: int,
    saved_times: int,
) -> None:
    x_m = np.arange(grid_size, dtype=np.float64) * 5.0
    z_m = np.arange(grid_size, dtype=np.float64) * 5.0
    time_s = np.arange(saved_times, dtype=np.float64) * 0.01
    velocities = np.stack(
        [
            np.full((grid_size, grid_size), 1800.0, dtype=np.float32),
            np.repeat(
                np.linspace(2400.0, 1800.0, grid_size, dtype=np.float32)[:, None],
                grid_size,
                axis=1,
            ),
            np.repeat(
                (
                    2000.0
                    + 150.0 * np.sin(np.linspace(0.0, 4.0, grid_size))
                )[:, None],
                grid_size,
                axis=1,
            ).astype(np.float32),
        ]
    )
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
        dtype=torch.float64,
        output_restriction_factor=1,
    )
    source_x_m = np.asarray([103.25, 93.25, 113.25], dtype=np.float64)
    source_z_m = np.asarray([52.75, 57.75, 62.75], dtype=np.float64)
    frequency_hz = np.full(3, 80.0, dtype=np.float64)
    source_t0_s = np.full(3, 1.5 / 80.0, dtype=np.float64)
    amplitude = np.ones(3, dtype=np.float64)
    wavefield = solver.simulate(
        velocities,
        source_x_m=source_x_m,
        source_z_m=source_z_m,
        source_f0_hz=frequency_hz,
        source_t0_s=source_t0_s,
        source_amplitude=amplitude,
    ).wavefield
    text = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "medium_type",
            data=np.asarray(["uniform", "layered", "marmousi"], dtype=text),
        )
        handle.create_dataset(
            "split", data=np.asarray(["validation"] * 3, dtype=text)
        )
        handle.create_dataset(
            "sample_id", data=np.asarray(["s0", "s1", "s2"], dtype=text)
        )
        handle.create_dataset(
            "group_id", data=np.asarray(["g0", "g1", "g2"], dtype=text)
        )
        handle.create_dataset(
            "sample_sha256", data=np.asarray(["h0", "h1", "h2"], dtype=text)
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


def test_synthetic_cpu_run_seals_before_metrics(tmp_path: Path) -> None:
    source_h5 = tmp_path / "tiny.h5"
    _make_three_family_fixture(source_h5, grid_size=41, saved_times=3)
    summary = run(
        source_h5=source_h5,
        output_dir=tmp_path / "result",
        device="cpu",
        internal_dt_s=2.0e-4,
        npml=8,
        seed=17,
        per_family=1,
        metric_block_size=2,
        make_plots=False,
        c_ref_mps=3000.0,
    )
    assert summary["record_count"] == 3
    assert summary["truth_opened_after_seal"] is True
    assert summary["gate"]["action"] == "direct_baseline"
    assert len(list((tmp_path / "result" / "predictions").glob("*.pt"))) == 3
    assert (tmp_path / "result" / "summary.json").is_file()
