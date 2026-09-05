#!/usr/bin/env python3
"""Evaluate a native-grid LWC-84 baseline against exact stored wavefields."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from saved_time_phase_operator_v4.coarse_lwc84 import (
    accumulate_exact_metrics,
    decide_coarse_gate,
    seal_prediction,
    select_validation_records,
)
from saved_time_phase_operator_v4.streaming_metrics import (
    ExactWavefieldMetricAccumulator,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--internal-dt-s", type=float, default=2.5e-4)
    parser.add_argument("--npml", type=int, default=20)
    parser.add_argument("--c-ref-mps", type=float, default=6750.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-family", type=int, default=1)
    parser.add_argument("--metric-block-size", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def assert_uniform_axis(
    values: np.ndarray,
    *,
    expected_step: float | None,
    name: str,
) -> float:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all():
        raise ValueError(f"{name} must be a finite one-dimensional axis")
    differences = np.diff(axis)
    step = float(np.median(differences))
    if step <= 0.0 or not np.allclose(
        differences, step, rtol=1.0e-6, atol=1.0e-10
    ):
        raise ValueError(f"{name} must be strictly increasing and uniform")
    if expected_step is not None and not np.isclose(
        step, expected_step, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"{name} step does not match the registered design")
    return step


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    encoded = json.dumps(
        _jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    try:
        with partial.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        partial.unlink(missing_ok=True)


def _save_figure(figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_velocity_source(
    velocity: np.ndarray,
    source_x_m: float,
    source_z_m: float,
    x_m: np.ndarray,
    z_m: np.ndarray,
    output_stem: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = axis.imshow(
        velocity,
        origin="upper",
        extent=[x_m[0], x_m[-1], z_m[-1], z_m[0]],
        aspect="equal",
        cmap="viridis",
    )
    axis.scatter(
        [source_x_m],
        [source_z_m],
        marker="*",
        s=150,
        c="red",
        edgecolors="white",
        linewidths=0.7,
    )
    axis.set(xlabel="x (m)", ylabel="z (m)", title="Velocity model and source")
    figure.colorbar(image, ax=axis, label="Velocity (m/s)")
    _save_figure(figure, output_stem)


def plot_wavefield_truth_coarse(
    truth: np.ndarray,
    coarse: np.ndarray,
    time_s: np.ndarray,
    indices: Sequence[int],
    output_stem: Path,
) -> None:
    chosen = tuple(
        dict.fromkeys(
            min(max(int(index), 0), len(time_s) - 1) for index in indices
        )
    )
    figure, axes = plt.subplots(
        3,
        len(chosen),
        figsize=(3.0 * len(chosen), 8.0),
        squeeze=False,
    )
    for column, index in enumerate(chosen):
        target = truth[index]
        prediction = coarse[index]
        error = prediction - target
        field_limit = max(
            float(np.max(np.abs(target))),
            float(np.max(np.abs(prediction))),
            1.0e-12,
        )
        error_limit = max(float(np.max(np.abs(error))), 1.0e-12)
        rows = (
            (target, field_limit, "Teacher"),
            (prediction, field_limit, "Coarse LWC-84"),
            (error, error_limit, "Error"),
        )
        for row, (field, limit, label) in enumerate(rows):
            axes[row, column].imshow(
                field,
                origin="upper",
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if column == 0:
                axes[row, column].set_ylabel(label)
        axes[0, column].set_title(f"t={time_s[index]:.4f} s")
    figure.tight_layout()
    _save_figure(figure, output_stem)


def plot_receiver_truth_coarse(
    truth: np.ndarray,
    coarse: np.ndarray,
    time_s: np.ndarray,
    receiver_indices: Sequence[tuple[int, int]],
    output_stem: Path,
) -> None:
    if len(receiver_indices) != 9:
        raise ValueError("receiver diagnostic requires exactly nine points")
    figure, axes = plt.subplots(
        3, 3, figsize=(12.0, 8.0), sharex=True, squeeze=False
    )
    for axis, (z_index, x_index) in zip(
        axes.flat, receiver_indices, strict=True
    ):
        axis.plot(
            time_s,
            truth[:, z_index, x_index],
            color="black",
            lw=1.1,
            label="Teacher",
        )
        axis.plot(
            time_s,
            coarse[:, z_index, x_index],
            color="tab:red",
            lw=0.9,
            label="Coarse LWC-84",
        )
        axis.set_title(f"receiver (z={z_index}, x={x_index})")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Pressure")
    figure.tight_layout()
    _save_figure(figure, output_stem)


def _dataset_identity(source_h5: Path) -> dict[str, object]:
    with h5py.File(source_h5, "r", swmr=True) as handle:
        return {
            "path": str(source_h5),
            "byte_count": source_h5.stat().st_size,
            "manifest_sha256": str(handle.attrs.get("manifest_sha256", "")),
            "config_sha256": str(handle.attrs.get("config_sha256", "")),
            "saved_grid_shape": _jsonable(
                handle.attrs.get("saved_grid_shape", [])
            ),
            "saved_dx_m": float(handle.attrs.get("saved_dx_m", np.nan)),
            "saved_dz_m": float(handle.attrs.get("saved_dz_m", np.nan)),
            "teacher_dt_used_s": float(handle.attrs.get("dt_used_s", np.nan)),
        }


def _warm_up_cuda(
    *,
    dx_m: float,
    dz_m: float,
    internal_dt_s: float,
    npml: int,
    c_ref_mps: float,
    device: torch.device,
) -> None:
    warmup = LWC84CPMLSolver(
        grid=AcousticGrid(
            nx=41,
            nz=41,
            dx_m=dx_m,
            dz_m=dz_m,
            lx_m=40.0 * dx_m,
            lz_m=40.0 * dz_m,
            centering="node",
        ),
        boundaries=BoundaryConfig(npml=min(npml, 8)),
        dt_s=internal_dt_s,
        output_times_s=np.asarray([0.0, internal_dt_s], dtype=np.float64),
        c_ref_mps=c_ref_mps,
        device=device,
        dtype=torch.float32,
        output_restriction_factor=1,
    )
    warmup.simulate(
        np.full((41, 41), min(2000.0, c_ref_mps), dtype=np.float32),
        source_x_m=20.0 * dx_m,
        source_z_m=10.0 * dz_m,
        source_f0_hz=20.0,
    )
    torch.cuda.synchronize(device)


def _update_accumulator(
    accumulator: ExactWavefieldMetricAccumulator,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    family: str,
    group_id: str,
    sample_id: str,
    onset_index: int,
    block_size: int,
) -> None:
    for start in range(0, prediction.shape[1], int(block_size)):
        stop = min(start + int(block_size), prediction.shape[1])
        indices = torch.arange(start, stop, dtype=torch.long)[None]
        accumulator.update(
            prediction[:, start:stop],
            target[:, start:stop],
            families=(family,),
            group_ids=(group_id,),
            sample_ids=(sample_id,),
            time_indices=indices,
            source_onset_indices=(onset_index,),
        )


def _receiver_points(height: int, width: int) -> tuple[tuple[int, int], ...]:
    z_index = max(1, height // 20)
    x_indices = np.linspace(5, width - 6, 9, dtype=int)
    return tuple((z_index, int(x_index)) for x_index in x_indices)


def _receiver_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    points: Sequence[tuple[int, int]],
) -> float:
    predicted = torch.stack(
        [prediction[:, :, z_index, x_index] for z_index, x_index in points],
        dim=-1,
    )
    truth = torch.stack(
        [target[:, :, z_index, x_index] for z_index, x_index in points],
        dim=-1,
    )
    denominator = truth.double().square().sum().sqrt().clamp_min(1.0e-16)
    return float((predicted.double() - truth.double()).square().sum().sqrt() / denominator)


def _record_inputs(source_h5: Path, source_index: int) -> dict[str, object]:
    with h5py.File(source_h5, "r", swmr=True) as handle:
        return {
            "velocity": np.asarray(
                handle["velocity_mps"][int(source_index)], dtype=np.float32
            ),
            "source_x_m": float(handle["source_x_m"][int(source_index)]),
            "source_z_m": float(handle["source_z_m"][int(source_index)]),
            "source_f0_hz": float(handle["source_f0_hz"][int(source_index)]),
            "source_t0_s": float(handle["source_t0_s"][int(source_index)]),
            "source_amplitude": float(
                handle["source_amplitude"][int(source_index)]
            ),
        }


def run(
    *,
    source_h5: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    internal_dt_s: float = 2.5e-4,
    npml: int = 20,
    c_ref_mps: float = 6750.0,
    seed: int = 17,
    per_family: int = 1,
    metric_block_size: int = 20,
    make_plots: bool = True,
) -> dict[str, object]:
    source_path = Path(source_h5).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if float(internal_dt_s) <= 0.0 or float(c_ref_mps) <= 0.0:
        raise ValueError("internal_dt_s and c_ref_mps must be positive")
    if int(metric_block_size) < 1:
        raise ValueError("metric_block_size must be positive")

    records = select_validation_records(
        source_path, seed=seed, per_family=per_family
    )
    with h5py.File(source_path, "r", swmr=True) as handle:
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        x_m = np.asarray(handle["x_m"][:], dtype=np.float64)
        z_m = np.asarray(handle["z_m"][:], dtype=np.float64)
    time_step_s = assert_uniform_axis(
        time_s, expected_step=None, name="time_s"
    )
    dx_m = assert_uniform_axis(x_m, expected_step=None, name="x_m")
    dz_m = assert_uniform_axis(z_m, expected_step=None, name="z_m")
    if not np.isclose(time_s[0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("stored time axis must start at zero")
    if not np.isclose(x_m[0], 0.0, rtol=0.0, atol=1.0e-12) or not np.isclose(
        z_m[0], 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("coarse solver axes must start at zero")
    if len(time_s) not in (3, 401):
        raise ValueError(
            "evaluation requires 401 production times or the 3-time test fixture"
        )
    if len(time_s) == 401 and (len(x_m), len(z_m)) != (201, 201):
        raise ValueError("production evaluation requires a 201 by 201 saved grid")
    if abs(dx_m - dz_m) > 1.0e-9:
        raise ValueError("LWC84 evaluation requires equal x/z spacing")
    aligned_steps = np.rint(time_s / float(internal_dt_s))
    if not np.allclose(
        time_s / float(internal_dt_s),
        aligned_steps,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("stored times do not align to the internal step")

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; refusing CPU fallback"
        )
    if requested_device.type == "cuda":
        _warm_up_cuda(
            dx_m=dx_m,
            dz_m=dz_m,
            internal_dt_s=float(internal_dt_s),
            npml=int(npml),
            c_ref_mps=float(c_ref_mps),
            device=requested_device,
        )

    grid = AcousticGrid(
        nx=len(x_m),
        nz=len(z_m),
        dx_m=dx_m,
        dz_m=dz_m,
        lx_m=float(x_m[-1] - x_m[0]),
        lz_m=float(z_m[-1] - z_m[0]),
        centering="node",
    )
    boundaries = BoundaryConfig(npml=int(npml))
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=boundaries,
        dt_s=float(internal_dt_s),
        output_times_s=time_s,
        c_ref_mps=float(c_ref_mps),
        device=requested_device,
        dtype=torch.float32,
        output_restriction_factor=1,
    )
    overall = ExactWavefieldMetricAccumulator(
        require_unique=True, stored_time_count=len(time_s)
    )
    run_started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    record_reports: list[dict[str, object]] = []
    order_checks: list[bool] = []

    for record in records:
        inputs = _record_inputs(source_path, record.source_index)
        velocity = np.asarray(inputs.pop("velocity"), dtype=np.float32)
        if velocity.shape != (len(z_m), len(x_m)):
            raise ValueError(f"velocity shape mismatch for {record.sample_id}")
        if not np.isfinite(velocity).all() or float(velocity.min()) <= 0.0:
            raise ValueError(f"velocity is not positive and finite for {record.sample_id}")
        if requested_device.type == "cuda":
            torch.cuda.synchronize(requested_device)
        record_start = time.perf_counter()
        result = solver.simulate(
            velocity,
            source_x_m=inputs["source_x_m"],
            source_z_m=inputs["source_z_m"],
            source_f0_hz=inputs["source_f0_hz"],
            source_t0_s=inputs["source_t0_s"],
            source_amplitude=inputs["source_amplitude"],
        )
        if requested_device.type == "cuda":
            torch.cuda.synchronize(requested_device)
        elapsed_s = time.perf_counter() - record_start
        prediction = torch.from_numpy(result.wavefield)
        expected_shape = (1, len(time_s), len(z_m), len(x_m))
        if tuple(prediction.shape) != expected_shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} != {expected_shape}"
            )
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError(f"non-finite prediction for {record.sample_id}")
        if float(prediction[:, :, 0, :].abs().max()) != 0.0:
            raise ValueError(f"free top surface is nonzero for {record.sample_id}")
        if not np.isclose(result.source_map_saved.sum(), 1.0, atol=2.0e-7):
            raise ValueError(f"source map is not conservative for {record.sample_id}")
        if not np.array_equal(result.velocity_saved_mps[0], velocity):
            raise ValueError(f"factor-one velocity changed for {record.sample_id}")
        solver_qc = dict(result.metrics[0])
        if not bool(solver_qc["finite"]) or float(solver_qc["lwc_qmax"]) >= 1.0:
            raise FloatingPointError(
                f"unstable LWC84 symbol for {record.sample_id}: {solver_qc}"
            )

        sealed = seal_prediction(
            destination / "predictions" / f"{record.sample_id}.pt",
            prediction,
            metadata={
                "sample_id": record.sample_id,
                "sample_sha256": record.sample_sha256,
                "source_index": record.source_index,
                "output_restriction_factor": 1,
            },
        )
        sealed_at_ns = time.monotonic_ns()
        with h5py.File(source_path, "r", swmr=True) as handle:
            truth = torch.from_numpy(
                np.asarray(
                    handle["wavefield"][record.source_index], dtype=np.float32
                )
            )[None]
        truth_opened_at_ns = time.monotonic_ns()
        opened_after_seal = truth_opened_at_ns > sealed_at_ns and sealed.path.is_file()
        if not opened_after_seal:
            raise RuntimeError("future truth was opened before prediction sealing")
        order_checks.append(opened_after_seal)
        if truth.shape != prediction.shape or not bool(torch.isfinite(truth).all()):
            raise ValueError(f"truth field is invalid for {record.sample_id}")

        onset_index = int(
            np.searchsorted(time_s, float(inputs["source_t0_s"]), side="left")
        )
        if onset_index < 0 or onset_index >= len(time_s):
            raise ValueError(f"source onset is outside stored times for {record.sample_id}")
        record_metrics = accumulate_exact_metrics(
            prediction,
            truth,
            families=(record.medium_type,),
            group_ids=(record.group_id,),
            sample_ids=(record.sample_id,),
            source_onset_indices=(onset_index,),
            block_size=int(metric_block_size),
        )
        _update_accumulator(
            overall,
            prediction,
            truth,
            family=record.medium_type,
            group_id=record.group_id,
            sample_id=record.sample_id,
            onset_index=onset_index,
            block_size=int(metric_block_size),
        )
        receivers = _receiver_points(len(z_m), len(x_m))
        receiver_error = _receiver_relative_l2(prediction, truth, receivers)
        record_dir = destination / "records" / (
            f"{record.medium_type}_{record.sample_id}"
        )
        if make_plots:
            plot_velocity_source(
                velocity,
                float(inputs["source_x_m"]),
                float(inputs["source_z_m"]),
                x_m,
                z_m,
                record_dir / "velocity_model",
            )
            plot_wavefield_truth_coarse(
                truth[0].numpy(),
                prediction[0].numpy(),
                time_s,
                [0, onset_index, 100, 200, 300, 400],
                record_dir / "wavefield_comparison",
            )
            plot_receiver_truth_coarse(
                truth[0].numpy(),
                prediction[0].numpy(),
                time_s,
                receivers,
                record_dir / "receiver_comparison",
            )
        report = {
            "record": asdict(record),
            "source": dict(inputs),
            "source_onset_index": onset_index,
            "solver_runtime_seconds": elapsed_s,
            "solver_qc": solver_qc,
            "metrics": record_metrics,
            "receiver_relative_l2": receiver_error,
            "receiver_indices": receivers,
            "prediction_seal": asdict(sealed),
            "prediction_sealed_monotonic_ns": sealed_at_ns,
            "truth_opened_monotonic_ns": truth_opened_at_ns,
            "truth_opened_after_seal": opened_after_seal,
        }
        record_reports.append(report)
        _write_json_atomic(record_dir / "evaluation.json", report)
        print(
            json.dumps(
                {
                    "event": "record_complete",
                    "sample_id": record.sample_id,
                    "family": record.medium_type,
                    "relative_l2": record_metrics["aggregate_relative_l2"],
                    "solver_runtime_seconds": elapsed_s,
                    "sealed_sha256": sealed.sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    metrics = overall.finalize()
    gate = decide_coarse_gate(
        float(metrics["aggregate_relative_l2"]),
        metrics["family_relative_l2"],
    )
    wall_seconds = time.perf_counter() - wall_start
    summary = {
        "schema": "coarse_lwc84_201_development_gate_v1",
        "development_gate_only": True,
        "sealed_480x401_acceptance_completed": False,
        "record_count": len(records),
        "selected_records": [asdict(record) for record in records],
        "dataset_identity": _dataset_identity(source_path),
        "numerics": {
            "grid": grid.as_dict(),
            "boundaries": boundaries.as_dict(),
            "internal_dt_s": float(internal_dt_s),
            "stored_dt_s": time_step_s,
            "stored_time_count": len(time_s),
            "c_ref_mps": float(c_ref_mps),
            "dtype": "float32",
            "device": str(requested_device),
            "output_restriction_factor": 1,
        },
        "metrics": metrics,
        "record_reports": record_reports,
        "gate": asdict(gate),
        "truth_opened_after_seal": bool(order_checks) and all(order_checks),
        "receiver_traces_are_diagnostics_only": True,
        "runtime": {
            "run_started_utc": run_started,
            "wall_seconds": wall_seconds,
            "solver_seconds": sum(
                float(report["solver_runtime_seconds"])
                for report in record_reports
            ),
        },
        "sealed_acceptance_target": {
            "record_count": 480,
            "stored_time_count": 401,
            "aggregate_relative_l2_lt": 0.10,
            "every_family_relative_l2_lt": 0.12,
        },
    }
    _write_json_atomic(destination / "summary.json", summary)
    print(
        json.dumps(
            {
                "event": "gate_complete",
                "aggregate_relative_l2": metrics["aggregate_relative_l2"],
                "family_relative_l2": metrics["family_relative_l2"],
                "gate_action": gate.action,
                "wall_seconds": wall_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    run(
        source_h5=args.source_h5,
        output_dir=args.output_dir,
        device=args.device,
        internal_dt_s=args.internal_dt_s,
        npml=args.npml,
        c_ref_mps=args.c_ref_mps,
        seed=args.seed,
        per_family=args.per_family,
        metric_block_size=args.metric_block_size,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
