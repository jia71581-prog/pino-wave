#!/usr/bin/env python
"""Evaluate a continuous-time V3 pilot checkpoint on held-out physical wavefields."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES, V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest
from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset
from grouped_ufno_mionet_v3.training.checkpoint import CHECKPOINT_FORMAT
from scripts.train_grouped_v3 import build_model
from scripts.train_grouped_v3_pilot import load_normalizer


COLORS = {"target": "#0072B2", "prediction": "#D55E00"}


def validate_evaluation_identity(
    identity: Mapping[str, object],
    checkpoint: Mapping[str, object],
    *,
    manifest_digest: str,
    model_config_digest: str,
) -> str:
    if identity.get("manifest_digest") != manifest_digest:
        raise ValueError("evaluation run identity manifest mismatch")
    pilot_config_digest = identity.get("pilot_config_digest")
    if pilot_config_digest is not None and pilot_config_digest != model_config_digest:
        raise ValueError("evaluation model config mismatch")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("evaluation checkpoint is not V3")
    if checkpoint.get("manifest_digest") != manifest_digest:
        raise ValueError("evaluation checkpoint manifest mismatch")
    current_digest = identity.get("run_digest")
    allowed_digests: set[str] = set()
    if isinstance(current_digest, str) and current_digest:
        allowed_digests.add(current_digest)
    parent = identity.get("parent")
    if isinstance(parent, Mapping):
        parent_digest = parent.get("run_digest")
        if isinstance(parent_digest, str) and parent_digest:
            allowed_digests.add(parent_digest)
    checkpoint_digest = checkpoint.get("config_digest")
    if checkpoint_digest not in allowed_digests:
        raise ValueError("evaluation checkpoint run mismatch")
    return str(checkpoint_digest)


def representative_validation_indices(manifest) -> dict[str, int]:
    """Return dataset-local indices for the first held-out record of each family."""
    selected: dict[str, int] = {}
    local_index = 0
    for record in manifest.records:
        if record.split != "validation":
            continue
        if record.medium_type in ALLOWED_MEDIUM_TYPES and record.medium_type not in selected:
            selected[record.medium_type] = local_index
        local_index += 1
    missing = [family for family in ALLOWED_MEDIUM_TYPES if family not in selected]
    if missing:
        raise ValueError(f"validation split is missing representative families: {missing}")
    return {family: selected[family] for family in ALLOWED_MEDIUM_TYPES}


def active_evaluation_times(
    time_s: torch.Tensor,
    *,
    source_t0_s: float,
    active_horizon_s: float = 0.60,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose three saved frames and three strict midpoints over active propagation."""
    axis = torch.as_tensor(time_s, dtype=torch.float64)
    if axis.ndim != 1 or len(axis) < 7 or active_horizon_s <= 0:
        raise ValueError("evaluation time axis or active horizon is invalid")
    onset = min(
        int(torch.searchsorted(axis, torch.tensor(source_t0_s, dtype=axis.dtype))),
        len(axis) - 2,
    )
    end_time = min(float(axis[-2]), float(source_t0_s) + float(active_horizon_s))
    end = min(
        int(torch.searchsorted(axis, torch.tensor(end_time, dtype=axis.dtype))),
        len(axis) - 2,
    )
    indices = np.arange(onset, end + 1, dtype=np.int64)
    phases = np.array_split(indices, 3)
    if any(len(phase) == 0 for phase in phases):
        raise ValueError("active evaluation horizon cannot form three phases")
    exact_indices = torch.tensor([int(phase[len(phase) // 2]) for phase in phases])
    exact = axis[exact_indices]
    midpoint = 0.5 * (exact + axis[exact_indices + 1])
    return exact.float(), midpoint.float()


def bilinear_sample_frames(
    frames: torch.Tensor,
    *,
    x_m: torch.Tensor,
    z_m: torch.Tensor,
    points_xy_m: torch.Tensor,
) -> torch.Tensor:
    """Sample ``[time,z,x]`` fields at arbitrary physical x/z coordinates."""
    values = torch.as_tensor(frames, dtype=torch.float32)
    x = torch.as_tensor(x_m, dtype=torch.float32)
    z = torch.as_tensor(z_m, dtype=torch.float32)
    points = torch.as_tensor(points_xy_m, dtype=torch.float32)
    if values.ndim != 3 or values.shape[-2:] != (len(z), len(x)):
        raise ValueError("wavefield frames and physical grid disagree")
    if points.ndim != 2 or points.shape[-1] != 2 or len(points) == 0:
        raise ValueError("arbitrary points must have shape [point,2]")
    if torch.any(points[:, 0] < x[0]) or torch.any(points[:, 0] > x[-1]):
        raise ValueError("arbitrary x coordinate lies outside the grid")
    if torch.any(points[:, 1] < z[0]) or torch.any(points[:, 1] > z[-1]):
        raise ValueError("arbitrary z coordinate lies outside the grid")
    normalized = torch.stack(
        (
            2.0 * (points[:, 0] - x[0]) / (x[-1] - x[0]) - 1.0,
            2.0 * (points[:, 1] - z[0]) / (z[-1] - z[0]) - 1.0,
        ),
        dim=-1,
    )
    grid = normalized[None, None].expand(len(values), -1, -1, -1)
    sampled = F.grid_sample(
        values[:, None], grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled[:, 0, 0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1.0e-20))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _plot_wavefields(
    target: np.ndarray,
    prediction: np.ndarray,
    times: np.ndarray,
    *,
    x_m: np.ndarray,
    z_m: np.ndarray,
    family: str,
    output: Path,
) -> None:
    _style()
    amplitude_limit = max(float(np.abs(target).max()), float(np.abs(prediction).max()), 1e-20)
    error = np.abs(prediction - target)
    error_limit = max(float(error.max()), 1e-20)
    fig, axes = plt.subplots(6, 3, figsize=(7.0, 10.0), constrained_layout=True)
    extent = (x_m[0] / 1000.0, x_m[-1] / 1000.0, z_m[-1] / 1000.0, z_m[0] / 1000.0)
    row_labels = (
        "Saved target",
        "Saved prediction",
        "Saved abs. error",
        "Midpoint target",
        "Midpoint prediction",
        "Midpoint abs. error",
    )
    for column in range(3):
        for local_row, frame_index in enumerate((column, column + 3)):
            row = 0 if frame_index < 3 else 3
            axes[row, column].imshow(
                target[frame_index], cmap="RdBu_r", vmin=-amplitude_limit, vmax=amplitude_limit,
                extent=extent, aspect="equal",
            )
            axes[row + 1, column].imshow(
                prediction[frame_index], cmap="RdBu_r", vmin=-amplitude_limit,
                vmax=amplitude_limit, extent=extent, aspect="equal",
            )
            axes[row + 2, column].imshow(
                error[frame_index], cmap="magma", vmin=0.0, vmax=error_limit,
                extent=extent, aspect="equal",
            )
            kind = "saved" if local_row == 0 else "midpoint"
            axes[row, column].set_title(f"{kind} t={times[frame_index]:.4f} s")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(f"{label}\nz (km)")
        for column in range(3):
            axes[row, column].set_xlabel("x (km)")
    amplitude_map = ScalarMappable(norm=Normalize(-amplitude_limit, amplitude_limit), cmap="RdBu_r")
    error_map = ScalarMappable(norm=Normalize(0.0, error_limit), cmap="magma")
    fig.colorbar(amplitude_map, ax=axes[[0, 1, 3, 4], :], shrink=0.55, label="Pressure (Pa)")
    fig.colorbar(error_map, ax=axes[[2, 5], :], shrink=0.55, label="Absolute error (Pa)")
    fig.suptitle(f"Held-out {family}: full 201×201 wavefield", fontweight="bold")
    _save(fig, output)


def _plot_point_traces(
    time_s: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    points: np.ndarray,
    *,
    family: str,
    output: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(len(points), 1, figsize=(6.75, 5.2), sharex=True, constrained_layout=True)
    for index, axis in enumerate(np.atleast_1d(axes)):
        axis.plot(time_s, target[index], color=COLORS["target"], label="Numerical target")
        axis.plot(
            time_s,
            prediction[index],
            color=COLORS["prediction"],
            ls="--",
            label="V3 arbitrary-point query",
        )
        axis.set_ylabel(f"Point {index + 1}\nPressure (Pa)")
        axis.text(
            0.99,
            0.88,
            f"({points[index, 0]:.1f}, {points[index, 1]:.1f}) m\nrel. L2={_relative_l2(prediction[index], target[index]):.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
        )
        axis.grid(alpha=0.15)
    axes[0].legend(ncol=2, loc="upper left")
    axes[-1].set_xlabel("Time after simulation start (s)")
    fig.suptitle(f"Held-out {family}: off-grid point traces", fontweight="bold")
    _save(fig, output)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _point_coordinates(points: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    rows = []
    for point in points:
        rows.append(
            torch.stack(
                (torch.full_like(times, point[0]), torch.full_like(times, point[1]), times),
                dim=-1,
            )
        )
    return torch.cat(rows, dim=0)[None]


@torch.no_grad()
def _evaluate_record(model, dataset, index: int, normalizer, device: torch.device, output: Path):
    record = dataset[index]
    exact_times, midpoint_times = active_evaluation_times(
        record.time_s, source_t0_s=float(record.source_parameters[3])
    )
    field_times = torch.cat((exact_times, midpoint_times))
    field_target = dataset.read_wavefield(index, field_times).values
    onset = int(torch.searchsorted(record.time_s, record.source_parameters[3]))
    trace_end = int(
        torch.searchsorted(
            record.time_s,
            torch.tensor(float(record.source_parameters[3]) + 0.60, dtype=record.time_s.dtype),
        )
    )
    trace_end = min(trace_end, len(record.time_s) - 1)
    trace_times = record.time_s[onset : trace_end + 1]
    trace_target = dataset.read_wavefield(index, trace_times).values
    x_extent = float(record.x_m[-1] - record.x_m[0])
    z_extent = float(record.z_m[-1] - record.z_m[0])
    points = torch.tensor(
        [
            [float(record.x_m[0]) + 0.17 * x_extent, float(record.z_m[0]) + 0.23 * z_extent],
            [float(record.x_m[0]) + 0.49 * x_extent, float(record.z_m[0]) + 0.52 * z_extent],
            [float(record.x_m[0]) + 0.83 * x_extent, float(record.z_m[0]) + 0.71 * z_extent],
        ],
        dtype=torch.float32,
    )

    _sync(device)
    started = time.perf_counter()
    medium = model.encode_medium(record.velocity_mps[None].to(device), normalizer)
    prepared = model.prepare_sources(
        medium,
        record.source_parameters[None].to(device),
        record.source_map[None].to(device),
        normalizer,
    )
    _sync(device)
    encode_seconds = time.perf_counter() - started

    started = time.perf_counter()
    field_prediction = model.predict_wavefield(
        prepared,
        field_times.to(device),
        x_m=record.x_m.to(device),
        z_m=record.z_m.to(device),
        time_block=1,
    )[0]
    _sync(device)
    dense_seconds = time.perf_counter() - started

    point_coords = _point_coordinates(points, trace_times).to(device)
    started = time.perf_counter()
    trace_prediction = model.query_pressure(prepared, point_coords, chunk_size=2048)[0]
    _sync(device)
    query_seconds = time.perf_counter() - started
    trace_prediction = trace_prediction.reshape(len(points), len(trace_times)).cpu()
    trace_target_points = bilinear_sample_frames(
        trace_target, x_m=record.x_m, z_m=record.z_m, points_xy_m=points
    ).T

    consistency_coords = _point_coordinates(points, field_times).to(device)
    point_field_prediction = model.query_pressure(
        prepared, consistency_coords, chunk_size=2048
    )[0].reshape(len(points), len(field_times)).T.cpu()
    dense_point_prediction = bilinear_sample_frames(
        field_prediction.cpu(), x_m=record.x_m, z_m=record.z_m, points_xy_m=points
    )

    target_np = field_target.numpy()
    prediction_np = field_prediction.cpu().numpy()
    trace_target_np = trace_target_points.numpy()
    trace_prediction_np = trace_prediction.numpy()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "arrays.npz",
        exact_time_s=exact_times.numpy(),
        midpoint_time_s=midpoint_times.numpy(),
        target_wavefield_pa=target_np,
        prediction_wavefield_pa=prediction_np,
        trace_time_s=trace_times.numpy(),
        trace_points_xy_m=points.numpy(),
        target_point_traces_pa=trace_target_np,
        prediction_point_traces_pa=trace_prediction_np,
    )
    _plot_wavefields(
        target_np,
        prediction_np,
        field_times.numpy(),
        x_m=record.x_m.numpy(),
        z_m=record.z_m.numpy(),
        family=record.medium_type,
        output=output / "wavefield_comparison",
    )
    _plot_point_traces(
        trace_times.numpy(),
        trace_target_np,
        trace_prediction_np,
        points.numpy(),
        family=record.medium_type,
        output=output / "arbitrary_point_traces",
    )
    return {
        "sample_id": record.sample_id,
        "group_id": record.group_id,
        "medium_type": record.medium_type,
        "source_index": record.source_index,
        "exact_time_s": exact_times.tolist(),
        "midpoint_time_s": midpoint_times.tolist(),
        "metrics": {
            "exact_dense_relative_l2": _relative_l2(prediction_np[:3], target_np[:3]),
            "midpoint_dense_relative_l2": _relative_l2(prediction_np[3:], target_np[3:]),
            "arbitrary_point_trace_relative_l2": _relative_l2(
                trace_prediction_np, trace_target_np
            ),
            "query_dense_point_consistency_relative_l2": _relative_l2(
                point_field_prediction.numpy(), dense_point_prediction.numpy()
            ),
        },
        "timing_seconds": {
            "encode_medium_and_source_once": encode_seconds,
            "six_full_wavefield_snapshots": dense_seconds,
            "three_off_grid_point_traces": query_seconds,
        },
        "artifacts": {
            "arrays": str((output / "arrays.npz").resolve()),
            "wavefield_png": str((output / "wavefield_comparison.png").resolve()),
            "wavefield_pdf": str((output / "wavefield_comparison.pdf").resolve()),
            "point_trace_png": str((output / "arbitrary_point_traces.png").resolve()),
            "point_trace_pdf": str((output / "arbitrary_point_traces.pdf").resolve()),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-identity", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    manifest = build_manifest(config.data.source_h5)
    with Path(args.run_identity).open(encoding="utf8") as handle:
        identity = json.load(handle)
    checkpoint_path = Path(args.checkpoint).resolve()
    device = torch.device(args.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_origin_run_digest = validate_evaluation_identity(
        identity,
        payload,
        manifest_digest=manifest.digest,
        model_config_digest=config.digest(),
    )
    normalizer = load_normalizer(config, manifest.digest)
    model = build_model(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    dataset = V3WavefieldDataset(config.data.source_h5, manifest, split="validation")
    selected = representative_validation_indices(manifest)
    output = Path(args.output_dir)
    records = {
        family: _evaluate_record(model, dataset, index, normalizer, device, output / family)
        for family, index in selected.items()
    }
    report = {
        "schema": "grouped_v3_checkpoint_evaluation_v2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "manifest_digest": manifest.digest,
        "run_digest": identity["run_digest"],
        "checkpoint_origin_run_digest": checkpoint_origin_run_digest,
        "records": records,
        "note": (
            "All examples are held-out validation records. The model inputs are medium and one "
            "source only; plotted points are post-hoc arbitrary-coordinate queries, never receivers "
            "or conditioning inputs. Midpoint targets are interpolated between numerical frames."
        ),
    }
    _atomic_json(report, output / "evaluation_report.json")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
