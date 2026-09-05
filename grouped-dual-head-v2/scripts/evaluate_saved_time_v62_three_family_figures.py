#!/usr/bin/env python
"""Generate three-family visual diagnostics for a saved-time V62 checkpoint.

The script evaluates one validation representative from each medium family on the
exact stored time axis only.  It saves velocity/source figures, wavefield
snapshot comparisons, receiver-like point traces, per-family arrays, and a JSON
summary under the requested artifact directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES
from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.evaluation import sha256_file, time_axis_sha256
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import _load_context, _load_parent_model


COLORS = {"target": "#0072B2", "prediction": "#D55E00"}


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
    return float(
        np.linalg.norm(prediction.astype(np.float64) - target.astype(np.float64))
        / max(np.linalg.norm(target.astype(np.float64)), 1.0e-20)
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def representative_validation_indices(manifest) -> dict[str, int]:
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


def snapshot_indices(time_s: np.ndarray, source_t0_s: float, *, count: int) -> np.ndarray:
    axis = np.asarray(time_s, dtype=np.float64)
    onset = int(np.searchsorted(axis, float(source_t0_s), side="left"))
    onset = min(max(onset, 0), len(axis) - 1)
    end_time = min(float(axis[-1]), float(source_t0_s) + 0.60)
    end = int(np.searchsorted(axis, end_time, side="right")) - 1
    end = min(max(end, onset), len(axis) - 1)
    if end - onset + 1 < count:
        return np.linspace(onset, len(axis) - 1, count).round().astype(np.int64)
    return np.linspace(onset, end, count).round().astype(np.int64)


def receiver_grid_indices(width: int, height: int) -> tuple[tuple[int, int], ...]:
    xs = np.linspace(0.15, 0.85, 3)
    zs = np.linspace(0.10, 0.55, 3)
    points: list[tuple[int, int]] = []
    for z in zs:
        for x in xs:
            points.append((int(round(z * (height - 1))), int(round(x * (width - 1)))))
    return tuple(points)


def source_location(record) -> tuple[float, float]:
    params = record.source_parameters.detach().cpu().numpy()
    return float(params[0]), float(params[1])


def _plot_velocity(record, family: str, output_dir: Path) -> None:
    _style()
    velocity = record.velocity_mps.squeeze(0).detach().cpu().numpy()
    x_m = record.x_m.detach().cpu().numpy()
    z_m = record.z_m.detach().cpu().numpy()
    sx_m, sz_m = source_location(record)
    extent = (x_m[0] / 1000.0, x_m[-1] / 1000.0, z_m[-1] / 1000.0, z_m[0] / 1000.0)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    image = ax.imshow(velocity, cmap="viridis", extent=extent, origin="upper", aspect="equal")
    ax.scatter([sx_m / 1000.0], [sz_m / 1000.0], marker="*", s=150, c="#D55E00", edgecolors="white", linewidths=0.8)
    ax.set_title(f"{family}: velocity model and source")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    cbar = fig.colorbar(image, ax=ax, shrink=0.86)
    cbar.set_label("velocity (m/s)")
    ax.text(
        sx_m / 1000.0,
        sz_m / 1000.0,
        f"  source\n  ({sx_m:.0f} m, {sz_m:.0f} m)",
        color="white",
        fontsize=8,
        va="center",
        ha="left",
        path_effects=[],
    )
    _save(fig, output_dir / f"{family}_velocity_source")


def _plot_wavefield_snapshots(
    target: np.ndarray,
    prediction: np.ndarray,
    times: np.ndarray,
    record,
    family: str,
    output_dir: Path,
) -> None:
    _style()
    x_m = record.x_m.detach().cpu().numpy()
    z_m = record.z_m.detach().cpu().numpy()
    sx_m, sz_m = source_location(record)
    extent = (x_m[0] / 1000.0, x_m[-1] / 1000.0, z_m[-1] / 1000.0, z_m[0] / 1000.0)
    limit = max(float(np.max(np.abs(target))), float(np.max(np.abs(prediction))), 1.0e-20)
    error = prediction - target
    err_limit = max(float(np.max(np.abs(error))), 1.0e-20)
    fig, axes = plt.subplots(3, len(times), figsize=(2.15 * len(times), 6.2), constrained_layout=True)
    rows = ("true p", "predicted p", "error")
    for col, t_s in enumerate(times):
        panels = (target[col], prediction[col], error[col])
        for row, panel in enumerate(panels):
            ax = axes[row, col]
            vmax = limit if row < 2 else err_limit
            image = ax.imshow(panel, cmap="seismic", vmin=-vmax, vmax=vmax, extent=extent, origin="upper", aspect="equal")
            ax.scatter([sx_m / 1000.0], [sz_m / 1000.0], marker="*", s=38, c="yellow", edgecolors="black", linewidths=0.4)
            if row == 0:
                ax.set_title(f"t={t_s:.3f}s")
            if col == 0:
                ax.set_ylabel(f"{rows[row]}\nz (km)")
            else:
                ax.set_yticks([])
            if row == 2:
                ax.set_xlabel("x (km)")
            else:
                ax.set_xticks([])
        cbar = fig.colorbar(image, ax=axes[:, col], shrink=0.72)
        cbar.set_label("Pa")
    fig.suptitle(f"{family}: exact saved-time wavefield snapshots", y=1.02)
    _save(fig, output_dir / f"{family}_wavefield_snapshots")


def _plot_receiver_traces(
    target: np.ndarray,
    prediction: np.ndarray,
    times: np.ndarray,
    record,
    family: str,
    output_dir: Path,
) -> dict[str, object]:
    _style()
    height, width = target.shape[-2:]
    points = receiver_grid_indices(width, height)
    x_m = record.x_m.detach().cpu().numpy()
    z_m = record.z_m.detach().cpu().numpy()
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 6.4), sharex=True, constrained_layout=True)
    trace_metrics: list[dict[str, object]] = []
    for ax, (iz, ix) in zip(axes.flat, points, strict=True):
        y_true = target[:, iz, ix]
        y_pred = prediction[:, iz, ix]
        trace_metrics.append(
            {
                "ix": int(ix),
                "iz": int(iz),
                "x_m": float(x_m[ix]),
                "z_m": float(z_m[iz]),
                "relative_l2": _relative_l2(y_pred, y_true),
            }
        )
        ax.plot(times, y_true, color=COLORS["target"], lw=1.0, label="true")
        ax.plot(times, y_pred, color=COLORS["prediction"], lw=0.9, ls="--", label="pred")
        ax.set_title(f"x={x_m[ix]/1000:.2f} km, z={z_m[iz]/1000:.2f} km")
        ax.grid(alpha=0.25, lw=0.4)
    axes[0, 0].legend(loc="upper right")
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("pressure (Pa)")
    fig.suptitle(f"{family}: receiver-point waveform diagnostics", y=1.02)
    _save(fig, output_dir / f"{family}_receiver_waveforms")
    return {"points": trace_metrics, "mean_relative_l2": float(np.mean([m["relative_l2"] for m in trace_metrics]))}


@torch.inference_mode()
def evaluate(
    *,
    config_path: Path,
    checkpoint_path: Path,
    run_identity_path: Path,
    output_dir: Path,
    device: torch.device,
    time_block: int,
    snapshot_count: int,
) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text())
    run_identity = json.loads(run_identity_path.read_text())
    base, manifest, parent_identity = _load_context(config)
    if run_identity["manifest_digest"] != manifest.digest:
        raise ValueError("run identity and dataset manifest disagree")

    model = _load_parent_model(config, base, manifest, parent_identity, device)
    metadata = load_checkpoint(
        checkpoint_path,
        model=model,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=run_identity["run_digest"],
        map_location=device,
    )
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)
    dataset = V3WavefieldDataset(base.data.source_h5, manifest, split="validation")
    selected = representative_validation_indices(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_global_step": int(metadata.global_step),
        "config": str(config_path),
        "run_identity": str(run_identity_path),
        "manifest_digest": manifest.digest,
        "time_axis_sha256": time_axis_sha256(manifest.time_s),
        "stored_times_only": True,
        "interpolated_targets": 0,
        "time_block": int(time_block),
        "families": {},
    }

    all_snapshot_rel: list[float] = []
    all_trace_rel: list[float] = []
    for family, dataset_index in selected.items():
        family_dir = output_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        record = dataset[dataset_index]
        full_times = record.time_s.detach().cpu().numpy().astype(np.float32)
        target_physical = dataset.read_wavefield(dataset_index, record.time_s)
        if not bool(target_physical.exact.all()) or not torch.equal(
            target_physical.left_index, target_physical.right_index
        ):
            raise RuntimeError("evaluation unexpectedly requested interpolated targets")

        velocity = record.velocity_mps.unsqueeze(0).to(device)
        source = record.source_parameters.unsqueeze(0).to(device)
        source_map = record.source_map.unsqueeze(0).to(device)
        prepared = model.prepare_sources(
            model.encode_medium(velocity, normalizer),
            source,
            source_map,
            normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
        )
        prediction_norm = model.dense_normalized(
            prepared,
            record.time_s.unsqueeze(0).to(device),
            x_m=record.x_m.to(device),
            z_m=record.z_m.to(device),
            time_block=int(time_block),
        )
        prediction = (
            normalizer.decode_pressure(prediction_norm.float(), source[:, 4])
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        target = target_physical.values.detach().cpu().numpy().astype(np.float32)
        source_t0_s = float(record.source_parameters[3].item())
        snap_idx = snapshot_indices(full_times, source_t0_s, count=snapshot_count)
        _plot_velocity(record, family, family_dir)
        _plot_wavefield_snapshots(
            target[snap_idx],
            prediction[snap_idx],
            full_times[snap_idx],
            record,
            family,
            family_dir,
        )
        receiver_report = _plot_receiver_traces(
            target,
            prediction,
            full_times,
            record,
            family,
            family_dir,
        )
        snapshot_rel = _relative_l2(prediction[snap_idx], target[snap_idx])
        full_rel = _relative_l2(prediction, target)
        trace_rel = float(receiver_report["mean_relative_l2"])
        all_snapshot_rel.append(snapshot_rel)
        all_trace_rel.append(trace_rel)
        np.savez_compressed(
            family_dir / f"{family}_representative_arrays.npz",
            target=target,
            prediction=prediction,
            time_s=full_times,
            velocity_mps=record.velocity_mps.squeeze(0).detach().cpu().numpy().astype(np.float32),
            x_m=record.x_m.detach().cpu().numpy().astype(np.float32),
            z_m=record.z_m.detach().cpu().numpy().astype(np.float32),
            source_parameters=record.source_parameters.detach().cpu().numpy().astype(np.float32),
            snapshot_indices=snap_idx.astype(np.int64),
        )
        report["families"][family] = {
            "dataset_index": int(dataset_index),
            "sample_id": record.sample_id,
            "group_id": record.group_id,
            "source_parameters": [float(v) for v in record.source_parameters.detach().cpu().tolist()],
            "snapshot_indices": [int(v) for v in snap_idx.tolist()],
            "snapshot_times_s": [float(v) for v in full_times[snap_idx].tolist()],
            "full_wavefield_relative_l2": full_rel,
            "snapshot_relative_l2": snapshot_rel,
            "receiver_mean_relative_l2": trace_rel,
            "receiver_points": receiver_report["points"],
            "outputs": {
                "velocity_source_png": str(family_dir / f"{family}_velocity_source.png"),
                "wavefield_snapshots_png": str(family_dir / f"{family}_wavefield_snapshots.png"),
                "receiver_waveforms_png": str(family_dir / f"{family}_receiver_waveforms.png"),
                "arrays_npz": str(family_dir / f"{family}_representative_arrays.npz"),
            },
        }
        print(
            json.dumps(
                {
                    "event": "family_complete",
                    "family": family,
                    "dataset_index": int(dataset_index),
                    "full_wavefield_relative_l2": full_rel,
                    "snapshot_relative_l2": snapshot_rel,
                    "receiver_mean_relative_l2": trace_rel,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report["aggregate"] = {
        "mean_snapshot_relative_l2": float(np.mean(all_snapshot_rel)),
        "mean_receiver_relative_l2": float(np.mean(all_trace_rel)),
    }
    _atomic_json(report, output_dir / "three_family_figure_evaluation_report.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time-block", type=int, default=8)
    parser.add_argument("--snapshot-count", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    report = evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        run_identity_path=args.run_identity,
        output_dir=args.output_dir,
        device=device,
        time_block=args.time_block,
        snapshot_count=args.snapshot_count,
    )
    print(json.dumps({"event": "complete", "aggregate": report["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
