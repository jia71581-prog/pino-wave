#!/usr/bin/env python3
"""Render background-suppressed reflection diagnostics from an audited result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Marmousi result directory containing render_report.json and plotted data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New output directory; defaults to RESULT_DIR/reflection_diagnostics.",
    )
    parser.add_argument(
        "--display-percentile",
        type=float,
        default=99.5,
        help="Absolute scattered-field percentile used for robust symmetric color limits.",
    )
    return parser.parse_args()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    return float(
        np.linalg.norm(prediction64 - target64)
        / max(float(np.linalg.norm(target64)), 1.0e-30)
    )


def _norm_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(numerator, dtype=np.float64))
        / max(float(np.linalg.norm(np.asarray(denominator, dtype=np.float64))), 1.0e-30)
    )


def _robust_limit(value: np.ndarray, percentile: float) -> float:
    limit = float(np.percentile(np.abs(np.asarray(value, dtype=np.float64)), percentile))
    return max(limit, 1.0e-30)


def _plot_scattered_snapshots(
    *,
    true_scattered: np.ndarray,
    predicted_scattered: np.ndarray,
    time_s: np.ndarray,
    snapshot_indices: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    source_parameters: np.ndarray,
    percentile: float,
    output: Path,
) -> tuple[list[float], list[float]]:
    _configure_style()
    errors = predicted_scattered - true_scattered
    correction_to_scatter: list[float] = []
    correction_errors: list[float] = []
    extent = (
        float(x_m[0]) / 1000.0,
        float(x_m[-1]) / 1000.0,
        float(z_m[-1]) / 1000.0,
        float(z_m[0]) / 1000.0,
    )
    fig, axes = plt.subplots(
        3,
        len(snapshot_indices),
        figsize=(2.35 * len(snapshot_indices), 6.8),
        constrained_layout=True,
    )
    row_labels = (
        r"True scattering $p_{true}-P_{bg}$",
        r"Predicted correction $p_{pred}-P_{bg}$",
        "Correction error",
    )
    source_x_km = float(source_parameters[0]) / 1000.0
    source_z_km = float(source_parameters[1]) / 1000.0
    for column, frame_index in enumerate(snapshot_indices):
        true_frame = true_scattered[column]
        predicted_frame = predicted_scattered[column]
        error_frame = errors[column]
        limit = _robust_limit(true_frame, percentile)
        correction_ratio = _norm_ratio(predicted_frame, true_frame)
        correction_error = _relative_l2(predicted_frame, true_frame)
        correction_to_scatter.append(correction_ratio)
        correction_errors.append(correction_error)
        for row, panel in enumerate((true_frame, predicted_frame, error_frame)):
            ax = axes[row, column]
            ax.imshow(
                panel,
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
                extent=extent,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
            )
            ax.scatter(
                [source_x_km],
                [source_z_km],
                marker="*",
                s=32,
                c="yellow",
                edgecolors="black",
                linewidths=0.35,
            )
            if row == 0:
                ax.set_title(
                    f"t={time_s[frame_index]:.3f} s\n"
                    f"p{percentile:g} scale +/-{limit:.1e}"
                )
            elif row == 1:
                ax.set_title(f"||pred corr||/||true scat||={correction_ratio:.2e}")
            else:
                ax.set_title(f"scattered-field relL2={correction_error:.2%}")
            if column == 0:
                ax.set_ylabel(f"{row_labels[row]}\nz (km)")
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel("x (km)")
            else:
                ax.set_xticklabels([])
    fig.suptitle(
        "Best Phase4b Helmholtz A+1 - Marmousi background-suppressed reflections",
        fontsize=13,
        y=1.025,
    )
    fig.savefig(output)
    plt.close(fig)
    return correction_to_scatter, correction_errors


def _plot_scattered_gather(
    *,
    true_scattered: np.ndarray,
    predicted_scattered: np.ndarray,
    time_s: np.ndarray,
    receiver_x_m: np.ndarray,
    source_x_m: float,
    percentile: float,
    output: Path,
) -> None:
    _configure_style()
    error = predicted_scattered - true_scattered
    limit = _robust_limit(true_scattered, percentile)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), constrained_layout=True)
    panels = (
        (true_scattered, r"True scattering $p_{true}-P_{bg}$"),
        (predicted_scattered, r"Predicted correction $p_{pred}-P_{bg}$"),
        (error, "Correction error"),
    )
    mesh = None
    for ax, (panel, title) in zip(axes, panels, strict=True):
        mesh = ax.pcolormesh(
            time_s,
            receiver_x_m / 1000.0,
            panel.T,
            cmap="seismic",
            shading="auto",
            vmin=-limit,
            vmax=limit,
        )
        ax.axhline(source_x_m / 1000.0, color="lime", linewidth=0.8, linestyle=":")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("receiver x (km)")
    assert mesh is not None
    fig.colorbar(mesh, ax=axes, shrink=0.82, label="pressure")
    fig.suptitle(
        f"Near-surface background-suppressed receiver gather (shared p{percentile:g} scale)",
        fontsize=12,
    )
    fig.savefig(output)
    plt.close(fig)


def _plot_scattered_waveforms(
    *,
    true_scattered: np.ndarray,
    predicted_scattered: np.ndarray,
    time_s: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_z_m: float,
    output: Path,
) -> None:
    _configure_style()
    fig, axes = plt.subplots(3, 3, figsize=(12.4, 8.0), constrained_layout=True)
    for trace_index, ax in enumerate(axes.flat):
        true_trace = true_scattered[:, trace_index]
        predicted_trace = predicted_scattered[:, trace_index]
        limit = max(
            float(np.max(np.abs(true_trace))),
            float(np.max(np.abs(predicted_trace))),
            1.0e-30,
        )
        ax.plot(time_s, true_trace, linewidth=1.0, label="true scattering")
        ax.plot(
            time_s,
            predicted_trace,
            linewidth=1.0,
            linestyle="--",
            label="predicted correction",
        )
        ax.set_ylim(-1.08 * limit, 1.08 * limit)
        ax.grid(alpha=0.2)
        ax.set_title(
            f"x={receiver_x_m[trace_index] / 1000.0:.2f} km, "
            f"z={receiver_z_m / 1000.0:.2f} km\n"
            f"corr/scat={_norm_ratio(predicted_trace, true_trace):.2e}"
        )
        if trace_index % 3 == 0:
            ax.set_ylabel("pressure")
        if trace_index >= 6:
            ax.set_xlabel("time (s)")
        if trace_index == 0:
            ax.legend(loc="upper right")
    fig.suptitle(
        "Marmousi near-surface background-suppressed receiver waveforms",
        fontsize=13,
    )
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    percentile = float(args.display_percentile)
    if not 90.0 <= percentile < 100.0:
        raise ValueError("--display-percentile must lie in [90, 100)")
    result_dir = args.result_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else result_dir / "reflection_diagnostics"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    report_path = result_dir / "render_report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete":
        raise ValueError("source render report is not complete")
    plotted_data_path = Path(str(report["outputs"]["plotted_data"])).resolve()
    sample_h5_path = Path(str(report["sample_h5"])).resolve()
    with np.load(plotted_data_path, allow_pickle=False) as plotted:
        time_s = np.asarray(plotted["time_s"], dtype=np.float32)
        snapshot_indices = np.asarray(plotted["snapshot_indices"], dtype=np.int64)
        snapshot_target = np.asarray(plotted["snapshot_target"], dtype=np.float32)
        snapshot_prediction = np.asarray(plotted["snapshot_prediction"], dtype=np.float32)
        receiver_x_indices = np.asarray(plotted["receiver_x_indices"], dtype=np.int64)
        receiver_z_index = int(np.asarray(plotted["receiver_z_index"]).item())
        receiver_target = np.asarray(plotted["receiver_target"], dtype=np.float32)
        receiver_prediction = np.asarray(plotted["receiver_prediction"], dtype=np.float32)
        x_m = np.asarray(plotted["x_m"], dtype=np.float32)
        z_m = np.asarray(plotted["z_m"], dtype=np.float32)
        source_parameters = np.asarray(plotted["source_parameters"], dtype=np.float32)
    with h5py.File(sample_h5_path, "r", swmr=True) as handle:
        if str(handle.attrs.get("status", "")) != "complete":
            raise ValueError("source HDF5 sample is not complete")
        h5_time_s = np.asarray(handle["time_s"][:], dtype=np.float32)
        background = np.asarray(handle["background_pbg"][:], dtype=np.float32)
        target = np.asarray(handle["wavefield_target"][:], dtype=np.float32)
    expected_shape = (len(time_s), len(z_m), len(x_m))
    if target.shape != expected_shape or background.shape != expected_shape:
        raise ValueError("source wavefield shapes disagree with plotted coordinates")
    if not np.array_equal(h5_time_s, time_s):
        raise ValueError("source HDF5 and plotted time axes disagree")
    if not np.array_equal(target[snapshot_indices], snapshot_target):
        raise ValueError("plotted target snapshots no longer match the audited sample")
    observed_background_rel = _relative_l2(background, target)
    if not np.isclose(
        observed_background_rel,
        float(report["background_only_relative_l2"]),
        rtol=2.0e-5,
        atol=1.0e-10,
    ):
        raise ValueError("background field no longer reproduces the render report")

    snapshot_background = background[snapshot_indices]
    true_scattered_snapshots = snapshot_target - snapshot_background
    predicted_scattered_snapshots = snapshot_prediction - snapshot_background
    receiver_background = background[:, receiver_z_index, receiver_x_indices]
    true_scattered_receivers = receiver_target - receiver_background
    predicted_scattered_receivers = receiver_prediction - receiver_background
    receiver_x_m = x_m[receiver_x_indices]
    receiver_z_m = float(z_m[receiver_z_index])

    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = output_dir / "marmousi_scattered_wavefield_snapshots.png"
    gather_path = output_dir / "marmousi_scattered_receiver_gather.png"
    waveform_path = output_dir / "marmousi_scattered_receiver_waveforms.png"
    data_path = output_dir / "marmousi_reflection_diagnostics_data.npz"
    diagnostics_path = output_dir / "reflection_diagnostics_report.json"
    correction_ratios, correction_errors = _plot_scattered_snapshots(
        true_scattered=true_scattered_snapshots,
        predicted_scattered=predicted_scattered_snapshots,
        time_s=time_s,
        snapshot_indices=snapshot_indices,
        x_m=x_m,
        z_m=z_m,
        source_parameters=source_parameters,
        percentile=percentile,
        output=snapshot_path,
    )
    _plot_scattered_gather(
        true_scattered=true_scattered_receivers,
        predicted_scattered=predicted_scattered_receivers,
        time_s=time_s,
        receiver_x_m=receiver_x_m,
        source_x_m=float(source_parameters[0]),
        percentile=percentile,
        output=gather_path,
    )
    _plot_scattered_waveforms(
        true_scattered=true_scattered_receivers,
        predicted_scattered=predicted_scattered_receivers,
        time_s=time_s,
        receiver_x_m=receiver_x_m,
        receiver_z_m=receiver_z_m,
        output=waveform_path,
    )
    np.savez_compressed(
        data_path,
        time_s=time_s,
        snapshot_indices=snapshot_indices,
        snapshot_background=snapshot_background,
        snapshot_true_scattered=true_scattered_snapshots,
        snapshot_predicted_scattered=predicted_scattered_snapshots,
        receiver_x_indices=receiver_x_indices,
        receiver_z_index=np.asarray(receiver_z_index),
        receiver_background=receiver_background,
        receiver_true_scattered=true_scattered_receivers,
        receiver_predicted_scattered=predicted_scattered_receivers,
        x_m=x_m,
        z_m=z_m,
        source_parameters=source_parameters,
    )
    per_frame = []
    for column, frame_index in enumerate(snapshot_indices):
        true_scattered = true_scattered_snapshots[column]
        target_frame = snapshot_target[column]
        per_frame.append(
            {
                "frame_index": int(frame_index),
                "time_s": float(time_s[frame_index]),
                "true_scattered_to_total_l2_ratio": _norm_ratio(
                    true_scattered, target_frame
                ),
                "true_scattered_to_total_peak_ratio": float(
                    np.max(np.abs(true_scattered))
                    / max(float(np.max(np.abs(target_frame))), 1.0e-30)
                ),
                "predicted_correction_to_true_scattered_l2_ratio": correction_ratios[
                    column
                ],
                "scattered_field_relative_l2": correction_errors[column],
            }
        )
    diagnostics = {
        "status": "complete",
        "schema": "phase4b_reflection_diagnostics_v1",
        "definition": {
            "true_scattering": "p_true - P_bg",
            "predicted_correction": "p_prediction - P_bg",
            "correction_error": "predicted_correction - true_scattering",
        },
        "display_absolute_percentile": percentile,
        "source_render_report": str(report_path),
        "source_plotted_data": str(plotted_data_path),
        "source_sample_h5": str(sample_h5_path),
        "checkpoint": str(report["checkpoint"]),
        "checkpoint_sha256": str(report["checkpoint_sha256"]),
        "background_to_target_relative_l2": observed_background_rel,
        "receiver_true_scattered_to_total_l2_ratio": _norm_ratio(
            true_scattered_receivers, receiver_target
        ),
        "receiver_predicted_correction_to_true_scattered_l2_ratio": _norm_ratio(
            predicted_scattered_receivers, true_scattered_receivers
        ),
        "receiver_scattered_field_relative_l2": _relative_l2(
            predicted_scattered_receivers, true_scattered_receivers
        ),
        "per_snapshot": per_frame,
        "outputs": {
            "scattered_wavefield_snapshots": str(snapshot_path),
            "scattered_receiver_gather": str(gather_path),
            "scattered_receiver_waveforms": str(waveform_path),
            "diagnostic_data": str(data_path),
        },
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "report": str(diagnostics_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
