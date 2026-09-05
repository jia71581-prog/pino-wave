#!/usr/bin/env python3
"""Render an audited linear superposition of eight fixed-frequency source fields."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from saved_time_phase_operator_v4.evaluation import sha256_file  # noqa: E402


CASE_ORDER = (
    "interp_left_shallow",
    "interp_left_deep",
    "interp_center",
    "interp_right_shallow",
    "interp_right_deep",
    "extrap_left",
    "extrap_right",
    "extrap_deep",
)
SNAPSHOT_INDICES = (80, 144, 208, 272, 336, 400)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    denominator = max(float(np.linalg.norm(target64.ravel())), 1.0e-30)
    return float(np.linalg.norm((prediction64 - target64).ravel()) / denominator)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _plot(
    output_stem: Path,
    *,
    velocity_mps: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    time_s: np.ndarray,
    source_parameters: np.ndarray,
    target_sum: np.ndarray,
    prediction_sum: np.ndarray,
    all_time_relative_l2: float,
) -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    indices = np.asarray(SNAPSHOT_INDICES, dtype=np.int64)
    target = target_sum[indices]
    prediction = prediction_sum[indices]
    error = prediction - target
    frame_errors = [_relative_l2(p, t) for p, t in zip(prediction, target, strict=True)]
    field_limit = max(float(np.max(np.abs(target))), float(np.max(np.abs(prediction))), 1.0e-30)
    error_limit = max(float(np.max(np.abs(error))), 1.0e-30)
    extent = (
        float(x_m[0]) / 1000.0,
        float(x_m[-1]) / 1000.0,
        float(z_m[-1]) / 1000.0,
        float(z_m[0]) / 1000.0,
    )
    field_norm = Normalize(vmin=-field_limit, vmax=field_limit)
    error_norm = Normalize(vmin=-error_limit, vmax=error_limit)
    fig = plt.figure(figsize=(19.6, 7.3))
    grid = fig.add_gridspec(
        3,
        8,
        width_ratios=(2.85, 1, 1, 1, 1, 1, 1, 0.13),
        left=0.035,
        right=0.982,
        bottom=0.12,
        top=0.875,
        wspace=0.22,
        hspace=0.16,
    )
    velocity_ax = fig.add_subplot(grid[:, 0])
    velocity_image = velocity_ax.imshow(
        velocity_mps,
        cmap="turbo",
        vmin=float(np.min(velocity_mps)),
        vmax=float(np.max(velocity_mps)),
        extent=extent,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    for number, source in enumerate(source_parameters, start=1):
        sx, sz = (float(source[0]), float(source[1]))
        velocity_ax.scatter(
            [sx / 1000.0],
            [sz / 1000.0],
            marker="*",
            s=76,
            c="white",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        velocity_ax.text(
            sx / 1000.0 + 0.035,
            sz / 1000.0 + 0.035,
            f"S{number}",
            color="white",
            fontsize=7.0,
            weight="bold",
            zorder=4,
        )
    velocity_ax.set_title(
        "True Marmousi velocity\ninput-complexity rank 30 of 30",
        fontsize=10.0,
    )
    velocity_ax.set_xlabel("x (km)")
    velocity_ax.set_ylabel("z (km)")
    velocity_cax = inset_axes(
        velocity_ax,
        width="96%",
        height="4.2%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.20, 1.0, 1.0),
        bbox_transform=velocity_ax.transAxes,
        borderpad=0,
    )
    velocity_cbar = fig.colorbar(velocity_image, cax=velocity_cax, orientation="horizontal")
    velocity_cbar.set_label("velocity (m s$^{-1}$)")

    field_mappable = None
    error_mappable = None
    row_labels = ("8-source LWC-84 reference", "8-source Phase4b sum", "Signed error")
    for column, (frame_index, frame_error) in enumerate(
        zip(indices, frame_errors, strict=True), start=1
    ):
        panels = (
            (target[column - 1], field_norm),
            (prediction[column - 1], field_norm),
            (error[column - 1], error_norm),
        )
        for row, (panel, norm) in enumerate(panels):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(
                panel,
                cmap="seismic",
                norm=norm,
                extent=extent,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(
                    f"t={float(time_s[frame_index]):.2f} s\nframe $e_r$={frame_error:.1%}"
                )
            if column == 1:
                ax.set_ylabel("z (km)", labelpad=1.0)
                ax.text(
                    0.03,
                    0.96,
                    row_labels[row],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.7,
                    weight="bold",
                    bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 1.2},
                )
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel("x (km)")
            else:
                ax.set_xticklabels([])
            if row < 2:
                field_mappable = image
            else:
                error_mappable = image
    if field_mappable is None or error_mappable is None:
        raise RuntimeError("superposition plot did not create all panels")
    field_formatter = ScalarFormatter(useMathText=True)
    field_formatter.set_powerlimits((-2, 2))
    field_cbar = fig.colorbar(
        field_mappable,
        cax=fig.add_subplot(grid[0:2, 7]),
        orientation="vertical",
        format=field_formatter,
    )
    field_cbar.set_label(f"summed pressure (global $\\pm${field_limit:.2e})")
    error_formatter = ScalarFormatter(useMathText=True)
    error_formatter.set_powerlimits((-2, 2))
    error_cbar = fig.colorbar(
        error_mappable,
        cax=fig.add_subplot(grid[2, 7]),
        orientation="vertical",
        format=error_formatter,
    )
    error_cbar.set_label(f"signed error (global $\\pm${error_limit:.2e})")
    fig.suptitle(
        "Eight-source linear superposition on the most input-complex Marmousi slice"
        f"; all-401-frame $e_r$={all_time_relative_l2:.2%}",
        fontsize=12.0,
    )
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rank", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_report_path = evaluation_dir / "report.json"
    prediction_manifest_path = evaluation_dir / "prediction_manifest.json"
    report = json.loads(evaluation_report_path.read_text(encoding="utf8"))
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf8"))
    expected_rank = int(args.expected_rank)
    if report.get("status") != "complete" or prediction_manifest.get("status") != "complete":
        raise ValueError("source Phase4b evaluation is incomplete")
    if int(report["panel"]["rank"]) != expected_rank or int(prediction_manifest["rank"]) != expected_rank:
        raise ValueError("source evaluation rank does not match --expected-rank")
    if len(report["panel"]["records"]) != 8 or len(prediction_manifest["records"]) != 8:
        raise ValueError("superposition requires exactly eight sealed source fields")
    prediction_rows = {row["case_id"]: row for row in prediction_manifest["records"]}
    report_rows = {row["case_id"]: row for row in report["panel"]["records"]}
    if set(prediction_rows) != set(CASE_ORDER) or set(report_rows) != set(CASE_ORDER):
        raise ValueError("source evaluation cases differ from the registered eight positions")

    velocity_path = Path(prediction_manifest["velocity_input"]).expanduser().resolve()
    if sha256_file(velocity_path) != prediction_manifest["velocity_input_sha256"]:
        raise ValueError("velocity input hash changed")
    with np.load(velocity_path) as values:
        velocity_mps = np.asarray(values["velocity_mps"], dtype=np.float32)
        time_s = np.asarray(values["time_s"], dtype=np.float64)
        x_m = np.asarray(values["x_m"], dtype=np.float32)
        z_m = np.asarray(values["z_m"], dtype=np.float32)
    if _array_sha256(velocity_mps) != report["panel"]["velocity_array_sha256"]:
        raise ValueError("velocity array digest changed")

    source_parameters: list[np.ndarray] = []
    target_fields: list[np.ndarray] = []
    prediction_fields: list[np.ndarray] = []
    source_bindings: list[dict[str, Any]] = []
    for case_id in CASE_ORDER:
        prediction_row = prediction_rows[case_id]
        report_row = report_rows[case_id]
        prediction_path = Path(prediction_row["prediction_path"]).expanduser().resolve()
        reference_path = Path(report_row["reference_path"]).expanduser().resolve()
        if sha256_file(prediction_path) != prediction_row["prediction_sha256"]:
            raise ValueError(f"prediction hash changed for {case_id}")
        if sha256_file(reference_path) != report_row["reference_sha256"]:
            raise ValueError(f"reference hash changed for {case_id}")
        with np.load(prediction_path) as values:
            prediction = np.asarray(values["prediction_tzx"], dtype=np.float32)
            source = np.asarray(values["source_parameters"], dtype=np.float32)
        with np.load(reference_path) as values:
            target = np.asarray(values["target_tzx"], dtype=np.float32)
            reference_source = np.asarray(values["source_parameters"], dtype=np.float32)
        if not np.array_equal(source, reference_source):
            raise ValueError(f"source mismatch for {case_id}")
        if prediction.shape != (401, 201, 201) or target.shape != prediction.shape:
            raise ValueError(f"wavefield shape changed for {case_id}")
        source_parameters.append(source)
        target_fields.append(target)
        prediction_fields.append(prediction)
        source_bindings.append(
            {
                "case_id": case_id,
                "source_parameters": [float(value) for value in source],
                "prediction_path": str(prediction_path),
                "prediction_sha256": prediction_row["prediction_sha256"],
                "reference_path": str(reference_path),
                "reference_sha256": report_row["reference_sha256"],
            }
        )
    sources = np.stack(source_parameters)
    if not (
        np.all(sources[:, 2] == np.float32(19.0))
        and np.all(sources[:, 3] == sources[0, 3])
        and np.all(sources[:, 4] == np.float32(1.0))
    ):
        raise ValueError("source frequency, onset, or amplitude varies across the sum")
    target_stack = np.stack(target_fields).astype(np.float64)
    prediction_stack = np.stack(prediction_fields).astype(np.float64)
    target_sum = np.sum(target_stack, axis=0, dtype=np.float64).astype(np.float32)
    prediction_sum = np.sum(prediction_stack, axis=0, dtype=np.float64).astype(np.float32)
    all_time_relative_l2 = _relative_l2(prediction_sum, target_sum)
    target_norms = np.linalg.norm(target_stack.reshape(8, -1), axis=1)
    prediction_norms = np.linalg.norm(prediction_stack.reshape(8, -1), axis=1)
    target_cancellation_ratio = float(
        np.linalg.norm(target_sum.astype(np.float64).ravel()) / np.sum(target_norms)
    )
    prediction_cancellation_ratio = float(
        np.linalg.norm(prediction_sum.astype(np.float64).ravel()) / np.sum(prediction_norms)
    )
    snapshot_indices = np.asarray(SNAPSHOT_INDICES, dtype=np.int64)
    snapshot_errors = [
        _relative_l2(prediction_sum[index], target_sum[index]) for index in snapshot_indices
    ]
    arrays_path = output_dir / "rank30_eight_source_superposition_snapshots.npz"
    _atomic_npz(
        arrays_path,
        velocity_mps=velocity_mps,
        x_m=x_m,
        z_m=z_m,
        time_s=time_s,
        source_parameters=sources,
        snapshot_indices=snapshot_indices,
        target_sum_snapshots=target_sum[snapshot_indices],
        prediction_sum_snapshots=prediction_sum[snapshot_indices],
        error_sum_snapshots=prediction_sum[snapshot_indices] - target_sum[snapshot_indices],
    )
    figures = _plot(
        output_dir / "marmousi_rank30_eight_source_superposition",
        velocity_mps=velocity_mps,
        x_m=x_m,
        z_m=z_m,
        time_s=time_s,
        source_parameters=sources,
        target_sum=target_sum,
        prediction_sum=prediction_sum,
        all_time_relative_l2=all_time_relative_l2,
    )
    output_report = {
        "schema": "phase4b_rank30_eight_source_linear_superposition_v1",
        "status": "complete",
        "selection": {
            "rank": expected_rank,
            "sample_id": report["panel"]["source_sample_id"],
            "group_id": report["panel"]["source_group_id"],
            "rule": "maximum registered input-only velocity first-difference energy",
            "prediction_or_target_error_used_for_selection": False,
        },
        "superposition": {
            "operation": "unscaled physical sum of eight independently predicted or simulated single-source fields",
            "linearity_basis": "the registered acoustic wave equation is linear in source amplitude",
            "source_count": 8,
            "fixed_frequency_hz": 19.0,
            "fixed_amplitude_each": 1.0,
            "fixed_onset_s": float(sources[0, 3]),
            "all_401_frame_relative_l2": all_time_relative_l2,
            "target_cancellation_ratio_norm_sum_over_sum_norms": target_cancellation_ratio,
            "prediction_cancellation_ratio_norm_sum_over_sum_norms": prediction_cancellation_ratio,
            "snapshot_indices": [int(value) for value in snapshot_indices],
            "snapshot_times_s": [float(time_s[index]) for index in snapshot_indices],
            "snapshot_relative_l2": snapshot_errors,
        },
        "claim_scope": {
            "one_input_selected_velocity_slice": True,
            "position_only": True,
            "frequency_generalization_claim_permitted": False,
            "population_accuracy_claim_permitted": False,
            "separate_g3_4p51_percent_result_not_depicted": True,
        },
        "checkpoint": report["checkpoint"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_evaluation_report": str(evaluation_report_path),
        "source_evaluation_report_sha256": sha256_file(evaluation_report_path),
        "prediction_manifest": str(prediction_manifest_path),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "source_bindings": source_bindings,
        "outputs": {
            "snapshot_arrays": {
                "path": str(arrays_path.resolve()),
                "sha256": sha256_file(arrays_path),
            },
            "figures": {
                key: {"path": value, "sha256": sha256_file(Path(value))}
                for key, value in figures.items()
            },
        },
    }
    report_path = output_dir / "report.json"
    _atomic_json(output_report, report_path)
    print(
        json.dumps(
            {
                "event": "complete",
                "all_401_frame_relative_l2": all_time_relative_l2,
                "target_cancellation_ratio": target_cancellation_ratio,
                "report": str(report_path.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
