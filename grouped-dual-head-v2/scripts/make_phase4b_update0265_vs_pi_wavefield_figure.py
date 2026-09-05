#!/usr/bin/env python3
"""Render the same-record Phase4b update-265 versus PI-DeepONet snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_SHA256 = "8b157cd45512c036d8992c07f8e7593d7bc7047d801a0cbbfcc46fedc623b1c0"
PI_SOURCE_SHA256 = "59fb6b04a59ade103b85754541278c3e0ed232845c1f34344369f11ba8e348ca"
SELECTION = (
    ("uniform", "train_uniform_00146", "ed12d212389e9503290854c7d140f95e3466c671b9e461dc350bd45ef0d75a04"),
    ("layered", "train_layered_00109", "5ada098503fe28a07f3d8d649a8dbd9d043ae676690bb3e268bb436136639079"),
    ("marmousi", "train_marmousi_00107", "410d16986d3220829493d0cc504aec5204cb2f198d4fe1abd49fa828747d896e"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    return float(
        math.sqrt(
            np.sum((prediction64 - target64) ** 2)
            / max(float(np.sum(target64**2)), 1.0e-30)
        )
    )


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    if _sha256(args.result) != RESULT_SHA256:
        raise ValueError("Phase4b same-record result binding changed")
    if _sha256(args.pi_source) != PI_SOURCE_SHA256:
        raise ValueError("registered PI snapshot source binding changed")
    result = json.loads(args.result.read_text())
    by_id = {str(row["sample_id"]): row for row in result["records"]}
    if result.get("status") != "complete" or len(by_id) != 6:
        raise ValueError("Phase4b result is not the complete six-record evaluation")

    family_label = {"uniform": "Uniform", "layered": "Layered", "marmousi": "Marmousi"}
    rows: list[dict[str, object]] = []
    with np.load(args.pi_source) as pi_source:
        for family, sample_id, expected_snapshot_sha256 in SELECTION:
            snapshot_path = args.snapshot_dir / f"{sample_id}_t0240.npz"
            if _sha256(snapshot_path) != expected_snapshot_sha256:
                raise ValueError(f"Phase4b snapshot binding changed for {sample_id}")
            with np.load(snapshot_path) as phase4b:
                reference = np.asarray(pi_source[f"{family}_reference"], dtype=np.float32)
                if not np.array_equal(reference, phase4b["target"]):
                    raise ValueError(f"reference arrays differ for {sample_id}")
                rows.append(
                    {
                        "family": family,
                        "sample_id": sample_id,
                        "reference": reference,
                        "phase4b": np.asarray(phase4b["prediction"], dtype=np.float32),
                        "pi": np.asarray(pi_source[f"{family}_pi_deeponet"], dtype=np.float32),
                        "x_m": np.asarray(pi_source[f"{family}_x_m"], dtype=np.float32),
                        "z_m": np.asarray(pi_source[f"{family}_z_m"], dtype=np.float32),
                    }
                )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    figure = plt.figure(figsize=(7.16, 4.45), constrained_layout=True)
    grid = figure.add_gridspec(
        3,
        7,
        width_ratios=(1.0, 1.0, 1.0, 0.055, 1.0, 1.0, 0.055),
        wspace=0.05,
        hspace=0.11,
    )
    titles = {
        0: "Reference",
        1: "SBH update 265",
        2: "PI-DeepONet",
        4: "SBH abs. error",
        5: "PI abs. error",
    }
    report_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        reference = np.asarray(row["reference"])
        phase4b = np.asarray(row["phase4b"])
        pi_field = np.asarray(row["pi"])
        phase4b_error = np.abs(phase4b - reference)
        pi_error = np.abs(pi_field - reference)
        pressure_values = np.concatenate(
            [np.abs(reference).ravel(), np.abs(phase4b).ravel(), np.abs(pi_field).ravel()]
        )
        error_values = np.concatenate([phase4b_error.ravel(), pi_error.ravel()])
        pressure_limit = max(float(np.quantile(pressure_values, 0.997)), 1.0e-30)
        error_limit = max(float(np.quantile(error_values, 0.997)), 1.0e-30)
        x_m = np.asarray(row["x_m"])
        z_m = np.asarray(row["z_m"])
        extent = (
            float(x_m[0] / 1000.0),
            float(x_m[-1] / 1000.0),
            float(z_m[-1] / 1000.0),
            float(z_m[0] / 1000.0),
        )
        pressure_norm = Normalize(vmin=-pressure_limit, vmax=pressure_limit)
        error_norm = Normalize(vmin=0.0, vmax=error_limit)
        axes: list[plt.Axes] = []
        for column, field, cmap, norm in (
            (0, reference, "RdBu_r", pressure_norm),
            (1, phase4b, "RdBu_r", pressure_norm),
            (2, pi_field, "RdBu_r", pressure_norm),
            (4, phase4b_error, "magma", error_norm),
            (5, pi_error, "magma", error_norm),
        ):
            axis = figure.add_subplot(grid[row_index, column])
            axis.imshow(
                field,
                cmap=cmap,
                norm=norm,
                extent=extent,
                interpolation="nearest",
                aspect="equal",
            )
            if row_index == 0:
                axis.set_title(titles[column], fontsize=7.2, pad=3.0)
            if column == 0:
                letter = chr(ord("a") + row_index)
                axis.set_ylabel(
                    f"({letter}) {family_label[str(row['family'])]}\nz (km)", fontsize=7.0
                )
            else:
                axis.set_yticklabels([])
            if row_index == len(rows) - 1:
                axis.set_xlabel("x (km)", fontsize=7.0, labelpad=1.5)
            else:
                axis.set_xticklabels([])
            axis.tick_params(labelsize=6.4, length=2.0, pad=1.2)
            axes.append(axis)
        pressure_axis = figure.add_subplot(grid[row_index, 3])
        pressure_bar = figure.colorbar(axes[2].images[0], cax=pressure_axis)
        pressure_bar.set_label("p (Pa)", fontsize=6.2, labelpad=1.5)
        pressure_bar.ax.tick_params(labelsize=5.5, length=1.7, pad=1.0)
        pressure_bar.formatter.set_powerlimits((-2, 2))
        pressure_bar.update_ticks()
        error_axis = figure.add_subplot(grid[row_index, 6])
        error_bar = figure.colorbar(axes[4].images[0], cax=error_axis)
        error_bar.set_label(r"$|\Delta p|$ (Pa)", fontsize=6.2, labelpad=1.5)
        error_bar.ax.tick_params(labelsize=5.5, length=1.7, pad=1.0)
        error_bar.formatter.set_powerlimits((-2, 2))
        error_bar.update_ticks()
        report_rows.append(
            {
                "sample_id": row["sample_id"],
                "family": row["family"],
                "snapshot_phase4b_relative_l2": _relative(phase4b, reference),
                "snapshot_pi_relative_l2": _relative(pi_field, reference),
                "complete_future_phase4b_relative_l2": float(
                    by_id[str(row["sample_id"])]["phase4b_parent_relative_l2"]
                ),
                "complete_future_pi_relative_l2": float(
                    by_id[str(row["sample_id"])]["pi_deeponet_relative_l2"]
                ),
            }
        )

    figure.suptitle(
        "Matched wavefield snapshots at t = 0.60 s (train-development records)",
        fontsize=8.0,
        y=1.01,
    )
    args.figure_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure_base.with_suffix(".pdf"))
    figure.savefig(args.figure_base.with_suffix(".png"), dpi=450)
    plt.close(figure)

    report = {
        "schema": "phase4b_update0265_vs_pi_wavefield_figure_v1",
        "status": "complete_train_development_figure",
        "selection": {
            "rule": "first preregistered r34 record in each family",
            "snapshot_index": 240,
            "snapshot_time_s": 0.60,
            "sample_ids": [sample_id for _, sample_id, _ in SELECTION],
        },
        "method_identity": {
            "ours": "phase4b_update0265_with_sigma2_lwc84_background",
            "baseline": "full_training_set_pi_deeponet_epoch96",
            "external_numerical_background_required": True,
        },
        "rows": report_rows,
        "artifacts": {
            "figure_pdf": str(args.figure_base.with_suffix(".pdf").resolve()),
            "figure_pdf_sha256": _sha256(args.figure_base.with_suffix(".pdf")),
            "figure_png": str(args.figure_base.with_suffix(".png").resolve()),
            "figure_png_sha256": _sha256(args.figure_base.with_suffix(".png")),
        },
        "bindings": {
            "result_sha256": RESULT_SHA256,
            "pi_source_sha256": PI_SOURCE_SHA256,
            "script_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(report, args.report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "results/phase4b_update0265_vs_pi_train_r35_20260816/result.json",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=ROOT / "results/phase4b_update0265_vs_pi_train_r35_20260816/snapshots",
    )
    parser.add_argument(
        "--pi-source",
        type=Path,
        default=ROOT
        / "paper/tgrs_helmholtz_operator/experiment_evidence_bundle_20260813"
        / "15_pi_deeponet_train_development_comparison"
        / "method_vs_pi_train_wavefield_snapshot_source.npz",
    )
    parser.add_argument(
        "--figure-base",
        type=Path,
        default=ROOT
        / "paper/tgrs_helmholtz_operator/figs"
        / "phase4b_update0265_vs_pi_train_wavefield_snapshot",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "paper/tgrs_helmholtz_operator/experiment_evidence_bundle_20260813"
        / "16_phase4b_update0265_pi_comparison"
        / "figure_report.json",
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "report": str(args.report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
