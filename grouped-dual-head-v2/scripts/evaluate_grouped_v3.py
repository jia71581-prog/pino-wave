#!/usr/bin/env python
"""Evaluate a V3 gate checkpoint and generate reproducible comparison figures."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest
from grouped_ufno_mionet_v3.training.checkpoint import CHECKPOINT_FORMAT
from grouped_ufno_mionet_v3.training.gates import evaluate_one_record_gate
from scripts.overfit_grouped_v3 import evaluate_model, load_gate_record, load_normalizer
from scripts.train_grouped_v3 import build_model


COLORS = {"target": "#0072B2", "prediction": "#D55E00", "error": "#CC79A7"}


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


def _save_figure(fig, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _plot_fields(prediction: torch.Tensor, target: torch.Tensor, output: Path) -> None:
    _style()
    indices = (2, 8, 15)
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.2), constrained_layout=True)
    row_labels = ("Target", "V3 prediction", "Absolute error")
    for column, frame in enumerate(indices):
        truth = target[0, frame].numpy()
        estimate = prediction[0, frame].numpy()
        error = np.abs(estimate - truth)
        limit = max(float(np.abs(truth).max()), float(np.abs(estimate).max()), 1.0e-8)
        images = (
            axes[0, column].imshow(truth, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper"),
            axes[1, column].imshow(estimate, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper"),
            axes[2, column].imshow(error, cmap="magma", vmin=0.0, origin="upper"),
        )
        axes[0, column].set_title(f"Saved frame {frame}")
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        fig.colorbar(images[1], ax=axes[:2, column], shrink=0.72, pad=0.02)
        fig.colorbar(images[2], ax=axes[2, column], shrink=0.72, pad=0.02)
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontweight="bold")
    _save_figure(fig, output)


def _receiver_diagnostic(
    model,
    record,
    normalizer,
    *,
    cache_path: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(cache_path, "r", swmr=True) as cache:
        ids = cache["sample_id"].asstr()[:]
        row = int(np.flatnonzero(ids == record.sample_id)[0])
        receiver_zx = np.asarray(cache["receiver_zx_indices"][row], np.int64)
        target_physical = torch.from_numpy(
            np.asarray(cache["receiver_target"][row], np.float32)
        )
    time_s = record.time_s.to(device)
    coords = []
    for z_index, x_index in receiver_zx:
        coords.append(
            torch.stack(
                (
                    torch.full_like(time_s, record.x_m[x_index]),
                    torch.full_like(time_s, record.z_m[z_index]),
                    time_s,
                ),
                dim=-1,
            )
        )
    coords_tensor = torch.cat(coords, dim=0)[None]
    with torch.no_grad():
        source = record.source_parameters[None].to(device)
        prepared = model.prepare_sources(
            model.encode_medium(record.velocity_mps[None].to(device), normalizer),
            source,
            record.source_map[None].to(device),
            normalizer,
        )
        prediction = model.query_normalized(prepared, coords_tensor, chunk_size=2048)
    prediction = prediction.reshape(len(receiver_zx), len(time_s)).cpu()
    target = normalizer.encode_pressure(target_physical, record.source_parameters[4])
    return time_s.cpu().numpy(), prediction.numpy(), target.numpy()


def _plot_waveforms(
    time_s: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    output: Path,
) -> None:
    _style()
    selected = (1, 4, 7)
    fig, axes = plt.subplots(3, 1, figsize=(6.75, 5.0), sharex=True, constrained_layout=True)
    for axis, receiver in zip(axes, selected, strict=True):
        axis.plot(time_s, target[receiver], color=COLORS["target"], label="Numerical target", lw=1.5)
        axis.plot(
            time_s,
            prediction[receiver],
            color=COLORS["prediction"],
            label="V3 arbitrary-point query",
            lw=1.2,
            ls="--",
        )
        relative = np.linalg.norm(prediction[receiver] - target[receiver]) / max(
            np.linalg.norm(target[receiver]), 1.0e-12
        )
        axis.set_ylabel(f"Point {receiver}\n$p/p_{{scale}}$")
        axis.text(0.99, 0.88, f"rel. L2={relative:.3f}", transform=axis.transAxes, ha="right")
        axis.grid(alpha=0.18)
    axes[0].legend(ncol=2, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    _save_figure(fig, output)


def _write_json(payload: dict[str, object], path: Path) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--sample-id",
        default="",
        help="Override data.gate_sample_id with any allowed non-anomaly training record.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write diagnostics even when the strict one-record memorization gate does not pass.",
    )
    args = parser.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    if args.sample_id:
        config = replace(config, data=replace(config.data, gate_sample_id=args.sample_id))
    device = torch.device(args.device)
    manifest = build_manifest(config.data.source_h5)
    normalizer = load_normalizer(config, manifest.digest)
    record = load_gate_record(config, require_uniform=not bool(args.sample_id))
    model = build_model(config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("evaluation checkpoint is not V3")
    if payload.get("manifest_digest") != manifest.digest:
        raise ValueError("evaluation checkpoint manifest mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    metrics, diagnostics, prediction, target, _ = evaluate_model(
        model, record, normalizer, device=device
    )
    if record.medium_type == "uniform":
        gate_decision = evaluate_one_record_gate(metrics)
        decision = {"applicable": True, **gate_decision.to_dict()}
        gate_passed = gate_decision.passed
    else:
        decision = {
            "applicable": False,
            "passed": None,
            "failures": [],
            "reason": "The strict one-record memorization gate is defined only for uniform media.",
        }
        gate_passed = False
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _plot_fields(prediction, target, output / "wavefield_comparison")
    time_s, receiver_prediction, receiver_target = _receiver_diagnostic(
        model,
        record,
        normalizer,
        cache_path=config.data.train_cache,
        device=device,
    )
    _plot_waveforms(time_s, receiver_prediction, receiver_target, output / "point_waveforms")
    receiver_relative = np.linalg.norm(receiver_prediction - receiver_target, axis=1) / np.maximum(
        np.linalg.norm(receiver_target, axis=1), 1.0e-12
    )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_format": payload["format"],
        "manifest_digest": manifest.digest,
        "sample_id": record.sample_id,
        "metrics": metrics.to_dict(),
        "decision": decision,
        "diagnostics": diagnostics,
        "diagnostic_point_relative_l2": receiver_relative.tolist(),
        "note": "Diagnostic point waveforms are derived outputs, never model inputs or training targets.",
    }
    _write_json(report, output / "evaluation_report.json")
    print(json.dumps(report, sort_keys=True))
    return 0 if gate_passed or args.report_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
