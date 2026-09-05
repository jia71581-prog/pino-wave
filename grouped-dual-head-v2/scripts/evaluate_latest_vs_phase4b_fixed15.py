#!/usr/bin/env python3
"""Compare one saved-time checkpoint with Phase4b on the same fixed-15-Hz cases.

The three inputs and their display times are inherited from the historical
Phase4b paper figures.  Prediction is completed and sealed before the target
arrays are opened.  The result is therefore a same-input comparison, but it is
development/qualitative evidence rather than a new confirmatory split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.evaluation import sha256_file, time_axis_sha256
from saved_time_phase_operator_v4.losses import apply_hard_causality, source_causality_onset_s
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import _load_context, _load_parent_model


SOURCE_H5 = Path(
    "/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/"
    "dataset_v1.h5"
)
CANONICAL_PHASE4B_REPORT = PROJECT_ROOT / (
    "results/current_best_phase4b_figures_canonical_ood_20260808T0328Z/"
    "render_report.json"
)
TRUE_MARMOUSI_PHASE4B_REPORT = PROJECT_ROOT / (
    "results/current_best_phase4b_figures_true_marmousi1_20260808T1201Z/"
    "marmousi/render_report.json"
)
TRUE_MARMOUSI_H5 = PROJECT_ROOT / (
    "results/current_best_phase4b_true_marmousi1_assets_20260808T0359Z/"
    "marmousi1_true_nowater_x4400_z400_f15.h5"
)
CANONICAL_TRAVEL_H5 = PROJECT_ROOT / (
    "results/current_best_phase4b_canonical_ood_assets_20260808T0328Z/"
    "hybrid_travel_ood_canonical.h5"
)
SAMPLE_IDS = {
    "uniform": "ood_uniform_c4000_f15",
    "layered": "ood_layered_c3000_c5000_f15",
    "marmousi": "marmousi1_true_nowater_x4400_z400_f15",
}


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}")
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
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    denominator = max(float(np.linalg.norm(target64.ravel())), 1.0e-20)
    return float(np.linalg.norm((prediction64 - target64).ravel()) / denominator)


def replacement_decision(
    latest: Mapping[str, Mapping[str, float]],
    historical: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Require every family to improve in full-field and receiver metrics."""

    if set(latest) != set(SAMPLE_IDS) or set(historical) != set(SAMPLE_IDS):
        raise ValueError("replacement decision requires all three fixed families")
    rows: dict[str, Any] = {}
    for family in SAMPLE_IDS:
        current = latest[family]
        previous = historical[family]
        full_new = float(current["full_wavefield_relative_l2"])
        full_old = float(previous["full_wavefield_relative_l2"])
        receiver_new = float(current["receiver_mean_relative_l2"])
        receiver_old = float(previous["receiver_mean_relative_l2"])
        if min(full_new, full_old, receiver_new, receiver_old) < 0.0 or not np.isfinite(
            [full_new, full_old, receiver_new, receiver_old]
        ).all():
            raise ValueError("replacement metrics must be finite and nonnegative")
        rows[family] = {
            "latest_full_wavefield_relative_l2": full_new,
            "historical_full_wavefield_relative_l2": full_old,
            "full_wavefield_improved": full_new < full_old,
            "full_wavefield_relative_improvement": (full_old - full_new)
            / max(full_old, 1.0e-20),
            "latest_receiver_mean_relative_l2": receiver_new,
            "historical_receiver_mean_relative_l2": receiver_old,
            "receiver_improved_or_equal": receiver_new <= receiver_old,
            "receiver_relative_improvement": (receiver_old - receiver_new)
            / max(receiver_old, 1.0e-20),
        }
    passed = all(
        row["full_wavefield_improved"] and row["receiver_improved_or_equal"]
        for row in rows.values()
    )
    return {
        "replacement_gate_passed": bool(passed),
        "rule": (
            "strict same-sample gate: every family must lower complete-wavefield "
            "relative L2 and must not regress mean near-surface receiver relative L2"
        ),
        "families": rows,
    }


def _historical_contract() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    canonical = json.loads(CANONICAL_PHASE4B_REPORT.read_text(encoding="utf8"))
    marmousi = json.loads(TRUE_MARMOUSI_PHASE4B_REPORT.read_text(encoding="utf8"))
    reports = {
        "uniform": canonical["families"]["uniform"],
        "layered": canonical["families"]["layered"],
        "marmousi": marmousi,
    }
    indices = {
        family: np.asarray(report["snapshot_indices"], dtype=np.int64)
        for family, report in reports.items()
    }
    for family, report in reports.items():
        if str(report["sample_id"]) != SAMPLE_IDS[family]:
            raise ValueError(f"historical {family} sample identity changed")
        if not np.allclose(np.asarray(report["source_parameters"])[2], 15.0):
            raise ValueError("same-sample comparison is no longer fixed at 15 Hz")
        if indices[family].shape != (6,) or np.any(np.diff(indices[family]) <= 0):
            raise ValueError("historical snapshot contract changed")
    return reports, indices


def _read_main_h5_input(sample_id: str) -> dict[str, np.ndarray]:
    with h5py.File(SOURCE_H5, "r", swmr=True) as handle:
        matches = np.flatnonzero(np.asarray(handle["sample_id"].asstr()[:]) == sample_id)
        if len(matches) != 1:
            raise ValueError(f"sample ID is not unique in source HDF5: {sample_id}")
        index = int(matches[0])
        item = {
            "velocity_mps": np.asarray(handle["velocity_mps"][index], dtype=np.float32),
            "source_map": np.asarray(handle["source_map"][index], dtype=np.float32),
            "source_parameters": np.asarray(
                [
                    handle["source_x_m"][index],
                    handle["source_z_m"][index],
                    handle["source_f0_hz"][index],
                    handle["source_t0_s"][index],
                    handle["source_amplitude"][index],
                ],
                dtype=np.float32,
            ),
            "time_s": np.asarray(handle["time_s"][:], dtype=np.float32),
            "x_m": np.asarray(handle["x_m"][:], dtype=np.float32),
            "z_m": np.asarray(handle["z_m"][:], dtype=np.float32),
            "source_index": np.asarray(index, dtype=np.int64),
        }
    with h5py.File(CANONICAL_TRAVEL_H5, "r") as handle:
        matches = np.flatnonzero(
            np.asarray(handle["sample_id"].asstr()[:]) == sample_id
        )
        if len(matches) != 1:
            raise ValueError(f"travel-time sample ID is not unique: {sample_id}")
        item["travel_time_s"] = np.asarray(
            handle["travel_time_s"][int(matches[0])], dtype=np.float32
        )
    return item


def _read_true_marmousi_input() -> dict[str, np.ndarray]:
    with h5py.File(TRUE_MARMOUSI_H5, "r") as handle:
        return {
            "velocity_mps": np.asarray(handle["velocity_mps"][:], dtype=np.float32),
            "source_map": np.asarray(handle["source_map"][:], dtype=np.float32),
            "source_parameters": np.asarray(handle["source_parameters"][:], dtype=np.float32),
            "time_s": np.asarray(handle["time_s"][:], dtype=np.float32),
            "x_m": np.asarray(handle["x_m"][:], dtype=np.float32),
            "z_m": np.asarray(handle["z_m"][:], dtype=np.float32),
            "source_index": np.asarray(-1, dtype=np.int64),
            "travel_time_s": np.asarray(handle["travel_time_s"][:], dtype=np.float32),
        }


def _read_target(family: str, source_index: int) -> np.ndarray:
    if family == "marmousi":
        with h5py.File(TRUE_MARMOUSI_H5, "r") as handle:
            return np.asarray(handle["wavefield_target"][:], dtype=np.float32)
    with h5py.File(SOURCE_H5, "r", swmr=True) as handle:
        return np.asarray(handle["wavefield"][int(source_index)], dtype=np.float32)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_snapshots(
    target: np.ndarray,
    prediction: np.ndarray,
    indices: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    family: str,
    output: Path,
) -> list[float]:
    _style()
    extent = (x_m[0] / 1000, x_m[-1] / 1000, z_m[-1] / 1000, z_m[0] / 1000)
    fig, axes = plt.subplots(3, len(indices), figsize=(2.3 * len(indices), 6.7), constrained_layout=True)
    errors: list[float] = []
    for column, frame in enumerate(indices):
        truth = target[frame]
        pred = prediction[frame]
        error = pred - truth
        field_limit = max(float(np.max(np.abs(truth))), float(np.max(np.abs(pred))), 1.0e-30)
        error_limit = max(float(np.max(np.abs(error))), 1.0e-30)
        rel = _relative_l2(pred, truth)
        errors.append(rel)
        for row, (panel, limit, label) in enumerate(
            ((truth, field_limit, "Reference"), (pred, field_limit, "Latest model"), (error, error_limit, "Error"))
        ):
            axis = axes[row, column]
            axis.imshow(panel, cmap="seismic", vmin=-limit, vmax=limit, extent=extent, origin="upper", aspect="equal")
            if row == 0:
                axis.set_title(f"t={time_s[frame]:.3f} s\nscale ±{field_limit:.1e}")
            elif row == 2:
                axis.set_title(f"relL2={rel:.2%}\nscale ±{error_limit:.1e}")
            if column == 0:
                axis.set_ylabel(f"{label}\nz (km)")
            else:
                axis.set_yticklabels([])
            if row == 2:
                axis.set_xlabel("x (km)")
            else:
                axis.set_xticklabels([])
    fig.suptitle(f"Latest saved-time model: {family} fixed-15-Hz same-sample snapshots", y=1.025)
    _save(fig, output)
    return errors


def _receiver_contract(height: int, width: int) -> tuple[int, np.ndarray]:
    receiver_z = min(max(int(round(0.05 * (height - 1))), 0), height - 1)
    receiver_x = np.rint(np.linspace(0.15, 0.95, 9) * (width - 1)).astype(np.int64)
    return receiver_z, receiver_x


def _plot_receivers(
    target: np.ndarray,
    prediction: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    family: str,
    output_dir: Path,
) -> tuple[int, np.ndarray, list[float]]:
    _style()
    receiver_z, receiver_x = _receiver_contract(target.shape[-2], target.shape[-1])
    errors: list[float] = []
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 7.2), sharex=True, constrained_layout=True)
    for axis, ix in zip(axes.flat, receiver_x, strict=True):
        truth = target[:, receiver_z, ix]
        pred = prediction[:, receiver_z, ix]
        rel = _relative_l2(pred, truth)
        errors.append(rel)
        axis.plot(time_s, truth, color="#0072B2", lw=1.05, label="reference")
        axis.plot(time_s, pred, color="#D55E00", lw=0.9, ls="--", label="latest")
        axis.set_title(f"x={x_m[ix]/1000:.2f} km, z={z_m[receiver_z]/1000:.2f} km\nrelL2={rel:.2%}")
        axis.grid(alpha=0.25, lw=0.45)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0, 0].legend(loc="upper right")
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("pressure")
    fig.suptitle(f"Latest saved-time model: {family} receiver waveforms", y=1.025)
    _save(fig, output_dir / f"{family}_receiver_waveforms")

    truth = target[:, receiver_z, :].T
    pred = prediction[:, receiver_z, :].T
    error = pred - truth
    field_limit = max(float(np.max(np.abs(truth))), float(np.max(np.abs(pred))), 1.0e-30)
    error_limit = max(float(np.max(np.abs(error))), 1.0e-30)
    extent = (float(time_s[0]), float(time_s[-1]), x_m[-1] / 1000, x_m[0] / 1000)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    for axis, (panel, limit, title) in zip(
        axes,
        ((truth, field_limit, "Reference"), (pred, field_limit, "Latest model"), (error, error_limit, "Error")),
        strict=True,
    ):
        image = axis.imshow(panel, cmap="seismic", vmin=-limit, vmax=limit, extent=extent, aspect="auto", interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("time (s)")
        axis.set_ylabel("receiver x (km)")
        fig.colorbar(image, ax=axis, shrink=0.84, label="pressure")
    fig.suptitle(f"Latest saved-time model: {family} gather at z={z_m[receiver_z]:.0f} m", y=1.02)
    _save(fig, output_dir / f"{family}_receiver_gather")
    return receiver_z, receiver_x, errors


@torch.inference_mode()
def evaluate(
    *,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_identity_path: Path,
    output_dir: Path,
    device: torch.device,
    time_block: int,
) -> dict[str, Any]:
    if (output_dir / "comparison_report.json").exists():
        raise FileExistsError("comparison output already exists; refusing overwrite")
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    identity = json.loads(checkpoint_identity_path.read_text(encoding="utf8"))
    base, manifest, parent_identity = _load_context(config)
    if identity.get("manifest_digest") != manifest.digest:
        raise ValueError("checkpoint identity and manifest disagree")
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    metadata = load_checkpoint(
        checkpoint_path,
        model=model,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=str(identity["run_digest"]),
        map_location=device,
    )
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)
    historical, snapshot_indices = _historical_contract()
    inputs = {
        "uniform": _read_main_h5_input(SAMPLE_IDS["uniform"]),
        "layered": _read_main_h5_input(SAMPLE_IDS["layered"]),
        "marmousi": _read_true_marmousi_input(),
    }
    for family, item in inputs.items():
        if item["velocity_mps"].shape != (201, 201) or item["time_s"].shape != (401,):
            raise ValueError(f"fixed comparison geometry changed for {family}")
        if not np.isclose(float(item["source_parameters"][2]), 15.0):
            raise ValueError("fixed-15-Hz comparison source changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_manifest: dict[str, Any] = {
        "schema": "latest_vs_phase4b_fixed15_prediction_seal_v1",
        "status": "complete",
        "truth_wavefield_access": False,
        "fixed_source_frequency_hz": 15.0,
        "frequency_generalization_claim_permitted": False,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_identity": str(checkpoint_identity_path.resolve()),
        "checkpoint_identity_sha256": sha256_file(checkpoint_identity_path),
        "checkpoint_epoch": int(metadata.epoch),
        "checkpoint_global_step": int(metadata.global_step),
        "manifest_digest": manifest.digest,
        "records": [],
    }
    predictions: dict[str, np.ndarray] = {}
    for family, item in inputs.items():
        velocity = torch.from_numpy(item["velocity_mps"])[None, None].to(device)
        source = torch.from_numpy(item["source_parameters"])[None].to(device)
        source_map = torch.from_numpy(item["source_map"])[None, None].to(device)
        time_tensor = torch.from_numpy(item["time_s"]).to(device)
        prepared = model.prepare_sources(
            model.encode_medium(velocity, normalizer),
            source,
            source_map,
            normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
        )
        dense_grid = model.prepare_dense_grid(
            prepared,
            x_m=torch.from_numpy(item["x_m"]).to(device),
            z_m=torch.from_numpy(item["z_m"]).to(device),
            travel_time_s=torch.from_numpy(item["travel_time_s"])[None].to(device),
        )
        normalized = model.dense_normalized(
            prepared,
            time_tensor,
            dense_grid=dense_grid,
            time_block=int(time_block),
        )
        if bool(config.get("loss", {}).get("hard_causality", False)):
            onset = source_causality_onset_s(
                source,
                lead_cycles=float(config["loss"].get("hard_causality_lead_cycles", 0.0)),
            )
            normalized = apply_hard_causality(normalized, time_tensor[None], onset)
        prediction = (
            normalizer.decode_pressure(normalized.float(), source[:, 4])
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if prediction.shape != (401, 201, 201) or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid latest-model prediction for {family}")
        prediction_path = output_dir / "sealed_predictions" / f"{family}_prediction.npz"
        _atomic_npz(
            prediction_path,
            prediction_tzx=prediction,
            source_parameters=item["source_parameters"],
            time_s=item["time_s"],
        )
        predictions[family] = prediction
        prediction_manifest["records"].append(
            {
                "family": family,
                "sample_id": SAMPLE_IDS[family],
                "input_velocity_sha256": _array_sha256(item["velocity_mps"]),
                "input_travel_time_sha256": _array_sha256(item["travel_time_s"]),
                "source_parameters": [float(value) for value in item["source_parameters"]],
                "time_axis_sha256": time_axis_sha256(item["time_s"]),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
            }
        )
    _atomic_json(prediction_manifest, output_dir / "prediction_manifest.json")

    report: dict[str, Any] = {
        "schema": "latest_vs_phase4b_fixed15_same_sample_comparison_v1",
        "status": "complete",
        "evidence_scope": "same-sample_development_comparison",
        "fixed_source_frequency_hz": 15.0,
        "frequency_generalization_claim_permitted": False,
        "prediction_sealed_before_target_access": True,
        "prediction_manifest": str((output_dir / "prediction_manifest.json").resolve()),
        "prediction_manifest_sha256": sha256_file(output_dir / "prediction_manifest.json"),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": prediction_manifest["checkpoint_sha256"],
        "hard_causality_applied": bool(config.get("loss", {}).get("hard_causality", False)),
        "families": {},
    }
    latest_metrics: dict[str, dict[str, float]] = {}
    historical_metrics: dict[str, dict[str, float]] = {}
    for family, item in inputs.items():
        target = _read_target(family, int(item["source_index"]))
        prediction = predictions[family]
        indices = snapshot_indices[family]
        family_dir = output_dir / family
        frame_errors = _plot_snapshots(
            target,
            prediction,
            indices,
            item["time_s"],
            item["x_m"],
            item["z_m"],
            family,
            family_dir / f"{family}_wavefield_snapshots",
        )
        receiver_z, receiver_x, trace_errors = _plot_receivers(
            target,
            prediction,
            item["time_s"],
            item["x_m"],
            item["z_m"],
            family,
            family_dir,
        )
        arrays_path = family_dir / f"{family}_complete_wavefield.npz"
        _atomic_npz(
            arrays_path,
            target_tzx=target,
            prediction_tzx=prediction,
            velocity_mps=item["velocity_mps"],
            source_parameters=item["source_parameters"],
            source_map=item["source_map"],
            travel_time_s=item["travel_time_s"],
            time_s=item["time_s"],
            x_m=item["x_m"],
            z_m=item["z_m"],
            snapshot_indices=indices,
            receiver_z_index=np.asarray(receiver_z, dtype=np.int64),
            receiver_x_indices=receiver_x,
        )
        current = {
            "full_wavefield_relative_l2": _relative_l2(prediction, target),
            "snapshot_relative_l2": _relative_l2(prediction[indices], target[indices]),
            "receiver_mean_relative_l2": float(np.mean(trace_errors)),
        }
        previous = {
            "full_wavefield_relative_l2": float(historical[family]["full_wavefield_relative_l2"]),
            "receiver_mean_relative_l2": float(historical[family]["receiver_mean_relative_l2"]),
        }
        latest_metrics[family] = current
        historical_metrics[family] = previous
        report["families"][family] = {
            "sample_id": SAMPLE_IDS[family],
            "source_parameters": [float(value) for value in item["source_parameters"]],
            "latest": current,
            "historical_phase4b": previous,
            "snapshot_indices": [int(value) for value in indices],
            "snapshot_frame_relative_l2": frame_errors,
            "receiver_z_m": float(item["z_m"][receiver_z]),
            "receiver_x_m": [float(item["x_m"][index]) for index in receiver_x],
            "receiver_trace_relative_l2": trace_errors,
            "outputs": {
                "complete_wavefield_npz": str(arrays_path.resolve()),
                "wavefield_snapshots_png": str((family_dir / f"{family}_wavefield_snapshots.png").resolve()),
                "wavefield_snapshots_pdf": str((family_dir / f"{family}_wavefield_snapshots.pdf").resolve()),
                "receiver_waveforms_png": str((family_dir / f"{family}_receiver_waveforms.png").resolve()),
                "receiver_gather_png": str((family_dir / f"{family}_receiver_gather.png").resolve()),
            },
        }
    report["replacement_decision"] = replacement_decision(latest_metrics, historical_metrics)
    _atomic_json(report, output_dir / "comparison_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time-block", type=int, default=8)
    args = parser.parse_args()
    report = evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        checkpoint_identity_path=args.checkpoint_identity,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        time_block=int(args.time_block),
    )
    print(json.dumps({"status": report["status"], **report["replacement_decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["replacement_decision"]
