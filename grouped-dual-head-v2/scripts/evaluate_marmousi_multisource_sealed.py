#!/usr/bin/env python3
"""Historical variable-frequency Marmousi diagnostic; not paper evidence.

This script evaluates five stored records whose source frequencies differ.  It
must not be used to support the fixed-frequency source-position claim.  The
authoritative controlled workflow is ``evaluate_marmousi_source_position_control.py``.

``predict`` never reads target wavefields.  It serializes physical-pressure
predictions and writes a SHA-256 manifest last.  ``score`` verifies those seals
before it opens exact stored target wavefields, computes 401-frame diagnostics,
and renders the preregistered source-configuration figure panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.confirmatory_metrics import complete_transient_metrics
from saved_time_phase_operator_v4.evaluation import sha256_file, time_axis_sha256
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import _load_context, _load_parent_model


DEFAULT_GROUP_ID = "validation:marmousi:x350.0:z0.0"
SNAPSHOT_INDICES = (80, 240, 400)
RECEIVER_Z_INDEX = 4
RECEIVER_X_INDICES = tuple(range(10, 191, 2))


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


def _load_config_context(config_path: Path):
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    base, manifest, parent_identity = _load_context(config)
    return config, base, manifest, parent_identity


def _selected_indices(dataset: V3WavefieldDataset, group_id: str) -> list[int]:
    indices = [
        index
        for index, record in enumerate(dataset.records)
        if record.group_id == group_id
    ]
    if len(indices) != 5:
        raise ValueError(f"locked group must contain exactly five records, found {len(indices)}")
    if any(dataset.records[index].medium_type != "marmousi" for index in indices):
        raise ValueError("locked group contains a non-Marmousi record")
    return indices


def _validate_locked_geometry(record) -> None:
    if len(record.time_s) != 401 or record.velocity_mps.shape[-2:] != (201, 201):
        raise ValueError("locked figure protocol requires [401,201,201]")
    if not np.allclose(record.time_s.detach().cpu().numpy()[list(SNAPSHOT_INDICES)], [0.2, 0.6, 1.0]):
        raise ValueError("locked snapshot indices no longer map to 0.2/0.6/1.0 s")
    if not np.isclose(float(record.z_m[RECEIVER_Z_INDEX]), 40.0):
        raise ValueError("locked receiver depth is no longer 40 m")
    receiver_x = record.x_m[list(RECEIVER_X_INDICES)].detach().cpu().numpy()
    if not np.allclose(receiver_x, np.arange(100.0, 1900.1, 20.0)):
        raise ValueError("locked receiver x coordinates changed")


@torch.inference_mode()
def predict(
    *,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_identity_path: Path,
    output_dir: Path,
    group_id: str,
    device: torch.device,
    time_block: int,
) -> dict[str, Any]:
    """Serialize predictions without calling the target-wavefield reader."""

    if (output_dir / "prediction_manifest.json").exists():
        raise FileExistsError("sealed prediction manifest already exists; refusing overwrite")
    config, base, manifest, parent_identity = _load_config_context(config_path)
    identity = json.loads(checkpoint_identity_path.read_text())
    if identity.get("manifest_digest") != manifest.digest:
        raise ValueError("checkpoint identity and manifest disagree")
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    metadata = load_checkpoint(
        checkpoint_path,
        model=model,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=identity["run_digest"],
        map_location=device,
    )
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)
    dataset = V3WavefieldDataset(base.data.source_h5, manifest, split="validation")
    indices = _selected_indices(dataset, group_id)

    entries: list[dict[str, Any]] = []
    reference_velocity: np.ndarray | None = None
    for dataset_index in indices:
        # V3WavefieldDataset.__getitem__ is input-only by contract.  Do not call
        # read_wavefield anywhere in this function.
        record = dataset[dataset_index]
        _validate_locked_geometry(record)
        velocity = record.velocity_mps.squeeze(0).detach().cpu().numpy()
        if reference_velocity is None:
            reference_velocity = velocity.copy()
        elif not np.array_equal(reference_velocity, velocity):
            raise ValueError("multi-source group does not share an identical velocity map")
        source = record.source_parameters.unsqueeze(0).to(device)
        prepared = model.prepare_sources(
            model.encode_medium(record.velocity_mps.unsqueeze(0).to(device), normalizer),
            source,
            record.source_map.unsqueeze(0).to(device),
            normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
        )
        prediction_normalized = model.dense_normalized(
            prepared,
            record.time_s.unsqueeze(0).to(device),
            x_m=record.x_m.to(device),
            z_m=record.z_m.to(device),
            time_block=int(time_block),
        )
        prediction_physical = (
            normalizer.decode_pressure(prediction_normalized.float(), source[:, 4])
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if prediction_physical.shape != (401, 201, 201) or not np.isfinite(prediction_physical).all():
            raise RuntimeError("prediction is non-finite or has the wrong shape")
        prediction_path = output_dir / "predictions" / f"{record.sample_id}.npz"
        _atomic_npz(
            prediction_path,
            prediction_tzx=prediction_physical,
            time_s=record.time_s.detach().cpu().numpy().astype(np.float64),
            x_m=record.x_m.detach().cpu().numpy().astype(np.float64),
            z_m=record.z_m.detach().cpu().numpy().astype(np.float64),
            velocity_mps=velocity.astype(np.float32),
            source_parameters=record.source_parameters.detach().cpu().numpy().astype(np.float64),
        )
        entries.append(
            {
                "dataset_index": dataset_index,
                "sample_id": record.sample_id,
                "sample_sha256": record.sample_sha256,
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
                "source_parameters": record.source_parameters.detach().cpu().tolist(),
            }
        )

    entries.sort(key=lambda row: (row["source_parameters"][0], row["source_parameters"][1], row["sample_id"]))
    payload = {
        "schema": "sealed_marmousi_multisource_predictions_v1",
        "status": "complete",
        "truth_wavefield_access": False,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(metadata.epoch),
        "checkpoint_global_step": int(metadata.global_step),
        "checkpoint_identity": str(checkpoint_identity_path.resolve()),
        "checkpoint_identity_sha256": sha256_file(checkpoint_identity_path),
        "model_config_digest": identity["run_digest"],
        "manifest_digest": manifest.digest,
        "time_axis_sha256": time_axis_sha256(manifest.time_s),
        "group_id": group_id,
        "snapshot_indices": list(SNAPSHOT_INDICES),
        "receiver_z_index": RECEIVER_Z_INDEX,
        "receiver_x_indices": list(RECEIVER_X_INDICES),
        "records": entries,
    }
    _atomic_json(payload, output_dir / "prediction_manifest.json")
    return payload


def _load_verified_predictions(manifest_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    payload = json.loads(manifest_path.read_text())
    if payload.get("status") != "complete" or payload.get("truth_wavefield_access") is not False:
        raise ValueError("prediction manifest is not a complete input-only seal")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for row in payload["records"]:
        path = Path(row["prediction_path"])
        if sha256_file(path) != row["prediction_sha256"]:
            raise ValueError(f"prediction seal mismatch: {row['sample_id']}")
        with np.load(path, allow_pickle=False) as archive:
            arrays[row["sample_id"]] = {name: archive[name] for name in archive.files}
    return payload, arrays


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = sorted(set.intersection(*(set(row["metrics"]) for row in rows)))
    return {name: float(np.mean([row["metrics"][name] for row in rows])) for name in names}


def _render_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
    prediction_arrays: dict[str, dict[str, np.ndarray]],
    targets: dict[str, np.ndarray],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    ordered = sorted(rows, key=lambda row: (row["source_parameters"][0], row["source_parameters"][1]))
    first = prediction_arrays[ordered[0]["sample_id"]]
    x_m, z_m = first["x_m"], first["z_m"]
    extent = (x_m[0] / 1000.0, x_m[-1] / 1000.0, z_m[-1] / 1000.0, z_m[0] / 1000.0)
    artifacts: list[str] = []

    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    image = ax.imshow(first["velocity_mps"], cmap="viridis", extent=extent, origin="upper", aspect="equal")
    for index, row in enumerate(ordered, 1):
        source = row["source_parameters"]
        ax.scatter(source[0] / 1000.0, source[1] / 1000.0, marker="*", s=90, label=f"S{index}: {source[2]:.1f} Hz")
    ax.set(xlabel="x (km)", ylabel="z (km)", title="Locked Marmousi slice and five source configurations")
    fig.colorbar(image, ax=ax, label="velocity (m/s)")
    ax.legend(loc="lower right", fontsize=7)
    path = output_dir / "figures" / "marmousi_velocity_sources.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    artifacts.append(str(path.resolve()))

    for time_index in SNAPSHOT_INDICES:
        target_stack = np.stack([targets[row["sample_id"]][time_index] for row in ordered])
        prediction_stack = np.stack([prediction_arrays[row["sample_id"]]["prediction_tzx"][time_index] for row in ordered])
        error_stack = prediction_stack - target_stack
        pressure_limit = max(float(np.abs(target_stack).max()), float(np.abs(prediction_stack).max()), 1.0e-20)
        error_limit = max(float(np.abs(error_stack).max()), 1.0e-20)
        fig, axes = plt.subplots(3, 5, figsize=(12.0, 6.7), constrained_layout=True)
        for column, row in enumerate(ordered):
            for panel_row, panel in enumerate((target_stack[column], prediction_stack[column], error_stack[column])):
                limit = pressure_limit if panel_row < 2 else error_limit
                image = axes[panel_row, column].imshow(panel, cmap="seismic", vmin=-limit, vmax=limit, extent=extent, origin="upper", aspect="equal")
                if panel_row == 0:
                    axes[panel_row, column].set_title(f"S{column + 1}, {row['source_parameters'][2]:.1f} Hz")
                if column == 0:
                    axes[panel_row, column].set_ylabel(("reference", "prediction", "error")[panel_row] + "\nz (km)")
                else:
                    axes[panel_row, column].set_yticks([])
                if panel_row == 2:
                    axes[panel_row, column].set_xlabel("x (km)")
                else:
                    axes[panel_row, column].set_xticks([])
            fig.colorbar(image, ax=axes[:, column], shrink=0.65)
        fig.suptitle(f"Marmousi multi-source wavefields at t={first['time_s'][time_index]:.1f} s")
        path = output_dir / "figures" / f"marmousi_multisource_t{time_index:03d}.pdf"
        fig.savefig(path)
        plt.close(fig)
        artifacts.append(str(path.resolve()))

    receiver_x = x_m[list(RECEIVER_X_INDICES)] / 1000.0
    time_s = first["time_s"]
    fig, axes = plt.subplots(5, 3, figsize=(9.5, 11.0), constrained_layout=True)
    gather_limit = max(
        max(float(np.abs(targets[row["sample_id"]][:, RECEIVER_Z_INDEX, list(RECEIVER_X_INDICES)]).max()) for row in ordered),
        max(float(np.abs(prediction_arrays[row["sample_id"]]["prediction_tzx"][:, RECEIVER_Z_INDEX, list(RECEIVER_X_INDICES)]).max()) for row in ordered),
        1.0e-20,
    )
    errors = [
        prediction_arrays[row["sample_id"]]["prediction_tzx"][:, RECEIVER_Z_INDEX, list(RECEIVER_X_INDICES)]
        - targets[row["sample_id"]][:, RECEIVER_Z_INDEX, list(RECEIVER_X_INDICES)]
        for row in ordered
    ]
    error_limit = max(max(float(np.abs(value).max()) for value in errors), 1.0e-20)
    gather_extent = (receiver_x[0], receiver_x[-1], time_s[-1], time_s[0])
    for row_index, (row, error) in enumerate(zip(ordered, errors, strict=True)):
        sample = row["sample_id"]
        reference = targets[sample][:, RECEIVER_Z_INDEX, list(RECEIVER_X_INDICES)]
        prediction_value = prediction_arrays[sample]["prediction_tzx"][:, RECEIVER_Z_INDEX, list(RECEIVER_X_INDICES)]
        for column, panel in enumerate((reference, prediction_value, error)):
            limit = gather_limit if column < 2 else error_limit
            axes[row_index, column].imshow(panel, cmap="seismic", vmin=-limit, vmax=limit, extent=gather_extent, origin="upper", aspect="auto")
            if row_index == 0:
                axes[row_index, column].set_title(("reference", "prediction", "difference")[column])
            if column == 0:
                axes[row_index, column].set_ylabel(f"S{row_index + 1}\ntime (s)")
            else:
                axes[row_index, column].set_yticks([])
            if row_index == 4:
                axes[row_index, column].set_xlabel("receiver x (km), z=40 m")
    path = output_dir / "figures" / "marmousi_multisource_shallow_gathers.pdf"
    fig.savefig(path)
    plt.close(fig)
    artifacts.append(str(path.resolve()))
    return artifacts


def score(
    *,
    config_path: Path,
    prediction_manifest_path: Path,
    output_dir: Path,
    device: torch.device,
    compute_komega: bool,
) -> dict[str, Any]:
    """Verify prediction seals, then open exact target wavefields and score them."""

    seal, prediction_arrays = _load_verified_predictions(prediction_manifest_path)
    config, base, manifest, _ = _load_config_context(config_path)
    del config
    if seal["manifest_digest"] != manifest.digest or seal["time_axis_sha256"] != time_axis_sha256(manifest.time_s):
        raise ValueError("prediction seal and evaluation manifest disagree")
    dataset = V3WavefieldDataset(base.data.source_h5, manifest, split="validation")
    index_by_sample = {record.sample_id: index for index, record in enumerate(dataset.records)}
    rows: list[dict[str, Any]] = []
    targets: dict[str, np.ndarray] = {}
    for sealed_row in seal["records"]:
        sample_id = sealed_row["sample_id"]
        dataset_index = index_by_sample[sample_id]
        record = dataset[dataset_index]
        if record.group_id != seal["group_id"] or record.sample_sha256 != sealed_row["sample_sha256"]:
            raise ValueError(f"target identity mismatch: {sample_id}")
        target_read = dataset.read_wavefield(dataset_index, record.time_s)
        if not bool(target_read.exact.all()) or not torch.equal(target_read.left_index, target_read.right_index):
            raise RuntimeError("scoring encountered an interpolated target")
        target = target_read.values.to(device)
        prediction = torch.from_numpy(prediction_arrays[sample_id]["prediction_tzx"]).to(device)
        onset_index = int(np.searchsorted(record.time_s.detach().cpu().numpy(), float(record.source_parameters[3])))
        metrics = complete_transient_metrics(
            prediction,
            target,
            record.time_s.to(device),
            onset_index=onset_index,
            receiver_z_index=RECEIVER_Z_INDEX,
            receiver_x_indices=RECEIVER_X_INDICES,
            compute_komega=compute_komega,
        )
        targets[sample_id] = target.detach().cpu().numpy().astype(np.float32)
        rows.append(
            {
                "sample_id": sample_id,
                "group_id": record.group_id,
                "source_parameters": record.source_parameters.detach().cpu().tolist(),
                "onset_index": onset_index,
                "metrics": metrics,
            }
        )
    figures = _render_figures(output_dir, rows, prediction_arrays, targets)
    report = {
        "schema": "sealed_marmousi_multisource_score_v1",
        "status": "complete",
        "prediction_manifest": str(prediction_manifest_path.resolve()),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "target_wavefield_access": True,
        "exact_stored_times": True,
        "group_id": seal["group_id"],
        "record_count": len(rows),
        "snapshot_indices": list(SNAPSHOT_INDICES),
        "receiver_z_index": RECEIVER_Z_INDEX,
        "receiver_x_indices": list(RECEIVER_X_INDICES),
        "aggregate_record_mean_metrics": _mean_metrics(rows),
        "records": rows,
        "figures": figures,
    }
    _atomic_json(report, output_dir / "score_report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict_parser = subparsers.add_parser("predict", help="seal predictions without target access")
    predict_parser.add_argument("--config", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--checkpoint-identity", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    predict_parser.add_argument("--device", default="cuda")
    predict_parser.add_argument("--time-block", type=int, default=16)
    score_parser = subparsers.add_parser("score", help="verify seals and then open target wavefields")
    score_parser.add_argument("--config", type=Path, required=True)
    score_parser.add_argument("--prediction-manifest", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--device", default="cuda")
    score_parser.add_argument("--skip-komega", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "predict":
        result = predict(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            checkpoint_identity_path=args.checkpoint_identity,
            output_dir=args.output,
            group_id=args.group_id,
            device=torch.device(args.device),
            time_block=args.time_block,
        )
    else:
        result = score(
            config_path=args.config,
            prediction_manifest_path=args.prediction_manifest,
            output_dir=args.output,
            device=torch.device(args.device),
            compute_komega=not args.skip_komega,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
