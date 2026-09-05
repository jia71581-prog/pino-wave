#!/usr/bin/env python3
"""Evaluate the archived Phase4b update-265 parent on the registered PI panel.

The archived checkpoint was trained before the Marmousi VDS repair.  This script
therefore treats its use on the repaired manifest as an explicit checkpoint
transfer, verifies a strict state-dict match, and evaluates it on the exact six
records and future windows already bound by the r34 PI-DeepONet comparison.

The smoothed-velocity LWC-84 background is part of Phase4b inference.  It must be
provided for all 401 saved times and is never hidden from the method identity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.run_v5_instance_adaptation import _predict_parent
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_probe import _model


EXPECTED_CHECKPOINT_SHA256 = (
    "f6efbb81dd1e9baab0eb34b32b125e2cb58cd3b3292ee0db33a29194c84bfea1"
)
EXPECTED_CHECKPOINT_MANIFEST = (
    "a20c9a65abbc65294062af443e2ceae241ead66450f940f652e2d95aaa0aa92b"
)
EXPECTED_EVALUATION_MANIFEST = (
    "55fbffa9a66b0cb547657d2d5cd8cc140c4f7d970e37f3828144e7778d182e09"
)
EXPECTED_REGISTERED_COMPARISON_SHA256 = (
    "22f9a22cd105995415ec66dcbb15a86bf3dc9c12108f1f1790a67c4667fc2a8b"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _phase4b_variant(identity: dict[str, object]) -> ProbeVariant:
    return ProbeVariant(
        depth=int(identity["dense_depth"]),
        use_local_phase=True,
        spectral_rank=int(identity["dense_spectral_rank"]),
        modes=int(identity["dense_modes"]),
        temporal_basis_rank=0,
        family_expert_rank=0,
        local_field=True,
        local_field_residual=False,
        local_field_helmholtz_synthesis=True,
        local_field_helmholtz_synthesis_frequencies=int(
            identity["helmholtz_frequencies"]
        ),
        local_field_helmholtz_synthesis_wkb_phase=bool(identity["helmholtz_wkb_phase"]),
        local_field_helmholtz_synthesis_rank=int(identity["helmholtz_rank"]),
        local_field_helmholtz_synthesis_late_rank=int(
            identity.get("helmholtz_late_rank", 0)
        ),
        local_field_helmholtz_synthesis_late_frequencies=int(
            identity.get("helmholtz_late_frequencies", 0)
        ),
    )


def _aggregate(rows: list[dict[str, object]], error_key: str) -> dict[str, object]:
    def one(selected: list[dict[str, object]]) -> dict[str, float]:
        error = sum(float(row[error_key]) for row in selected)
        truth = sum(float(row["future_truth_squared_norm"]) for row in selected)
        return {
            "relative_l2": float(math.sqrt(error / max(truth, 1.0e-30))),
            "squared_error": error,
            "truth_squared_norm": truth,
        }

    families = sorted({str(row["family"]) for row in rows})
    return {
        "global": one(rows),
        "by_family": {
            family: one([row for row in rows if row["family"] == family])
            for family in families
        },
        "mean_record_relative_l2": float(
            sum(float(row[error_key.replace("squared_error", "relative_l2")]) for row in rows)
            / len(rows)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard count)")
    if _sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Phase4b update-265 checkpoint binding changed")
    if _sha256(args.registered_comparison) != EXPECTED_REGISTERED_COMPARISON_SHA256:
        raise ValueError("registered r34 PI comparison binding changed")

    registered = json.loads(args.registered_comparison.read_text())
    if (
        registered.get("status") != "complete_train_only_comparison"
        or registered.get("protocol", {}).get("record_count") != 6
    ):
        raise ValueError("registered comparison is not the complete six-record result")
    registered_rows = list(registered["records"])
    selected_rows = registered_rows[args.shard_index :: args.shard_count]
    if not selected_rows:
        raise ValueError("evaluation shard is empty")

    base = V3Config.from_yaml(args.base_config)
    manifest = build_manifest(base.data.source_h5)
    if manifest.digest != EXPECTED_EVALUATION_MANIFEST:
        raise ValueError("repaired evaluation manifest binding changed")
    identity = json.loads(args.run_identity.read_text())
    if identity.get("manifest_digest") != EXPECTED_CHECKPOINT_MANIFEST:
        raise ValueError("archived Phase4b run identity changed")

    device = torch.device(args.device)
    model = _model(base, manifest, _phase4b_variant(identity)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("manifest_digest") != EXPECTED_CHECKPOINT_MANIFEST:
        raise ValueError("checkpoint does not carry the archived manifest digest")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)
    background = BackgroundFieldProvider(args.background_cache)

    sample_ids = tuple(str(row["sample_id"]) for row in selected_rows)
    if not background.covers(sample_ids, range(len(manifest.time_s))):
        raise ValueError("background cache does not cover every selected record and time")
    dataset = GuardedOnsetDataset(
        base.data.source_h5,
        manifest,
        split="train",
        sample_ids=sample_ids,
        travel_time_h5=args.travel_time_h5,
    )
    registered_by_id = {str(row["sample_id"]): row for row in selected_rows}
    output_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for position in range(len(dataset)):
            record = dataset[position]
            registered_row = registered_by_id[record.sample_id]
            observed = tuple(int(value) for value in registered_row["observed_indices"])
            if tuple(record.observed_indices) != observed:
                raise RuntimeError("guarded onset indices differ from the r34 comparison")

            with torch.inference_mode():
                prediction = _predict_parent(
                    model,
                    normalizer,
                    record,
                    device,
                    normalized=False,
                    background_provider=background,
                    time_block=args.time_block,
                )
            prediction_np = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
            prediction_sha256 = _array_sha256(prediction_np)

            # Future truth is opened only after the prediction has been fixed and hashed.
            with h5py.File(base.data.source_h5, "r", swmr=True) as source:
                truth_np = np.asarray(
                    source["wavefield"][int(record.source_index)], dtype=np.float32
                )[None]
            future = slice(observed[1] + 1, len(manifest.time_s))
            prediction64 = prediction_np[:, future].astype(np.float64)
            truth64 = truth_np[:, future].astype(np.float64)
            squared_error = float(np.sum((prediction64 - truth64) ** 2))
            truth_squared_norm = float(np.sum(truth64**2))
            registered_truth = float(registered_row["future_truth_squared_norm"])
            if not math.isclose(
                truth_squared_norm, registered_truth, rel_tol=2.0e-6, abs_tol=1.0e-30
            ):
                raise RuntimeError("Phase4b and PI future truth energies differ")
            relative_l2 = float(
                math.sqrt(squared_error / max(truth_squared_norm, 1.0e-30))
            )

            snapshot_index = int(args.snapshot_index)
            snapshot_path = args.snapshot_dir / f"{record.sample_id}_t{snapshot_index:04d}.npz"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                snapshot_path,
                prediction=prediction_np[0, snapshot_index],
                target=truth_np[0, snapshot_index],
                velocity_mps=record.velocity_mps.detach().cpu().numpy(),
                source_parameters=record.source_parameters.detach().cpu().numpy(),
                time_s=np.asarray(float(record.time_s[snapshot_index])),
            )
            row = {
                "sample_id": record.sample_id,
                "source_index": int(record.source_index),
                "family": str(registered_row["family"]),
                "observed_indices": list(observed),
                "future_frame_count": len(manifest.time_s) - observed[1] - 1,
                "future_truth_squared_norm": truth_squared_norm,
                "phase4b_parent_squared_error": squared_error,
                "phase4b_parent_relative_l2": relative_l2,
                "pi_deeponet_squared_error": float(
                    registered_row["pi_deeponet_squared_error"]
                ),
                "pi_deeponet_relative_l2": float(
                    registered_row["pi_deeponet_relative_l2"]
                ),
                "prediction_sha256_before_truth_open": prediction_sha256,
                "snapshot_path": str(snapshot_path.resolve()),
                "snapshot_sha256": _sha256(snapshot_path),
            }
            output_rows.append(row)
            print(
                f"[{position + 1}/{len(dataset)}] {record.sample_id} "
                f"phase4b={relative_l2:.6g} "
                f"pi={row['pi_deeponet_relative_l2']:.6g}",
                flush=True,
            )
    finally:
        background.close()
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    family_counts: dict[str, int] = defaultdict(int)
    for row in output_rows:
        family_counts[str(row["family"])] += 1
    report = {
        "schema": "phase4b_update0265_vs_pi_train_worker_v1",
        "status": "complete",
        "role": "same_record_development_evidence_not_validation_or_test_evidence",
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "elapsed_s": float(time.perf_counter() - started),
        "method_identity": {
            "ours": "archived_phase4b_update0265_with_sigma2_lwc84_background",
            "baseline": "full_training_set_pi_deeponet",
            "external_numerical_background_required": True,
            "instance_adaptation_applied": False,
        },
        "protocol": {
            "split": "train",
            "record_count": len(output_rows),
            "family_count": dict(family_counts),
            "frames": "all saved frames strictly after the second guarded onset observation",
            "same_records_as_pi": True,
            "same_future_time_indices_as_pi": True,
            "physical_pressure_float64_energy_metric": True,
            "prediction_hashed_before_future_truth_open": True,
            "checkpoint_transfer_from_archived_to_repaired_manifest": True,
        },
        "records": output_rows,
        "metrics": {
            "phase4b_parent": _aggregate(
                output_rows, "phase4b_parent_squared_error"
            ),
            "pi_deeponet": _aggregate(output_rows, "pi_deeponet_squared_error"),
        },
        "bindings": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "checkpoint_manifest_digest": EXPECTED_CHECKPOINT_MANIFEST,
            "evaluation_manifest_digest": manifest.digest,
            "strict_state_dict_match": True,
            "registered_comparison": str(args.registered_comparison.resolve()),
            "registered_comparison_sha256": _sha256(args.registered_comparison),
            "run_identity_sha256": _sha256(args.run_identity),
            "base_config_sha256": _sha256(args.base_config),
            "normalization_sha256": _sha256(Path(base.data.normalization_json)),
            "travel_time_h5_sha256": _sha256(args.travel_time_h5),
            "background_cache": [str(path.resolve()) for path in args.background_cache],
            "background_cache_sha256": {
                str(path.resolve()): _sha256(path) for path in args.background_cache
            },
            "evaluation_script_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(report, args.output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-identity", type=Path, required=True)
    parser.add_argument("--registered-comparison", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT
        / "configs/grouped_v3/continuous_pilot_w128_legacy_norm_marmousi1_4m_v2.yaml",
    )
    parser.add_argument("--travel-time-h5", type=Path, required=True)
    parser.add_argument("--background-cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--snapshot-index", type=int, default=240)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-block", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
