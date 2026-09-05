#!/usr/bin/env python3
"""Build a truth-leaking residual-scaled oracle diagnostic.

This utility intentionally cannot create an artifact that looks like a model
prediction.  It uses the reference wavefield after evaluation and is suitable
only for an explicitly labelled oracle/visualization diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_KIND = "truth_leaking_oracle_not_model_prediction"
DEFAULT_PREDICTION_MANIFEST = Path(
    "paper/tgrs_helmholtz_operator/experiment_evidence_bundle_20260813/"
    "09_pending_position_evaluation/prediction_manifest.json"
)
DEFAULT_REFERENCE_MANIFEST = Path(
    "paper/tgrs_helmholtz_operator/experiment_evidence_bundle_20260813/"
    "09_pending_position_evaluation/reference_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/truth_leaking_oracle_residual_scaled_5x_diagnostic_20260814"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    numerator = np.sum((prediction64 - target64) ** 2, dtype=np.float64)
    denominator = max(
        float(np.sum(target64**2, dtype=np.float64)),
        1.0e-16,
    )
    return float(np.sqrt(numerator / denominator))


def residual_scaled_oracle(
    prediction: np.ndarray,
    target: np.ndarray,
    shrink_factor: float,
) -> np.ndarray:
    if not np.isfinite(shrink_factor) or shrink_factor <= 1.0:
        raise ValueError("shrink_factor must be finite and greater than one")
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    return (target64 + (prediction64 - target64) / shrink_factor).astype(
        np.float32
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"manifest has no record list: {path}")
    return payload


def _record_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload["records"]
    result = {str(item["record_id"]): item for item in records}
    if len(result) != len(records):
        raise ValueError("duplicate record_id in manifest")
    return result


def _validate_output_dir(path: Path) -> None:
    resolved_parts = {part.lower() for part in path.resolve().parts}
    if "truth_leaking_oracle" not in path.name.lower():
        raise ValueError("output directory must contain 'truth_leaking_oracle'")
    forbidden = {"predictions", "paper", "manuscript", "figs"}
    overlap = sorted(resolved_parts & forbidden)
    if overlap:
        raise ValueError(f"refusing oracle output under protected path: {overlap}")
    if path.exists():
        marker = path / "NOT_A_MODEL_PREDICTION"
        if not marker.is_file():
            raise ValueError("existing output directory lacks oracle guard marker")
        generated = list(path.glob("*.npz")) + list(path.glob("*.json"))
        if generated:
            raise FileExistsError("refusing to overwrite existing oracle artifacts")


def build(args: argparse.Namespace) -> dict[str, Any]:
    prediction_manifest_path = args.prediction_manifest.resolve()
    reference_manifest_path = args.reference_manifest.resolve()
    output_dir = args.output_dir.resolve()
    _validate_output_dir(output_dir)

    prediction_manifest = _load_manifest(prediction_manifest_path)
    reference_manifest = _load_manifest(reference_manifest_path)
    prediction_records = _record_map(prediction_manifest)
    reference_records = _record_map(reference_manifest)

    selected_ids = sorted(
        record_id
        for record_id, record in prediction_records.items()
        if int(record["slice_rank"]) == args.slice_rank
    )
    if len(selected_ids) != args.expected_records:
        raise ValueError(
            f"expected {args.expected_records} slice-rank records, got "
            f"{len(selected_ids)}"
        )
    if set(selected_ids) - set(reference_records):
        raise ValueError("prediction/reference record sets do not match")

    output_dir.mkdir(parents=True, exist_ok=True)
    record_summaries: list[dict[str, Any]] = []
    for record_id in selected_ids:
        prediction_record = prediction_records[record_id]
        reference_record = reference_records[record_id]
        prediction_path = Path(prediction_record["prediction_path"])
        reference_path = Path(reference_record["reference_path"])

        prediction_sha256 = sha256_file(prediction_path)
        reference_sha256 = sha256_file(reference_path)
        if prediction_sha256 != prediction_record["prediction_sha256"]:
            raise ValueError(f"prediction digest mismatch: {record_id}")
        if reference_sha256 != reference_record["reference_sha256"]:
            raise ValueError(f"reference digest mismatch: {record_id}")

        with np.load(prediction_path, allow_pickle=False) as pred_npz:
            prediction = pred_npz["prediction_tzx"]
            pred_source_map = pred_npz["source_map_zx"]
            pred_source_parameters = pred_npz["source_parameters"]
        with np.load(reference_path, allow_pickle=False) as ref_npz:
            target = ref_npz["target_tzx"]
            ref_source_map = ref_npz["source_map_zx"]
            ref_source_parameters = ref_npz["source_parameters"]

        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError(f"invalid wavefield shapes for {record_id}")
        if max(args.frames) >= prediction.shape[0]:
            raise ValueError(f"frame index outside wavefield for {record_id}")
        if not np.array_equal(pred_source_map, ref_source_map):
            raise ValueError(f"source-map mismatch for {record_id}")
        if not np.array_equal(pred_source_parameters, ref_source_parameters):
            raise ValueError(f"source-parameter mismatch for {record_id}")

        frame_indices = np.asarray(args.frames, dtype=np.int64)
        prediction_frames = np.asarray(prediction[frame_indices], dtype=np.float32)
        target_frames = np.asarray(target[frame_indices], dtype=np.float32)
        oracle_frames = residual_scaled_oracle(
            prediction_frames,
            target_frames,
            args.shrink_factor,
        )
        original_error = relative_l2(prediction_frames, target_frames)
        oracle_error = relative_l2(oracle_frames, target_frames)
        measured_ratio = oracle_error / original_error
        expected_ratio = 1.0 / args.shrink_factor
        if not np.isclose(measured_ratio, expected_ratio, rtol=2e-6, atol=2e-8):
            raise ValueError(
                f"oracle error-ratio check failed for {record_id}: "
                f"{measured_ratio} vs {expected_ratio}"
            )

        output_path = output_dir / f"{record_id}__{ARTIFACT_KIND}.npz"
        np.savez_compressed(
            output_path,
            artifact_kind=np.asarray(ARTIFACT_KIND),
            truth_accessed=np.asarray(True),
            may_be_used_as_model_prediction=np.asarray(False),
            shrink_factor=np.asarray(args.shrink_factor, dtype=np.float64),
            frame_indices=frame_indices,
            original_prediction_snapshots_tzx=prediction_frames,
            reference_snapshots_tzx=target_frames,
            truth_leaking_oracle_snapshots_tzx=oracle_frames,
            source_map_zx=pred_source_map,
            source_parameters=pred_source_parameters,
            source_prediction_sha256=np.asarray(prediction_sha256),
            source_reference_sha256=np.asarray(reference_sha256),
        )
        record_summaries.append(
            {
                "artifact_kind": ARTIFACT_KIND,
                "record_id": record_id,
                "role": prediction_record["role"],
                "slice_rank": int(prediction_record["slice_rank"]),
                "source_prediction_path": str(prediction_path),
                "source_prediction_sha256": prediction_sha256,
                "source_reference_path": str(reference_path),
                "source_reference_sha256": reference_sha256,
                "snapshot_output_path": str(output_path),
                "snapshot_output_sha256": sha256_file(output_path),
                "snapshot_original_relative_l2": original_error,
                "snapshot_oracle_relative_l2": oracle_error,
                "measured_error_ratio": measured_ratio,
            }
        )

    manifest: dict[str, Any] = {
        "schema": "truth_leaking_residual_scaled_oracle_diagnostic_v1",
        "status": "diagnostic_only_not_model_output",
        "artifact_kind": ARTIFACT_KIND,
        "truth_accessed": True,
        "may_be_used_as_model_prediction": False,
        "may_be_used_for_model_selection_or_scoring": False,
        "may_be_used_as_paper_experimental_result": False,
        "allowed_use": "oracle illustration and debugging only",
        "formula": "oracle = target + (prediction - target) / shrink_factor",
        "shrink_factor": args.shrink_factor,
        "expected_relative_error_ratio": 1.0 / args.shrink_factor,
        "complete_wavefield_materialized": False,
        "virtual_complete_wavefield_definition_only": True,
        "materialized_snapshot_frames": list(args.frames),
        "slice_rank": args.slice_rank,
        "prediction_manifest": str(prediction_manifest_path),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "records": record_summaries,
    }
    manifest_path = output_dir / "oracle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-manifest",
        type=Path,
        default=DEFAULT_PREDICTION_MANIFEST,
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=DEFAULT_REFERENCE_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slice-rank", type=int, default=15)
    parser.add_argument("--frames", type=int, nargs="+", default=[80, 240, 400])
    parser.add_argument("--shrink-factor", type=float, default=5.0)
    parser.add_argument("--expected-records", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(
        json.dumps(
            {
                "artifact_kind": result["artifact_kind"],
                "records": len(result["records"]),
                "expected_relative_error_ratio": result[
                    "expected_relative_error_ratio"
                ],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
