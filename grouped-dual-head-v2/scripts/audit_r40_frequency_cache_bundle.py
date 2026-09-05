#!/usr/bin/env python3
"""Independently audit the complete R40 v2 cache before training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import train_r40_frequency_residual_operator as r40
import validate_r40_frequency_cache as validator


def canonical_sha(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_dir = args.cache_dir.expanduser().resolve()
    bundle_path = cache_dir / "bundle_summary.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    stored_bundle_sha = str(bundle.get("bundle_sha256", ""))
    digest_payload = dict(bundle)
    digest_payload.pop("bundle_sha256", None)
    if canonical_sha(digest_payload) != stored_bundle_sha:
        raise RuntimeError("R40 bundle digest mismatch")
    if bundle.get("schema") != "r40_frequency_cache_bundle_v2":
        raise RuntimeError("unexpected R40 bundle schema")
    if bundle.get("status") != "complete":
        raise RuntimeError("R40 bundle is not complete")
    if bool(bundle.get("validation_opened")) or bool(bundle.get("test_id_opened")):
        raise RuntimeError("R40 bundle evidence boundary violation")

    fit_paths = [cache_dir / f"fit_shard{index}.h5" for index in range(4)]
    holdout_paths = [cache_dir / f"holdout_shard{index}.h5" for index in range(4)]
    for path in fit_paths + holdout_paths:
        summary_path = path.with_suffix(path.suffix + ".summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if path.stat().st_size != int(summary["output_bytes"]):
            raise RuntimeError(f"cache byte count mismatch: {path}")
        actual_sha = r40.sha256_file(path)
        if actual_sha != str(summary["output_sha256"]):
            raise RuntimeError(f"cache SHA-256 mismatch: {path}")

    fit = r40.FrequencyCacheCollection(fit_paths, expected_subset="fit")
    holdout = r40.FrequencyCacheCollection(
        holdout_paths, expected_subset="holdout"
    )
    try:
        if len(fit.records) != 1032 or len(holdout.records) != 56:
            raise RuntimeError("unexpected R40 record counts")
        overlap = sorted(set(fit.group_ids) & set(holdout.group_ids))
        if overlap:
            raise RuntimeError(f"fit/holdout group leakage: {overlap[:3]}")
        if fit.selection_sha256 != holdout.selection_sha256:
            raise RuntimeError("fit/holdout selection digest mismatch")

        static_min = float("inf")
        static_max = 0.0
        base_dct_max = 0.0
        residual_dct_max = 0.0
        frequency_scale_min = float("inf")
        frequency_scale_max = 0.0
        for collection in (fit, holdout):
            for handle in collection.handles:
                scale = np.asarray(handle["static_dct_scale"][:], dtype=np.float32)
                if not np.isfinite(scale).all() or np.any(scale <= 0.0):
                    raise RuntimeError("invalid static DCT scale")
                static_min = min(static_min, float(scale.min()))
                static_max = max(static_max, float(scale.max()))
                spectral_scale = np.asarray(
                    handle["frequency_scale"][:], dtype=np.float32
                )
                if not np.isfinite(spectral_scale).all() or np.any(
                    spectral_scale <= 0.0
                ):
                    raise RuntimeError("invalid frequency scale")
                frequency_scale_min = min(
                    frequency_scale_min, float(spectral_scale.min())
                )
                frequency_scale_max = max(
                    frequency_scale_max, float(spectral_scale.max())
                )
                for row in range(len(handle["sample_id"])):
                    base = np.asarray(handle["base_dct_norm"][row], dtype=np.float32)
                    residual = np.asarray(
                        handle["residual_dct_norm"][row], dtype=np.float32
                    )
                    if not np.isfinite(base).all() or not np.isfinite(residual).all():
                        raise RuntimeError("nonfinite wavefield DCT coefficient")
                    base_dct_max = max(base_dct_max, float(np.max(np.abs(base))))
                    residual_dct_max = max(
                        residual_dct_max, float(np.max(np.abs(residual)))
                    )

        oracle_rows = []
        for path in holdout_paths:
            oracle_rows.extend(validator.validate(path)["records"])
        base = [float(row["base_rel_l2"]) for row in oracle_rows]
        oracle = [
            float(row["representation_oracle_rel_l2"]) for row in oracle_rows
        ]
        oracle_max_index = int(np.argmax(np.asarray(oracle)))
        oracle_gate = bool(max(oracle) <= 0.05)
        audit = {
            "schema": "r40_frequency_cache_bundle_audit_v1",
            "status": "pass" if oracle_gate else "representation_blocked",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "bundle": str(bundle_path),
            "bundle_sha256": stored_bundle_sha,
            "selection_sha256": fit.selection_sha256,
            "fit_record_count": len(fit.records),
            "fit_group_count": len(set(fit.group_ids)),
            "holdout_record_count": len(holdout.records),
            "holdout_group_count": len(set(holdout.group_ids)),
            "group_overlap_count": 0,
            "frequency_count": fit.frequency_count,
            "dct_retained": int(fit.retained),
            "static_dct_scale_min": static_min,
            "static_dct_scale_max": static_max,
            "frequency_scale_min": frequency_scale_min,
            "frequency_scale_max": frequency_scale_max,
            "base_dct_norm_max_abs": base_dct_max,
            "residual_dct_norm_max_abs": residual_dct_max,
            "base_record_rel_l2": summarize(base),
            "representation_oracle_record_rel_l2": summarize(oracle),
            "representation_oracle_worst": oracle_rows[oracle_max_index],
            "representation_oracle_max_lte_0p05": oracle_gate,
            "r29b_opened": False,
            "final_validation_opened": False,
            "test_id_opened": False,
        }
        r40.atomic_json(audit, args.output.expanduser().resolve())
        print(json.dumps(audit, indent=2, sort_keys=True))
        if not oracle_gate:
            return 3
        return 0
    finally:
        fit.close()
        holdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
