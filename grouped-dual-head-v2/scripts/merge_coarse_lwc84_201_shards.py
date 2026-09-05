#!/usr/bin/env python3
"""Merge four independently sealed LWC-84 validation shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from saved_time_phase_operator_v4.coarse_lwc84 import (
    EXPECTED_VALIDATION_FAMILY_COUNTS,
    merge_record_rows,
    select_all_validation_records,
    shard_records,
)
from scripts.evaluate_coarse_lwc84_201 import _write_json_atomic
from scripts.evaluate_coarse_lwc84_201_shard import _source_identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    return parser


def merge_shards(
    *,
    source_h5: str | Path,
    artifact_dir: str | Path,
    expected_family_counts: Mapping[str, int] = EXPECTED_VALIDATION_FAMILY_COUNTS,
    shard_count: int = 4,
) -> dict[str, object]:
    source_path = Path(source_h5).expanduser().resolve()
    artifact = Path(artifact_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if int(shard_count) < 1:
        raise ValueError("shard_count must be positive")
    expected = select_all_validation_records(
        source_path,
        expected_family_counts=expected_family_counts,
    )
    identity = _source_identity(source_path)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    numerical_contract: dict[str, object] | None = None
    prediction_byte_count = 0
    prediction_file_count = 0

    for shard_index in range(int(shard_count)):
        shard_name = f"shard_{shard_index:02d}"
        shard_dir = artifact / "shards" / shard_name
        summary_path = shard_dir / "shard_summary.json"
        rows_path = shard_dir / "records.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"{shard_name} summary is missing: {summary_path}")
        if not rows_path.is_file():
            raise FileNotFoundError(f"{shard_name} records are missing: {rows_path}")
        summary = json.loads(summary_path.read_text())
        rows = json.loads(rows_path.read_text())
        expected_shard = shard_records(
            expected,
            shard_index=shard_index,
            shard_count=int(shard_count),
        )
        expected_ids = [record.sample_id for record in expected_shard]
        if summary.get("status") != "complete":
            raise ValueError(f"{shard_name} is not complete")
        if int(summary.get("shard_index", -1)) != shard_index or int(
            summary.get("shard_count", -1)
        ) != int(shard_count):
            raise ValueError(f"{shard_name} identity is inconsistent")
        if summary.get("source_identity") != identity:
            raise ValueError(f"{shard_name} source identity mismatch")
        if summary.get("sample_ids") != expected_ids:
            raise ValueError(f"{shard_name} record assignment mismatch")
        if int(summary.get("record_count", -1)) != len(expected_shard) or len(
            rows
        ) != len(expected_shard):
            raise ValueError(f"{shard_name} record count mismatch")
        if not bool(
            summary.get("truth_opened_after_all_shard_predictions_sealed")
        ):
            raise ValueError(f"{shard_name} violates seal-before-truth ordering")
        observed_numerics = dict(summary.get("numerical_contract", {}))
        if numerical_contract is None:
            numerical_contract = observed_numerics
        elif observed_numerics != numerical_contract:
            raise ValueError(f"{shard_name} numerical contract mismatch")
        for row in rows:
            prediction = Path(str(row["prediction_path"]))
            expected_bytes = int(row["prediction_byte_count"])
            if not prediction.is_file() or prediction.stat().st_size != expected_bytes:
                raise ValueError(f"sealed prediction missing or truncated: {prediction}")
            if not bool(
                row["truth_opened_after_all_shard_predictions_sealed"]
            ):
                raise ValueError(
                    f"record {row['sample_id']} violates seal-before-truth ordering"
                )
            prediction_file_count += 1
            prediction_byte_count += expected_bytes
        if int(summary.get("prediction_file_count", -1)) != len(rows):
            raise ValueError(f"{shard_name} prediction count mismatch")
        if int(summary.get("prediction_byte_count", -1)) != sum(
            int(row["prediction_byte_count"]) for row in rows
        ):
            raise ValueError(f"{shard_name} prediction byte count mismatch")
        all_rows.extend(rows)
        summaries.append(summary)

    metrics = merge_record_rows(all_rows, expected_records=expected)
    family_counts = {
        family: sum(record.medium_type == family for record in expected)
        for family in expected_family_counts
    }
    truth_after_seal = all(
        bool(summary["truth_opened_after_all_shard_predictions_sealed"])
        for summary in summaries
    )
    passed = metrics["gate"]["action"] == "direct_baseline"
    result = {
        "schema": "coarse_lwc84_201_sealed480_summary_v1",
        "status": "passed" if passed else "rejected",
        "source_identity": identity,
        "numerical_contract": numerical_contract,
        "shard_count": int(shard_count),
        "family_counts": family_counts,
        "prediction_file_count": prediction_file_count,
        "prediction_byte_count": prediction_byte_count,
        "truth_opened_after_all_predictions_sealed": truth_after_seal,
        "metrics": metrics,
        "shard_runtime": [summary.get("runtime", {}) for summary in summaries],
        "peak_cuda_bytes": max(
            int(summary.get("peak_cuda_bytes", 0)) for summary in summaries
        ),
    }
    artifact.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(artifact / "sealed480_summary.json", result)
    decision_path = artifact / ("success.json" if passed else "rejection.json")
    _write_json_atomic(
        decision_path,
        {
            "status": result["status"],
            "metrics": metrics,
            "sealed480_summary": str(artifact / "sealed480_summary.json"),
        },
    )
    print(
        json.dumps(
            {
                "event": "sealed480_merge_complete",
                "status": result["status"],
                "aggregate_relative_l2": metrics["aggregate_relative_l2"],
                "family_relative_l2": metrics["family_relative_l2"],
                "prediction_file_count": prediction_file_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    merge_shards(
        source_h5=args.source_h5,
        artifact_dir=args.artifact_dir,
        shard_count=args.shard_count,
    )


if __name__ == "__main__":
    main()
