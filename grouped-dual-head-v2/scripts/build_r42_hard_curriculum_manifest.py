#!/usr/bin/env python3
"""Build a train-only hard-record curriculum manifest from audited R40 caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampling_weight(error: float) -> float:
    if error >= 0.05:
        return 32.0
    if error >= 0.04:
        return 12.0
    if error >= 0.03:
        return 4.0
    return 1.0


def load_log(path: Path, expected_shard: int) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "cached_record":
                continue
            if int(event["shard"]) != expected_shard:
                raise RuntimeError(f"log shard mismatch: {path}")
            row = int(event["row"])
            if row in rows:
                raise RuntimeError(f"duplicate row {row} in {path}")
            rows[row] = event
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--fit-log", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.fit_cache) != len(args.fit_log):
        raise ValueError("fit-cache and fit-log counts differ")

    records: list[dict[str, Any]] = []
    selection_sha256 = None
    cache_evidence: list[dict[str, Any]] = []
    for shard, (cache_arg, log_arg) in enumerate(zip(args.fit_cache, args.fit_log)):
        cache_path = cache_arg.expanduser().resolve()
        log_path = log_arg.expanduser().resolve()
        log_rows = load_log(log_path, shard)
        with h5py.File(cache_path, "r", swmr=True) as handle:
            if str(handle.attrs.get("schema", "")) != "r40_frequency_residual_cache_v2":
                raise RuntimeError(f"unexpected cache schema: {cache_path}")
            if str(handle.attrs.get("subset", "")) != "fit":
                raise RuntimeError(f"non-fit cache supplied: {cache_path}")
            current_selection = str(handle.attrs["selection_sha256"])
            if selection_sha256 is None:
                selection_sha256 = current_selection
            elif current_selection != selection_sha256:
                raise RuntimeError("fit cache selection digests disagree")
            count = int(handle["sample_id"].shape[0])
            if set(log_rows) != set(range(count)):
                raise RuntimeError(f"log/cache row coverage mismatch: {cache_path}")
            for row in range(count):
                sample_id = str(handle["sample_id"].asstr()[row])
                group_id = str(handle["group_id"].asstr()[row])
                family = str(handle["family"].asstr()[row])
                event = log_rows[row]
                if str(event["sample_id"]) != sample_id or str(event["family"]) != family:
                    raise RuntimeError(f"log/cache identity mismatch: {cache_path} row {row}")
                error = float(event["base_rel_l2"])
                records.append(
                    {
                        "record_position": len(records),
                        "cache_basename": cache_path.name,
                        "shard": shard,
                        "local_row": row,
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "family": family,
                        "base_record_rel_l2": error,
                        "sampling_weight": sampling_weight(error),
                    }
                )
        cache_evidence.append(
            {
                "path": str(cache_path),
                "bytes": cache_path.stat().st_size,
                "sha256": sha256_file(cache_path),
                "log_path": str(log_path),
                "log_bytes": log_path.stat().st_size,
                "log_sha256": sha256_file(log_path),
            }
        )

    if not records or selection_sha256 is None:
        raise RuntimeError("empty fit curriculum")
    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("duplicate sample IDs in curriculum")
    weighted_total = sum(float(record["sampling_weight"]) for record in records)
    tiers = {}
    for label, predicate in {
        "gte_0p05": lambda value: value >= 0.05,
        "0p04_to_0p05": lambda value: 0.04 <= value < 0.05,
        "0p03_to_0p04": lambda value: 0.03 <= value < 0.04,
        "lt_0p03": lambda value: value < 0.03,
    }.items():
        selected = [record for record in records if predicate(record["base_record_rel_l2"])]
        tier_weight = sum(float(record["sampling_weight"]) for record in selected)
        tiers[label] = {
            "record_count": len(selected),
            "sampling_weight_sum": tier_weight,
            "sampling_probability_mass": tier_weight / weighted_total,
        }

    payload = {
        "schema": "r42_train_only_hard_record_curriculum_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": selection_sha256,
        "record_count": len(records),
        "group_count": len({record["group_id"] for record in records}),
        "sampling_rule": {
            "base_record_rel_l2_gte_0p05": 32.0,
            "base_record_rel_l2_gte_0p04": 12.0,
            "base_record_rel_l2_gte_0p03": 4.0,
            "otherwise": 1.0,
            "replacement": True,
            "epoch_draw_count": "fixed_by_training_preregistration"
        },
        "weighted_total": weighted_total,
        "tiers": tiers,
        "cache_evidence": cache_evidence,
        "records": records,
        "data_boundary": {
            "fit_truth_used": True,
            "opened_development_holdout_used": False,
            "r29b_opened": False,
            "final_validation_opened": False,
            "test_id_opened": False,
            "paper_modified": False,
        },
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "event": "r42_curriculum_complete",
                "output": str(output),
                "record_count": len(records),
                "group_count": payload["group_count"],
                "tiers": tiers,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
