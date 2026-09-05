#!/usr/bin/env python3
"""Build a hard-record curriculum for the verified R40+R46 fit-cache union."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import r46_union_frequency_cache as union_cache


SCHEMA = "r46_train_only_union_hard_curriculum_v1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def load_log(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "cached_record":
                continue
            row = int(event["row"])
            if row in rows:
                raise RuntimeError(f"duplicate row {row} in {path}")
            rows[row] = event
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--base-fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--base-fit-log", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--supplement-fit-cache", type=Path, nargs="+", required=True
    )
    parser.add_argument("--supplement-fit-log", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.base_fit_cache) != len(args.base_fit_log):
        raise ValueError("base cache/log counts differ")
    if len(args.supplement_fit_cache) != len(args.supplement_fit_log):
        raise ValueError("supplement cache/log counts differ")

    r40_path = args.r40_script.expanduser().resolve()
    r40 = load_module(r40_path, "r46_union_curriculum_r40")
    base_paths = [path.expanduser().resolve() for path in args.base_fit_cache]
    supplement_paths = [
        path.expanduser().resolve() for path in args.supplement_fit_cache
    ]
    collection = union_cache.UnionFrequencyCacheCollection(
        r40, [base_paths, supplement_paths], expected_subset="fit"
    )
    try:
        verified_cache_evidence = {}
        for cache_path in collection.paths:
            evidence = dict(collection.cache_evidence[str(cache_path)])
            actual_sha256 = sha256_file(cache_path)
            if actual_sha256 != str(evidence["sha256"]):
                raise RuntimeError(f"cache SHA-256 mismatch: {cache_path}")
            evidence["verified_sha256"] = actual_sha256
            verified_cache_evidence[str(cache_path)] = evidence
        log_by_basename = {}
        log_evidence = []
        for cache_arg, log_arg in list(
            zip(args.base_fit_cache, args.base_fit_log)
        ) + list(zip(args.supplement_fit_cache, args.supplement_fit_log)):
            cache_path = cache_arg.expanduser().resolve()
            log_path = log_arg.expanduser().resolve()
            if cache_path.name in log_by_basename:
                raise RuntimeError(f"duplicate cache basename: {cache_path.name}")
            log_by_basename[cache_path.name] = load_log(log_path)
            log_evidence.append(
                {
                    "cache_basename": cache_path.name,
                    "log_path": str(log_path),
                    "log_bytes": log_path.stat().st_size,
                    "log_sha256": sha256_file(log_path),
                }
            )

        records = []
        for position, (file_index, local_row) in enumerate(collection.records):
            cache_path = collection.paths[file_index]
            events = log_by_basename.get(cache_path.name)
            if events is None or int(local_row) not in events:
                raise RuntimeError(
                    f"missing log event: {cache_path.name} row {local_row}"
                )
            event = events[int(local_row)]
            sample_id = str(collection.sample_ids[position])
            family = str(collection.families[position])
            if str(event["sample_id"]) != sample_id or str(event["family"]) != family:
                raise RuntimeError(
                    f"log/cache identity mismatch: {cache_path.name} row {local_row}"
                )
            error = float(event["base_rel_l2"])
            records.append(
                {
                    "record_position": position,
                    "cache_basename": cache_path.name,
                    "local_row": int(local_row),
                    "sample_id": sample_id,
                    "group_id": str(collection.group_ids[position]),
                    "family": family,
                    "base_record_rel_l2": error,
                    "sampling_weight": sampling_weight(error),
                }
            )

        weighted_total = sum(float(row["sampling_weight"]) for row in records)
        tiers = {}
        predicates = {
            "gte_0p05": lambda value: value >= 0.05,
            "0p04_to_0p05": lambda value: 0.04 <= value < 0.05,
            "0p03_to_0p04": lambda value: 0.03 <= value < 0.04,
            "lt_0p03": lambda value: value < 0.03,
        }
        for label, predicate in predicates.items():
            chosen = [row for row in records if predicate(row["base_record_rel_l2"])]
            mass = sum(float(row["sampling_weight"]) for row in chosen)
            tiers[label] = {
                "record_count": len(chosen),
                "sampling_weight_sum": mass,
                "sampling_probability_mass": mass / weighted_total,
            }

        payload = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": collection.selection_sha256,
            "union_identity": collection.union_identity,
            "component_selection_sha256": list(
                collection.component_selection_sha256
            ),
            "record_count": len(records),
            "group_count": len(set(collection.group_ids)),
            "family_record_counts": {
                family: sum(row["family"] == family for row in records)
                for family in sorted(set(collection.families))
            },
            "sampling_rule": {
                "base_record_rel_l2_gte_0p05": 32.0,
                "base_record_rel_l2_gte_0p04": 12.0,
                "base_record_rel_l2_gte_0p03": 4.0,
                "otherwise": 1.0,
                "replacement": True,
                "epoch_draw_count": "fixed_by_R46_training_preregistration",
            },
            "weighted_total": weighted_total,
            "tiers": tiers,
            "cache_evidence": verified_cache_evidence,
            "log_evidence": log_evidence,
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
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "event": "r46_union_curriculum_complete",
                    "output": str(output),
                    "selection_sha256": collection.selection_sha256,
                    "record_count": len(records),
                    "group_count": payload["group_count"],
                    "family_record_counts": payload["family_record_counts"],
                    "tiers": tiers,
                    "r29b_opened": False,
                    "paper_modified": False,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        collection.close()


if __name__ == "__main__":
    raise SystemExit(main())
