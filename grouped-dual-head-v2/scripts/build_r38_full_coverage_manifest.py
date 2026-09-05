#!/usr/bin/env python3
"""Freeze R38 full train coverage with two untouched group reserves.

Every train record from the uniform/layered/Marmousi families is assigned to
fit unless its group belongs to either the already-opened R28 development
holdout or the preregistered, never-opened R29B fresh holdout.  The R28
development records remain the only R38 development holdout.  Validation and
test_id are never selected.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
BASE_PATH = SCRIPT_PATH.with_name("build_r25_coarse_residual_cache.py")
SPEC = importlib.util.spec_from_file_location("r38_cache_components", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import cache components: {BASE_PATH}")
r25 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r25
SPEC.loader.exec_module(r25)


def load_manifest(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != r25.SCHEMA_MANIFEST:
        raise RuntimeError(f"unexpected manifest schema: {path}")
    digest_payload = dict(payload)
    digest = str(digest_payload.pop("selection_sha256"))
    if r25.canonical_json_sha256(digest_payload) != digest:
        raise RuntimeError(f"manifest digest mismatch: {path}")
    return payload, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source_h5 = args.source_h5.expanduser().resolve()
    development_path = args.development_manifest.expanduser().resolve()
    fresh_path = args.fresh_manifest.expanduser().resolve()
    development, development_digest = load_manifest(development_path)
    fresh, fresh_digest = load_manifest(fresh_path)
    if Path(str(development["source_h5"])).resolve() != source_h5:
        raise RuntimeError("R28 development manifest source mismatch")
    if Path(str(fresh["source_h5"])).resolve() != source_h5:
        raise RuntimeError("R29B fresh manifest source mismatch")

    development_groups = {
        family: {
            str(row["group_id"])
            for row in development["holdout_records"]
            if str(row["family"]) == family
        }
        for family in r25.FAMILIES
    }
    fresh_groups = {
        family: {
            str(row["group_id"])
            for row in fresh["holdout_records"]
            if str(row["family"]) == family
        }
        for family in r25.FAMILIES
    }
    for family in r25.FAMILIES:
        if development_groups[family] & fresh_groups[family]:
            raise RuntimeError(f"development/fresh reserve overlap for {family}")

    with h5py.File(source_h5, "r", swmr=True) as handle:
        split = r25.text_array(handle, "split")
        families = r25.text_array(handle, "medium_type")
        sample_ids = r25.text_array(handle, "sample_id")
        group_ids = r25.text_array(handle, "group_id")
        sample_hashes = r25.text_array(handle, "sample_sha256")
        source_manifest_sha256 = str(handle.attrs.get("manifest_sha256", ""))
        source_config_sha256 = str(handle.attrs.get("config_sha256", ""))

    fit_rows: list[dict] = []
    audit: dict[str, dict] = {}
    for family in r25.FAMILIES:
        candidates = np.flatnonzero((split == "train") & (families == family))
        reserved = development_groups[family] | fresh_groups[family]
        selected = [
            int(index) for index in candidates if str(group_ids[int(index)]) not in reserved
        ]
        selected_groups = {str(group_ids[index]) for index in selected}
        if selected_groups & reserved:
            raise RuntimeError(f"reserved group entered R38 fit for {family}")
        fit_rows.extend(
            r25.record_row(
                index,
                sample_ids=sample_ids,
                group_ids=group_ids,
                sample_hashes=sample_hashes,
                families=families,
            )
            for index in selected
        )
        audit[family] = {
            "source_train_records": int(candidates.size),
            "source_train_groups": len({str(group_ids[int(index)]) for index in candidates}),
            "fit_records": len(selected),
            "fit_groups": len(selected_groups),
            "development_reserved_records": sum(
                1 for row in development["holdout_records"] if str(row["family"]) == family
            ),
            "development_reserved_groups": len(development_groups[family]),
            "fresh_reserved_records": sum(
                1 for row in fresh["holdout_records"] if str(row["family"]) == family
            ),
            "fresh_reserved_groups": len(fresh_groups[family]),
        }

    holdout_rows = [dict(row) for row in development["holdout_records"]]
    fit_group_set = {str(row["group_id"]) for row in fit_rows}
    development_group_set = {str(row["group_id"]) for row in holdout_rows}
    fresh_group_set = {str(row["group_id"]) for row in fresh["holdout_records"]}
    if fit_group_set & development_group_set or fit_group_set & fresh_group_set:
        raise RuntimeError("R38 fit leaks into a reserved group")
    if development_group_set & fresh_group_set:
        raise RuntimeError("R38 reserve sets overlap")
    if len({row["sample_id"] for row in fit_rows + holdout_rows}) != len(fit_rows) + len(holdout_rows):
        raise RuntimeError("R38 sample IDs are not unique")

    payload = {
        "schema": r25.SCHEMA_MANIFEST,
        "research_variant": "r38_full_train_coverage_dual_group_reserve_v1",
        "status": "frozen_before_cache_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_h5": str(source_h5),
        "source_h5_byte_count": source_h5.stat().st_size,
        "source_manifest_sha256": source_manifest_sha256,
        "source_config_sha256": source_config_sha256,
        "development_manifest": str(development_path),
        "development_selection_sha256": development_digest,
        "development_manifest_file_sha256": r25.sha256_file(development_path),
        "fresh_manifest": str(fresh_path),
        "fresh_selection_sha256": fresh_digest,
        "fresh_manifest_file_sha256": r25.sha256_file(fresh_path),
        "split_policy": {
            "fit": "all_three_family_train_records_excluding_both_reserved_group_sets",
            "development_holdout": "exact_R28_already_opened_group_disjoint_holdout",
            "fresh_holdout": "R29B_preregistered_never_opened_groups_excluded_from_fit_and_not_cached_here",
            "validation_opened": False,
            "test_id_opened": False,
        },
        "numerical_contract": dict(development["numerical_contract"]),
        "fit_time_indices": list(development["fit_time_indices"]),
        "holdout_time_indices": list(development["holdout_time_indices"]),
        "family_group_counts": audit,
        "fit_records": fit_rows,
        "holdout_records": holdout_rows,
        "deployment_input_contract": list(development["deployment_input_contract"]),
        "truth_policy": "high_fidelity wavefield is train-only supervision and is never a deployment input",
        "absolute_goal": {
            "primary": "R28 development holdout record-relative L2 before one-shot R29B evaluation",
            "mean_lte": 0.05,
            "max_lte": 0.05,
        },
        "evidence_boundary": "R38 development success only authorizes one R29B fresh-holdout evaluation.",
    }
    payload["selection_sha256"] = r25.canonical_json_sha256(payload)
    r25.atomic_json(payload, output)
    print(json.dumps({
        "event": "R38_MANIFEST_FROZEN",
        "selection_sha256": payload["selection_sha256"],
        "fit_records": len(fit_rows),
        "development_holdout_records": len(holdout_rows),
        "fresh_holdout_records_reserved": len(fresh["holdout_records"]),
        "family_group_counts": audit,
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": r25.sha256_file(SCRIPT_PATH),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
