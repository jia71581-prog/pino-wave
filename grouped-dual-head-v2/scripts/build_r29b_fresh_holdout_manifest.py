#!/usr/bin/env python3
"""Freeze a fresh R29B train-only holdout while reusing the R28 fit cache.

The R28 fit records and time indices are copied exactly.  New holdout groups
are selected only from train groups absent from both the R28 fit and its
already-opened development holdout.  The resulting holdout cache can therefore
be evaluated once after the R29A-selected training schedule is frozen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
BASE_PATH = SCRIPT_PATH.with_name("build_r25_coarse_residual_cache.py")
SPEC = importlib.util.spec_from_file_location("r29b_cache_components", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import cache components: {BASE_PATH}")
r25 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r25
SPEC.loader.exec_module(r25)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=290829)
    parser.add_argument("--holdout-uniform", type=int, default=16)
    parser.add_argument("--holdout-layered", type=int, default=20)
    parser.add_argument("--holdout-marmousi", type=int, default=20)
    args = parser.parse_args()

    source_h5 = args.source_h5.expanduser().resolve()
    prior_path = args.prior_manifest.expanduser().resolve()
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("schema") != r25.SCHEMA_MANIFEST:
        raise RuntimeError("unexpected prior manifest schema")
    prior_payload = dict(prior)
    prior_digest = str(prior_payload.pop("selection_sha256"))
    if r25.canonical_json_sha256(prior_payload) != prior_digest:
        raise RuntimeError("prior manifest selection digest mismatch")

    with h5py.File(source_h5, "r", swmr=True) as handle:
        split = r25.text_array(handle, "split")
        families = r25.text_array(handle, "medium_type")
        sample_ids = r25.text_array(handle, "sample_id")
        group_ids = r25.text_array(handle, "group_id")
        sample_hashes = r25.text_array(handle, "sample_sha256")
        source_manifest_sha256 = str(handle.attrs.get("manifest_sha256", ""))
        source_config_sha256 = str(handle.attrs.get("config_sha256", ""))

    requested = {
        "uniform": int(args.holdout_uniform),
        "layered": int(args.holdout_layered),
        "marmousi": int(args.holdout_marmousi),
    }
    opened: dict[str, set[str]] = {family: set() for family in r25.FAMILIES}
    for row in list(prior["fit_records"]) + list(prior["holdout_records"]):
        opened[str(row["family"])].add(str(row["group_id"]))

    holdout_rows: list[dict] = []
    audit: dict[str, dict] = {}
    for family_index, family in enumerate(r25.FAMILIES):
        candidates = np.flatnonzero((split == "train") & (families == family))
        group_to_indices: dict[str, list[int]] = {}
        for raw_index in candidates:
            index = int(raw_index)
            group_to_indices.setdefault(str(group_ids[index]), []).append(index)
        all_groups = sorted(group_to_indices)
        never_opened = [group for group in all_groups if group not in opened[family]]
        rng = np.random.default_rng(int(args.seed) + 1009 * family_index)
        rng.shuffle(never_opened)
        selected, selected_groups, _ = r25.take_group_exclusive_records(
            never_opened,
            group_to_indices,
            count=requested[family],
        )
        selected_group_set = set(selected_groups)
        if selected_group_set & opened[family]:
            raise RuntimeError(f"R29B holdout reuses an opened {family} group")
        holdout_rows.extend(
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
            "source_train_groups": len(all_groups),
            "r28_opened_groups": len(opened[family]),
            "available_never_opened_groups": len(never_opened),
            "holdout_records": len(selected),
            "holdout_groups": len(selected_group_set),
            "holdout_groups_never_opened_before_r29b": True,
        }

    fit_rows = [dict(row) for row in prior["fit_records"]]
    fit_groups = {str(row["group_id"]) for row in fit_rows}
    holdout_groups = {str(row["group_id"]) for row in holdout_rows}
    if fit_groups & holdout_groups:
        raise RuntimeError("R29B fit/holdout group leakage")
    if len({row["sample_id"] for row in fit_rows + holdout_rows}) != len(
        fit_rows + holdout_rows
    ):
        raise RuntimeError("R29B selected sample IDs are not unique")

    payload = {
        "schema": r25.SCHEMA_MANIFEST,
        "research_variant": "r29b_fresh_late_tail_holdout_v1",
        "status": "frozen_before_new_holdout_cache_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "source_h5": str(source_h5),
        "source_h5_byte_count": source_h5.stat().st_size,
        "source_manifest_sha256": source_manifest_sha256,
        "source_config_sha256": source_config_sha256,
        "prior_r28_manifest": str(prior_path),
        "prior_r28_selection_sha256": prior_digest,
        "prior_r28_manifest_file_sha256": r25.sha256_file(prior_path),
        "split_policy": {
            "fit": "exact_R28_fit_records_and_fit_time_indices_reused_without_opening_new_data",
            "holdout": "train_only_groups_absent_from_all_R28_fit_and_development_holdout_groups",
            "holdout_evaluation": "once_after_R29A_selected_schedule_is_frozen",
            "validation_opened": False,
            "test_id_opened": False,
        },
        "numerical_contract": dict(prior["numerical_contract"]),
        "fit_time_indices": list(prior["fit_time_indices"]),
        "holdout_time_indices": list(prior["holdout_time_indices"]),
        "family_group_counts": audit,
        "fit_records": fit_rows,
        "holdout_records": holdout_rows,
        "deployment_input_contract": list(prior["deployment_input_contract"]),
        "truth_policy": "high_fidelity wavefield is train-only supervision and is never a deployment input",
        "absolute_goal": {
            "primary": "fresh group-disjoint train holdout record-relative L2",
            "mean_lte": 0.05,
            "max_lte": 0.05,
        },
    }
    payload["selection_sha256"] = r25.canonical_json_sha256(payload)
    output = args.output.expanduser().resolve()
    r25.atomic_json(payload, output)
    print(
        json.dumps(
            {
                "event": "R29B_FRESH_HOLDOUT_MANIFEST_FROZEN",
                "selection_sha256": payload["selection_sha256"],
                "fit_records_reused": len(fit_rows),
                "holdout_records": len(holdout_rows),
                "family_group_counts": audit,
                "validation_opened": False,
                "test_id_opened": False,
                "script_sha256": r25.sha256_file(SCRIPT_PATH),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
