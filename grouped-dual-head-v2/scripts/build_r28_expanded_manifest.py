#!/usr/bin/env python3
"""Freeze an expanded train-only manifest with a never-opened holdout.

R25 fit and development-holdout groups are promoted to R28 fitting data.
R28 holdout groups are selected only from train groups absent from the entire
R25 manifest, so they remain independent of architecture and loss tuning.
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
SPEC = importlib.util.spec_from_file_location("r25_cache_builder", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R25 cache builder: {BASE_PATH}")
r25 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r25
SPEC.loader.exec_module(r25)


def family_counts(args: argparse.Namespace, prefix: str) -> dict[str, int]:
    return {
        "uniform": int(getattr(args, f"{prefix}_uniform")),
        "layered": int(getattr(args, f"{prefix}_layered")),
        "marmousi": int(getattr(args, f"{prefix}_marmousi")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=280828)
    parser.add_argument("--fit-time-count", type=int, default=64)
    parser.add_argument("--fit-uniform", type=int, default=128)
    parser.add_argument("--fit-layered", type=int, default=256)
    parser.add_argument("--fit-marmousi", type=int, default=512)
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

    prior_groups: dict[str, set[str]] = {family: set() for family in r25.FAMILIES}
    for row in list(prior["fit_records"]) + list(prior["holdout_records"]):
        prior_groups[str(row["family"])].add(str(row["group_id"]))

    fit_target = family_counts(args, "fit")
    holdout_target = family_counts(args, "holdout")
    fit_rows: list[dict] = []
    holdout_rows: list[dict] = []
    audit: dict[str, dict] = {}
    for family_index, family in enumerate(r25.FAMILIES):
        candidates = np.flatnonzero((split == "train") & (families == family))
        group_to_indices: dict[str, list[int]] = {}
        for raw_index in candidates:
            index = int(raw_index)
            group_to_indices.setdefault(str(group_ids[index]), []).append(index)
        all_groups = sorted(group_to_indices)
        previously_seen = set(prior_groups[family])
        missing_prior = previously_seen - set(all_groups)
        if missing_prior:
            raise RuntimeError(f"prior groups missing from source for {family}")

        rng = np.random.default_rng(int(args.seed) + 1009 * family_index)
        never_opened = [group for group in all_groups if group not in previously_seen]
        rng.shuffle(never_opened)
        held, held_groups, _ = r25.take_group_exclusive_records(
            never_opened,
            group_to_indices,
            count=holdout_target[family],
        )
        held_group_set = set(held_groups)
        if held_group_set & previously_seen:
            raise RuntimeError(f"R28 holdout reuses an opened group for {family}")

        prior_fit_order = sorted(previously_seen - held_group_set)
        fresh_fit_order = [
            group
            for group in all_groups
            if group not in held_group_set and group not in previously_seen
        ]
        rng.shuffle(prior_fit_order)
        rng.shuffle(fresh_fit_order)
        fit, fit_groups, _ = r25.take_group_exclusive_records(
            prior_fit_order + fresh_fit_order,
            group_to_indices,
            count=fit_target[family],
        )
        fit_group_set = set(fit_groups)
        if held_group_set & fit_group_set:
            raise RuntimeError(f"R28 fit/holdout leakage for {family}")
        if not previously_seen.issubset(fit_group_set):
            raise RuntimeError(
                f"fit target is too small to promote all prior groups for {family}"
            )

        holdout_rows.extend(
            r25.record_row(
                index,
                sample_ids=sample_ids,
                group_ids=group_ids,
                sample_hashes=sample_hashes,
                families=families,
            )
            for index in held
        )
        fit_rows.extend(
            r25.record_row(
                index,
                sample_ids=sample_ids,
                group_ids=group_ids,
                sample_hashes=sample_hashes,
                families=families,
            )
            for index in fit
        )
        audit[family] = {
            "source_train_records": int(candidates.size),
            "source_train_groups": len(all_groups),
            "prior_opened_groups_promoted_to_fit": len(previously_seen),
            "fit_records": len(fit),
            "fit_groups": len(fit_group_set),
            "holdout_records": len(held),
            "holdout_groups": len(held_group_set),
            "holdout_groups_never_opened_before_r28": True,
        }

    fit_group_set = {row["group_id"] for row in fit_rows}
    holdout_group_set = {row["group_id"] for row in holdout_rows}
    if fit_group_set & holdout_group_set:
        raise RuntimeError("global R28 fit/holdout group leakage")
    if len({row["sample_id"] for row in fit_rows + holdout_rows}) != len(
        fit_rows + holdout_rows
    ):
        raise RuntimeError("R28 selected sample IDs are not unique")

    fit_times = np.rint(
        np.linspace(0, r25.STORED_TIME_COUNT - 1, int(args.fit_time_count))
    ).astype(np.int64)
    if len(np.unique(fit_times)) != int(args.fit_time_count):
        raise RuntimeError("R28 fit time selection is not unique")
    payload = {
        "schema": r25.SCHEMA_MANIFEST,
        "research_variant": "r28_expanded_complex_medium_coverage_v1",
        "status": "frozen_before_cache_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "source_h5": str(source_h5),
        "source_h5_byte_count": source_h5.stat().st_size,
        "source_manifest_sha256": source_manifest_sha256,
        "source_config_sha256": source_config_sha256,
        "prior_development_manifest": str(prior_path),
        "prior_development_selection_sha256": prior_digest,
        "prior_development_manifest_file_sha256": r25.sha256_file(prior_path),
        "split_policy": {
            "fit": "train_only_including_promoted_R25_development_groups",
            "holdout": "train_only_group_disjoint_and_never_opened_in_R25",
            "validation_opened": False,
            "test_id_opened": False,
        },
        "numerical_contract": {
            "method": "LWC-84",
            "grid": [r25.GRID_SIZE, r25.GRID_SIZE],
            "spacing_m": 10.0,
            "internal_dt_s": r25.INTERNAL_DT_S,
            "stored_dt_s": 0.0025,
            "stored_time_count": r25.STORED_TIME_COUNT,
            "npml": r25.NPML,
            "c_ref_mps": r25.C_REF_MPS,
            "dtype": "float32",
        },
        "fit_time_indices": fit_times.tolist(),
        "holdout_time_indices": list(range(r25.STORED_TIME_COUNT)),
        "family_group_counts": audit,
        "fit_records": fit_rows,
        "holdout_records": holdout_rows,
        "deployment_input_contract": list(prior["deployment_input_contract"]),
        "truth_policy": "high_fidelity wavefield is train-only supervision and is never a deployment input",
        "absolute_goal": {
            "primary": "never-opened group-disjoint train holdout record-relative L2",
            "mean_lte": 0.05,
            "max_lte": 0.05,
        },
    }
    payload["selection_sha256"] = r25.canonical_json_sha256(payload)
    r25.atomic_json(payload, args.output.expanduser().resolve())
    print(
        json.dumps(
            {
                "event": "R28_MANIFEST_FROZEN",
                "selection_sha256": payload["selection_sha256"],
                "fit_records": len(fit_rows),
                "holdout_records": len(holdout_rows),
                "family_group_counts": audit,
                "validation_opened": False,
                "test_id_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
