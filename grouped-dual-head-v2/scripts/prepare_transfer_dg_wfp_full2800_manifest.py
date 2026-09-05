#!/usr/bin/env python3
"""Expand the fixed E1 group split to every one of the 2800 train records."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "results/transfer_dg_wfp_train_pool_2800_20260902.json"
E1 = ROOT / "results/transfer_dg_wfp_e1_manifest_256_20260902.json"
OUTPUT = ROOT / "results/transfer_dg_wfp_full2800_manifest_20260902.json"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    pool = json.loads(POOL.read_text())
    e1 = json.loads(E1.read_text())
    calibration_groups, confirmation_groups = set(), set()
    for family in FAMILIES:
        holdout = sorted(
            [row for row in e1["records"] if row["family"] == family and row["role"] == "holdout"],
            key=lambda row: row["sample_id"],
        )
        calibration_groups.update(row["group_id"] for row in holdout[::2])
        confirmation_groups.update(row["group_id"] for row in holdout[1::2])
    if calibration_groups & confirmation_groups:
        raise RuntimeError("calibration/confirmation groups overlap")
    rows = []
    for source in pool["records"]:
        row = dict(source)
        group = row["group_id"]
        row["role"] = (
            "calibration" if group in calibration_groups
            else "confirmation" if group in confirmation_groups
            else "fit"
        )
        rows.append(row)
    summary = {
        family: {
            role: sum(row["family"] == family and row["role"] == role for row in rows)
            for role in ("fit", "calibration", "confirmation")
        }
        for family in FAMILIES
    }
    payload = {
        "schema": "transfer_dg_wfp_e1_manifest_v1",
        "scope": "all_2800_train_records",
        "source_h5": pool["source_h5"],
        "source_h5_sha256": pool["source_h5_sha256"],
        "source_manifest_sha256": pool["source_manifest_sha256"],
        "parent_pool": str(POOL.resolve()),
        "parent_e1": str(E1.resolve()),
        "split": "train",
        "record_count": len(rows),
        "role_counts": {
            role: sum(row["role"] == role for row in rows)
            for role in ("fit", "calibration", "confirmation")
        },
        "family_role_counts": summary,
        "calibration_group_count": len(calibration_groups),
        "confirmation_group_count": len(confirmation_groups),
        "role_group_overlap_count": 0,
        "records": rows,
        "training_full_truth_allowed": True,
        "test_wavefield_access": "registered_early_prefix_only",
        "test_future_truth_access": False,
        "validation_opened": False,
        "test_id_opened": False,
    }
    payload["selection_sha256"] = canonical(rows)
    temporary = OUTPUT.with_name(f"{OUTPUT.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT), "record_count": len(rows),
        "role_counts": payload["role_counts"], "family_role_counts": summary,
        "selection_sha256": payload["selection_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
