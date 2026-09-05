#!/usr/bin/env python3
"""Freeze fresh group-disjoint train roles for the phase/scatter-64 pilot."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/transfer_dg_wfp_full2800_manifest_20260902.json"
OUTPUT = ROOT / "results/transfer_dg_phase_scatter64_pilot_manifest_20260903.json"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
GROUP_PLAN = {
    "uniform": {"fit": 16, "calibration": 8, "confirmation": 8},
    "layered": {"fit": 4, "calibration": 2, "confirmation": 2},
    "anomaly": {"fit": 8, "calibration": 4, "confirmation": 4},
    "marmousi": {"fit": 4, "calibration": 2, "confirmation": 2},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def group_order(group_id: str) -> str:
    return hashlib.sha256(f"phase-scatter64-v1:{group_id}".encode()).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text())
    pools = defaultdict(lambda: defaultdict(list))
    for row in source["records"]:
        if row["role"] == "fit":
            pools[row["family"]][row["group_id"]].append(row)
    selected = []
    role_groups = {role: set() for role in ("fit", "calibration", "confirmation")}
    family_role_counts = {}
    for family in FAMILIES:
        groups = sorted(pools[family], key=group_order)
        offset = 0
        family_role_counts[family] = {}
        for role in ("fit", "calibration", "confirmation"):
            count = GROUP_PLAN[family][role]
            chosen = groups[offset:offset + count]
            if len(chosen) != count:
                raise RuntimeError(f"insufficient {family} groups for {role}")
            offset += count
            role_groups[role].update(chosen)
            role_rows = []
            for group in chosen:
                for source_row in pools[family][group]:
                    row = dict(source_row)
                    row["parent_role"] = row["role"]
                    row["role"] = role
                    role_rows.append(row)
            selected.extend(role_rows)
            family_role_counts[family][role] = len(role_rows)
    if any(role_groups[a] & role_groups[b] for a, b in (
        ("fit", "calibration"), ("fit", "confirmation"),
        ("calibration", "confirmation")
    )):
        raise RuntimeError("fresh phase/scatter roles overlap by group")
    payload = {
        "schema": "transfer_dg_phase_scatter64_pilot_manifest_v1",
        "split": "train",
        "source_h5": source["source_h5"],
        "source_h5_sha256": source["source_h5_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "parent_manifest": str(SOURCE.resolve()),
        "parent_manifest_sha256": sha256(SOURCE),
        "record_count": len(selected),
        "role_counts": {
            role: sum(row["role"] == role for row in selected)
            for role in ("fit", "calibration", "confirmation")
        },
        "family_role_counts": family_role_counts,
        "role_group_counts": {role: len(groups) for role, groups in role_groups.items()},
        "role_group_overlap_count": 0,
        "selection_policy": "hash-order groups from the prior fit role; fixed per-family group counts",
        "selection_uses_wavefield_truth": False,
        "records": selected,
        "selection_sha256": canonical(selected),
        "training_full_truth_allowed": True,
        "validation_opened": False,
        "test_id_opened": False,
    }
    temporary = OUTPUT.with_name(f"{OUTPUT.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({key: payload[key] for key in (
        "record_count", "role_counts", "family_role_counts",
        "role_group_counts", "role_group_overlap_count", "selection_sha256",
        "validation_opened", "test_id_opened"
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
