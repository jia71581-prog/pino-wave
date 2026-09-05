#!/usr/bin/env python3
"""Freeze unseen train-only groups for the restored branch-trunk U-FNO pilot."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/transfer_dg_wfp_full2800_manifest_20260902.json"
EXCLUDES = (
    ROOT / "results/transfer_dg_phase_scatter64_pilot_manifest_20260903.json",
    ROOT / "results/transfer_dg_coupled_mhc_muon_pilot_manifest_20260903.json",
)
OUTPUT = ROOT / "results/transfer_dg_coupled_mhc_muon_bt_ufno_pilot_manifest_20260903.json"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
GROUP_PLAN = {
    "uniform": {"fit": 16, "calibration": 8, "confirmation": 8},
    "layered": {"fit": 4, "calibration": 2, "confirmation": 2},
    "anomaly": {"fit": 8, "calibration": 4, "confirmation": 4},
    "marmousi": {"fit": 4, "calibration": 2, "confirmation": 2},
}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def order(group: str) -> str:
    return hashlib.sha256(f"coupled-mhc-muon-bt-ufno-v1:{group}".encode()).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text())
    excluded_groups = {
        row["group_id"]
        for path in EXCLUDES
        for row in json.loads(path.read_text())["records"]
    }
    pools = defaultdict(lambda: defaultdict(list))
    for row in source["records"]:
        if row["role"] == "fit" and row["group_id"] not in excluded_groups:
            pools[row["family"]][row["group_id"]].append(row)

    selected = []
    role_groups = {name: set() for name in ("fit", "calibration", "confirmation")}
    family_role_counts = {}
    for family in FAMILIES:
        groups = sorted(pools[family], key=order)
        cursor = 0
        family_role_counts[family] = {}
        for role in ("fit", "calibration", "confirmation"):
            count = GROUP_PLAN[family][role]
            chosen = groups[cursor : cursor + count]
            if len(chosen) != count:
                raise RuntimeError(f"insufficient unseen {family} groups for {role}")
            cursor += count
            role_groups[role].update(chosen)
            rows = []
            for group in chosen:
                for original in pools[family][group]:
                    row = dict(original)
                    row["role"] = role
                    rows.append(row)
            selected.extend(rows)
            family_role_counts[family][role] = len(rows)

    overlaps = sum(
        len(role_groups[a] & role_groups[b])
        for a, b in (
            ("fit", "calibration"),
            ("fit", "confirmation"),
            ("calibration", "confirmation"),
        )
    )
    if overlaps or any(row["group_id"] in excluded_groups for row in selected):
        raise RuntimeError("restored pilot split overlaps prior train-only evidence")
    payload = {
        "schema": "transfer_dg_coupled_mhc_muon_bt_ufno_pilot_manifest_v1",
        "split": "train",
        "source_h5": source["source_h5"],
        "record_count": len(selected),
        "role_counts": {
            role: sum(row["role"] == role for row in selected)
            for role in ("fit", "calibration", "confirmation")
        },
        "family_role_counts": family_role_counts,
        "role_group_overlap_count": overlaps,
        "excluded_prior_group_count": len(excluded_groups),
        "selection_uses_wavefield_truth": False,
        "selection_policy": "hash-order unseen parent-fit groups after excluding both prior pilots",
        "records": selected,
        "selection_sha256": digest(selected),
        "validation_opened": False,
        "test_id_opened": False,
    }
    temporary = OUTPUT.with_name(f"{OUTPUT.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "record_count",
                    "role_counts",
                    "family_role_counts",
                    "role_group_overlap_count",
                    "excluded_prior_group_count",
                    "selection_sha256",
                    "validation_opened",
                    "test_id_opened",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
