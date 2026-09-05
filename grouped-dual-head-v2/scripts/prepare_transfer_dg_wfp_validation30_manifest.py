#!/usr/bin/env python3
"""Freeze an evenly spaced 10-record panel for each required validation family."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/transfer_dg_wfp_validation600_manifest_20260903.json"
OUTPUT = ROOT / "results/transfer_dg_wfp_validation30_manifest_20260903.json"
FAMILIES = ("uniform", "layered", "marmousi")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: str(row["sample_id"]))
    indices = [math.floor((position + 0.5) * len(ordered) / count) for position in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("panel selection produced duplicate indices")
    return [ordered[index] for index in indices]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text())
    if source.get("validation_future_truth_opened") or source.get("test_id_opened"):
        raise RuntimeError("source manifest violates frozen split boundary")
    selected = []
    family_indices = {}
    for family in FAMILIES:
        candidates = [row for row in source["records"] if row["family"] == family]
        rows = evenly_spaced(candidates, 10)
        selected.extend(rows)
        family_indices[family] = [str(row["sample_id"]) for row in rows]
    payload = {
        "schema": "transfer_dg_wfp_validation30_public_manifest_v1",
        "split": "validation",
        "record_count": 30,
        "frame_count": 401,
        "family_counts": {family: 10 for family in FAMILIES},
        "records": selected,
        "selection_sha256": canonical(selected),
        "selection_policy": "sort sample_id within family and select ten midpoint-stratified equal-width positions",
        "selection_uses_future_truth": False,
        "selected_sample_ids": family_indices,
        "parent_manifest": str(SOURCE.resolve()),
        "parent_manifest_sha256": sha256(SOURCE),
        "source_h5": source["source_h5"],
        "source_h5_sha256": source["source_h5_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "allowed_model_inputs": source["allowed_model_inputs"],
        "model_input_wavefield_frames": 0,
        "validation_public_inputs_opened": True,
        "validation_future_truth_opened": False,
        "test_id_opened": False,
    }
    temporary = OUTPUT.with_name(f"{OUTPUT.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT),
        "record_count": payload["record_count"],
        "family_counts": payload["family_counts"],
        "selected_sample_ids": family_indices,
        "selection_sha256": payload["selection_sha256"],
        "validation_future_truth_opened": False,
        "test_id_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
