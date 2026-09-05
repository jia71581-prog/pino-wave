#!/usr/bin/env python3
"""Freeze the unused Marmousi portion of the existing R38 fit pool.

The supplement is the exact set difference between R38 fit Marmousi groups
and the R40 selected Marmousi groups.  Reserved development/R29B groups stay
excluded because the source set is R38 ``fit_records`` only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "r40_frequency_residual_manifest_v1"


def canonical_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verification = dict(payload)
    expected = str(verification.pop("selection_sha256"))
    if canonical_sha(verification) != expected:
        raise RuntimeError(f"manifest digest mismatch: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r38-manifest", type=Path, required=True)
    parser.add_argument("--r40-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    r38_path = args.r38_manifest.expanduser().resolve()
    r40_path = args.r40_manifest.expanduser().resolve()
    r38 = verified_payload(r38_path)
    r40 = verified_payload(r40_path)
    if r38.get("status") != "frozen_before_cache_generation":
        raise RuntimeError("R38 manifest is not frozen")
    if r40.get("status") != "frozen_before_cache_generation":
        raise RuntimeError("R40 manifest is not frozen")
    if str(r40.get("source_r38_selection_sha256")) != str(
        r38["selection_sha256"]
    ):
        raise RuntimeError("R40 does not derive from the supplied R38 manifest")
    for key in (
        "source_h5",
        "source_h5_byte_count",
        "source_manifest_sha256",
        "source_config_sha256",
    ):
        if r40[key] != r38[key]:
            raise RuntimeError(f"R38/R40 source mismatch for {key}")

    selected = set(map(str, r40["selected_groups"]["marmousi"]))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in r38["fit_records"]:
        if str(row["family"]) != "marmousi":
            continue
        grouped.setdefault(str(row["group_id"]), []).append(dict(row))
    missing_groups = sorted(set(grouped) - selected)
    if len(grouped) != 132 or len(selected) != 104 or len(missing_groups) != 28:
        raise RuntimeError(
            "unexpected Marmousi coverage: "
            f"R38={len(grouped)}, R40={len(selected)}, missing={len(missing_groups)}"
        )
    fit_records = []
    for group_id in missing_groups:
        fit_records.extend(
            sorted(grouped[group_id], key=lambda row: str(row["sample_id"]))
        )
    fit_records.sort(key=lambda row: int(row["source_index"]))
    holdout_records = [dict(row) for row in r38["holdout_records"]]
    fit_groups = {str(row["group_id"]) for row in fit_records}
    holdout_groups = {str(row["group_id"]) for row in holdout_records}
    if fit_groups & holdout_groups:
        raise RuntimeError("supplement leaks into development groups")
    if {str(row["sample_id"]) for row in fit_records} & {
        str(row["sample_id"]) for row in holdout_records
    }:
        raise RuntimeError("supplement leaks into development records")
    if len(fit_records) != 140:
        raise RuntimeError(f"expected 140 supplemental records, got {len(fit_records)}")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "frozen_before_cache_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_variant": "R46_unused_R38_fit_Marmousi_supplement_v1",
        "source_r38_manifest": str(r38_path),
        "source_r38_manifest_sha256": sha256_file(r38_path),
        "source_r38_selection_sha256": r38["selection_sha256"],
        "source_r40_manifest": str(r40_path),
        "source_r40_manifest_sha256": sha256_file(r40_path),
        "source_r40_selection_sha256": r40["selection_sha256"],
        "source_h5": r38["source_h5"],
        "source_h5_byte_count": r38["source_h5_byte_count"],
        "source_manifest_sha256": r38["source_manifest_sha256"],
        "source_config_sha256": r38["source_config_sha256"],
        "base_checkpoint": r40["base_checkpoint"],
        "base_checkpoint_sha256": r40["base_checkpoint_sha256"],
        "fit_records": fit_records,
        "holdout_records": holdout_records,
        "selected_groups": {
            "uniform": [],
            "layered": [],
            "marmousi": missing_groups,
        },
        "group_targets": {"uniform": 0, "layered": 0, "marmousi": 28},
        "frequency_policy": dict(r40["frequency_policy"]),
        "spatial_compression": dict(r40["spatial_compression"]),
        "fit_policy": (
            "exact unused Marmousi group set from R38 fit after subtracting R40; "
            "no development or fresh-reserve group is eligible"
        ),
        "holdout_policy": (
            "identity copy of the already-opened R38 development holdout; not "
            "scheduled for supplemental cache generation"
        ),
        "truth_policy": "fit supervision and opened-development evaluation only",
        "evidence_boundary": (
            "R46 supplement may be combined only with R40 fit caches; R29B, "
            "final validation, test, and manuscript remain frozen"
        ),
        "validation_opened": False,
        "test_id_opened": False,
        "r29b_opened": False,
        "paper_modified": False,
    }
    payload["selection_sha256"] = canonical_sha(payload)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "event": "r46_supplement_manifest_frozen",
                "output": str(output),
                "selection_sha256": payload["selection_sha256"],
                "fit_records": len(fit_records),
                "fit_groups": len(fit_groups),
                "marmousi_groups": missing_groups,
                "fit_development_group_overlap": 0,
                "r29b_opened": False,
                "paper_modified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
