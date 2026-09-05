#!/usr/bin/env python3
"""Freeze a train-only, group-disjoint manifest for R40 frequency refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "r40_frequency_residual_manifest_v1"
GROUP_TARGETS = {"uniform": 128, "layered": 96, "marmousi": 104}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r38-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.r38_manifest.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "frozen_before_cache_generation":
        raise RuntimeError("R38 manifest is not frozen")
    verification = dict(source)
    selection = str(verification.pop("selection_sha256"))
    if canonical_sha(verification) != selection:
        raise RuntimeError("R38 manifest digest mismatch")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        family: {} for family in GROUP_TARGETS
    }
    for row in source["fit_records"]:
        family = str(row["family"])
        grouped[family].setdefault(str(row["group_id"]), []).append(dict(row))

    fit_records: list[dict[str, Any]] = []
    selected_groups: dict[str, list[str]] = {}
    for family, target in GROUP_TARGETS.items():
        groups = sorted(
            grouped[family],
            key=lambda value: hashlib.sha256(
                f"r40-frequency-v1|{family}|{value}".encode("utf-8")
            ).hexdigest(),
        )
        if len(groups) < target:
            raise RuntimeError(f"not enough {family} groups: {len(groups)} < {target}")
        chosen = groups[:target]
        selected_groups[family] = chosen
        for group in chosen:
            fit_records.extend(sorted(grouped[family][group], key=lambda row: row["sample_id"]))

    holdout_records = [dict(row) for row in source["holdout_records"]]
    fit_groups = {str(row["group_id"]) for row in fit_records}
    holdout_groups = {str(row["group_id"]) for row in holdout_records}
    if fit_groups & holdout_groups:
        raise RuntimeError("R40 fit leaks into development holdout groups")
    fit_ids = {str(row["sample_id"]) for row in fit_records}
    holdout_ids = {str(row["sample_id"]) for row in holdout_records}
    if fit_ids & holdout_ids:
        raise RuntimeError("R40 fit/holdout sample overlap")

    checkpoint = args.base_checkpoint.expanduser().resolve()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "frozen_before_cache_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_variant": "R40_spatially_varying_frequency_domain_residual",
        "source_r38_manifest": str(source_path),
        "source_r38_manifest_sha256": sha256_file(source_path),
        "source_r38_selection_sha256": selection,
        "source_h5": source["source_h5"],
        "source_h5_byte_count": source["source_h5_byte_count"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_config_sha256": source["source_config_sha256"],
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": sha256_file(checkpoint),
        "fit_records": sorted(fit_records, key=lambda row: row["source_index"]),
        "holdout_records": holdout_records,
        "selected_groups": selected_groups,
        "group_targets": GROUP_TARGETS,
        "frequency_policy": {
            "minimum_hz": 5.0,
            "maximum_hz": 80.0,
            "selection": "all_rfft_bins_in_closed_band",
            "temporal_fft_norm": "ortho",
        },
        "spatial_compression": {
            "transform": "DCT-II_ortho",
            "retained_shape": [96, 96],
            "full_shape": [201, 201],
        },
        "fit_policy": (
            "whole groups selected deterministically from R38 fit only; Marmousi "
            "groups intentionally emphasized after opened-dev spectral diagnosis"
        ),
        "holdout_policy": "exact already-opened R28/R38 group-disjoint development holdout",
        "truth_policy": "train supervision and opened-development evaluation only",
        "evidence_boundary": (
            "R40 development success only authorizes the single preregistered R29B "
            "fresh group-disjoint evaluation"
        ),
        "validation_opened": False,
        "test_id_opened": False,
    }
    payload["selection_sha256"] = canonical_sha(payload)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "selection_sha256": payload["selection_sha256"],
                "fit_records": len(fit_records),
                "fit_groups": len(fit_groups),
                "holdout_records": len(holdout_records),
                "holdout_groups": len(holdout_groups),
                "family_record_counts": {
                    family: sum(row["family"] == family for row in fit_records)
                    for family in GROUP_TARGETS
                },
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
