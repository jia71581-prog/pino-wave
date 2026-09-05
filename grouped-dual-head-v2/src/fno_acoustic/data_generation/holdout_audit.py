"""Pre-truth identity audit for independently registered LWC84 holdouts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _medium_group_payload(row: dict[str, Any]) -> dict[str, Any]:
    medium = str(row["medium_type"])
    parameters = dict(row.get("medium_parameters", {}))
    # These two families are defined completely by their material parameters;
    # the seed is merely how those parameters/crops were selected.
    if medium in {"uniform", "marmousi"}:
        parameters.pop("seed", None)
    if medium == "marmousi":
        # A prepared transpose and its raw source are the same geology. Bind the
        # semantic identity to the raw/geological digest when it is available.
        geology_sha256 = parameters.pop(
            "geology_sha256", parameters.pop("source_sha256", None)
        )
        parameters.pop("source_sha256", None)
        parameters["geology_sha256"] = geology_sha256
    payload: dict[str, Any] = {
        "medium_type": medium,
        "medium_parameters": parameters,
    }
    if medium == "marmousi":
        payload.update(
            {
                "crop_x0_m": float(row["crop_x0_m"]),
                "crop_z0_m": float(row["crop_z0_m"]),
            }
        )
    return payload


def manifest_identity_sets(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    identities = {
        "sample_id": set(),
        "group_id": set(),
        "semantic_group_sha256": set(),
        "problem_sha256": set(),
    }
    for row in rows:
        group_payload = _medium_group_payload(row)
        semantic_group = _canonical_sha256(group_payload)
        problem_payload = {
            "semantic_group_sha256": semantic_group,
            "source_x_m": float(row["source_x_m"]),
            "source_z_m": float(row["source_z_m"]),
            "source_f0_hz": float(row["source_f0_hz"]),
            "source_t0_s": float(row["source_t0_s"]),
            "source_amplitude": float(row["source_amplitude"]),
        }
        identities["sample_id"].add(str(row["sample_id"]))
        identities["group_id"].add(str(row["group_id"]))
        identities["semantic_group_sha256"].add(semantic_group)
        identities["problem_sha256"].add(_canonical_sha256(problem_payload))
    return identities


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"historical manifest is missing: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"historical manifest is empty: {path}")
    return rows


def audit_historical_manifest_disjointness(
    candidate_rows: list[dict[str, Any]],
    historical_manifests: Iterable[str | Path],
) -> dict[str, Any]:
    """Require literal and semantic identities to be new before truth exists."""
    candidate = manifest_identity_sets(candidate_rows)
    historical_union = {name: set() for name in candidate}
    sources: list[dict[str, Any]] = []
    for value in historical_manifests:
        path = Path(value).resolve()
        rows = _read_jsonl(path)
        identities = manifest_identity_sets(rows)
        for name in historical_union:
            historical_union[name].update(identities[name])
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": len(rows),
            }
        )
    if not sources:
        raise ValueError("at least one historical manifest is required")

    intersections = {
        name: sorted(candidate[name] & historical_union[name])
        for name in candidate
    }
    passed = all(not values for values in intersections.values())
    return {
        "schema": "lwc84-holdout-pretruth-disjointness-v1",
        "status": "passed" if passed else "blocked_overlap",
        "passed": passed,
        "candidate_row_count": len(candidate_rows),
        "candidate_unique_counts": {
            name: len(values) for name, values in candidate.items()
        },
        "historical_sources": sources,
        "historical_unique_counts": {
            name: len(values) for name, values in historical_union.items()
        },
        "intersection_counts": {
            name: len(values) for name, values in intersections.items()
        },
        "intersection_examples": {
            name: values[:10] for name, values in intersections.items()
        },
        "post_generation_sample_sha256_audit": "required_before_truth_opening",
    }


def audit_candidate_split_disjointness(
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove all candidate splits are literal and semantic holdouts of each other."""
    splits = tuple(
        sorted(
            {str(row["split"]) for row in candidate_rows},
            key=("train", "validation", "test_id", "ood_canonical").index,
        )
    )
    identities = {
        split: manifest_identity_sets(
            row for row in candidate_rows if str(row["split"]) == split
        )
        for split in splits
    }
    pairwise: dict[str, dict[str, int]] = {}
    examples: dict[str, dict[str, list[str]]] = {}
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            key = f"{left}__vs__{right}"
            overlaps = {
                name: sorted(identities[left][name] & identities[right][name])
                for name in identities[left]
            }
            pairwise[key] = {name: len(values) for name, values in overlaps.items()}
            examples[key] = {name: values[:10] for name, values in overlaps.items()}
    passed = all(
        count == 0
        for intersections in pairwise.values()
        for count in intersections.values()
    )
    return {
        "schema": "lwc84-holdout-internal-split-disjointness-v1",
        "status": "passed" if passed else "blocked_overlap",
        "passed": passed,
        "splits": list(splits),
        "pairwise_intersection_counts": pairwise,
        "pairwise_intersection_examples": examples,
    }


def _decode_hdf5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _sample_hashes(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"sample-hash dataset is missing: {path}")
    with h5py.File(path, "r") as handle:
        if "sample_sha256" not in handle:
            raise ValueError(f"sample_sha256 is missing from {path}")
        values = {
            _decode_hdf5_text(value).strip()
            for value in handle["sample_sha256"][:]
        }
    if not values or "" in values:
        raise ValueError(f"sample_sha256 is empty or incomplete in {path}")
    return values


def audit_generated_sample_sha256_disjointness(
    candidate_dataset: str | Path,
    historical_datasets: Iterable[str | Path],
) -> dict[str, Any]:
    """Post-generation exact content-hash audit, before future truth is opened."""
    candidate_path = Path(candidate_dataset).resolve()
    candidate = _sample_hashes(candidate_path)
    historical: set[str] = set()
    sources: list[dict[str, Any]] = []
    for value in historical_datasets:
        path = Path(value).resolve()
        hashes = _sample_hashes(path)
        historical.update(hashes)
        sources.append({"path": str(path), "sample_hash_count": len(hashes)})
    if not sources:
        raise ValueError("at least one historical dataset is required")
    overlap = sorted(candidate & historical)
    return {
        "schema": "lwc84-holdout-posttruth-sample-sha256-v1",
        "status": "passed" if not overlap else "blocked_overlap",
        "passed": not overlap,
        "candidate_dataset": str(candidate_path),
        "candidate_sample_hash_count": len(candidate),
        "historical_sources": sources,
        "historical_sample_hash_count": len(historical),
        "intersection_count": len(overlap),
        "intersection_examples": overlap[:10],
        "future_truth_opened_by_evaluator": False,
    }


__all__ = [
    "audit_candidate_split_disjointness",
    "audit_generated_sample_sha256_disjointness",
    "audit_historical_manifest_disjointness",
    "manifest_identity_sets",
]
