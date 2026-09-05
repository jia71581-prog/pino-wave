#!/usr/bin/env python3
"""Read-only full numeric audit of the manifest-approved acoustic train split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)
    if args.progress_every <= 0:
        raise ValueError("progress cadence must be positive")

    source = Path(args.source_h5).resolve()
    manifest = build_manifest(source)
    validate_expected_counts(
        manifest,
        {"train": 2240, "validation": 480, "test_id": 480, "ood_canonical": 3},
    )
    records = tuple(record for record in manifest.records if record.split == "train")
    source_indices = tuple(int(record.source_index) for record in records)
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("manifest-approved train source indices overlap")

    family_counts: dict[str, int] = {}
    for record in records:
        family_counts[record.medium_type] = family_counts.get(record.medium_type, 0) + 1
    expected_families = {"uniform": 420, "layered": 1120, "marmousi": 700}
    if family_counts != expected_families:
        raise ValueError(f"train family census changed: {family_counts}")

    failures: list[dict[str, object]] = []
    max_abs = 0.0
    min_abs_nonzero = float("inf")
    top_surface_max = 0.0
    velocity_min = float("inf")
    velocity_max = float("-inf")
    with h5py.File(source, "r", swmr=True) as h5:
        ids = tuple(_decode(value) for value in h5["sample_id"][:])
        hashes = tuple(_decode(value) for value in h5["sample_sha256"][:])
        splits = tuple(_decode(value) for value in h5["split"][:])
        media = tuple(_decode(value) for value in h5["medium_type"][:])
        completed = np.asarray(h5["completed_mask"], dtype=bool)
        if not bool(completed.all()):
            raise ValueError("raw VDS completed_mask contains false entries")
        if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
            raise ValueError("raw VDS sample IDs or sample hashes are not unique")
        split_hashes = {
            split: {hashes[i] for i, value in enumerate(splits) if value == split}
            for split in set(splits)
        }
        overlaps = {
            f"{left}_vs_{right}": len(split_hashes[left] & split_hashes[right])
            for left in sorted(split_hashes)
            for right in sorted(split_hashes)
            if left < right
        }
        if any(overlaps.values()):
            raise ValueError(f"raw split sample-hash overlap detected: {overlaps}")

        for ordinal, record in enumerate(records, start=1):
            index = int(record.source_index)
            try:
                if ids[index] != record.sample_id:
                    raise ValueError("sample_id binding mismatch")
                if hashes[index] != record.sample_sha256:
                    raise ValueError("sample_sha256 binding mismatch")
                if splits[index] != "train" or media[index] != record.medium_type:
                    raise ValueError("split/family binding mismatch")
                velocity = np.asarray(h5["velocity_mps"][index])
                wavefield = np.asarray(h5["wavefield"][index])
                source_map = np.asarray(h5["source_map"][index])
                if velocity.shape != (201, 201) or wavefield.shape != (401, 201, 201):
                    raise ValueError("velocity or wavefield shape changed")
                if not (
                    np.isfinite(velocity).all()
                    and np.isfinite(wavefield).all()
                    and np.isfinite(source_map).all()
                ):
                    raise ValueError("non-finite numeric value")
                sample_max = float(np.max(np.abs(wavefield)))
                if sample_max <= 0.0:
                    raise ValueError("zero wavefield")
                mass_error = abs(float(source_map.sum(dtype=np.float64)) - 1.0)
                if mass_error > 2.0e-6:
                    raise ValueError(f"source-map mass error {mass_error}")
                max_abs = max(max_abs, sample_max)
                min_abs_nonzero = min(min_abs_nonzero, sample_max)
                top_surface_max = max(
                    top_surface_max, float(np.max(np.abs(wavefield[:, 0, :])))
                )
                velocity_min = min(velocity_min, float(velocity.min()))
                velocity_max = max(velocity_max, float(velocity.max()))
            except Exception as exc:
                failures.append(
                    {
                        "ordinal": ordinal,
                        "source_index": index,
                        "sample_id": record.sample_id,
                        "error": str(exc),
                    }
                )
            if ordinal % int(args.progress_every) == 0 or ordinal == len(records):
                print(
                    f"audited {ordinal}/{len(records)} train records failures={len(failures)}",
                    flush=True,
                )

    payload = {
        "schema": "full_train_vds_numeric_audit_v1",
        "status": "pass" if not failures and top_surface_max <= 1.0e-7 else "fail",
        "source_h5": str(source),
        "manifest_digest": manifest.digest,
        "manifest_source_sha256": manifest.source_manifest_sha256,
        "train_record_count": len(records),
        "train_family_counts": family_counts,
        "anomaly_records_in_manifest_train": sum(
            record.medium_type == "anomaly" for record in records
        ),
        "split_sample_hash_overlaps": overlaps,
        "max_abs_wavefield": max_abs,
        "minimum_record_max_abs_wavefield": min_abs_nonzero,
        "max_abs_top_surface": top_surface_max,
        "velocity_min_mps": velocity_min,
        "velocity_max_mps": velocity_max,
        "failure_count": len(failures),
        "failures": failures[:100],
        "validation_wavefield_opened": False,
        "test_id_wavefield_opened": False,
    }
    _atomic_json(payload, Path(args.output).resolve())
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
