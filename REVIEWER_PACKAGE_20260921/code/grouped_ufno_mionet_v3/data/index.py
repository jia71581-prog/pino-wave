"""Read-only three-family index and content-bound V3 manifest."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES


REQUIRED_INDEX_DATASETS = {
    "medium_type",
    "split",
    "sample_id",
    "group_id",
    "sample_sha256",
    "split_id",
    "time_s",
    "x_m",
    "z_m",
}


def assert_allowed_families(families: Iterable[str]) -> None:
    observed = {str(value) for value in families}
    forbidden = observed - set(ALLOWED_MEDIUM_TYPES)
    if "anomaly" in forbidden:
        raise ValueError("anomaly medium is forbidden by the V3 data contract")
    if forbidden:
        raise ValueError(f"unknown medium families: {sorted(forbidden)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8")


def _text_array(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()[:], dtype=str)


@dataclass(frozen=True)
class V3RecordIndex:
    source_index: int
    sample_id: str
    group_id: str
    sample_sha256: str
    split: str
    split_id: int
    medium_type: str


@dataclass(frozen=True)
class V3DataManifest:
    schema: str
    source_path: str
    source_file_sha256: str
    source_manifest_sha256: str
    source_config_sha256: str
    allowed_medium_types: tuple[str, ...]
    excluded_medium_types: tuple[str, ...]
    counts_before: dict[str, dict[str, int]]
    counts_after: dict[str, int]
    indices_by_split: dict[str, tuple[int, ...]]
    records: tuple[V3RecordIndex, ...]
    time_s: tuple[float, ...]
    x_m: tuple[float, ...]
    z_m: tuple[float, ...]
    digest: str

    def content_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("digest")
        return payload

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


def build_manifest(source_h5: str | Path) -> V3DataManifest:
    path = Path(source_h5).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r", swmr=True) as h5:
        missing = sorted(REQUIRED_INDEX_DATASETS - set(h5))
        if missing:
            raise ValueError(f"source HDF5 is missing index datasets: {missing}")
        medium_types = _text_array(h5["medium_type"])
        splits = _text_array(h5["split"])
        sample_ids = _text_array(h5["sample_id"])
        group_ids = _text_array(h5["group_id"])
        sample_sha256 = _text_array(h5["sample_sha256"])
        split_ids = np.asarray(h5["split_id"][:], dtype=np.int64)
        length = len(medium_types)
        arrays = (splits, sample_ids, group_ids, sample_sha256, split_ids)
        if any(len(values) != length for values in arrays):
            raise ValueError("source index datasets have inconsistent lengths")

        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for split, family in zip(splits, medium_types, strict=True):
            counts[str(split)][str(family)] += 1

        keep = np.isin(medium_types, np.asarray(ALLOWED_MEDIUM_TYPES))
        indices_by_split_mut: dict[str, list[int]] = defaultdict(list)
        records: list[V3RecordIndex] = []
        for source_index in np.flatnonzero(keep):
            index = int(source_index)
            split = str(splits[index])
            family = str(medium_types[index])
            assert_allowed_families((family,))
            indices_by_split_mut[split].append(index)
            records.append(
                V3RecordIndex(
                    source_index=index,
                    sample_id=str(sample_ids[index]),
                    group_id=str(group_ids[index]),
                    sample_sha256=str(sample_sha256[index]),
                    split=split,
                    split_id=int(split_ids[index]),
                    medium_type=family,
                )
            )

        counts_before = {
            split: dict(sorted(family_counts.items()))
            for split, family_counts in sorted(counts.items())
        }
        indices_by_split = {
            split: tuple(values) for split, values in sorted(indices_by_split_mut.items())
        }
        counts_after = {split: len(values) for split, values in indices_by_split.items()}
        excluded = tuple(sorted(set(medium_types.tolist()) - set(ALLOWED_MEDIUM_TYPES)))
        base = V3DataManifest(
            schema="phase_aligned_complex_fno_mionet_v3_manifest",
            source_path=str(path),
            source_file_sha256=_sha256_file(path),
            source_manifest_sha256=str(h5.attrs.get("manifest_sha256", "")),
            source_config_sha256=str(h5.attrs.get("config_sha256", "")),
            allowed_medium_types=ALLOWED_MEDIUM_TYPES,
            excluded_medium_types=excluded,
            counts_before=counts_before,
            counts_after=counts_after,
            indices_by_split=indices_by_split,
            records=tuple(records),
            time_s=tuple(float(value) for value in np.asarray(h5["time_s"][:], dtype=np.float64)),
            x_m=tuple(float(value) for value in np.asarray(h5["x_m"][:], dtype=np.float64)),
            z_m=tuple(float(value) for value in np.asarray(h5["z_m"][:], dtype=np.float64)),
            digest="",
        )
    digest = hashlib.sha256(_canonical_bytes(base.content_payload())).hexdigest()
    return V3DataManifest(**{**base.__dict__, "digest": digest})


def validate_expected_counts(
    manifest: V3DataManifest,
    expected: Mapping[str, int],
) -> None:
    mismatches = {
        split: (int(expected_count), manifest.counts_after.get(split, 0))
        for split, expected_count in expected.items()
        if manifest.counts_after.get(split, 0) != int(expected_count)
    }
    if mismatches:
        raise ValueError(f"filtered V3 census mismatch: {mismatches}")


def write_manifest_atomic(manifest: V3DataManifest, output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    encoded = json.dumps(manifest.to_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    try:
        with partial.open("x", encoding="utf8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


__all__ = [
    "ALLOWED_MEDIUM_TYPES",
    "V3DataManifest",
    "V3RecordIndex",
    "assert_allowed_families",
    "build_manifest",
    "validate_expected_counts",
    "write_manifest_atomic",
]

