"""Small, split-aware manifest for the LWC-84 HDF5/VDS dataset.

The production file is a virtual HDF5 dataset.  Reading the one-dimensional
metadata arrays is cheap, while wavefields remain on disk until a dataset item
is requested.  Group IDs are treated as medium IDs; the dataset performs an
exact velocity check before sharing an encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import numpy as np

import h5py


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf8", errors="replace")
    return value.item() if hasattr(value, "item") else value


def _read_column(h5: h5py.File, name: str, n: int, default=None) -> list:
    if name not in h5:
        return [default] * n
    return [_decode(v) for v in h5[name][:]]


@dataclass(frozen=True)
class ManifestRow:
    index: int
    sample_id: str
    group_id: str
    split: str
    source_x_m: float | None = None
    source_z_m: float | None = None
    source_f0_hz: float | None = None


@dataclass(frozen=True)
class GroupManifest:
    """Metadata index for one split of an HDF5 dataset."""

    path: str
    split: str
    rows: tuple[ManifestRow, ...]
    all_splits: tuple[str, ...] = ()

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(row.index for row in self.rows)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row.sample_id for row in self.rows)

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(row.group_id for row in self.rows)

    @property
    def local_by_global(self) -> dict[int, int]:
        return {row.index: pos for pos, row in enumerate(self.rows)}

    @property
    def groups(self) -> dict[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = {}
        for row in self.rows:
            grouped.setdefault(row.group_id, []).append(row.index)
        return {key: tuple(value) for key, value in grouped.items()}

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def from_hdf5(cls, path: str | Path, split: str = "train") -> "GroupManifest":
        path = str(path)
        with h5py.File(path, "r", swmr=True) as h5:
            if "velocity_mps" not in h5:
                raise ValueError(f"{path} is missing velocity_mps")
            n = int(h5["velocity_mps"].shape[0])
            splits = _read_column(h5, "split", n, "train")
            split_names = tuple(dict.fromkeys(str(v) for v in splits))
            if split not in split_names:
                raise ValueError(f"split {split!r} not found; available={split_names}")
            sample_ids = _read_column(h5, "sample_id", n)
            groups = _read_column(h5, "group_id", n)
            xs = _read_column(h5, "source_x_m", n)
            zs = _read_column(h5, "source_z_m", n)
            fs = _read_column(h5, "source_f0_hz", n)
            rows = tuple(
                ManifestRow(
                    index=i,
                    sample_id=str(sample_ids[i] if sample_ids[i] is not None else i),
                    group_id=str(groups[i] if groups[i] is not None else f"medium-{i}"),
                    split=str(splits[i]),
                    source_x_m=None if xs[i] is None else float(xs[i]),
                    source_z_m=None if zs[i] is None else float(zs[i]),
                    source_f0_hz=None if fs[i] is None else float(fs[i]),
                )
                for i in range(n)
                if str(splits[i]) == split
            )
            # A group is a medium-sharing optimization only when all of its
            # records are byte-identical in velocity.  Check this before a
            # loader is allowed to reuse an encoder result.
            by_group: dict[str, list[ManifestRow]] = {}
            for row in rows:
                by_group.setdefault(row.group_id, []).append(row)
            for group_id, group_rows in by_group.items():
                reference = None
                reference_digest = None
                for row in group_rows:
                    array = np.ascontiguousarray(h5["velocity_mps"][row.index])
                    digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
                    if reference_digest is None:
                        reference_digest, reference = digest, array
                    elif digest != reference_digest or not np.array_equal(array, reference):
                        ids = [item.sample_id for item in group_rows]
                        raise ValueError(f"group_id={group_id} contains different velocity fields; sample_ids={ids}")
        return cls(path=path, split=split, rows=rows, all_splits=split_names)


def validate_split_isolation(path: str | Path) -> dict[str, object]:
    """Validate that a sample ID is assigned to at most one split.

    Returns a compact report and raises ``ValueError`` for duplicate IDs or
    duplicate ``(group_id, sample_id)`` assignments.  Duplicate medium IDs are
    explicitly allowed because a medium is expected to have multiple sources.
    """
    path = str(path)
    with h5py.File(path, "r", swmr=True) as h5:
        if "velocity_mps" not in h5:
            raise ValueError(f"{path} is missing velocity_mps")
        n = int(h5["velocity_mps"].shape[0])
        splits = [str(_decode(v)) for v in _read_column(h5, "split", n, "train")]
        raw_ids = _read_column(h5, "sample_id", n)
        ids = [str(_decode(v) if v is not None else i) for i, v in enumerate(raw_ids)]
        seen: dict[str, str] = {}
        group_seen: dict[str, str] = {}
        duplicates: list[str] = []
        group_values = [str(_decode(v)) for v in _read_column(h5, "group_id", n)]
    for sample_id, split, group_id in zip(ids, splits, group_values):
        if sample_id in seen and seen[sample_id] != split:
            duplicates.append(sample_id)
        seen.setdefault(sample_id, split)
        if group_id in group_seen and group_seen[group_id] != split:
            duplicates.append(f"group_id={group_id}")
        group_seen.setdefault(group_id, split)
    if duplicates:
        raise ValueError(f"split leakage across sample/group IDs: {sorted(set(duplicates))[:5]}")
    return {
        "path": path,
        "sample_count": n,
        "splits": {name: splits.count(name) for name in dict.fromkeys(splits)},
        "sample_ids_unique": len(set(ids)) == len(ids),
        "cross_split_duplicates": tuple(sorted(set(duplicates))),
    }
