"""Dataset backed by the compact, precomputed sparse-query HDF5 cache.

The cache is intentionally a *training* representation.  It contains the
velocity, one source, and the already selected query coordinates/pressure
values for every record.  In particular it never opens, follows, or indexes
the source HDF5 ``wavefield`` VDS.

The cache writer's canonical layout is top-level datasets::

    velocity_mps      [N, z, x]
    source_parameters [N, 5]
    query_coords      [N, frames, blocks, receivers, 3]
    targets           [N, frames, blocks, receivers]

It also carries ``sample_id``, ``group_id`` and ``global_index`` columns (and
the cache split as an attribute), so its
:class:`~grouped_ufno_mionet.data.manifest.GroupManifest` is compatible with
:class:`GroupedBatchSampler`.  A few unambiguous older names are accepted to
make partially built caches usable after a restart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..contracts import SingleSourceRecord, SourceParameters
from .dataset import GroupedSample
from .manifest import GroupManifest, ManifestRow


_VELOCITY_NAMES = ("velocity_mps", "records/velocity_mps")
_SOURCE_NAMES = ("source_parameters", "records/source_parameters")
_COORD_NAMES = (
    "query_tzxs", "query_coords", "sparse_query_tzxs", "records/query_tzxs",
)
_TARGET_NAMES = ("targets", "target", "sparse_targets", "records/targets")
_SOURCE_MAP_NAMES = ("source_map", "records/source_map")


def _first_present(h5, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in h5), None)


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf8", errors="replace")
    return value.item() if hasattr(value, "item") else value


class SparseCacheDataset(Dataset):
    """Load precomputed sparse queries without reading a wavefield VDS.

    The public shape and manifest contract deliberately matches
    :class:`GroupedWavefieldDataset`: indices are local to ``split`` while the
    manifest retains storage/global indices for :class:`GroupedBatchSampler`.
    ``frames``, ``blocks`` and ``receivers`` are checked against the cache,
    rather than used to resample data.
    """

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        frames: int | None = 32,
        blocks: int | None = 4,
        receivers: int | None = 512,
        *,
        return_full_field: bool = False,
        seed: int = 17,
    ):
        if return_full_field:
            raise ValueError(
                "SparseCacheDataset has no full wavefield; set return_full_field=False"
            )
        for name, value in (("frames", frames), ("blocks", blocks), ("receivers", receivers)):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive or None")
        self.path, self.split = str(path), str(split)
        self.frames = None if frames is None else int(frames)
        self.blocks = None if blocks is None else int(blocks)
        self.receivers = None if receivers is None else int(receivers)
        self.return_full_field = False
        # Kept for drop-in construction compatibility.  Queries are fixed by
        # the cache and this value is intentionally never used for sampling.
        self.seed = int(seed)
        self._h5 = None
        self._dataset_names = self._inspect_layout()
        self.manifest, self._storage_by_global = self._build_manifest()

    def _inspect_layout(self) -> dict[str, str]:
        """Validate the compact cache once, without touching ``wavefield``."""
        import h5py

        with h5py.File(self.path, "r", swmr=True) as h5:
            found = {
                "velocity": _first_present(h5, _VELOCITY_NAMES),
                "source": _first_present(h5, _SOURCE_NAMES),
                "coords": _first_present(h5, _COORD_NAMES),
                "target": _first_present(h5, _TARGET_NAMES),
                "source_map": _first_present(h5, _SOURCE_MAP_NAMES),
            }
            missing = [key for key, value in found.items() if value is None]
            if missing:
                raise ValueError(
                    f"{self.path} is not a complete sparse cache; missing {missing}"
                )
            velocity = h5[found["velocity"]]
            source = h5[found["source"]]
            coords = h5[found["coords"]]
            target = h5[found["target"]]
            source_map = h5[found["source_map"]]
            if velocity.ndim != 3:
                raise ValueError("cached velocity_mps must have shape [records,z,x]")
            if source.shape != (velocity.shape[0], 5):
                raise ValueError("cached source_parameters must have shape [records,5]")
            if source_map.shape != velocity.shape:
                raise ValueError("cached source_map must have shape [records,z,x]")
            if coords.ndim != 5 or coords.shape[-1] != 3:
                raise ValueError("cached query_tzxs must have shape [records,frames,blocks,receivers,3]")
            if target.shape != coords.shape[:-1]:
                raise ValueError("cached targets must match query_tzxs without its coordinate axis")
            if coords.shape[0] != velocity.shape[0]:
                raise ValueError("cached record arrays have inconsistent lengths")
            if "completed" in h5 and not bool(np.asarray(h5["completed"][:], dtype=bool).all()):
                raise ValueError("sparse cache contains incomplete records")
            actual = tuple(int(v) for v in coords.shape[1:4])
            expected = (self.frames, self.blocks, self.receivers)
            mismatches = [
                f"{name}={got} (cache) != {want} (requested)"
                for name, got, want in zip(("frames", "blocks", "receivers"), actual, expected)
                if want is not None and got != want
            ]
            if mismatches:
                raise ValueError("sparse cache query shape mismatch: " + "; ".join(mismatches))
        return {key: str(value) for key, value in found.items()}

    def _build_manifest(self) -> tuple[GroupManifest, dict[int, int]]:
        """Build a manifest from cache metadata without indexing a VDS.

        ``global_index`` preserves the source-file identifier used by the
        regular manifest/sampler, while cache rows are compacted.  Keep the
        explicit translation so an arbitrary subset cache remains valid.
        """
        import h5py

        with h5py.File(self.path, "r", swmr=True) as h5:
            n = int(h5[self._dataset_names["velocity"]].shape[0])

            def column(name: str, default):
                if name not in h5:
                    return [default] * n
                values = h5[name][:]
                if len(values) != n:
                    raise ValueError(f"cached {name} has length {len(values)}, expected {n}")
                return [_decode(value) for value in values]

            # Split is optional for a cache written for exactly one split.
            # The writer records that split as an attribute in this case.
            if "split" in h5:
                splits = [str(value) for value in column("split", self.split)]
            else:
                cache_split = str(_decode(h5.attrs.get("split", self.split)))
                if cache_split != self.split:
                    raise ValueError(
                        f"cache was built for split {cache_split!r}, not {self.split!r}"
                    )
                splits = [cache_split] * n
            sample_ids = column("sample_id", None)
            group_ids = column("group_id", None)
            global_ids = column("global_index", None)
            all_splits = tuple(dict.fromkeys(splits))
            if self.split not in all_splits:
                raise ValueError(f"split {self.split!r} not found; available={all_splits}")
            storage_by_global: dict[int, int] = {}
            rows: list[ManifestRow] = []
            for storage_index, (row_split, sample_id, group_id, global_id) in enumerate(
                zip(splits, sample_ids, group_ids, global_ids)
            ):
                index = storage_index if global_id is None else int(global_id)
                if index in storage_by_global:
                    raise ValueError(f"cached global_index is not unique: {index}")
                storage_by_global[index] = storage_index
                if row_split == self.split:
                    rows.append(
                        ManifestRow(
                            index=index,
                            sample_id=str(storage_index if sample_id is None else sample_id),
                            group_id=str(f"medium-{index}" if group_id is None else group_id),
                            split=row_split,
                        )
                    )
        return (
            GroupManifest(path=self.path, split=self.split, rows=tuple(rows), all_splits=all_splits),
            storage_by_global,
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def _file(self):
        if self._h5 is None or not self._h5.id.valid:
            import h5py

            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    def __del__(self):
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass

    @staticmethod
    def _source(values: np.ndarray) -> SourceParameters:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (5,):
            raise ValueError(f"cached source_parameters must be [5], got {values.shape}")
        return SourceParameters(*(float(value) for value in values))

    def __getitem__(self, item: int) -> GroupedSample:
        row = self.manifest.rows[int(item)]
        # Manifest indices remain source/global IDs.  The explicit map lets
        # grouped samplers translate those IDs to split-local Dataset indices
        # even though this cache contains compacted storage rows.
        index = row.index
        storage_index = self._storage_by_global[index]
        h5 = self._file()
        names = self._dataset_names
        velocity = np.asarray(h5[names["velocity"]][storage_index], dtype=np.float32)
        source = self._source(h5[names["source"]][storage_index])
        coords = np.asarray(h5[names["coords"]][storage_index], dtype=np.float32)
        target = np.asarray(h5[names["target"]][storage_index], dtype=np.float32)
        if coords.shape != target.shape + (3,):
            raise ValueError("cached query/target item shapes disagree")
        record = SingleSourceRecord(
            velocity_mps=torch.from_numpy(velocity[None]),
            source_parameters=source,
            pressure_tzx=None,
            sample_id=row.sample_id,
            group_id=row.group_id,
            metadata={
                "index": index,
                "storage_index": storage_index,
                "split": row.split,
                "frames_read": int(coords.shape[0]),
                "storage": "sparse_cache",
            },
        )
        return GroupedSample(
            record,
            target=torch.from_numpy(target),
            query_tzxs=torch.from_numpy(coords),
        )
