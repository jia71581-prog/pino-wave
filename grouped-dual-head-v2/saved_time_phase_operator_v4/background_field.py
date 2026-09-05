"""Direction A+1: smoothed-velocity background P_bg residual coupling.

Because encode_pressure is LINEAR (value / (pressure_scale * amplitude)),
    encode(wf) - encode(P_bg) = encode(wf - P_bg) = encode(scat).
So the operator can keep predicting a normalized field, but we retarget it onto the
SCATTERING residual by subtracting the (fixed, physical) background P_bg from the target,
and add encode(P_bg) back when scoring the full field. At deploy / held-out (G3) the same
P_bg comes from a cheap smoothed-velocity solve -- no target needed for that direction.

This helper reads per-(record, frame) P_bg from a NumericalTeacherCache-format file
(built by scripts/build_smoothed_background_cache.py, all 401 frames) and aligns it to a
PilotBatch by sample_id + exact frame index.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from saved_time_phase_operator_v4.multifidelity import NumericalTeacherCache


class BackgroundFieldProvider:
    """Per-(record, frame) physical background P_bg aligned to a batch."""

    def __init__(self, cache_path: str | Path | Sequence[str | Path]):
        paths = (
            (cache_path,)
            if isinstance(cache_path, (str, Path))
            else tuple(cache_path)
        )
        if not paths:
            raise ValueError("background provider requires at least one cache")
        self._caches = tuple(NumericalTeacherCache(path) for path in paths)
        spatial_shapes = {cache.spatial_shape for cache in self._caches}
        if len(spatial_shapes) != 1:
            self.close()
            raise ValueError("background cache spatial shapes disagree")

        sample_ids: list[str] = []
        sample_caches: dict[str, list[int]] = {}
        for cache_index, cache in enumerate(self._caches):
            for sample_id in cache.sample_ids:
                sample_caches.setdefault(sample_id, []).append(cache_index)
                if sample_id not in sample_ids:
                    sample_ids.append(sample_id)
        self._sample_ids = tuple(sample_ids)
        self._sample_caches = {
            sample_id: tuple(cache_indices)
            for sample_id, cache_indices in sample_caches.items()
        }
        common_times = set(self._caches[0].time_indices)
        for cache in self._caches[1:]:
            common_times.intersection_update(cache.time_indices)
        if not common_times:
            self.close()
            raise ValueError("background caches have no common saved-time indices")
        self._time_indices = tuple(sorted(int(value) for value in common_times))

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    @property
    def time_indices(self) -> tuple[int, ...]:
        """Strict common time pool available for every cache shard."""

        return self._time_indices

    def covers(
        self,
        sample_ids: Sequence[str],
        time_indices: Sequence[int] | None = None,
    ) -> bool:
        requested_times = None if time_indices is None else set(int(v) for v in time_indices)
        for sample_id in sample_ids:
            candidates = self._sample_caches.get(str(sample_id), ())
            if not candidates:
                return False
            if requested_times is not None and not any(
                requested_times.issubset(self._caches[index].time_indices)
                for index in candidates
            ):
                return False
        return True

    def _cache_for(self, sample_id: str, time_indices: Sequence[int]) -> NumericalTeacherCache:
        requested = set(int(value) for value in time_indices)
        for cache_index in self._sample_caches.get(str(sample_id), ()):
            cache = self._caches[cache_index]
            if requested.issubset(cache.time_indices):
                return cache
        raise KeyError(
            f"no background cache covers sample {sample_id!r} at "
            f"{len(requested)} requested time indices"
        )

    def close(self) -> None:
        for cache in getattr(self, "_caches", ()):
            cache.close()

    def physical(
        self,
        sample_ids: Sequence[str],
        frame_indices: torch.Tensor,   # (records, count) long, EXACT saved-time indices
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return P_bg physical field [records, count, H, W] aligned to (sample, frame).

        The cache read() requires strictly-increasing distinct time columns per record, so
        we gather unique frames then scatter back to the requested (possibly repeated /
        unordered) layout -- keeping the query-invariance-friendly per-frame independence.
        """
        samples = [str(s) for s in sample_ids]
        fi = torch.as_tensor(frame_indices, device="cpu", dtype=torch.long)
        if fi.ndim != 2 or fi.shape[0] != len(samples):
            raise ValueError("frame_indices must be (records, count) matching sample_ids")
        records, count = fi.shape
        out_rows = []
        for r, sample in enumerate(samples):
            frames = fi[r].numpy()
            uniq, inverse = np.unique(frames, return_inverse=True)
            cache = self._cache_for(sample, uniq)
            block = cache.read([sample], torch.as_tensor(uniq[None]))        # [1,U,H,W]
            block = block[0]                                                 # [U,H,W]
            gathered = block[torch.as_tensor(inverse, dtype=torch.long)]     # [count,H,W]
            out_rows.append(gathered)
        stacked = torch.stack(out_rows, dim=0)                               # [R,count,H,W]
        return stacked.to(device=device, dtype=dtype)

    def full_physical(
        self,
        sample_ids: Sequence[str],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the complete cached ``P_bg`` time axis for each sample.

        The background-conditioned Helmholtz branch transforms this fixed complete
        axis once per record.  Its conditioning therefore does not depend on which
        saved frames happen to be requested in a particular query batch.
        """

        samples = tuple(str(value) for value in sample_ids)
        if not samples:
            raise ValueError("full background lookup requires at least one sample")
        indices = torch.as_tensor(self.time_indices, dtype=torch.long)
        requested = indices[None].expand(len(samples), -1)
        return self.physical(
            samples,
            requested,
            device=device,
            dtype=dtype,
        )
