"""Lazy exact and continuous-time reads from the full V3 source VDS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .index import V3DataManifest, V3RecordIndex, assert_allowed_families


def select_balanced_exact_times(
    time_s: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    source_t0_s: float,
    count: int,
) -> np.ndarray:
    """Choose equal deterministic exact-frame counts from early/middle/late propagation."""
    axis = np.asarray(torch.as_tensor(time_s, dtype=torch.float64).cpu())
    if axis.ndim != 1 or len(axis) < 3 or count < 3 or count % 3:
        raise ValueError("time axis must be 1-D and count must be a positive multiple of three")
    onset = min(int(np.searchsorted(axis, source_t0_s, side="left")), len(axis) - 1)
    candidates = np.arange(onset, len(axis), dtype=np.int64)
    if len(candidates) < count:
        raise ValueError("not enough post-onset frames for balanced exact selection")
    per_phase = count // 3
    selected: list[np.ndarray] = []
    for phase in np.array_split(candidates, 3):
        if len(phase) < per_phase:
            raise ValueError("not enough frames in an early/middle/late phase")
        offsets = np.rint(np.linspace(0, len(phase) - 1, per_phase)).astype(np.int64)
        selected.append(phase[offsets])
    result = np.concatenate(selected)
    result[0] = onset
    result[-1] = len(axis) - 1
    return result


@dataclass(frozen=True)
class V3Record:
    velocity_mps: torch.Tensor
    source_parameters: torch.Tensor
    source_map: torch.Tensor
    time_s: torch.Tensor
    x_m: torch.Tensor
    z_m: torch.Tensor
    source_index: int
    sample_id: str
    group_id: str
    medium_type: str
    split: str
    sample_sha256: str


@dataclass(frozen=True)
class WavefieldTarget:
    values: torch.Tensor
    requested_time_s: torch.Tensor
    left_index: torch.Tensor
    right_index: torch.Tensor
    alpha: torch.Tensor
    exact: torch.Tensor


class V3WavefieldDataset(Dataset[V3Record]):
    """A filtered record view; `__getitem__` never reads the wavefield tensor."""

    REQUIRED = {
        "velocity_mps",
        "wavefield",
        "source_map",
        "source_x_m",
        "source_z_m",
        "source_f0_hz",
        "source_t0_s",
        "source_amplitude",
    }

    def __init__(
        self,
        source_h5: str | Path,
        manifest: V3DataManifest,
        *,
        split: str | Sequence[str],
    ) -> None:
        self.path = str(Path(source_h5).expanduser().resolve())
        if self.path != manifest.source_path:
            raise ValueError("source path does not match the V3 manifest")
        self.manifest_digest = manifest.digest
        splits = (str(split),) if isinstance(split, str) else tuple(str(value) for value in split)
        if not splits or len(splits) != len(set(splits)):
            raise ValueError("V3 splits must be a nonempty unique sequence")
        self.splits = splits
        # Keep the historical attribute for callers that inspect a single-split view.
        self.split = splits[0] if len(splits) == 1 else splits
        selected = set(splits)
        self.records = tuple(record for record in manifest.records if record.split in selected)
        if not self.records:
            raise ValueError(f"V3 split view is empty: {self.splits}")
        assert_allowed_families(record.medium_type for record in self.records)
        self.time_s = torch.tensor(manifest.time_s, dtype=torch.float32)
        self.x_m = torch.tensor(manifest.x_m, dtype=torch.float32)
        self.z_m = torch.tensor(manifest.z_m, dtype=torch.float32)
        self._h5: h5py.File | None = None
        with h5py.File(self.path, "r", swmr=True) as h5:
            missing = sorted(self.REQUIRED - set(h5))
            if missing:
                raise ValueError(f"source HDF5 is missing V3 record datasets: {missing}")
            if h5["wavefield"].shape[1] != len(self.time_s):
                raise ValueError("wavefield and time axis lengths disagree")
            if h5["velocity_mps"].shape[-2:] != (len(self.z_m), len(self.x_m)):
                raise ValueError("velocity and coordinate grid shapes disagree")

    def __len__(self) -> int:
        return len(self.records)

    def _file(self) -> h5py.File:
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    def _metadata(self, index: int) -> V3RecordIndex:
        if index < 0:
            index += len(self.records)
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        return self.records[index]

    def __getitem__(self, index: int) -> V3Record:
        metadata = self._metadata(int(index))
        h5 = self._file()
        source_index = metadata.source_index
        source = torch.tensor(
            [
                h5["source_x_m"][source_index],
                h5["source_z_m"][source_index],
                h5["source_f0_hz"][source_index],
                h5["source_t0_s"][source_index],
                h5["source_amplitude"][source_index],
            ],
            dtype=torch.float32,
        )
        source_map = torch.from_numpy(np.asarray(h5["source_map"][source_index], np.float32))[None]
        if torch.any(source_map < 0) or not torch.allclose(
            source_map.sum(), torch.tensor(1.0), atol=2.0e-4, rtol=2.0e-4
        ):
            raise ValueError("source map must be nonnegative with unit mass")
        return V3Record(
            velocity_mps=torch.from_numpy(
                np.asarray(h5["velocity_mps"][source_index], np.float32)
            )[None],
            source_parameters=source,
            source_map=source_map,
            time_s=self.time_s,
            x_m=self.x_m,
            z_m=self.z_m,
            source_index=source_index,
            sample_id=metadata.sample_id,
            group_id=metadata.group_id,
            medium_type=metadata.medium_type,
            split=metadata.split,
            sample_sha256=metadata.sample_sha256,
        )

    def read_wavefield(
        self,
        index: int,
        requested_time_s: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> WavefieldTarget:
        metadata = self._metadata(int(index))
        requested = np.asarray(torch.as_tensor(requested_time_s, dtype=torch.float64).cpu())
        if requested.ndim != 1 or requested.size == 0 or not np.isfinite(requested).all():
            raise ValueError("requested times must be a nonempty finite vector")
        axis = self.time_s.double().numpy()
        tolerance = max(np.finfo(np.float64).eps * 16, 1.0e-10)
        if requested.min() < axis[0] - tolerance or requested.max() > axis[-1] + tolerance:
            raise ValueError("requested time is outside the saved physical time axis")
        requested = np.clip(requested, axis[0], axis[-1])
        right = np.searchsorted(axis, requested, side="left").clip(0, len(axis) - 1)
        exact = np.isclose(axis[right], requested, rtol=0.0, atol=tolerance)
        left = np.where(exact, right, right - 1)
        if np.any(left < 0):
            raise ValueError("continuous target has no valid left frame")
        denominator = axis[right] - axis[left]
        alpha = np.zeros_like(requested)
        interpolated = ~exact
        alpha[interpolated] = (
            requested[interpolated] - axis[left[interpolated]]
        ) / denominator[interpolated]

        unique = np.unique(np.concatenate((left, right))).astype(np.int64)
        h5 = self._file()
        frames = np.asarray(h5["wavefield"][metadata.source_index, unique, :, :], np.float32)
        position = {int(frame_index): offset for offset, frame_index in enumerate(unique)}
        left_values = frames[[position[int(value)] for value in left]]
        right_values = frames[[position[int(value)] for value in right]]
        weight = alpha.astype(np.float32)[:, None, None]
        values = left_values * (1.0 - weight) + right_values * weight
        return WavefieldTarget(
            values=torch.from_numpy(values),
            requested_time_s=torch.from_numpy(requested.astype(np.float32)),
            left_index=torch.from_numpy(left.astype(np.int64)),
            right_index=torch.from_numpy(right.astype(np.int64)),
            alpha=torch.from_numpy(alpha.astype(np.float32)),
            exact=torch.from_numpy(exact),
        )


__all__ = [
    "V3Record",
    "V3WavefieldDataset",
    "WavefieldTarget",
    "select_balanced_exact_times",
]
