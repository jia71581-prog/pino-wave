"""Lazy HDF5 dataset exposing one source per training record."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from ..contracts import SingleSourceRecord, SourceParameters
from .manifest import GroupManifest


@dataclass
class QueryBlock:
    """One independently sampled block backed by the same loaded frames."""
    coords: torch.Tensor  # [frames, receivers, 3] in [x,z,t] order
    targets: torch.Tensor  # [frames, receivers]


@dataclass
class GroupedSample:
    """A record plus deterministic receiver/time queries.

    ``pressure_tzx`` is retained for full-field losses; ``target`` contains a
    compact ``[frames, blocks, receivers]`` view for query losses.
    """
    record: SingleSourceRecord
    target: torch.Tensor | None = None
    query_tzxs: torch.Tensor | None = None

    @property
    def query_blocks(self) -> tuple[QueryBlock, ...]:
        if self.query_tzxs is None or self.target is None:
            return ()
        return tuple(QueryBlock(self.query_tzxs[:, i], self.target[:, i])
                     for i in range(self.query_tzxs.shape[1]))

    @property
    def velocity_mps(self): return self.record.velocity_mps
    @property
    def source_parameters(self): return self.record.source_parameters
    @property
    def pressure_tzx(self): return self.record.pressure_tzx
    @property
    def query_coords(self): return self.query_tzxs
    @property
    def sample_id(self): return self.record.sample_id
    @property
    def group_id(self): return self.record.group_id

    def __getitem__(self, key: str):
        values = {
            "velocity_mps": self.velocity_mps,
            "source_parameters": self.source_parameters,
            "pressure_tzx": self.pressure_tzx,
            "target": self.target,
            "query_tzxs": self.query_tzxs,
            "query_coords": self.query_tzxs,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "metadata": self.record.metadata,
        }
        return values[key]


class GroupedWavefieldDataset(Dataset):
    """Read LWC-84 records lazily and keep each record's source independent."""

    def __init__(self, path: str, split: str = "train", frames: int = 32,
                 blocks: int = 4, receivers: int = 512, *, return_full_field: bool = True,
                 seed: int = 17):
        if frames <= 0 or blocks <= 0 or receivers <= 0:
            raise ValueError("frames, blocks, and receivers must be positive")
        self.path, self.split = str(path), split
        self.frames, self.blocks, self.receivers = int(frames), int(blocks), int(receivers)
        self.return_full_field, self.seed = return_full_field, int(seed)
        self.manifest = GroupManifest.from_hdf5(path, split)
        self._h5 = None

    def __len__(self): return len(self.manifest)

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
            try: h5.close()
            except Exception: pass

    def _time_indices(self, nt: int) -> np.ndarray:
        return np.rint(np.linspace(0, nt - 1, min(self.frames, nt))).astype(np.int64)

    def _query(self, frames: np.ndarray, time_idx: np.ndarray, idx: int, h5,
               *, nt: int, nz: int, nx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build four query blocks from frames already read exactly once.

        ``frames`` has shape ``[len(time_idx), z, x]``.  Training therefore
        reads 32 saved frames rather than the complete 401-frame wavefield.
        """
        if frames.shape != (len(time_idx), nz, nx):
            raise ValueError("loaded wavefield frames have an unexpected shape")
        flat = nz * nx
        rng = np.random.default_rng(self.seed + int(idx) * 1009)
        block_edges = np.linspace(0, flat, self.blocks + 1, dtype=np.int64)
        coords = np.empty((len(time_idx), self.blocks, self.receivers, 3), dtype=np.float32)
        target = np.empty((len(time_idx), self.blocks, self.receivers), dtype=np.float32)
        x = np.asarray(h5["x_m"][:], dtype=np.float32) if "x_m" in h5 else np.arange(nx, dtype=np.float32)
        z = np.asarray(h5["z_m"][:], dtype=np.float32) if "z_m" in h5 else np.arange(nz, dtype=np.float32)
        t = np.asarray(h5["time_s"][:], dtype=np.float32) if "time_s" in h5 else np.arange(nt, dtype=np.float32)
        for bi, (lo, hi) in enumerate(zip(block_edges[:-1], block_edges[1:])):
            pool = np.arange(lo, max(lo + 1, hi), dtype=np.int64)
            picks = rng.choice(pool, self.receivers, replace=len(pool) < self.receivers)
            zz, xx = np.divmod(picks, nx)
            for ti, frame in enumerate(time_idx):
                coords[ti, bi, :, 0] = x[xx]
                coords[ti, bi, :, 1] = z[zz]
                coords[ti, bi, :, 2] = t[frame]
                target[ti, bi] = frames[ti, zz, xx]
        return torch.from_numpy(coords), torch.from_numpy(target)

    def __getitem__(self, item: int) -> GroupedSample:
        row = self.manifest.rows[int(item)]
        h5 = self._file()
        i = row.index
        velocity = np.asarray(h5["velocity_mps"][i], dtype=np.float32)
        pressure = None
        frames = None
        time_idx = None
        if "wavefield" in h5:
            wavefield = h5["wavefield"]
            nt, nz, nx = wavefield.shape[1:]
            time_idx = self._time_indices(nt)
            # Full fields are reserved for evaluation/export.  The training
            # path below never reads more than the configured saved frames.
            if self.return_full_field:
                pressure = np.asarray(wavefield[i], dtype=np.float32)
                frames = pressure[time_idx]
            else:
                frames = np.asarray(wavefield[i, time_idx, :, :], dtype=np.float32)
        def scalar(name, default):
            return float(h5[name][i]) if name in h5 else float(default)
        source = SourceParameters(
            scalar("source_x_m", 0.5 * float(h5["x_m"][-1]) if "x_m" in h5 else 0.0),
            scalar("source_z_m", 0.5 * float(h5["z_m"][-1]) if "z_m" in h5 else 0.0),
            scalar("source_f0_hz", 10.0), scalar("source_t0_s", 0.0),
            scalar("source_amplitude", 1.0),
        )
        pressure_tensor = torch.from_numpy(pressure) if self.return_full_field and pressure is not None else None
        record = SingleSourceRecord(
            velocity_mps=torch.from_numpy(velocity[None]), source_parameters=source,
            pressure_tzx=pressure_tensor, sample_id=row.sample_id, group_id=row.group_id,
            metadata={"index": i, "split": row.split,
                      "frames_read": 0 if time_idx is None else int(time_idx.size)},
        )
        if frames is None or time_idx is None:
            return GroupedSample(record)
        query_coords, target = self._query(frames, time_idx, i, h5, nt=nt, nz=nz, nx=nx)
        return GroupedSample(record, target=target, query_tzxs=query_coords)
