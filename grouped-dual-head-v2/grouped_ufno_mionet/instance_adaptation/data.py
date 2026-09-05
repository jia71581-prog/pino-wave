"""Readers that make future-label access impossible during deployment adaptation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from .causal import WavefieldAccessAudit, onset_window


@dataclass(frozen=True)
class OnsetDeploymentExample:
    velocity_mps: torch.Tensor
    source: torch.Tensor
    time_s: torch.Tensor
    observed_wavefield: torch.Tensor
    observed_indices: tuple[int, int]
    accessed_wavefield_indices: tuple[int, int]
    sample_index: int


class OnsetDeploymentDataset(Dataset):
    """HDF5 adapter reader which accesses exactly two onset-aligned snapshots."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        with h5py.File(self.path, "r", swmr=True) as h5:
            if not {"velocity_mps", "wavefield", "time_s", "source_t0_s"} <= set(h5):
                raise ValueError("dataset lacks required grouped onset-adaptation keys")
            self.length = int(h5["velocity_mps"].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> OnsetDeploymentExample:
        with h5py.File(self.path, "r", swmr=True) as h5:
            time_s = torch.as_tensor(h5["time_s"][:], dtype=torch.float32)
            t0 = float(h5["source_t0_s"][index])
            window = onset_window(time_s, t0)
            audit = WavefieldAccessAudit(window.observed_indices)
            audit.record(window.observed_indices)
            observed = torch.as_tensor(h5["wavefield"][index, list(window.observed_indices)], dtype=torch.float32)
            def value(name, default): return float(h5[name][index]) if name in h5 else default
            source = torch.tensor([value("source_x_m", 0.0), value("source_z_m", 0.0),
                                   value("source_f0_hz", 10.0), t0, value("source_amplitude", 1.0)])
            velocity = torch.as_tensor(h5["velocity_mps"][index], dtype=torch.float32)
        return OnsetDeploymentExample(velocity, source, time_s, observed, window.observed_indices,
                                      audit.accessed_indices, int(index))
