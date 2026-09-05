"""Strict reader for structured dual-head V2 cache files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def select_dense_times(time_s, source_t0_s: float, count: int = 16) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=np.float64)
    if time_s.ndim != 1 or count < 2 or count > len(time_s):
        raise ValueError("time axis and dense frame count are incompatible")
    k0 = min(int(np.searchsorted(time_s, source_t0_s, side="left")), len(time_s) - 2)
    selected = [k0, k0 + 1]
    candidates = np.rint(np.linspace(k0 + 2, len(time_s) - 1, max(count * 2, 2))).astype(int)
    for index in candidates:
        if index not in selected:
            selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in range(k0 + 2, len(time_s)):
            if index not in selected:
                selected.append(index)
            if len(selected) == count:
                break
    selected[-1] = len(time_s) - 1
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class V2CacheRecord:
    velocity_mps: torch.Tensor
    source_parameters: torch.Tensor
    source_map: torch.Tensor
    dense_time_indices: torch.Tensor
    dense_target: torch.Tensor
    receiver_zx_indices: torch.Tensor
    receiver_target: torch.Tensor
    query_coords: torch.Tensor
    query_target: torch.Tensor
    sample_probability: torch.Tensor
    time_s: torch.Tensor
    sample_id: str
    group_id: str
    medium_type: str
    metadata: dict[str, object]


class StructuredCacheDataset(Dataset):
    REQUIRED = {
        "velocity_mps", "source_parameters", "source_map", "dense_time_indices", "dense_target",
        "receiver_zx_indices", "receiver_target", "query_coords", "query_target",
        "sample_probability", "time_s", "sample_id", "group_id", "medium_type",
    }

    def __init__(self, path: str | Path, *, expected_split: str):
        self.path = str(path)
        self.expected_split = str(expected_split)
        self._h5 = None
        with h5py.File(self.path, "r", swmr=True) as h5:
            missing = sorted(self.REQUIRED - set(h5))
            if missing:
                raise ValueError(f"V2 cache missing datasets: {missing}")
            if h5.attrs.get("schema", "") != "grouped_dual_head_v2":
                raise ValueError("unexpected V2 cache schema")
            if str(h5.attrs.get("split", "")) != self.expected_split:
                raise ValueError("cache split does not match expected split")
            if not bool(h5.attrs.get("complete", False)):
                raise ValueError("incomplete V2 cache")
            self.length = int(h5["velocity_mps"].shape[0])
            if h5["source_parameters"].shape != (self.length, 5):
                raise ValueError("source_parameters must be [records,5]")
            if h5["source_map"].shape != h5["velocity_mps"].shape:
                raise ValueError("source_map must match velocity spatial shape")
            if h5["dense_target"].shape[:2] != h5["dense_time_indices"].shape:
                raise ValueError("dense targets and time indices disagree")
            if h5["query_coords"].shape[:-1] != h5["query_target"].shape or h5["query_coords"].shape[-1] != 3:
                raise ValueError("query cache tensors disagree")
            if h5["sample_probability"].shape != h5["query_target"].shape:
                raise ValueError("sample probabilities must match query targets")

    def __len__(self):
        return self.length

    def _file(self):
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getstate__(self):
        state = dict(self.__dict__); state["_h5"] = None; return state

    def __getitem__(self, index: int) -> V2CacheRecord:
        h5 = self._file(); i = int(index)
        text = lambda name: h5[name].asstr()[i]
        probability = torch.from_numpy(np.asarray(h5["sample_probability"][i], np.float32))
        if not torch.isfinite(probability).all() or torch.any(probability <= 0):
            raise ValueError("sample probabilities must be finite and positive")
        source_map = torch.from_numpy(np.asarray(h5["source_map"][i], np.float32))[None]
        if torch.any(source_map < 0) or not torch.allclose(source_map.sum(), torch.tensor(1.0), atol=2e-4):
            raise ValueError("source map must be nonnegative with unit mass")
        return V2CacheRecord(
            torch.from_numpy(np.asarray(h5["velocity_mps"][i], np.float32))[None],
            torch.from_numpy(np.asarray(h5["source_parameters"][i], np.float32)), source_map,
            torch.from_numpy(np.asarray(h5["dense_time_indices"][i], np.int64)),
            torch.from_numpy(np.asarray(h5["dense_target"][i], np.float32)),
            torch.from_numpy(np.asarray(h5["receiver_zx_indices"][i], np.int64)),
            torch.from_numpy(np.asarray(h5["receiver_target"][i], np.float32)),
            torch.from_numpy(np.asarray(h5["query_coords"][i], np.float32)),
            torch.from_numpy(np.asarray(h5["query_target"][i], np.float32)), probability,
            torch.from_numpy(np.asarray(h5["time_s"], np.float32)), text("sample_id"), text("group_id"),
            text("medium_type"), {"split": self.expected_split,
            "source_dataset_sha256": str(h5.attrs["source_dataset_sha256"]),
            "normalization_sha256": str(h5.attrs["normalization_sha256"])},
        )
