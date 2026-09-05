"""Deployment records that expose only the two real onset snapshots."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from grouped_ufno_mionet_v3.data.index import V3DataManifest
from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset
from saved_time_phase_operator_v4.eikonal import EikonalTravelCache

from .contracts import SnapshotAccessAudit, onset_indices


def _digest_tensors(values: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("utf8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class GuardedOnsetRecord:
    velocity_mps: torch.Tensor
    source_parameters: torch.Tensor
    source_map: torch.Tensor
    time_s: torch.Tensor
    x_m: torch.Tensor
    z_m: torch.Tensor
    sample_id: str
    group_id: str
    medium_type: str
    source_index: int
    observed_indices: tuple[int, int]
    observed_wavefield: torch.Tensor
    input_digest: str
    audit: SnapshotAccessAudit
    dense_travel_time_s: torch.Tensor | None = None

    @property
    def public_keys(self) -> tuple[str, ...]:
        return (
            "velocity_mps",
            "source_parameters",
            "source_map",
            "time_s",
            "x_m",
            "z_m",
            "observed_wavefield",
            "observed_indices",
            "dense_travel_time_s",
        )

    def read_truth_forbidden(self, indices: Iterable[int]) -> None:
        self.audit.read(indices)
        raise RuntimeError("deployment records do not expose future truth")


class GuardedOnsetDataset(Dataset[GuardedOnsetRecord]):
    """V3 manifest-backed dataset with an explicit two-frame truth boundary."""

    def __init__(
        self,
        source_h5: str | Path,
        manifest: V3DataManifest,
        *,
        split: str,
        sample_ids: Sequence[str] | None = None,
        travel_time_h5: str | Path | None = None,
    ) -> None:
        self.records = V3WavefieldDataset(source_h5, manifest, split=split)
        wanted = None if sample_ids is None else {str(value) for value in sample_ids}
        self._indices = tuple(
            index
            for index, metadata in enumerate(self.records.records)
            if wanted is None or metadata.sample_id in wanted
        )
        if not self._indices:
            raise ValueError("guarded onset dataset selection is empty")
        self.travel_cache = (
            None
            if travel_time_h5 is None
            else EikonalTravelCache(travel_time_h5, source_h5=source_h5)
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> GuardedOnsetRecord:
        record_index = self._indices[int(index)]
        metadata = self.records._metadata(record_index)
        metadata_record = self.records[record_index]
        observed = onset_indices(
            metadata_record.time_s,
            t0_s=float(metadata_record.source_parameters[3]),
            f0_hz=float(metadata_record.source_parameters[2]),
        )
        audit = SnapshotAccessAudit(observed)
        audit.read(observed)
        h5 = self.records._file()
        frames = np.asarray(
            h5["wavefield"][metadata.source_index, list(observed), :, :], dtype=np.float32
        )
        observed_wavefield = torch.from_numpy(frames)
        input_digest = _digest_tensors(
            (
                metadata_record.velocity_mps,
                metadata_record.source_parameters,
                metadata_record.source_map,
                observed_wavefield,
            )
        )
        return GuardedOnsetRecord(
            velocity_mps=metadata_record.velocity_mps,
            source_parameters=metadata_record.source_parameters,
            source_map=metadata_record.source_map,
            time_s=metadata_record.time_s,
            x_m=metadata_record.x_m,
            z_m=metadata_record.z_m,
            sample_id=metadata.sample_id,
            group_id=metadata.group_id,
            medium_type=metadata.medium_type,
            source_index=metadata.source_index,
            observed_indices=observed,
            observed_wavefield=observed_wavefield,
            input_digest=input_digest,
            audit=audit,
            dense_travel_time_s=(
                None
                if self.travel_cache is None
                else self.travel_cache.read((metadata.sample_id,))[0]
            ),
        )

    def close(self) -> None:
        handle = getattr(self.records, "_h5", None)
        if handle is not None and handle.id.valid:
            handle.close()
        travel_handle = getattr(getattr(self, "travel_cache", None), "_h5", None)
        if travel_handle is not None and travel_handle.id.valid:
            travel_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["GuardedOnsetDataset", "GuardedOnsetRecord"]
