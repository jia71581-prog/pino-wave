"""Group media without ever combining independent single-source targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .cache import V2CacheRecord


@dataclass(frozen=True)
class V2MacroBatch:
    velocity_mps: torch.Tensor
    record_to_medium: torch.Tensor
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
    sample_id: tuple[str, ...]
    group_id: tuple[str, ...]
    medium_type: tuple[str, ...]


def pack_v2_groups(records: Sequence[V2CacheRecord]) -> V2MacroBatch:
    if not records:
        raise ValueError("cannot pack an empty V2 batch")
    media, mapping, by_group = [], [], {}
    for record in records:
        velocity = torch.as_tensor(record.velocity_mps, dtype=torch.float32)
        if velocity.ndim != 3 or velocity.shape[0] != 1:
            raise ValueError("velocity must be [1,z,x]")
        if record.group_id in by_group:
            medium_index = by_group[record.group_id]
            if not torch.equal(media[medium_index], velocity):
                raise ValueError(f"group {record.group_id} contains different velocity fields")
        else:
            medium_index = len(media); by_group[record.group_id] = medium_index; media.append(velocity)
        source_map = torch.as_tensor(record.source_map, dtype=torch.float32)
        if source_map.ndim != 3 or source_map.shape[0] != 1 or torch.any(source_map < 0):
            raise ValueError("source map must be nonnegative [1,z,x]")
        if not torch.allclose(source_map.sum(), torch.tensor(1.0), atol=2e-4, rtol=2e-4):
            raise ValueError("source map must have unit mass")
        mapping.append(medium_index)
    time_s = torch.as_tensor(records[0].time_s, dtype=torch.float32)
    if any(not torch.equal(torch.as_tensor(record.time_s), time_s) for record in records[1:]):
        raise ValueError("all records in a batch must share the saved-time axis")
    stack = lambda name, dtype=None: torch.stack([
        torch.as_tensor(getattr(record, name), dtype=dtype) for record in records
    ])
    return V2MacroBatch(
        torch.stack(media), torch.tensor(mapping, dtype=torch.long),
        stack("source_parameters", torch.float32), stack("source_map", torch.float32),
        stack("dense_time_indices", torch.long), stack("dense_target", torch.float32),
        stack("receiver_zx_indices", torch.long), stack("receiver_target", torch.float32),
        stack("query_coords", torch.float32), stack("query_target", torch.float32),
        stack("sample_probability", torch.float32), time_s,
        tuple(record.sample_id for record in records), tuple(record.group_id for record in records),
        tuple(record.medium_type for record in records),
    )
