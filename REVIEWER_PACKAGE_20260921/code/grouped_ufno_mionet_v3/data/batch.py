"""Pack independent sources while reusing identical medium encodings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .index import assert_allowed_families
from .records import V3Record


@dataclass(frozen=True)
class V3MacroBatch:
    velocity_mps: torch.Tensor
    record_to_medium: torch.Tensor
    source_parameters: torch.Tensor
    source_map: torch.Tensor
    time_s: torch.Tensor
    x_m: torch.Tensor
    z_m: torch.Tensor
    source_index: torch.Tensor
    sample_id: tuple[str, ...]
    group_id: tuple[str, ...]
    medium_type: tuple[str, ...]


def pack_v3_groups(records: Sequence[V3Record]) -> V3MacroBatch:
    if not records:
        raise ValueError("cannot pack an empty V3 batch")
    assert_allowed_families(record.medium_type for record in records)
    media: list[torch.Tensor] = []
    record_to_medium: list[int] = []
    by_group: dict[str, int] = {}
    for record in records:
        velocity = torch.as_tensor(record.velocity_mps, dtype=torch.float32)
        if velocity.ndim != 3 or velocity.shape[0] != 1:
            raise ValueError("velocity must be [1,z,x]")
        if record.group_id in by_group:
            medium_index = by_group[record.group_id]
            if not torch.equal(media[medium_index], velocity):
                raise ValueError(f"group {record.group_id} contains different velocity fields")
        else:
            medium_index = len(media)
            by_group[record.group_id] = medium_index
            media.append(velocity)
        record_to_medium.append(medium_index)
    for name in ("time_s", "x_m", "z_m"):
        reference = torch.as_tensor(getattr(records[0], name))
        if any(not torch.equal(torch.as_tensor(getattr(record, name)), reference) for record in records[1:]):
            raise ValueError(f"all records must share {name}")
    return V3MacroBatch(
        velocity_mps=torch.stack(media),
        record_to_medium=torch.tensor(record_to_medium, dtype=torch.long),
        source_parameters=torch.stack([record.source_parameters for record in records]),
        source_map=torch.stack([record.source_map for record in records]),
        time_s=records[0].time_s,
        x_m=records[0].x_m,
        z_m=records[0].z_m,
        source_index=torch.tensor([record.source_index for record in records], dtype=torch.long),
        sample_id=tuple(record.sample_id for record in records),
        group_id=tuple(record.group_id for record in records),
        medium_type=tuple(record.medium_type for record in records),
    )


__all__ = ["V3MacroBatch", "pack_v3_groups"]

