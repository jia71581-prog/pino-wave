"""Medium-aware batching: encode each velocity field once per macro-batch."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Iterator

import torch
from torch.utils.data import Sampler

from ..contracts import GroupedMacroBatch, SingleSourceRecord
from .dataset import QueryBlock


def _get(sample, key):
    if isinstance(sample, Mapping):
        return sample[key]
    if hasattr(sample, key):
        return getattr(sample, key)
    if hasattr(sample, "record") and hasattr(sample.record, key):
        return getattr(sample.record, key)
    raise TypeError(f"sample has no {key!r}")


def pack_groups(samples: Sequence, max_records: int = 24) -> GroupedMacroBatch:
    """Pack independent source records while deduplicating identical media.

    ``samples`` may contain :class:`GroupedSample`, ``SingleSourceRecord`` or
    dictionaries with equivalent fields.  Targets stay record-specific and
    are stacked only when all records expose tensors of the same shape.
    """
    if not samples:
        raise ValueError("cannot pack an empty group")
    if max_records <= 0 or len(samples) > max_records:
        raise ValueError(f"records={len(samples)} exceeds max_records={max_records}")
    media: list[torch.Tensor] = []
    medium_index: dict[str, int] = {}
    record_to_medium: list[int] = []
    source_params: list[torch.Tensor] = []
    targets = []
    sample_blocks = []
    ids: list[str | int] = []
    for sample in samples:
        velocity = torch.as_tensor(_get(sample, "velocity_mps"), dtype=torch.float32)
        if velocity.ndim == 2:
            velocity = velocity.unsqueeze(0)
        if velocity.ndim != 3:
            raise ValueError(f"velocity must be [1,z,x] or [z,x], got {tuple(velocity.shape)}")
        key = str(_get(sample, "group_id") or f"velocity-{len(media)}")
        if key in medium_index:
            mi = medium_index[key]
            if media[mi].shape != velocity.shape or not torch.equal(media[mi], velocity):
                raise ValueError(f"group_id {key!r} maps to different velocity fields")
        else:
            mi = len(media)
            medium_index[key] = mi
            media.append(velocity)
        record_to_medium.append(mi)
        src = _get(sample, "source_parameters")
        if hasattr(src, "as_tensor"):
            src = src.as_tensor(dtype=torch.float32)
        src = torch.as_tensor(src, dtype=torch.float32)
        if src.shape != (5,):
            raise ValueError(f"source_parameters must have shape [5], got {tuple(src.shape)}")
        source_params.append(src)
        target = None
        try: target = _get(sample, "target")
        except TypeError: pass
        if target is None:
            try: target = _get(sample, "pressure_tzx")
            except TypeError: pass
        targets.append(None if target is None else torch.as_tensor(target))
        sample_blocks.append(getattr(sample, "query_blocks", ()))
        try: ids.append(_get(sample, "sample_id"))
        except TypeError: ids.append(len(ids))
    target_tensor = None
    if targets and all(t is not None and t.shape == targets[0].shape for t in targets):
        target_tensor = torch.stack(targets)
    return GroupedMacroBatch(
        velocity_mps=torch.stack(media),
        record_to_medium=torch.tensor(record_to_medium, dtype=torch.long),
        source_parameters=torch.stack(source_params),
        targets=target_tensor,
        sample_id=tuple(ids),
        query_blocks=tuple(
            QueryBlock(torch.stack([blocks[i].coords for blocks in sample_blocks]),
                       torch.stack([blocks[i].targets for blocks in sample_blocks]))
            for i in range(len(sample_blocks[0]))
        ) if sample_blocks and all(sample_blocks) and len({len(v) for v in sample_blocks}) == 1 else (),
    )


class GroupedBatchSampler(Sampler[list[int]]):
    """Yield bounded batches with nearby records from the same medium."""
    def __init__(self, manifest_or_dataset, batch_size: int = 24, *, shuffle: bool = True, seed: int = 17):
        if batch_size <= 0: raise ValueError("batch_size must be positive")
        self.manifest = getattr(manifest_or_dataset, "manifest", manifest_or_dataset)
        self.batch_size, self.shuffle, self.seed = int(batch_size), bool(shuffle), int(seed)
        # Manifest rows retain global HDF5 indices; Dataset indices are local
        # to the selected split, so translate before yielding batches.
        local = self.manifest.local_by_global
        self._groups = [[local[index] for index in group] for group in self.manifest.groups.values()]
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return (sum(len(g) for g in self._groups) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        groups = [list(g) for g in self._groups]
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            order = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[i] for i in order]
            for group in groups:
                if len(group) > 1:
                    perm = torch.randperm(len(group), generator=generator).tolist()
                    group[:] = [group[i] for i in perm]
        # Pack records from successive groups into a macro-batch.  Group IDs
        # are retained by ``pack_groups`` so media are still encoded once,
        # while small (1--5 source) groups no longer underfill the GPU.
        records = [index for group in groups for index in group]
        for start in range(0, len(records), self.batch_size):
            yield records[start:start + self.batch_size]


MediumGroupedBatchSampler = GroupedBatchSampler
