"""HDF5 data plumbing for the grouped single-source operator."""

from .manifest import GroupManifest, ManifestRow, validate_split_isolation
from .dataset import GroupedSample, GroupedWavefieldDataset
from .sparse_cache import SparseCacheDataset
from .group_sampler import GroupedBatchSampler, MediumGroupedBatchSampler, pack_groups
from .prefetch import (
    BoundedPrefetchLoader, PrefetchLoader, bounded_prefetch,
    identity_collate, make_dataloader, prefetch_dataloader,
)

__all__ = [
    "GroupManifest",
    "ManifestRow",
    "validate_split_isolation",
    "GroupedSample",
    "GroupedWavefieldDataset",
    "SparseCacheDataset",
    "GroupedBatchSampler",
    "MediumGroupedBatchSampler",
    "pack_groups",
    "BoundedPrefetchLoader",
    "PrefetchLoader",
    "identity_collate",
    "bounded_prefetch",
    "make_dataloader",
    "prefetch_dataloader",
]
