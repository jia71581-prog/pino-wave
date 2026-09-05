#!/usr/bin/env python3
"""R54: from-scratch full-pool training with worker-parallel data loading.

Delegates everything to the R28 entrypoint (same architecture, loss, lr,
batch, sampling).  The only additions are infrastructure, not mathematics:

1. Worker-safe HDF5 handles.  The r25 CacheCollection opens h5py handles in
   the parent process; DataLoader workers must never share them.  A per-PID
   handle map (the pattern R53 validated with --num-workers 4) opens fresh
   read-only handles inside each worker.
2. num_workers for the fit DataLoader only.  The r25 driver hardcodes
   num_workers=0, which serializes HDF5 decompression with GPU compute and
   was measured at 1.64 steps/s on the full pool (26 h projected).  Sample
   order comes from DistributedSampler with the preregistered seed, and
   FitFrameDataset.__getitem__ is a deterministic read, so worker count does
   not alter the optimization trajectory.

Batch size, learning rate, loss weights and sampling are untouched.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import h5py

SCRIPT_PATH = Path(__file__).resolve()
R28_PATH = SCRIPT_PATH.with_name("train_r28_expanded_tail_spectral.py")
SPEC = importlib.util.spec_from_file_location("r28_entry", R28_PATH)
r28 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r28
SPEC.loader.exec_module(r28)
r26 = r28.r26
r25 = r28.r25

FIT_NUM_WORKERS = int(os.environ.get("R54_FIT_NUM_WORKERS", "8"))


class WorkerSafeHandles:
    """List-like façade over CacheCollection handles.

    The parent process keeps its original handles; any other PID lazily opens
    its own read-only set the first time it touches the cache.
    """

    def __init__(self, paths, parent_handles):
        self.paths = tuple(paths)
        self.parent_pid = os.getpid()
        self.parent_handles = parent_handles
        self._per_pid: dict[int, list[h5py.File]] = {}

    def _local(self):
        pid = os.getpid()
        if pid == self.parent_pid:
            return self.parent_handles
        if pid not in self._per_pid:
            self._per_pid[pid] = [
                h5py.File(path, "r", swmr=True) for path in self.paths
            ]
        return self._per_pid[pid]

    def __getitem__(self, index):
        return self._local()[index]

    def __iter__(self):
        return iter(self._local())

    def __len__(self):
        return len(self.parent_handles)


_original_collection_init = r25.CacheCollection.__init__


def _patched_collection_init(self, paths, *, expected_subset):
    _original_collection_init(self, paths, expected_subset=expected_subset)
    self.handles = WorkerSafeHandles(self.paths, list(self.handles))


r25.CacheCollection.__init__ = _patched_collection_init

_original_loader = r25.DataLoader


def _patched_loader(dataset, *args, **kwargs):
    if isinstance(dataset, r25.FitFrameDataset) and kwargs.get("num_workers") == 0:
        kwargs["num_workers"] = FIT_NUM_WORKERS
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return _original_loader(dataset, *args, **kwargs)


r25.DataLoader = _patched_loader


def main() -> None:
    r28.SCRIPT_PATH = SCRIPT_PATH
    r28.main()


if __name__ == "__main__":
    main()
