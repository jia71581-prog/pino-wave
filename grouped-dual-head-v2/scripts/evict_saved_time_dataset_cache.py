#!/usr/bin/env python
"""Evict clean page-cache entries for one saved-time VDS and its sources."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import h5py


def vds_source_paths(manifest_path: Path) -> tuple[Path, ...]:
    """Return the identity-bound VDS manifest and all registered source shards."""

    manifest = manifest_path.resolve(strict=True)
    with h5py.File(manifest, "r", swmr=True) as handle:
        encoded = handle.attrs.get("vds_source_shards")
    if encoded is None:
        raise ValueError("saved-time manifest has no vds_source_shards binding")
    if isinstance(encoded, bytes):
        encoded = encoded.decode("utf8")
    sources = json.loads(str(encoded))
    if not isinstance(sources, list) or not sources:
        raise ValueError("saved-time VDS source binding is invalid")
    paths = [manifest]
    for source in sources:
        path = Path(str(source)).resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return tuple(dict.fromkeys(paths))


def evict_clean_file_pages(paths: Iterable[Path]) -> dict[str, object]:
    """Advise the kernel that clean cached pages for explicit files are expendable."""

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    files = tuple(dict.fromkeys(Path(path).resolve(strict=True) for path in paths))
    total_bytes = 0
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            total_bytes += path.stat().st_size
        finally:
            os.close(descriptor)
    return {
        "schema": "saved_time_dataset_cache_eviction_v1",
        "status": "complete",
        "file_count": len(files),
        "total_file_bytes": total_bytes,
        "data_mutated": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extra-file", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    paths = (*vds_source_paths(args.manifest), *args.extra_file)
    print(json.dumps(evict_clean_file_pages(paths), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

