from __future__ import annotations

import json
import os

import h5py

from scripts.evict_saved_time_dataset_cache import (
    evict_clean_file_pages,
    vds_source_paths,
)
from scripts.steward_saved_time_dataset_cache import (
    cache_decision_events,
    memory_pressure_requires_eviction,
)


def test_vds_sources_are_identity_bound_and_cache_eviction_is_read_only(tmp_path):
    shard = tmp_path / "shard.h5"
    with h5py.File(shard, "w") as handle:
        handle.create_dataset("value", data=[1.0, 2.0])
    manifest = tmp_path / "manifest.h5"
    with h5py.File(manifest, "w") as handle:
        handle.attrs["vds_source_shards"] = json.dumps([str(shard)])
    before = shard.read_bytes()

    paths = vds_source_paths(manifest)
    report = evict_clean_file_pages(paths)

    assert paths == (manifest.resolve(), shard.resolve())
    assert report["status"] == "complete"
    assert report["file_count"] == 2
    assert report["data_mutated"] is False
    assert shard.read_bytes() == before


def test_cache_eviction_deduplicates_explicit_files(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(os.urandom(128))

    report = evict_clean_file_pages((path, path))

    assert report["file_count"] == 1
    assert report["total_file_bytes"] == 128


def test_cache_steward_only_reacts_to_new_accept_or_reject_decisions():
    lines = [
        json.dumps({"event": "validation_baseline", "epoch": 0}),
        json.dumps({"event": "epoch_accepted", "epoch": 1}),
        json.dumps({"event": "epoch_rejected", "epoch": 2}),
    ]

    seen, decisions = cache_decision_events(lines, start=1)

    assert seen == 3
    assert [(row["event"], row["epoch"]) for row in decisions] == [
        ("epoch_accepted", 1),
        ("epoch_rejected", 2),
    ]


def test_cache_steward_respects_memory_high_watermark_and_interval():
    kwargs = {
        "high_watermark_bytes": 220,
        "minimum_interval_seconds": 60.0,
    }
    assert memory_pressure_requires_eviction(
        221, seconds_since_last_eviction=61.0, **kwargs
    )
    assert not memory_pressure_requires_eviction(
        219, seconds_since_last_eviction=61.0, **kwargs
    )
    assert not memory_pressure_requires_eviction(
        221, seconds_since_last_eviction=59.0, **kwargs
    )
