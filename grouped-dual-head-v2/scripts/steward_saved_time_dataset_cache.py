#!/usr/bin/env python
"""Evict saved-time dataset pages after each validation-control decision."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evict_saved_time_dataset_cache import (
    evict_clean_file_pages,
    vds_source_paths,
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _memory_current() -> int | None:
    path = Path("/sys/fs/cgroup/memory.current")
    return int(path.read_text()) if path.is_file() else None


def cache_decision_events(lines: list[str], *, start: int) -> tuple[int, list[dict]]:
    """Return newly committed accept/reject decisions from an append-only history."""

    rows = [json.loads(line) for line in lines[start:] if line.strip()]
    decisions = [
        row
        for row in rows
        if row.get("event") in {"epoch_accepted", "epoch_rejected"}
    ]
    return len(lines), decisions


def memory_pressure_requires_eviction(
    current_bytes: int | None,
    *,
    high_watermark_bytes: int,
    seconds_since_last_eviction: float,
    minimum_interval_seconds: float,
) -> bool:
    return (
        current_bytes is not None
        and int(current_bytes) >= int(high_watermark_bytes)
        and float(seconds_since_last_eviction) >= float(minimum_interval_seconds)
    )


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extra-file", type=Path, action="append", default=[])
    parser.add_argument("--control-history", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--memory-high-watermark-gib", type=float, default=220.0)
    parser.add_argument("--minimum-eviction-interval-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    paths = (*vds_source_paths(args.manifest), *args.extra_file)
    seen = 0
    if args.control_history.is_file():
        seen = len(args.control_history.read_text().splitlines())
    high_watermark_bytes = int(float(args.memory_high_watermark_gib) * 1024**3)
    minimum_interval = float(args.minimum_eviction_interval_seconds)
    if high_watermark_bytes <= 0 or minimum_interval < 0.0:
        raise ValueError("cache-steward memory thresholds are invalid")
    last_eviction = 0.0
    while _alive(args.training_pid):
        lines = (
            args.control_history.read_text().splitlines()
            if args.control_history.is_file()
            else []
        )
        seen, decisions = cache_decision_events(lines, start=seen)
        current_memory = _memory_current()
        pressure = memory_pressure_requires_eviction(
            current_memory,
            high_watermark_bytes=high_watermark_bytes,
            seconds_since_last_eviction=time.monotonic() - last_eviction,
            minimum_interval_seconds=minimum_interval,
        )
        triggers = list(decisions)
        if pressure and not triggers:
            triggers.append({"event": "memory_high_watermark"})
        for decision in triggers:
            before = _memory_current()
            report = evict_clean_file_pages(paths)
            after = _memory_current()
            last_eviction = time.monotonic()
            _append(
                args.output,
                {
                    "event": "dataset_cache_eviction",
                    "trigger": decision.get("event"),
                    "epoch": decision.get("epoch"),
                    "attempt": decision.get("attempt"),
                    "high_watermark_bytes": high_watermark_bytes,
                    "memory_current_before": before,
                    "memory_current_after": after,
                    **report,
                },
            )
        time.sleep(max(float(args.poll_seconds), 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
