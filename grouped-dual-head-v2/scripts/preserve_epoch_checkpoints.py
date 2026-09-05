#!/usr/bin/env python3
"""Preserve epoch checkpoints from an already-running retention-limited job.

The watcher creates same-filesystem hard links in ``preserved_checkpoints``.
Removing the original retention-managed name therefore cannot remove the
underlying checkpoint.  It never opens or rewrites checkpoint payloads.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any


SCHEMA = "saved_time_preserved_epoch_checkpoints_v1"


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _owner_is_alive(owner_pid: int | None) -> bool:
    if owner_pid is None:
        return True
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        return False
    return True


def preserve_available(run_dir: Path, owner_pid: int | None) -> list[dict[str, Any]]:
    checkpoint_dir = run_dir / "checkpoints"
    archive_dir = run_dir / "preserved_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    for source in sorted(checkpoint_dir.glob("epoch_[0-9][0-9][0-9][0-9].pt")):
        target = archive_dir / source.name
        if not target.exists():
            try:
                os.link(source, target)
            except FileExistsError:
                pass
            except FileNotFoundError:
                # The trainer may have pruned the source between glob and link.
                continue
        if not source.exists():
            continue
        source_stat = source.stat()
        target_stat = target.stat()
        if (source_stat.st_dev, source_stat.st_ino) != (
            target_stat.st_dev,
            target_stat.st_ino,
        ):
            raise RuntimeError(f"archive collision is not a hard link: {target}")

    preserved: list[dict[str, Any]] = []
    for target in sorted(archive_dir.glob("epoch_[0-9][0-9][0-9][0-9].pt")):
        stat = target.stat()
        preserved.append(
            {
                "name": target.name,
                "bytes": int(stat.st_size),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "link_count": int(stat.st_nlink),
            }
        )
    _atomic_json(
        {
            "schema": SCHEMA,
            "run_dir": str(run_dir.resolve()),
            "owner_pid": owner_pid,
            "owner_alive": _owner_is_alive(owner_pid),
            "updated_unix_s": time.time(),
            "preserved": preserved,
        },
        archive_dir / "manifest.json",
    )
    return preserved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    previous_names: tuple[str, ...] = ()
    while True:
        preserved = preserve_available(run_dir, args.owner_pid)
        current_names = tuple(item["name"] for item in preserved)
        if current_names != previous_names:
            print(
                json.dumps(
                    {
                        "event": "preserved_epoch_checkpoints",
                        "count": len(current_names),
                        "names": current_names,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            previous_names = current_names
        if args.once:
            return 0
        if not _owner_is_alive(args.owner_pid):
            # One last sweep happened above after the owner exited.
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
