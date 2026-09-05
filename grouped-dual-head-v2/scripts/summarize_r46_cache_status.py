#!/usr/bin/env python3
"""Print one compact status object for the four R46 supplement shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    shards = []
    for shard in range(4):
        log_path = output / f"fit_supplement_shard{shard}.log"
        final_path = output / f"fit_supplement_shard{shard}.h5"
        summary_path = final_path.with_suffix(final_path.suffix + ".summary.json")
        partials = list(output.glob(f".fit_supplement_shard{shard}.h5.partial-*"))
        cached = []
        errors = []
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if "Traceback" in line or "Error" in line or "Exception" in line:
                        errors.append(line.strip())
                    continue
                if event.get("event") == "cached_record":
                    cached.append(event)
        summary = None
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shards.append(
            {
                "shard": shard,
                "cached_records": len(cached),
                "expected_records": 35,
                "last_sample_id": cached[-1].get("sample_id") if cached else None,
                "last_base_rel_l2": cached[-1].get("base_rel_l2") if cached else None,
                "partial_bytes": sum(path.stat().st_size for path in partials),
                "final_exists": final_path.is_file(),
                "summary_status": summary.get("status") if summary else None,
                "summary_sha256": summary.get("output_sha256") if summary else None,
                "error_lines": errors[-3:],
            }
        )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "cached_records": sum(row["cached_records"] for row in shards),
                "expected_records": 140,
                "complete_shards": sum(
                    row["summary_status"] == "complete" for row in shards
                ),
                "has_errors": any(row["error_lines"] for row in shards),
                "shards": shards,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
