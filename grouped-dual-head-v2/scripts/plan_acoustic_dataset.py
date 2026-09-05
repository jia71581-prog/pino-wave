#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fno_acoustic.data_generation.config import artifact_root, grid_from_config, load_config, time_from_config
from fno_acoustic.data_generation.split_manifest import build_manifest, validate_manifest, write_manifest_artifacts
from fno_acoustic.data_generation.pipeline_lwc84 import plan_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan and freeze v3 acoustic dataset manifest without computing wavefields.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if str(config.get("schema_version", "")).startswith("acoustic-lwc84"):
        output = Path(args.output) if args.output else Path(config["paths"]["output_root"])
        summary, exit_code = plan_dataset(config, output=output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return exit_code
    if not args.profile:
        raise ValueError("legacy dataset planning requires --profile")
    from fno_acoustic.data_generation.cfl import compute_cfl_plan

    rows = build_manifest(config, profile=args.profile, enabled_categories=None)
    if args.splits:
        allowed = {s.strip() for s in args.splits.split(",") if s.strip()}
        rows = [row for row in rows if row["split"] in allowed]
    summary = validate_manifest(rows, config, profile=args.profile)
    grid = grid_from_config(config)
    time = time_from_config(config)
    cfl = compute_cfl_plan(c_global_max_mps=5500.0, grid=grid, time=time)
    disk = shutil.disk_usage(config["paths"]["output_root"])
    summary.update(
        {
            "c_global_min_mps": 1500.0,
            "c_global_max_mps": 5500.0,
            "cfl_plan": cfl.as_dict(),
            "estimated_wavefield_bytes": int(len(rows) * grid.nz * grid.nx * time.nt_out * 4),
            "disk_free_bytes": int(disk.free),
        }
    )
    if args.write_manifest:
        summary["manifest_artifacts"] = write_manifest_artifacts(rows, config, summary)
    output = Path(args.output) if args.output else artifact_root(config) / "plan_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
