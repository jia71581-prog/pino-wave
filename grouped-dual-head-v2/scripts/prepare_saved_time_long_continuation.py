#!/usr/bin/env python3
"""Select a controlled pilot and write its evidence-bound long continuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.continuation import (
    build_long_continuation_config,
    select_best_pilot_candidate,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-config", action="append", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--physical-microbatch-records", type=int, default=2)
    args = parser.parse_args(argv)

    selection = select_best_pilot_candidate(tuple(args.candidate_config))
    config = build_long_continuation_config(
        selection,
        artifact_dir=args.artifact_dir,
        epochs=args.epochs,
        physical_microbatch_records=args.physical_microbatch_records,
    )
    output_path = Path(args.output_config)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "name": selection.name,
        "config_path": str(selection.config_path),
        "epoch": selection.epoch,
        "score": selection.score,
        "family_relative_l2": selection.family_relative_l2,
        "checkpoint": str(selection.checkpoint),
        "checkpoint_identity": str(selection.checkpoint_identity),
        "long_config": str(output_path.resolve()),
        "long_artifact_dir": str(Path(args.artifact_dir).resolve()),
        "physical_microbatch_records": int(config["microbatch_records"]),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
