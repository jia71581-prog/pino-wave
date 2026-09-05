#!/usr/bin/env python3
"""Materialize V15 from the best comparable completed saved-time pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.continuation import (
    build_architecture_pilot_config,
    select_best_pilot_candidate,
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(value)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-config", action="append", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    selection = select_best_pilot_candidate(tuple(args.candidate_config))
    if "high" not in dict(selection.metrics.get("spectrum_relative_l2", {})):
        raise ValueError("selected V15 parent lacks high-band fixed-panel evidence")
    if "delta" not in selection.loss_components:
        raise ValueError("selected V15 parent lacks delta-loss evidence")
    config = build_architecture_pilot_config(
        selection,
        artifact_dir=args.artifact_dir,
    )
    output_path = Path(args.output_config).resolve()
    report_path = Path(args.report).resolve()
    _atomic_text(output_path, yaml.safe_dump(config, sort_keys=False))
    report = {
        "schema": "saved_time_v15_parent_selection_v1",
        "name": selection.name,
        "config_path": str(selection.config_path),
        "epoch": selection.epoch,
        "score": selection.score,
        "checkpoint": str(selection.checkpoint),
        "checkpoint_identity": str(selection.checkpoint_identity),
        "parent_metrics": selection.metrics,
        "parent_loss_components": selection.loss_components,
        "generated_config": str(output_path),
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
    }
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
