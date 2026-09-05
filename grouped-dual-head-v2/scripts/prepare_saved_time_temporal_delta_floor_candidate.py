#!/usr/bin/env python3
"""Materialize a temporal-basis continuation with a floored delta target."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.continuation import (
    build_temporal_delta_floor_pilot_config,
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
    parser.add_argument("--delta-energy-floor-fraction", type=float, default=0.1)
    parser.add_argument("--dense-learning-rate-multiplier", type=float, default=2.0)
    parser.add_argument("--pilot-epochs", type=int, default=3)
    args = parser.parse_args(argv)

    selection = select_best_pilot_candidate(tuple(args.candidate_config))
    config = build_temporal_delta_floor_pilot_config(
        selection,
        artifact_dir=args.artifact_dir,
        delta_energy_floor_fraction=args.delta_energy_floor_fraction,
        dense_learning_rate_multiplier=args.dense_learning_rate_multiplier,
        pilot_epochs=args.pilot_epochs,
    )
    output = Path(args.output_config).resolve()
    report = Path(args.report).resolve()
    _atomic_text(output, yaml.safe_dump(config, sort_keys=False))
    updates_per_epoch = math.ceil(
        2240 / (int(config["macro_records"]) * int(config["macros_per_update"]))
    )
    rank = int(config["variant_overrides"]["temporal_basis_rank"])
    evidence = {
        "schema": "saved_time_temporal_delta_floor_parent_selection_v1",
        "name": selection.name,
        "config_path": str(selection.config_path),
        "epoch": int(selection.epoch),
        "score": float(selection.score),
        "parent_metrics": selection.metrics,
        "parent_loss_components": selection.loss_components,
        "checkpoint": str(selection.checkpoint),
        "checkpoint_identity": str(selection.checkpoint_identity),
        "generated_config": str(output),
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "temporal_basis_rank": rank,
        "effective_batch": int(config["macro_records"])
        * int(config["macros_per_update"]),
        "physical_microbatch_records": int(config["microbatch_records"]),
        "delta_energy_floor_fraction": float(args.delta_energy_floor_fraction),
        "dense_learning_rate_multiplier": float(
            args.dense_learning_rate_multiplier
        ),
        "pilot_epochs": int(args.pilot_epochs),
        "updates_per_epoch": int(updates_per_epoch),
        "expected_optimizer_updates": int(updates_per_epoch * args.pilot_epochs),
    }
    _atomic_text(report, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
