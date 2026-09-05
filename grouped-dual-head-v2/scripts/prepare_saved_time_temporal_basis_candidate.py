#!/usr/bin/env python3
"""Materialize an evidence-bound query-invariant temporal-basis pilot."""
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
    build_temporal_basis_pilot_config,
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
    parser.add_argument("--temporal-basis-rank", type=int, default=96)
    parser.add_argument("--effective-batch", type=int, default=96)
    parser.add_argument("--pilot-epochs", type=int, default=3)
    args = parser.parse_args(argv)

    selection = select_best_pilot_candidate(tuple(args.candidate_config))
    config = build_temporal_basis_pilot_config(
        selection,
        artifact_dir=args.artifact_dir,
        temporal_basis_rank=args.temporal_basis_rank,
        effective_batch=args.effective_batch,
        pilot_epochs=args.pilot_epochs,
    )
    output = Path(args.output_config).resolve()
    report = Path(args.report).resolve()
    _atomic_text(output, yaml.safe_dump(config, sort_keys=False))
    updates_per_epoch = math.ceil(
        2240 / (int(config["macro_records"]) * int(config["macros_per_update"]))
    )
    evidence = {
        "schema": "saved_time_temporal_basis_parent_selection_v1",
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
        "temporal_basis_rank": int(args.temporal_basis_rank),
        "effective_batch": int(args.effective_batch),
        "physical_microbatch_records": int(config["microbatch_records"]),
        "pilot_epochs": int(args.pilot_epochs),
        "updates_per_epoch": int(updates_per_epoch),
        "expected_optimizer_updates": int(updates_per_epoch * args.pilot_epochs),
    }
    _atomic_text(report, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
