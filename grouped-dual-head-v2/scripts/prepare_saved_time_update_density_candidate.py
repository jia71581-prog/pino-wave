#!/usr/bin/env python3
"""Materialize an evidence-bound saved-time update-density pilot."""
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
    build_update_density_pilot_config,
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
    parser.add_argument("--effective-batch", type=int, default=96)
    parser.add_argument("--pilot-epochs", type=int, default=3)
    parser.add_argument("--physical-microbatch-records", type=int, default=3)
    family_group = parser.add_mutually_exclusive_group()
    family_group.add_argument("--family-curriculum", action="store_true")
    family_group.add_argument("--balanced-families", action="store_true")
    parser.add_argument("--family-gradient-report")
    args = parser.parse_args(argv)

    selection = select_best_pilot_candidate(tuple(args.candidate_config))
    gradient_report_path = (
        None
        if args.family_gradient_report is None
        else Path(args.family_gradient_report).resolve()
    )
    family_gradient_norms = None
    if gradient_report_path is not None:
        gradient_evidence = json.loads(gradient_report_path.read_text())
        if gradient_evidence.get("schema") != "saved_time_family_gradient_conflict_v1":
            raise ValueError("family gradient report schema is invalid")
        family_gradient_norms = gradient_evidence.get("gradient_report", {}).get(
            "gradient_norm"
        )
    config = build_update_density_pilot_config(
        selection,
        artifact_dir=args.artifact_dir,
        effective_batch=args.effective_batch,
        pilot_epochs=args.pilot_epochs,
        physical_microbatch_records=args.physical_microbatch_records,
        family_curriculum=args.family_curriculum,
        balanced_families=args.balanced_families,
        family_gradient_norms=family_gradient_norms,
    )
    output_path = Path(args.output_config).resolve()
    report_path = Path(args.report).resolve()
    _atomic_text(output_path, yaml.safe_dump(config, sort_keys=False))

    updates_per_epoch = math.ceil(
        2240
        / (int(config["macro_records"]) * int(config["macros_per_update"]))
    )
    report = {
        "schema": "saved_time_update_density_parent_selection_v1",
        "name": selection.name,
        "config_path": str(selection.config_path),
        "epoch": int(selection.epoch),
        "score": float(selection.score),
        "parent_metrics": selection.metrics,
        "parent_loss_components": selection.loss_components,
        "checkpoint": str(selection.checkpoint),
        "checkpoint_identity": str(selection.checkpoint_identity),
        "generated_config": str(output_path),
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "effective_batch": int(args.effective_batch),
        "physical_microbatch_records": int(config["microbatch_records"]),
        "pilot_epochs": int(args.pilot_epochs),
        "updates_per_epoch": int(updates_per_epoch),
        "expected_optimizer_updates": int(updates_per_epoch * args.pilot_epochs),
        "intervention": (
            "family_curriculum"
            if args.family_curriculum
            else "balanced_families"
            if args.balanced_families
            else "update_density"
        ),
        "family_gradient_report": (
            None if gradient_report_path is None else str(gradient_report_path)
        ),
        "family_gradient_weights": config.get("family_gradient_weights"),
    }
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
