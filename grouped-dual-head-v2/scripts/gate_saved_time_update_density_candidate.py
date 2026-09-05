#!/usr/bin/env python3
"""Gate an update-density pilot against its exact corrected parent."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.continuation import update_density_candidate_gate


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _best_epoch(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    epochs = [
        row
        for row in rows
        if row.get("event") == "epoch"
        and row.get("validation_scope") == "pilot_fixed_panel"
    ]
    if not epochs:
        raise ValueError("update-density metrics contain no fixed-panel epoch")
    return min(
        epochs,
        key=lambda row: (
            float(row["metrics"]["aggregate_relative_l2"]),
            int(row["epoch"]),
        ),
    )


def _completed_optimizer_updates(path: Path) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    epochs = [
        row
        for row in rows
        if row.get("event") == "epoch"
        and row.get("validation_scope") == "pilot_fixed_panel"
    ]
    if not epochs:
        raise ValueError("update-density metrics contain no fixed-panel epoch")
    try:
        return max(int(row["global_step"]) for row in epochs)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("update-density epoch is missing optimizer progress") from error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    selection_path = Path(args.selection_report)
    selection = json.loads(selection_path.read_text())
    config = yaml.safe_load(Path(selection["generated_config"]).read_text())
    candidate_path = Path(args.candidate_metrics)
    candidate = _best_epoch(candidate_path)
    gate = config["gate"]
    report = update_density_candidate_gate(
        parent_metrics=selection["parent_metrics"],
        parent_loss_components=selection["parent_loss_components"],
        candidate_row=candidate,
        family_tolerance=float(gate["family_regression_tolerance"]),
        maximum_peak_cuda_gib=float(gate["maximum_peak_cuda_gib"]),
        expected_global_macros_per_update=int(config["macros_per_update"]),
        expected_physical_microbatch=int(config["microbatch_records"]),
        minimum_optimizer_updates=int(selection["expected_optimizer_updates"]),
        completed_optimizer_updates=_completed_optimizer_updates(candidate_path),
        minimum_relative_improvement=float(
            gate.get("minimum_relative_improvement", 1.0e-3)
        ),
    )
    report.update(
        {
            "schema": "saved_time_update_density_evidence_gate_v1",
            "selection_report": str(selection_path.resolve()),
            "candidate_metrics": str(candidate_path.resolve()),
            "candidate_epoch": int(candidate["epoch"]),
            "candidate_checkpoint": str(candidate["checkpoint"]),
            "intervention": selection.get("intervention", "update_density"),
        }
    )
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passes"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
