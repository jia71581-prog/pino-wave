#!/usr/bin/env python3
"""Apply the parent-relative V15 evidence gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.continuation import (
    local_differential_candidate_gate,
)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _final_epoch(path: Path) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    epochs = [
        row
        for row in rows
        if row.get("event") == "epoch"
        and row.get("validation_scope") == "pilot_fixed_panel"
    ]
    if not epochs:
        raise ValueError("V15 metrics contain no fixed-panel epoch")
    return max(epochs, key=lambda row: int(row["epoch"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    selection_path = Path(args.selection_report)
    selection = json.loads(selection_path.read_text())
    config = yaml.safe_load(Path(selection["generated_config"]).read_text())
    gate = config["gate"]
    candidate = _final_epoch(Path(args.candidate_metrics))
    report = local_differential_candidate_gate(
        parent_metrics=selection["parent_metrics"],
        parent_loss_components=selection["parent_loss_components"],
        candidate_row=candidate,
        family_tolerance=float(gate["family_regression_tolerance"]),
        maximum_peak_cuda_gib=float(gate["maximum_peak_cuda_gib"]),
    )
    report.update(
        {
            "schema": "saved_time_v15_evidence_gate_v1",
            "selection_report": str(selection_path.resolve()),
            "candidate_metrics": str(Path(args.candidate_metrics).resolve()),
            "candidate_epoch": int(candidate["epoch"]),
            "candidate_checkpoint": str(candidate["checkpoint"]),
        }
    )
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passes"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
