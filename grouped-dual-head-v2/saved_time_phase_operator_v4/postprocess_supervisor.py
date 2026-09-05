"""Choose the next safe action after the V65 pilot and V66 long run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence


FAMILIES = {"uniform", "layered", "marmousi"}


def _sealed_complete(report: object) -> bool:
    if not isinstance(report, dict):
        return False
    metrics = report.get("metrics")
    family = metrics.get("family_relative_l2") if isinstance(metrics, dict) else None
    return bool(
        report.get("status") == "complete"
        and report.get("stored_times_only") is True
        and int(report.get("interpolated_targets", -1)) == 0
        and isinstance(metrics, dict)
        and int(metrics.get("record_count", -1)) == 480
        and int(metrics.get("unique_time_index_count", -1)) == 401
        and isinstance(family, dict)
        and set(family) == FAMILIES
    )


def _accuracy_target_met(report: dict[str, object]) -> bool:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return False
    family = metrics.get("family_relative_l2")
    if not isinstance(family, dict) or set(family) != FAMILIES:
        return False
    try:
        aggregate = float(metrics["aggregate_relative_l2"])
        family_values = tuple(float(family[name]) for name in FAMILIES)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        report.get("passes_accuracy_gate") is True
        and math.isfinite(aggregate)
        and aggregate < 0.10
        and all(math.isfinite(value) and value < 0.12 for value in family_values)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--run-terminal", type=Path, required=True)
    parser.add_argument("--sealed-report", type=Path, required=True)
    parser.add_argument("--figures-report", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.gate.is_file():
        print(
            json.dumps(
                {
                    "action": "wait_gate",
                    "reason": "pilot gate is not available",
                },
                sort_keys=True,
            )
        )
        return 0
    gate = json.loads(args.gate.read_text())
    if gate.get("passes") is not True:
        print(
            json.dumps(
                {
                    "action": "pilot_rejected",
                    "reason": "same-panel pilot gate did not pass",
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.run_terminal.is_file():
        print(
            json.dumps(
                {
                    "action": "wait_run",
                    "reason": "V66 long training is not complete",
                },
                sort_keys=True,
            )
        )
        return 0
    run_terminal = json.loads(args.run_terminal.read_text())
    if run_terminal.get("status") != "complete":
        print(
            json.dumps(
                {
                    "action": "run_failed",
                    "reason": "V66 long training terminal is not complete",
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.sealed_report.is_file():
        print(
            json.dumps(
                {
                    "action": "evaluate_sealed",
                    "reason": "full held-out evaluation is missing",
                },
                sort_keys=True,
            )
        )
        return 0
    sealed_report = json.loads(args.sealed_report.read_text())
    if not _sealed_complete(sealed_report):
        print(
            json.dumps(
                {
                    "action": "evaluate_sealed",
                    "reason": "full held-out evaluation is incomplete",
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.figures_report.is_file():
        print(
            json.dumps(
                {
                    "action": "render_figures",
                    "reason": "three-family visual diagnostics are missing",
                },
                sort_keys=True,
            )
        )
        return 0
    figures_report = json.loads(args.figures_report.read_text())
    figure_families = (
        figures_report.get("families") if isinstance(figures_report, dict) else None
    )
    if not (
        isinstance(figures_report, dict)
        and figures_report.get("status") == "complete"
        and isinstance(figure_families, dict)
        and set(figure_families) == FAMILIES
    ):
        print(
            json.dumps(
                {
                    "action": "render_figures",
                    "reason": "three-family visual diagnostics are incomplete",
                },
                sort_keys=True,
            )
        )
        return 0
    if not _accuracy_target_met(sealed_report):
        print(
            json.dumps(
                {
                    "action": "continue_experiments",
                    "reason": "sealed neural-operator accuracy target is not met",
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "action": "complete",
                "reason": "sealed neural-operator accuracy and visual evidence are complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
