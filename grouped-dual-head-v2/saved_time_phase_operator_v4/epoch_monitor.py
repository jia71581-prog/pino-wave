"""Small, deterministic summaries for detached saved-time training runs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence


FAMILIES = ("uniform", "layered", "marmousi")


def _validated_epochs(rows: Sequence[Mapping[str, object]], *, name: str):
    epochs = [row for row in rows if row.get("event") == "epoch"]
    for row in epochs:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{name} epoch metrics are missing")
        family = metrics.get("family_relative_l2")
        if not isinstance(family, Mapping) or set(family) != set(FAMILIES):
            raise ValueError(f"{name} epoch metrics must contain all three families")
        source = metrics.get("source_relative_l2")
        if not isinstance(source, Mapping) or not source:
            raise ValueError(f"{name} epoch metrics must contain source panel identity")
        values = [
            metrics.get("aggregate_relative_l2"),
            *(family[key] for key in FAMILIES),
            *source.values(),
        ]
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"{name} epoch metrics must be finite")
    return epochs


def summarize_epoch_metrics(
    candidate_rows: Sequence[Mapping[str, object]],
    parent_rows: Sequence[Mapping[str, object]],
    *,
    target_aggregate: float = 0.10,
    target_family: float = 0.12,
    family_regression_tolerance: float = 0.03,
) -> dict[str, object]:
    """Compare completed candidate epochs with the best registered parent epoch."""

    candidate = _validated_epochs(candidate_rows, name="candidate")
    if not candidate:
        return {"status": "waiting_for_epoch"}
    parent = _validated_epochs(parent_rows, name="parent")
    if not parent:
        return {
            "status": "waiting_for_parent_panel",
            "completed_epochs": len(candidate),
        }
    best_candidate = min(
        candidate, key=lambda row: float(row["metrics"]["aggregate_relative_l2"])
    )
    best_parent = min(
        parent, key=lambda row: float(row["metrics"]["aggregate_relative_l2"])
    )
    latest = max(candidate, key=lambda row: int(row["epoch"]))
    candidate_metrics = best_candidate["metrics"]
    parent_metrics = best_parent["metrics"]
    same_validation_panel = tuple(
        sorted(str(value) for value in candidate_metrics["source_relative_l2"])
    ) == tuple(sorted(str(value) for value in parent_metrics["source_relative_l2"]))
    aggregate = float(candidate_metrics["aggregate_relative_l2"])
    parent_aggregate = float(parent_metrics["aggregate_relative_l2"])
    family = {
        key: float(candidate_metrics["family_relative_l2"][key]) for key in FAMILIES
    }
    parent_family = {
        key: float(parent_metrics["family_relative_l2"][key]) for key in FAMILIES
    }
    family_ok = all(
        family[key] <= parent_family[key] + float(family_regression_tolerance)
        for key in FAMILIES
    )
    completed = len(candidate)
    latest_elapsed = float(latest.get("elapsed_seconds", 0.0))
    return {
        "status": "epochs_available",
        "completed_epochs": completed,
        "latest_epoch": int(latest["epoch"]),
        "latest_train_loss": float(latest.get("train_loss", math.nan)),
        "best_epoch": int(best_candidate["epoch"]),
        "best_checkpoint": str(best_candidate.get("checkpoint", "")),
        "best_aggregate_relative_l2": aggregate,
        "best_family_relative_l2": family,
        "parent_aggregate_relative_l2": parent_aggregate,
        "parent_family_relative_l2": parent_family,
        "relative_improvement_vs_parent": (parent_aggregate - aggregate)
        / parent_aggregate,
        "same_validation_panel": same_validation_panel,
        "family_regression_ok": family_ok,
        "pilot_promotion_eligible": same_validation_panel
        and aggregate < parent_aggregate
        and family_ok,
        "final_target_met": aggregate < float(target_aggregate)
        and all(value < float(target_family) for value in family.values()),
        "seconds_per_completed_epoch": latest_elapsed / max(int(latest["epoch"]), 1),
    }


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_parent_rows(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        return []
    text = path.read_text()
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return _read_jsonl(path)
    if (
        isinstance(document, Mapping)
        and document.get("schema") == "saved_time_family_expert_same_panel_parent_v1"
        and document.get("status") == "complete"
    ):
        return [
            {
                "event": "epoch",
                "epoch": 0,
                "metrics": document.get("metrics"),
            }
        ]
    return [document] if isinstance(document, Mapping) else []


def write_epoch_summary(
    candidate_metrics: str | Path,
    parent_metrics: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Write one atomic machine-readable monitor snapshot."""

    candidate_path = Path(candidate_metrics)
    parent_path = Path(parent_metrics)
    output_path = Path(output)
    report = summarize_epoch_metrics(
        _read_jsonl(candidate_path), _read_parent_rows(parent_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--parent-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = write_epoch_summary(
        args.candidate_metrics, args.parent_report, args.output
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["summarize_epoch_metrics", "write_epoch_summary"]
