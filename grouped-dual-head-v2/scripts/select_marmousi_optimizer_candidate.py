#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def select_candidate(
    rows: list[dict[str, Any]],
    threshold: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    finite = [
        row
        for row in rows
        if math.isfinite(float(row["validation_relative_l2"]))
    ]
    eligible = [
        row
        for row in finite
        if float(row["validation_relative_l2"]) < float(threshold)
    ]
    if not eligible:
        return {
            "accepted": False,
            "selected": None,
            "candidates": rows,
            "threshold": float(threshold),
            "tie_tolerance": float(tie_tolerance),
        }

    best_metric = min(float(row["validation_relative_l2"]) for row in eligible)
    tied = [
        row
        for row in eligible
        if float(row["validation_relative_l2"]) <= best_metric + float(tie_tolerance)
    ]
    selected = min(tied, key=lambda row: float(row["learning_rate"]))
    return {
        "accepted": True,
        "selected": selected,
        "candidates": rows,
        "threshold": float(threshold),
        "tie_tolerance": float(tie_tolerance),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a Marmousi optimizer candidate from sealed validation metrics."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("TAG", "LEARNING_RATE", "METRICS_JSON"),
        required=True,
    )
    parser.add_argument("--threshold", type=float, default=0.1045)
    parser.add_argument("--tie-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for tag, learning_rate, metrics_path_raw in args.candidate:
        metrics_path = Path(metrics_path_raw)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "tag": tag,
                "learning_rate": float(learning_rate),
                "validation_relative_l2": float(payload["validation_relative_l2"]),
                "metrics_path": str(metrics_path),
                "checkpoint_best": payload.get("checkpoint_best"),
            }
        )

    result = select_candidate(rows, args.threshold, args.tie_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
