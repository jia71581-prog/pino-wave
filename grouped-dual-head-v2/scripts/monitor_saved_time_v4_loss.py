#!/usr/bin/env python
"""Locally analyze each newly written V4 epoch without chat-side polling."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    output = args.artifact_dir / "loss_analysis.jsonl"
    seen: set[tuple[str, int]] = set()
    history: dict[str, list[float]] = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line); seen.add((str(row["variant"]), int(row["epoch"])))
            history.setdefault(str(row["variant"]), []).append(float(row["validation_relative_l2"]))
    idle_after_exit = 0
    while True:
        wrote = False
        for path in sorted((args.artifact_dir / "variants").glob("*/metrics.jsonl")):
            variant = path.parent.name
            for line in path.read_text().splitlines():
                report = json.loads(line); key = (variant, int(report["epoch"]))
                if key in seen:
                    continue
                metrics = report["metrics"]
                validation = float(metrics["aggregate_floored_relative_l2"])
                train = float(metrics["train_loss"])
                values = history.setdefault(variant, [])
                previous = values[-1] if values else None
                window = (*values[-4:], validation)
                slope = 0.0 if len(window) < 2 else (window[-1] - window[0]) / (len(window) - 1)
                delta = None if previous is None else validation - previous
                if not math.isfinite(train) or not math.isfinite(validation):
                    state = "nonfinite"
                elif delta is None:
                    state = "initial"
                elif delta < -1.0e-4:
                    state = "improving"
                elif delta > 1.0e-4:
                    state = "regressing"
                else:
                    state = "plateau"
                row = {
                    "variant": variant, "epoch": key[1], "global_step": int(report["global_step"]),
                    "train_loss": train, "validation_relative_l2": validation,
                    "delta_from_previous": delta, "best_so_far": min((*values, validation)),
                    "rolling_5_epoch_slope": slope, "state": state,
                }
                with output.open("a", encoding="utf8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
                values.append(validation); seen.add(key); wrote = True
        if _alive(args.training_pid):
            idle_after_exit = 0
        else:
            idle_after_exit = 0 if wrote else idle_after_exit + 1
            if idle_after_exit >= 2:
                break
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
