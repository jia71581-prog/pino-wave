#!/usr/bin/env python3
"""Summarize the paired two-seed Transfer DG A1 train-only experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def summarize(terminals: list[dict]) -> dict:
    if len(terminals) != 4:
        raise ValueError("A1 summary requires four lane terminals")
    grouped = {
        arm: sorted(
            [row for row in terminals if row["arm"] == arm], key=lambda row: row["seed"]
        )
        for arm in ("halo", "control")
    }
    if any([row["seed"] for row in grouped[arm]] != [372, 733] for arm in grouped):
        raise RuntimeError("A1 requires paired seeds 372 and 733")
    halo = np.asarray([row["best_score"] for row in grouped["halo"]], dtype=np.float64)
    control = np.asarray(
        [row["best_score"] for row in grouped["control"]], dtype=np.float64
    )
    parent = np.asarray(
        [row["parent_score"] for row in grouped["halo"]], dtype=np.float64
    )
    paired = control - halo
    accepted = bool(halo.mean() < control.mean() and halo.mean() < parent.mean())
    return {
        "decision": "accepted_train_only" if accepted else "rejected",
        "gate": "halo two-seed mean must be strictly below paired control and P1b hard-boundary parent means; no minimum percentage",
        "halo_mean_best_score": float(halo.mean()),
        "control_mean_best_score": float(control.mean()),
        "parent_mean_score": float(parent.mean()),
        "relative_gain_vs_control": float((control.mean() - halo.mean()) / control.mean()),
        "relative_gain_vs_parent": float((parent.mean() - halo.mean()) / parent.mean()),
        "paired_control_minus_halo": {
            str(seed): float(value) for seed, value in zip((372, 733), paired)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-terminal", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.lane_terminal) != 4 or args.output.exists():
        raise RuntimeError("four lane terminals and a fresh output are required")
    rows = [json.loads(path.read_text()) for path in args.lane_terminal]
    if any(row.get("status") == "failed" for row in rows):
        raise RuntimeError("cannot summarize a failed A1 lane")
    payload = {
        "schema": "transfer_dg_a1_paired_summary_v1",
        "status": "complete",
        **summarize(rows),
        "lane_terminals": {
            str(path): {"sha256": _sha256(path), **row}
            for path, row in zip(args.lane_terminal, rows)
        },
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
