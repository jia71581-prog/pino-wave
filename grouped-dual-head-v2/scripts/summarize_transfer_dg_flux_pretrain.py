#!/usr/bin/env python3
"""Summarize the paired Transfer DG interface-flux pretraining pilot."""
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
    grouped = {
        arm: sorted(
            [row for row in terminals if row["arm"] == arm], key=lambda row: row["seed"]
        )
        for arm in ("dg_flux", "control")
    }
    if len(terminals) != 4 or any(
        [row["seed"] for row in grouped[arm]] != [372, 733] for arm in grouped
    ):
        raise RuntimeError("paired seeds 372 and 733 are required for both arms")
    dg = np.asarray([row["best_score"] for row in grouped["dg_flux"]])
    control = np.asarray([row["best_score"] for row in grouped["control"]])
    parent = np.asarray([row["parent_score"] for row in grouped["dg_flux"]])
    accepted = bool(dg.mean() < control.mean() and dg.mean() < parent.mean())
    return {
        "decision": "accepted_train_only" if accepted else "rejected",
        "gate": "DG-flux two-seed mean must be strictly below paired control and hard-boundary parent means; no minimum percentage",
        "dg_flux_mean_best_score": float(dg.mean()),
        "control_mean_best_score": float(control.mean()),
        "parent_mean_score": float(parent.mean()),
        "relative_gain_vs_control": float((control.mean() - dg.mean()) / control.mean()),
        "relative_gain_vs_parent": float((parent.mean() - dg.mean()) / parent.mean()),
        "paired_control_minus_dg_flux": {
            str(seed): float(value)
            for seed, value in zip((372, 733), control - dg)
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
        raise RuntimeError("cannot summarize a failed lane")
    payload = {
        "schema": "transfer_dg_flux_pretrain_summary_v1",
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
