#!/usr/bin/env python3
"""Summarize four local passive-DtN training lanes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


FAMILIES = ("uniform", "layered", "marmousi")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def summarize(rows: list[dict]) -> dict:
    if len(rows) != 4 or sorted(row["seed"] for row in rows) != [372, 733, 1049, 1403]:
        raise RuntimeError("four frozen local-DtN seeds are required")
    aggregate = float(np.mean([row["best_aggregate"] for row in rows]))
    baseline = float(np.mean([row["zero_flux_baseline"] for row in rows]))
    per_family = {
        family: float(
            np.mean([row["best_metrics"]["per_family"][family] for row in rows])
        )
        for family in FAMILIES
    }
    accepted = bool(
        aggregate < baseline
        and all(row["best_metrics"]["dissipation_minimum"] >= -1.0e-6 for row in rows)
    )
    return {
        "decision": "accepted_local_operator_pilot" if accepted else "rejected",
        "mean_best_aggregate": aggregate,
        "mean_zero_flux_baseline": baseline,
        "relative_gain": (baseline - aggregate) / max(baseline, 1.0e-16),
        "mean_per_family": per_family,
        "all_passive": all(
            row["best_metrics"]["dissipation_minimum"] >= -1.0e-6 for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.terminal) != 4 or args.output.exists():
        raise RuntimeError("four lane terminals and a fresh output are required")
    rows = [json.loads(path.read_text()) for path in args.terminal]
    if any(row.get("status") == "failed" for row in rows):
        raise RuntimeError("cannot summarize failed local DtN lanes")
    payload = {
        "schema": "transfer_dg_local_dtn_summary_v1",
        "status": "complete",
        **summarize(rows),
        "lane_terminals": {
            str(path): {"sha256": _sha256(path), **row}
            for path, row in zip(args.terminal, rows)
        },
        "claim_scope": "train_only local trace-to-flux capacity; not wavefield promotion",
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
