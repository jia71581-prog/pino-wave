#!/usr/bin/env python3
"""Compare four HCAIS local-DtN lanes with the frozen uniform baseline."""
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


def summarize(hcais: list[dict], baseline: list[dict]) -> dict:
    expected = [372, 733, 1049, 1403]
    if sorted(row["seed"] for row in hcais) != expected or sorted(
        row["seed"] for row in baseline
    ) != expected:
        raise RuntimeError("paired four-seed summaries are required")
    hcais = sorted(hcais, key=lambda row: row["seed"])
    baseline = sorted(baseline, key=lambda row: row["seed"])
    adaptive_mean = float(np.mean([row["best_aggregate"] for row in hcais]))
    baseline_mean = float(np.mean([row["best_aggregate"] for row in baseline]))
    ess_pass = all(
        row["sampler"]["minimum_epoch_ess_p5_fraction"] >= 0.50 for row in hcais
    )
    overhead_pass = all(row["sampler"]["sampling_overhead_fraction"] <= 0.10 for row in hcais)
    accepted = bool(adaptive_mean < baseline_mean and ess_pass and overhead_pass)
    return {
        "decision": "accepted_sampling_pilot" if accepted else "rejected",
        "hcais_mean_best_aggregate": adaptive_mean,
        "uniform_mean_best_aggregate": baseline_mean,
        "relative_gain": (baseline_mean - adaptive_mean) / max(baseline_mean, 1.0e-16),
        "paired_uniform_minus_hcais": {
            str(left["seed"]): left["best_aggregate"] - right["best_aggregate"]
            for left, right in zip(baseline, hcais)
        },
        "mean_per_family": {
            arm: {
                family: float(
                    np.mean(
                        [row["best_metrics"]["per_family"][family] for row in values]
                    )
                )
                for family in FAMILIES
            }
            for arm, values in (("hcais", hcais), ("uniform", baseline))
        },
        "ess_gate_pass": ess_pass,
        "overhead_gate_pass": overhead_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hcais-terminal", type=Path, action="append", required=True)
    parser.add_argument("--baseline-terminal", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.hcais_terminal) != 4 or len(args.baseline_terminal) != 4 or args.output.exists():
        raise RuntimeError("four paired terminals per arm and a fresh output are required")
    hcais = [json.loads(path.read_text()) for path in args.hcais_terminal]
    baseline = [json.loads(path.read_text()) for path in args.baseline_terminal]
    if any(row.get("status") == "failed" for row in hcais + baseline):
        raise RuntimeError("cannot compare failed lanes")
    payload = {
        "schema": "transfer_dg_local_dtn_hcais_comparison_v1",
        "status": "complete",
        **summarize(hcais, baseline),
        "hcais_terminals": {
            str(path): _sha256(path) for path in args.hcais_terminal
        },
        "baseline_terminals": {
            str(path): _sha256(path) for path in args.baseline_terminal
        },
        "claim_scope": "train-only local flux sampling efficiency; not full-wavefield promotion",
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
