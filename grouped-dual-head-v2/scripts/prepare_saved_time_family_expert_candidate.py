#!/usr/bin/env python3
"""Materialize an identity-bound velocity-routed family-expert pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.continuation import (
    build_family_expert_pilot_config,
    select_best_pilot_candidate,
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(value)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-config", action="append", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--family-expert-rank", type=int, default=16)
    parser.add_argument("--physical-microbatch-records", type=int, default=3)
    args = parser.parse_args(argv)

    selection = select_best_pilot_candidate(tuple(args.candidate_config))
    config = build_family_expert_pilot_config(
        selection,
        artifact_dir=args.artifact_dir,
        physical_microbatch_records=args.physical_microbatch_records,
        family_expert_rank=args.family_expert_rank,
    )
    output_path = Path(args.output_config).resolve()
    report_path = Path(args.report).resolve()
    _atomic_text(output_path, yaml.safe_dump(config, sort_keys=False))
    evidence = {
        "schema": "saved_time_family_expert_parent_selection_v1",
        "name": selection.name,
        "config_path": str(selection.config_path),
        "config_sha256": _sha256(selection.config_path),
        "epoch": int(selection.epoch),
        "score": float(selection.score),
        "parent_metrics": selection.metrics,
        "parent_loss_components": selection.loss_components,
        "checkpoint": str(selection.checkpoint),
        "checkpoint_sha256": _sha256(selection.checkpoint),
        "checkpoint_identity": str(selection.checkpoint_identity),
        "checkpoint_identity_sha256": _sha256(selection.checkpoint_identity),
        "generated_config": str(output_path),
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "family_expert_rank": int(config["variant_overrides"]["family_expert_rank"]),
        "effective_batch": int(config["macro_records"])
        * int(config["macros_per_update"]),
        "physical_microbatch_records": int(config["microbatch_records"]),
        "pilot_epochs": int(config["gate"]["pilot_epochs"]),
    }
    _atomic_text(report_path, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
