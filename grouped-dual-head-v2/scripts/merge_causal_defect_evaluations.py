#!/usr/bin/env python3
"""Merge mutually exclusive sealed CPADC validation shards and apply one gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from scripts.evaluate_v5_instance_adaptation import write_report
from scripts.run_causal_defect_adaptation import (
    _promotion_gate,
    build_complete_evaluation_manifest,
)
from scripts.run_v5_instance_adaptation import (
    _write_manifest,
)


def merge(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    shard_dirs: tuple[str | Path, ...],
    evaluation_split: str = "validation",
) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("CPADC config must contain a mapping")
    manifest = build_manifest(config["source_h5"])
    normalized_split = str(evaluation_split)
    expected_rows = build_complete_evaluation_manifest(
        manifest, split=normalized_split
    )
    expected_ids = tuple(row.sample_id for row in expected_rows)
    expected_set = set(expected_ids)

    reports_by_id: dict[str, dict[str, object]] = {}
    basis: dict[str, object] | None = None
    shard_metadata: list[dict[str, object]] = []
    normalized_dirs = tuple(Path(value).expanduser().resolve() for value in shard_dirs)
    if not normalized_dirs:
        raise ValueError("at least one shard directory is required")
    for shard_dir in normalized_dirs:
        summary_path = shard_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if str(summary.get("selection_split", "validation")) != normalized_split:
            raise ValueError(f"evaluation split mismatch in {summary_path}")
        shard_basis = dict(summary["basis"])
        if basis is None:
            basis = shard_basis
        elif shard_basis != basis:
            raise ValueError(f"basis identity mismatch in {summary_path}")
        metadata = dict(
            summary.get("evaluation_shard")
            or summary.get("validation_shard", {})
        )
        if int(metadata.get("count", 1)) != len(normalized_dirs):
            raise ValueError(f"shard count mismatch in {summary_path}")
        shard_metadata.append(metadata)
        for report_raw in summary["records"]:
            report = dict(report_raw)
            sample_id = str(report["sample_id"])
            if sample_id in reports_by_id:
                raise ValueError(f"duplicate sample across shards: {sample_id}")
            if sample_id not in expected_set:
                raise ValueError(f"unexpected validation sample: {sample_id}")
            adaptation = dict(report["adaptation"])
            if bool(adaptation.get("future_truth_used", True)):
                raise ValueError(f"future truth was used online for {sample_id}")
            artifact = shard_dir / sample_id / "adaptation.pt"
            evaluation = shard_dir / sample_id / "evaluation.json"
            if not artifact.is_file() or not evaluation.is_file():
                raise FileNotFoundError(f"unsealed or incomplete record: {sample_id}")
            reports_by_id[sample_id] = report

    shard_indices = sorted(int(value.get("index", -1)) for value in shard_metadata)
    if shard_indices != list(range(len(normalized_dirs))):
        raise ValueError(f"shard indices are incomplete: {shard_indices}")
    actual_set = set(reports_by_id)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"full validation coverage mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )

    reports = [reports_by_id[sample_id] for sample_id in expected_ids]
    gate = dict(config.get("deployment_gate", {}) or {})
    promotion = _promotion_gate(reports, gate, all_validation=True)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_manifest(expected_rows, output / "instance_manifest.json")
    summary = {
        "records": reports,
        "family_count": {
            family: sum(item["medium_type"] == family for item in reports)
            for family in ALLOWED_MEDIUM_TYPES
        },
        "basis": basis,
        "future_truth_opened_only_after_seal": True,
        "parallel_validation": {
            "shard_count": len(normalized_dirs),
            "shard_dirs": [str(value) for value in normalized_dirs],
            "exact_manifest_coverage": True,
        },
        "evaluation_split": normalized_split,
        "promotion_gate": promotion,
    }
    write_report(summary, output / "summary.json")
    terminal = {
        "status": "complete",
        "schema": "cpadc_same_protocol_terminal_v1",
        "same_protocol_validation_passed": bool(promotion["passed"]),
        "same_protocol_evaluation_passed": bool(promotion["passed"]),
        "evaluation_split": normalized_split,
        "promotion_gate": promotion,
        "basis": basis,
        "claim": (
            "same-protocol CPADC gate passed"
            if promotion["passed"]
            else "evaluation complete; accuracy promotion not authorized"
        ),
    }
    write_report(terminal, output / "terminal.json")
    return terminal


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument(
        "--evaluation-split", choices=("validation", "test_id"), default="validation"
    )
    args = parser.parse_args(argv)
    merge(
        args.config,
        output_dir=args.output_dir,
        shard_dirs=tuple(args.shard_dir),
        evaluation_split=args.evaluation_split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
