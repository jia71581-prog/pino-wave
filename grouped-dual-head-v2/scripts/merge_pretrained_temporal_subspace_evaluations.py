#!/usr/bin/env python3
"""Merge exact-coverage PTLSA validation shards and apply one promotion gate."""
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
from scripts.run_causal_defect_adaptation import _promotion_gate
from scripts.run_pretrained_temporal_subspace_adaptation import PTLSA_SCHEMA
from scripts.run_v5_instance_adaptation import _write_manifest
from scripts.train_v5_residual_meta import _sha256


def merge(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    shard_dirs: tuple[str | Path, ...],
    evaluation_split: str = "validation",
) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("PTLSA config must contain a mapping")
    manifest = build_manifest(config["source_h5"])
    normalized_split = str(evaluation_split)
    if normalized_split not in {"validation", "test_id"}:
        raise ValueError("PTLSA evaluation split must be validation or test_id")
    expected_rows = tuple(row for row in manifest.records if row.split == normalized_split)
    if not expected_rows or {row.medium_type for row in expected_rows} != set(
        ALLOWED_MEDIUM_TYPES
    ):
        raise ValueError("complete evaluation manifest is invalid")
    expected_ids = tuple(row.sample_id for row in expected_rows)
    expected_set = set(expected_ids)

    normalized_dirs = tuple(Path(value).expanduser().resolve() for value in shard_dirs)
    if not normalized_dirs:
        raise ValueError("at least one PTLSA shard directory is required")
    reports_by_id: dict[str, dict[str, object]] = {}
    artifact_hashes: dict[str, str] = {}
    parent_identity: dict[str, object] | None = None
    method_identity: dict[str, object] | None = None
    shard_indices: list[int] = []
    for shard_dir in normalized_dirs:
        summary_path = shard_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if summary.get("schema") != PTLSA_SCHEMA:
            raise ValueError(f"PTLSA schema mismatch in {summary_path}")
        if summary.get("selection_split") != normalized_split:
            raise ValueError(f"evaluation split mismatch in {summary_path}")
        shard_parent = dict(summary["parent"])
        shard_method = dict(summary["method"])
        if parent_identity is None:
            parent_identity = shard_parent
            method_identity = shard_method
        elif shard_parent != parent_identity or shard_method != method_identity:
            raise ValueError(f"parent or method identity mismatch in {summary_path}")
        metadata = dict(summary.get("evaluation_shard") or {})
        if (
            int(metadata.get("count", 0)) != len(normalized_dirs)
            or not bool(metadata.get("complete_evaluation_requested", False))
            or metadata.get("evaluation_split") != normalized_split
        ):
            raise ValueError(f"invalid shard contract in {summary_path}")
        shard_indices.append(int(metadata.get("index", -1)))
        for raw_report in summary["records"]:
            report = dict(raw_report)
            sample_id = str(report["sample_id"])
            if sample_id in reports_by_id:
                raise ValueError(f"duplicate sample across shards: {sample_id}")
            if sample_id not in expected_set:
                raise ValueError(f"unexpected evaluation sample: {sample_id}")
            adaptation = dict(report["adaptation"])
            if bool(adaptation.get("future_truth_used", True)):
                raise ValueError(f"future truth was used online for {sample_id}")
            if dict(adaptation.get("parent") or {}) != parent_identity:
                raise ValueError(f"parent identity drift for {sample_id}")
            if tuple(adaptation.get("apply_families") or ()) != tuple(
                method_identity.get("apply_families") or ()
            ):
                raise ValueError(f"family gate drift for {sample_id}")
            artifact = shard_dir / sample_id / "adaptation.pt"
            evaluation = shard_dir / sample_id / "evaluation.json"
            if not artifact.is_file() or not evaluation.is_file():
                raise FileNotFoundError(f"unsealed or incomplete record: {sample_id}")
            artifact_hashes[sample_id] = _sha256(artifact)
            reports_by_id[sample_id] = report

    if sorted(shard_indices) != list(range(len(normalized_dirs))):
        raise ValueError(f"shard indices are incomplete: {sorted(shard_indices)}")
    if set(reports_by_id) != expected_set:
        missing = sorted(expected_set - set(reports_by_id))
        extra = sorted(set(reports_by_id) - expected_set)
        raise ValueError(
            f"full evaluation coverage mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )

    reports = [reports_by_id[sample_id] for sample_id in expected_ids]
    promotion = _promotion_gate(
        reports,
        dict(config.get("deployment_gate", {}) or {}),
        all_validation=True,
    )
    output = Path(output_dir).expanduser().resolve()
    if (output / "summary.json").exists() or (output / "terminal.json").exists():
        raise FileExistsError("refusing to overwrite an existing PTLSA merge")
    output.mkdir(parents=True, exist_ok=True)
    _write_manifest(expected_rows, output / "instance_manifest.json")
    summary = {
        "schema": PTLSA_SCHEMA,
        "records": reports,
        "family_count": {
            family: sum(item["medium_type"] == family for item in reports)
            for family in ALLOWED_MEDIUM_TYPES
        },
        "parent": parent_identity,
        "method": method_identity,
        "artifact_sha256": artifact_hashes,
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
        "schema": PTLSA_SCHEMA,
        "same_protocol_validation_passed": bool(promotion["passed"]),
        "same_protocol_evaluation_passed": bool(promotion["passed"]),
        "evaluation_split": normalized_split,
        "promotion_gate": promotion,
        "parent": parent_identity,
        "method": method_identity,
        "claim": (
            "same-protocol PTLSA gate passed"
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
