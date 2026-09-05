#!/usr/bin/env python3
"""Run CPADC on a train-only audit set disjoint from all checkpoint provenance.

The production evaluator intentionally exposes only registered calibration,
validation, and test selections.  This wrapper supplies a deterministic audit
manifest after excluding basis fitting, risk calibration, and any development
set recorded by a family-selective checkpoint.  Prediction and evaluation are
still delegated unchanged to ``run_causal_defect_adaptation``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from scripts import run_causal_defect_adaptation as evaluator
from scripts.train_v5_feature_meta import build_balanced_meta_episodes
from scripts.train_v5_residual_meta import _sha256


def _provenance_exclusions(payload: dict[str, object]) -> set[str]:
    calibration = dict(payload.get("risk_calibration") or {})
    excluded = {
        str(value) for value in calibration.get("development_sample_ids", [])
    }
    parent = dict(calibration.get("parent_risk_calibration") or {})
    excluded.update(str(value) for value in calibration.get("sample_ids", []))
    excluded.update(str(value) for value in parent.get("sample_ids", []))
    excluded.update(
        str(value)
        for value in calibration.get("provenance_exclusion_sample_ids", [])
    )
    return excluded


def run_audit(
    config_path: str | Path,
    *,
    basis_checkpoint: str | Path,
    output_dir: str | Path,
    audit_per_family: int,
    audit_seed: int,
    shard_index: int,
    shard_count: int,
    device_name: str,
) -> list[dict[str, object]]:
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("CPADC config must contain a mapping")
    checkpoint = Path(basis_checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "causal_physics_aligned_defect_correction_v1":
        raise ValueError("basis checkpoint type mismatch")
    manifest = build_manifest(config["source_h5"])
    basis_per_family = int((config.get("cpadc", {}) or {}).get("per_family", 0))
    basis_seed = int(config.get("seed", 372))
    basis_rows = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=basis_per_family,
        seed=basis_seed,
    )
    excluded = {row.sample_id for row in basis_rows}
    excluded.update(_provenance_exclusions(payload))
    registered_train_ids = {
        row.sample_id for row in manifest.records if row.split == "train"
    }
    if not excluded <= registered_train_ids:
        raise ValueError("checkpoint provenance includes non-train samples")
    allowed = registered_train_ids - excluded
    audit_rows = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=int(audit_per_family),
        seed=int(audit_seed),
        allowed_sample_ids=allowed,
    )
    audit_ids = {row.sample_id for row in audit_rows}
    if audit_ids & excluded:
        raise RuntimeError("audit selection overlaps checkpoint provenance")
    if len(audit_ids) != len(audit_rows):
        raise RuntimeError("audit selection contains duplicate samples")
    if {
        family: sum(row.medium_type == family for row in audit_rows)
        for family in ALLOWED_MEDIUM_TYPES
    } != {family: int(audit_per_family) for family in ALLOWED_MEDIUM_TYPES}:
        raise RuntimeError("audit selection is not family balanced")

    canonical = json.dumps(
        {
            "checkpoint_sha256": _sha256(checkpoint),
            "manifest_digest": manifest.digest,
            "basis_seed": basis_seed,
            "basis_per_family": basis_per_family,
            "excluded_sample_ids": sorted(excluded),
            "audit_seed": int(audit_seed),
            "audit_per_family": int(audit_per_family),
            "audit_sample_ids": [row.sample_id for row in audit_rows],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    selection_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    original_selector = evaluator.build_risk_calibration_manifest

    def fixed_selector(
        selected_manifest,
        *,
        basis_per_family: int,
        basis_seed: int,
        calibration_per_family: int,
        calibration_seed: int,
    ):
        if selected_manifest.digest != manifest.digest:
            raise ValueError("audit evaluator manifest changed")
        if int(basis_per_family) != int((config.get("cpadc", {}) or {}).get("per_family", 0)):
            raise ValueError("audit evaluator basis count changed")
        if int(calibration_per_family) != int(audit_per_family):
            raise ValueError("audit evaluator count changed")
        if int(calibration_seed) != int(audit_seed):
            raise ValueError("audit evaluator seed changed")
        return audit_rows

    evaluator.build_risk_calibration_manifest = fixed_selector
    try:
        reports = evaluator.run(
            config_path,
            basis_checkpoint=checkpoint,
            output_dir=output_dir,
            device_name=device_name,
            adaptation_device_name="cpu",
            shard_index=int(shard_index),
            shard_count=int(shard_count),
            calibration_per_family=int(audit_per_family),
            calibration_seed=int(audit_seed),
            basis_training_per_family=basis_per_family,
            save_fields=False,
        )
    finally:
        evaluator.build_risk_calibration_manifest = original_selector

    output = Path(output_dir).expanduser().resolve()
    selected_for_shard = audit_rows[int(shard_index) :: int(shard_count)]
    contract = {
        "status": "complete",
        "schema": "cpadc_disjoint_train_audit_selection_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest_digest": manifest.digest,
        "future_truth_scope": "fresh_disjoint_train_split_only",
        "online_future_truth_used": False,
        "basis_record_count": len(basis_rows),
        "provenance_exclusion_count": len(excluded),
        "audit_seed": int(audit_seed),
        "audit_per_family": int(audit_per_family),
        "audit_record_count": len(audit_rows),
        "audit_sample_ids": [row.sample_id for row in audit_rows],
        "selection_digest": selection_digest,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "shard_record_count": len(selected_for_shard),
        "evaluated_report_count": len(reports),
        "overlap_with_checkpoint_provenance": 0,
    }
    (output / "audit_selection.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--basis-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-per-family", type=int, default=32)
    parser.add_argument("--audit-seed", type=int, default=42315)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run_audit(
        args.config,
        basis_checkpoint=args.basis_checkpoint,
        output_dir=args.output_dir,
        audit_per_family=args.audit_per_family,
        audit_seed=args.audit_seed,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
