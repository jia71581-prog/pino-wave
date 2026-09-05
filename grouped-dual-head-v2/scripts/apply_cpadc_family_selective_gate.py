#!/usr/bin/env python3
"""Apply a train-only family-selective abstention gate to a CPADC checkpoint.

The source checkpoint already contains family-specific lower strength floors.
This tool can disable an unsafe family by replacing its floor with a finite,
unreachable value.  It never changes the learned basis or opens validation or
test data.  The development reports used to choose the enabled families must
be train-only and disjoint from both basis fitting and risk calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.cpadc_contract import (
    cpadc_implementation_digests,
)
from scripts.train_v5_feature_meta import build_balanced_meta_episodes
from scripts.train_v5_residual_meta import _sha256


DISABLED_STRENGTH_FLOOR = 1.0e30


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_development_reports(
    shard_dirs: tuple[Path, ...],
    *,
    expected_checkpoint_sha256: str | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not shard_dirs:
        raise ValueError("at least one train-only development shard is required")
    reports: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    for shard_dir in shard_dirs:
        terminal_path = shard_dir / "terminal.json"
        summary_path = shard_dir / "summary.json"
        if not terminal_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"incomplete development shard: {shard_dir}")
        terminal = json.loads(terminal_path.read_text())
        summary = json.loads(summary_path.read_text())
        if terminal.get("status") != "complete":
            raise ValueError(f"development shard is not complete: {shard_dir}")
        if summary.get("selection_split") != "train" or not bool(
            summary.get("risk_calibration_requested")
        ):
            raise ValueError(f"development shard is not train-only: {shard_dir}")
        basis = dict(summary.get("basis") or {})
        if (
            expected_checkpoint_sha256 is not None
            and basis.get("checkpoint_sha256") != expected_checkpoint_sha256
        ):
            raise ValueError(f"development shard checkpoint mismatch: {shard_dir}")
        shard = dict(summary.get("validation_shard") or {})
        metadata.append(
            {
                "directory": str(shard_dir),
                "summary_sha256": _sha256_bytes(summary_path),
                "terminal_sha256": _sha256_bytes(terminal_path),
                "checkpoint_sha256": str(basis.get("checkpoint_sha256", "")),
                "index": int(shard.get("index", -1)),
                "count": int(shard.get("count", -1)),
            }
        )
        for raw in summary.get("records", []):
            report = dict(raw)
            sample_id = str(report.get("sample_id", ""))
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"duplicate or missing development sample: {sample_id}")
            if bool(dict(report.get("adaptation") or {}).get("future_truth_used", True)):
                raise ValueError(f"future truth was used online for {sample_id}")
            sample_ids.add(sample_id)
            reports.append(report)
    shard_count = len(shard_dirs)
    if sorted(item["index"] for item in metadata) != list(range(shard_count)):
        raise ValueError("development shard indices are incomplete")
    if any(item["count"] != shard_count for item in metadata):
        raise ValueError("development shard count metadata is inconsistent")
    return reports, metadata


def _selection_metrics(
    reports: list[dict[str, object]],
    *,
    enabled_families: set[str],
    deployed_strength_floors: dict[str, float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for report in reports:
        family = str(report["medium_type"])
        parent = float(report["parent_future_fullfield_relative_l2"])
        enabled = family in enabled_families
        adaptation = dict(report["adaptation"])
        accepted = (
            enabled
            and bool(adaptation["accepted"])
            and float(adaptation["unconstrained_correction_ratio"])
            >= float(deployed_strength_floors[family])
        )
        output = (
            float(report["future_fullfield_relative_l2"])
            if accepted
            else parent
        )
        rows.append(
            {
                "family": family,
                "parent": parent,
                "output": output,
                "accepted": accepted,
                "nonworse": output <= parent * (1.0 + 1.0e-9),
                "relative_improvement": (parent - output) / max(parent, 1.0e-12),
            }
        )
    if not rows:
        raise ValueError("development reports are empty")
    families: dict[str, dict[str, object]] = {}
    for family in ALLOWED_MEDIUM_TYPES:
        selected = [row for row in rows if row["family"] == family]
        if not selected:
            raise ValueError(f"development reports omit family {family}")
        parent_mean = sum(float(row["parent"]) for row in selected) / len(selected)
        output_mean = sum(float(row["output"]) for row in selected) / len(selected)
        families[family] = {
            "record_count": len(selected),
            "enabled": family in enabled_families,
            "accepted_fraction": sum(bool(row["accepted"]) for row in selected)
            / len(selected),
            "nonworse_fraction": sum(bool(row["nonworse"]) for row in selected)
            / len(selected),
            "relative_improvement": (parent_mean - output_mean)
            / max(parent_mean, 1.0e-12),
        }
    metrics = {
        "record_count": len(rows),
        "accepted_fraction": sum(bool(row["accepted"]) for row in rows) / len(rows),
        "nonworse_fraction": sum(bool(row["nonworse"]) for row in rows) / len(rows),
        "mean_relative_improvement": sum(
            float(row["relative_improvement"]) for row in rows
        )
        / len(rows),
        "families": families,
    }
    checks = {
        "minimum_nonworse_fraction": float(metrics["nonworse_fraction"]) >= 0.95,
        "minimum_mean_improvement": float(metrics["mean_relative_improvement"])
        >= 0.01,
        "every_family_mean_nonworse": all(
            float(item["relative_improvement"]) >= 0.0
            for item in families.values()
        ),
    }
    metrics["checks"] = checks
    if not all(checks.values()):
        raise RuntimeError(f"family-selective development gate failed: {checks}")
    return metrics


def apply_gate(
    config_path: str | Path,
    *,
    source_checkpoint: str | Path,
    output_dir: str | Path,
    shard_dirs: tuple[str | Path, ...],
    enabled_families: tuple[str, ...],
    minimum_strength_floors: dict[str, float] | None = None,
    exclusion_shard_dirs: tuple[str | Path, ...] = (),
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("CPADC config must contain a mapping")
    enabled = {str(value) for value in enabled_families}
    allowed = set(ALLOWED_MEDIUM_TYPES)
    if not enabled or not enabled <= allowed:
        raise ValueError("enabled families must be a nonempty registered subset")

    source = Path(source_checkpoint).expanduser().resolve()
    source_sha256 = _sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema") != "causal_physics_aligned_defect_correction_v1":
        raise ValueError("source checkpoint type mismatch")
    if dict(payload.get("implementation_digests") or {}) != cpadc_implementation_digests(
        ROOT
    ):
        raise ValueError("source checkpoint implementation contract mismatch")
    parent_calibration = dict(payload.get("risk_calibration") or {})
    calibration_ids = {str(value) for value in parent_calibration.get("sample_ids", [])}
    if not calibration_ids:
        raise ValueError("source checkpoint lacks train-only risk calibration provenance")

    normalized_dirs = tuple(Path(value).expanduser().resolve() for value in shard_dirs)
    reports, shard_metadata = _load_development_reports(
        normalized_dirs, expected_checkpoint_sha256=source_sha256
    )
    development_ids = {str(report["sample_id"]) for report in reports}
    if development_ids & calibration_ids:
        raise ValueError("development reports overlap source risk calibration")
    manifest = build_manifest(config["source_h5"])
    basis_per_family = int((config.get("cpadc", {}) or {}).get("per_family", 0))
    basis_rows = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=basis_per_family,
        seed=int(config.get("seed", 372)),
    )
    if development_ids & {row.sample_id for row in basis_rows}:
        raise ValueError("development reports overlap basis meta-training")
    registered_train_ids = {
        row.sample_id for row in manifest.records if row.split == "train"
    }
    if not development_ids <= registered_train_ids:
        raise ValueError("development reports contain non-train samples")

    solve_contract = dict(payload.get("online_solve_contract") or {})
    floors = dict(
        solve_contract.get("minimum_unconstrained_correction_ratio_by_family")
        or {}
    )
    if set(floors) != allowed or not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in floors.values()
    ):
        raise ValueError("source checkpoint lacks valid family strength floors")
    overrides = {
        str(family): float(value)
        for family, value in dict(minimum_strength_floors or {}).items()
    }
    if not set(overrides) <= enabled:
        raise ValueError("strength-floor overrides require an enabled family")
    if not all(math.isfinite(value) and value >= 0.0 for value in overrides.values()):
        raise ValueError("strength-floor overrides must be finite and nonnegative")
    if any(overrides[family] < float(floors[family]) for family in overrides):
        raise ValueError("selective gate may only make a source strength floor stricter")
    deployed_floors = {
        family: (
            float(overrides.get(family, floors[family]))
            if family in enabled
            else DISABLED_STRENGTH_FLOOR
        )
        for family in ALLOWED_MEDIUM_TYPES
    }
    metrics = _selection_metrics(
        reports,
        enabled_families=enabled,
        deployed_strength_floors=deployed_floors,
    )
    normalized_exclusion_dirs = tuple(
        Path(value).expanduser().resolve() for value in exclusion_shard_dirs
    )
    exclusion_reports, exclusion_metadata = _load_development_reports(
        normalized_exclusion_dirs, expected_checkpoint_sha256=None
    ) if normalized_exclusion_dirs else ([], [])
    provenance_exclusion_ids = {
        str(report["sample_id"]) for report in exclusion_reports
    }
    if provenance_exclusion_ids & development_ids:
        raise ValueError("provenance-only exclusions overlap selection development")
    if provenance_exclusion_ids & calibration_ids:
        raise ValueError("provenance-only exclusions overlap source calibration")
    if provenance_exclusion_ids & {row.sample_id for row in basis_rows}:
        raise ValueError("provenance-only exclusions overlap basis meta-training")
    if not provenance_exclusion_ids <= registered_train_ids:
        raise ValueError("provenance-only exclusions contain non-train samples")
    canonical = json.dumps(
        {
            "source_checkpoint_sha256": source_sha256,
            "enabled_families": sorted(enabled),
            "deployed_strength_floors": deployed_floors,
            "development_sample_ids": sorted(development_ids),
            "development_summary_sha256s": sorted(
                item["summary_sha256"] for item in shard_metadata
            ),
            "provenance_exclusion_sample_ids": sorted(provenance_exclusion_ids),
            "provenance_exclusion_summary_sha256s": sorted(
                item["summary_sha256"] for item in exclusion_metadata
            ),
            "metrics": metrics,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    selection_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    solve_contract["minimum_unconstrained_correction_ratio_by_family"] = (
        deployed_floors
    )
    solve_contract["calibration_digest"] = selection_digest
    payload["online_solve_contract"] = solve_contract
    payload["selective_gate_from_checkpoint"] = str(source)
    payload["selective_gate_from_checkpoint_sha256"] = source_sha256
    payload["risk_calibration"] = {
        "schema": "cpadc_train_family_selective_risk_gate_v1",
        "future_truth_scope": "disjoint_train_split_only",
        # Keep the common calibrated-checkpoint manifest fields at the top
        # level so the immutable deployment loader can validate this derived
        # checkpoint without knowing the richer selective-gate schema.
        "record_count": len(reports),
        "sample_ids": sorted(development_ids),
        "selection_digest": selection_digest,
        "enabled_families": sorted(enabled),
        "disabled_families": sorted(allowed - enabled),
        "disabled_strength_floor": DISABLED_STRENGTH_FLOOR,
        "deployed_strength_floor_by_family": deployed_floors,
        "development_record_count": len(reports),
        "development_sample_ids": sorted(development_ids),
        "development_shards": shard_metadata,
        "provenance_exclusion_sample_ids": sorted(provenance_exclusion_ids),
        "provenance_exclusion_shards": exclusion_metadata,
        "selection": metrics,
        "parent_risk_calibration": parent_calibration,
    }

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "selective.pt"
    temporary = output / ".selective.pt.tmp"
    if checkpoint.exists() or temporary.exists() or (output / "terminal.json").exists():
        raise FileExistsError(f"refusing to overwrite selective gate output: {output}")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    terminal = {
        "status": "complete",
        "schema": "cpadc_train_family_selective_gate_terminal_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_sha256,
        "selection_digest": selection_digest,
        "enabled_families": sorted(enabled),
        "disabled_families": sorted(allowed - enabled),
        "deployed_strength_floor_by_family": deployed_floors,
        "provenance_exclusion_record_count": len(provenance_exclusion_ids),
        "selection": metrics,
        "online_future_truth_used": False,
        "claim": "train-only selective risk gate complete; validation not yet run",
    }
    (output / "terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n"
    )
    return checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument(
        "--enable-family", action="append", required=True, choices=ALLOWED_MEDIUM_TYPES
    )
    parser.add_argument(
        "--minimum-strength-floor",
        action="append",
        default=[],
        metavar="FAMILY=VALUE",
    )
    parser.add_argument("--exclude-shard-dir", action="append", default=[])
    args = parser.parse_args(argv)
    overrides: dict[str, float] = {}
    for raw in args.minimum_strength_floor:
        family, separator, value = str(raw).partition("=")
        if separator != "=" or family not in ALLOWED_MEDIUM_TYPES:
            parser.error("--minimum-strength-floor must be FAMILY=VALUE")
        if family in overrides:
            parser.error(f"duplicate strength-floor override for {family}")
        try:
            overrides[family] = float(value)
        except ValueError:
            parser.error(f"invalid strength-floor value for {family}: {value}")
    apply_gate(
        args.config,
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
        shard_dirs=tuple(args.shard_dir),
        enabled_families=tuple(args.enable_family),
        minimum_strength_floors=overrides,
        exclusion_shard_dirs=tuple(args.exclude_shard_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
