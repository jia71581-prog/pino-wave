#!/usr/bin/env python3
"""Calibrate a train-only causal-strength abstention rule for a CPADC basis."""
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
from saved_time_phase_operator_v4.instance_adaptation.defect_correction import (
    CPADC_SCHEMA_VERSION,
)
from saved_time_phase_operator_v4.instance_adaptation.cpadc_contract import (
    cpadc_implementation_digests,
    validate_dataset_cpml_contract,
)
from saved_time_phase_operator_v4.instance_adaptation.forced_defect import (
    saved_grid_cpml_config,
)
from scripts.run_causal_defect_adaptation import build_risk_calibration_manifest
from scripts.train_v5_residual_meta import _sha256


def _metrics(
    reports: list[dict[str, object]],
    *,
    strength_floor: float | dict[str, float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for report in reports:
        parent = float(report["parent_future_fullfield_relative_l2"])
        candidate = float(report["future_fullfield_relative_l2"])
        adaptation = dict(report["adaptation"])
        family = str(report["medium_type"])
        selected_floor = (
            float(strength_floor[family])
            if isinstance(strength_floor, dict)
            else float(strength_floor)
        )
        accepted = bool(adaptation["accepted"]) and float(
            adaptation["unconstrained_correction_ratio"]
        ) >= selected_floor
        output = candidate if accepted else parent
        rows.append(
            {
                "family": family,
                "parent": parent,
                "output": output,
                "accepted": accepted,
                "relative_improvement": (parent - output) / max(parent, 1.0e-12),
                "nonworse": output <= parent * (1.0 + 1.0e-9),
            }
        )
    families: dict[str, dict[str, object]] = {}
    present_families = tuple(
        family
        for family in ALLOWED_MEDIUM_TYPES
        if any(row["family"] == family for row in rows)
    )
    for family in present_families:
        selected = [row for row in rows if row["family"] == family]
        parent_mean = sum(float(row["parent"]) for row in selected) / len(selected)
        output_mean = sum(float(row["output"]) for row in selected) / len(selected)
        families[family] = {
            "record_count": len(selected),
            "accepted_fraction": sum(bool(row["accepted"]) for row in selected)
            / len(selected),
            "nonworse_fraction": sum(bool(row["nonworse"]) for row in selected)
            / len(selected),
            "relative_improvement": (parent_mean - output_mean)
            / max(parent_mean, 1.0e-12),
        }
    result = {
        "record_count": len(rows),
        "accepted_fraction": sum(bool(row["accepted"]) for row in rows) / len(rows),
        "nonworse_fraction": sum(bool(row["nonworse"]) for row in rows) / len(rows),
        "mean_relative_improvement": sum(
            float(row["relative_improvement"]) for row in rows
        )
        / len(rows),
        "families": families,
    }
    if isinstance(strength_floor, dict):
        result["strength_floor_by_family"] = {
            family: float(strength_floor[family]) for family in ALLOWED_MEDIUM_TYPES
        }
    else:
        result["strength_floor"] = float(strength_floor)
    return result


def select_strength_threshold(
    reports: list[dict[str, object]],
    *,
    minimum_nonworse_fraction: float = 0.90,
    minimum_family_nonworse_fraction: float = 0.90,
    minimum_mean_improvement: float = 0.01,
) -> dict[str, object]:
    """Choose the safest threshold that retains the required train benefit."""

    if not reports:
        raise ValueError("risk calibration requires reports")
    if {str(report["medium_type"]) for report in reports} != set(
        ALLOWED_MEDIUM_TYPES
    ):
        raise ValueError("risk calibration must cover every medium family")
    values = sorted(
        {
            float(dict(report["adaptation"])["unconstrained_correction_ratio"])
            for report in reports
        }
    )
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("calibration strength values must be finite and nonnegative")
    feasible: list[dict[str, object]] = []
    for value in values:
        metrics = _metrics(reports, strength_floor=value)
        family_metrics = dict(metrics["families"])
        checks = {
            "overall_nonworse_buffer": float(metrics["nonworse_fraction"])
            >= float(minimum_nonworse_fraction),
            "every_family_nonworse_buffer": all(
                float(item["nonworse_fraction"])
                >= float(minimum_family_nonworse_fraction)
                for item in family_metrics.values()
            ),
            "every_family_mean_nonworse": all(
                float(item["relative_improvement"]) >= 0.0
                for item in family_metrics.values()
            ),
            "mean_improvement": float(metrics["mean_relative_improvement"])
            >= float(minimum_mean_improvement),
        }
        metrics["checks"] = checks
        if all(checks.values()):
            feasible.append(metrics)
    if not feasible:
        raise RuntimeError("no causal-strength threshold satisfies calibration safety")
    selected = max(
        feasible,
        key=lambda item: (
            float(item["nonworse_fraction"]),
            min(
                float(value["nonworse_fraction"])
                for value in dict(item["families"]).values()
            ),
            float(item["mean_relative_improvement"]),
            float(item["accepted_fraction"]),
        ),
    )
    selected["requirements"] = {
        "minimum_nonworse_fraction": float(minimum_nonworse_fraction),
        "minimum_family_nonworse_fraction": float(
            minimum_family_nonworse_fraction
        ),
        "minimum_mean_improvement": float(minimum_mean_improvement),
    }
    selected["candidate_threshold_count"] = len(values)
    return selected


def select_family_strength_thresholds(
    reports: list[dict[str, object]],
    *,
    minimum_nonworse_fraction: float = 0.90,
    minimum_family_nonworse_fraction: float = 0.95,
    minimum_mean_improvement: float = 0.01,
) -> dict[str, object]:
    """Choose one train-only benefit-maximizing strength floor per family."""

    if not reports:
        raise ValueError("risk calibration requires reports")
    present = {str(report["medium_type"]) for report in reports}
    if present != set(ALLOWED_MEDIUM_TYPES):
        raise ValueError("family calibration must cover every medium family")
    if not 0.0 <= minimum_nonworse_fraction <= 1.0:
        raise ValueError("minimum_nonworse_fraction must lie in [0,1]")
    if not 0.0 <= minimum_family_nonworse_fraction <= 1.0:
        raise ValueError("minimum_family_nonworse_fraction must lie in [0,1]")

    floors: dict[str, float] = {}
    candidate_counts: dict[str, int] = {}
    for family in ALLOWED_MEDIUM_TYPES:
        family_reports = [
            report for report in reports if str(report["medium_type"]) == family
        ]
        values = sorted(
            {
                float(dict(report["adaptation"])["unconstrained_correction_ratio"])
                for report in family_reports
            }
        )
        if not values or not all(
            math.isfinite(value) and value >= 0.0 for value in values
        ):
            raise ValueError(
                f"{family} calibration strengths must be finite and nonnegative"
            )
        feasible: list[dict[str, object]] = []
        for value in values:
            metrics = _metrics(family_reports, strength_floor=value)
            if (
                float(metrics["nonworse_fraction"])
                >= float(minimum_family_nonworse_fraction)
                and float(metrics["mean_relative_improvement"]) >= 0.0
            ):
                feasible.append(metrics)
        if not feasible:
            raise RuntimeError(
                f"no {family} strength threshold satisfies calibration safety"
            )
        selected = max(
            feasible,
            key=lambda item: (
                float(item["mean_relative_improvement"]),
                float(item["nonworse_fraction"]),
                float(item["accepted_fraction"]),
                float(item["strength_floor"]),
            ),
        )
        floors[family] = float(selected["strength_floor"])
        candidate_counts[family] = len(values)

    combined = _metrics(reports, strength_floor=floors)
    family_metrics = dict(combined["families"])
    checks = {
        "overall_nonworse_buffer": float(combined["nonworse_fraction"])
        >= float(minimum_nonworse_fraction),
        "every_family_nonworse_buffer": all(
            float(item["nonworse_fraction"])
            >= float(minimum_family_nonworse_fraction)
            for item in family_metrics.values()
        ),
        "every_family_mean_nonworse": all(
            float(item["relative_improvement"]) >= 0.0
            for item in family_metrics.values()
        ),
        "mean_improvement": float(combined["mean_relative_improvement"])
        >= float(minimum_mean_improvement),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"family-specific thresholds fail aggregate calibration safety: {checks}"
        )
    combined["checks"] = checks
    combined["requirements"] = {
        "minimum_nonworse_fraction": float(minimum_nonworse_fraction),
        "minimum_family_nonworse_fraction": float(
            minimum_family_nonworse_fraction
        ),
        "minimum_mean_improvement": float(minimum_mean_improvement),
    }
    combined["candidate_threshold_count_by_family"] = candidate_counts
    combined["selection_strategy"] = (
        "family_specific_benefit_with_nonworse_buffer_v1"
    )
    return combined


def calibrate(
    config_path: str | Path,
    *,
    basis_checkpoint: str | Path,
    output_dir: str | Path,
    shard_dirs: tuple[str | Path, ...],
    calibration_per_family: int,
    calibration_seed: int,
    minimum_nonworse_fraction: float,
    minimum_family_nonworse_fraction: float,
    minimum_mean_improvement: float,
    family_specific: bool = False,
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("CPADC config must contain a mapping")
    manifest = build_manifest(config["source_h5"])
    basis_per_family = int((config.get("cpadc", {}) or {}).get("per_family", 0))
    expected_rows = build_risk_calibration_manifest(
        manifest,
        basis_per_family=basis_per_family,
        basis_seed=int(config.get("seed", 372)),
        calibration_per_family=int(calibration_per_family),
        calibration_seed=int(calibration_seed),
    )
    expected_ids = tuple(row.sample_id for row in expected_rows)
    expected_set = set(expected_ids)
    normalized_dirs = tuple(Path(value).expanduser().resolve() for value in shard_dirs)
    if len(normalized_dirs) < 1:
        raise ValueError("at least one calibration shard is required")

    reports_by_id: dict[str, dict[str, object]] = {}
    basis_identity: dict[str, object] | None = None
    shard_indices: list[int] = []
    for shard_dir in normalized_dirs:
        summary_path = shard_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing calibration summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if summary.get("selection_split") != "train" or not bool(
            summary.get("risk_calibration_requested")
        ):
            raise ValueError(f"not a train-only risk calibration shard: {summary_path}")
        identity = dict(summary["basis"])
        if basis_identity is None:
            basis_identity = identity
        elif identity != basis_identity:
            raise ValueError(f"basis identity mismatch in {summary_path}")
        shard = dict(summary["validation_shard"])
        if int(shard["count"]) != len(normalized_dirs):
            raise ValueError(f"calibration shard count mismatch in {summary_path}")
        shard_indices.append(int(shard["index"]))
        for raw in summary["records"]:
            report = dict(raw)
            sample_id = str(report["sample_id"])
            if sample_id in reports_by_id:
                raise ValueError(f"duplicate calibration sample: {sample_id}")
            if sample_id not in expected_set:
                raise ValueError(f"unexpected calibration sample: {sample_id}")
            adaptation = dict(report["adaptation"])
            if bool(adaptation.get("future_truth_used", True)):
                raise ValueError(f"future truth used by online solve: {sample_id}")
            if not (shard_dir / sample_id / "adaptation.pt").is_file():
                raise FileNotFoundError(f"missing sealed adaptation: {sample_id}")
            reports_by_id[sample_id] = report
    if sorted(shard_indices) != list(range(len(normalized_dirs))):
        raise ValueError(f"calibration shard indices are incomplete: {shard_indices}")
    if set(reports_by_id) != expected_set:
        missing = sorted(expected_set - set(reports_by_id))
        raise ValueError(f"calibration coverage is incomplete: {missing[:3]}")

    reports = [reports_by_id[sample_id] for sample_id in expected_ids]
    selector = (
        select_family_strength_thresholds
        if family_specific
        else select_strength_threshold
    )
    selected = selector(
        reports,
        minimum_nonworse_fraction=minimum_nonworse_fraction,
        minimum_family_nonworse_fraction=minimum_family_nonworse_fraction,
        minimum_mean_improvement=minimum_mean_improvement,
    )
    source_checkpoint = Path(basis_checkpoint).expanduser().resolve()
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "causal_physics_aligned_defect_correction_v1":
        raise ValueError("basis checkpoint type mismatch")
    # Calibration may alter only the train-only risk threshold.  It must never
    # relabel a legacy interior-only defect as the CPML-aware schema.
    if int(payload.get("schema_version", 0)) != CPADC_SCHEMA_VERSION:
        raise ValueError("basis checkpoint schema mismatch")
    defect_contract = payload.get("defect_contract") or {}
    cpml_config = saved_grid_cpml_config(config.get("pde_cpml"))
    if cpml_config is None:
        raise ValueError("risk calibration requires the registered CPML contract")
    dataset_numerical_contract = validate_dataset_cpml_contract(
        config["source_h5"],
        saved_grid_cpml=cpml_config.as_dict(),
        saved_dt_s=float(config.get("dt_s", 0.0025)),
        saved_dx_m=float(config.get("dx_m", 10.0)),
        saved_dz_m=float(config.get("dz_m", 10.0)),
    )
    implementation_digests = cpadc_implementation_digests(ROOT)
    if (
        defect_contract.get("name")
        != "source_consistent_saved_grid_lwc84_cfs_cpml_v3"
        or not isinstance(defect_contract.get("cpml"), dict)
        or defect_contract.get("saved_source_contract")
        != "direct_saved_grid_bilinear_unit_mass_reinjection"
        or defect_contract.get("exact_fine_source_restriction") is not False
        or dict(payload.get("dataset_numerical_contract") or {})
        != dataset_numerical_contract
        or dict(payload.get("implementation_digests") or {})
        != implementation_digests
    ):
        raise ValueError("basis checkpoint numerical or implementation contract mismatch")
    if basis_identity is None or basis_identity.get("checkpoint_sha256") != _sha256(
        source_checkpoint
    ):
        raise ValueError("calibration shards are not bound to the source checkpoint")

    canonical = json.dumps(
        {
            "sample_ids": expected_ids,
            "selected": selected,
            "source_checkpoint_sha256": _sha256(source_checkpoint),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    calibration_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    calibration = {
        "schema": "cpadc_disjoint_train_risk_calibration_v1",
        "future_truth_scope": "disjoint_train_split_only",
        "record_count": len(reports),
        "per_family": int(calibration_per_family),
        "basis_training_per_family": basis_per_family,
        "basis_seed": int(config.get("seed", 372)),
        "calibration_seed": int(calibration_seed),
        "sample_ids": list(expected_ids),
        "calibration_digest": calibration_digest,
        "selection": selected,
    }
    payload["schema_version"] = CPADC_SCHEMA_VERSION
    payload["calibrated_from_checkpoint"] = str(source_checkpoint)
    payload["calibrated_from_checkpoint_sha256"] = _sha256(source_checkpoint)
    payload["risk_calibration"] = calibration
    source_solve_contract = dict(payload.get("online_solve_contract") or {})
    source_solve_name = str(source_solve_contract.get("name", ""))
    sparse_online_defect = source_solve_name in {
        "cpu_causal_sparse_defect_observation_bridge_ridge_v6",
        "cpu_causal_weak_sparse_defect_observation_bridge_ridge_v7",
    }
    weak_sparse_online_defect = source_solve_name == (
        "cpu_causal_weak_sparse_defect_observation_bridge_ridge_v7"
    )
    solve_contract = {
        "name": (
            (
                (
                    "cpu_causal_weak_sparse_defect_observation_bridge_"
                    "family_calibrated_ridge_v7"
                    if family_specific
                    else "cpu_causal_weak_sparse_defect_observation_bridge_"
                    "calibrated_ridge_v7"
                )
                if weak_sparse_online_defect
                else (
                    "cpu_causal_sparse_defect_observation_bridge_"
                    "family_calibrated_ridge_v6"
                    if family_specific
                    else "cpu_causal_sparse_defect_observation_bridge_"
                    "calibrated_ridge_v6"
                )
            )
            if sparse_online_defect
            else (
                "cpu_causal_observation_bridge_family_calibrated_ridge_v5"
                if family_specific
                else "cpu_causal_observation_bridge_calibrated_ridge_v5"
            )
        ),
        "projection_uses_parent_rms": True,
        "trust_fraction_learned_offline": True,
        "observed_probe_design": bool(
            (payload.get("online_solve_contract") or {}).get(
                "observed_probe_design", False
            )
        ),
        "strength_feature": "unconstrained_correction_rms_over_parent_rms",
        "acceptance_comparator": "greater_than_or_equal",
        "calibration_digest": calibration_digest,
        "online_defect_weight": float(
            source_solve_contract.get("online_defect_weight", -1.0)
        ),
        "online_defect_design": source_solve_contract.get(
            "online_defect_design", "materialized"
        ),
        "online_defect_time_order": int(
            source_solve_contract.get("online_defect_time_order", 4)
        ),
        "online_defect_test_function_width": int(
            source_solve_contract.get("online_defect_test_function_width", 1)
        ),
        "online_defect_test_function_normalization": source_solve_contract.get(
            "online_defect_test_function_normalization", "missing"
        ),
        "online_physics_point_count": int(
            source_solve_contract.get("online_physics_point_count", 1)
        ),
        "online_physics_sampling": source_solve_contract.get(
            "online_physics_sampling", "fixed"
        ),
        "online_rad_k": float(source_solve_contract.get("online_rad_k", 1.0)),
        "online_rad_c": float(source_solve_contract.get("online_rad_c", 1.0)),
        "online_rad_time_tilt": float(
            source_solve_contract.get("online_rad_time_tilt", 1.5)
        ),
        "online_physics_seed": int(
            source_solve_contract.get("online_physics_seed", 372)
        ),
        "online_prior_weight": float(
            source_solve_contract.get("online_prior_weight", 1.0e-4)
        ),
        "online_spatial_stride": int(
            source_solve_contract.get("online_spatial_stride", 4)
        ),
        "online_observed_weight": float(
            source_solve_contract.get("online_observed_weight", 0.0)
        ),
        "online_bridge_weight": float(
            source_solve_contract.get("online_bridge_weight", 0.0)
        ),
        "coefficient_solve_device": source_solve_contract.get(
            "coefficient_solve_device"
        ),
        "coefficient_objective_device": source_solve_contract.get(
            "coefficient_objective_device"
        ),
        "correction_materialization_device": source_solve_contract.get(
            "correction_materialization_device"
        ),
        "synthetic_bridge_device": source_solve_contract.get(
            "synthetic_bridge_device"
        ),
        "offline_physics_operator": source_solve_contract.get(
            "offline_physics_operator"
        ),
    }
    if family_specific:
        solve_contract["minimum_unconstrained_correction_ratio_by_family"] = {
            family: float(selected["strength_floor_by_family"][family])
            for family in ALLOWED_MEDIUM_TYPES
        }
    else:
        solve_contract["minimum_unconstrained_correction_ratio"] = float(
            selected["strength_floor"]
        )
    payload["online_solve_contract"] = solve_contract

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "calibrated.pt"
    temporary = output / ".calibrated.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    report = {
        "status": "complete",
        "schema": "cpadc_risk_calibration_terminal_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "risk_calibration": calibration,
        "online_future_truth_used": False,
        "claim": "train-only risk calibration complete; validation not yet run",
    }
    (output / "terminal.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--basis-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--calibration-per-family", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=10372)
    parser.add_argument("--minimum-nonworse-fraction", type=float, default=0.90)
    parser.add_argument(
        "--minimum-family-nonworse-fraction", type=float, default=0.90
    )
    parser.add_argument("--minimum-mean-improvement", type=float, default=0.01)
    parser.add_argument("--family-specific", action="store_true")
    args = parser.parse_args(argv)
    calibrate(
        args.config,
        basis_checkpoint=args.basis_checkpoint,
        output_dir=args.output_dir,
        shard_dirs=tuple(args.shard_dir),
        calibration_per_family=args.calibration_per_family,
        calibration_seed=args.calibration_seed,
        minimum_nonworse_fraction=args.minimum_nonworse_fraction,
        minimum_family_nonworse_fraction=args.minimum_family_nonworse_fraction,
        minimum_mean_improvement=args.minimum_mean_improvement,
        family_specific=args.family_specific,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
