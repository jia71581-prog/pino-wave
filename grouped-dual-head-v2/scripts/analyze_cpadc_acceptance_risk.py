#!/usr/bin/env python3
"""Diagnose CPADC accepted-correction risk without authorizing a new policy.

The guard thresholds are fitted only on the disjoint train calibration rows.
Validation and test_id are retrospective diagnostics and can never promote the
result produced by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest


FEATURES = (
    "strength",
    "strength_margin_fraction",
    "log10_condition_number",
    "projection_scale",
    "trust_fraction",
    "effective_correction_ratio_limit",
    "correction_ratio",
    "causal_objective_gain_fraction",
    "coefficient_l2",
    "coefficient_linf_over_l2",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _load_floors(checkpoint_path: str | Path) -> tuple[dict[str, float], str]:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = dict(payload.get("online_solve_contract") or {})
    if contract.get("name") != (
        "ridge_direction_family_calibrated_strength_abstention_v1"
    ):
        raise ValueError("checkpoint does not contain the frozen R7 solve contract")
    raw = dict(contract.get("minimum_unconstrained_correction_ratio_by_family") or {})
    if set(raw) != set(ALLOWED_MEDIUM_TYPES):
        raise ValueError("checkpoint is missing family-specific strength floors")
    floors = {family: float(raw[family]) for family in ALLOWED_MEDIUM_TYPES}
    if not all(math.isfinite(value) and value > 0.0 for value in floors.values()):
        raise ValueError("family-specific strength floors must be finite and positive")
    return floors, _sha256(path)


def _adaptation_index(shard_dirs: tuple[Path, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for shard_dir in shard_dirs:
        for path in shard_dir.glob("*/adaptation.pt"):
            sample_id = path.parent.name
            if sample_id in result:
                raise ValueError(f"duplicate adaptation artifact: {sample_id}")
            result[sample_id] = path
    return result


def _extract_row(
    report: dict[str, object],
    *,
    adaptation_path: Path,
    group_id: str,
    strength_floor: float,
    calibration_role: bool,
) -> dict[str, object]:
    adaptation = dict(report["adaptation"])
    if bool(adaptation.get("future_truth_used", True)):
        raise ValueError(f"online future truth was used: {report['sample_id']}")
    stored = torch.load(adaptation_path, map_location="cpu", weights_only=False)
    stored_adaptation = dict(stored["adaptation"])
    coefficients = torch.as_tensor(stored_adaptation["coefficients"]).float()
    expected_digest = str(dict(adaptation["coefficients"])["tensor_sha256"])
    if _tensor_sha256(coefficients) != expected_digest:
        raise ValueError(f"coefficient digest mismatch: {report['sample_id']}")
    if int(coefficients.numel()) != int(adaptation["coefficient_count"]):
        raise ValueError(f"coefficient count mismatch: {report['sample_id']}")

    family = str(report["medium_type"])
    strength = float(adaptation["unconstrained_correction_ratio"])
    if calibration_role:
        baseline_accepted = bool(adaptation["accepted"]) and strength >= strength_floor
    else:
        selected_floor = float(adaptation["minimum_unconstrained_correction_ratio"])
        if not math.isclose(selected_floor, strength_floor, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"R7 strength-floor mismatch: {report['sample_id']}")
        baseline_accepted = bool(adaptation["accepted"])

    parent = float(report["parent_future_fullfield_relative_l2"])
    candidate = float(report["future_fullfield_relative_l2"])
    baseline_output = candidate if baseline_accepted else parent
    before = float(adaptation["objective_before"])
    after = float(adaptation["objective_after"])
    coefficient_l2 = float(torch.linalg.vector_norm(coefficients))
    coefficient_linf = float(torch.max(torch.abs(coefficients)))
    return {
        "sample_id": str(report["sample_id"]),
        "group_id": str(group_id),
        "family": family,
        "parent": parent,
        "candidate": candidate,
        "baseline_output": baseline_output,
        "baseline_accepted": baseline_accepted,
        "strength": strength,
        "strength_margin_fraction": strength / strength_floor - 1.0,
        "log10_condition_number": math.log10(
            max(float(adaptation["condition_number"]), 1.0e-30)
        ),
        "projection_scale": float(adaptation["projection_scale"]),
        "trust_fraction": float(adaptation["trust_fraction"]),
        "effective_correction_ratio_limit": float(
            adaptation["effective_correction_ratio_limit"]
        ),
        "correction_ratio": float(adaptation["correction_ratio"]),
        "causal_objective_gain_fraction": (before - after)
        / max(abs(before), 1.0e-12),
        "coefficient_l2": coefficient_l2,
        "coefficient_linf_over_l2": coefficient_linf
        / max(coefficient_l2, 1.0e-12),
    }


def _load_calibration_rows(
    *,
    config_path: Path,
    shard_dirs: tuple[Path, ...],
    floors: dict[str, float],
) -> tuple[list[dict[str, object]], list[str]]:
    config = yaml.safe_load(config_path.read_text())
    manifest = build_manifest(config["source_h5"])
    groups = {row.sample_id: row.group_id for row in manifest.records}
    adaptations = _adaptation_index(shard_dirs)
    rows: list[dict[str, object]] = []
    summary_hashes: list[str] = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("selection_split") != "train" or not bool(
            summary.get("risk_calibration_requested")
        ):
            raise ValueError(f"not a train risk-calibration shard: {summary_path}")
        summary_hashes.append(_sha256(summary_path))
        for raw in summary["records"]:
            report = dict(raw)
            sample_id = str(report["sample_id"])
            if sample_id not in groups or sample_id not in adaptations:
                raise ValueError(f"incomplete calibration identity: {sample_id}")
            family = str(report["medium_type"])
            rows.append(
                _extract_row(
                    report,
                    adaptation_path=adaptations[sample_id],
                    group_id=groups[sample_id],
                    strength_floor=floors[family],
                    calibration_role=True,
                )
            )
    if len(rows) != len({str(row["sample_id"]) for row in rows}):
        raise ValueError("duplicate calibration rows")
    return rows, summary_hashes


def _load_evaluation_rows(
    *,
    artifact_dir: Path,
    shard_subdir: str,
    expected_split: str,
    floors: dict[str, float],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    evaluation = artifact_dir / "evaluation"
    summary_path = evaluation / "summary.json"
    manifest_path = evaluation / "instance_manifest.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    manifest_splits = {str(row["split"]) for row in manifest}
    if manifest_splits != {expected_split}:
        raise ValueError(
            f"evaluation split mismatch: expected {expected_split}, got {manifest_splits}"
        )
    summary_split = str(summary.get("evaluation_split", expected_split))
    if summary_split != expected_split:
        raise ValueError(
            f"summary split mismatch: expected {expected_split}, got {summary_split}"
        )
    groups = {str(row["sample_id"]): str(row["group_id"]) for row in manifest}
    shard_dirs = tuple(sorted((artifact_dir / shard_subdir).glob("shard_*")))
    adaptations = _adaptation_index(shard_dirs)
    rows: list[dict[str, object]] = []
    for raw in summary["records"]:
        report = dict(raw)
        sample_id = str(report["sample_id"])
        if sample_id not in groups or sample_id not in adaptations:
            raise ValueError(f"incomplete evaluation identity: {sample_id}")
        family = str(report["medium_type"])
        rows.append(
            _extract_row(
                report,
                adaptation_path=adaptations[sample_id],
                group_id=groups[sample_id],
                strength_floor=floors[family],
                calibration_role=False,
            )
        )
    if len(rows) != len(manifest) or len(rows) != len(set(groups)):
        raise ValueError(f"evaluation manifest coverage mismatch: {artifact_dir}")
    identity = {
        "summary_sha256": _sha256(summary_path),
        "instance_manifest_sha256": _sha256(manifest_path),
        "evaluation_split": summary_split,
    }
    return rows, identity


def _guard_decisions(
    rows: list[dict[str, object]],
    *,
    coefficient_l2_cap: float | None,
    trust_fraction_floor: float | None,
) -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    for row in rows:
        keep = bool(row["baseline_accepted"])
        if coefficient_l2_cap is not None:
            keep = keep and float(row["coefficient_l2"]) <= coefficient_l2_cap
        if trust_fraction_floor is not None:
            keep = keep and float(row["trust_fraction"]) >= trust_fraction_floor
        decisions[str(row["sample_id"])] = keep
    return decisions


def _metrics_for_decisions(
    rows: list[dict[str, object]], decisions: dict[str, bool]
) -> dict[str, object]:
    outputs: list[float] = []
    improvements: list[float] = []
    kept: list[bool] = []
    for row in rows:
        keep = bool(decisions[str(row["sample_id"])])
        output = float(row["candidate"]) if keep else float(row["parent"])
        parent = float(row["parent"])
        outputs.append(output)
        improvements.append((parent - output) / max(parent, 1.0e-12))
        kept.append(keep)
    families: dict[str, dict[str, object]] = {}
    present_families = tuple(
        family
        for family in ALLOWED_MEDIUM_TYPES
        if any(row["family"] == family for row in rows)
    )
    for family in present_families:
        indices = [i for i, row in enumerate(rows) if row["family"] == family]
        parent_mean = float(np.mean([float(rows[i]["parent"]) for i in indices]))
        output_mean = float(np.mean([outputs[i] for i in indices]))
        families[family] = {
            "record_count": len(indices),
            "accepted_count": sum(kept[i] for i in indices),
            "strict_harm_count": sum(improvements[i] < 0.0 for i in indices),
            "relative_improvement": (parent_mean - output_mean)
            / max(parent_mean, 1.0e-12),
        }
    return {
        "record_count": len(rows),
        "accepted_count": sum(kept),
        "acceptance_fraction": float(np.mean(kept)),
        "strict_harm_count": sum(value < 0.0 for value in improvements),
        "nonworse_fraction": float(
            np.mean([value >= -1.0e-9 for value in improvements])
        ),
        "mean_record_improvement": float(np.mean(improvements)),
        "families": families,
    }


def guard_metrics(
    rows: list[dict[str, object]],
    *,
    coefficient_l2_cap: float | None = None,
    trust_fraction_floor: float | None = None,
) -> dict[str, object]:
    return _metrics_for_decisions(
        rows,
        _guard_decisions(
            rows,
            coefficient_l2_cap=coefficient_l2_cap,
            trust_fraction_floor=trust_fraction_floor,
        ),
    )


def _feasible(
    metrics: dict[str, object],
    *,
    minimum_mean_improvement: float,
    minimum_nonworse_fraction: float,
) -> bool:
    return (
        float(metrics["mean_record_improvement"]) >= minimum_mean_improvement
        and float(metrics["nonworse_fraction"]) >= minimum_nonworse_fraction
        and all(
            float(item["relative_improvement"]) >= 0.0
            for item in dict(metrics["families"]).values()
        )
    )


def select_safety_guard(
    rows: list[dict[str, object]],
    *,
    minimum_mean_improvement: float = 0.01,
    minimum_nonworse_fraction: float = 0.95,
) -> dict[str, object]:
    """Fit the safety-first two-feature guard on calibration rows only."""

    accepted = [row for row in rows if bool(row["baseline_accepted"])]
    if not accepted:
        raise ValueError("guard selection requires accepted calibration rows")
    if {str(row["family"]) for row in rows} != set(ALLOWED_MEDIUM_TYPES):
        raise ValueError("guard selection requires every medium family")
    coefficient_caps = sorted({float(row["coefficient_l2"]) for row in accepted})
    trust_floors = sorted({float(row["trust_fraction"]) for row in accepted})
    raw_candidates: list[tuple[float | None, float | None]] = [
        (cap, floor) for cap in coefficient_caps for floor in trust_floors
    ]
    raw_candidates.extend((cap, None) for cap in coefficient_caps)
    raw_candidates.extend((None, floor) for floor in trust_floors)
    raw_candidates.append((None, None))

    feasible: list[tuple[tuple[float, ...], dict[str, object]]] = []
    for cap, floor in raw_candidates:
        metrics = guard_metrics(
            rows,
            coefficient_l2_cap=cap,
            trust_fraction_floor=floor,
        )
        if not _feasible(
            metrics,
            minimum_mean_improvement=minimum_mean_improvement,
            minimum_nonworse_fraction=minimum_nonworse_fraction,
        ):
            continue
        key = (
            float(metrics["strict_harm_count"]),
            -float(metrics["mean_record_improvement"]),
            -float(metrics["accepted_count"]),
            float(cap) if cap is not None else math.inf,
            -float(floor) if floor is not None else math.inf,
        )
        feasible.append(
            (
                key,
                {
                    "coefficient_l2_cap": cap,
                    "trust_fraction_floor": floor,
                    "metrics": metrics,
                },
            )
        )
    if not feasible:
        raise RuntimeError("no train-only safety guard satisfies the requirements")
    feasible.sort(key=lambda item: item[0])
    selected = feasible[0][1]
    selected["feasible_candidate_count"] = len(feasible)
    selected["requirements"] = {
        "minimum_mean_improvement": float(minimum_mean_improvement),
        "minimum_nonworse_fraction": float(minimum_nonworse_fraction),
        "require_every_family_nonworse": True,
        "selection_order": (
            "minimum_strict_harm_then_maximum_mean_improvement_then_acceptance"
        ),
    }
    return selected


def _auc_higher_predicts_harm(values: list[float], harms: list[bool]) -> float:
    positive = np.asarray([value for value, harm in zip(values, harms) if harm])
    negative = np.asarray([value for value, harm in zip(values, harms) if not harm])
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUC requires both harmful and nonharmful accepted rows")
    greater = (positive[:, None] > negative[None, :]).mean()
    equal = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * equal)


def feature_diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted = [row for row in rows if bool(row["baseline_accepted"])]
    improvements = [
        (float(row["parent"]) - float(row["candidate"]))
        / max(float(row["parent"]), 1.0e-12)
        for row in accepted
    ]
    harms = [value < 0.0 for value in improvements]
    features: dict[str, dict[str, object]] = {}
    for feature in FEATURES:
        values = [float(row[feature]) for row in accepted]
        harmful = [value for value, harm in zip(values, harms) if harm]
        nonharmful = [value for value, harm in zip(values, harms) if not harm]
        auc = _auc_higher_predicts_harm(values, harms)
        features[feature] = {
            "harm_median": float(np.median(harmful)),
            "nonharm_median": float(np.median(nonharmful)),
            "auc_higher_predicts_harm": auc,
            "higher_is_risk": auc >= 0.5,
            "best_direction_auc": max(auc, 1.0 - auc),
        }
    harmful_records = []
    for row, improvement, harm in zip(accepted, improvements, harms):
        if not harm:
            continue
        harmful_records.append(
            {
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "family": row["family"],
                "relative_improvement": improvement,
                "coefficient_l2": row["coefficient_l2"],
                "trust_fraction": row["trust_fraction"],
            }
        )
    harmful_records.sort(key=lambda item: float(item["relative_improvement"]))
    return {
        "accepted_count": len(accepted),
        "strict_harm_count": sum(harms),
        "strict_harm_group_count": len(
            {str(item["group_id"]) for item in harmful_records}
        ),
        "features": features,
        "harmful_records": harmful_records,
    }


def crossfit_guard(
    rows: list[dict[str, object]],
    *,
    fold_count: int,
    seed: int,
    minimum_mean_improvement: float,
    minimum_nonworse_fraction: float,
) -> dict[str, object]:
    if fold_count < 2:
        raise ValueError("cross-fitting requires at least two folds")
    rng = np.random.default_rng(seed)
    group_fold: dict[str, int] = {}
    for family in ALLOWED_MEDIUM_TYPES:
        groups = sorted(
            {str(row["group_id"]) for row in rows if row["family"] == family}
        )
        rng.shuffle(groups)
        for index, group_id in enumerate(groups):
            group_fold[group_id] = index % fold_count

    out_of_fold: dict[str, bool] = {}
    folds: list[dict[str, object]] = []
    for fold in range(fold_count):
        fitting = [
            row for row in rows if group_fold[str(row["group_id"])] != fold
        ]
        holdout = [
            row for row in rows if group_fold[str(row["group_id"])] == fold
        ]
        selected = select_safety_guard(
            fitting,
            minimum_mean_improvement=minimum_mean_improvement,
            minimum_nonworse_fraction=minimum_nonworse_fraction,
        )
        cap = selected["coefficient_l2_cap"]
        floor = selected["trust_fraction_floor"]
        decisions = _guard_decisions(
            holdout,
            coefficient_l2_cap=None if cap is None else float(cap),
            trust_fraction_floor=None if floor is None else float(floor),
        )
        out_of_fold.update(decisions)
        folds.append(
            {
                "fold": fold,
                "fitting_record_count": len(fitting),
                "holdout_record_count": len(holdout),
                "coefficient_l2_cap": cap,
                "trust_fraction_floor": floor,
                "fitting_metrics": selected["metrics"],
                "holdout_metrics": _metrics_for_decisions(holdout, decisions),
            }
        )
    if set(out_of_fold) != {str(row["sample_id"]) for row in rows}:
        raise ValueError("cross-fit coverage is incomplete")
    return {
        "fold_count": fold_count,
        "seed": seed,
        "group_disjoint": True,
        "folds": folds,
        "out_of_fold_metrics": _metrics_for_decisions(rows, out_of_fold),
    }


def analyze(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    floors, checkpoint_sha256 = _load_floors(checkpoint_path)
    calibration_dirs = tuple(
        Path(value).expanduser().resolve() for value in args.calibration_shard_dir
    )
    calibration, calibration_hashes = _load_calibration_rows(
        config_path=config_path,
        shard_dirs=calibration_dirs,
        floors=floors,
    )
    validation, validation_identity = _load_evaluation_rows(
        artifact_dir=Path(args.validation_artifact_dir).expanduser().resolve(),
        shard_subdir="validation_shards",
        expected_split="validation",
        floors=floors,
    )
    test_id, test_identity = _load_evaluation_rows(
        artifact_dir=Path(args.test_artifact_dir).expanduser().resolve(),
        shard_subdir="test_id_shards",
        expected_split="test_id",
        floors=floors,
    )

    baseline_calibration = guard_metrics(calibration)
    selected = select_safety_guard(
        calibration,
        minimum_mean_improvement=args.minimum_mean_improvement,
        minimum_nonworse_fraction=args.minimum_nonworse_fraction,
    )
    cap = float(selected["coefficient_l2_cap"])
    trust_floor = float(selected["trust_fraction_floor"])
    validation_baseline = guard_metrics(validation)
    test_baseline = guard_metrics(test_id)
    validation_guarded = guard_metrics(
        validation,
        coefficient_l2_cap=cap,
        trust_fraction_floor=trust_floor,
    )
    test_guarded = guard_metrics(
        test_id,
        coefficient_l2_cap=cap,
        trust_fraction_floor=trust_floor,
    )
    validation_groups = {str(row["group_id"]) for row in validation}
    test_groups = {str(row["group_id"]) for row in test_id}

    return {
        "schema": "cpadc_posthoc_acceptance_risk_diagnostic_v1",
        "status": "complete",
        "claim_authorized": False,
        "policy_status": "candidate_only_do_not_deploy",
        "reason": (
            "feature hypothesis was formed after test_id outcomes were inspected"
        ),
        "required_confirmation": (
            "preregister and evaluate once on a newly generated group-disjoint holdout"
        ),
        "input_identity": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "calibration_summary_sha256": calibration_hashes,
            "validation": validation_identity,
            "test_id": test_identity,
        },
        "protocol": {
            "threshold_fitting_split": "disjoint_train_calibration_only",
            "retrospective_splits": ["validation", "test_id"],
            "retrospective_results_are_confirmatory": False,
            "online_features_only": True,
            "validation_test_group_overlap_count": len(
                validation_groups & test_groups
            ),
            "strength_floor_by_family": floors,
        },
        "feature_diagnostics": {
            "train_calibration": feature_diagnostics(calibration),
            "test_id_posthoc": feature_diagnostics(test_id),
        },
        "candidate_guard": {
            "name": "coefficient_norm_cap_and_trust_floor_v1",
            "coefficient_l2_cap": cap,
            "trust_fraction_floor": trust_floor,
            "selection": selected,
            "train_baseline": baseline_calibration,
            "group_crossfit": crossfit_guard(
                calibration,
                fold_count=args.fold_count,
                seed=args.fold_seed,
                minimum_mean_improvement=args.minimum_mean_improvement,
                minimum_nonworse_fraction=args.minimum_nonworse_fraction,
            ),
            "validation_retrospective": {
                "baseline": validation_baseline,
                "guarded": validation_guarded,
            },
            "test_id_retrospective": {
                "baseline": test_baseline,
                "guarded": test_guarded,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-shard-dir", action="append", required=True)
    parser.add_argument("--validation-artifact-dir", required=True)
    parser.add_argument("--test-artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260810)
    parser.add_argument("--minimum-mean-improvement", type=float, default=0.01)
    parser.add_argument("--minimum-nonworse-fraction", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"status": result["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
