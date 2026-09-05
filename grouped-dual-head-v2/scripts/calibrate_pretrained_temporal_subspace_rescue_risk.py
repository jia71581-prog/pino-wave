#!/usr/bin/env python3
"""Fit a train-only linear rescue branch on top of a frozen PTLSA norm gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_v5_instance_adaptation import write_report
from scripts.run_pretrained_temporal_subspace_adaptation import (
    PTLSA_RESCUE_FEATURES,
    PTLSA_RESCUE_MODEL_SCHEMA,
    PTLSA_RESCUE_RISK_SCHEMA,
    PTLSA_RISK_SCHEMA,
    PTLSA_SCHEMA,
)
from scripts.train_v5_residual_meta import _sha256


def _load_raw_rows(
    run_dirs: tuple[Path, ...],
    *,
    family: str,
    role: str,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    rows: list[dict[str, object]] = []
    parent_identity: dict[str, object] | None = None
    method_identity: dict[str, object] | None = None
    seen: set[str] = set()
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("schema") != PTLSA_SCHEMA or summary.get("selection_split") != "train":
            raise ValueError(f"rescue calibration requires train-only PTLSA: {summary_path}")
        shard_parent = dict(summary["parent"])
        shard_method = dict(summary.get("method") or {})
        if shard_method.get("risk_calibration") is not None:
            raise ValueError("rescue calibration requires ungated raw PTLSA runs")
        comparable_method = {
            key: value for key, value in shard_method.items() if key != "config_sha256"
        }
        if parent_identity is None:
            parent_identity = shard_parent
            method_identity = comparable_method
        elif shard_parent != parent_identity or comparable_method != method_identity:
            raise ValueError("rescue calibration parent or method identities do not match")
        for raw_report in summary["records"]:
            report = dict(raw_report)
            if str(report["medium_type"]) != str(family):
                continue
            sample_id = str(report["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate rescue calibration sample: {sample_id}")
            seen.add(sample_id)
            adaptation = dict(report["adaptation"])
            if bool(adaptation.get("future_truth_used", True)):
                raise ValueError(f"future truth was used online for {sample_id}")
            if float(adaptation.get("observed_probe_weight", 0.0)) != 100.0:
                raise ValueError("rescue calibration requires the frozen weight-100 probe")
            artifact = run_dir / sample_id / "adaptation.pt"
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            sealed = dict(payload.get("adaptation") or {})
            solver_coefficients = torch.as_tensor(
                sealed.get("solver_coefficients", sealed["coefficients"])
            ).double()
            coefficient_norm = float(solver_coefficients.norm())
            objective_before = float(sealed["objective_before"])
            objective_after = float(sealed["objective_after"])
            objective_improvement = (objective_before - objective_after) / max(
                abs(objective_before), 1.0e-12
            )
            parent_error = float(report["parent_future_fullfield_relative_l2"])
            adapted_error = float(report["future_fullfield_relative_l2"])
            rows.append(
                {
                    "role": role,
                    "sample_id": sample_id,
                    "features": {
                        PTLSA_RESCUE_FEATURES[0]: coefficient_norm,
                        PTLSA_RESCUE_FEATURES[1]: objective_improvement,
                    },
                    "parent_error": parent_error,
                    "adapted_error": adapted_error,
                    "relative_improvement": (
                        (parent_error - adapted_error) / max(parent_error, 1.0e-12)
                    ),
                    "artifact_sha256": _sha256(artifact),
                }
            )
    if not rows or parent_identity is None or method_identity is None:
        raise ValueError(f"{role} rescue selection is empty")
    return rows, parent_identity, method_identity


def _fit_ridge(
    rows: list[dict[str, object]], *, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [float(dict(row["features"])[name]) for name in PTLSA_RESCUE_FEATURES]
            for row in rows
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [float(row["relative_improvement"]) for row in rows], dtype=torch.float64
    )
    means = features.mean(dim=0)
    scales = features.std(dim=0, unbiased=False)
    if not torch.isfinite(scales).all() or bool((scales <= 1.0e-12).any()):
        raise ValueError("rescue feature scale is degenerate")
    standardized = (features - means) / scales
    design = torch.cat(
        (torch.ones((len(rows), 1), dtype=torch.float64), standardized), dim=1
    )
    regularizer = torch.diag(
        torch.tensor((0.0,) + (float(alpha),) * len(PTLSA_RESCUE_FEATURES))
    ).double()
    weights = torch.linalg.solve(
        design.T @ design + regularizer, design.T @ target
    )
    if not torch.isfinite(weights).all():
        raise ValueError("rescue ridge fit is nonfinite")
    return means, scales, weights


def _score_rows(
    rows: list[dict[str, object]],
    *,
    means: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
) -> list[float]:
    features = torch.tensor(
        [
            [float(dict(row["features"])[name]) for name in PTLSA_RESCUE_FEATURES]
            for row in rows
        ],
        dtype=torch.float64,
    )
    standardized = (features - means) / scales
    design = torch.cat(
        (torch.ones((len(rows), 1), dtype=torch.float64), standardized), dim=1
    )
    return [float(value) for value in design @ weights]


def _metrics(
    rows: list[dict[str, object]],
    scores: list[float],
    *,
    base_threshold: float,
    rescue_threshold: float | None,
) -> dict[str, object]:
    base = [
        float(dict(row["features"])[PTLSA_RESCUE_FEATURES[0]]) <= base_threshold
        for row in rows
    ]
    rescue = [
        (not base_safe)
        and rescue_threshold is not None
        and score >= rescue_threshold
        for base_safe, score in zip(base, scores)
    ]
    accepted = [left or right for left, right in zip(base, rescue)]
    improvements = [float(row["relative_improvement"]) for row in rows]
    gated = [value if keep else 0.0 for value, keep in zip(improvements, accepted)]
    parent_mean = sum(float(row["parent_error"]) for row in rows) / len(rows)
    adapted_mean = sum(
        float(row["adapted_error"]) if keep else float(row["parent_error"])
        for row, keep in zip(rows, accepted)
    ) / len(rows)
    return {
        "record_count": len(rows),
        "accepted_count": sum(accepted),
        "acceptance_fraction": sum(accepted) / len(rows),
        "base_accepted_count": sum(base),
        "rescued_count": sum(rescue),
        "base_regression_count": sum(
            keep and value < -1.0e-9 for keep, value in zip(base, improvements)
        ),
        "added_rescue_regression_count": sum(
            keep and value < -1.0e-9 for keep, value in zip(rescue, improvements)
        ),
        "nonworse_fraction": sum(value >= -1.0e-9 for value in gated) / len(rows),
        "mean_per_record_relative_improvement": sum(gated) / len(rows),
        "ratio_of_mean_improvement": (parent_mean - adapted_mean)
        / max(parent_mean, 1.0e-12),
    }


def calibrate(
    *,
    fit_run_dirs: tuple[str | Path, ...],
    calibration_run_dirs: tuple[str | Path, ...],
    base_risk_calibration: str | Path,
    output: str | Path,
    family: str = "layered",
    ridge_alpha: float = 1.0,
    minimum_nonworse_fraction: float = 0.95,
    minimum_rescue_count_per_split: int = 1,
    maximum_added_rescue_regressions_per_split: int = 0,
) -> dict[str, object]:
    fit_dirs = tuple(Path(value).expanduser().resolve() for value in fit_run_dirs)
    calibration_dirs = tuple(
        Path(value).expanduser().resolve() for value in calibration_run_dirs
    )
    if not fit_dirs or not calibration_dirs:
        raise ValueError("fit and calibration PTLSA runs are both required")
    if ridge_alpha <= 0.0:
        raise ValueError("ridge alpha must be positive")
    if not 0.0 <= minimum_nonworse_fraction <= 1.0:
        raise ValueError("minimum nonworse fraction must lie in [0,1]")
    if minimum_rescue_count_per_split < 1:
        raise ValueError("minimum rescue count per split must be positive")
    if maximum_added_rescue_regressions_per_split < 0:
        raise ValueError("maximum added regressions must be nonnegative")

    fit_rows, parent_identity, method_identity = _load_raw_rows(
        fit_dirs, family=family, role="fit"
    )
    calibration_rows, calibration_parent, calibration_method = _load_raw_rows(
        calibration_dirs, family=family, role="calibration"
    )
    if calibration_parent != parent_identity or calibration_method != method_identity:
        raise ValueError("fit and calibration PTLSA identities do not match")
    fit_ids = {str(row["sample_id"]) for row in fit_rows}
    calibration_ids = {str(row["sample_id"]) for row in calibration_rows}
    if fit_ids.intersection(calibration_ids):
        raise ValueError("fit and rescue calibration samples overlap")

    base_path = Path(base_risk_calibration).expanduser().resolve()
    base_payload = json.loads(base_path.read_text())
    if (
        base_payload.get("schema") != PTLSA_RISK_SCHEMA
        or dict(base_payload.get("parent") or {}) != parent_identity
        or str(base_payload.get("family")) != str(family)
    ):
        raise ValueError("base risk calibration identity mismatch")
    base_threshold = float(base_payload.get("maximum_coefficient_l2_norm", 0.0))
    if base_threshold <= 0.0 or not torch.isfinite(torch.tensor(base_threshold)).item():
        raise ValueError("base risk threshold is invalid")

    means, scales, weights = _fit_ridge(fit_rows, alpha=ridge_alpha)
    fit_scores = _score_rows(fit_rows, means=means, scales=scales, weights=weights)
    calibration_scores = _score_rows(
        calibration_rows, means=means, scales=scales, weights=weights
    )
    candidates = sorted(set(fit_scores + calibration_scores), reverse=True)
    eligible: list[dict[str, object]] = []
    for threshold in candidates:
        fit_metrics = _metrics(
            fit_rows,
            fit_scores,
            base_threshold=base_threshold,
            rescue_threshold=threshold,
        )
        calibration_metrics = _metrics(
            calibration_rows,
            calibration_scores,
            base_threshold=base_threshold,
            rescue_threshold=threshold,
        )
        if all(
            int(metrics["rescued_count"]) >= minimum_rescue_count_per_split
            and int(metrics["added_rescue_regression_count"])
            <= maximum_added_rescue_regressions_per_split
            and float(metrics["nonworse_fraction"]) >= minimum_nonworse_fraction
            and float(metrics["mean_per_record_relative_improvement"]) > 0.0
            for metrics in (fit_metrics, calibration_metrics)
        ):
            eligible.append(
                {
                    "threshold": threshold,
                    "fit_metrics": fit_metrics,
                    "calibration_metrics": calibration_metrics,
                }
            )
    if not eligible:
        raise RuntimeError("no linear rescue threshold satisfies the train-only gate")
    selected = max(
        eligible,
        key=lambda value: (
            int(dict(value["fit_metrics"])["rescued_count"])
            + int(dict(value["calibration_metrics"])["rescued_count"]),
            int(dict(value["calibration_metrics"])["rescued_count"]),
            float(dict(value["calibration_metrics"])[
                "mean_per_record_relative_improvement"
            ]),
        ),
    )
    rescue_threshold = float(selected["threshold"])
    all_rows = fit_rows + calibration_rows
    all_scores = fit_scores + calibration_scores
    checkpoint = {
        "schema": PTLSA_RESCUE_RISK_SCHEMA,
        "future_truth_scope": "train_split_after_adaptation_seal_only",
        "family": str(family),
        "record_count": len(all_rows),
        "sample_ids": [str(row["sample_id"]) for row in all_rows],
        "fit_source_run_dirs": [str(value) for value in fit_dirs],
        "calibration_source_run_dirs": [str(value) for value in calibration_dirs],
        "parent": parent_identity,
        "base_method": method_identity,
        "base_gate": {
            "checkpoint": str(base_path),
            "checkpoint_sha256": _sha256(base_path),
            "schema": PTLSA_RISK_SCHEMA,
            "maximum_coefficient_l2_norm": base_threshold,
            "record_count": int(base_payload["record_count"]),
        },
        "rescue_model": {
            "schema": PTLSA_RESCUE_MODEL_SCHEMA,
            "features": list(PTLSA_RESCUE_FEATURES),
            "feature_means": [float(value) for value in means],
            "feature_scales": [float(value) for value in scales],
            "weights": [float(value) for value in weights],
            "minimum_score": rescue_threshold,
            "ridge_alpha": float(ridge_alpha),
        },
        "selection_constraints": {
            "minimum_nonworse_fraction": float(minimum_nonworse_fraction),
            "minimum_rescue_count_per_split": int(minimum_rescue_count_per_split),
            "maximum_added_rescue_regressions_per_split": int(
                maximum_added_rescue_regressions_per_split
            ),
        },
        "fit_metrics": selected["fit_metrics"],
        "calibration_metrics": selected["calibration_metrics"],
        "records": [
            {
                **row,
                "rescue_score": score,
            }
            for row, score in zip(all_rows, all_scores)
        ],
    }
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite a PTLSA rescue risk checkpoint")
    write_report(checkpoint, output_path)
    return checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run-dir", action="append", required=True)
    parser.add_argument("--calibration-run-dir", action="append", required=True)
    parser.add_argument("--base-risk-calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", default="layered")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--minimum-nonworse-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-rescue-count-per-split", type=int, default=1)
    parser.add_argument(
        "--maximum-added-rescue-regressions-per-split", type=int, default=0
    )
    args = parser.parse_args(argv)
    calibrate(
        fit_run_dirs=tuple(args.fit_run_dir),
        calibration_run_dirs=tuple(args.calibration_run_dir),
        base_risk_calibration=args.base_risk_calibration,
        output=args.output,
        family=args.family,
        ridge_alpha=args.ridge_alpha,
        minimum_nonworse_fraction=args.minimum_nonworse_fraction,
        minimum_rescue_count_per_split=args.minimum_rescue_count_per_split,
        maximum_added_rescue_regressions_per_split=(
            args.maximum_added_rescue_regressions_per_split
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
