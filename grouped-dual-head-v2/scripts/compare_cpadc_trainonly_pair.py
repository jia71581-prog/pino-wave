#!/usr/bin/env python3
"""Compare frozen, sample-paired CPADC train-only evaluation summaries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence


FAMILIES = ("uniform", "layered", "marmousi")


def _finite(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _records_by_id(summary: Mapping[str, object], *, arm: str) -> dict[str, dict]:
    if summary.get("selection_split") != "train":
        raise ValueError(f"{arm} summary is not train-only")
    if summary.get("risk_calibration_requested") is not True:
        raise ValueError(f"{arm} summary did not use disjoint train selection")
    raw = summary.get("records")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{arm} summary has no records")
    records: dict[str, dict] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{arm} summary contains a malformed record")
        sample_id = str(item.get("sample_id", ""))
        if not sample_id or sample_id in records:
            raise ValueError(f"{arm} summary has missing or duplicate sample IDs")
        adaptation = item.get("adaptation")
        if not isinstance(adaptation, dict) or adaptation.get("future_truth_used") is not False:
            raise ValueError(f"{arm} adaptation is not sealed from future truth")
        records[sample_id] = item
    return records


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator <= 0.0:
        return None
    return sum(a * b for a, b in zip(centered_left, centered_right, strict=True)) / denominator


def compare_trainonly_pair(
    candidate_summary: Mapping[str, object],
    ablation_summary: Mapping[str, object],
    *,
    minimum_mean_gain_percentage_points: float = 1.0,
    minimum_marmousi_gain_percentage_points: float = 0.5,
    minimum_nonworse_fraction: float = 0.95,
    maximum_p95_adaptation_runtime_s: float = 2.5,
    minimum_effective_design_rank: int = 24,
    maximum_low_rank_fraction: float = 0.05,
    numerical_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Return a fail-closed paired gate without using validation or test_id."""

    candidate_basis = candidate_summary.get("basis")
    ablation_basis = ablation_summary.get("basis")
    if not isinstance(candidate_basis, dict) or not isinstance(ablation_basis, dict):
        raise ValueError("both summaries require basis provenance")
    provenance_keys = (
        "parent_checkpoint_sha256",
        "basis_rank",
        "phase_rank",
        "basis_manifest_digest",
        "evaluation_manifest_digest",
    )
    for key in provenance_keys:
        if candidate_basis.get(key) != ablation_basis.get(key):
            raise ValueError(f"paired basis provenance mismatch: {key}")

    candidate_records = _records_by_id(candidate_summary, arm="candidate")
    ablation_records = _records_by_id(ablation_summary, arm="ablation")
    if set(candidate_records) != set(ablation_records):
        raise ValueError("candidate and ablation sample sets differ")

    rows: list[dict[str, object]] = []
    for sample_id in sorted(candidate_records):
        candidate = candidate_records[sample_id]
        ablation = ablation_records[sample_id]
        family = str(candidate.get("medium_type", ""))
        if family not in FAMILIES or ablation.get("medium_type") != family:
            raise ValueError(f"paired family mismatch for {sample_id}")
        parent = _finite(
            candidate.get("parent_future_fullfield_relative_l2"),
            name=f"candidate parent error {sample_id}",
        )
        ablation_parent = _finite(
            ablation.get("parent_future_fullfield_relative_l2"),
            name=f"ablation parent error {sample_id}",
        )
        if not math.isclose(parent, ablation_parent, rel_tol=0.0, abs_tol=numerical_tolerance):
            raise ValueError(f"paired parent error mismatch for {sample_id}")
        if parent <= 0.0:
            raise ValueError(f"parent error must be positive for {sample_id}")
        candidate_error = _finite(
            candidate.get("future_fullfield_relative_l2"),
            name=f"candidate error {sample_id}",
        )
        ablation_error = _finite(
            ablation.get("future_fullfield_relative_l2"),
            name=f"ablation error {sample_id}",
        )
        adaptation = candidate["adaptation"]
        objective_before = _finite(
            adaptation.get("objective_before"), name=f"objective before {sample_id}"
        )
        objective_after = _finite(
            adaptation.get("objective_after"), name=f"objective after {sample_id}"
        )
        candidate_improvement = (parent - candidate_error) / parent
        ablation_improvement = (parent - ablation_error) / parent
        rows.append(
            {
                "sample_id": sample_id,
                "medium_type": family,
                "parent_relative_l2": parent,
                "candidate_relative_l2": candidate_error,
                "ablation_relative_l2": ablation_error,
                "candidate_improvement": candidate_improvement,
                "ablation_improvement": ablation_improvement,
                "paired_gain_percentage_points": 100.0
                * (candidate_improvement - ablation_improvement),
                "candidate_nonworse": candidate_error
                <= ablation_error + numerical_tolerance,
                "adaptation_elapsed_s": _finite(
                    adaptation.get("adaptation_elapsed_s"),
                    name=f"adaptation runtime {sample_id}",
                ),
                "effective_design_rank": int(adaptation.get("effective_design_rank", -1)),
                "objective_relative_improvement": (
                    objective_before - objective_after
                )
                / max(abs(objective_before), 1.0e-12),
                "future_error_improvement": candidate_improvement,
            }
        )

    family_metrics: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["medium_type"] == family]
        if not selected:
            raise ValueError(f"paired comparison is missing family {family}")
        family_metrics[family] = {
            "record_count": len(selected),
            "mean_paired_gain_percentage_points": fmean(
                float(row["paired_gain_percentage_points"]) for row in selected
            ),
            "candidate_mean_relative_l2": fmean(
                float(row["candidate_relative_l2"]) for row in selected
            ),
            "ablation_mean_relative_l2": fmean(
                float(row["ablation_relative_l2"]) for row in selected
            ),
            "nonworse_fraction": fmean(
                float(bool(row["candidate_nonworse"])) for row in selected
            ),
            "objective_error_pearson": _pearson(
                [float(row["objective_relative_improvement"]) for row in selected],
                [float(row["future_error_improvement"]) for row in selected],
            ),
        }

    mean_gain = fmean(float(row["paired_gain_percentage_points"]) for row in rows)
    nonworse_fraction = fmean(float(bool(row["candidate_nonworse"])) for row in rows)
    p95_runtime = _percentile(
        [float(row["adaptation_elapsed_s"]) for row in rows], 0.95
    )
    low_rank_fraction = fmean(
        float(int(row["effective_design_rank"]) < int(minimum_effective_design_rank))
        for row in rows
    )
    checks = {
        "mean_paired_gain": mean_gain >= float(minimum_mean_gain_percentage_points),
        "marmousi_paired_gain": family_metrics["marmousi"][
            "mean_paired_gain_percentage_points"
        ]
        >= float(minimum_marmousi_gain_percentage_points),
        "nonworse_fraction": nonworse_fraction >= float(minimum_nonworse_fraction),
        "every_family_mean_nonworse": all(
            float(metrics["candidate_mean_relative_l2"])
            <= float(metrics["ablation_mean_relative_l2"]) + numerical_tolerance
            for metrics in family_metrics.values()
        ),
        "p95_adaptation_runtime": p95_runtime
        <= float(maximum_p95_adaptation_runtime_s),
        "effective_design_rank": low_rank_fraction <= float(maximum_low_rank_fraction),
    }
    return {
        "schema": "cpadc_trainonly_paired_gate_v1",
        "passed": all(checks.values()),
        "record_count": len(rows),
        "mean_paired_gain_percentage_points": mean_gain,
        "nonworse_fraction": nonworse_fraction,
        "p95_adaptation_runtime_s": p95_runtime,
        "low_effective_rank_fraction": low_rank_fraction,
        "checks": checks,
        "families": family_metrics,
        "records": rows,
        "future_truth_scope": "disjoint_train_split_only",
        "validation_access": False,
        "test_id_access": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--ablation-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    candidate = json.loads(Path(args.candidate_summary).read_text())
    ablation = json.loads(Path(args.ablation_summary).read_text())
    result = compare_trainonly_pair(candidate, ablation)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: result[key] for key in ("passed", "record_count", "checks")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
