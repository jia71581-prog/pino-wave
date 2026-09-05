#!/usr/bin/env python3
"""Paired comparison on the sealed fixed-19-Hz source-position control.

The velocity slice, not each of its eight source positions, is the independent
unit.  Positive oriented differences always favour the candidate method.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from saved_time_phase_operator_v4.evaluation import sha256_file


SCORE_SCHEMA = "marmousi_fixed_frequency_source_position_scores_v1"
COMPARISON_SCHEMA = "paired_marmousi_fixed19hz_source_position_comparison_v1"

# The first three endpoints form the preregistered accuracy claim family.
PRIMARY_METRICS: tuple[str, ...] = (
    "record_relative_l2",
    "late_relative_l2",
    "spectrum_high_relative_l2",
)
METRIC_DIRECTIONS: dict[str, str] = {
    "record_relative_l2": "lower",
    "late_relative_l2": "lower",
    "spectrum_high_relative_l2": "lower",
    "receiver_relative_l2": "lower",
    "receiver_lag_abs_s": "lower",
    "receiver_phase_error": "lower",
    "receiver_phase_coherence": "higher",
    "receiver_xcorr_peak": "higher",
    "arrival_mae_s": "lower",
    "arrival_miss_rate": "lower",
    "late_error_slope_per_s": "lower",
    "komega_relative_l2": "lower",
    "komega_high": "lower",
    "komega_relative_l2_late": "lower",
    "komega_high_late": "lower",
}


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _load_score(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if payload.get("schema") != SCORE_SCHEMA or payload.get("status") != "complete":
        raise ValueError(f"score is not a complete position-control artifact: {path}")
    if payload.get("source_generalization_variable") != "position_only":
        raise ValueError("score is not restricted to source-position generalization")
    if payload.get("frequency_generalization_claim_permitted") is not False:
        raise ValueError("score does not explicitly forbid a frequency-generalization claim")
    if not math.isclose(float(payload.get("fixed_source_frequency_hz")), 19.0):
        raise ValueError("score is not the fixed-19-Hz control")
    return payload


def _record_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("records", [])
    mapping = {str(row["record_id"]): row for row in rows}
    if len(mapping) != len(rows) or len(mapping) != 240:
        raise ValueError("score must contain exactly 240 unique locked records")
    return mapping


def _verify_pair(candidate: Mapping[str, Any], comparator: Mapping[str, Any]) -> None:
    for name in ("protocol_sha256", "reference_manifest_sha256"):
        if candidate.get(name) != comparator.get(name):
            raise ValueError(f"candidate and comparator have different {name}")
    left, right = _record_map(candidate), _record_map(comparator)
    if set(left) != set(right):
        raise ValueError("candidate and comparator record identifiers differ")
    for record_id in sorted(left):
        for name in ("slice_rank", "case_id", "role", "source_parameters"):
            if left[record_id].get(name) != right[record_id].get(name):
                raise ValueError(f"paired record metadata differ for {record_id}: {name}")
        if sorted(left[record_id]["metrics"]) != sorted(right[record_id]["metrics"]):
            raise ValueError(f"paired metric keys differ for {record_id}")


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: (float(p_values[name]), name))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(p_values[name])))
        adjusted[name] = running
    return adjusted


def compare_scores(
    candidate_path: Path,
    comparator_path: Path,
    *,
    repetitions: int = 100_000,
    seed: int = 20260813,
) -> dict[str, Any]:
    """Return slice-clustered paired effects; positive effects favour candidate."""

    if repetitions < 100:
        raise ValueError("paired inference requires at least 100 repetitions")
    candidate = _load_score(candidate_path)
    comparator = _load_score(comparator_path)
    _verify_pair(candidate, comparator)
    candidate_rows = _record_map(candidate)
    comparator_rows = _record_map(comparator)
    ranks = sorted({int(row["slice_rank"]) for row in candidate_rows.values()})
    if ranks != list(range(1, 31)):
        raise ValueError("paired comparison requires all 30 locked velocity slices")
    available_metrics = sorted(candidate_rows[next(iter(candidate_rows))]["metrics"])
    metric_names = sorted(set(available_metrics) & set(METRIC_DIRECTIONS))
    excluded_non_directional_metrics = sorted(set(available_metrics) - set(metric_names))

    rng = np.random.default_rng(int(seed))
    cluster_draw = rng.integers(0, len(ranks), size=(int(repetitions), len(ranks)))
    results: dict[str, Any] = {}
    raw_primary_p: dict[str, float] = {}
    for metric in metric_names:
        direction = METRIC_DIRECTIONS[metric]
        slice_effects: list[float] = []
        role_effects: dict[str, list[float]] = {}
        for rank in ranks:
            ids = sorted(
                record_id
                for record_id, row in candidate_rows.items()
                if int(row["slice_rank"]) == rank
            )
            if len(ids) != 8:
                raise ValueError(f"slice {rank} does not contain eight paired positions")
            effects: list[float] = []
            for record_id in ids:
                candidate_value = float(candidate_rows[record_id]["metrics"][metric])
                comparator_value = float(comparator_rows[record_id]["metrics"][metric])
                if not math.isfinite(candidate_value) or not math.isfinite(comparator_value):
                    raise ValueError(f"non-finite paired metric for {record_id}: {metric}")
                effect = (
                    comparator_value - candidate_value
                    if direction == "lower"
                    else candidate_value - comparator_value
                )
                effects.append(effect)
                role = str(candidate_rows[record_id]["role"])
                role_effects.setdefault(role, []).append(effect)
            slice_effects.append(float(np.mean(effects)))
        values = np.asarray(slice_effects, dtype=np.float64)
        bootstrap = values[cluster_draw].mean(axis=1)
        # Paired sign-flip randomization at the independent slice level.
        signs = rng.integers(0, 2, size=(int(repetitions), len(ranks)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null_means = (signs * values[None]).mean(axis=1)
        one_sided_p = float((1 + np.count_nonzero(null_means >= values.mean())) / (repetitions + 1))
        results[metric] = {
            "direction": direction,
            "oriented_effect_definition": (
                "comparator_minus_candidate" if direction == "lower" else "candidate_minus_comparator"
            ),
            "mean_candidate_improvement": float(values.mean()),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "one_sided_p": one_sided_p,
            "slice_effects": [float(value) for value in values],
            "mean_effect_by_position_role": {
                role: float(np.mean(role_values)) for role, role_values in sorted(role_effects.items())
            },
        }
        if metric in PRIMARY_METRICS:
            raw_primary_p[metric] = one_sided_p

    missing_primary = sorted(set(PRIMARY_METRICS) - set(results))
    if missing_primary:
        raise ValueError(f"required primary metrics are absent: {missing_primary}")
    adjusted = _holm_adjust(raw_primary_p)
    for metric, value in adjusted.items():
        results[metric]["holm_adjusted_one_sided_p"] = float(value)
    accuracy_gate = all(
        results[metric]["ci95_low"] > 0.0
        and results[metric]["holm_adjusted_one_sided_p"] < 0.05
        and all(value > 0.0 for value in results[metric]["mean_effect_by_position_role"].values())
        for metric in PRIMARY_METRICS
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "complete",
        "candidate_score": str(candidate_path.resolve()),
        "candidate_score_sha256": sha256_file(candidate_path),
        "comparator_score": str(comparator_path.resolve()),
        "comparator_score_sha256": sha256_file(comparator_path),
        "protocol_sha256": candidate["protocol_sha256"],
        "reference_manifest_sha256": candidate["reference_manifest_sha256"],
        "fixed_source_frequency_hz": 19.0,
        "source_generalization_variable": "position_only",
        "frequency_generalization_claim_permitted": False,
        "independent_unit": "velocity_slice",
        "independent_velocity_slice_count": 30,
        "repeated_source_positions_per_slice": 8,
        "bootstrap_repetitions": int(repetitions),
        "randomization_repetitions": int(repetitions),
        "seed": int(seed),
        "primary_metrics": list(PRIMARY_METRICS),
        "excluded_non_directional_metrics": excluded_non_directional_metrics,
        "multiplicity": "Holm correction across three prespecified primary accuracy metrics",
        "accuracy_superiority_gate_passed": bool(accuracy_gate),
        "claim_boundary": (
            "accuracy comparison applies only to the tested methods on the fixed-19-Hz "
            "source-position protocol; it does not establish frequency generalization"
        ),
        "metrics": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-score", type=Path, required=True)
    parser.add_argument("--comparator-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    payload = compare_scores(
        args.candidate_score,
        args.comparator_score,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["compare_scores", "METRIC_DIRECTIONS", "PRIMARY_METRICS"]
