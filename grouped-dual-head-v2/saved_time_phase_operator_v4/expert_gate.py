"""Pure evidence gate for velocity-routed family expert continuations."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


FAMILIES = ("uniform", "layered", "marmousi")
HIGH_BAND_NUMERICAL_TOLERANCE = 2.0e-5


def _metric_bundle(
    name: str, raw: Mapping[str, object]
) -> tuple[float, dict[str, float], tuple[str, ...]]:
    try:
        aggregate = float(raw["aggregate_relative_l2"])
        family_raw = raw["family_relative_l2"]
        if not isinstance(family_raw, Mapping):
            raise TypeError("family metrics are not a mapping")
        family = {key: float(family_raw[key]) for key in FAMILIES}
        source_raw = raw["source_relative_l2"]
        if not isinstance(source_raw, Mapping) or not source_raw:
            raise TypeError("source metrics are not a populated mapping")
        source = {str(key): float(value) for key, value in source_raw.items()}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"family expert gate {name} metrics are malformed") from error
    if aggregate < 0.0 or not all(
        math.isfinite(value) and value >= 0.0
        for value in (aggregate, *family.values(), *source.values())
    ):
        raise ValueError(f"family expert gate {name} metrics must be finite")
    return aggregate, family, tuple(sorted(source))


def _better_than_coarse(candidate: Mapping[str, object], aggregate: float) -> bool:
    if "relative_improvement_vs_coarse" in candidate:
        improvement = float(candidate["relative_improvement_vs_coarse"])
        if not math.isfinite(improvement):
            raise ValueError("family expert gate coarse improvement must be finite")
        return improvement > 0.0
    try:
        coarse = float(candidate["coarse_aggregate_relative_l2"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("family expert gate candidate lacks coarse evidence") from error
    if not math.isfinite(coarse) or coarse < 0.0:
        raise ValueError("family expert gate coarse metric must be finite")
    return aggregate < coarse


def family_expert_gate(
    *,
    direct_parent: Mapping[str, object],
    global_parent: Mapping[str, object],
    candidate: Mapping[str, object],
    overfit_reduction: float,
    router_accuracy: float,
    route_probabilities: Sequence[float],
    high_band_parent: float,
    high_band_candidate: float,
    peak_cuda_bytes: int,
    physical_microbatch: int,
    minimum_physical_microbatch: int,
    effective_batch: int,
    coarse_anchor_required: bool,
) -> dict[str, object]:
    """Apply accuracy, routing, resource and throughput admission checks."""

    if not all(isinstance(value, Mapping) for value in (direct_parent, global_parent, candidate)):
        raise ValueError("family expert gate metric bundles must be mappings")
    direct_aggregate, _, direct_panel = _metric_bundle("direct parent", direct_parent)
    global_aggregate, global_family, global_panel = _metric_bundle(
        "global parent", global_parent
    )
    candidate_aggregate, candidate_family, candidate_panel = _metric_bundle(
        "candidate", candidate
    )
    try:
        overfit = float(overfit_reduction)
        accuracy = float(router_accuracy)
        probabilities = tuple(float(value) for value in route_probabilities)
        parent_high = float(high_band_parent)
        candidate_high = float(high_band_candidate)
        peak_bytes = int(peak_cuda_bytes)
        microbatch = int(physical_microbatch)
        minimum_microbatch = int(minimum_physical_microbatch)
        batch = int(effective_batch)
    except (TypeError, ValueError) as error:
        raise ValueError("family expert gate scalar evidence is malformed") from error
    scalar_values = (overfit, accuracy, parent_high, candidate_high, *probabilities)
    if (
        len(probabilities) != len(FAMILIES)
        or not all(math.isfinite(value) for value in scalar_values)
        or overfit < 0.0
        or not 0.0 <= accuracy <= 1.0
        or any(not 0.0 <= value <= 1.0 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-4)
        or parent_high < 0.0
        or candidate_high < 0.0
        or peak_bytes < 0
        or microbatch <= 0
        or minimum_microbatch <= 0
        or batch <= 0
        or not isinstance(coarse_anchor_required, bool)
    ):
        raise ValueError("family expert gate scalar evidence is invalid")

    direct_improvement = (direct_aggregate - candidate_aggregate) / max(
        direct_aggregate, 1.0e-16
    )
    global_improvement = (global_aggregate - candidate_aggregate) / max(
        global_aggregate, 1.0e-16
    )
    direct_same_panel = direct_panel == candidate_panel
    global_same_panel = global_panel == candidate_panel
    checks = {
        "direct_parent_same_panel": direct_same_panel,
        "global_parent_same_panel": global_same_panel,
        "overfit_reduction_20pct": overfit >= 0.20,
        "direct_parent_improvement_1pct": direct_same_panel
        and candidate_aggregate <= 0.99 * direct_aggregate,
        "global_parent_improvement_1pct": global_same_panel
        and candidate_aggregate <= 0.99 * global_aggregate,
        "families_safe": global_same_panel
        and all(
            candidate_family[family] <= global_family[family] + 0.01
            for family in FAMILIES
        ),
        "high_band_safe": global_same_panel
        and candidate_high <= parent_high + HIGH_BAND_NUMERICAL_TOLERANCE,
        # A jointly trained source/fusion/geometry/backbone changes the coarse
        # output itself, so only a head-only expert continuation has a fixed
        # coarse anchor that can be used as an admission requirement.
        "better_than_coarse": (not coarse_anchor_required)
        or _better_than_coarse(candidate, candidate_aggregate),
        "router_accuracy_95pct": accuracy >= 0.95,
        "no_route_collapse": all(value >= 0.10 for value in probabilities),
        "cuda_peak_safe": peak_bytes < 23 * 1024**3,
        "physical_microbatch_stage_safe": microbatch >= minimum_microbatch,
        "effective_batch_96": batch == 96,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_overfit_reduction": 0.20,
            "minimum_parent_relative_improvement": 0.01,
            "family_regression_tolerance": 0.01,
            "minimum_router_accuracy": 0.95,
            "minimum_route_probability": 0.10,
            "maximum_peak_cuda_gib": 23.0,
            "high_band_numerical_tolerance": HIGH_BAND_NUMERICAL_TOLERANCE,
            "minimum_physical_microbatch": minimum_microbatch,
        },
        "direct_parent": {"aggregate_relative_l2": direct_aggregate},
        "global_parent": {
            "aggregate_relative_l2": global_aggregate,
            "family_relative_l2": global_family,
            "high_band_relative_l2": parent_high,
        },
        "candidate": {
            "aggregate_relative_l2": candidate_aggregate,
            "family_relative_l2": candidate_family,
            "high_band_relative_l2": candidate_high,
            "relative_improvement_vs_direct_parent": direct_improvement,
            "relative_improvement_vs_global_parent": global_improvement,
            "overfit_reduction": overfit,
            "router_accuracy": accuracy,
            "route_probabilities": dict(zip(FAMILIES, probabilities, strict=True)),
            "peak_cuda_bytes": peak_bytes,
            "physical_microbatch_records": microbatch,
            "coarse_anchor_required": coarse_anchor_required,
            "effective_batch": batch,
        },
    }


__all__ = [
    "FAMILIES",
    "HIGH_BAND_NUMERICAL_TOLERANCE",
    "family_expert_gate",
]
