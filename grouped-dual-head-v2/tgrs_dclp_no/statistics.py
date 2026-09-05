"""Paired grouped bootstrap and the machine-readable dispersion claim gate.

All comparison metrics are *errors* (lower is better), so a paired difference
``proposed - baseline`` whose 95% bootstrap interval lies entirely below zero is a
significant improvement.  The claim gate counts significant improvements among the
registered field and receiver metrics and decides whether a "dispersion suppression"
claim is supportable, along with the permitted title prefix and forbidden wording.
"""
from __future__ import annotations

import numpy as np


FIELD_METRICS = (
    "wavefront_radius_error",
    "zero_crossing_error",
    "dominant_wavenumber_error",
    "komega_ridge_error",
    "post_front_ringing_ratio",
    "spectrum_high_error",
)
RECEIVER_METRICS = (
    "receiver_lag",
    "receiver_phase_residual",
    "receiver_coherence_error",
)


def paired_group_bootstrap(proposed, baseline, group_ids, *, replicates: int, seed: int) -> dict:
    """Grouped paired bootstrap of the mean error difference ``proposed - baseline``.

    Resamples whole groups (with replacement) so records sharing a group are not treated
    as independent.  Returns the observed mean difference and its 95% percentile interval.
    """
    proposed = np.asarray(proposed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=str)
    if proposed.shape != baseline.shape or proposed.shape != groups.shape:
        raise ValueError("paired arrays and group ids must share one shape")
    if proposed.ndim != 1 or proposed.size == 0:
        raise ValueError("paired arrays must be non-empty 1-D")
    if not np.isfinite(proposed).all() or not np.isfinite(baseline).all():
        raise ValueError("paired arrays must be finite")
    unique = np.unique(groups)
    rng = np.random.default_rng(int(seed))
    observed = float(np.mean(proposed - baseline))
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    draws = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([by_group[group] for group in sampled])
        draws[index] = float(np.mean(proposed[positions] - baseline[positions]))
    return {
        "mean_difference": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def holm_adjusted_pvalues(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment of a name->p-value mapping."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    total = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, raw) in enumerate(items):
        value = min(1.0, float(raw) * (total - rank))
        running = max(running, value)  # enforce monotonic non-decreasing
        adjusted[name] = running
    return adjusted


def _metric_passed(entry: dict, *, alpha: float) -> bool:
    """A metric improves if its bootstrap CI upper bound is below zero and, when a
    (Holm-adjusted) p-value is registered, it is below ``alpha``."""
    if float(entry.get("ci95_high", 0.0)) >= 0.0:
        return False
    adjusted = entry.get("holm_pvalue", entry.get("pvalue"))
    if adjusted is not None and float(adjusted) >= float(alpha):
        return False
    return True


def dispersion_claim_gate(results: dict[str, dict], *, alpha: float = 0.05) -> dict:
    """Decide whether the dispersion-suppression claim is supportable.

    ``results`` maps metric names (from FIELD_METRICS / RECEIVER_METRICS) to entries with
    at least ``ci95_high`` and optionally ``pvalue``.  If any ``pvalue`` values are present
    they are Holm-adjusted across all supplied metrics before the per-metric decision.
    """
    working = {name: dict(entry) for name, entry in results.items()}
    raw_pvalues = {name: entry["pvalue"] for name, entry in working.items() if "pvalue" in entry}
    if raw_pvalues:
        adjusted = holm_adjusted_pvalues(raw_pvalues)
        for name, value in adjusted.items():
            working[name]["holm_pvalue"] = value

    field_pass_count = sum(
        1 for name in FIELD_METRICS if name in working and _metric_passed(working[name], alpha=alpha)
    )
    receiver_pass_count = sum(
        1 for name in RECEIVER_METRICS if name in working and _metric_passed(working[name], alpha=alpha)
    )
    supported = field_pass_count >= 2 and receiver_pass_count >= 1
    return {
        "schema": "dclp_no_claim_gate_v1",
        "dispersion_suppression_supported": bool(supported),
        "field_pass_count": int(field_pass_count),
        "receiver_pass_count": int(receiver_pass_count),
        "alpha": float(alpha),
        "permitted_title_prefix": "Dispersion-Controlled" if supported else "Physics-Conditioned",
        "forbidden_phrases": ["dispersion-free", "eliminates numerical dispersion"],
    }
