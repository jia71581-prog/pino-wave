from __future__ import annotations

import numpy as np

from tgrs_dclp_no.statistics import (
    dispersion_claim_gate,
    holm_adjusted_pvalues,
    paired_group_bootstrap,
)


def test_paired_bootstrap_detects_clear_improvement():
    baseline = np.arange(30, dtype=np.float64) + 10.0
    proposed = baseline - 2.0
    groups = np.asarray([f"g{index}" for index in range(30)])
    result = paired_group_bootstrap(proposed, baseline, groups, replicates=2000, seed=372)
    assert result["mean_difference"] < 0.0
    assert result["ci95_high"] < 0.0


def test_paired_bootstrap_no_improvement_straddles_zero():
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=40)
    proposed = baseline + rng.normal(scale=1.0e-3, size=40)  # essentially identical
    groups = np.asarray([f"g{index}" for index in range(40)])
    result = paired_group_bootstrap(proposed, baseline, groups, replicates=2000, seed=372)
    assert result["ci95_low"] < 0.0 < result["ci95_high"]


def test_dispersion_gate_requires_two_field_metrics_and_receiver_agreement():
    passed = dispersion_claim_gate(
        {
            "wavefront_radius_error": {"ci95_high": -0.1},
            "komega_ridge_error": {"ci95_high": -0.2},
            "receiver_lag": {"ci95_high": -0.01},
        }
    )
    failed = dispersion_claim_gate(
        {
            "wavefront_radius_error": {"ci95_high": -0.1},
            "komega_ridge_error": {"ci95_high": 0.2},  # no longer improving
            "receiver_lag": {"ci95_high": -0.01},
        }
    )
    assert passed["dispersion_suppression_supported"]
    assert passed["permitted_title_prefix"] == "Dispersion-Controlled"
    assert not failed["dispersion_suppression_supported"]
    assert failed["permitted_title_prefix"] == "Physics-Conditioned"


def test_gate_requires_receiver_metric_even_with_many_field_passes():
    result = dispersion_claim_gate(
        {
            "wavefront_radius_error": {"ci95_high": -0.1},
            "komega_ridge_error": {"ci95_high": -0.2},
            "spectrum_high_error": {"ci95_high": -0.05},
            # no receiver metric supplied
        }
    )
    assert result["field_pass_count"] == 3
    assert result["receiver_pass_count"] == 0
    assert not result["dispersion_suppression_supported"]


def test_holm_adjustment_respects_alpha_in_gate():
    # Two field metrics with tiny CIs but large p-values must NOT pass once Holm-adjusted.
    result = dispersion_claim_gate(
        {
            "wavefront_radius_error": {"ci95_high": -0.1, "pvalue": 0.30},
            "komega_ridge_error": {"ci95_high": -0.1, "pvalue": 0.40},
            "receiver_lag": {"ci95_high": -0.01, "pvalue": 0.001},
        }
    )
    assert result["field_pass_count"] == 0
    assert not result["dispersion_suppression_supported"]


def test_holm_is_monotonic_and_bounded():
    adjusted = holm_adjusted_pvalues({"a": 0.01, "b": 0.02, "c": 0.9})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0.0 <= v <= 1.0 for v in adjusted.values())
