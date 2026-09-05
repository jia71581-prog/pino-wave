from __future__ import annotations

import pytest

from scripts.analyze_cpadc_acceptance_risk import (
    _auc_higher_predicts_harm,
    guard_metrics,
    select_safety_guard,
)


def _row(
    sample_id: str,
    family: str,
    *,
    improvement: float,
    coefficient_l2: float,
    trust_fraction: float,
):
    return {
        "sample_id": sample_id,
        "group_id": sample_id,
        "family": family,
        "parent": 1.0,
        "candidate": 1.0 - improvement,
        "baseline_output": 1.0 - improvement,
        "baseline_accepted": True,
        "coefficient_l2": coefficient_l2,
        "trust_fraction": trust_fraction,
    }


def test_auc_reports_the_direction_of_higher_risk():
    assert _auc_higher_predicts_harm([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert _auc_higher_predicts_harm([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_guard_restores_parent_for_screened_corrections():
    rows = [
        _row("good", "uniform", improvement=0.20, coefficient_l2=0.10, trust_fraction=0.92),
        _row("bad", "uniform", improvement=-0.10, coefficient_l2=0.20, trust_fraction=0.90),
    ]
    baseline = guard_metrics(rows)
    guarded = guard_metrics(
        rows, coefficient_l2_cap=0.15, trust_fraction_floor=0.91
    )
    assert baseline["strict_harm_count"] == 1
    assert guarded["strict_harm_count"] == 0
    assert guarded["accepted_count"] == 1
    assert guarded["mean_record_improvement"] == pytest.approx(0.10)


def test_safety_first_guard_is_fit_without_losing_required_mean_gain():
    rows = []
    for family in ("uniform", "layered", "marmousi"):
        rows.extend(
            [
                _row(
                    f"{family}-good-a",
                    family,
                    improvement=0.08,
                    coefficient_l2=0.08,
                    trust_fraction=0.93,
                ),
                _row(
                    f"{family}-good-b",
                    family,
                    improvement=0.04,
                    coefficient_l2=0.10,
                    trust_fraction=0.92,
                ),
                _row(
                    f"{family}-bad",
                    family,
                    improvement=-0.03,
                    coefficient_l2=0.20,
                    trust_fraction=0.90,
                ),
            ]
        )
    selected = select_safety_guard(rows)
    assert selected["metrics"]["strict_harm_count"] == 0
    assert selected["metrics"]["mean_record_improvement"] >= 0.01
    assert selected["coefficient_l2_cap"] <= 0.10
    assert selected["trust_fraction_floor"] >= 0.92
