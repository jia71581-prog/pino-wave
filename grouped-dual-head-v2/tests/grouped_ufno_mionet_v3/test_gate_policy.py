from __future__ import annotations

import pytest

from grouped_ufno_mionet_v3.training.gates import (
    NineRecordGateMetrics,
    OneRecordGateMetrics,
    evaluate_nine_record_gate,
    evaluate_one_record_gate,
    require_one_record_gate,
)


def _passing(**updates) -> OneRecordGateMetrics:
    values = dict(
        sample_id="train_uniform_00370",
        medium_type="uniform",
        early_relative_l2=0.04,
        middle_relative_l2=0.04,
        late_relative_l2=0.04,
        query_relative_l2=0.04,
        radial_centroid_displacement_m=9.0,
        grid_spacing_m=10.0,
        zero_prediction_relative_l2=1.0,
        missing_gradient_groups=(),
    )
    values.update(updates)
    return OneRecordGateMetrics(**values)


def test_one_record_gate_requires_every_independent_threshold():
    decision = evaluate_one_record_gate(_passing())
    assert decision.passed
    assert decision.failures == ()
    for field in (
        "early_relative_l2",
        "middle_relative_l2",
        "late_relative_l2",
        "query_relative_l2",
    ):
        failed = evaluate_one_record_gate(_passing(**{field: 0.05}))
        assert not failed.passed
        assert any(field in failure for failure in failed.failures)
    centroid = evaluate_one_record_gate(_passing(radial_centroid_displacement_m=10.0))
    assert not centroid.passed
    assert any("centroid" in failure for failure in centroid.failures)


def test_one_record_gate_rejects_anomaly_zero_baseline_and_missing_gradients():
    with pytest.raises(ValueError, match="anomaly"):
        evaluate_one_record_gate(_passing(medium_type="anomaly"))
    baseline = evaluate_one_record_gate(_passing(query_relative_l2=1.0))
    assert not baseline.passed
    missing = evaluate_one_record_gate(_passing(missing_gradient_groups=("dense_spectral",)))
    assert not missing.passed
    assert any("gradient" in failure for failure in missing.failures)


def test_failed_one_record_gate_blocks_next_stage():
    with pytest.raises(RuntimeError, match="one-record gate"):
        require_one_record_gate(_passing(late_relative_l2=0.2))
    require_one_record_gate(_passing())


def _passing_nine(**updates) -> NineRecordGateMetrics:
    values = dict(
        record_count_by_family={"uniform": 3, "layered": 3, "marmousi": 3},
        aggregate_query_relative_l2=0.08,
        aggregate_dense_relative_l2=0.08,
        family_query_relative_l2={"uniform": 0.08, "layered": 0.08, "marmousi": 0.08},
        family_dense_relative_l2={"uniform": 0.08, "layered": 0.08, "marmousi": 0.08},
        family_late_relative_l2={"uniform": 0.08, "layered": 0.08, "marmousi": 0.08},
        zero_prediction_relative_l2=1.0,
        missing_gradient_groups=(),
    )
    values.update(updates)
    return NineRecordGateMetrics(**values)


def test_nine_record_gate_requires_three_records_and_every_family_metric():
    assert evaluate_nine_record_gate(_passing_nine()).passed
    bad_count = evaluate_nine_record_gate(
        _passing_nine(record_count_by_family={"uniform": 3, "layered": 3, "marmousi": 2})
    )
    assert not bad_count.passed and any("count" in item for item in bad_count.failures)
    bad_aggregate = evaluate_nine_record_gate(
        _passing_nine(aggregate_dense_relative_l2=0.1)
    )
    assert not bad_aggregate.passed
    for metric in (
        "family_query_relative_l2",
        "family_dense_relative_l2",
        "family_late_relative_l2",
    ):
        values = {"uniform": 0.08, "layered": 0.1, "marmousi": 0.08}
        failed = evaluate_nine_record_gate(_passing_nine(**{metric: values}))
        assert not failed.passed
        assert any("layered" in item for item in failed.failures)


def test_nine_record_gate_rejects_anomaly_or_missing_family_keys():
    with pytest.raises(ValueError, match="anomaly"):
        evaluate_nine_record_gate(
            _passing_nine(
                record_count_by_family={
                    "uniform": 3,
                    "layered": 3,
                    "marmousi": 3,
                    "anomaly": 3,
                }
            )
        )
    with pytest.raises(ValueError, match="family"):
        evaluate_nine_record_gate(
            _passing_nine(family_query_relative_l2={"uniform": 0.08, "layered": 0.08})
        )
