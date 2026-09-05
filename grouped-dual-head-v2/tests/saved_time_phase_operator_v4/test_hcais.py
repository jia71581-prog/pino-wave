from __future__ import annotations

import numpy as np

from saved_time_phase_operator_v4.hcais import (
    coverage_distribution,
    effective_sample_size,
    hcais_distribution,
    inverse_probability_weights,
    regularized_leverage_scores,
    within_stratum_percentile,
)


def test_percentiles_and_coverage_preserve_every_stratum():
    strata = np.asarray([0, 0, 0, 1, 1])
    values = np.asarray([3.0, 1.0, 2.0, 9.0, 9.0])
    rank = within_stratum_percentile(values, strata)
    coverage = coverage_distribution(strata)
    assert rank[0] > rank[2] > rank[1]
    assert rank[3] == rank[4]
    assert np.isclose(coverage.sum(), 1.0)
    assert np.all(coverage >= 0.5 / len(strata))


def test_leverage_favors_a_unique_direction():
    features = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    score = regularized_leverage_scores(features, ridge=0.1)
    assert score[2] > score[0]
    assert score[0] == score[1]


def test_hcais_is_positive_normalized_and_defensively_bounded():
    strata = np.asarray([0, 0, 0, 1, 1, 1])
    q = hcais_distribution(
        np.asarray([100.0, 0.0, 0.0, 0.0, 0.0, 10.0]),
        np.asarray([0.0, 0.0, 10.0, 0.0, 4.0, 0.0]),
        strata,
        coverage_mix=0.4,
    )
    assert np.isclose(q.sum(), 1.0)
    assert np.all(q > 0.0)
    assert np.all(q >= 0.2 / len(q))


def test_inverse_weights_recover_uniform_expectation():
    q = np.asarray([0.6, 0.3, 0.1])
    loss = np.asarray([2.0, 5.0, 11.0])
    indices = np.arange(3)
    weights = inverse_probability_weights(indices, q)
    recovered = np.sum(q * weights * loss)
    assert np.isclose(recovered, loss.mean())
    assert effective_sample_size(np.ones(8)) == 8.0
