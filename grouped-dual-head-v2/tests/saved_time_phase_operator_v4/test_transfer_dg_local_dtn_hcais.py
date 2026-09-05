from __future__ import annotations

import numpy as np

from scripts.summarize_transfer_dg_local_dtn_hcais import summarize
from scripts.train_transfer_dg_local_dtn_hcais import (
    build_hcais_strata,
    combined_difficulty,
)


def test_hcais_strata_cover_family_frequency_and_element_type():
    families = np.asarray(["uniform", "uniform", "layered", "marmousi"])
    frequency = np.asarray(
        [[0.25, 2 / 3, 0, 0], [0.50, 2 / 3, 0, 0], [0.50, 2 / 3, 0, 0], [0.50, 2 / 3, 0, 0]]
    )
    origins = np.asarray([[20, 20], [0, 20], [20, 20], [20, 20]])
    interface = np.asarray([0.0, 0.0, 0.4, 0.8])
    strata = build_hcais_strata(families, frequency, origins, interface)
    assert len(np.unique(strata)) == 4


def test_combined_difficulty_is_finite_bounded_and_error_sensitive():
    strata = np.zeros(4, dtype=np.int64)
    score = combined_difficulty(
        np.asarray([1.0, 2.0, 3.0, 9.0]),
        np.ones(4),
        np.asarray([0.0, 0.1, 0.2, 0.3]),
        np.asarray([0.0, 0.0, 0.2, 0.5]),
        strata,
    )
    assert np.isfinite(score).all()
    assert np.all((score >= 0.0) & (score <= 1.0))
    assert score[-1] > score[0]


def test_hcais_summary_requires_accuracy_ess_and_overhead():
    hcais, baseline = [], []
    for seed in (372, 733, 1049, 1403):
        common = {
            "seed": seed,
            "best_metrics": {
                "per_family": {"uniform": 0.5, "layered": 0.6, "marmousi": 0.7}
            },
        }
        hcais.append(
            {
                **common,
                "best_aggregate": 0.6,
                "sampler": {
                    "minimum_epoch_ess_p5_fraction": 0.7,
                    "sampling_overhead_fraction": 0.05,
                },
            }
        )
        baseline.append({**common, "best_aggregate": 0.7})
    assert summarize(hcais, baseline)["decision"] == "accepted_sampling_pilot"
    hcais[0]["sampler"]["sampling_overhead_fraction"] = 0.2
    assert summarize(hcais, baseline)["decision"] == "rejected"
