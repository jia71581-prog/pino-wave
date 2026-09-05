from __future__ import annotations

import numpy as np
import torch

from scripts.summarize_transfer_dg_local_dtn import summarize
from scripts.train_transfer_dg_local_dtn import (
    robust_flux_relative,
    source_group_split,
)


def test_source_split_is_group_disjoint_and_balanced():
    sample, family = [], []
    for name in ("uniform", "layered", "marmousi"):
        for record in range(8):
            sample.extend([f"{name}_{record}"] * 3)
            family.extend([name] * 3)
    fit, holdout = source_group_split(np.asarray(sample), np.asarray(family))
    assert fit.sum() == 54
    assert holdout.sum() == 18
    assert set(np.asarray(sample)[fit]).isdisjoint(set(np.asarray(sample)[holdout]))


def test_robust_flux_relative_has_exact_zero_and_denominator_floor():
    target = torch.full((2, 2, 4, 8), 1.0e-8)
    exact = robust_flux_relative(target, target)
    zero = robust_flux_relative(torch.zeros_like(target), target)
    assert torch.count_nonzero(exact) == 0
    assert torch.isfinite(zero).all()
    assert zero.max() < 1.0e-6


def test_four_seed_summary_requires_gain_and_passivity():
    rows = []
    for seed in (372, 733, 1049, 1403):
        rows.append(
            {
                "seed": seed,
                "best_aggregate": 0.8,
                "zero_flux_baseline": 1.0,
                "best_metrics": {
                    "per_family": {
                        "uniform": 0.7,
                        "layered": 0.8,
                        "marmousi": 0.9,
                    },
                    "dissipation_minimum": 0.0,
                },
            }
        )
    assert summarize(rows)["decision"] == "accepted_local_operator_pilot"
    rows[0]["best_metrics"]["dissipation_minimum"] = -1.0
    assert summarize(rows)["decision"] == "rejected"
