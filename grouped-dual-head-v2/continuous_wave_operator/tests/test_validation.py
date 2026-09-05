from __future__ import annotations

import numpy as np
import torch

from continuous_wave_operator.training.validation import fixed_validation_indices, validation_metrics


def test_fixed_validation_queries_and_metrics_are_deterministic() -> None:
    first = fixed_validation_indices("sample-7", seed=2026, shape=(8, 12, 10), count=16)
    second = fixed_validation_indices("sample-7", seed=2026, shape=(8, 12, 10), count=16)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))

    target = torch.tensor([[0.0, 1.0, -1.0, 0.5], [0.0, 2.0, -2.0, 1.0]])
    metrics = validation_metrics(target.clone(), target, medium_types=("uniform", "layered"))
    assert metrics["relative_l2"] == 0.0
    assert metrics["normalized_rmse"] == 0.0
    assert metrics["trace_correlation"] > 0.9999
    assert set(metrics["per_medium"]) == {"uniform", "layered"}
