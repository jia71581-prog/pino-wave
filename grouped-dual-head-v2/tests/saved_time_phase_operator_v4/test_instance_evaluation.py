from __future__ import annotations

import pytest
import torch

from scripts.evaluate_v5_instance_adaptation import evaluate_after_adaptation
from saved_time_phase_operator_v4.instance_adaptation.visualization import derive_receivers_from_field


def _fields():
    truth = torch.randn(1, 401, 17, 17)
    parent = truth + 0.1
    adapted = truth + 0.05
    return truth, parent, adapted


def test_evaluator_refuses_unsealed_future_truth():
    truth, parent, adapted = _fields()
    with pytest.raises(RuntimeError, match="sealed"):
        evaluate_after_adaptation(
            {"parent_field": parent, "adapted_field": adapted}, truth,
            observed_indices=((2, 3),), families=("uniform",), group_ids=("g",),
            sample_ids=("s",), sealed=False,
        )


def test_evaluator_streams_all_401_saved_times():
    truth, parent, adapted = _fields()
    report = evaluate_after_adaptation(
        {"parent_field": parent, "adapted_field": adapted}, truth,
        observed_indices=((2, 3),), families=("uniform",), group_ids=("g",),
        sample_ids=("s",), sealed=True,
    )
    assert report["metrics"]["unique_time_index_count"] == 401
    assert report["all_saved_time_indices"] == 401
    assert report["future_truth_squared_norm"] > 0.0
    assert report["future_adapted_squared_error"] >= 0.0
    assert report["future_parent_squared_error"] >= 0.0


def test_receiver_traces_are_derived_from_full_field():
    truth, _, _ = _fields()
    result = derive_receivers_from_field(truth, ((1, 3), (2, 4)))
    assert result.shape == (2, 401)
