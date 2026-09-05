from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.training.pilot import PilotIdentity
from scripts.train_grouped_v3_pilot import (
    combine_validation_metrics,
    dense_at_pilot_queries,
    pilot_run_digest,
    relative_metrics_by_family,
)


def _identity(**updates) -> PilotIdentity:
    values = dict(
        manifest_digest="manifest",
        gate_config_digest="gate-config",
        gate_step=750,
        parent_checkpoint_path="/tmp/parent.pt",
        parent_checkpoint_sha256="a" * 64,
        prerequisite_report_path="/tmp/report.json",
        prerequisite_report_sha256="b" * 64,
    )
    values.update(updates)
    return PilotIdentity(**values)


def test_pilot_run_digest_binds_config_manifest_and_parent():
    first = pilot_run_digest("pilot-config", _identity())
    assert first == pilot_run_digest("pilot-config", _identity())
    assert first != pilot_run_digest("other-config", _identity())
    assert first != pilot_run_digest(
        "pilot-config", replace(_identity(), parent_checkpoint_sha256="c" * 64)
    )


def test_dense_at_pilot_queries_matches_each_independent_record():
    dense = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    x_m = torch.arange(5, dtype=torch.float32) * 10.0
    z_m = torch.arange(4, dtype=torch.float32) * 10.0
    times = torch.tensor([[0.1, 0.3, 0.7], [0.2, 0.4, 0.8]])
    coords = torch.tensor(
        [
            [[20.0, 10.0, 0.3], [40.0, 30.0, 0.7]],
            [[0.0, 0.0, 0.2], [30.0, 20.0, 0.8]],
        ]
    )
    sampled = dense_at_pilot_queries(dense, coords, times, x_m=x_m, z_m=z_m)
    expected = torch.stack(
        (dense[0, torch.tensor([1, 2]), torch.tensor([1, 3]), torch.tensor([2, 4])],
         dense[1, torch.tensor([0, 2]), torch.tensor([0, 2]), torch.tensor([0, 3])])
    )
    torch.testing.assert_close(sampled, expected)
    with pytest.raises(RuntimeError, match="time"):
        dense_at_pilot_queries(
            dense, coords + torch.tensor([0.0, 0.0, 0.01]), times, x_m=x_m, z_m=z_m
        )


def test_validation_reports_exact_and_interpolated_metrics_separately():
    target_dense = torch.ones(6, 3, 2, 2)
    target_query = torch.ones(6, 5)
    prediction_dense = target_dense.clone()
    prediction_query = target_query.clone()
    prediction_dense[2:4] *= 0.8
    prediction_query[2:4] *= 0.7
    families = ("uniform", "uniform", "layered", "layered", "marmousi", "marmousi")
    exact = relative_metrics_by_family(
        prediction_dense,
        prediction_query,
        target_dense,
        target_query,
        families,
        dense_at_query=prediction_query * 0.9,
    )
    interpolated = relative_metrics_by_family(
        prediction_dense * 0.9, prediction_query * 0.9, target_dense, target_query, families
    )
    report = combine_validation_metrics(exact, interpolated)
    assert set(report) == {"exact", "interpolated", "score"}
    assert report["exact"]["family_query_relative_l2"]["layered"] == pytest.approx(0.3)
    assert report["exact"]["family_query_relative_l2"]["uniform"] == pytest.approx(0.0)
    assert report["interpolated"]["aggregate_dense_relative_l2"] > 0
    assert report["exact"]["aggregate_head_consistency_relative_l2"] == pytest.approx(0.09)
    assert report["score"] == pytest.approx(
        report["exact"]["aggregate_query_relative_l2"]
        + report["exact"]["aggregate_dense_relative_l2"]
        + report["interpolated"]["aggregate_query_relative_l2"]
        + report["interpolated"]["aggregate_dense_relative_l2"]
    )


def test_continuous_pilot_config_is_epoch_checkpointed_and_full_data():
    config = V3Config.from_yaml("configs/grouped_v3/continuous_pilot.yaml")
    assert config.train.batch_records == 12
    assert config.train.evaluation_every == config.train.steps_per_epoch == 187
    assert config.train.max_steps == config.train.epochs * config.train.steps_per_epoch
    assert config.data.continuous_fraction == 0.25
    assert config.train.workers >= 1 and config.train.prefetch_factor >= 2
