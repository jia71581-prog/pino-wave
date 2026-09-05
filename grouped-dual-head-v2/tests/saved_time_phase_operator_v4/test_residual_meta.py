from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.train_meta_hypernet import (
    _parent_normalized_at_times,
    residual_adaptive_time_indices,
)
from scripts.train_v5_residual_meta import _time_indices


def test_meta_time_indices_always_include_both_onset_frames():
    indices = _time_indices(401, (73, 74), 16)
    assert 73 in indices.tolist()
    assert 74 in indices.tolist()
    assert indices.tolist() == sorted(set(indices.tolist()))


def test_meta_time_indices_are_bounded():
    indices = _time_indices(5, (1, 2), 64)
    assert indices.tolist() == [0, 1, 2, 3, 4]


def test_residual_adaptive_time_indices_prioritize_all_budgeted_hotspots():
    candidates = torch.arange(20)
    scores = torch.arange(20, dtype=torch.float32)
    selected = residual_adaptive_time_indices(
        candidates, scores, (2, 3), 8, high_fraction=0.75
    )
    assert selected.tolist() == [0, 2, 3, 15, 16, 17, 18, 19]


def test_residual_adaptive_time_indices_are_deterministic_and_cover_axis():
    candidates = torch.arange(0, 40, 2)
    scores = torch.zeros(len(candidates))
    scores[7:13] = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    first = residual_adaptive_time_indices(
        candidates, scores, (4, 6), 10, high_fraction=0.5
    )
    repeated = residual_adaptive_time_indices(
        candidates, scores, (4, 6), 10, high_fraction=0.5
    )
    assert torch.equal(first, repeated)
    assert len(first) == 10
    assert 4 in first.tolist() and 6 in first.tolist()
    assert {18, 20, 22, 24}.issubset(set(first.tolist()))
    assert int(first.min()) == 0
    assert int(first.max()) >= 36


def test_meta_parent_uses_cached_dense_travel_time():
    sentinel_grid = object()

    class Parent:
        def encode_medium(self, velocity, normalizer):
            return velocity

        def prepare_sources(self, medium, source, source_map, normalizer, **kwargs):
            return SimpleNamespace(source_parameters=source)

        def prepare_dense_grid(self, prepared, *, x_m, z_m, travel_time_s=None):
            assert travel_time_s is not None
            assert travel_time_s.shape == (1, 3, 4)
            assert torch.equal(travel_time_s[0], record.dense_travel_time_s)
            return sentinel_grid

        def dense_normalized(self, prepared, time_s, *, dense_grid, time_block):
            assert dense_grid is sentinel_grid
            assert time_block == 1
            return torch.zeros(1, len(time_s), 3, 4)

    record = SimpleNamespace(
        velocity_mps=torch.ones(1, 3, 4),
        source_parameters=torch.tensor([100.0, 100.0, 20.0, 0.1, 1.0]),
        source_map=torch.ones(1, 3, 4),
        x_m=torch.arange(4, dtype=torch.float32),
        z_m=torch.arange(3, dtype=torch.float32),
        dense_travel_time_s=torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )
    result = _parent_normalized_at_times(
        Parent(),
        object(),
        record,
        torch.device("cpu"),
        torch.tensor([0.0, 0.1]),
    )
    assert result.shape == (1, 2, 3, 4)
