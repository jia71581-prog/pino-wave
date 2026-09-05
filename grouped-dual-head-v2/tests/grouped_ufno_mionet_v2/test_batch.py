import pytest
import torch

from grouped_ufno_mionet_v2.data.batch import pack_v2_groups
from grouped_ufno_mionet_v2.data.cache import V2CacheRecord


def record(sample_id, source_x, target_value, velocity=None):
    velocity = torch.full((1, 9, 9), 2000.0) if velocity is None else velocity
    source_map = torch.zeros(1, 9, 9); source_map[0, 2, int(source_x)] = 1
    return V2CacheRecord(
        velocity, torch.tensor([float(source_x), 20, 10, .005, 1]), source_map,
        torch.arange(3), torch.full((3, 9, 9), float(target_value)),
        torch.tensor([[1, 1], [1, 7]]), torch.full((2, 5), float(target_value)),
        torch.zeros(7, 3), torch.full((7,), float(target_value)), torch.full((7,), 1/7),
        torch.arange(5) * .0025, sample_id, "medium-a", "uniform", {"split": "train"},
    )


def test_grouped_batch_deduplicates_medium_but_keeps_source_targets():
    batch = pack_v2_groups([record("a", 2, 1), record("b", 6, 2)])
    assert batch.velocity_mps.shape[0] == 1
    assert batch.source_parameters.shape[0] == 2
    assert batch.source_map.shape == (2, 1, 9, 9)
    assert batch.dense_target.shape[0] == 2
    assert not torch.equal(batch.dense_target[0], batch.dense_target[1])
    assert torch.equal(batch.record_to_medium, torch.tensor([0, 0]))


def test_grouped_batch_rejects_different_velocities_with_same_group():
    changed = torch.full((1, 9, 9), 2100.0)
    with pytest.raises(ValueError, match="different velocity"):
        pack_v2_groups([record("a", 2, 1), record("b", 6, 2, changed)])


def test_grouped_batch_rejects_non_unit_source_map():
    bad = record("a", 2, 1); bad.source_map.mul_(2)
    with pytest.raises(ValueError, match="unit mass"):
        pack_v2_groups([bad])
