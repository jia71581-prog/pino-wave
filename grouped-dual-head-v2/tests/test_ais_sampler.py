from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import pytest
import torch

from fno_acoustic.ais_sampler import (
    AdaptiveSpatialSampler,
    SpatialSamplingFeatures,
)


AI_MIXTURE = (0.30, 0.20, 0.15, 0.20, 0.05, 0.10)


def synthetic_features(height: int = 20, width: int = 20, offset: int = 0):
    n = height * width
    x = torch.arange(n, dtype=torch.float64)
    return SpatialSamplingFeatures(
        uniform=torch.ones(n),
        interface=(x.remainder(17) + 1 + offset),
        source_wavefront=(n - x + offset),
        physical_edge=(x.remainder(width).sub(width / 2).abs() + 1),
        receiver_neighborhood=(x.remainder(11) + 1 + 2 * offset),
    )


def make_sampler(seed: int = 13):
    return AdaptiveSpatialSampler(20, 20, AI_MIXTURE, seed, tile_size=5)


def test_mixture_has_positive_support_and_normalized_components():
    sampler = make_sampler(seed=7)
    features = synthetic_features()

    components = sampler._component_distributions(11, features)
    proposal = sampler.proposal(sample_id=11, features=features)

    assert components.shape == (6, 400)
    assert torch.allclose(components.sum(dim=1), torch.ones(6, dtype=torch.float64))
    assert proposal.shape == (400,)
    assert torch.all(proposal > 0)
    assert torch.allclose(proposal.sum(), torch.tensor(1.0, dtype=proposal.dtype))


def test_zero_mass_feature_components_fall_back_to_explicit_uniform():
    sampler = AdaptiveSpatialSampler(4, 5, (0, 1, 0, 0, 0, 0), seed=7)
    zeros = torch.zeros(20)
    features = SpatialSamplingFeatures(zeros, zeros, zeros, zeros, zeros)

    components = sampler._component_distributions(1, features)
    proposal = sampler.proposal(1, features)

    assert torch.equal(components, torch.full((6, 20), 1 / 20, dtype=torch.float64))
    assert torch.equal(proposal, torch.full((20,), 1 / 20, dtype=torch.float64))


def test_each_scene_draws_its_own_sites_and_reports_batch_diagnostics():
    sampler = make_sampler(seed=9)
    draw = sampler.draw(
        [3, 4], [synthetic_features(offset=0), synthetic_features(offset=3)], count=64
    )

    assert draw.site_indices.shape == (2, 64)
    assert draw.draw_probability.shape == (2, 64)
    assert draw.component.shape == (2, 64)
    assert not torch.equal(draw.site_indices[0], draw.site_indices[1])
    assert set(draw.diagnostics) == {
        "ess",
        "duplicate_fraction",
        "coverage",
        "max_median_inverse_weight",
    }
    assert 0 < draw.diagnostics["ess"] <= 128
    assert 0 <= draw.diagnostics["duplicate_fraction"] <= 1
    assert 0 < draw.diagnostics["coverage"] <= 1
    assert draw.diagnostics["max_median_inverse_weight"] >= 1


def test_draw_groups_site_sampling_by_component(monkeypatch):
    sampler = AdaptiveSpatialSampler(400, 400, AI_MIXTURE, seed=9)
    features = synthetic_features(400, 400)
    original = torch.multinomial
    calls = 0

    def counted_multinomial(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", counted_multinomial)
    draw = sampler.draw([3], [features], count=2048)

    assert draw.site_indices.shape == (1, 2048)
    assert calls <= 7  # one component draw plus at most one site draw per component


def test_sampler_state_round_trip_reproduces_exact_next_draw():
    first = make_sampler(seed=13)
    first.draw([1], [synthetic_features()], count=32)
    state = first.state_dict()
    expected = first.draw([1], [synthetic_features()], count=32)

    restored = make_sampler(seed=999)
    restored.load_state_dict(state)
    actual = restored.draw([1], [synthetic_features()], count=32)

    assert torch.equal(actual.site_indices, expected.site_indices)
    assert torch.equal(actual.component, expected.component)
    assert torch.equal(actual.draw_probability, expected.draw_probability)


def test_residual_updates_are_lagged_until_epoch_commit():
    sampler = make_sampler(seed=21)
    before = sampler.proposal(1, synthetic_features())
    sampler.update_residual_tiles(
        torch.tensor([1]), torch.tensor([[7]]), torch.tensor([[100.0]])
    )

    assert torch.equal(sampler.proposal(1, synthetic_features()), before)
    assert sampler.ema_version == 0
    sampler.commit_epoch()
    assert not torch.equal(sampler.proposal(1, synthetic_features()), before)
    assert sampler.ema_version == 1
    assert sampler.pending_updates == []


def test_commit_pools_all_pending_observations_before_one_tile_ema():
    sampler = AdaptiveSpatialSampler(
        20, 20, AI_MIXTURE, seed=21, tile_size=5, ema_momentum=0.5
    )
    sampler.update_residual_tiles(
        torch.tensor([1]), torch.tensor([[1]]), torch.tensor([[1.0]])
    )
    sampler.update_residual_tiles(
        torch.tensor([1]), torch.tensor([[2]]), torch.tensor([[9.0]])
    )

    sampler.commit_epoch()

    assert sampler.residual_tiles[1][0, 0] == pytest.approx(3.0)


def test_commit_is_invariant_to_pending_partition_and_order():
    combined = make_sampler(seed=21)
    split = make_sampler(seed=21)
    combined.update_residual_tiles(
        torch.tensor([1]), torch.tensor([[1, 2]]), torch.tensor([[1.0, 9.0]])
    )
    split.update_residual_tiles(
        torch.tensor([1]), torch.tensor([[2]]), torch.tensor([[9.0]])
    )
    split.update_residual_tiles(
        torch.tensor([1]), torch.tensor([[1]]), torch.tensor([[1.0]])
    )

    combined.commit_epoch()
    split.commit_epoch()

    assert torch.equal(combined.residual_tiles[1], split.residual_tiles[1])


def test_residual_tiles_use_tile_size_boundaries_when_grid_is_not_divisible():
    sampler = AdaptiveSpatialSampler(20, 20, AI_MIXTURE, seed=2, tile_size=16)
    sampler.residual_tiles[1] = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64
    )

    dense = sampler.residual_distribution(1).reshape(20, 20)

    assert dense[15, 15] == 1
    assert dense[15, 16] == 2
    assert dense[16, 15] == 3
    assert dense[16, 16] == 4


def test_draw_probability_is_final_mixture_proposal_at_drawn_sites():
    sampler = make_sampler(seed=17)
    features = synthetic_features()
    draw = sampler.draw([5], [features], count=128)
    expected = sampler.proposal(5, features).index_select(0, draw.site_indices[0])
    assert torch.equal(draw.draw_probability[0], expected)


def test_coverage_diagnostic_is_scene_tile_coverage():
    sampler = AdaptiveSpatialSampler(20, 20, AI_MIXTURE, seed=17, tile_size=5)
    draw = sampler.draw(
        [5, 6], [synthetic_features(), synthetic_features(offset=2)], count=128
    )
    x = torch.div(draw.site_indices, 20, rounding_mode="floor")
    z = draw.site_indices.remainder(20)
    tile = torch.div(x, 5, rounding_mode="floor") * 4 + torch.div(
        z, 5, rounding_mode="floor"
    )
    pairs = torch.stack(
        (draw.sample_ids[:, None].expand_as(tile).reshape(-1), tile.reshape(-1)), dim=1
    )
    expected = torch.unique(pairs, dim=0).shape[0] / (2 * 16)

    assert draw.diagnostics["coverage"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "mixture",
    [
        (1, 0, 0, 0, 0, 0),
        (0.375, 0.25, 0.1875, 0, 0.0625, 0.125),
        (0.30, 0, 0, 0.70, 0, 0),
        AI_MIXTURE,
    ],
    ids=["B1", "S", "A", "AI"],
)
def test_all_ablation_mixtures_can_draw(mixture):
    draw = AdaptiveSpatialSampler(20, 20, mixture, 3).draw(
        [8], [synthetic_features()], count=12
    )
    assert draw.site_indices.shape == (1, 12)
    assert torch.all(draw.draw_probability > 0)


@pytest.mark.parametrize(
    "mixture",
    [
        (1, 0, 0),
        (1, 0, 0, 0, 0, -0.1),
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ],
)
def test_invalid_mixture_is_rejected(mixture):
    with pytest.raises(ValueError, match="six nonnegative weights summing to one"):
        AdaptiveSpatialSampler(20, 20, mixture, 1)


def test_bad_feature_length_and_empty_batch_are_rejected():
    sampler = make_sampler()
    bad = synthetic_features()
    bad = SpatialSamplingFeatures(
        bad.uniform[:-1],
        bad.interface,
        bad.source_wavefront,
        bad.physical_edge,
        bad.receiver_neighborhood,
    )
    with pytest.raises(ValueError, match="feature vectors"):
        sampler.proposal(1, bad)
    with pytest.raises(ValueError, match="nonempty"):
        sampler.draw([], [], count=4)


@pytest.mark.parametrize(
    ("sample_ids", "site_indices", "squared_error"),
    [
        (torch.tensor([1, 2]), torch.tensor([[1, 2]]), torch.ones(1, 2)),
        (torch.tensor([1]), torch.tensor([[1, 2]]), torch.ones(1, 3)),
        (torch.tensor([1]), torch.tensor([[1, -1]]), torch.ones(1, 2)),
    ],
)
def test_bad_residual_update_shapes_or_values_are_rejected(
    sample_ids, site_indices, squared_error
):
    with pytest.raises(ValueError):
        make_sampler().update_residual_tiles(sample_ids, site_indices, squared_error)


def test_pending_updates_survive_state_restore_and_commit_identically():
    first = make_sampler(seed=41)
    first.update_residual_tiles(
        torch.tensor([2]), torch.tensor([[0, 7, 399]]), torch.tensor([[2.0, 8.0, 32.0]])
    )
    restored = make_sampler(seed=999)
    restored.load_state_dict(first.state_dict())

    assert len(restored.pending_updates) == 1
    first.commit_epoch()
    restored.commit_epoch()
    assert torch.equal(restored.residual_tiles[2], first.residual_tiles[2])
    assert restored.ema_version == first.ema_version == 1


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_bad_checkpoint_load_is_transactional_for_every_mutable_field():
    source = make_sampler(seed=31)
    source.draw([2], [synthetic_features()], 4)
    source.update_residual_tiles(
        torch.tensor([2]), torch.tensor([[1, 2]]), torch.tensor([[3.0, 4.0]])
    )
    source.commit_epoch()
    good = source.state_dict()

    corruptions = []
    bad = copy.deepcopy(good)
    bad["seed"] = True
    corruptions.append(bad)
    bad = copy.deepcopy(good)
    bad["ema_version"] = 1.5
    corruptions.append(bad)
    bad = copy.deepcopy(good)
    bad["residual_tiles"][2] = torch.ones(3, 3)
    corruptions.append(bad)
    bad = copy.deepcopy(good)
    bad["residual_tiles"][2][0, 0] = torch.nan
    corruptions.append(bad)
    bad = copy.deepcopy(good)
    key = next(iter(bad["generator_states"]))
    bad["generator_states"][key] = torch.ones(3)
    corruptions.append(bad)
    bad = copy.deepcopy(good)
    bad["pending_updates"] = [(torch.tensor([2.0]), torch.tensor([[1]]), torch.ones(1, 1))]
    corruptions.append(bad)
    bad = copy.deepcopy(good)
    bad["pending_updates"] = [(torch.tensor([2]), torch.tensor([[400]]), torch.ones(1, 1))]
    corruptions.append(bad)

    for bad_state in corruptions:
        target = make_sampler(seed=77)
        target.draw([9], [synthetic_features()], 3)
        target.update_residual_tiles(
            torch.tensor([9]), torch.tensor([[7]]), torch.tensor([[8.0]])
        )
        before = target.state_dict()
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            target.load_state_dict(bad_state)
        _assert_nested_equal(target.state_dict(), before)


def test_restored_pending_update_commits_with_exact_saved_momentum():
    first = AdaptiveSpatialSampler(20, 20, AI_MIXTURE, 41, tile_size=5, ema_momentum=0.2)
    first.update_residual_tiles(
        torch.tensor([2]), torch.tensor([[7]]), torch.tensor([[11.0]])
    )
    restored = AdaptiveSpatialSampler(
        20, 20, AI_MIXTURE, 999, tile_size=5, ema_momentum=0.2
    )
    restored.load_state_dict(first.state_dict())

    restored.commit_epoch()

    assert restored.residual_tiles[2][0, 1] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 21},
        {"width": 21},
        {"tile_size": 4},
        {"ema_momentum": 0.3},
    ],
)
def test_state_restore_rejects_different_spatial_or_ema_configuration(kwargs):
    state = make_sampler(seed=3).state_dict()
    config = dict(
        height=20,
        width=20,
        mixture=AI_MIXTURE,
        seed=999,
        tile_size=5,
        ema_momentum=0.2,
    )
    config.update(kwargs)
    restored = AdaptiveSpatialSampler(**config)

    with pytest.raises(ValueError, match="configuration differs"):
        restored.load_state_dict(state)


def test_mixture_mismatch_is_rejected_when_loading_state():
    state = make_sampler().state_dict()
    other = AdaptiveSpatialSampler(
        20, 20, (1, 0, 0, 0, 0, 0), seed=1, tile_size=5
    )
    with pytest.raises(ValueError, match="mixture differs"):
        other.load_state_dict(state)


def test_public_records_are_frozen():
    draw = make_sampler().draw([1], [synthetic_features()], 1)
    with pytest.raises(FrozenInstanceError):
        draw.ema_version = 99
    features = synthetic_features()
    with pytest.raises(FrozenInstanceError):
        features.uniform = torch.zeros(400)


def test_mixture_storage_is_not_aliased_and_features_normalize_to_cpu_float64():
    mixture = torch.tensor(AI_MIXTURE, dtype=torch.float64)
    sampler = AdaptiveSpatialSampler(2, 2, mixture, seed=1)
    mixture.zero_()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = SpatialSamplingFeatures(
        torch.ones(4, dtype=torch.float32, device=device),
        torch.ones(4, dtype=torch.int64),
        torch.ones(4, dtype=torch.float16),
        torch.ones(4, dtype=torch.float32),
        torch.ones(4, dtype=torch.float64),
    )

    components = sampler._component_distributions(1, features)

    assert torch.equal(sampler.mixture, torch.tensor(AI_MIXTURE, dtype=torch.float64))
    assert components.device.type == "cpu"
    assert components.dtype == torch.float64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 20.9},
        {"width": 20.9},
        {"tile_size": 5.8},
        {"height": True},
    ],
)
def test_constructor_rejects_non_integer_discrete_inputs(kwargs):
    config = dict(height=20, width=20, mixture=AI_MIXTURE, seed=1, tile_size=5)
    config.update(kwargs)
    with pytest.raises((TypeError, ValueError), match="integer"):
        AdaptiveSpatialSampler(**config)


def test_draw_and_proposal_reject_non_integer_count_and_sample_id():
    sampler = make_sampler()
    with pytest.raises((TypeError, ValueError), match="integer"):
        sampler.draw([1], [synthetic_features()], count=1.9)
    with pytest.raises((TypeError, ValueError), match="integer"):
        sampler.proposal(1.9, synthetic_features())
    with pytest.raises((TypeError, ValueError), match="integer"):
        sampler.proposal(torch.tensor(True), synthetic_features())


def test_all_public_sampler_entries_reject_negative_sample_ids():
    sampler = make_sampler()
    with pytest.raises(ValueError, match="nonnegative"):
        sampler.generator_for(-1)
    with pytest.raises(ValueError, match="nonnegative"):
        sampler.residual_distribution(-1)
    with pytest.raises(ValueError, match="nonnegative"):
        sampler.proposal(-1, synthetic_features())
    with pytest.raises(ValueError, match="nonnegative"):
        sampler.draw([-1], [synthetic_features()], count=1)
    with pytest.raises(ValueError, match="nonnegative"):
        sampler.update_residual_tiles(
            torch.tensor([-1]), torch.tensor([[3]]), torch.ones(1, 1)
        )


@pytest.mark.parametrize(
    ("sample_ids", "site_indices"),
    [
        (torch.tensor([1.0]), torch.tensor([[3]])),
        (torch.tensor([True]), torch.tensor([[3]])),
        (torch.tensor([1]), torch.tensor([[3.9]])),
        (torch.tensor([1]), torch.tensor([[True]])),
    ],
)
def test_residual_updates_reject_non_integer_discrete_tensor_dtypes(
    sample_ids, site_indices
):
    with pytest.raises((TypeError, ValueError), match="integer"):
        make_sampler().update_residual_tiles(
            sample_ids, site_indices, torch.ones_like(site_indices, dtype=torch.float64)
        )
