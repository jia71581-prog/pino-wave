from __future__ import annotations

import pytest
import torch

from grouped_ufno_mionet_v3.model.medium import ComplexMediumEncoder, sample_medium_pyramid
from grouped_ufno_mionet_v3.model.source import SourceEncoderV3


def _medium_encoder() -> ComplexMediumEncoder:
    return ComplexMediumEncoder(
        width=8,
        rank=6,
        spectral_rank=4,
        modes=(4, 3),
        token_grid=2,
        position_bands=2,
    )


def test_medium_encoder_returns_multiscale_position_aware_state():
    encoder = _medium_encoder()
    velocity = torch.zeros(2, 1, 17, 19)
    state = encoder(velocity)
    assert len(state.pyramid) == 2
    assert state.pyramid[0].shape == (2, 8, 17, 19)
    assert state.pyramid[1].shape == (2, 8, 9, 10)
    assert state.tokens.shape == (2, 4, 8)
    assert state.token_positions.shape == (2, 4, 2)
    assert state.rank.shape == (2, 6)
    assert not torch.allclose(state.tokens[:, 0], state.tokens[:, -1])


def test_medium_pyramid_sampling_preserves_local_geometry_and_mapping():
    encoder = _medium_encoder()
    velocity = torch.randn(2, 1, 15, 15)
    state = encoder(velocity)
    coords = torch.tensor(
        [
            [[0.1, 0.2], [0.8, 0.7]],
            [[0.1, 0.2], [0.8, 0.7]],
            [[0.1, 0.2], [0.8, 0.7]],
        ]
    )
    local = sample_medium_pyramid(state, coords, torch.tensor([0, 0, 1]))
    assert local.shape == (3, 2, 16)
    torch.testing.assert_close(local[0], local[1])
    assert not torch.allclose(local[0], local[2])


def test_all_medium_spectral_weights_receive_nonzero_gradients():
    torch.manual_seed(4)
    encoder = _medium_encoder()
    velocity = torch.randn(2, 1, 17, 19, requires_grad=True)
    state = encoder(velocity)
    loss = state.rank.square().mean()
    loss = loss + sum(level.square().mean() for level in state.pyramid)
    loss.backward()
    spectral = {
        name: parameter
        for name, parameter in encoder.named_parameters()
        if "spectral.weight_" in name
    }
    assert spectral
    for name, parameter in spectral.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_source_encoder_keeps_position_frequency_amplitude_map_and_local_medium():
    torch.manual_seed(5)
    medium = _medium_encoder()(torch.randn(1, 1, 17, 19))
    encoder = SourceEncoderV3(width=8, rank=6)
    source = torch.tensor(
        [
            [0.25, 0.50, 0.20, 0.10, 1.0],
            [0.75, 0.50, 0.40, 0.10, 2.0],
        ]
    )
    source_map = torch.zeros(2, 1, 17, 19)
    source_map[0, 0, 8, 4] = 1.0
    source_map[1, 0, 8, 14] = 1.0
    encoded = encoder(source, source_map, medium, torch.tensor([0, 0]))
    assert encoded.hidden.shape == (2, 8)
    assert encoded.rank.shape == (2, 6)
    assert encoded.map_field.shape == (2, 8, 17, 19)
    assert encoded.local_medium.shape == (2, 8)
    assert not torch.allclose(encoded.hidden[0], encoded.hidden[1])

    encoded.hidden.square().mean().backward()
    assert encoder.parameter_mlp[0].weight.grad is not None
    for column in range(5):
        assert encoder.parameter_mlp[0].weight.grad[:, column].abs().sum() > 0


def test_source_encoder_rejects_non_unit_or_negative_source_maps():
    medium = _medium_encoder()(torch.zeros(1, 1, 9, 9))
    encoder = SourceEncoderV3(width=8, rank=6)
    source = torch.tensor([[0.5, 0.5, 0.2, 0.1, 1.0]])
    bad = torch.zeros(1, 1, 9, 9)
    with pytest.raises(ValueError, match="unit mass"):
        encoder(source, bad, medium, torch.tensor([0]))
    bad[0, 0, 4, 4] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        encoder(source, bad, medium, torch.tensor([0]))


def test_branch_gradient_groups_are_explicit_and_nonempty():
    medium = _medium_encoder()
    source = SourceEncoderV3(width=8, rank=6)
    assert set(medium.required_gradient_groups()) == {
        "medium_spectral",
        "medium_local",
        "medium_tokens",
        "medium_rank",
    }
    assert set(source.required_gradient_groups()) == {
        "source_parameters",
        "source_map",
        "source_local_medium",
        "source_rank",
    }
    assert all(medium.required_gradient_groups().values())
    assert all(source.required_gradient_groups().values())
