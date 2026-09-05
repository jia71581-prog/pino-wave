from __future__ import annotations

import torch

from continuous_wave_operator.config import ModelConfig
from continuous_wave_operator.medium_encoder import MediumEncoder


def test_medium_encoder_builds_reusable_multiscale_cache() -> None:
    encoder = MediumEncoder(
        ModelConfig(width=16, token_grid_size=6, spectral_modes=(8, 6, 4, 3))
    )
    velocity = torch.full((2, 1, 201, 201), 3000.0, requires_grad=True)

    cache = encoder(velocity)

    assert [feature.shape[-2:] for feature in cache.local_features] == [
        (201, 201),
        (101, 101),
        (51, 51),
        (26, 26),
    ]
    assert cache.global_tokens.shape == (2, 36, 16)
    cache.global_tokens.sum().backward()
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()
