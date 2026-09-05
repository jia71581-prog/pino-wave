from __future__ import annotations

import torch

from continuous_wave_operator.config import ModelConfig
from continuous_wave_operator.medium_encoder import MediumEncoder
from continuous_wave_operator.source_encoder import SourceEncoder


def test_source_encoder_preserves_mass_and_supports_shots() -> None:
    config = ModelConfig(width=16, spectral_modes=(8, 6, 4, 3))
    medium = MediumEncoder(config)(torch.full((2, 1, 201, 201), 2500.0))
    source_map = torch.zeros(2, 3, 1, 201, 201)
    source_map[:, :, :, 10, 20] = 1.0
    parameters = torch.tensor([[[200.0, 100.0, 15.0, 0.1, 1.0]]]).expand(2, 3, 5).clone()

    result = SourceEncoder(config)(medium, source_map, parameters)

    assert result.latent.shape == (2, 3, 16)
    assert torch.allclose(result.source_mass, torch.ones(2, 3), atol=1.0e-6)
    assert torch.allclose(result.amplitude, torch.ones(2, 3))
