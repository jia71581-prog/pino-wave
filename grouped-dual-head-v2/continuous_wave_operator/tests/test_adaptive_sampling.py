from __future__ import annotations

import torch

from continuous_wave_operator.training.adaptive_sampling import HierarchicalAIS


def test_hierarchical_ais_prioritizes_residual_without_losing_support() -> None:
    ais = HierarchicalAIS(sample_count=4, time_bins=8, spatial_shape=(12, 12), seed=2026)
    ais.update(
        torch.tensor([1]),
        torch.tensor([[3]]),
        torch.tensor([[[7, 8]]]),
        torch.tensor([[[10.0]]]),
    )

    probabilities = ais.probabilities(epoch=5)

    assert probabilities.sample[1] > probabilities.sample[0]
    assert probabilities.time[1, 3] > probabilities.time[1, 0]
    assert probabilities.spatial[1, 7, 8] > probabilities.spatial[1, 0, 0]
    assert torch.all(probabilities.sample > 0)
    assert ais.inverse_probability_weights(torch.tensor([0.9, 0.01])).max() <= ais.config.max_importance_weight
    restored = HierarchicalAIS.from_state_dict(ais.state_dict())
    assert torch.equal(restored.sample_ema, ais.sample_ema)
    assert restored.step == ais.step
