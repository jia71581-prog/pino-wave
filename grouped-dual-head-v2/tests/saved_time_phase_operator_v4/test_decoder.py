import pytest

from saved_time_phase_operator_v4.decoder import PropagationConditionedDenseDecoder


def _decoder(*, family_expert_rank: int) -> PropagationConditionedDenseDecoder:
    return PropagationConditionedDenseDecoder(
        width=8,
        spectral_rank=6,
        pyramid_levels=2,
        modes=4,
        depth=2,
        saved_time_count=401,
        time_block=2,
        domain_t_s=1.0,
        domain_diagonal_m=2828.0,
        use_local_phase=True,
        activation_checkpointing=False,
        family_expert_rank=family_expert_rank,
    )


def test_decoder_constructs_optional_three_family_experts():
    disabled = _decoder(family_expert_rank=0)
    enabled = _decoder(family_expert_rank=4)

    assert disabled.family_experts is None
    assert len(enabled.family_experts.experts) == 3


def test_decoder_rejects_negative_family_expert_rank():
    with pytest.raises(ValueError, match="family expert rank"):
        _decoder(family_expert_rank=-1)


import torch
from saved_time_phase_operator_v4.decoder import _HighFrequencyResidualHead


def test_high_frequency_head_is_zero_initialized_and_shaped():
    head = _HighFrequencyResidualHead(width=8, hidden=16, depth=3)
    x = torch.randn(3, 8, 17, 17)
    y = head(x)
    assert y.shape == (3, 1, 17, 17)
    # zero-init output conv -> exact no-op at load (warm-start safe)
    assert float(y.abs().max()) == 0.0
    # gradients still flow into the body so it can learn off zero
    y.sum().backward()
    assert all(p.grad is not None for p in head.body.parameters())


def test_decoder_high_frequency_head_optional_and_default_off():
    off = _decoder(family_expert_rank=0)
    assert off.high_frequency_head is None
    on = PropagationConditionedDenseDecoder(
        width=8, spectral_rank=6, pyramid_levels=2, modes=4, depth=2,
        saved_time_count=401, time_block=2, domain_t_s=1.0, domain_diagonal_m=2828.0,
        use_local_phase=True, activation_checkpointing=False, family_expert_rank=0,
        high_frequency_residual=True, high_frequency_hidden=16, high_frequency_depth=2,
    )
    assert on.high_frequency_head is not None
    assert on.high_frequency_head.output.in_channels == 16
