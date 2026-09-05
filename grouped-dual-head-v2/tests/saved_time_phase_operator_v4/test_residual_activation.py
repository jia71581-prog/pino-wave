import pytest
import torch
from torch import nn

from saved_time_phase_operator_v4.residual_activation import (
    activate_residual_head,
    residual_activation_config,
)


class _Decoder(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.output = nn.Conv2d(3, 1, kernel_size=1)
        self.correction_scale = nn.Parameter(torch.tensor(scale))


@pytest.mark.parametrize("scale", [-7.5e-7, 0.0, 0.25])
def test_absorb_is_function_preserving_and_fixes_the_gate(scale: float):
    torch.manual_seed(17)
    decoder = _Decoder(scale)
    features = torch.randn(2, 3, 5, 4)
    before = decoder.correction_scale * decoder.output(features)

    report = activate_residual_head(decoder, {"activation_mode": "absorb"})
    after = decoder.correction_scale * decoder.output(features)

    torch.testing.assert_close(after, before, rtol=1.0e-6, atol=1.0e-8)
    assert decoder.correction_scale.item() == pytest.approx(1.0)
    assert not decoder.correction_scale.requires_grad
    assert report["mode"] == "absorb"
    assert report["pre_correction_scale"] == pytest.approx(scale)
    assert report["post_correction_scale"] == pytest.approx(1.0)


def test_absorbed_zero_gate_restores_output_head_gradient():
    torch.manual_seed(23)
    decoder = _Decoder(0.0)
    features = torch.randn(2, 3, 5, 4)
    target = torch.randn(2, 1, 5, 4)
    activate_residual_head(decoder, {"activation_mode": "absorb"})

    loss = (decoder.correction_scale * decoder.output(features) - target).square().mean()
    loss.backward()

    assert decoder.output.weight.grad is not None
    assert decoder.output.weight.grad.norm().item() > 0.0


def test_activation_config_accepts_explicit_absorb_mode():
    resolved = residual_activation_config(
        {"activation_mode": "absorb"},
        {"parent_optimizer_state": False},
    )
    assert resolved["activation_mode"] == "absorb"
