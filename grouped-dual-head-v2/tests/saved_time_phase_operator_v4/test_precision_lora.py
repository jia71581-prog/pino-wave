from __future__ import annotations

import torch
from torch import nn

from saved_time_phase_operator_v4.instance_adaptation.precision_lora import (
    PrecisionFirstLoRAConv2d,
    PrecisionFirstLoRALinear,
    QuantizedLoRAConv2dCandidate,
    QuantizedLoRALinearCandidate,
    guarded_quantized_forward,
    quantization_policy,
)


def test_full_precision_linear_lora_is_exact_and_only_adapter_is_trainable():
    torch.manual_seed(431)
    base = nn.Linear(9, 7)
    value = torch.randn(5, 9)
    expected = base(value)
    wrapped = PrecisionFirstLoRALinear(base, rank=3, train_magnitude=True)
    actual = wrapped(value)
    assert torch.equal(actual, expected)
    assert all(not parameter.requires_grad for parameter in wrapped.base.parameters())
    assert all(parameter.requires_grad for parameter in wrapped.adapter_parameters())


def test_full_precision_conv_lora_is_exact_and_merge_equivalent():
    torch.manual_seed(433)
    base = nn.Conv2d(4, 6, 3, padding=1)
    value = torch.randn(2, 4, 11, 13)
    wrapped = PrecisionFirstLoRAConv2d(base, rank=3)
    assert torch.equal(wrapped(value), base(value))
    with torch.no_grad():
        wrapped.lora_up.weight.normal_(std=0.01)
    weight, bias = wrapped.merged_weight_bias()
    merged = torch.nn.functional.conv2d(value, weight, bias, padding=1)
    torch.testing.assert_close(wrapped(value), merged, atol=2.0e-6, rtol=2.0e-6)


def test_svd_compensation_reduces_quantized_linear_output_error():
    torch.manual_seed(439)
    base = nn.Linear(7, 5)
    value = torch.randn(32, 7)
    plain = QuantizedLoRALinearCandidate(
        base, rank=5, bits=2, group_size=4, compensate=False
    )
    compensated = QuantizedLoRALinearCandidate(
        base, rank=5, bits=2, group_size=4, compensate=True
    )
    expected = base(value)
    plain_error = (plain(value) - expected).norm()
    compensated_error = (compensated(value) - expected).norm()
    assert compensated_error < plain_error * 1.0e-4


def test_svd_compensation_reduces_quantized_pointwise_conv_error():
    torch.manual_seed(441)
    base = nn.Conv2d(3, 4, 1)
    value = torch.randn(2, 3, 9, 11)
    plain = QuantizedLoRAConv2dCandidate(
        base, rank=3, bits=2, group_size=2, compensate=False
    )
    compensated = QuantizedLoRAConv2dCandidate(
        base, rank=3, bits=2, group_size=2, compensate=True
    )
    expected = base(value)
    plain_error = (plain(value) - expected).norm()
    compensated_error = (compensated(value) - expected).norm()
    assert compensated_error < plain_error * 1.0e-4


def test_precision_guard_falls_back_to_exact_full_precision_output():
    torch.manual_seed(443)
    base = nn.Linear(8, 6)
    value = torch.randn(16, 8)
    unsafe = QuantizedLoRALinearCandidate(
        base, rank=1, bits=2, group_size=4, compensate=False
    )
    result = guarded_quantized_forward(base, unsafe, value, tolerance=0.0)
    assert not result.accepted_quantized
    assert result.relative_error > 0.0
    assert torch.equal(result.output, base(value))


def test_conservative_quantization_policy_protects_wave_physics_modules():
    assert quantization_policy("blocks.0.spatial.frequency", nn.Linear(8, 8)) == "full_precision_protected"
    assert quantization_policy("physical_head.4", nn.Conv2d(8, 2, 1)) == "full_precision_protected"
    assert quantization_policy("branch_context.0", nn.Linear(8, 8)) == "quantized_candidate"
    assert quantization_policy("pyramid_context.0", nn.Conv2d(8, 8, 1)) == "quantized_candidate"
    assert quantization_policy("pyramid_context.0", nn.Conv2d(8, 8, 3)) == "full_precision_default"
