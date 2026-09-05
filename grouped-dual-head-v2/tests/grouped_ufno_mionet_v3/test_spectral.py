from __future__ import annotations

import io

import pytest
import torch

from grouped_ufno_mionet_v3.model.spectral import (
    ComplexSpectralResidualBlock,
    LearnedComplexSpectralConv2d,
)


def _explicit(layer: LearnedComplexSpectralConv2d, x: torch.Tensor) -> torch.Tensor:
    x_fft = torch.fft.rfft2(x.float(), norm=layer.fft_norm)
    output_fft = torch.zeros(
        x.shape[0],
        layer.out_channels,
        x.shape[-2],
        x.shape[-1] // 2 + 1,
        dtype=x_fft.dtype,
        device=x.device,
    )
    my, mx = layer.retained_modes(x.shape[-2], x.shape[-1])
    top = torch.view_as_complex(layer.weight_top.contiguous())[:, :, :my, :mx]
    bottom = torch.view_as_complex(layer.weight_bottom.contiguous())[:, :, :my, :mx]
    output_fft[:, :, :my, :mx] = torch.einsum(
        "bixy,ioxy->boxy", x_fft[:, :, :my, :mx], top
    )
    output_fft[:, :, -my:, :mx] = torch.einsum(
        "bixy,ioxy->boxy", x_fft[:, :, -my:, :mx], bottom
    )
    return torch.fft.irfft2(output_fft, s=x.shape[-2:], norm=layer.fft_norm)


def test_learned_complex_contraction_matches_explicit_fourier_product():
    torch.manual_seed(7)
    layer = LearnedComplexSpectralConv2d(2, 3, modes_y=2, modes_x=3)
    x = torch.randn(4, 2, 7, 9)
    actual = layer(x)
    expected = _explicit(layer, x)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_positive_and_negative_vertical_modes_have_independent_weights():
    layer = LearnedComplexSpectralConv2d(1, 1, modes_y=2, modes_x=2)
    with torch.no_grad():
        layer.weight_top.zero_()
        layer.weight_bottom.zero_()
        layer.weight_top[..., 0].fill_(1.0)
    x = torch.randn(1, 1, 9, 9)
    top_only = layer(x)
    with torch.no_grad():
        layer.weight_top.zero_()
        layer.weight_bottom[..., 0].fill_(1.0)
    bottom_only = layer(x)
    assert not torch.allclose(top_only, bottom_only)
    assert layer.weight_top.data_ptr() != layer.weight_bottom.data_ptr()


def test_odd_201_grid_restores_shape_and_all_weights_receive_gradients():
    torch.manual_seed(3)
    layer = LearnedComplexSpectralConv2d(2, 2, modes_y=5, modes_x=6)
    x = torch.randn(1, 2, 201, 201, requires_grad=True)
    output = layer(x)
    assert output.shape == (1, 2, 201, 201)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for parameter in (layer.weight_top, layer.weight_bottom):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_complex_weights_are_paired_real_parameters_and_serialize():
    layer = LearnedComplexSpectralConv2d(2, 3, modes_y=2, modes_x=3)
    assert layer.weight_top.dtype == torch.float32
    assert layer.weight_top.shape == (2, 3, 2, 3, 2)
    buffer = io.BytesIO()
    torch.save(layer.state_dict(), buffer)
    buffer.seek(0)
    restored = LearnedComplexSpectralConv2d(2, 3, modes_y=2, modes_x=3)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    x = torch.randn(2, 2, 11, 13)
    torch.testing.assert_close(layer(x), restored(x))


def test_residual_block_combines_local_and_mode_dependent_paths():
    block = ComplexSpectralResidualBlock(
        width=8,
        spectral_rank=4,
        modes_y=3,
        modes_x=3,
    )
    x = torch.randn(2, 8, 17, 19, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    required = {
        "spectral.weight_top",
        "spectral.weight_bottom",
        "in_projection.weight",
        "out_projection.weight",
        "local_depthwise.weight",
        "local_pointwise.weight",
    }
    gradients = {name for name, parameter in block.named_parameters() if parameter.grad is not None}
    assert required <= gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_spectral_contraction_runs_fp32_under_cuda_autocast():
    layer = LearnedComplexSpectralConv2d(2, 2, modes_y=3, modes_x=3).cuda()
    x = torch.randn(1, 2, 21, 23, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = layer(x)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
