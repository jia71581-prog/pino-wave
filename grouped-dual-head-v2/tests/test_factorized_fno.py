"""Tests for FactorizedAcousticFNO — factorized spatiotemporal architecture."""

from __future__ import annotations

import pytest
import torch

from fno_acoustic.model_factorized import (
    FactorizedAcousticFNO,
    SpatialEncoder,
    SpectralTemporalMixer,
    SpectralConv2d,
    TemporalMixer,
)


class TestSpectralConv2d:
    def test_forward_shape(self) -> None:
        conv = SpectralConv2d(in_channels=8, out_channels=8, modes_x=12, modes_z=12)
        x = torch.randn(2, 8, 32, 32)
        out = conv(x)
        assert out.shape == (2, 8, 32, 32)

    def test_different_in_out_channels(self) -> None:
        conv = SpectralConv2d(in_channels=4, out_channels=8, modes_x=8, modes_z=8)
        x = torch.randn(2, 4, 16, 16)
        out = conv(x)
        assert out.shape == (2, 8, 16, 16)

    def test_modes_exceed_nyquist_raises(self) -> None:
        conv = SpectralConv2d(in_channels=4, out_channels=4, modes_x=16, modes_z=16)
        x = torch.randn(2, 4, 8, 8)  # Nyquist: 4
        with pytest.raises(ValueError):
            conv(x)

    def test_forward_finite(self) -> None:
        conv = SpectralConv2d(in_channels=4, out_channels=4, modes_x=8, modes_z=8)
        x = torch.randn(2, 4, 16, 16)
        out = conv(x)
        assert torch.isfinite(out).all()

    def test_backward(self) -> None:
        conv = SpectralConv2d(in_channels=4, out_channels=4, modes_x=8, modes_z=8)
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        out = conv(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_double_forward_backward_uses_complex128_weights(self) -> None:
        conv = SpectralConv2d(2, 3, modes_x=2, modes_z=2).double()
        x = torch.randn(1, 2, 6, 6, dtype=torch.float64, requires_grad=True)

        conv(x).square().mean().backward()

        assert conv.weights1.dtype == torch.complex128
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_pure_device_migration_preserves_existing_complex_dtype(self) -> None:
        conv = SpectralConv2d(2, 2, modes_x=2, modes_z=2).double().to("cpu")

        assert conv.weights1.dtype == torch.complex128


class TestSpatialEncoder:
    def test_forward_single_frame_shape(self) -> None:
        encoder = SpatialEncoder(
            in_features=3, spatial_width=16,
            spatial_modes_x=8, spatial_modes_z=8,
            spatial_layers=2,
        )
        x = torch.randn(2, 32, 32, 3)  # [B,H,W,C]
        out = encoder.forward_single_frame(x)
        assert out.shape == (2, 32, 32, 16)

    def test_forward_multiframe_shape(self) -> None:
        encoder = SpatialEncoder(
            in_features=3, spatial_width=16,
            spatial_modes_x=8, spatial_modes_z=8,
            spatial_layers=2,
        )
        x = torch.randn(2, 32, 32, 10, 3)  # [B,H,W,T,C]
        out = encoder(x)
        assert out.shape == (2, 32, 32, 10, 16)

    def test_forward_finite(self) -> None:
        encoder = SpatialEncoder(
            in_features=3, spatial_width=16,
            spatial_modes_x=8, spatial_modes_z=8,
            spatial_layers=2,
        )
        x = torch.randn(2, 32, 32, 5, 3)
        out = encoder(x)
        assert torch.isfinite(out).all()

    def test_backward(self) -> None:
        encoder = SpatialEncoder(
            in_features=3, spatial_width=16,
            spatial_modes_x=8, spatial_modes_z=8,
            spatial_layers=2,
        )
        x = torch.randn(2, 32, 32, 5, 3, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_chunked_matches_unchunked_for_multiple_batches_and_times(self) -> None:
        encoder = SpatialEncoder(
            in_features=2,
            spatial_width=4,
            spatial_modes_x=1,
            spatial_modes_z=1,
            spatial_layers=1,
            padding_ratio=0,
        ).eval()
        x = torch.randn(2, 4, 4, 4, 2)

        with torch.no_grad():
            unchunked = encoder(x)
            chunked = encoder(x, spatial_chunk_size=4)

        assert torch.equal(chunked, unchunked)


class TestTemporalMixer:
    def test_forward_shape(self) -> None:
        mixer = TemporalMixer(
            spatial_width=16, temporal_width=16,
            temporal_kernel=5, temporal_layers=2,
        )
        x = torch.randn(128, 20, 16)  # [N, T, spatial_width]
        out = mixer(x)
        assert out.shape == (128, 20, 16)

    def test_forward_finite(self) -> None:
        mixer = TemporalMixer(
            spatial_width=16, temporal_width=16,
            temporal_kernel=5, temporal_layers=2,
        )
        x = torch.randn(64, 10, 16)
        out = mixer(x)
        assert torch.isfinite(out).all()

    def test_backward(self) -> None:
        mixer = TemporalMixer(
            spatial_width=16, temporal_width=16,
            temporal_kernel=5, temporal_layers=2,
        )
        x = torch.randn(32, 8, 16, requires_grad=True)
        out = mixer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_long_sequence(self) -> None:
        """T=160 like the real data."""
        mixer = TemporalMixer(
            spatial_width=16, temporal_width=16,
            temporal_kernel=7, temporal_layers=2,
        )
        x = torch.randn(256, 160, 16)  # 16*16 spatial points × 160 time steps
        out = mixer(x)
        assert out.shape == (256, 160, 16)
        assert torch.isfinite(out).all()


class TestSpectralTemporalMixer:
    def test_forward_shape_and_backward(self) -> None:
        mixer = SpectralTemporalMixer(
            spatial_width=8,
            temporal_width=8,
            temporal_modes=6,
            temporal_layers=2,
        )
        value = torch.randn(12, 17, 8, requires_grad=True)
        output = mixer(value)
        assert output.shape == value.shape
        output.square().mean().backward()
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in mixer.parameters()
        )

    def test_modes_cannot_exceed_rfft_support(self) -> None:
        mixer = SpectralTemporalMixer(
            spatial_width=4,
            temporal_width=4,
            temporal_modes=7,
            temporal_layers=1,
        )
        with pytest.raises(ValueError, match="exceed rFFT bins"):
            mixer(torch.randn(2, 10, 4))


class TestFactorizedAcousticFNO:
    @staticmethod
    def make_model(**overrides: object) -> FactorizedAcousticFNO:
        kwargs: dict[str, object] = {
            "in_features": 3,
            "spatial_modes_x": 8,
            "spatial_modes_z": 8,
            "spatial_width": 16,
            "spatial_layers": 2,
            "temporal_width": 16,
            "temporal_kernel": 5,
            "temporal_layers": 2,
            "head_hidden": 64,
        }
        kwargs.update(overrides)
        return FactorizedAcousticFNO(**kwargs)  # type: ignore[arg-type]

    def test_forward_shape(self) -> None:
        model = self.make_model()
        x = torch.randn(2, 32, 32, 20, 3)
        out = model(x)
        assert out.shape == (2, 32, 32, 20)

    def test_spectral_temporal_mixer_forward_and_backward(self) -> None:
        model = self.make_model(temporal_mixer_type="spectral", temporal_modes=6)
        value = torch.randn(1, 16, 16, 20, 3, requires_grad=True)
        output = model(value)
        assert output.shape == (1, 16, 16, 20)
        output.square().mean().backward()
        assert any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if "temporal_mixer.spectral" in name
        )

    def test_forward_shape_64x64x160(self) -> None:
        """Target resolution for first real training run."""
        model = self.make_model()
        x = torch.randn(1, 64, 64, 160, 3)
        out = model(x)
        assert out.shape == (1, 64, 64, 160)

    def test_forward_finite(self) -> None:
        model = self.make_model()
        x = torch.randn(2, 32, 32, 10, 3)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_backward(self) -> None:
        model = self.make_model()
        x = torch.randn(2, 16, 16, 8, 3, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # Check that spectral weights got gradients
        for name, param in model.named_parameters():
            if "weights" in name:
                assert param.grad is not None, f"{name} has no grad"

    def test_batch_size_gt_1(self) -> None:
        model = self.make_model()
        x = torch.randn(4, 32, 32, 10, 3)
        out = model(x)
        assert out.shape == (4, 32, 32, 10)

    def test_wrong_input_ndim_raises(self) -> None:
        model = self.make_model()
        x = torch.randn(2, 32, 32, 20)  # missing C dim
        with pytest.raises(ValueError):
            model(x)

    def test_wrong_input_channels_raises(self) -> None:
        model = self.make_model(in_features=3)
        x = torch.randn(2, 32, 32, 10, 5)  # 5 channels, expected 3
        with pytest.raises(ValueError):
            model(x)

    def test_memory_estimate_reasonable(self) -> None:
        """At 64x64x160, spatial_width=32, temporal_width=32, batch=1:
        spatial: 1*64*64*32*4 = 524 KB
        temporal: 1*64*64*160*32*4 = 83.9 MB
        Total ~84 MB for hidden tensors — much less than dense 3D (~1.6 GB at width=32).
        """
        # Just verify the model can run forward at this size
        model = FactorizedAcousticFNO(
            in_features=3,
            spatial_modes_x=32,
            spatial_modes_z=32,
            spatial_width=32,
            spatial_layers=2,
            temporal_width=32,
            temporal_kernel=7,
            temporal_layers=2,
            head_hidden=128,
        )
        x = torch.randn(1, 64, 64, 160, 3)
        out = model(x)
        assert out.shape == (1, 64, 64, 160)
        assert torch.isfinite(out).all()

    def test_cross_resolution(self) -> None:
        """Model trained at one resolution should run at another (no hardcoded sizes)."""
        model = self.make_model()
        # Train shape
        x1 = torch.randn(1, 32, 32, 10, 3)
        out1 = model(x1)
        assert out1.shape == (1, 32, 32, 10)
        # Different eval shape
        x2 = torch.randn(1, 48, 48, 15, 3)
        out2 = model(x2)
        assert out2.shape == (1, 48, 48, 15)

    def test_repr(self) -> None:
        model = self.make_model()
        rep = repr(model)
        assert "FactorizedAcousticFNO" in rep
