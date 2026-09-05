"""
Factorized Spatiotemporal FNO (FS-FNO).

Decomposes 3D operator learning into:
1. Per-frame 2D spatial FNO (processes [B,H,W,C] for each time frame)
2. Per-point 1D temporal mixer (processes [B,T,D] for each spatial point)
3. Output projection

Memory: O(H*W*spatial_width + H*W*T*temporal_width) vs O(H*W*T*width) for dense 3D.
At 400x400x160, this is ~6 GB vs ~48 GB for width=32.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


class SpectralConv2d(nn.Module):
    """2D spectral convolution — FFT over H×W spatial dimensions only."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_z: int,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_z = int(modes_z)
        scale = 1.0 / max(1, in_channels * out_channels)
        shape = (in_channels, out_channels, self.modes_x, self.modes_z)
        # 2D FFT produces 4 quadrants (like SpectralConv3d but with 2 dims)
        self.weights1 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixz,ioxz->boxz", x, weights)

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> "SpectralConv2d":
        current_real_dtype = (
            torch.float64 if self.weights1.dtype == torch.complex128 else torch.float32
        )
        real_probe = torch.empty(
            (), dtype=current_real_dtype, device=self.weights1.device
        )
        converted = fn(real_probe)
        complex_dtype = (
            torch.complex128
            if converted.dtype == torch.float64
            else torch.complex64
        )

        def convert_parameter(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=converted.device, dtype=complex_dtype)

        return super()._apply(convert_parameter, recurse=recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  (spatial only — call per time frame)
        Returns:
            [B, out_channels, H, W]
        """
        if x.ndim != 4:
            raise ValueError(f"SpectralConv2d expects [B,C,H,W], got {tuple(x.shape)}")
        input_dtype = x.dtype
        if x.dtype in {torch.float16, torch.bfloat16}:
            x = x.float()
        batch, _, h, w = x.shape
        if self.modes_x > h // 2 or self.modes_z > w // 2:
            raise ValueError(
                f"modes ({self.modes_x},{self.modes_z}) exceed Nyquist for grid ({h},{w})"
            )
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(
            batch, self.out_channels, h, w // 2 + 1,
            dtype=x_ft.dtype, device=x.device,
        )
        mx, mz = self.modes_x, self.modes_z
        weights1 = self.weights1.to(x_ft.dtype)
        weights2 = self.weights2.to(x_ft.dtype)
        weights3 = self.weights3.to(x_ft.dtype)
        weights4 = self.weights4.to(x_ft.dtype)
        out_ft[:, :, :mx, :mz] = self.compl_mul2d(x_ft[:, :, :mx, :mz], weights1)
        out_ft[:, :, -mx:, :mz] = self.compl_mul2d(x_ft[:, :, -mx:, :mz], weights2)
        out_ft[:, :, :mx, -mz:] = self.compl_mul2d(x_ft[:, :, :mx, -mz:], weights3)
        out_ft[:, :, -mx:, -mz:] = self.compl_mul2d(x_ft[:, :, -mx:, -mz:], weights4)
        out = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))
        return out.to(input_dtype) if out.dtype != input_dtype else out


class SpatialFNOBlock(nn.Module):
    """One 2D FNO block: SpectralConv2d + Conv2d 1×1 + Norm + Activation.

    Optional softmax gates (identity at init, disabled by default):
      - spatial gate: per-channel softmax over flattened H*W applied to the
        block output — spatial attention that can concentrate on the wavefront /
        high-residual region (complements residual-based importance sampling).
      - mode gate: per-channel softmax over the kept 2D Fourier modes scaling
        the spectrum inside SpectralConv2d — Fourier-domain attention.
    """

    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_z: int,
        normalization: str = "group",
        num_groups: int = 8,
        spatial_softmax: bool = False,
        mode_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_z)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = _make_norm_2d(normalization, width, num_groups=num_groups)
        self.spatial_softmax = bool(spatial_softmax)
        self.mode_softmax = bool(mode_softmax)
        if self.mode_softmax:
            # 4 FFT quadrants, each (modes_x, modes_z); score table per channel.
            self.mode_gate_scores = nn.Parameter(
                torch.zeros(1, width, 4, int(modes_x), int(modes_z))
            )
        if self.spatial_softmax:
            # Registered lazily on first forward (grid size unknown until then).
            self.register_parameter("spatial_gate_scores", None)

    def _ensure_spatial_gate(self, x: torch.Tensor) -> None:
        if self.spatial_gate_scores is None:
            h, w = int(x.shape[-2]), int(x.shape[-1])
            self.spatial_gate_scores = nn.Parameter(
                torch.zeros(1, x.shape[1], h, w, device=x.device, dtype=x.dtype)
            )
        elif tuple(self.spatial_gate_scores.shape[-2:]) != tuple(x.shape[-2:]):
            raise ValueError(
                f"spatial gate grid {tuple(self.spatial_gate_scores.shape[-2:])} "
                f"!= input grid {tuple(x.shape[-2:])}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode_softmax:
            spectral_out = self._gated_spectral(x)
        else:
            spectral_out = self.spectral(x)
        out = F.gelu(self.norm(spectral_out + self.pointwise(x)))
        if self.spatial_softmax:
            self._ensure_spatial_gate(out)
            n = int(out.shape[-2]) * int(out.shape[-1])
            gate = torch.softmax(
                self.spatial_gate_scores.reshape(1, out.shape[1], n), dim=-1
            ).reshape(self.spatial_gate_scores.shape) * float(n)
            out = out * gate.to(out.dtype)
        return out

    def _gated_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """SpectralConv2d with a per-channel softmax gate over kept modes.

        Mirrors SpectralConv2d.forward exactly (same FFT convention, same
        weights1..4 quadrant packing), inserting a per-channel softmax gate on
        the input spectrum of each quadrant before the learned complex mixing.
        At zero-init scores the gate is 1 everywhere -> bit-identical to
        self.spectral(x).
        """
        s = self.spectral
        input_dtype = x.dtype
        work = x.float() if x.dtype in {torch.float16, torch.bfloat16} else x
        batch, _, h, w = work.shape
        if s.modes_x > h // 2 or s.modes_z > w // 2:
            raise ValueError(
                f"modes ({s.modes_x},{s.modes_z}) exceed Nyquist for grid ({h},{w})"
            )
        x_ft = torch.fft.rfft2(work, dim=(-2, -1))
        out_ft = torch.zeros(
            batch, s.out_channels, h, w // 2 + 1,
            dtype=x_ft.dtype, device=x.device,
        )
        mx, mz = s.modes_x, s.modes_z
        gate_flat = torch.softmax(
            self.mode_gate_scores.reshape(1, x.shape[1], 4, mx * mz), dim=-1
        ) * float(mx * mz)
        gate = gate_flat.reshape_as(self.mode_gate_scores).to(x_ft.real.dtype)
        quadrants = [
            (x_ft[:, :, :mx, :mz], s.weights1, 0, (slice(None), slice(None), slice(0, mx), slice(0, mz))),
            (x_ft[:, :, -mx:, :mz], s.weights2, 1, (slice(None), slice(None), slice(-mx, None), slice(0, mz))),
            (x_ft[:, :, :mx, -mz:], s.weights3, 2, (slice(None), slice(None), slice(0, mx), slice(-mz, None))),
            (x_ft[:, :, -mx:, -mz:], s.weights4, 3, (slice(None), slice(None), slice(-mx, None), slice(-mz, None))),
        ]
        for spec_q, weight_q, qi, out_slice in quadrants:
            gated = spec_q * gate[:, :, qi].to(spec_q.dtype)
            out_ft[out_slice] = s.compl_mul2d(gated, weight_q.to(x_ft.dtype))
        out = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))
        return out.to(input_dtype) if out.dtype != input_dtype else out


class SpatialEncoder(nn.Module):
    """
    2D spatial FNO encoder.
    Processes each time frame independently: [B, C_in, H, W] → [B, spatial_width, H, W].
    """

    def __init__(
        self,
        in_features: int,
        spatial_width: int = 32,
        spatial_modes_x: int = 32,
        spatial_modes_z: int = 32,
        spatial_layers: int = 3,
        padding_ratio: float = 0.0625,
        normalization: str = "group",
        num_groups: int = 8,
        spatial_softmax: bool = False,
        mode_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_width = int(spatial_width)
        self.spatial_layers = int(spatial_layers)
        self.padding_ratio = float(padding_ratio)

        self.fc_in = nn.Linear(int(in_features), self.spatial_width)
        self.blocks = nn.ModuleList([
            SpatialFNOBlock(
                self.spatial_width,
                int(spatial_modes_x),
                int(spatial_modes_z),
                normalization=normalization,
                num_groups=num_groups,
                spatial_softmax=spatial_softmax,
                mode_softmax=mode_softmax,
            )
            for _ in range(self.spatial_layers)
        ])

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.padding_ratio <= 0:
            return x, (0, 0)
        h, w = x.shape[-2:]
        ph = max(0, int(round(h * self.padding_ratio)))
        pw = max(0, int(round(w * self.padding_ratio)))
        return F.pad(x, (0, pw, 0, ph)), (ph, pw)

    @staticmethod
    def _unpad(x: torch.Tensor, pads: tuple[int, int]) -> torch.Tensor:
        ph, pw = pads
        if ph:
            x = x[:, :, :-ph, :]
        if pw:
            x = x[:, :, :, :-pw]
        return x

    def forward_single_frame(self, x: torch.Tensor, use_checkpointing: bool = False) -> torch.Tensor:
        """
        Process one time frame.
        Args:
            x: [B, H, W, C_in]
            use_checkpointing: if True, use gradient checkpointing on FNO blocks
        Returns:
            [B, H, W, spatial_width]
        """
        original_hw = tuple(x.shape[1:3])
        x = self.fc_in(x)                          # [B,H,W,spatial_width]
        x = x.permute(0, 3, 1, 2).contiguous()     # [B,spatial_width,H,W]
        x, pads = self._pad(x)
        for block in self.blocks:
            if use_checkpointing and self.training and x.requires_grad:
                x = activation_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self._unpad(x, pads)
        if tuple(x.shape[-2:]) != original_hw:
            raise RuntimeError(
                f"unpadding changed shape to {tuple(x.shape[-2:])}, expected {original_hw}"
            )
        return x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,spatial_width]

    def forward(self, x: torch.Tensor, spatial_chunk_size: int = 0, use_checkpointing: bool = False) -> torch.Tensor:
        """
        Process all time frames — batched for GPU parallelism.
        Use spatial_chunk_size to limit peak memory for large H×W.
        Args:
            x: [B, H, W, T, C_in]
            spatial_chunk_size: max B*T frames per chunk (0 = all at once)
            use_checkpointing: gradient checkpointing on FNO blocks
        Returns:
            [B, H, W, T, spatial_width]
        """
        batch, h, w, t, _ = x.shape
        total_frames = batch * t
        chunk = max(1, int(spatial_chunk_size)) if spatial_chunk_size and spatial_chunk_size > 0 else total_frames
        ckpt = bool(use_checkpointing) and self.training

        if chunk >= total_frames:
            # Fuse batch and time: [B,H,W,T,C] → [B*T,H,W,C]
            x_flat = x.permute(0, 3, 1, 2, 4).contiguous().reshape(total_frames, h, w, x.shape[-1])
            encoded = self.forward_single_frame(x_flat, use_checkpointing=ckpt)
            return encoded.reshape(batch, t, h, w, -1).permute(0, 2, 3, 1, 4).contiguous()

        # Micro-batched: slice time dim BEFORE reshaping to avoid materializing full tensor
        encoded_chunks: list[torch.Tensor] = []
        for start_t in range(0, t, max(1, chunk // batch)):
            end_t = min(start_t + max(1, chunk // batch), t)
            # Slice along time: [B,H,W,chunk_T,C]
            x_chunk = x[:, :, :, start_t:end_t, :]
            chunk_frames = batch * (end_t - start_t)
            x_flat = x_chunk.permute(0, 3, 1, 2, 4).contiguous().reshape(chunk_frames, h, w, x.shape[-1])
            encoded = self.forward_single_frame(x_flat, use_checkpointing=ckpt)
            encoded_chunks.append(
                encoded.reshape(batch, end_t - start_t, h, w, -1)
            )
        encoded = torch.cat(encoded_chunks, dim=1)
        return encoded.permute(0, 2, 3, 1, 4).contiguous()


class TemporalMixer(nn.Module):
    """
    1D temporal mixer — processes each spatial point's time series independently.
    Uses grouped 1D convolutions for efficiency.

    Input: [B*H*W, T, spatial_width]
    Output: [B*H*W, T, temporal_width]
    """

    def __init__(
        self,
        spatial_width: int,
        temporal_width: int = 32,
        temporal_kernel: int = 7,
        temporal_layers: int = 3,
        normalization: str = "group",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.temporal_width = int(temporal_width)
        self.temporal_layers = int(temporal_layers)

        self.input_proj = nn.Conv1d(spatial_width, temporal_width, kernel_size=1)
        layers = []
        for _ in range(temporal_layers):
            padding = int(temporal_kernel) // 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        temporal_width, temporal_width,
                        kernel_size=int(temporal_kernel),
                        padding=padding,
                        groups=_num_groups_for_channels(temporal_width, int(num_groups)),
                    ),
                    _make_norm_1d(normalization, temporal_width, int(num_groups)),
                    nn.GELU(),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.output_proj = nn.Conv1d(temporal_width, temporal_width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*H*W, T, spatial_width]
        Returns:
            [B*H*W, T, temporal_width]
        """
        # Conv1d expects [N, C, L]
        x = x.permute(0, 2, 1).contiguous()         # [N, spatial_width, T]
        x = self.input_proj(x)                       # [N, temporal_width, T]
        residual = x
        for layer in self.layers:
            x = layer(x) + residual
            residual = x
        x = self.output_proj(x)                      # [N, temporal_width, T]
        return x.permute(0, 2, 1).contiguous()       # [N, T, temporal_width]


class SpectralConv1d(nn.Module):
    """Global Fourier convolution over the complete temporal axis."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if int(modes) <= 0:
            raise ValueError("temporal modes must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                self.modes,
                dtype=torch.cfloat,
            )
        )

    def _apply(self, fn, recurse: bool = True):
        real_probe = torch.empty(
            (),
            dtype=(torch.float64 if self.weight.dtype == torch.complex128 else torch.float32),
            device=self.weight.device,
        )
        converted = fn(real_probe)
        complex_dtype = torch.complex128 if converted.dtype == torch.float64 else torch.complex64

        def convert_parameter(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=converted.device, dtype=complex_dtype)

        return super()._apply(convert_parameter, recurse=recurse)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError("SpectralConv1d expects [batch,channel,time]")
        input_dtype = value.dtype
        work = value.float() if value.dtype in {torch.float16, torch.bfloat16} else value
        spectrum = torch.fft.rfft(work, dim=-1)
        available = spectrum.shape[-1]
        if self.modes > available:
            raise ValueError(
                f"temporal modes={self.modes} exceed rFFT bins={available}"
            )
        output = torch.zeros(
            value.shape[0],
            self.out_channels,
            available,
            dtype=spectrum.dtype,
            device=value.device,
        )
        output[..., : self.modes] = torch.einsum(
            "bim,iom->bom",
            spectrum[..., : self.modes],
            self.weight.to(spectrum.dtype),
        )
        result = torch.fft.irfft(output, n=value.shape[-1], dim=-1)
        return result.to(input_dtype) if result.dtype != input_dtype else result


class SpectralTemporalMixer(nn.Module):
    """Residual global temporal Fourier mixer for every spatial point.

    Optional per-layer softmax gates (disabled by default to keep the legacy
    checkpoint layout bit-compatible):
      - spectral gate: per-channel softmax over the rFFT mode axis scales the
        complex spectrum before the learned mode mixing, letting each channel
        re-weight its frequency content (Fourier-domain attention).
      - time gate: per-channel softmax over the time axis applied to the layer
        output, letting each spatial point concentrate on its active window.
    Both gates are exact identities at initialization: score tables are
    zero-initialized, so softmax(scores) = uniform = 1/K and (K * uniform) = 1,
    reproducing the un-gated forward pass bit-exactly.
    """

    def __init__(
        self,
        spatial_width: int,
        temporal_width: int = 32,
        temporal_modes: int = 32,
        temporal_layers: int = 3,
        normalization: str = "group",
        num_groups: int = 8,
        spectral_softmax: bool = False,
        time_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.temporal_width = int(temporal_width)
        self.spectral_softmax = bool(spectral_softmax)
        self.time_softmax = bool(time_softmax)
        self.input_proj = nn.Conv1d(int(spatial_width), self.temporal_width, 1)
        self.spectral = nn.ModuleList(
            [
                SpectralConv1d(
                    self.temporal_width, self.temporal_width, int(temporal_modes)
                )
                for _ in range(int(temporal_layers))
            ]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(self.temporal_width, self.temporal_width, 1) for _ in self.spectral]
        )
        self.norms = nn.ModuleList(
            [
                _make_norm_1d(
                    normalization, self.temporal_width, int(num_groups)
                )
                for _ in self.spectral
            ]
        )
        if self.spectral_softmax:
            # Gate sees all rFFT bins (T//2+1), sized lazily is impossible for
            # parameters, so the caller guarantees the time axis is fixed.
            # We register scores relative to temporal_modes (the mixed prefix);
            # bins beyond `modes` are dropped by SpectralConv1d anyway.
            self.spectral_gate_scores = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, self.temporal_width, int(temporal_modes)))
                    for _ in self.spectral
                ]
            )
        self.time_softmax_steps = 0
        if self.time_softmax:
            raise NotImplementedError(
                "time_softmax requires a fixed time axis; pass time_softmax_steps"
            )
        self.output_proj = nn.Conv1d(self.temporal_width, self.temporal_width, 1)

    def enable_time_softmax(self, time_steps: int) -> None:
        """Attach per-channel softmax-over-time gates (fixed time axis)."""
        time_steps = int(time_steps)
        if time_steps <= 0:
            raise ValueError("time_softmax_steps must be positive")
        self.time_softmax_steps = time_steps
        self.time_softmax = True
        device = self.output_proj.weight.device
        dtype = self.output_proj.weight.dtype
        self.time_gate_scores = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, self.temporal_width, time_steps, device=device, dtype=dtype))
                for _ in self.spectral
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError("SpectralTemporalMixer expects [point,time,channel]")
        work = self.input_proj(value.permute(0, 2, 1).contiguous())
        time_steps = int(work.shape[-1])
        for layer_idx, (spectral, pointwise, norm) in enumerate(
            zip(self.spectral, self.pointwise, self.norms, strict=True)
        ):
            if self.spectral_softmax:
                update = F.gelu(norm(self._gated_spectral(spectral, layer_idx, work) + pointwise(work)))
            else:
                update = F.gelu(norm(spectral(work) + pointwise(work)))
            if getattr(self, "time_softmax", False):
                if self.time_softmax_steps != time_steps:
                    raise ValueError(
                        f"time_softmax_steps={self.time_softmax_steps} != input time axis {time_steps}"
                    )
                scores = self.time_gate_scores[layer_idx].to(update.dtype)
                gate = torch.softmax(scores, dim=-1) * float(time_steps)
                update = update * gate
            work = work + update
        return self.output_proj(work).permute(0, 2, 1).contiguous()

    def _gated_spectral(
        self, spectral: SpectralConv1d, layer_idx: int, value: torch.Tensor
    ) -> torch.Tensor:
        """SpectralConv1d with a per-channel softmax gate over kept modes."""
        input_dtype = value.dtype
        work_v = value.float() if value.dtype in {torch.float16, torch.bfloat16} else value
        spectrum = torch.fft.rfft(work_v, dim=-1)
        available = spectrum.shape[-1]
        if spectral.modes > available:
            raise ValueError(
                f"temporal modes={spectral.modes} exceed rFFT bins={available}"
            )
        kept = spectrum[..., : spectral.modes]
        scores = self.spectral_gate_scores[layer_idx].to(kept.real.dtype)
        gate = torch.softmax(scores, dim=-1) * float(spectral.modes)
        kept = kept * gate.to(kept.dtype)
        output = torch.zeros(
            value.shape[0],
            spectral.out_channels,
            available,
            dtype=spectrum.dtype,
            device=value.device,
        )
        output[..., : spectral.modes] = torch.einsum(
            "bim,iom->bom", kept, spectral.weight.to(spectrum.dtype)
        )
        result = torch.fft.irfft(output, n=value.shape[-1], dim=-1)
        return result.to(input_dtype) if result.dtype != input_dtype else result


class FactorizedAcousticFNO(nn.Module):
    """
    Factorized Spatiotemporal FNO for acoustic wavefield prediction.

    Architecture:
      Input [B,H,W,T,C]
        → SpatialEncoder: per-frame 2D FNO → [B,H,W,T,spatial_W]
        → TemporalMixer: per-point 1D Conv → [B,H,W,T,temporal_W]
        → OutputHead: MLP → [B,H,W,T]

    Memory advantage: spatial features are computed frame-by-frame (streaming),
    temporal mixing is per-point. No [B,width,H,W,T] dense tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_channels: int = 1,
        spatial_modes_x: int = 32,
        spatial_modes_z: int = 32,
        spatial_width: int = 32,
        spatial_layers: int = 3,
        temporal_width: int = 32,
        temporal_kernel: int = 7,
        temporal_layers: int = 3,
        temporal_mixer_type: str = "local",
        temporal_modes: int = 32,
        padding_ratio: float = 0.0625,
        normalization: str = "group",
        num_groups: int = 8,
        head_hidden: int = 128,
        temporal_chunk_size: int = 4096,
        activation_checkpointing: bool = False,
        spectral_softmax: bool = False,
        time_softmax_steps: int = 0,
        spatial_softmax: bool = False,
        mode_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_channels = int(out_channels)
        self.spatial_width = int(spatial_width)
        self.temporal_width = int(temporal_width)
        self.head_hidden = int(head_hidden)
        self.temporal_chunk_size = int(temporal_chunk_size)
        self.activation_checkpointing = bool(activation_checkpointing)

        self.spatial_encoder = SpatialEncoder(
            in_features=in_features,
            spatial_width=spatial_width,
            spatial_modes_x=int(spatial_modes_x),
            spatial_modes_z=int(spatial_modes_z),
            spatial_layers=int(spatial_layers),
            padding_ratio=float(padding_ratio),
            normalization=normalization,
            num_groups=num_groups,
            spatial_softmax=bool(spatial_softmax),
            mode_softmax=bool(mode_softmax),
        )
        mixer_type = str(temporal_mixer_type).lower()
        if mixer_type == "local":
            self.temporal_mixer = TemporalMixer(
                spatial_width=spatial_width,
                temporal_width=temporal_width,
                temporal_kernel=int(temporal_kernel),
                temporal_layers=int(temporal_layers),
                normalization=normalization,
                num_groups=num_groups,
            )
        elif mixer_type == "spectral":
            self.temporal_mixer = SpectralTemporalMixer(
                spatial_width=spatial_width,
                temporal_width=temporal_width,
                temporal_modes=int(temporal_modes),
                temporal_layers=int(temporal_layers),
                normalization=normalization,
                num_groups=num_groups,
                spectral_softmax=bool(spectral_softmax),
            )
            if int(time_softmax_steps) > 0:
                self.temporal_mixer.enable_time_softmax(int(time_softmax_steps))
        else:
            raise ValueError("temporal_mixer_type must be local or spectral")
        self.spatial_chunk_size = int(temporal_chunk_size)  # reuse for spatial micro-batching
        self.spatial_checkpointing = bool(activation_checkpointing)
        self.head = nn.Sequential(
            nn.Linear(temporal_width, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, T, C]
        Returns:
            [B, H, W, T] if out_channels==1 else [B, H, W, T, out_channels]
        """
        if x.ndim != 5:
            raise ValueError(
                f"FactorizedAcousticFNO expects [B,H,W,T,C], got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"input feature count {x.shape[-1]} != configured {self.in_features}"
            )

        batch, h, w, t, _ = x.shape
        n_spatial = batch * h * w

        # Spatial encoding with micro-batching + optional gradient checkpointing
        spatial_features = self.spatial_encoder(
            x, spatial_chunk_size=self.spatial_chunk_size,
            use_checkpointing=self.spatial_checkpointing,
        )

        # Reshape for per-point temporal mixing
        # [B,H,W,T,spatial_W] → [B*H*W, T, spatial_W]
        flat = spatial_features.reshape(n_spatial, t, self.spatial_width)

        # Chunked temporal mixing to control peak memory.
        # At 400×400, n_spatial=160000 would create a [160000,T,W] tensor (~820 MB
        # at T=160,W=32). Chunking keeps the temporal mixer's working set bounded.
        chunk = int(self.temporal_chunk_size)
        if chunk > 0 and n_spatial > chunk:
            temporal_chunks: list[torch.Tensor] = []
            for start in range(0, n_spatial, chunk):
                end = min(start + chunk, n_spatial)
                temporal_chunks.append(self.temporal_mixer(flat[start:end]))
            temporal_features = torch.cat(temporal_chunks, dim=0)
        else:
            temporal_features = self.temporal_mixer(flat)  # [B*H*W, T, temporal_W]

        # Output head
        # [B*H*W, T, temporal_W] → [B*H*W, T, out_channels]
        out = self.head(temporal_features)

        # Reshape back to [B, H, W, T, out_channels]
        out = out.reshape(batch, h, w, t, self.out_channels)

        if self.out_channels == 1:
            return out[..., 0]
        return out


# ── helpers ──────────────────────────────────────────────────────────────


def _num_groups_for_channels(channels: int, max_groups: int) -> int:
    groups = max(1, min(int(max_groups), int(channels)))
    while int(channels) % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _make_norm_2d(
    norm_type: str,
    channels: int,
    num_groups: int = 8,
) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "group":
        return nn.GroupNorm(
            num_groups=_num_groups_for_channels(channels, num_groups),
            num_channels=channels,
        )
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type in {"none", None}:
        return nn.Identity()
    raise ValueError(f"unsupported 2D normalization: {norm_type}")


def _make_norm_1d(
    norm_type: str,
    channels: int,
    num_groups: int = 8,
) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "group":
        return nn.GroupNorm(
            num_groups=_num_groups_for_channels(channels, num_groups),
            num_channels=channels,
        )
    if norm_type in {"none", None}:
        return nn.Identity()
    if norm_type == "instance":
        return nn.InstanceNorm1d(channels, affine=True)
    raise ValueError(f"unsupported 1D normalization: {norm_type}")
