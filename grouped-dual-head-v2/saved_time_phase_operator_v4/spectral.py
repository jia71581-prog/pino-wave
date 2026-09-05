"""Deep axis-factorized learned-complex spectral residual layers."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .boundary_consistency import crop_pressure_halo, odd_top_three_side_halo


def _group_count(width: int) -> int:
    for groups in (8, 4, 2, 1):
        if width % groups == 0:
            return groups
    return 1


class AxisFactorizedComplexSpectralConv2d(nn.Module):
    """Sum independent learned complex contractions along x and z."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        modes: int,
        fft_norm: str = "ortho",
        boundary_halo_radius: int = 0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or modes <= 0:
            raise ValueError("factorized spectral channels and modes must be positive")
        if fft_norm not in {"backward", "forward", "ortho"}:
            raise ValueError(f"unsupported FFT normalization: {fft_norm}")
        if int(boundary_halo_radius) < 0:
            raise ValueError("boundary_halo_radius must be nonnegative")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        self.fft_norm = str(fft_norm)
        self.boundary_halo_radius = int(boundary_halo_radius)
        shape = (self.in_channels, self.out_channels, self.modes, 2)
        self.weight_x = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.weight_z = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        nn.init.uniform_(self.weight_x, -scale, scale)
        nn.init.uniform_(self.weight_z, -scale, scale)

    def _transform_x(self, real_value: torch.Tensor) -> torch.Tensor:
        height, width = real_value.shape[-2:]
        modes_x = min(self.modes, width // 2 + 1)
        x_fft = torch.fft.rfft(real_value, dim=-1, norm=self.fft_norm)
        x_output = torch.zeros(
            real_value.shape[0],
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=x_fft.dtype,
            device=real_value.device,
        )
        weight_x = torch.view_as_complex(self.weight_x.contiguous())[:, :, :modes_x]
        x_output[..., :modes_x] = torch.einsum(
            "bihw,iow->bohw", x_fft[..., :modes_x], weight_x
        )
        return torch.fft.irfft(x_output, n=width, dim=-1, norm=self.fft_norm)

    def _transform_z(self, real_value: torch.Tensor) -> torch.Tensor:
        height, width = real_value.shape[-2:]
        modes_z = min(self.modes, height // 2 + 1)
        z_fft = torch.fft.rfft(real_value, dim=-2, norm=self.fft_norm)
        z_output = torch.zeros(
            real_value.shape[0],
            self.out_channels,
            height // 2 + 1,
            width,
            dtype=z_fft.dtype,
            device=real_value.device,
        )
        weight_z = torch.view_as_complex(self.weight_z.contiguous())[:, :, :modes_z]
        z_output[:, :, :modes_z] = torch.einsum(
            "bihw,ioh->bohw", z_fft[:, :, :modes_z], weight_z
        )
        return torch.fft.irfft(z_output, n=height, dim=-2, norm=self.fft_norm)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.in_channels:
            raise ValueError(
                f"factorized spectral input must be [batch,{self.in_channels},z,x]"
            )
        with torch.autocast(device_type=value.device.type, enabled=False):
            real_value = value.float()
            if self.boundary_halo_radius > 0:
                real_value = odd_top_three_side_halo(
                    real_value, radius=self.boundary_halo_radius
                )
            transformed = self._transform_x(real_value) + self._transform_z(real_value)
            if self.boundary_halo_radius > 0:
                transformed = crop_pressure_halo(
                    transformed, radius=self.boundary_halo_radius
                )
            return transformed

    def coupled(self, value: torch.Tensor) -> torch.Tensor:
        """Compose both axis operators to obtain a separable 2-D response."""

        if self.in_channels != self.out_channels:
            raise ValueError("coupled axis transform requires equal channel dimensions")
        if value.ndim != 4 or value.shape[1] != self.in_channels:
            raise ValueError(
                f"factorized spectral input must be [batch,{self.in_channels},z,x]"
            )
        with torch.autocast(device_type=value.device.type, enabled=False):
            real_value = value.float()
            x_then_z = self._transform_z(self._transform_x(real_value))
            z_then_x = self._transform_x(self._transform_z(real_value))
            return (x_then_z + z_then_x) / math.sqrt(2.0)


class LocalDifferentialResidual2d(nn.Module):
    """Zero-gated learned mixture of fixed local derivatives and a dilated path."""

    def __init__(
        self,
        width: int,
        *,
        grid_height: int = 201,
        grid_width: int = 201,
    ) -> None:
        super().__init__()
        channels = int(width)
        height = int(grid_height)
        columns = int(grid_width)
        if channels <= 0 or height <= 1 or columns <= 1:
            raise ValueError("local differential dimensions must be positive")
        dx = 1.0 / float(columns - 1)
        dz = 1.0 / float(height - 1)
        basis = torch.tensor(
            (
                ((0.0, 0.0, 0.0), (-0.5 / dx, 0.0, 0.5 / dx), (0.0, 0.0, 0.0)),
                ((0.0, -0.5 / dz, 0.0), (0.0, 0.0, 0.0), (0.0, 0.5 / dz, 0.0)),
                ((0.0, 0.0, 0.0), (1.0 / dx**2, -2.0 / dx**2, 1.0 / dx**2), (0.0, 0.0, 0.0)),
                ((0.0, 1.0 / dz**2, 0.0), (0.0, -2.0 / dz**2, 0.0), (0.0, 1.0 / dz**2, 0.0)),
                (
                    (0.25 / (dx * dz), 0.0, -0.25 / (dx * dz)),
                    (0.0, 0.0, 0.0),
                    (-0.25 / (dx * dz), 0.0, 0.25 / (dx * dz)),
                ),
            ),
            dtype=torch.float32,
        )
        self.width = channels
        self.register_buffer("basis", basis[:, None], persistent=True)
        self.coefficients = nn.Parameter(torch.ones(channels, len(basis)))
        self.dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
        )
        self.projection = nn.Conv2d(channels * 6, channels, kernel_size=1)
        nn.init.normal_(self.projection.weight, mean=0.0, std=1.0e-5)
        nn.init.zeros_(self.projection.bias)
        self.scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.width:
            raise ValueError(
                f"local differential input must be [batch,{self.width},z,x]"
            )
        with torch.autocast(device_type=value.device.type, enabled=False):
            real_value = value.float()
            padded = F.pad(real_value, (1, 1, 1, 1), mode="replicate")
            kernels = self.basis.repeat(self.width, 1, 1, 1)
            raw = F.conv2d(padded, kernels, groups=self.width)
            raw = raw.reshape(
                value.shape[0], self.width, 5, value.shape[-2], value.shape[-1]
            )
            derivatives = raw * self.coefficients[None, :, :, None, None]
            features = torch.cat(
                (derivatives.flatten(1, 2), self.dilated(real_value)), dim=1
            )
            return torch.tanh(self.scale) * self.projection(features)


class LowRankCoupledSpectralResidual2d(nn.Module):
    """Zero-gated CP-factorized response over the complete 2-D rFFT grid."""

    def __init__(
        self,
        width: int,
        *,
        coupling_rank: int,
        grid_height: int = 201,
        grid_width: int = 201,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()
        channels = int(width)
        rank = int(coupling_rank)
        height = int(grid_height)
        columns = int(grid_width)
        if channels <= 0 or rank <= 0 or height <= 1 or columns <= 1:
            raise ValueError("coupled spectral dimensions must be positive")
        if fft_norm not in {"backward", "forward", "ortho"}:
            raise ValueError(f"unsupported FFT normalization: {fft_norm}")
        self.width = channels
        self.coupling_rank = rank
        self.grid_height = height
        self.grid_width = columns
        self.fft_norm = str(fft_norm)
        self.channel_in = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.channel_out = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        self.factor_z = nn.Parameter(torch.empty(rank, height, 2))
        self.factor_x = nn.Parameter(torch.empty(rank, columns // 2 + 1, 2))
        self.scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.channel_in.weight, a=math.sqrt(5.0))
        nn.init.kaiming_uniform_(self.channel_out.weight, a=math.sqrt(5.0))
        with torch.no_grad():
            self.factor_z.zero_()
            self.factor_x.zero_()
            self.factor_z[..., 0].normal_(mean=1.0, std=1.0e-2)
            self.factor_x[..., 0].normal_(mean=1.0, std=1.0e-2)
            self.scale.zero_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        expected = (self.grid_height, self.grid_width)
        if value.ndim != 4 or value.shape[1] != self.width:
            raise ValueError(
                f"coupled spectral input must be [batch,{self.width},z,x]"
            )
        if tuple(value.shape[-2:]) != expected:
            raise ValueError(
                f"coupled spectral grid must be {expected}, got {tuple(value.shape[-2:])}"
            )
        with torch.autocast(device_type=value.device.type, enabled=False):
            latent = self.channel_in(value.float())
            spectrum = torch.fft.rfft2(latent, norm=self.fft_norm)
            factor_z = torch.view_as_complex(self.factor_z.contiguous())[:, :, None]
            factor_x = torch.view_as_complex(self.factor_x.contiguous())[:, None, :]
            filtered = spectrum * (factor_z * factor_x)[None]
            field = torch.fft.irfft2(
                filtered,
                s=expected,
                norm=self.fft_norm,
            )
            return torch.tanh(self.scale) * self.channel_out(field)


class FactorizedComplexResidualBlock(nn.Module):
    """Pre-normalized factorized spectral and V2 local residual update."""

    def __init__(
        self,
        *,
        width: int,
        spectral_rank: int,
        modes: int,
        coupled_axes: bool = False,
        local_differential_residual: bool = False,
        coupled_2d_rank: int = 0,
        boundary_halo_radius: int = 0,
    ) -> None:
        super().__init__()
        if width <= 0 or spectral_rank <= 0:
            raise ValueError("factorized residual widths must be positive")
        self.norm = nn.GroupNorm(_group_count(width), width)
        self.in_projection = nn.Conv2d(width, spectral_rank, kernel_size=1)
        self.spectral = AxisFactorizedComplexSpectralConv2d(
            spectral_rank,
            spectral_rank,
            modes=modes,
            boundary_halo_radius=int(boundary_halo_radius),
        )
        self.out_projection = nn.Conv2d(spectral_rank, width, kernel_size=1)
        self.local_depthwise = nn.Conv2d(
            width, width, kernel_size=3, padding=1, groups=width
        )
        self.local_pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.local_differential = (
            LocalDifferentialResidual2d(width)
            if bool(local_differential_residual)
            else None
        )
        self.coupled_2d = (
            LowRankCoupledSpectralResidual2d(width, coupling_rank=coupled_2d_rank)
            if int(coupled_2d_rank) > 0
            else None
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.coupled_axes = bool(coupled_axes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError("factorized residual input must be [batch,channel,z,x]")
        normalized = self.norm(value)
        projected = self.in_projection(normalized)
        spectral_latent = self.spectral(projected)
        if self.coupled_axes:
            spectral_latent = spectral_latent + torch.tanh(
                self.residual_scale
            ) * self.spectral.coupled(projected)
        spectral = self.out_projection(spectral_latent)
        local = self.local_pointwise(self.local_depthwise(normalized))
        differential = (
            torch.zeros_like(local)
            if self.local_differential is None
            else self.local_differential(normalized)
        )
        coupled_2d = (
            torch.zeros_like(local)
            if self.coupled_2d is None
            else self.coupled_2d(normalized)
        )
        return value + self.residual_scale * F.gelu(
            spectral + local + differential + coupled_2d
        )


class FactorizedComplexResidualStack(nn.Module):
    """Deep residual-fork stack with optional activation recomputation."""

    def __init__(
        self,
        width: int,
        spectral_rank: int,
        modes: int,
        depth: int,
        *,
        activation_checkpointing: bool = True,
        fork_every: int = 2,
        coupled_axes: bool = False,
        local_differential_residual: bool = False,
        coupled_2d_rank: int = 0,
        boundary_halo_radius: int = 0,
    ) -> None:
        super().__init__()
        if depth <= 0 or fork_every <= 0:
            raise ValueError("factorized stack depth and fork interval must be positive")
        self.activation_checkpointing = bool(activation_checkpointing)
        self.fork_every = int(fork_every)
        self.blocks = nn.ModuleList(
            FactorizedComplexResidualBlock(
                width=int(width),
                spectral_rank=int(spectral_rank),
                modes=int(modes),
                coupled_axes=bool(coupled_axes),
                local_differential_residual=bool(local_differential_residual),
                coupled_2d_rank=int(coupled_2d_rank),
                boundary_halo_radius=int(boundary_halo_radius),
            )
            for _ in range(int(depth))
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = value
        fork = value
        for index, block in enumerate(self.blocks):
            if self.activation_checkpointing and self.training and hidden.requires_grad:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
            if (index + 1) % self.fork_every == 0:
                hidden = (hidden + fork) / math.sqrt(2.0)
                fork = hidden
        return hidden


@torch.no_grad()
def transfer_expanded_spectral_modes(
    parent: FactorizedComplexResidualStack,
    candidate: FactorizedComplexResidualStack,
) -> dict[str, int]:
    """Copy a trained stack into a wider-mode stack without changing its output."""

    parent_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    if set(parent_state) != set(candidate_state):
        raise ValueError("spectral mode transfer requires identical parameter names")
    transferred: dict[str, torch.Tensor] = {}
    source_modes: set[int] = set()
    target_modes: set[int] = set()
    expanded = 0
    spectral_suffixes = ("spectral.weight_x", "spectral.weight_z")
    for name, target in candidate_state.items():
        source = parent_state[name]
        if source.shape == target.shape:
            transferred[name] = source.to(device=target.device, dtype=target.dtype)
            continue
        valid = (
            name.endswith(spectral_suffixes)
            and source.ndim == target.ndim == 4
            and source.shape[:2] == target.shape[:2]
            and source.shape[3:] == target.shape[3:]
            and target.shape[2] > source.shape[2]
        )
        if not valid:
            raise ValueError(
                f"spectral mode transfer must strictly expand {name}: "
                f"{tuple(source.shape)} -> {tuple(target.shape)}"
            )
        value = torch.zeros_like(target)
        value[:, :, : source.shape[2], :] = source.to(
            device=target.device, dtype=target.dtype
        )
        transferred[name] = value
        source_modes.add(int(source.shape[2]))
        target_modes.add(int(target.shape[2]))
        expanded += 1
    if expanded == 0 or len(source_modes) != 1 or len(target_modes) != 1:
        raise ValueError("spectral mode transfer must strictly expand one mode width")
    candidate.load_state_dict(transferred, strict=True)
    return {
        "source_modes": source_modes.pop(),
        "target_modes": target_modes.pop(),
        "expanded_tensors": int(expanded),
    }


__all__ = [
    "AxisFactorizedComplexSpectralConv2d",
    "FactorizedComplexResidualBlock",
    "FactorizedComplexResidualStack",
    "LocalDifferentialResidual2d",
    "LowRankCoupledSpectralResidual2d",
    "transfer_expanded_spectral_modes",
]
