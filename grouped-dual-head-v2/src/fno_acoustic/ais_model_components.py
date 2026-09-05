"""Reusable local-query components for AIS-MQFNO model variants."""

from __future__ import annotations

import math
import numbers

import torch
import torch.nn.functional as F
from torch import nn

from .temporal_operator import (
    _ValidatedTimeGrid,
    _make_validated_time_grid,
    _require_validated_time_grid,
)


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_queries(query_xz: torch.Tensor, batch: int) -> None:
    if query_xz.ndim != 3 or query_xz.shape[0] != batch or query_xz.shape[-1] != 2:
        raise ValueError("query_xz must have shape [B,Q,2]")
    if not query_xz.is_floating_point() or not bool(torch.isfinite(query_xz).all()):
        raise ValueError("query_xz must contain finite floating values")
    if bool(torch.any((query_xz < 0) | (query_xz > 1))):
        raise ValueError("normalized query_xz coordinates must lie in [0,1]")


class LocalQueryEncoder(nn.Module):
    """Extract fixed native-grid border-padded halos around normalized sites."""

    def __init__(self, in_channels: int, local_dim: int, halo_size: int = 17) -> None:
        super().__init__()
        if (
            not isinstance(halo_size, int)
            or isinstance(halo_size, bool)
            or halo_size < 1
            or halo_size % 2 == 0
        ):
            raise ValueError("halo_size must be a positive odd integer")
        self.in_channels = _positive_integer("in_channels", in_channels)
        self.local_dim = _positive_integer("local_dim", local_dim)
        self.halo_size = int(halo_size)
        self.encoder = nn.Sequential(
            nn.Linear(self.in_channels * self.halo_size**2, self.local_dim),
            nn.GELU(),
            nn.Linear(self.local_dim, self.local_dim),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def extract_patches(
        self, native_static: torch.Tensor, query_xz: torch.Tensor
    ) -> torch.Tensor:
        """Return ``[B,Q,C,halo,halo]`` patches using an explicit border policy."""

        if native_static.ndim != 4:
            raise ValueError("native_static must have shape [B,C,H,W]")
        batch, channels, height, width = native_static.shape
        if channels != self.in_channels:
            raise ValueError(
                f"native_static channels must equal {self.in_channels}, got {channels}"
            )
        _validate_queries(query_xz, batch)
        if native_static.device != query_xz.device:
            raise ValueError("native_static and query_xz must be on the same device")

        query_grid = query_xz.to(dtype=native_static.dtype)
        radius = self.halo_size // 2
        offset_x = torch.arange(
            -radius, radius + 1, device=query_xz.device, dtype=query_grid.dtype
        ) * (2.0 / max(height - 1, 1))
        offset_z = torch.arange(
            -radius, radius + 1, device=query_xz.device, dtype=query_grid.dtype
        ) * (2.0 / max(width - 1, 1))
        delta_x, delta_z = torch.meshgrid(offset_x, offset_z, indexing="ij")
        center = query_grid.mul(2).sub(1)
        horizontal_z = center[..., 1, None, None] + delta_z
        vertical_x = center[..., 0, None, None] + delta_x
        grid = torch.stack((horizontal_z, vertical_x), dim=-1)
        query_count = query_xz.shape[1]
        grid = grid.reshape(batch, query_count * self.halo_size, self.halo_size, 2)
        patches = F.grid_sample(
            native_static,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return (
            patches.reshape(
                batch, channels, query_count, self.halo_size, self.halo_size
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    def forward(
        self, native_static: torch.Tensor, query_xz: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(self.extract_patches(native_static, query_xz).flatten(2))


class MultiScaleLocalQueryEncoder(nn.Module):
    """Concatenate independent 9x9 and 25x25 native-grid halo encoders."""

    output_dim = 24

    def __init__(self, in_channels: int, branch_dim: int = 12) -> None:
        super().__init__()
        if not isinstance(branch_dim, int) or isinstance(branch_dim, bool) or branch_dim != 12:
            raise ValueError("branch_dim must be exactly 12")
        self.small = LocalQueryEncoder(in_channels, branch_dim, halo_size=9)
        self.large = LocalQueryEncoder(in_channels, branch_dim, halo_size=25)

    def forward(
        self, native_static: torch.Tensor, query_xz: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (self.small(native_static, query_xz), self.large(native_static, query_xz)),
            dim=-1,
        )


class DispersionResidualHead(nn.Module):
    """Real 24-mode phase residual conditioned on local velocity and grid spacing."""

    def __init__(
        self,
        local_dim: int,
        hidden_dim: int = 48,
        modes: int = 24,
        velocity_mean: float = 3000.0,
        velocity_std: float = 500.0,
    ) -> None:
        super().__init__()
        if not isinstance(local_dim, int) or isinstance(local_dim, bool) or local_dim < 1:
            raise ValueError("local_dim must be a positive integer")
        if hidden_dim != 48:
            raise ValueError("dispersion hidden_dim must be exactly 48")
        if modes != 24:
            raise ValueError("dispersion modes must be exactly 24")
        mean = self._finite_real(velocity_mean, "velocity_mean")
        std = self._finite_real(velocity_std, "velocity_std", positive=True)
        self.local_dim = local_dim
        self.hidden_dim = 48
        self.modes = 24
        self.register_buffer("velocity_mean", torch.tensor(mean, dtype=torch.float64))
        self.register_buffer("velocity_std", torch.tensor(std, dtype=torch.float64))
        self.mlp = nn.Sequential(
            nn.Linear(local_dim + 3, 48),
            nn.GELU(),
            nn.Linear(48, 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real number")
        result = float(value)
        if not math.isfinite(result) or (positive and result <= 0.0):
            qualifier = "positive finite" if positive else "finite"
            raise ValueError(f"{name} must be {qualifier}")
        return result

    @staticmethod
    def _validate_condition(condition: torch.Tensor, message: str) -> None:
        """Synchronously reject invalid user data without poisoning CUDA state."""

        if condition.ndim != 0 or condition.dtype != torch.bool:
            raise RuntimeError("dispersion validation condition must be scalar boolean")
        if not bool(condition):
            raise ValueError(message)

    @staticmethod
    def _compute_scalar(
        value: float | torch.Tensor,
        reference: torch.Tensor,
        name: str,
        *,
        host_validated: bool = False,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.ndim != 0 or not value.is_floating_point():
                raise ValueError(f"{name} must be a scalar floating value")
            return value.to(reference)
        if not host_validated:
            DispersionResidualHead._finite_real(value, name, positive=True)
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

    def _dispersion_features(
        self,
        velocity_hat: torch.Tensor,
        dx_m: float,
        dz_m: float,
        duration_s: float | torch.Tensor,
        *,
        spacing_host_validated: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dx = self._compute_scalar(
            dx_m, velocity_hat, "dx_m", host_validated=spacing_host_validated
        )
        dz = self._compute_scalar(
            dz_m, velocity_hat, "dz_m", host_validated=spacing_host_validated
        )
        duration = self._compute_scalar(duration_s, velocity_hat, "duration_s")
        mean = self.velocity_mean.to(velocity_hat)
        std = self.velocity_std.to(velocity_hat)
        scalar_condition = (
            torch.isfinite(torch.stack((mean, std, dx, dz, duration))).all()
            & (std > 0)
            & (dx > 0)
            & (dz > 0)
            & (duration > 0)
        )
        velocity = mean + std * velocity_hat
        mode = torch.arange(
            1, self.modes + 1, dtype=velocity_hat.dtype, device=velocity_hat.device
        )
        frequency = mode / duration
        mode_fraction = (mode / self.modes).view(1, 1, -1).expand(
            *velocity_hat.shape, -1
        )
        inverse_velocity = velocity.reciprocal()[..., None]
        features = torch.stack(
            (
                mode_fraction,
                frequency.view(1, 1, -1) * dx * inverse_velocity,
                frequency.view(1, 1, -1) * dz * inverse_velocity,
            ),
            dim=-1,
        )
        computed_condition = (
            torch.isfinite(velocity).all()
            & (velocity > 0).all()
            & torch.isfinite(frequency).all()
            & torch.isfinite(features).all()
        )
        return features, scalar_condition & computed_condition

    def dispersion_features(
        self,
        velocity_hat: torch.Tensor,
        dx_m: float,
        dz_m: float,
        duration_s: float | torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(velocity_hat, torch.Tensor) or velocity_hat.ndim != 2:
            raise ValueError("velocity_hat must have shape [B,Q]")
        if not velocity_hat.is_floating_point():
            raise ValueError("velocity_hat must contain finite floating values")
        features, condition = self._dispersion_features(
            velocity_hat,
            dx_m,
            dz_m,
            duration_s,
            spacing_host_validated=False,
        )
        self._validate_condition(
            torch.isfinite(velocity_hat).all() & condition,
            "physical query velocity, modal frequencies, and dispersion features "
            "must be finite, positive, and representable",
        )
        return features

    @staticmethod
    def _shared_full160_time(time_s: torch.Tensor, batch: int) -> torch.Tensor:
        if not isinstance(time_s, torch.Tensor) or not time_s.is_floating_point():
            raise ValueError("time_s must contain 160 floating values")
        if time_s.ndim == 1 and time_s.shape == (160,):
            time = time_s
        elif time_s.ndim == 2 and time_s.shape == (batch, 160):
            if not torch.equal(time_s, time_s[0].expand_as(time_s)):
                raise ValueError("batched time_s must contain one shared time vector")
            time = time_s[0]
        else:
            raise ValueError("time_s must contain exactly 160 saved times")
        condition = torch.isfinite(time).all() & torch.all(torch.diff(time) > 0)
        DispersionResidualHead._validate_condition(
            condition, "time_s must be finite and strictly increasing"
        )
        return time

    def forward(
        self,
        local_embedding: torch.Tensor,
        velocity_hat: torch.Tensor,
        dx_m: float,
        dz_m: float,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not isinstance(local_embedding, torch.Tensor)
            or local_embedding.ndim != 3
            or local_embedding.shape[-1] != self.local_dim
        ):
            raise ValueError(
                f"local_embedding must have shape [B,Q,{self.local_dim}]"
            )
        time = self._shared_full160_time(time_s, local_embedding.shape[0])
        return self._forward_validated(
            local_embedding,
            velocity_hat,
            dx_m,
            dz_m,
            _make_validated_time_grid(time),
            spacing_host_validated=False,
        )

    def _forward_validated(
        self,
        local_embedding: torch.Tensor,
        velocity_hat: torch.Tensor,
        dx_m: float,
        dz_m: float,
        time_grid: _ValidatedTimeGrid,
        *,
        spacing_host_validated: bool,
    ) -> torch.Tensor:
        if (
            not isinstance(local_embedding, torch.Tensor)
            or local_embedding.ndim != 3
            or local_embedding.shape[-1] != self.local_dim
        ):
            raise ValueError(
                f"local_embedding must have shape [B,Q,{self.local_dim}]"
            )
        if velocity_hat.shape != local_embedding.shape[:2]:
            raise ValueError("velocity_hat must match local embedding batch and queries")
        if not local_embedding.is_floating_point() or not velocity_hat.is_floating_point():
            raise ValueError("local_embedding must contain finite floating values")
        input_condition = (
            torch.isfinite(local_embedding).all()
            & torch.isfinite(velocity_hat).all()
        )
        time = _require_validated_time_grid(time_grid)
        if time.shape != (160,):
            raise ValueError("time_s must contain exactly 160 saved times")
        duration = time[-1] - time[0]
        velocity = velocity_hat.to(local_embedding)
        dispersion, feature_condition = self._dispersion_features(
            velocity,
            dx_m,
            dz_m,
            duration.to(local_embedding),
            spacing_host_validated=spacing_host_validated,
        )
        local = local_embedding[:, :, None, :].expand(-1, -1, self.modes, -1)
        coefficients = self.mlp(torch.cat((local, dispersion), dim=-1))
        mode = torch.arange(
            1, self.modes + 1,
            dtype=local_embedding.dtype,
            device=local_embedding.device,
        )
        frequency = mode / duration.to(local_embedding)
        phase = 2 * torch.pi * (time.to(local_embedding) - time[0].to(local_embedding))[
            :, None
        ] * frequency[None, :]
        cosine, sine = torch.cos(phase), torch.sin(phase)
        residual = (
            coefficients[..., 0] @ cosine.transpose(0, 1)
            + coefficients[..., 1] @ sine.transpose(0, 1)
        ) / math.sqrt(self.modes)
        dynamic_condition = (
            input_condition
            & feature_condition
            & torch.isfinite(coefficients).all()
            & torch.isfinite(phase).all()
            & torch.isfinite(residual).all()
        )
        self._validate_condition(
            dynamic_condition,
            "local embeddings, physical query velocity, dispersion features, phase, "
            "and residual output must be finite, positive, and representable",
        )
        return residual


__all__ = [
    "DispersionResidualHead",
    "LocalQueryEncoder",
    "MultiScaleLocalQueryEncoder",
]
