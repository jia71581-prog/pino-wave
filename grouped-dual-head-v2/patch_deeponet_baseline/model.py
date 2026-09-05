"""Parameter-matched local Patch-DeepONet for acoustic wavefields.

The branch preserves a downsampled spatial latent field instead of globally
pooling the medium. Query-local branch features interact with an independent
time/source trunk through the standard DeepONet inner product. No proposed-model
weights, predictions, pressure observations, or target-derived features enter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PatchDeepONetConfig:
    static_channels: int = 4
    query_channels: int = 10
    # Most of the parameter budget belongs to the once-per-medium spatial
    # branch. Keeping the per-query trunk compact is essential for a fair dense
    # 401 x 201 x 201 comparison; a parameter-matched 3k-wide query MLP would
    # make the baseline computationally unusable for reasons unrelated to its
    # operator formulation.
    branch_width: int = 64
    branch_blocks: int = 4
    branch_downsample_stages: int = 4
    branch_global_width: int = 5591
    branch_bottleneck_width: int = 32
    latent_dim: int = 64
    trunk_width: int = 48
    trunk_depth: int = 4
    fourier_bands: int = 16
    fourier_coordinate_indices: tuple[int, ...] = (2, 9)
    local_residual_width: int = 32
    ricker_bias_init: float = 0.1
    target_parameters: int = 32_420_564
    parameter_tolerance_fraction: float = 0.001

    def validate(self) -> None:
        integers = (
            self.static_channels,
            self.query_channels,
            self.branch_width,
            self.branch_blocks,
            self.branch_global_width,
            self.branch_bottleneck_width,
            self.latent_dim,
            self.trunk_width,
            self.trunk_depth,
            self.fourier_bands,
            self.local_residual_width,
            self.target_parameters,
        )
        if any(int(value) <= 0 for value in integers):
            raise ValueError("Patch-DeepONet dimensions and parameter target must be positive")
        if self.trunk_depth < 3:
            raise ValueError("Patch-DeepONet trunk depth must be at least three")
        if not 0 <= int(self.branch_downsample_stages) <= 4:
            raise ValueError("branch downsample stages must lie in [0,4]")
        coordinate_indices = tuple(int(value) for value in self.fourier_coordinate_indices)
        if (
            not coordinate_indices
            or len(set(coordinate_indices)) != len(coordinate_indices)
            or any(value < 0 or value >= self.query_channels for value in coordinate_indices)
        ):
            raise ValueError("Fourier coordinate indices must be unique query-channel indices")
        if not 0.0 <= float(self.parameter_tolerance_fraction) <= 0.01:
            raise ValueError("parameter tolerance must lie in [0,0.01]")
        if not math.isfinite(float(self.ricker_bias_init)):
            raise ValueError("Ricker bias initialization must be finite")


class _StaticResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = max(group for group in (8, 4, 2, 1) if width % group == 0)
        self.norm = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.gelu(self.norm(value)))
        return value + self.conv2(F.gelu(hidden))


class _TrunkResidualBlock(nn.Module):
    """One-matrix pre-normalized residual block for an auditable capacity budget."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + F.gelu(self.linear(self.norm(value)))


class PatchDeepONet(nn.Module):
    """Local branch/trunk product producing normalized scalar pressure queries."""

    def __init__(self, config: PatchDeepONetConfig | None = None) -> None:
        super().__init__()
        self.config = config or PatchDeepONetConfig()
        self.config.validate()
        c = self.config
        stem: list[nn.Module] = []
        for stage in range(4):
            stem.extend(
                (
                    nn.Conv2d(
                        c.static_channels if stage == 0 else c.branch_width,
                        c.branch_width,
                        5 if stage == 0 else 3,
                        stride=2 if stage < c.branch_downsample_stages else 1,
                        padding=2 if stage == 0 else 1,
                    ),
                    nn.GroupNorm(8, c.branch_width),
                    nn.GELU(),
                )
            )
        self.static_stem = nn.Sequential(*stem)
        self.static_blocks = nn.Sequential(
            *(_StaticResidualBlock(c.branch_width) for _ in range(c.branch_blocks))
        )
        self.branch_bottleneck = nn.Sequential(
            nn.Conv2d(c.branch_width, c.branch_bottleneck_width, 1),
            nn.GELU(),
            nn.Conv2d(c.branch_bottleneck_width, c.branch_width, 1),
        )
        self.branch_global = nn.Sequential(
            nn.Linear(c.branch_width, c.branch_global_width),
            nn.GELU(),
            nn.Linear(c.branch_global_width, c.branch_global_width),
            nn.GELU(),
            nn.Linear(c.branch_global_width, c.latent_dim),
        )
        self.branch_projection = nn.Conv2d(c.branch_width, c.latent_dim, 1)

        trunk_input = c.query_channels + 2 * len(c.fourier_coordinate_indices) * c.fourier_bands
        self.trunk_input = nn.Linear(trunk_input, c.trunk_width)
        self.trunk_blocks = nn.Sequential(
            *(_TrunkResidualBlock(c.trunk_width) for _ in range(c.trunk_depth - 2))
        )
        self.trunk_norm = nn.LayerNorm(c.trunk_width)
        self.trunk_output = nn.Linear(c.trunk_width, c.latent_dim)

        self.local_residual = nn.Sequential(
            nn.Linear(c.static_channels + c.query_channels, c.local_residual_width),
            nn.GELU(),
            nn.Linear(c.local_residual_width, 1),
        )
        self.ricker_bias = nn.Parameter(torch.tensor(float(c.ricker_bias_init)))
        bands = 2.0 ** torch.arange(c.fourier_bands, dtype=torch.float32)
        self.register_buffer("fourier_bands", bands, persistent=True)

    def encode_static(self, static_features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(static_features, dtype=torch.float32)
        if features.ndim != 4 or features.shape[1] != self.config.static_channels:
            raise ValueError("static features must be [record,4,z,x]")
        if not torch.isfinite(features).all():
            raise ValueError("static features must be finite")
        hidden = self.static_blocks(self.static_stem(features))
        hidden = hidden + self.branch_bottleneck(hidden)
        local = self.branch_projection(hidden)
        global_branch = self.branch_global(hidden.mean(dim=(-2, -1)))
        return local + global_branch[:, :, None, None]

    @staticmethod
    def _sample(field: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
        if query_xy.ndim != 3 or query_xy.shape[-1] != 2:
            raise ValueError("query coordinates must be [record,query,2]")
        sampled = F.grid_sample(
            field,
            query_xy[:, :, None, :],
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def _trunk_features(self, descriptors: torch.Tensor) -> torch.Tensor:
        # Only time and arrival coordinates receive harmonic expansion. Source
        # frequency remains fixed in the registered position experiment but stays
        # an ordinary conditioning scalar for the general training baseline.
        coordinates = descriptors[..., list(self.config.fourier_coordinate_indices)]
        phase = 2.0 * math.pi * coordinates[..., None] * self.fourier_bands
        harmonics = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        return torch.cat((descriptors, harmonics), dim=-1)

    def query_encoded(
        self,
        static_features: torch.Tensor,
        encoded_static: torch.Tensor,
        query_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        descriptors = torch.as_tensor(
            query_descriptors, dtype=encoded_static.dtype, device=encoded_static.device
        )
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.config.query_channels:
            raise ValueError("query descriptors must be [record,query,10]")
        if descriptors.shape[0] != encoded_static.shape[0] or not torch.isfinite(descriptors).all():
            raise ValueError("query descriptors and encoded records do not match")
        query_xy = descriptors[..., :2]
        branch = self._sample(encoded_static, query_xy)
        hidden = F.gelu(self.trunk_input(self._trunk_features(descriptors)))
        trunk = self.trunk_output(self.trunk_norm(self.trunk_blocks(hidden)))
        product = (branch * trunk).sum(dim=-1) / math.sqrt(float(self.config.latent_dim))
        local_static = self._sample(static_features, query_xy)
        residual = self.local_residual(torch.cat((local_static, descriptors), dim=-1)).squeeze(-1)
        arrival = descriptors[..., 9]
        ricker = (1.0 - 2.0 * math.pi**2 * arrival.square()) * torch.exp(
            -math.pi**2 * arrival.square()
        )
        return product + residual + self.ricker_bias * ricker

    def forward(
        self,
        static_features: torch.Tensor,
        query_descriptors: torch.Tensor,
        *,
        query_chunk: int | None = None,
    ) -> torch.Tensor:
        encoded = self.encode_static(static_features)
        descriptors = torch.as_tensor(query_descriptors)
        chunk = descriptors.shape[1] if query_chunk is None else int(query_chunk)
        if chunk <= 0:
            raise ValueError("query chunk must be positive")
        return torch.cat(
            [
                self.query_encoded(static_features, encoded, descriptors[:, start : start + chunk])
                for start in range(0, descriptors.shape[1], chunk)
            ],
            dim=1,
        )

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def parameter_match(self) -> dict[str, float | int | bool]:
        count = self.parameter_count()
        target = int(self.config.target_parameters)
        relative = abs(count - target) / float(target)
        return {
            "parameter_count": count,
            "target_parameters": target,
            "absolute_difference": count - target,
            "relative_difference": relative,
            "within_tolerance": relative <= float(self.config.parameter_tolerance_fraction),
        }


def resolve_parameter_matched_width(
    config: PatchDeepONetConfig | None = None,
    *,
    search_radius: int = 128,
) -> PatchDeepONetConfig:
    """Return the nearest trunk width to the frozen parameter target."""

    base = config or PatchDeepONetConfig()
    base.validate()
    if int(search_radius) < 0:
        raise ValueError("search radius must be nonnegative")
    candidates = range(max(1, base.trunk_width - search_radius), base.trunk_width + search_radius + 1)
    best = min(
        candidates,
        key=lambda width: abs(
            PatchDeepONet(replace(base, trunk_width=width)).parameter_count()
            - base.target_parameters
        ),
    )
    resolved = replace(base, trunk_width=int(best))
    if not bool(PatchDeepONet(resolved).parameter_match()["within_tolerance"]):
        raise ValueError("no parameter-matched trunk width lies in the bounded search")
    return resolved


__all__ = ["PatchDeepONet", "PatchDeepONetConfig", "resolve_parameter_matched_width"]
