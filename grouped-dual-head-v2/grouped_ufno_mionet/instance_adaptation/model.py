"""Frozen grouped operator with a small onset-conditioned full-field adapter."""
from __future__ import annotations

import torch
from torch import nn

from .adapters import SpatialTemporalFiLM
from .causal import hard_project_observations


class OnsetAdaptedOperator(nn.Module):
    """Expose a complete wavefield while adapting only rank-basis FiLM parameters."""

    def __init__(self, base: nn.Module, adapter_rank: int = 2):
        super().__init__()
        if adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        rank = int(base.dense_head.spatial.out_channels)
        self.film = SpatialTemporalFiLM(rank)

    def trainable_parameters(self):
        return tuple(parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, velocity, source, time_s, observed_wavefield, observed_indices, *, source_map=None):
        source, mapping, medium, source_hidden, _ = self.base.prepare(velocity, source, source_map=source_map)
        local = medium.pyramid[0][mapping]
        spatial = self.base.dense_head.spatial(local) + self.base.dense_head.source_bias(source_hidden)[:, :, None, None]
        times = torch.as_tensor(time_s, dtype=torch.float32, device=source.device)
        if times.ndim == 1:
            times = times[None].expand(source.shape[0], -1)
        coords = torch.stack((torch.zeros_like(times), torch.zeros_like(times), times), dim=-1)
        temporal = self.base.dense_head.time(coords)
        basis = temporal[:, :, :, None, None] * spatial[:, None]
        pressure = self.film(basis).sum(dim=2)
        gate_t = torch.clamp((times - source[:, None, 3]) / 0.05, 0.0, 1.0)[:, :, None, None]
        z = torch.linspace(0.0, self.base.config.model.domain_z_m, pressure.shape[-2], device=pressure.device)
        gate_z = torch.tanh(z / 20.0).square()[None, None, :, None]
        pressure = pressure * gate_t * gate_z * source[:, None, 4, None, None]
        return hard_project_observations(pressure, observed_wavefield, observed_indices)
