from __future__ import annotations
import torch
from torch import nn
from ..config import OperatorConfig
from .ufno import UFNOMediumEncoder
from .source import SourceEncoder
from .trunk import CoordinateTrunk
from .query_head import AttentionQueryHead
from .dense_head import DenseWavefieldHead


class GroupedSingleSourceUFNOMIONetOperator(nn.Module):
    """Shared-medium, independent-source neural wave operator.

    ``query_pressure`` and ``predict_wavefield`` retain record dimension. A
    caller may pass several records sharing one velocity tensor via
    ``record_to_medium``; the encoder is then invoked once on unique media.
    """
    def __init__(self, config: OperatorConfig | None = None, *, width=None, rank=None):
        super().__init__()
        cfg = config or OperatorConfig()
        self.config = cfg
        m = cfg.model
        width, rank = width or m.width, rank or m.rank
        self.medium_encoder = UFNOMediumEncoder(width, m.fourier_modes, rank, m.token_grid)
        self.domain_x_m, self.domain_z_m = m.domain_x_m, m.domain_z_m
        self.source_encoder = SourceEncoder(width, rank, m.domain_x_m, m.domain_z_m, width)
        self.trunk = CoordinateTrunk(width, rank, m.domain_x_m, m.domain_z_m)
        self.query_head = AttentionQueryHead(width, rank, m.heads)
        self.dense_head = DenseWavefieldHead(width, rank)

    def _sources(self, source):
        if hasattr(source, "as_tensor"):
            source = source.as_tensor()[None]
        source = torch.as_tensor(source)
        if source.ndim == 1:
            source = source[None]
        if source.ndim != 2 or source.shape[-1] != 5:
            raise ValueError("source must contain exactly one [x,z,f0,t0,amplitude] per record")
        if not torch.isfinite(source).all() or (source[:, 2] <= 0).any() or (source[:, 4] == 0).any():
            raise ValueError("source parameters must be finite, positive-frequency, non-zero-amplitude")
        if (source[:, 0] < 0).any() or (source[:, 0] > self.domain_x_m).any() or (source[:, 1] < 0).any() or (source[:, 1] > self.domain_z_m).any() or (source[:, 3] < 0).any():
            raise ValueError("source coordinates/delay lie outside the configured physical domain")
        return source

    def encode_medium(self, velocity, record_to_medium=None):
        velocity = torch.as_tensor(velocity)
        if velocity.ndim == 3:
            velocity = velocity[None]
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("velocity must have shape [medium,1,z,x]")
        return self.medium_encoder(velocity), record_to_medium

    def prepare(self, velocity, source, record_to_medium=None, source_map=None):
        source = self._sources(source).to(device=next(self.parameters()).device)
        velocity = torch.as_tensor(velocity, dtype=torch.float32, device=source.device)
        if velocity.ndim == 3:
            velocity = velocity[None]
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("velocity must have shape [medium,1,z,x]")
        if not torch.isfinite(velocity).all() or torch.any(velocity <= 0):
            raise ValueError("velocity must be finite and strictly positive")
        if record_to_medium is None:
            record_to_medium = torch.arange(source.shape[0], device=source.device)
            if velocity.shape[0] == 1 and source.shape[0] > 1:
                record_to_medium = torch.zeros(source.shape[0], dtype=torch.long, device=source.device)
        else:
            record_to_medium = torch.as_tensor(record_to_medium, dtype=torch.long, device=source.device)
        if record_to_medium.shape != (source.shape[0],):
            raise ValueError("record_to_medium must have one entry per source record")
        if record_to_medium.numel() and record_to_medium.max() >= velocity.shape[0]:
            raise ValueError("record_to_medium index exceeds velocity batch")
        medium = self.medium_encoder(velocity)
        source_hidden, source_rank = self.source_encoder(source, medium, record_to_medium, source_map)
        return source, record_to_medium, medium, source_hidden, source_rank

    _prepare = prepare

    def query_encoded(self, cache, coords, *, chunk_size=None):
        source, mapping, medium, source_hidden, source_rank = cache
        coords = torch.as_tensor(coords, dtype=torch.float32, device=source.device)
        if coords.ndim == 2:
            coords = coords[None].expand(source.shape[0], -1, -1)
        if coords.ndim != 3 or coords.shape[0] != source.shape[0] or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [records,queries,3]")
        chunks = []
        step = coords.shape[1] if chunk_size is None else max(1, int(chunk_size))
        for start in range(0, coords.shape[1], step):
            current = coords[:, start:start + step]
            hidden, trunk_rank = self.trunk(current)
            raw = self.query_head(hidden, trunk_rank, medium.tokens[mapping], medium.medium_rank[mapping], source_rank, source_hidden)
            t = current[..., 2]
            gate = torch.clamp((t - source[:, None, 3]) / 0.05, min=0.0, max=1.0)
            surface = torch.tanh(torch.clamp(current[..., 1], min=0.0) / 20.0).square()
            chunks.append(raw * gate * surface * source[:, None, 4])
        return torch.cat(chunks, dim=1)

    def query_pressure(self, velocity, source, coords, *, chunk_size=None, record_to_medium=None, source_map=None):
        cache = self.prepare(velocity, source, record_to_medium, source_map)
        return self.query_encoded(cache, coords, chunk_size=chunk_size)

    def predict_wavefield(self, velocity, source, time_s, *, record_to_medium=None, source_map=None):
        source, mapping, medium, source_hidden, source_rank = self._prepare(velocity, source, record_to_medium, source_map)
        local = medium.pyramid[0][mapping]
        pressure = self.dense_head(local, source_hidden, torch.as_tensor(time_s, dtype=torch.float32, device=source.device))
        t = torch.as_tensor(time_s, dtype=torch.float32, device=source.device)
        if t.ndim == 1:
            t = t[None].expand(source.shape[0], -1)
        gate_t = torch.clamp((t - source[:, None, 3]) / 0.05, min=0.0, max=1.0)[:, :, None, None]
        nz = pressure.shape[-2]
        z = torch.linspace(0.0, self.config.model.domain_z_m, nz, device=pressure.device, dtype=pressure.dtype)
        gate_z = torch.tanh(z / 20.0).square()[None, None, :, None]
        return pressure * gate_t * gate_z * source[:, None, 4, None, None]
