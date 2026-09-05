from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .ufno import MediumEncoding


class SourceEncoder(nn.Module):
    def __init__(self, width: int, rank: int, domain_x_m=2000.0, domain_z_m=2000.0, out_dim=64):
        super().__init__()
        self.domain_x_m, self.domain_z_m = domain_x_m, domain_z_m
        self.mlp = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, out_dim), nn.GELU())
        self.local = nn.Linear(width, out_dim)
        self.rank = nn.Linear(out_dim, rank)
        self.map_proj = nn.Sequential(nn.Linear(16, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, source: torch.Tensor, medium: MediumEncoding, record_to_medium=None, source_map=None):
        if source.ndim != 2 or source.shape[1] != 5:
            raise ValueError("source must have shape [records,5] (one source per record)")
        source = source.float()
        if source_map is not None:
            source_map = torch.as_tensor(source_map, device=source.device, dtype=source.dtype)
            if source_map.ndim == 3:
                source_map = source_map[:, None]
            if source_map.ndim != 4 or source_map.shape[0] != source.shape[0] or source_map.shape[1] != 1:
                raise ValueError("source_map must have shape [records,1,z,x]")
            if not torch.isfinite(source_map).all() or torch.any(source_map < 0):
                raise ValueError("source_map must be finite and non-negative")
            mass = source_map.sum(dim=(-2, -1))
            if not torch.allclose(mass, torch.ones_like(mass), atol=2e-4, rtol=2e-4):
                raise ValueError("each source_map must have unit mass")
        if record_to_medium is None:
            record_to_medium = torch.arange(source.shape[0], device=source.device)
        medium_local = medium.pyramid[0][record_to_medium]
        gx = source[:, 0] / self.domain_x_m * 2 - 1
        gz = source[:, 1] / self.domain_z_m * 2 - 1
        grid = torch.stack((gx, gz), dim=-1).view(-1, 1, 1, 2)
        sampled = torch.nn.functional.grid_sample(medium_local, grid, mode="bilinear", align_corners=True).flatten(1)
        # Use a compact spatial average so source parameters remain the primary
        # source identity while local velocity information is retained.
        local = sampled[:, : self.local.in_features]
        # Amplitude is intentionally *not* folded into this latent.  It is
        # applied once as an explicit physical scale in both public heads;
        # consequently changing amplitude has the required linear effect and
        # cannot accidentally alter source identity through a nonlinear MLP.
        feat = self.mlp(torch.stack((source[:, 0] / self.domain_x_m, source[:, 1] / self.domain_z_m, source[:, 2] / 50.0, source[:, 3], torch.zeros_like(source[:, 4])), dim=-1))
        feat = feat + self.local(local)
        # Encode the full (possibly sub-grid) source distribution separately;
        # the coordinates remain the primary source identity.
        if source_map is not None:
            map_features = F.adaptive_avg_pool2d(source_map, (4, 4)).flatten(1)
            feat = feat + self.map_proj(map_features)
        return feat, self.rank(feat)
