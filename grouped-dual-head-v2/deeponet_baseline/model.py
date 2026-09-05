"""Pure DeepONet baseline for the saved-time acoustic operator comparison.

This is the textbook DeepONet contrast to our SavedTimePhaseOperatorV4:
  * Branch: encodes the input functions (velocity model + source map, plus the
    5 scalar source parameters) into a latent coefficient vector p ∈ R^K.
  * Trunk: encodes the query coordinate (x, z, t) — with Fourier positional
    features — into a basis vector τ ∈ R^K.
  * Output: u(x,z,t) = <p, τ> + b   (the classic low-rank inner-product form).

Deliberately NO complex-FNO front end, NO travel-time branch, NO MIONet
multi-branch product, NO coarse field, NO dense spectral decoder. It is exactly
the "global low-rank product" whose structural mismatch with locally
translation-invariant wavefronts the capacity ladder identified as our
bottleneck — so this baseline measures how much our extra structure buys.

Everything trains and is scored on the identical data contract
(velocity_mps / source_parameters / source_map / requested_time_s /
dense_target_physical / x_m / z_m) and the identical
ExactWavefieldMetricAccumulator + target (agg < 0.10, every family < 0.12) used
by the capacity ladder, so the numbers are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DeepONetConfig:
    branch_width: int = 256
    branch_blocks: int = 4          # residual CNN blocks over the 201x201 input maps
    latent_dim: int = 256           # K: inner-product rank between branch and trunk
    trunk_width: int = 384
    trunk_depth: int = 6
    fourier_bands: int = 16         # positional Fourier features per (x,z,t) axis
    source_param_dim: int = 5


def _mlp(sizes: list[int], *, activation: type[nn.Module] = nn.GELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class _TrunkMLP(nn.Module):
    """Pre-norm residual MLP for the trunk.

    A plain deep GELU stack makes the trunk output collapse toward zero (each
    layer shrinks the signal; with no norm/residual a 10-12 layer trunk emits
    tau~1e-4, so the branch·trunk inner product is ~0 and training stalls at
    relL2=1.0). Pre-LayerNorm + residual blocks keep the signal scale stable
    across depth, which is what lets the larger-capacity rungs train at all.
    """

    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList()
        for _ in range(depth - 1):
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(width),
                "fc1": nn.Linear(width, width),
                "fc2": nn.Linear(width, width),
            }))
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for blk in self.blocks:
            r = blk["fc2"](nn.functional.gelu(blk["fc1"](blk["norm"](h))))
            h = h + r
        return self.out(self.out_norm(h))


class _ConvBlock(nn.Module):
    """Strided residual conv block that halves spatial resolution."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, stride=1, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DeepONetBaseline(nn.Module):
    """Branch(velocity+source) ⊗ Trunk(x,z,t) → normalized pressure field."""

    def __init__(self, config: DeepONetConfig | None = None) -> None:
        super().__init__()
        self.config = config or DeepONetConfig()
        c = self.config

        # --- Branch: CNN over [velocity, source_map] + scalar source params ---
        channels = [2, 32, 64, 128, c.branch_width]
        conv_layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            conv_layers.append(_ConvBlock(channels[i], channels[i + 1]))
        self.branch_cnn = nn.Sequential(*conv_layers)
        self.branch_pool = nn.AdaptiveAvgPool2d(1)
        self.branch_head = _mlp(
            [c.branch_width + c.source_param_dim, c.branch_width, c.branch_width, c.latent_dim]
        )

        # --- Trunk: Fourier-featured MLP over (x, z, t) ---
        trunk_in = 3 + 3 * 2 * c.fourier_bands
        self.trunk = _TrunkMLP(trunk_in, c.trunk_width, c.trunk_depth, c.latent_dim)
        # log-spaced frequencies for the positional features
        bands = torch.exp(torch.linspace(0.0, 4.0, c.fourier_bands)) * torch.pi
        self.register_buffer("fourier_bands", bands, persistent=True)

        self.bias = nn.Parameter(torch.zeros(1))

    # -- input encoders ---------------------------------------------------
    def encode_branch(
        self,
        velocity_encoded: torch.Tensor,   # (B,1,H,W) normalized velocity
        source_map: torch.Tensor,         # (B,1,H,W)
        source_scalars: torch.Tensor,     # (B,5) normalized source params
    ) -> torch.Tensor:
        maps = torch.cat([velocity_encoded, source_map], dim=1)  # (B,2,H,W)
        feat = self.branch_pool(self.branch_cnn(maps)).flatten(1)  # (B,branch_width)
        return self.branch_head(torch.cat([feat, source_scalars], dim=1))  # (B,K)

    def _fourier(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (...,3) already scaled to ~[-1,1]; append sin/cos features
        proj = coords.unsqueeze(-1) * self.fourier_bands  # (...,3,bands)
        feats = torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(-2)  # (...,3*2*bands)
        return torch.cat([coords, feats], dim=-1)

    def encode_trunk(self, coords: torch.Tensor) -> torch.Tensor:
        return self.trunk(self._fourier(coords))  # (...,K)

    # -- full-field forward on the stored grid ----------------------------
    def forward(
        self,
        velocity_encoded: torch.Tensor,   # (B,1,H,W)
        source_map: torch.Tensor,         # (B,1,H,W)
        source_scalars: torch.Tensor,     # (B,5)
        coords_grid: torch.Tensor,        # (B,F,H,W,3) scaled query coords
        *,
        frame_chunk: int = 4,
        checkpoint_trunk: bool = False,
    ) -> torch.Tensor:
        """Evaluate u = <branch, trunk> over the grid, chunked over time frames.

        The trunk is a pointwise MLP applied at every (x,z,t); materializing all
        F*H*W hidden activations at once blows up memory, so we stream over time
        in ``frame_chunk``-sized blocks. Set ``frame_chunk`` smaller under memory
        pressure. Behaviour is identical to a single-shot forward.

        ``checkpoint_trunk`` (training only) recomputes each block's trunk+einsum
        in the backward pass instead of storing its activations — trades compute
        for memory so the 20M+ full-field model fits in 24 GB.
        """
        p = self.encode_branch(velocity_encoded, source_map, source_scalars)  # (B,K)
        B, F = coords_grid.shape[0], coords_grid.shape[1]

        def _block(block_coords, pk):
            tau = self.encode_trunk(block_coords)                 # (B,f,H,W,K)
            return torch.einsum("bk,bfhwk->bfhw", pk, tau) + self.bias

        outputs = []
        for start in range(0, F, int(frame_chunk)):
            block = coords_grid[:, start:start + int(frame_chunk)]      # (B,f,H,W,3)
            if checkpoint_trunk and self.training:
                out = torch.utils.checkpoint.checkpoint(_block, block, p, use_reentrant=False)
            else:
                out = _block(block, p)
            outputs.append(out)
        return torch.cat(outputs, dim=1)  # (B,F,H,W)

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))


__all__ = ["DeepONetBaseline", "DeepONetConfig"]
