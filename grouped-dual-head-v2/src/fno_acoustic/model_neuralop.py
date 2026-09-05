"""Official neuraloperator (neuralop 2.0.0) baselines with the AcousticFNO3D contract.

Wraps ``neuralop.models.{FNO,TFNO,UNO}`` so they consume the same
``[B, H, W, T, C]`` batch layout as :class:`fno_acoustic.model.AcousticFNO3D` and
return ``[B, H, W, T]``.  The official models are channels-first N-D operators that
take ``[B, C, *spatial]``; the only glue is a permute in and out.

Using the upstream, community-validated implementations removes any doubt that a
weak baseline is an artifact of our own operator code.  The three architectures are
sized to ~25.8M trainable parameters to match A+1 for a fair equal-capacity table.

Dispatched from train.py via model.name in {neuralop_fno, neuralop_tfno, neuralop_uno}.
"""
from __future__ import annotations

import torch
from torch import nn

from neuralop.models import FNO, TFNO, UNO


class _ChannelsLastWrapper(nn.Module):
    """Adapt a channels-first neuralop model to the [B,H,W,T,C] -> [B,H,W,T] contract."""

    def __init__(self, net: nn.Module, in_features: int, out_channels: int) -> None:
        super().__init__()
        self.net = net
        self.in_features = int(in_features)
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expects [B,H,W,T,C], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"input feature count {x.shape[-1]} != configured {self.in_features}")
        # [B,H,W,T,C] -> [B,C,H,W,T]
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        y = self.net(x)  # [B,out,H,W,T]
        y = y.permute(0, 2, 3, 4, 1).contiguous()  # [B,H,W,T,out]
        if self.out_channels == 1:
            return y[..., 0]
        return y


def build_neuralop_fno(
    *,
    in_features: int,
    out_channels: int = 1,
    modes_x: int = 16,
    modes_z: int = 16,
    modes_t: int = 12,
    hidden_channels: int = 60,
    n_layers: int = 4,
) -> nn.Module:
    net = FNO(
        n_modes=(int(modes_x), int(modes_z), int(modes_t)),
        in_channels=int(in_features),
        out_channels=int(out_channels),
        hidden_channels=int(hidden_channels),
        n_layers=int(n_layers),
    )
    return _ChannelsLastWrapper(net, in_features, out_channels)


def build_neuralop_tfno(
    *,
    in_features: int,
    out_channels: int = 1,
    modes_x: int = 16,
    modes_z: int = 16,
    modes_t: int = 12,
    hidden_channels: int = 96,
    n_layers: int = 4,
    rank: float = 0.42,
) -> nn.Module:
    net = TFNO(
        n_modes=(int(modes_x), int(modes_z), int(modes_t)),
        in_channels=int(in_features),
        out_channels=int(out_channels),
        hidden_channels=int(hidden_channels),
        n_layers=int(n_layers),
        factorization="tucker",
        rank=float(rank),
    )
    return _ChannelsLastWrapper(net, in_features, out_channels)


def build_neuralop_uno(
    *,
    in_features: int,
    out_channels: int = 1,
    hidden_channels: int = 58,
    n_layers: int = 4,
) -> nn.Module:
    hc = int(hidden_channels)
    net = UNO(
        in_channels=int(in_features),
        out_channels=int(out_channels),
        hidden_channels=hc,
        n_layers=int(n_layers),
        uno_out_channels=[hc, hc, hc, hc],
        uno_n_modes=[[16, 16, 12], [12, 12, 8], [12, 12, 8], [16, 16, 12]],
        uno_scalings=[[1, 1, 1], [0.5, 0.5, 1], [2, 2, 1], [1, 1, 1]],
        channel_mlp_skip="linear",
    )
    return _ChannelsLastWrapper(net, in_features, out_channels)
