"""U-NO (U-shaped neural operator) adapter for the acoustic wavefield baseline.

Wraps ``UFNO/ufno3d.py::UFNO3d`` so it consumes the same ``[B, H, W, T, C]``
batch layout as :class:`fno_acoustic.model.AcousticFNO3D` and returns
``[B, H, W, T]``.  Two glue steps are needed because the raw ``UFNO3d`` is
channels-first and has a single 2x down/up level:

* permute ``[B,H,W,T,C] -> [B,C,H,W,T]`` for the Conv3d lift, and back on output;
* symmetric-pad H, W, T up to an even size so the stride-2 down block and the
  stride-2 transpose-up block round-trip to the input resolution, then crop.

This is a thin, faithful wrapper: the operator math (spectral + local + skip
U-blocks) is unchanged, so the baseline remains a genuine U-NO.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_UFNO_DIR = Path(__file__).resolve().parents[2] / "UFNO"
if str(_UFNO_DIR) not in sys.path:
    sys.path.insert(0, str(_UFNO_DIR))

from ufno3d import UFNO3d  # noqa: E402  (path injected above)


def _pad_to_even(size: int) -> int:
    return size + (size % 2)


class AcousticUNO3D(nn.Module):
    """U-NO baseline with the AcousticFNO3D input/output contract."""

    def __init__(
        self,
        in_features: int,
        out_channels: int = 1,
        modes_x: int = 12,
        modes_z: int = 12,
        modes_t: int = 12,
        width: int = 24,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_channels = int(out_channels)
        self.net = UFNO3d(
            in_channels=self.in_features,
            out_channels=self.out_channels,
            width=int(width),
            modes_x=int(modes_x),
            modes_y=int(modes_z),
            modes_z=int(modes_t),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"AcousticUNO3D expects [B,H,W,T,C], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"input feature count {x.shape[-1]} != configured {self.in_features}")
        b, h, w, t, _ = x.shape
        # [B,H,W,T,C] -> [B,C,H,W,T]
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        ph, pw, pt = _pad_to_even(h) - h, _pad_to_even(w) - w, _pad_to_even(t) - t
        if ph or pw or pt:
            # F.pad order is (T_left,T_right, W_left,W_right, H_left,H_right)
            x = F.pad(x, (0, pt, 0, pw, 0, ph))
        y = self.net(x)  # [B,out,H',W',T']
        if ph or pw or pt:
            y = y[:, :, :h, :w, :t]
        # [B,out,H,W,T] -> [B,H,W,T,out]
        y = y.permute(0, 2, 3, 4, 1).contiguous()
        if self.out_channels == 1:
            return y[..., 0]
        return y
