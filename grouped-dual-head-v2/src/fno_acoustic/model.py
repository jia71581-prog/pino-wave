from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_z: int, modes_t: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_z = int(modes_z)
        self.modes_t = int(modes_t)
        scale = 1.0 / max(1, in_channels * out_channels)
        shape = (in_channels, out_channels, self.modes_x, self.modes_z, self.modes_t)
        self.weights1 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def compl_mul3d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixzt,ioxzt->boxzt", x, weights)

    def _validate_modes(self, h: int, w: int, t: int) -> None:
        if self.modes_x > h // 2:
            raise ValueError(f"modes_x={self.modes_x} exceeds H//2={h // 2}")
        if self.modes_z > w // 2:
            raise ValueError(f"modes_z={self.modes_z} exceeds W//2={w // 2}")
        if self.modes_t > t // 2 + 1:
            raise ValueError(f"modes_t={self.modes_t} exceeds T//2+1={t // 2 + 1}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"SpectralConv3d expects [B,C,H,W,T], got {tuple(x.shape)}")
        input_dtype = x.dtype
        if x.dtype in {torch.float16, torch.bfloat16}:
            x = x.float()
        batch, _, h, w, t = x.shape
        self._validate_modes(h, w, t)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            h,
            w,
            t // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        mx, mz, mt = self.modes_x, self.modes_z, self.modes_t
        out_ft[:, :, :mx, :mz, :mt] = self.compl_mul3d(x_ft[:, :, :mx, :mz, :mt], self.weights1)
        out_ft[:, :, -mx:, :mz, :mt] = self.compl_mul3d(x_ft[:, :, -mx:, :mz, :mt], self.weights2)
        out_ft[:, :, :mx, -mz:, :mt] = self.compl_mul3d(x_ft[:, :, :mx, -mz:, :mt], self.weights3)
        out_ft[:, :, -mx:, -mz:, :mt] = self.compl_mul3d(x_ft[:, :, -mx:, -mz:, :mt], self.weights4)
        out = torch.fft.irfftn(out_ft, s=(h, w, t), dim=(-3, -2, -1))
        return out.to(input_dtype) if out.dtype != input_dtype else out


class AcousticFNO3D(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_channels: int = 1,
        modes_x: int = 16,
        modes_z: int = 16,
        modes_t: int = 16,
        width: int = 32,
        n_layers: int = 4,
        padding_ratio: float = 0.0,
        activation: str = "gelu",
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_z = int(modes_z)
        self.modes_t = int(modes_t)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.padding_ratio = float(padding_ratio)
        self.activation_name = activation
        self.normalization_name = normalization

        self.fc0 = nn.Linear(self.in_features, self.width)
        self.spectral = nn.ModuleList(
            [SpectralConv3d(self.width, self.width, self.modes_x, self.modes_z, self.modes_t) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList([nn.Conv3d(self.width, self.width, kernel_size=1) for _ in range(n_layers)])
        if normalization == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm3d(self.width) for _ in range(n_layers)])
        elif normalization in {"none", None}:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
        else:
            raise ValueError(f"unsupported normalization: {normalization}")
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "gelu":
            return F.gelu(x)
        if self.activation_name == "relu":
            return F.relu(x)
        raise ValueError(f"unsupported activation: {self.activation_name}")

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if self.padding_ratio <= 0:
            return x, (0, 0, 0)
        h, w, t = x.shape[-3:]
        pads = (
            max(0, int(round(h * self.padding_ratio))),
            max(0, int(round(w * self.padding_ratio))),
            max(0, int(round(t * self.padding_ratio))),
        )
        return F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0])), pads

    @staticmethod
    def _unpad(x: torch.Tensor, pads: tuple[int, int, int]) -> torch.Tensor:
        ph, pw, pt = pads
        if ph:
            x = x[:, :, :-ph, :, :]
        if pw:
            x = x[:, :, :, :-pw, :]
        if pt:
            x = x[:, :, :, :, :-pt]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"AcousticFNO3D expects [B,H,W,T,C], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"input feature count {x.shape[-1]} != configured {self.in_features}")
        original_hwt = tuple(int(v) for v in x.shape[1:4])
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x, pads = self._pad(x)
        for spectral, pointwise, norm in zip(self.spectral, self.pointwise, self.norms):
            x = spectral(x) + pointwise(x)
            x = norm(x)
            x = self._activation(x)
        x = self._unpad(x, pads)
        if tuple(x.shape[-3:]) != original_hwt:
            raise RuntimeError(f"unpadding changed shape to {tuple(x.shape[-3:])}, expected {original_hwt}")
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self._activation(self.fc1(x))
        x = self.fc2(x)
        if self.out_channels == 1:
            return x[..., 0]
        return x


def build_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(config["model"])
    if model_cfg.get("in_features") == "auto":
        model_cfg["in_features"] = len(config["data"].get("input_features", []))
    return model_cfg
