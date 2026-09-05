#!/usr/bin/env python3
"""R52 arm B: spatially varying patch HFS gains (structure change).

Replaces the per-channel scalar lambda_dc/lambda_hfc of R39 with per-channel
8x8 spatial gain maps that are bilinearly interpolated to the patch grid.
Loading an R39 checkpoint broadcasts its scalar gains into the maps, so the
loaded network is function-identical to the R39 model (constant maps
interpolate to the same constant).  Data, loss, DDP, and evaluation paths are
unchanged; sealed holdouts remain unopened.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import train_r39_hfs_tail_finetune as r39

SCRIPT_PATH = Path(__file__).resolve()
SPATIAL_GRID = 8


class SpatialPatchHighFrequencyScaling(r39.PatchHighFrequencyScaling):
    """Patch HFS with per-channel spatial gain maps instead of scalars."""

    def __init__(self, channels: int, *, patch_size: int = r39.PATCH_SIZE):
        super().__init__(channels, patch_size=patch_size)
        shape = (1, self.channels, SPATIAL_GRID, SPATIAL_GRID, 1, 1)
        self.lambda_dc = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.lambda_hfc = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.lambda_dc._r39_hfs_parameter = True
        self.lambda_hfc._r39_hfs_parameter = True

    def _gain(self, parameter: torch.Tensor, grid_h: int, grid_w: int, dtype):
        gain = parameter[:, :, :, :, 0, 0].to(dtype=dtype)
        gain = F.interpolate(
            gain, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        return gain[..., None, None]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_dtype = value.dtype
        batch, channels, height, width = value.shape
        if channels != self.channels:
            raise RuntimeError(
                f"HFS channel mismatch: expected {self.channels}, got {channels}"
            )
        patch = self.patch_size
        pad_h = (-height) % patch
        pad_w = (-width) % patch
        padded = F.pad(value.float(), (0, pad_w, 0, pad_h), mode="replicate")
        padded_h, padded_w = padded.shape[-2:]
        grid_h = padded_h // patch
        grid_w = padded_w // patch
        patches = padded.reshape(
            batch, channels, grid_h, patch, grid_w, patch
        ).permute(0, 1, 2, 4, 3, 5)
        direct = patches.mean(dim=(2, 3), keepdim=True)
        high = patches - direct
        scaled = (
            patches
            + self._gain(self.lambda_dc, grid_h, grid_w, patches.dtype) * direct
            + self._gain(self.lambda_hfc, grid_h, grid_w, patches.dtype) * high
        )
        restored = scaled.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, channels, padded_h, padded_w
        )
        return restored[:, :, :height, :width].to(dtype=original_dtype)


class SpatialHFSTailSpectralResidualUNet(r39.HFSTailSpectralResidualUNet):
    """R39 network with spatial HFS modules and scalar-broadcast loading."""

    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.25):
        super().__init__(base_width=base_width, correction_cap=correction_cap)
        for name, module in list(self.named_children()):
            if name.startswith("hfs_"):
                setattr(
                    self,
                    name,
                    SpatialPatchHighFrequencyScaling(
                        module.channels, patch_size=module.patch_size
                    ),
                )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = dict(state_dict)
        for name, parameter in self.named_parameters():
            if not name.startswith("hfs_") or name not in state_dict:
                continue
            value = state_dict[name]
            if value.shape != parameter.shape and value.numel() == parameter.shape[1]:
                state_dict[name] = (
                    value.reshape(1, -1, 1, 1, 1, 1).expand(parameter.shape).contiguous()
                )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def main() -> int:
    r39.r29a.r26.TailSpectralResidualUNet = SpatialHFSTailSpectralResidualUNet
    r39.SCRIPT_PATH = SCRIPT_PATH
    r39.r29a.SCRIPT_PATH = SCRIPT_PATH
    r39.r29a.torch.optim.AdamW = r39._r39_adamw
    r39._write_method_record()
    return r39.r29a.main()


if __name__ == "__main__":
    raise SystemExit(main())
