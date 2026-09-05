#!/usr/bin/env python3
"""R39: identity-initialized HFS fine-tuning on the frozen R38 split.

This experiment follows the patchwise high-frequency scaling idea of
Khodakarami et al. (2025), but starts every HFS gain at zero so loading a
previous R28/R38 checkpoint preserves its predictions exactly.  The opened
development holdout remains development-only; this script never reads the
fresh R29B holdout, validation, or test data.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import train_r29a_late_tail_finetune as r29a


SCRIPT_PATH = Path(__file__).resolve()
PATCH_SIZE = 8
HFS_LR_MULTIPLIER = 100.0
_ORIGINAL_ADAMW = torch.optim.AdamW


class PatchHighFrequencyScaling(nn.Module):
    """Scale common patch content and patch-specific deviations separately."""

    def __init__(self, channels: int, *, patch_size: int = PATCH_SIZE):
        super().__init__()
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        shape = (1, self.channels, 1, 1, 1, 1)
        # Zero is the identity under X + lambda_dc * DC + lambda_hfc * HFC.
        self.lambda_dc = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.lambda_hfc = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.lambda_dc._r39_hfs_parameter = True
        self.lambda_hfc._r39_hfs_parameter = True

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
        # PyTorch 2.1 CUDA does not implement replication_pad2d for BF16.
        # Do the inexpensive HFS bookkeeping in FP32 and cast back so AMP is
        # supported without changing the identity initialization.
        padded = F.pad(
            value.float(), (0, pad_w, 0, pad_h), mode="replicate"
        )
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
            + self.lambda_dc.to(dtype=patches.dtype) * direct
            + self.lambda_hfc.to(dtype=patches.dtype) * high
        )
        restored = scaled.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, channels, padded_h, padded_w
        )
        return restored[:, :, :height, :width].to(dtype=original_dtype)


class HFSTailSpectralResidualUNet(r29a.r26.TailSpectralResidualUNet):
    """R28 network augmented with identity-initialized patch HFS modules."""

    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.25):
        super().__init__(base_width=base_width, correction_cap=correction_cap)
        width = int(base_width)
        self.hfs_x0 = PatchHighFrequencyScaling(width)
        self.hfs_x1 = PatchHighFrequencyScaling(width * 2)
        self.hfs_x2 = PatchHighFrequencyScaling(width * 3)
        self.hfs_x3 = PatchHighFrequencyScaling(width * 4)
        self.hfs_y2 = PatchHighFrequencyScaling(width * 3)
        self.hfs_y1 = PatchHighFrequencyScaling(width * 2)
        self.hfs_y0 = PatchHighFrequencyScaling(width)
        self.hfs_s0 = PatchHighFrequencyScaling(16)
        self.hfs_s1 = PatchHighFrequencyScaling(16)
        self.hfs_s2 = PatchHighFrequencyScaling(16)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        bad_missing = [key for key in result.missing_keys if not key.startswith("hfs_")]
        if strict and (bad_missing or result.unexpected_keys):
            raise RuntimeError(
                "incompatible R39 checkpoint: "
                f"missing={bad_missing}, unexpected={result.unexpected_keys}"
            )
        return result

    def forward(
        self, features: torch.Tensor, *, active: torch.Tensor | None = None
    ) -> torch.Tensor:
        x0 = self.hfs_x0(self.enc0(self.stem(features)))
        x1 = self.hfs_x1(self.enc1(self.down1(x0)))
        x2 = self.hfs_x2(self.enc2(self.down2(x1)))
        x3 = self.hfs_x3(self.bottleneck(self.down3(x2)))
        y2 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.hfs_y2(self.dec2(self.up2(torch.cat([y2, x2], dim=1))))
        y1 = F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.hfs_y1(self.dec1(self.up1(torch.cat([y1, x1], dim=1))))
        y0 = F.interpolate(y1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y0 = self.hfs_y0(self.dec0(self.up0(torch.cat([y0, x0], dim=1))))
        local = self.correction_cap * torch.tanh(self.output(y0)[:, 0])

        spectral = self.hfs_s0(self.spectral_stem(features))
        spectral = self.hfs_s1(self.spectral_blocks[0](spectral))
        spectral = self.hfs_s2(self.spectral_blocks[1](spectral))
        spectral = self.spectral_cap * torch.tanh(self.spectral_output(spectral)[:, 0])
        if active is not None:
            local = local * active[:, None, None]
            spectral = spectral * active[:, None, None]
        correction = torch.clamp(
            local + spectral, -self.correction_cap, self.correction_cap
        ).clone()
        correction[:, 0, :] = 0.0
        return correction


def _argument_value(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _r39_adamw(parameters, *, lr: float, weight_decay: float, **kwargs):
    parameters = list(parameters)
    hfs = [
        parameter
        for parameter in parameters
        if bool(getattr(parameter, "_r39_hfs_parameter", False))
    ]
    hfs_ids = {id(parameter) for parameter in hfs}
    base = [parameter for parameter in parameters if id(parameter) not in hfs_ids]
    if not hfs or not base:
        raise RuntimeError(
            f"R39 optimizer partition failed: base={len(base)}, hfs={len(hfs)}"
        )
    return _ORIGINAL_ADAMW(
        [
            {"params": base, "lr": float(lr), "weight_decay": float(weight_decay)},
            {
                "params": hfs,
                "lr": float(lr) * HFS_LR_MULTIPLIER,
                "weight_decay": 0.0,
            },
        ],
        lr=float(lr),
        weight_decay=float(weight_decay),
        **kwargs,
    )


def _write_method_record() -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output = _argument_value("--output-dir")
    if output is None:
        return
    directory = Path(output).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "r39_hfs_method_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "R28_tail_spectral_residual_unet_plus_patch_HFS",
        "patch_size": PATCH_SIZE,
        "optimizer": {
            "base_learning_rate": _argument_value("--learning-rate"),
            "hfs_learning_rate_multiplier": HFS_LR_MULTIPLIER,
            "hfs_weight_decay": 0.0,
        },
        "initialization": {
            "lambda_dc": 0.0,
            "lambda_hfc": 0.0,
            "property": "exact identity before fine-tuning",
        },
        "evidence_boundary": (
            "R28 opened development holdout only; R29B fresh holdout, validation, "
            "and test remain unopened"
        ),
        "script": str(SCRIPT_PATH),
    }
    temporary = directory / ".r39_method.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(directory / "r39_method.json")


def main() -> int:
    # R29A constructs this symbol dynamically; replacing it keeps its audited
    # data loader, loss, DDP, and evaluation path unchanged.
    r29a.r26.TailSpectralResidualUNet = HFSTailSpectralResidualUNet
    r29a.SCRIPT_PATH = SCRIPT_PATH
    r29a.torch.optim.AdamW = _r39_adamw
    _write_method_record()
    return r29a.main()


if __name__ == "__main__":
    raise SystemExit(main())
