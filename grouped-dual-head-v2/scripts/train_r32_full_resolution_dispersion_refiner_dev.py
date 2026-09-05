#!/usr/bin/env python3
"""R32 development: frozen R28 plus a full-resolution dispersion refiner.

The R28 local U-Net and spectral branch remain frozen.  A zero-start branch at
the original 201x201 resolution receives the 16 dispersion-conditioned
features plus the frozen R28 correction.  Dilated residual blocks enlarge the
receptive field without pooling, preserving the high-wavenumber residuals that
dominate the high-frequency Marmousi tail.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
R29E_PATH = SCRIPT_PATH.with_name("train_r29e_dispersion_conditioned_dev.py")
SPEC = importlib.util.spec_from_file_location("r32_r29e_components", R29E_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R29E components: {R29E_PATH}")
r29e = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r29e
SPEC.loader.exec_module(r29e)
r25 = r29e.r25


BASE_MODEL = r29e.BASE_MODEL
REFINER_WIDTH = 24
REFINER_CAP = 0.10


class FullResolutionDispersionRefiner(nn.Module):
    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.25):
        super().__init__()
        if int(r25.INPUT_CHANNELS) != int(r29e.BASE_INPUT_CHANNELS):
            raise RuntimeError("R32 frozen base must be constructed with 13 input channels")
        self.base_width = int(base_width)
        self.correction_cap = float(correction_cap)
        self.base = BASE_MODEL(
            base_width=self.base_width, correction_cap=self.correction_cap
        )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        input_channels = int(r29e.DISPERSION_INPUT_CHANNELS) + 1
        self.refiner_stem = r25.ConvNormAct(input_channels, REFINER_WIDTH)
        self.refiner_blocks = nn.Sequential(
            r25.ResidualBlock(REFINER_WIDTH, dilation=1),
            r25.ResidualBlock(REFINER_WIDTH, dilation=2),
            r25.ResidualBlock(REFINER_WIDTH, dilation=4),
            r25.ResidualBlock(REFINER_WIDTH, dilation=8),
            r25.ResidualBlock(REFINER_WIDTH, dilation=4),
            r25.ResidualBlock(REFINER_WIDTH, dilation=2),
            r25.ResidualBlock(REFINER_WIDTH, dilation=1),
        )
        self.refiner_output = nn.Conv2d(REFINER_WIDTH, 1, kernel_size=1)
        nn.init.zeros_(self.refiner_output.weight)
        nn.init.zeros_(self.refiner_output.bias)

    def forward(
        self, features: torch.Tensor, *, active: torch.Tensor | None = None
    ) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != r29e.DISPERSION_INPUT_CHANNELS:
            raise ValueError(f"unexpected R32 feature shape: {tuple(features.shape)}")
        with torch.no_grad():
            base_correction = self.base(
                features[:, : r29e.BASE_INPUT_CHANNELS], active=active
            )
        refiner_input = torch.cat([features, base_correction[:, None]], dim=1)
        hidden = self.refiner_blocks(self.refiner_stem(refiner_input))
        update = REFINER_CAP * torch.tanh(self.refiner_output(hidden)[:, 0])
        if active is not None:
            update = update * active[:, None, None]
        correction = torch.clamp(
            base_correction + update, -self.correction_cap, self.correction_cap
        )
        correction = correction.clone()
        correction[:, 0, :] = 0.0
        return correction

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        if any(key.startswith("base.") for key in state_dict):
            return super().load_state_dict(state_dict, strict=strict)
        full = self.state_dict()
        base_keys = self.base.state_dict().keys()
        missing = [key for key in base_keys if key not in state_dict]
        if missing:
            raise RuntimeError(f"R28 checkpoint is missing base keys: {missing[:5]}")
        for key in base_keys:
            full[f"base.{key}"] = state_dict[key]
        return super().load_state_dict(full, strict=True)


def argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"missing required argument {name}") from error


def write_sidecar(stage: str) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output_dir = Path(argument_value("--output-dir")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": f"r32_full_resolution_dispersion_refiner_{stage}_v1",
        "role": "R28_already_opened_train_holdout_development_only",
        "frozen_parent": "R28_local_UNet_plus_global_spectral_branch",
        "refiner": {
            "resolution": [201, 201],
            "input_channels": int(r29e.DISPERSION_INPUT_CHANNELS) + 1,
            "inputs": "16_dispersion_conditioned_features_plus_frozen_R28_correction",
            "width": REFINER_WIDTH,
            "dilations": [1, 2, 4, 8, 4, 2, 1],
            "pooling_or_resampling": False,
            "correction_cap": REFINER_CAP,
            "zero_start": True,
        },
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": r25.sha256_file(SCRIPT_PATH),
        "r29e_components_sha256": r25.sha256_file(R29E_PATH),
    }
    if stage == "terminal":
        base = json.loads((output_dir / "terminal.json").read_text(encoding="utf-8"))
        payload.update(
            {
                "status": base["status"],
                "best_epoch": base["best_epoch"],
                "best_metrics": base["best_metrics"],
                "checkpoint": base["checkpoint"],
                "checkpoint_sha256": base["checkpoint_sha256"],
                "absolute_goal_passed": base["absolute_goal_passed"],
            }
        )
    else:
        payload.update(
            {
                "status": "frozen_before_development_training",
                "success_gate": "mean_and_max_record_relative_L2_lte_0p05",
                "evidence_boundary": "A pass only authorizes a fresh group-disjoint train holdout experiment.",
            }
        )
    r25.atomic_json(payload, output_dir / f"r32_{stage}.json")


def main() -> int:
    if int(r25.INPUT_CHANNELS) != int(r29e.BASE_INPUT_CHANNELS):
        r25.INPUT_CHANNELS = int(r29e.BASE_INPUT_CHANNELS)
    r29e.r29a.LateTailFitFrameDataset = r29e.DispersionConditionedDataset
    r29e.r29a.r26.TailSpectralResidualUNet = FullResolutionDispersionRefiner
    r29e.r29a.r25.evaluate = r29e.evaluate_dispersion
    write_sidecar("preregistration")
    result = r29e.r29a.main()
    write_sidecar("terminal")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
