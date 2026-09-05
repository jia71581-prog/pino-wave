#!/usr/bin/env python3
"""R29F development: gated high-frequency dispersion adapters on frozen R28."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPT_PATH = Path(__file__).resolve()
R29E_PATH = SCRIPT_PATH.with_name("train_r29e_dispersion_conditioned_dev.py")
SPEC = importlib.util.spec_from_file_location("r29f_r29e_components", R29E_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R29E components: {R29E_PATH}")
r29e = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r29e
SPEC.loader.exec_module(r29e)


GATE_START_HZ = 22.0
GATE_FULL_HZ = 27.0
BASE_ADD_DISPERSION_FEATURES = r29e.add_dispersion_features


def gated_dispersion_features(
    features: torch.Tensor, source_f0_hz: torch.Tensor
) -> torch.Tensor:
    result = BASE_ADD_DISPERSION_FEATURES(features, source_f0_hz)
    squeeze = result.ndim == 3
    if squeeze:
        result = result[None]
        f0 = source_f0_hz.reshape(1)
    else:
        f0 = source_f0_hz.reshape(result.shape[0])
    gate = ((f0.to(result.device, result.dtype) - GATE_START_HZ) / (
        GATE_FULL_HZ - GATE_START_HZ
    )).clamp(0.0, 1.0)
    result = result.clone()
    result[:, r29e.BASE_INPUT_CHANNELS :] *= gate[:, None, None, None]
    return result[0] if squeeze else result


class GatedFrozenDispersionAdapter(r29e.DispersionConditionedResidualUNet):
    """Freeze R28 and train only the three zero-start dispersion input slices."""

    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.25):
        super().__init__(base_width=base_width, correction_cap=correction_cap)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.stem.conv.weight.requires_grad_(True)
        self.spectral_stem.weight.requires_grad_(True)
        base_channels = int(r29e.BASE_INPUT_CHANNELS)

        def mask_backbone_channels(gradient: torch.Tensor) -> torch.Tensor:
            masked = gradient.clone()
            masked[:, :base_channels] = 0.0
            return masked

        self.stem.conv.weight.register_hook(mask_backbone_channels)
        self.spectral_stem.weight.register_hook(mask_backbone_channels)


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
        "schema": f"r29f_gated_frozen_dispersion_adapter_{stage}_v1",
        "role": "R28_already_opened_train_holdout_development_only",
        "frequency_gate": {
            "zero_at_or_below_hz": GATE_START_HZ,
            "linear_between_hz": [GATE_START_HZ, GATE_FULL_HZ],
            "one_at_or_above_hz": GATE_FULL_HZ,
        },
        "trainable_parameters": [
            "stem.conv.weight[:,13:16,:,:]",
            "spectral_stem.weight[:,13:16,:,:]",
        ],
        "frozen_parameters": "all_R28_backbone_parameters_and_original_13_input_channel_slices",
        "required_optimizer_weight_decay": 0.0,
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": r29e.r25.sha256_file(SCRIPT_PATH),
        "r29e_components_sha256": r29e.r25.sha256_file(R29E_PATH),
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
    r29e.r25.atomic_json(payload, output_dir / f"r29f_{stage}.json")


def main() -> int:
    r29e.add_dispersion_features = gated_dispersion_features
    r29e.r29a.LateTailFitFrameDataset = r29e.DispersionConditionedDataset
    r29e.r29a.r26.TailSpectralResidualUNet = GatedFrozenDispersionAdapter
    r29e.r29a.r25.evaluate = r29e.evaluate_dispersion
    r29e.r29a.r25.INPUT_CHANNELS = int(r29e.DISPERSION_INPUT_CHANNELS)
    write_sidecar("preregistration")
    result = r29e.r29a.main()
    write_sidecar("terminal")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
