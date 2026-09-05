#!/usr/bin/env python3
"""R29D development: train only zero-start temporal-context input adapters."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPT_PATH = Path(__file__).resolve()
R29C_PATH = SCRIPT_PATH.with_name("train_r29c_temporal_context_dev.py")
SPEC = importlib.util.spec_from_file_location("r29d_r29c_components", R29C_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R29C components: {R29C_PATH}")
r29c = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r29c
SPEC.loader.exec_module(r29c)


class FrozenContextAdapterModel(r29c.ContextTailSpectralResidualUNet):
    """Keep R28 fixed and train only the two new input-channel slices."""

    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.25):
        super().__init__(base_width=base_width, correction_cap=correction_cap)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.stem.conv.weight.requires_grad_(True)
        self.spectral_stem.weight.requires_grad_(True)

        base_channels = int(r29c.BASE_INPUT_CHANNELS)

        def mask_old_channels(gradient: torch.Tensor) -> torch.Tensor:
            masked = gradient.clone()
            masked[:, :base_channels] = 0.0
            return masked

        self.stem.conv.weight.register_hook(mask_old_channels)
        self.spectral_stem.weight.register_hook(mask_old_channels)


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
        "schema": f"r29d_frozen_context_adapter_{stage}_v1",
        "role": "R28_already_opened_train_holdout_development_only",
        "context_lag_s": float(r29c.CONTEXT_LAG_S),
        "trainable_parameters": [
            "stem.conv.weight[:,13:15,:,:]",
            "spectral_stem.weight[:,13:15,:,:]",
        ],
        "frozen_parameters": "all_R28_backbone_parameters_and_original_13_input_channel_slices",
        "required_optimizer_weight_decay": 0.0,
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": r29c.r25.sha256_file(SCRIPT_PATH),
        "r29c_components_sha256": r29c.r25.sha256_file(R29C_PATH),
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
    r29c.r25.atomic_json(payload, output_dir / f"r29d_{stage}.json")


def main() -> int:
    r29c.r29a.LateTailFitFrameDataset = r29c.ContextLateTailDataset
    r29c.r29a.r26.TailSpectralResidualUNet = FrozenContextAdapterModel
    r29c.r29a.r25.evaluate = r29c.evaluate_context
    r29c.r29a.r25.INPUT_CHANNELS = int(r29c.CONTEXT_INPUT_CHANNELS)
    write_sidecar("preregistration")
    result = r29c.r29a.main()
    write_sidecar("terminal")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
