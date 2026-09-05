#!/usr/bin/env python3
"""R29E development: explicitly condition the residual operator on dispersion.

Three deployment-available channels are added: normalized source frequency,
local nondimensional wavenumber kh=2*pi*f0*dx/c(x), and accumulated travel
cycles f0*T(x).  R28 is embedded exactly with zero new-channel weights.  A
gradient multiplier gives the new channels ten times the effective learning
rate of the existing backbone.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
R29A_PATH = SCRIPT_PATH.with_name("train_r29a_late_tail_finetune.py")
SPEC = importlib.util.spec_from_file_location("r29e_r29a_driver", R29A_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R29A driver: {R29A_PATH}")
r29a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r29a
SPEC.loader.exec_module(r29a)
r25 = r29a.r25


BASE_DATASET = r29a.LateTailFitFrameDataset
BASE_MODEL = r29a.r26.TailSpectralResidualUNet
BASE_INPUT_CHANNELS = int(r25.INPUT_CHANNELS)
DISPERSION_INPUT_CHANNELS = BASE_INPUT_CHANNELS + 3
GRID_SPACING_M = 10.0
BACKBONE_GRADIENT_MULTIPLIER = 0.1


def add_dispersion_features(
    features: torch.Tensor,
    source_f0_hz: torch.Tensor,
) -> torch.Tensor:
    """Append f0, local kh, and accumulated cycles to [B,13,H,W]."""

    squeeze = features.ndim == 3
    if squeeze:
        features = features[None]
        source_f0_hz = source_f0_hz.reshape(1)
    if features.ndim != 4 or features.shape[1] != BASE_INPUT_CHANNELS:
        raise ValueError(f"unexpected base feature shape: {tuple(features.shape)}")
    batch, _, height, width = features.shape
    f0 = source_f0_hz.to(device=features.device, dtype=features.dtype).reshape(batch)
    velocity = (features[:, 1] * 2500.0 + 4500.0).clamp(500.0, 8000.0)
    travel_s = features[:, 5].clamp(0.0, 1.5)
    f0_map = ((f0 - 20.0) / 10.0).clamp(-1.5, 1.5)[:, None, None]
    f0_map = f0_map.expand(batch, height, width)
    kh = (2.0 * math.pi * f0[:, None, None] * GRID_SPACING_M / velocity).clamp(0.0, 2.5)
    travel_cycles = (f0[:, None, None] * travel_s / 20.0).clamp(0.0, 2.5)
    result = torch.cat(
        [features, f0_map[:, None], kh[:, None], travel_cycles[:, None]], dim=1
    )
    return result[0] if squeeze else result


class DispersionConditionedDataset(BASE_DATASET):
    def __getitem__(self, index: int):
        base_index = int(self.mapping[int(index)])
        record_position, _ = divmod(base_index, self.time_count)
        file_index, local_index = self.collection.records[record_position]
        handle = self.collection.handles[file_index]
        features, coarse, truth, mean_energy, active = r25.FitFrameDataset.__getitem__(
            self, base_index
        )
        f0 = torch.tensor(float(handle["source_f0_hz"][local_index]), dtype=torch.float32)
        return (
            add_dispersion_features(features, f0),
            coarse,
            truth,
            mean_energy,
            active,
        )


class DispersionConditionedResidualUNet(BASE_MODEL):
    """R28 with three zero-start inputs and differential gradient scaling."""

    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.25):
        super().__init__(base_width=base_width, correction_cap=correction_cap)
        base_channels = int(BASE_INPUT_CHANNELS)
        multiplier = float(BACKBONE_GRADIENT_MULTIPLIER)

        def scale_stem(gradient: torch.Tensor) -> torch.Tensor:
            scaled = gradient.clone()
            scaled[:, :base_channels] *= multiplier
            return scaled

        for name, parameter in self.named_parameters():
            if name in {"stem.conv.weight", "spectral_stem.weight"}:
                parameter.register_hook(scale_stem)
            else:
                parameter.register_hook(lambda gradient, m=multiplier: gradient * m)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        target = self.state_dict()
        adapted: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key not in target or tuple(value.shape) == tuple(target[key].shape):
                adapted[key] = value
                continue
            expected = target[key]
            if (
                key in {"stem.conv.weight", "spectral_stem.weight"}
                and value.ndim == 4
                and expected.ndim == 4
                and value.shape[0] == expected.shape[0]
                and value.shape[1] == BASE_INPUT_CHANNELS
                and expected.shape[1] == DISPERSION_INPUT_CHANNELS
                and tuple(value.shape[2:]) == tuple(expected.shape[2:])
            ):
                padded = torch.zeros_like(expected)
                padded[:, :BASE_INPUT_CHANNELS].copy_(value)
                adapted[key] = padded
                continue
            raise RuntimeError(
                f"unsupported R29E checkpoint shape for {key}: "
                f"{tuple(value.shape)} -> {tuple(expected.shape)}"
            )
        return super().load_state_dict(adapted, strict=strict)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "candidate_mean": None,
            "candidate_max": None,
            "candidate_median": None,
            "parent_mean": None,
            "parent_max": None,
            "mean_relative_improvement": None,
            "max_relative_improvement": None,
        }
    candidate = np.asarray([row["candidate_rel_l2"] for row in rows], dtype=np.float64)
    parent = np.asarray([row["parent_rel_l2"] for row in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "candidate_mean": float(candidate.mean()),
        "candidate_max": float(candidate.max()),
        "candidate_median": float(np.median(candidate)),
        "parent_mean": float(parent.mean()),
        "parent_max": float(parent.max()),
        "mean_relative_improvement": float(1.0 - candidate.mean() / parent.mean()),
        "max_relative_improvement": float(1.0 - candidate.max() / parent.max()),
    }


@torch.inference_mode()
def evaluate_dispersion(
    model: torch.nn.Module,
    collection: Any,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    family_rows: dict[str, list[dict[str, Any]]] = {
        family: [] for family in r25.FAMILIES
    }
    for file_index, local_index in collection.records:
        handle = collection.handles[file_index]
        sample_id = str(handle["sample_id"].asstr()[local_index])
        family = str(handle["family"].asstr()[local_index])
        static = torch.from_numpy(
            np.asarray(handle["static_features"][local_index], dtype=np.float32)
        ).to(device)
        f0 = float(handle["source_f0_hz"][local_index])
        t0 = float(handle["source_t0_s"][local_index])
        times = np.asarray(handle["time_s"][:], dtype=np.float32)
        coarse_all = handle["coarse_norm"][local_index]
        truth_all = handle["truth_norm"][local_index]
        candidate_error = 0.0
        parent_error = 0.0
        target_square = 0.0
        correction_square = 0.0
        for start in range(0, len(times), int(batch_size)):
            stop = min(start + int(batch_size), len(times))
            coarse = torch.from_numpy(
                np.asarray(coarse_all[start:stop], dtype=np.float32)
            ).to(device)
            truth = torch.from_numpy(
                np.asarray(truth_all[start:stop], dtype=np.float32)
            ).to(device)
            block = stop - start
            time_tensor = torch.from_numpy(times[start:stop]).to(device)
            f0_tensor = torch.full((block,), f0, device=device)
            features = r25.make_dynamic_features(
                coarse,
                static[None].expand(block, -1, -1, -1),
                time_s=time_tensor,
                source_f0_hz=f0_tensor,
                source_t0_s=torch.full((block,), t0, device=device),
            )
            features = add_dispersion_features(features, f0_tensor)
            active = (time_tensor >= t0).float()
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp
                else nullcontext()
            )
            with context:
                correction = model(features, active=active).float()
            prediction = coarse + correction
            candidate_error += float((prediction.double() - truth.double()).square().sum())
            parent_error += float((coarse.double() - truth.double()).square().sum())
            target_square += float(truth.double().square().sum())
            correction_square += float(correction.double().square().sum())
        candidate_rel = math.sqrt(candidate_error / max(target_square, 1.0e-30))
        parent_rel = math.sqrt(parent_error / max(target_square, 1.0e-30))
        row = {
            "sample_id": sample_id,
            "family": family,
            "candidate_rel_l2": candidate_rel,
            "parent_rel_l2": parent_rel,
            "relative_improvement": 1.0 - candidate_rel / max(parent_rel, 1.0e-30),
            "candidate_error_square": candidate_error,
            "parent_error_square": parent_error,
            "target_square": target_square,
            "correction_square": correction_square,
        }
        rows.append(row)
        family_rows[family].append(row)
    aggregate = summarize(rows)
    return {
        "aggregate": aggregate,
        "per_family": {
            family: summarize(values) for family, values in family_rows.items()
        },
        "absolute_goal": {
            "mean_lte_0p05": bool(aggregate["candidate_mean"] <= 0.05),
            "max_lte_0p05": bool(aggregate["candidate_max"] <= 0.05),
            "passed": bool(
                aggregate["candidate_mean"] <= 0.05
                and aggregate["candidate_max"] <= 0.05
            ),
        },
        "records": rows,
    }


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
        "schema": f"r29e_dispersion_conditioned_{stage}_v1",
        "role": "R28_already_opened_train_holdout_development_only",
        "features": [
            "normalized_source_frequency_(f0-20)/10",
            "local_nondimensional_wavenumber_2pi_f0_dx_over_c",
            "normalized_accumulated_travel_cycles_f0T_over_20",
        ],
        "grid_spacing_m": GRID_SPACING_M,
        "input_channels": DISPERSION_INPUT_CHANNELS,
        "backbone_gradient_multiplier": BACKBONE_GRADIENT_MULTIPLIER,
        "new_channel_gradient_multiplier": 1.0,
        "required_optimizer_weight_decay": 0.0,
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": r25.sha256_file(SCRIPT_PATH),
        "r29a_driver_sha256": r25.sha256_file(R29A_PATH),
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
    r25.atomic_json(payload, output_dir / f"r29e_{stage}.json")


def main() -> int:
    r29a.LateTailFitFrameDataset = DispersionConditionedDataset
    r29a.r26.TailSpectralResidualUNet = DispersionConditionedResidualUNet
    r29a.r25.evaluate = evaluate_dispersion
    r29a.r25.INPUT_CHANNELS = int(DISPERSION_INPUT_CHANNELS)
    write_sidecar("preregistration")
    result = r29a.main()
    write_sidecar("terminal")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
