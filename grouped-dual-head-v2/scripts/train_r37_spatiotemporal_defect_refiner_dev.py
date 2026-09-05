#!/usr/bin/env python3
"""R37 development: frozen R28 plus a spatiotemporal-defect refiner.

The zero-start full-resolution branch receives deployment-only temporal
neighbours, explicit dispersion coordinates, temporal curvature/slope, and a
spatial high-pass defect.  The R28 base is frozen and reproduced exactly at
initialization.  Only the already-opened R28 train holdout is used.
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
from torch import nn
from torch.nn import functional as F


SCRIPT_PATH = Path(__file__).resolve()
R29C_PATH = SCRIPT_PATH.with_name("train_r29c_temporal_context_dev.py")
SPEC = importlib.util.spec_from_file_location("r37_r29c_components", R29C_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R29C components: {R29C_PATH}")
r29c = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r29c
SPEC.loader.exec_module(r29c)
r29a = r29c.r29a
r25 = r29c.r25


BASE_INPUT_CHANNELS = int(r29c.BASE_INPUT_CHANNELS)
CONTEXT_INPUT_CHANNELS = int(r29c.CONTEXT_INPUT_CHANNELS)
DISPERSION_INPUT_CHANNELS = CONTEXT_INPUT_CHANNELS + 3
REFINER_INPUT_CHANNELS = DISPERSION_INPUT_CHANNELS + 4
REFINER_WIDTH = 24
REFINER_CAP = 0.05
GRID_SPACING_M = 10.0
BASE_MODEL = r29c.BASE_MODEL


def add_dispersion_features(
    features: torch.Tensor, source_f0_hz: torch.Tensor
) -> torch.Tensor:
    squeeze = features.ndim == 3
    if squeeze:
        features = features[None]
        source_f0_hz = source_f0_hz.reshape(1)
    if features.ndim != 4 or features.shape[1] != CONTEXT_INPUT_CHANNELS:
        raise ValueError(f"unexpected R37 feature shape: {tuple(features.shape)}")
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


class SpatiotemporalDefectDataset(r29c.ContextLateTailDataset):
    def __getitem__(self, index: int):
        base_index = int(self.mapping[int(index)])
        record_position, _ = divmod(base_index, self.time_count)
        file_index, local_index = self.collection.records[record_position]
        handle = self.collection.handles[file_index]
        features, coarse, truth, mean_energy, active = super().__getitem__(index)
        f0 = torch.tensor(float(handle["source_f0_hz"][local_index]), dtype=torch.float32)
        return (
            add_dispersion_features(features, f0),
            coarse,
            truth,
            mean_energy,
            active,
        )


class SpatiotemporalDefectRefiner(nn.Module):
    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.25):
        super().__init__()
        if int(r25.INPUT_CHANNELS) != BASE_INPUT_CHANNELS:
            raise RuntimeError("R37 frozen base must be constructed with 13 channels")
        self.base_width = int(base_width)
        self.correction_cap = float(correction_cap)
        self.base = BASE_MODEL(
            base_width=self.base_width, correction_cap=self.correction_cap
        )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.refiner_stem = r25.ConvNormAct(REFINER_INPUT_CHANNELS, REFINER_WIDTH)
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
        if features.ndim != 4 or features.shape[1] != DISPERSION_INPUT_CHANNELS:
            raise ValueError(f"unexpected R37 model input: {tuple(features.shape)}")
        with torch.no_grad():
            base_correction = self.base(
                features[:, :BASE_INPUT_CHANNELS], active=active
            )
        previous_delta = features[:, BASE_INPUT_CHANNELS]
        following_delta = features[:, BASE_INPUT_CHANNELS + 1]
        temporal_curvature = previous_delta + following_delta
        temporal_slope = following_delta - previous_delta
        coarse = features[:, 0]
        spatial_highpass = coarse - F.avg_pool2d(
            coarse[:, None], kernel_size=5, stride=1, padding=2
        )[:, 0]
        base_highpass = base_correction - F.avg_pool2d(
            base_correction[:, None], kernel_size=5, stride=1, padding=2
        )[:, 0]
        refiner_input = torch.cat(
            [
                features,
                base_correction[:, None],
                temporal_curvature[:, None],
                temporal_slope[:, None],
                (spatial_highpass + base_highpass)[:, None],
            ],
            dim=1,
        )
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
            raise RuntimeError(f"R28 checkpoint missing base keys: {missing[:5]}")
        for key in base_keys:
            full[f"base.{key}"] = state_dict[key]
        return super().load_state_dict(full, strict=True)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate = np.asarray([row["candidate_rel_l2"] for row in rows], dtype=np.float64)
    parent = np.asarray([row["parent_rel_l2"] for row in rows], dtype=np.float64)
    if not len(rows):
        return {"count": 0}
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
def evaluate_r37(
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
        previous_positions, next_positions = r29c.context_indices(times.astype(np.float64))
        coarse_all = np.asarray(handle["coarse_norm"][local_index], dtype=np.float32)
        truth_all = np.asarray(handle["truth_norm"][local_index], dtype=np.float32)
        candidate_error = parent_error = target_square = correction_square = 0.0
        for start in range(0, len(times), int(batch_size)):
            stop = min(start + int(batch_size), len(times))
            positions = np.arange(start, stop)
            coarse = torch.from_numpy(coarse_all[start:stop]).to(device)
            truth = torch.from_numpy(truth_all[start:stop]).to(device)
            previous = torch.from_numpy(coarse_all[previous_positions[positions]]).to(device)
            following = torch.from_numpy(coarse_all[next_positions[positions]]).to(device)
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
            features = torch.cat(
                [features, (previous - coarse)[:, None], (following - coarse)[:, None]],
                dim=1,
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
        "schema": f"r37_spatiotemporal_defect_refiner_{stage}_v1",
        "role": "R28_already_opened_train_holdout_development_only",
        "frozen_parent": "R28_local_plus_spectral_residual_operator",
        "deployment_features": [
            "coarse_t_minus_lag_minus_current", "coarse_t_plus_lag_minus_current",
            "temporal_curvature", "temporal_slope", "spatial_highpass",
            "source_frequency", "local_kh", "accumulated_travel_cycles",
        ],
        "context_lag_s": float(r29c.CONTEXT_LAG_S),
        "refiner_width": REFINER_WIDTH,
        "refiner_cap": REFINER_CAP,
        "zero_start": True,
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": r25.sha256_file(SCRIPT_PATH),
        "r29c_components_sha256": r25.sha256_file(R29C_PATH),
    }
    if stage == "terminal":
        terminal = json.loads((output_dir / "terminal.json").read_text(encoding="utf-8"))
        payload.update({
            "status": terminal["status"],
            "best_epoch": terminal["best_epoch"],
            "best_metrics": terminal["best_metrics"],
            "checkpoint": terminal["checkpoint"],
            "checkpoint_sha256": terminal["checkpoint_sha256"],
            "absolute_goal_passed": terminal["absolute_goal_passed"],
        })
    else:
        payload.update({
            "status": "frozen_before_development_training",
            "success_gate": "mean_and_max_record_relative_L2_lte_0p05",
            "evidence_boundary": "A pass only authorizes the preregistered fresh holdout.",
        })
    r25.atomic_json(payload, output_dir / f"r37_{stage}.json")


def main() -> int:
    if int(r25.INPUT_CHANNELS) != BASE_INPUT_CHANNELS:
        r25.INPUT_CHANNELS = BASE_INPUT_CHANNELS
    r29a.LateTailFitFrameDataset = SpatiotemporalDefectDataset
    r29a.r26.TailSpectralResidualUNet = SpatiotemporalDefectRefiner
    r29a.r25.evaluate = evaluate_r37
    write_sidecar("preregistration")
    result = r29a.main()
    write_sidecar("terminal")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
