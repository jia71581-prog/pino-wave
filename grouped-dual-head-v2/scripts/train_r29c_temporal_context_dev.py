#!/usr/bin/env python3
"""R29C development wrapper with coarse-wavefield temporal context.

Two deployment-available channels, u(t-lag)-u(t) and u(t+lag)-u(t), are added
to the R28/R29A feature tensor.  Existing R28 weights are embedded exactly and
the new input weights start at zero, so the initial prediction must reproduce
R28.  Only the already-opened R28 train holdout is used for development.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
R29A_PATH = SCRIPT_PATH.with_name("train_r29a_late_tail_finetune.py")
SPEC = importlib.util.spec_from_file_location("r29c_r29a_driver", R29A_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R29A driver: {R29A_PATH}")
r29a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r29a
SPEC.loader.exec_module(r29a)
r25 = r29a.r25


def pop_float_argument(name: str, default: float) -> float:
    if name not in sys.argv:
        return float(default)
    position = sys.argv.index(name)
    try:
        value = float(sys.argv[position + 1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"invalid {name}") from error
    del sys.argv[position : position + 2]
    return value


CONTEXT_LAG_S = pop_float_argument("--context-lag-s", 0.015)
if CONTEXT_LAG_S <= 0.0:
    raise ValueError("context lag must be positive")

BASE_DATASET = r29a.LateTailFitFrameDataset
BASE_MODEL = r29a.r26.TailSpectralResidualUNet
BASE_INPUT_CHANNELS = int(r25.INPUT_CHANNELS)
CONTEXT_INPUT_CHANNELS = BASE_INPUT_CHANNELS + 2


def nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    insertion = np.searchsorted(times, targets)
    high = np.clip(insertion, 0, len(times) - 1)
    low = np.clip(insertion - 1, 0, len(times) - 1)
    choose_low = np.abs(times[low] - targets) <= np.abs(times[high] - targets)
    return np.where(choose_low, low, high).astype(np.int64)


def context_indices(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        nearest_indices(times, times - float(CONTEXT_LAG_S)),
        nearest_indices(times, times + float(CONTEXT_LAG_S)),
    )


class ContextLateTailDataset(BASE_DATASET):
    def __init__(self, collection: Any, *, late_start_s: float):
        super().__init__(collection, late_start_s=late_start_s)
        times = np.asarray(collection.handles[0]["time_s"][:], dtype=np.float64)
        self.previous_positions, self.next_positions = context_indices(times)

    def __getitem__(self, index: int):
        base_index = int(self.mapping[int(index)])
        record_position, frame_position = divmod(base_index, self.time_count)
        file_index, local_index = self.collection.records[record_position]
        handle = self.collection.handles[file_index]
        features, coarse, truth, mean_energy, active = r25.FitFrameDataset.__getitem__(
            self, base_index
        )
        previous = torch.from_numpy(
            np.asarray(
                handle["coarse_norm"][
                    local_index, int(self.previous_positions[frame_position])
                ],
                dtype=np.float32,
            )
        )
        following = torch.from_numpy(
            np.asarray(
                handle["coarse_norm"][
                    local_index, int(self.next_positions[frame_position])
                ],
                dtype=np.float32,
            )
        )
        context = torch.stack([previous - coarse, following - coarse], dim=0)
        return torch.cat([features, context], dim=0), coarse, truth, mean_energy, active


class ContextTailSpectralResidualUNet(BASE_MODEL):
    """R28 model with two zero-initialized additional input channels."""

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
                and expected.shape[1] == CONTEXT_INPUT_CHANNELS
                and tuple(value.shape[2:]) == tuple(expected.shape[2:])
            ):
                padded = torch.zeros_like(expected)
                padded[:, :BASE_INPUT_CHANNELS].copy_(value)
                adapted[key] = padded
                continue
            raise RuntimeError(
                f"unsupported R29C checkpoint shape for {key}: "
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
def evaluate_context(
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
        previous_positions, next_positions = context_indices(times.astype(np.float64))
        coarse_all = np.asarray(handle["coarse_norm"][local_index], dtype=np.float32)
        truth_all = np.asarray(handle["truth_norm"][local_index], dtype=np.float32)
        candidate_error = 0.0
        parent_error = 0.0
        target_square = 0.0
        correction_square = 0.0
        for start in range(0, len(times), int(batch_size)):
            stop = min(start + int(batch_size), len(times))
            positions = np.arange(start, stop)
            coarse = torch.from_numpy(coarse_all[start:stop]).to(device)
            truth = torch.from_numpy(truth_all[start:stop]).to(device)
            previous = torch.from_numpy(coarse_all[previous_positions[positions]]).to(device)
            following = torch.from_numpy(coarse_all[next_positions[positions]]).to(device)
            block = stop - start
            time_tensor = torch.from_numpy(times[start:stop]).to(device)
            features = r25.make_dynamic_features(
                coarse,
                static[None].expand(block, -1, -1, -1),
                time_s=time_tensor,
                source_f0_hz=torch.full((block,), f0, device=device),
                source_t0_s=torch.full((block,), t0, device=device),
            )
            features = torch.cat(
                [features, (previous - coarse)[:, None], (following - coarse)[:, None]],
                dim=1,
            )
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
    if int(__import__("os").environ.get("RANK", "0")) != 0:
        return
    output_dir = Path(argument_value("--output-dir")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": f"r29c_temporal_context_{stage}_v1",
        "role": "R28_already_opened_train_holdout_development_only",
        "context_lag_s": float(CONTEXT_LAG_S),
        "context_channels": ["coarse_previous_minus_current", "coarse_following_minus_current"],
        "input_channels": int(CONTEXT_INPUT_CHANNELS),
        "initial_embedding": "R28_first_13_channel_weights_exact_and_two_new_channel_weights_zero",
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
    r25.atomic_json(payload, output_dir / f"r29c_{stage}.json")


def main() -> int:
    r29a.LateTailFitFrameDataset = ContextLateTailDataset
    r29a.r26.TailSpectralResidualUNet = ContextTailSpectralResidualUNet
    r29a.r25.evaluate = evaluate_context
    r29a.r25.INPUT_CHANNELS = int(CONTEXT_INPUT_CHANNELS)
    write_sidecar("preregistration")
    result = r29a.main()
    write_sidecar("terminal")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
