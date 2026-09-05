#!/usr/bin/env python3
"""R44 train-only physical-space frequency-residual capacity diagnostic.

R40/R42 applied a spatial FNO to the *index plane* of retained DCT
coefficients.  R44 first synthesizes those coefficients back to the physical
x-z grid and applies the operator there.  This pilot deliberately trains only
on the hardest fit records and uses the already-opened group-disjoint
development set only for a final direction check.  Frozen R29B/final/test
partitions are never opened.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.fft import idct, idctn
import torch
from torch import nn
from torch.nn import functional as F


GRID_SIZE = 201
PHYSICAL_INPUT_CHANNELS = 18


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("r40_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import R40 module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def make_synthesis_matrix(grid: int, retained: int) -> np.ndarray:
    basis = np.eye(int(grid), dtype=np.float32)
    matrix = idct(basis, type=2, norm="ortho", axis=0)
    return np.asarray(matrix[:, : int(retained)], dtype=np.float32)


def synthesize(coefficients: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Synthesize retained orthonormal DCT coefficients on both spatial axes."""
    if coefficients.ndim != 4:
        raise ValueError("coefficients must be [batch,channels,retained,retained]")
    return torch.einsum(
        "ir,bcrs,js->bcij", matrix, coefficients.float(), matrix
    )


def verify_synthesis_contract(matrix: np.ndarray, retained: int) -> float:
    rng = np.random.default_rng(4401)
    coefficients = rng.normal(size=(2, retained, retained)).astype(np.float32)
    padded = np.zeros((2, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    padded[:, :retained, :retained] = coefficients
    reference = idctn(padded, type=2, norm="ortho", axes=(-2, -1))
    torch_value = synthesize(
        torch.from_numpy(coefficients[:, None]), torch.from_numpy(matrix)
    )[:, 0].numpy()
    error = float(np.max(np.abs(reference - torch_value)))
    if error > 2.0e-5:
        raise RuntimeError(f"R44 DCT synthesis contract failed: {error}")
    return error


def physical_features(
    base_coefficients: torch.Tensor,
    static_spatial: torch.Tensor,
    *,
    synthesis_matrix: torch.Tensor,
    frequency_hz: torch.Tensor,
    frequency_scale: torch.Tensor,
    source_f0_hz: torch.Tensor,
    source_t0_s: torch.Tensor,
) -> torch.Tensor:
    if base_coefficients.ndim != 4 or base_coefficients.shape[1] != 2:
        raise ValueError("base coefficients must be [batch,2,retained,retained]")
    batch = int(base_coefficients.shape[0])
    if static_spatial.shape != (batch, 7, GRID_SIZE, GRID_SIZE):
        raise ValueError(f"unexpected static physical shape: {static_spatial.shape}")
    base_spatial = synthesize(base_coefficients, synthesis_matrix)
    dtype = base_spatial.dtype
    device = base_spatial.device

    coordinate_x = static_spatial[:, 5:6].to(dtype=dtype)
    coordinate_z = static_spatial[:, 6:7].to(dtype=dtype)
    coordinate_radius = torch.sqrt(
        coordinate_x.square() + coordinate_z.square()
    ) / math.sqrt(2.0)

    frequency = (frequency_hz / 80.0).clamp(0.0, 1.25)
    f0 = (source_f0_hz / 30.0).clamp(0.0, 1.5)
    t0 = (source_t0_s / 0.10).clamp(0.0, 2.0)
    ratio = (frequency_hz / source_f0_hz.clamp_min(1.0) / 3.0).clamp(
        0.0, 2.0
    )
    mean_velocity = ((4500.0 + 2500.0 * static_spatial[:, 0].mean((1, 2))) / 6000.0).clamp(
        0.0, 1.5
    )
    log_frequency_scale = (
        torch.log10(frequency_scale.clamp_min(1.0e-8)).clamp(-8.0, 2.0)
        / 4.0
    )
    scalars = torch.stack(
        (frequency, f0, t0, ratio, mean_velocity, log_frequency_scale), dim=1
    )
    scalar_maps = scalars[:, :, None, None].expand(
        -1, -1, GRID_SIZE, GRID_SIZE
    )
    features = torch.cat(
        (
            base_spatial,
            static_spatial.to(dtype=dtype),
            scalar_maps,
            coordinate_x,
            coordinate_z,
            coordinate_radius,
        ),
        dim=1,
    )
    if features.shape[1] != PHYSICAL_INPUT_CHANNELS:
        raise RuntimeError(f"unexpected R44 feature shape: {features.shape}")
    return features


class PhysicalFrequencyResidualFNO(nn.Module):
    def __init__(
        self,
        r40,
        *,
        width: int,
        modes: int,
        blocks: int,
        correction_cap: float,
    ):
        super().__init__()
        self.width = int(width)
        self.modes = int(modes)
        self.blocks_count = int(blocks)
        self.correction_cap = float(correction_cap)
        self.stem = nn.Conv2d(PHYSICAL_INPUT_CHANNELS, self.width, 1)
        self.blocks = nn.Sequential(
            *[
                r40.FrequencyFNOBlock(self.width, self.modes)
                for _ in range(self.blocks_count)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(self.width, self.width, 1),
            nn.GELU(),
            nn.Conv2d(self.width, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.blocks(self.stem(features))
        return self.correction_cap * torch.tanh(self.head(value))


def load_curriculum(
    path: Path, fit
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "r42_train_only_hard_record_curriculum_v1":
        raise RuntimeError("unexpected hard-curriculum schema")
    if bool(payload.get("data_boundary", {}).get("opened_development_holdout_used")):
        raise RuntimeError("hard curriculum unexpectedly used development truth")
    records = list(payload.get("records", []))
    if len(records) != len(fit.records):
        raise RuntimeError("hard curriculum record count mismatch")
    base_errors = np.empty(len(records), dtype=np.float64)
    for expected_position, record in enumerate(records):
        position = int(record["record_position"])
        if position != expected_position:
            raise RuntimeError("hard curriculum record positions are not canonical")
        if str(record["sample_id"]) != fit.sample_ids[position]:
            raise RuntimeError("hard curriculum sample/cache mismatch")
        if str(record["group_id"]) != fit.group_ids[position]:
            raise RuntimeError("hard curriculum group/cache mismatch")
        base_errors[position] = float(record["base_record_rel_l2"])
    return payload, records, base_errors


def preload_hard_records(
    fit,
    manifest_records: Sequence[Mapping[str, Any]],
    base_errors: np.ndarray,
    *,
    count: int,
    synthesis_matrix_cpu: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[int]]:
    positions = sorted(
        range(len(fit.records)),
        key=lambda position: (-float(base_errors[position]), position),
    )[: int(count)]
    loaded: list[dict[str, Any]] = []
    for position in positions:
        file_index, local_index = fit.records[position]
        handle = fit.handles[file_index]
        if str(handle["sample_id"].asstr()[local_index]) != fit.sample_ids[position]:
            raise RuntimeError("preload sample identity mismatch")
        static_norm = torch.from_numpy(
            np.asarray(handle["static_dct_norm"][local_index], dtype=np.float32)
        )[None]
        static_scale = torch.from_numpy(
            np.asarray(handle["static_dct_scale"][local_index], dtype=np.float32)
        )
        static_coefficients = static_norm * static_scale[None, :, None, None]
        static_spatial = synthesize(
            static_coefficients, synthesis_matrix_cpu
        )[0].contiguous()
        loaded.append(
            {
                "record_position": int(position),
                "sample_id": fit.sample_ids[position],
                "group_id": fit.group_ids[position],
                "family": fit.families[position],
                "base_error": float(base_errors[position]),
                "manifest_record": dict(manifest_records[position]),
                "base_coefficients": np.asarray(
                    handle["base_dct_norm"][local_index], dtype=np.float16
                ),
                "residual_coefficients": np.asarray(
                    handle["residual_dct_norm"][local_index], dtype=np.float16
                ),
                "static_spatial": static_spatial.numpy().astype(np.float32),
                "frequency_hz": np.asarray(
                    handle["frequency_hz"], dtype=np.float32
                ),
                "frequency_scale": np.asarray(
                    handle["frequency_scale"][local_index], dtype=np.float32
                ),
                "source_f0_hz": float(handle["source_f0_hz"][local_index]),
                "source_t0_s": float(handle["source_t0_s"][local_index]),
                "target_square_total": float(
                    handle["target_square_total"][local_index]
                ),
            }
        )
    return loaded, positions


def make_batch(
    record: Mapping[str, Any],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count = int(len(indices))
    static = torch.from_numpy(np.asarray(record["static_spatial"])).to(device)
    batch = {
        "base": torch.from_numpy(
            np.asarray(record["base_coefficients"])[indices].astype(np.float32)
        ).to(device),
        "static": static[None].expand(count, -1, -1, -1),
        "frequency_hz": torch.from_numpy(
            np.asarray(record["frequency_hz"])[indices].astype(np.float32)
        ).to(device),
        "frequency_scale": torch.from_numpy(
            np.asarray(record["frequency_scale"])[indices].astype(np.float32)
        ).to(device),
        "source_f0_hz": torch.full(
            (count,), float(record["source_f0_hz"]), device=device
        ),
        "source_t0_s": torch.full(
            (count,), float(record["source_t0_s"]), device=device
        ),
    }
    if "residual_coefficients" in record:
        batch["residual"] = torch.from_numpy(
            np.asarray(record["residual_coefficients"])[indices].astype(np.float32)
        ).to(device)
    return batch


@torch.inference_mode()
def predict_record(
    model: nn.Module,
    record: Mapping[str, Any],
    *,
    synthesis_matrix: torch.Tensor,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    model.eval()
    frequency_count = len(record["frequency_hz"])
    blocks: list[np.ndarray] = []
    for start in range(0, frequency_count, int(batch_size)):
        stop = min(start + int(batch_size), frequency_count)
        indices = np.arange(start, stop, dtype=np.int64)
        batch = make_batch(record, indices, device=device)
        features = physical_features(
            batch["base"],
            batch["static"],
            synthesis_matrix=synthesis_matrix,
            frequency_hz=batch["frequency_hz"],
            frequency_scale=batch["frequency_scale"],
            source_f0_hz=batch["source_f0_hz"],
            source_t0_s=batch["source_t0_s"],
        )
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp
            else nullcontext()
        )
        with context:
            prediction = model(features)
        blocks.append(prediction.float().cpu().numpy())
    return np.concatenate(blocks, axis=0)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate = np.asarray(
        [float(row["candidate_rel_l2"]) for row in rows], dtype=np.float64
    )
    parent = np.asarray(
        [float(row["parent_rel_l2"]) for row in rows], dtype=np.float64
    )
    return {
        "count": int(len(rows)),
        "candidate_mean": float(candidate.mean()),
        "candidate_max": float(candidate.max()),
        "candidate_median": float(np.median(candidate)),
        "parent_mean": float(parent.mean()),
        "parent_max": float(parent.max()),
        "mean_relative_improvement": float(
            1.0 - candidate.mean() / max(float(parent.mean()), 1.0e-30)
        ),
        "max_relative_improvement": float(
            1.0 - candidate.max() / max(float(parent.max()), 1.0e-30)
        ),
    }


@torch.inference_mode()
def evaluate_hard_fit_scales(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    frequency_weight: np.ndarray,
    scales: Sequence[float],
    synthesis_matrix: torch.Tensor,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    rows_by_scale: dict[float, list[dict[str, Any]]] = {
        float(scale): [] for scale in scales
    }
    matrix_cpu = synthesis_matrix.detach().cpu()
    for record in records:
        prediction = predict_record(
            model,
            record,
            synthesis_matrix=synthesis_matrix,
            device=device,
            batch_size=batch_size,
            amp=amp,
        ).astype(np.float64)
        residual_coefficients = torch.from_numpy(
            np.asarray(record["residual_coefficients"], dtype=np.float32)
        )
        target_spatial = synthesize(
            residual_coefficients, matrix_cpu
        ).numpy().astype(np.float64)
        frequency_scale = np.asarray(record["frequency_scale"], dtype=np.float64)
        target_total = float(record["target_square_total"])
        parent_retained = float(
            np.sum(
                frequency_weight
                * frequency_scale**2
                * np.sum(target_spatial**2, axis=(1, 2, 3), dtype=np.float64)
            )
        )
        parent_total = float(record["base_error"]) ** 2 * target_total
        irreducible = max(parent_total - parent_retained, 0.0)
        for scale in rows_by_scale:
            error = float(scale) * prediction - target_spatial
            candidate_retained = float(
                np.sum(
                    frequency_weight
                    * frequency_scale**2
                    * np.sum(error**2, axis=(1, 2, 3), dtype=np.float64)
                )
            )
            candidate_rel = math.sqrt(
                (irreducible + candidate_retained) / max(target_total, 1.0e-30)
            )
            rows_by_scale[scale].append(
                {
                    "record_position": int(record["record_position"]),
                    "sample_id": str(record["sample_id"]),
                    "group_id": str(record["group_id"]),
                    "family": str(record["family"]),
                    "candidate_rel_l2": candidate_rel,
                    "parent_rel_l2": float(record["base_error"]),
                    "representation_oracle_rel_l2": math.sqrt(
                        irreducible / max(target_total, 1.0e-30)
                    ),
                    "relative_improvement": float(
                        1.0
                        - candidate_rel / max(float(record["base_error"]), 1.0e-30)
                    ),
                }
            )
    return {
        str(scale): {
            "correction_scale": float(scale),
            "aggregate": summarize(rows),
            "records": rows,
        }
        for scale, rows in rows_by_scale.items()
    }


def cache_record_for_prediction(handle, local_index: int, *, matrix_cpu: torch.Tensor):
    static_norm = torch.from_numpy(
        np.asarray(handle["static_dct_norm"][local_index], dtype=np.float32)
    )[None]
    static_scale = torch.from_numpy(
        np.asarray(handle["static_dct_scale"][local_index], dtype=np.float32)
    )
    static_spatial = synthesize(
        static_norm * static_scale[None, :, None, None], matrix_cpu
    )[0].numpy()
    return {
        "base_coefficients": np.asarray(
            handle["base_dct_norm"][local_index], dtype=np.float16
        ),
        "static_spatial": static_spatial.astype(np.float32),
        "frequency_hz": np.asarray(handle["frequency_hz"], dtype=np.float32),
        "frequency_scale": np.asarray(
            handle["frequency_scale"][local_index], dtype=np.float32
        ),
        "source_f0_hz": float(handle["source_f0_hz"][local_index]),
        "source_t0_s": float(handle["source_t0_s"][local_index]),
    }


@torch.inference_mode()
def evaluate_holdout_scales(
    r40,
    model: nn.Module,
    collection,
    *,
    scales: Sequence[float],
    synthesis_matrix: torch.Tensor,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    if collection.expected_subset != "holdout":
        raise ValueError("R44 holdout evaluation requires opened holdout caches")
    rows_by_scale: dict[float, list[dict[str, Any]]] = {
        float(scale): [] for scale in scales
    }
    family_by_scale = {
        float(scale): {name: [] for name in r40.FAMILIES} for scale in scales
    }
    frequency_weight = r40.rfft_weights(r40.TIME_COUNT)[
        collection.frequency_indices
    ]
    matrix_cpu = synthesis_matrix.detach().cpu()
    for file_index, local_index in collection.records:
        handle = collection.handles[file_index]
        record = cache_record_for_prediction(
            handle, local_index, matrix_cpu=matrix_cpu
        )
        prediction = predict_record(
            model,
            record,
            synthesis_matrix=synthesis_matrix,
            device=device,
            batch_size=batch_size,
            amp=amp,
        )
        correction = (
            prediction[:, 0] + 1j * prediction[:, 1]
        ) * np.asarray(record["frequency_scale"])[:, None, None]
        base_selected = r40.channels_to_complex(
            np.asarray(handle["base_spectrum_selected"][local_index], dtype=np.float32)
        )
        truth_selected = r40.channels_to_complex(
            np.asarray(handle["truth_spectrum_selected"][local_index], dtype=np.float32)
        )
        unselected = float(handle["base_error_square_unselected"][local_index])
        target_total = float(handle["target_square_total"][local_index])
        sample_id = str(handle["sample_id"].asstr()[local_index])
        group_id = str(handle["group_id"].asstr()[local_index])
        family = str(handle["family"].asstr()[local_index])
        parent_selected = float(
            np.sum(
                frequency_weight
                * np.sum(
                    np.abs(base_selected - truth_selected) ** 2,
                    axis=(1, 2),
                    dtype=np.float64,
                )
            )
        )
        parent_rel = math.sqrt(
            (unselected + parent_selected) / max(target_total, 1.0e-30)
        )
        for scale in rows_by_scale:
            candidate_selected = float(
                np.sum(
                    frequency_weight
                    * np.sum(
                        np.abs(
                            base_selected
                            + float(scale) * correction
                            - truth_selected
                        )
                        ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                )
            )
            candidate_rel = math.sqrt(
                (unselected + candidate_selected) / max(target_total, 1.0e-30)
            )
            row = {
                "sample_id": sample_id,
                "group_id": group_id,
                "family": family,
                "candidate_rel_l2": candidate_rel,
                "parent_rel_l2": parent_rel,
                "relative_improvement": float(
                    1.0 - candidate_rel / max(parent_rel, 1.0e-30)
                ),
            }
            rows_by_scale[scale].append(row)
            family_by_scale[scale][family].append(row)
    evaluations: dict[str, Any] = {}
    for scale, rows in rows_by_scale.items():
        aggregate = summarize(rows)
        passed = bool(
            float(aggregate["candidate_mean"]) <= 0.05
            and float(aggregate["candidate_max"]) <= 0.05
        )
        evaluations[str(scale)] = {
            "correction_scale": float(scale),
            "aggregate": aggregate,
            "per_family": {
                family: summarize(values)
                for family, values in family_by_scale[scale].items()
            },
            "absolute_goal": {
                "mean_lte_0p05": float(aggregate["candidate_mean"]) <= 0.05,
                "max_lte_0p05": float(aggregate["candidate_max"]) <= 0.05,
                "passed": passed,
            },
            "records": rows,
        }
    return evaluations


def choose_scale(evaluations: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        evaluations.values(),
        key=lambda item: (
            float(item["aggregate"]["candidate_max"])
            + 0.1 * float(item["aggregate"]["candidate_mean"]),
            float(item["correction_scale"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--fit-caches", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-caches", type=Path, nargs="+", required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-records", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--frequency-batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--correction-cap", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=4401)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    if args.hard_records < 1 or args.steps < 1 or args.frequency_batch < 1:
        raise ValueError("R44 counts must be positive")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("R44 requested CUDA but CUDA is unavailable")

    r40_path = args.r40_script.expanduser().resolve()
    r40 = load_module(r40_path)
    fit = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.fit_caches],
        expected_subset="fit",
    )
    holdout = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.holdout_caches],
        expected_subset="holdout",
    )
    curriculum_path = args.curriculum.expanduser().resolve()
    curriculum, manifest_records, base_errors = load_curriculum(
        curriculum_path, fit
    )
    retained = int(fit.retained)
    if int(holdout.retained) != retained:
        raise RuntimeError("fit/holdout retained DCT counts disagree")
    if not np.array_equal(fit.frequency_indices, holdout.frequency_indices):
        raise RuntimeError("fit/holdout frequency selections disagree")

    synthesis_array = make_synthesis_matrix(GRID_SIZE, retained)
    synthesis_error = verify_synthesis_contract(synthesis_array, retained)
    synthesis_cpu = torch.from_numpy(synthesis_array)
    synthesis_device = synthesis_cpu.to(device)
    hard_records, hard_positions = preload_hard_records(
        fit,
        manifest_records,
        base_errors,
        count=min(int(args.hard_records), len(fit.records)),
        synthesis_matrix_cpu=synthesis_cpu,
    )
    frequency_weight = r40.rfft_weights(r40.TIME_COUNT)[fit.frequency_indices].astype(
        np.float64
    )

    model = PhysicalFrequencyResidualFNO(
        r40,
        width=int(args.width),
        modes=int(args.modes),
        blocks=int(args.blocks),
        correction_cap=float(args.correction_cap),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    with torch.inference_mode():
        probe_indices = np.arange(
            min(int(args.frequency_batch), fit.frequency_count), dtype=np.int64
        )
        probe = make_batch(hard_records[0], probe_indices, device=device)
        probe_features = physical_features(
            probe["base"],
            probe["static"],
            synthesis_matrix=synthesis_device,
            frequency_hz=probe["frequency_hz"],
            frequency_scale=probe["frequency_scale"],
            source_f0_hz=probe["source_f0_hz"],
            source_t0_s=probe["source_t0_s"],
        )
        zero_output = float(model(probe_features).abs().max().cpu())
    if zero_output != 0.0:
        raise RuntimeError(f"R44 zero-output identity contract failed: {zero_output}")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".pt")
    scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    rng = np.random.default_rng(int(args.seed))
    record_order = rng.permutation(len(hard_records))
    losses: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    model.train()
    for step in range(1, int(args.steps) + 1):
        cycle_position = (step - 1) % len(hard_records)
        if cycle_position == 0 and step > 1:
            record_order = rng.permutation(len(hard_records))
        record = hard_records[int(record_order[cycle_position])]
        frequency_count = len(record["frequency_hz"])
        indices = rng.choice(
            frequency_count,
            size=min(int(args.frequency_batch), frequency_count),
            replace=False,
        ).astype(np.int64)
        batch = make_batch(record, indices, device=device)
        target_spatial = synthesize(batch["residual"], synthesis_device)
        features = physical_features(
            batch["base"],
            batch["static"],
            synthesis_matrix=synthesis_device,
            frequency_hz=batch["frequency_hz"],
            frequency_scale=batch["frequency_scale"],
            source_f0_hz=batch["source_f0_hz"],
            source_t0_s=batch["source_t0_s"],
        )
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if bool(args.amp)
            else nullcontext()
        )
        with context:
            prediction = model(features)
        weights = torch.from_numpy(frequency_weight[indices].astype(np.float32)).to(
            device
        ) * batch["frequency_scale"].float().square()
        error_square = (prediction.float() - target_spatial.float()).square().sum(
            dim=(1, 2, 3)
        )
        parent_square = target_spatial.float().square().sum(dim=(1, 2, 3))
        target_total = torch.tensor(
            float(record["target_square_total"]),
            device=device,
            dtype=torch.float32,
        ).clamp_min(1.0e-12)
        candidate_contribution = (
            float(frequency_count) * weights * error_square / target_total
        )
        parent_contribution = (
            float(frequency_count) * weights * parent_square / target_total
        )
        physical_mean = candidate_contribution.mean()
        tail_count = max(1, int(math.ceil(0.25 * len(indices))))
        physical_tail = torch.topk(
            candidate_contribution, k=tail_count
        ).values.mean()
        hinge = F.relu(
            torch.sqrt(candidate_contribution.clamp_min(1.0e-14))
            - torch.sqrt(parent_contribution.detach().clamp_min(1.0e-14))
        ).square().mean()
        shape = F.smooth_l1_loss(
            prediction.float(), target_spatial.float(), beta=0.02
        )
        objective = physical_mean + physical_tail + 2.0 * hinge + 0.002 * shape
        if not torch.isfinite(objective):
            raise FloatingPointError(f"nonfinite R44 loss at step {step}")
        objective.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu()
        )
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == int(args.steps):
            row = {
                "step": int(step),
                "sample_id": str(record["sample_id"]),
                "objective": float(objective.detach().cpu()),
                "physical_mean": float(physical_mean.detach().cpu()),
                "physical_tail": float(physical_tail.detach().cpu()),
                "hinge": float(hinge.detach().cpu()),
                "shape": float(shape.detach().cpu()),
                "gradient_norm": gradient_norm,
            }
            losses.append(row)
            print(json.dumps({"event": "r44_train", **row}), flush=True)

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            hard_evaluations = evaluate_hard_fit_scales(
                model,
                hard_records,
                frequency_weight=frequency_weight,
                scales=scales,
                synthesis_matrix=synthesis_device,
                device=device,
                batch_size=int(args.eval_batch),
                amp=bool(args.amp),
            )
            selected = choose_scale(hard_evaluations)
            evaluation = {
                "step": int(step),
                "hard_fit_scales": hard_evaluations,
                "selected": selected,
            }
            evaluations.append(evaluation)
            print(
                json.dumps(
                    {
                        "event": "r44_hard_fit_evaluation",
                        "step": int(step),
                        "selected_scale": float(selected["correction_scale"]),
                        "aggregate": selected["aggregate"],
                    }
                ),
                flush=True,
            )
            model.train()

    final_holdout = evaluate_holdout_scales(
        r40,
        model,
        holdout,
        scales=scales,
        synthesis_matrix=synthesis_device,
        device=device,
        batch_size=int(args.eval_batch),
        amp=bool(args.amp),
    )
    selected_holdout = choose_scale(final_holdout)
    torch.save(
        {
            "schema": "r44_physical_space_hard_pilot_checkpoint_v1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "configuration": vars(args),
            "hard_positions": hard_positions,
            "selected_holdout_scale": float(selected_holdout["correction_scale"]),
        },
        checkpoint_path,
    )
    payload = {
        "schema": "r44_physical_space_hard_pilot_result_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "purpose": "train_only_physical_space_operator_capacity_and_opened_dev_direction_check",
        "data_boundary": {
            "fit_truth_used": True,
            "opened_group_disjoint_development_used": True,
            "r29b_opened": False,
            "final_validation_opened": False,
            "test_id_opened": False,
            "paper_modified": False,
        },
        "contracts": {
            "dct_synthesis_max_abs_error": synthesis_error,
            "zero_output_max_abs": zero_output,
            "grid_size": GRID_SIZE,
            "retained_dct": retained,
            "frequency_count": int(fit.frequency_count),
        },
        "configuration": {
            "hard_records": len(hard_records),
            "steps": int(args.steps),
            "frequency_batch": int(args.frequency_batch),
            "eval_batch": int(args.eval_batch),
            "eval_every": int(args.eval_every),
            "width": int(args.width),
            "modes": int(args.modes),
            "blocks": int(args.blocks),
            "correction_cap": float(args.correction_cap),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "amp": bool(args.amp),
            "parameter_count": int(r40.parameter_count(model)),
            "objective": "record_balanced_full_wavefield_relative_energy_with_tail_hinge",
        },
        "evidence": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "r40_script": str(r40_path),
            "r40_script_sha256": sha256_file(r40_path),
            "curriculum": str(curriculum_path),
            "curriculum_sha256": sha256_file(curriculum_path),
            "fit_selection_sha256": fit.selection_sha256,
            "holdout_selection_sha256": holdout.selection_sha256,
            "checkpoint": str(checkpoint_path),
        },
        "hard_positions": hard_positions,
        "hard_records": [
            {
                "record_position": int(record["record_position"]),
                "sample_id": str(record["sample_id"]),
                "group_id": str(record["group_id"]),
                "family": str(record["family"]),
                "base_record_rel_l2": float(record["base_error"]),
            }
            for record in hard_records
        ],
        "loss_trace": losses,
        "hard_fit_evaluations": evaluations,
        "opened_development_final_scales": final_holdout,
        "opened_development_selected": selected_holdout,
        "absolute_goal": selected_holdout["absolute_goal"],
    }
    atomic_json(payload, output_path)
    print(
        json.dumps(
            {
                "event": "r44_complete",
                "output": str(output_path),
                "checkpoint": str(checkpoint_path),
                "selected_scale": float(selected_holdout["correction_scale"]),
                "aggregate": selected_holdout["aggregate"],
                "absolute_goal": selected_holdout["absolute_goal"],
            }
        ),
        flush=True,
    )
    fit.close()
    holdout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
