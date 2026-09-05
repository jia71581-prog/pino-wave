#!/usr/bin/env python3
"""Train R40 on train-only temporal-frequency/spatial-DCT residual caches.

The network receives only deployment-available base spectra, medium/source
features, and query frequency. Truth residuals are used only by the objective
and the already-opened group-disjoint development metric.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from scipy.fft import idctn
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


CACHE_SCHEMA = "r40_frequency_residual_cache_v2"
SUMMARY_SCHEMA = "r40_frequency_residual_cache_summary_v2"
CHECKPOINT_SCHEMA = "r40_frequency_residual_checkpoint_v1"
FAMILIES = ("uniform", "layered", "marmousi")
FAMILY_WEIGHTS = {"uniform": 1.0, "layered": 1.25, "marmousi": 1.75}
INPUT_CHANNELS = 32
TIME_COUNT = 401
SOURCE_GRID_SIZE = 201


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
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def text_values(dataset: h5py.Dataset) -> list[str]:
    return [str(value) for value in dataset.asstr()[:]]


def channels_to_complex(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 2:
        raise ValueError(f"expected [frequency,2,height,width], got {array.shape}")
    return array[:, 0] + 1j * array[:, 1]


def rfft_weights(count: int) -> np.ndarray:
    result = np.full(count // 2 + 1, 2.0, dtype=np.float64)
    result[0] = 1.0
    if count % 2 == 0:
        result[-1] = 1.0
    return result


class FrequencyCacheCollection:
    def __init__(self, paths: Sequence[Path], *, expected_subset: str):
        if not paths:
            raise ValueError(f"no {expected_subset} caches supplied")
        self.expected_subset = str(expected_subset)
        self.paths = tuple(path.expanduser().resolve() for path in paths)
        self.handles: list[h5py.File] = []
        self.records: list[tuple[int, int]] = []
        self.cache_evidence: dict[str, dict[str, Any]] = {}
        self.selection_sha256: str | None = None
        self.frequency_indices: np.ndarray | None = None
        self.frequency_hz: np.ndarray | None = None
        self.retained: int | None = None
        self.stored_dt_s: float | None = None
        try:
            for file_index, path in enumerate(self.paths):
                if not path.is_file():
                    raise FileNotFoundError(path)
                summary_path = path.with_suffix(path.suffix + ".summary.json")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("schema") != SUMMARY_SCHEMA:
                    raise RuntimeError(f"unexpected cache summary schema: {summary_path}")
                if summary.get("status") != "complete":
                    raise RuntimeError(f"cache summary is not complete: {summary_path}")
                if summary.get("subset") != self.expected_subset:
                    raise RuntimeError(f"cache summary subset mismatch: {summary_path}")
                if bool(summary.get("validation_opened")) or bool(
                    summary.get("test_id_opened")
                ):
                    raise RuntimeError(f"evidence boundary violation: {summary_path}")
                if int(summary.get("output_bytes", -1)) != path.stat().st_size:
                    raise RuntimeError(f"cache byte count changed: {path}")

                handle = h5py.File(path, "r", swmr=True)
                self.handles.append(handle)
                if str(handle.attrs.get("schema", "")) != CACHE_SCHEMA:
                    raise RuntimeError(f"unexpected cache schema: {path}")
                if str(handle.attrs.get("status", "")) != "complete":
                    raise RuntimeError(f"cache is not complete: {path}")
                if str(handle.attrs.get("subset", "")) != self.expected_subset:
                    raise RuntimeError(f"cache subset mismatch: {path}")
                if str(handle.attrs.get("truth_policy", "")) != (
                    "train_supervision_or_opened_development_only"
                ):
                    raise RuntimeError(f"truth policy mismatch: {path}")

                selection = str(handle.attrs.get("selection_sha256", ""))
                if selection != str(summary.get("selection_sha256", "")):
                    raise RuntimeError(f"summary/cache selection mismatch: {path}")
                if self.selection_sha256 is None:
                    self.selection_sha256 = selection
                elif selection != self.selection_sha256:
                    raise RuntimeError("cache selection digests disagree")

                indices = np.asarray(handle["frequency_indices"], dtype=np.int64)
                frequencies = np.asarray(handle["frequency_hz"], dtype=np.float32)
                retained = int(handle["base_dct_norm"].shape[-1])
                dt = float(handle.attrs["stored_dt_s"])
                if self.frequency_indices is None:
                    self.frequency_indices = indices
                    self.frequency_hz = frequencies
                    self.retained = retained
                    self.stored_dt_s = dt
                else:
                    if not np.array_equal(indices, self.frequency_indices):
                        raise RuntimeError("cache frequency indices disagree")
                    if not np.allclose(frequencies, self.frequency_hz, atol=1.0e-6):
                        raise RuntimeError("cache frequencies disagree")
                    if retained != self.retained or abs(dt - float(self.stored_dt_s)) > 1e-12:
                        raise RuntimeError("cache geometry disagrees")

                count = int(handle["base_dct_norm"].shape[0])
                expected = (count, len(indices), 2, retained, retained)
                if handle["base_dct_norm"].shape != expected:
                    raise RuntimeError(f"unexpected base DCT shape: {path}")
                if handle["residual_dct_norm"].shape != expected:
                    raise RuntimeError(f"unexpected residual DCT shape: {path}")
                if handle["static_dct_norm"].shape != (count, 7, retained, retained):
                    raise RuntimeError(f"unexpected static DCT shape: {path}")
                if handle["static_dct_scale"].shape != (count, 7):
                    raise RuntimeError(f"missing static DCT scales: {path}")
                self.records.extend((file_index, local) for local in range(count))
                self.cache_evidence[str(path)] = {
                    "bytes": int(summary["output_bytes"]),
                    "sha256": str(summary["output_sha256"]),
                    "summary": str(summary_path),
                }
        except Exception:
            self.close()
            raise

        if (
            self.selection_sha256 is None
            or self.frequency_indices is None
            or self.frequency_hz is None
            or self.retained is None
            or self.stored_dt_s is None
        ):
            self.close()
            raise RuntimeError("empty R40 cache collection")
        expected_hz = np.fft.rfftfreq(TIME_COUNT, d=self.stored_dt_s)[
            self.frequency_indices
        ]
        if not np.allclose(expected_hz, self.frequency_hz, atol=1.0e-5):
            self.close()
            raise RuntimeError("stored frequencies do not match the frozen 401-frame axis")

        sample_ids: list[str] = []
        group_ids: list[str] = []
        families: list[str] = []
        for handle in self.handles:
            sample_ids.extend(text_values(handle["sample_id"]))
            group_ids.extend(text_values(handle["group_id"]))
            families.extend(text_values(handle["family"]))
        if len(sample_ids) != len(set(sample_ids)):
            self.close()
            raise RuntimeError("cache collection has duplicate sample IDs")
        if any(family not in FAMILIES for family in families):
            self.close()
            raise RuntimeError("cache collection has an unexpected family")
        self.sample_ids = tuple(sample_ids)
        self.group_ids = tuple(group_ids)
        self.families = tuple(families)

    @property
    def frequency_count(self) -> int:
        return int(len(self.frequency_indices))

    def close(self) -> None:
        for handle in getattr(self, "handles", []):
            try:
                handle.close()
            except Exception:
                pass
        self.handles = []

    def __del__(self):
        self.close()


class FitFrequencyDataset(Dataset):
    def __init__(self, collection: FrequencyCacheCollection):
        if collection.expected_subset != "fit":
            raise ValueError("FitFrequencyDataset requires fit caches")
        self.collection = collection
        self.frequency_count = collection.frequency_count
        self.frequency_weights = rfft_weights(TIME_COUNT)[
            collection.frequency_indices
        ].astype(np.float32)

    def __len__(self) -> int:
        return len(self.collection.records) * self.frequency_count

    def __getitem__(self, index: int):
        record_position, frequency_position = divmod(
            int(index), self.frequency_count
        )
        file_index, local_index = self.collection.records[record_position]
        handle = self.collection.handles[file_index]
        family = str(handle["family"].asstr()[local_index])
        return (
            torch.from_numpy(
                np.asarray(
                    handle["base_dct_norm"][local_index, frequency_position],
                    dtype=np.float32,
                )
            ),
            torch.from_numpy(
                np.asarray(
                    handle["residual_dct_norm"][local_index, frequency_position],
                    dtype=np.float32,
                )
            ),
            torch.from_numpy(
                np.asarray(handle["static_dct_norm"][local_index], dtype=np.float32)
            ),
            torch.from_numpy(
                np.asarray(handle["static_dct_scale"][local_index], dtype=np.float32)
            ),
            torch.tensor(
                float(handle["frequency_hz"][frequency_position]), dtype=torch.float32
            ),
            torch.tensor(
                float(handle["source_f0_hz"][local_index]), dtype=torch.float32
            ),
            torch.tensor(
                float(handle["source_t0_s"][local_index]), dtype=torch.float32
            ),
            torch.tensor(
                float(handle["frequency_scale"][local_index, frequency_position]),
                dtype=torch.float32,
            ),
            torch.tensor(
                float(handle["target_square_total"][local_index]),
                dtype=torch.float32,
            ),
            torch.tensor(
                float(self.frequency_weights[frequency_position]), dtype=torch.float32
            ),
            torch.tensor(float(FAMILY_WEIGHTS[family]), dtype=torch.float32),
        )


def make_features(
    base: torch.Tensor,
    static_norm: torch.Tensor,
    static_scale: torch.Tensor,
    *,
    frequency_hz: torch.Tensor,
    frequency_scale: torch.Tensor,
    source_f0_hz: torch.Tensor,
    source_t0_s: torch.Tensor,
) -> torch.Tensor:
    if base.ndim != 4 or base.shape[1] != 2:
        raise ValueError("base must be [batch,2,height,width]")
    if static_norm.ndim != 4 or static_norm.shape[1] != 7:
        raise ValueError("static DCT must be [batch,7,height,width]")
    batch, _, height, width = base.shape
    if static_scale.shape != (batch, 7):
        raise ValueError("static scale must be [batch,7]")
    device = base.device
    dtype = base.dtype
    ky = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    kx = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    zz, xx = torch.meshgrid(ky, kx, indexing="ij")
    radius = torch.sqrt(xx.square() + zz.square()) / math.sqrt(2.0)
    coordinate = torch.stack((xx, zz, radius), dim=0)[None].expand(batch, -1, -1, -1)

    log_scale = torch.log10(static_scale.clamp_min(1.0e-6)).clamp(-6.0, 3.0) / 3.0
    scale_maps = log_scale[:, :, None, None].expand(-1, -1, height, width)
    raw_static = (
        static_norm * static_scale[:, :, None, None] / float(SOURCE_GRID_SIZE)
    )
    mean_velocity_feature = raw_static[:, 0, 0, 0]
    mean_velocity = (
        (4500.0 + 2500.0 * mean_velocity_feature) / 6000.0
    ).clamp(0.0, 1.5)
    frequency = (frequency_hz / 80.0).clamp(0.0, 1.25)
    f0 = (source_f0_hz / 30.0).clamp(0.0, 1.5)
    t0 = (source_t0_s / 0.10).clamp(0.0, 2.0)
    ratio = (frequency_hz / source_f0_hz.clamp_min(1.0) / 3.0).clamp(0.0, 2.0)
    log_frequency_scale = (
        torch.log10(frequency_scale.clamp_min(1.0e-8)).clamp(-8.0, 2.0) / 4.0
    )
    scalars = torch.stack(
        (frequency, f0, t0, ratio, mean_velocity, log_frequency_scale), dim=1
    )
    scalar_maps = scalars[:, :, None, None].expand(-1, -1, height, width)
    features = torch.cat(
        (base, static_norm, raw_static, scale_maps, scalar_maps, coordinate), dim=1
    )
    if features.shape[1] != INPUT_CHANNELS:
        raise RuntimeError(f"unexpected R40 input channel count: {features.shape}")
    return features


class LearnedComplexSpectralConv2d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = int(width)
        self.modes = int(modes)
        shape = (self.width, self.width, self.modes, self.modes, 2)
        self.weight_top = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.weight_bottom = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        scale = 1.0 / math.sqrt(self.width * self.width)
        nn.init.uniform_(self.weight_top, -scale, scale)
        nn.init.uniform_(self.weight_bottom, -scale, scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        modes_y = min(self.modes, height // 2)
        modes_x = min(self.modes, width // 2 + 1)
        with torch.autocast(device_type=value.device.type, enabled=False):
            spectrum = torch.fft.rfft2(value.float(), norm="ortho")
            output = torch.zeros(
                value.shape[0],
                self.width,
                height,
                width // 2 + 1,
                dtype=spectrum.dtype,
                device=value.device,
            )
            top = torch.view_as_complex(self.weight_top.contiguous())[
                :, :, :modes_y, :modes_x
            ]
            bottom = torch.view_as_complex(self.weight_bottom.contiguous())[
                :, :, :modes_y, :modes_x
            ]
            output[:, :, :modes_y, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, :modes_y, :modes_x], top
            )
            output[:, :, -modes_y:, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, -modes_y:, :modes_x], bottom
            )
            return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class FrequencyFNOBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = LearnedComplexSpectralConv2d(width, modes)
        self.local = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1),
        )
        groups = 8 if width % 8 == 0 else 4
        self.norm = nn.GroupNorm(groups, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.spectral(value) + self.local(value)
        return value + F.gelu(self.norm(update))


class FrequencyResidualFNO(nn.Module):
    def __init__(
        self,
        *,
        width: int = 48,
        modes: int = 16,
        blocks: int = 4,
        correction_cap: float = 1.5,
    ):
        super().__init__()
        self.width = int(width)
        self.modes = int(modes)
        self.blocks_count = int(blocks)
        self.correction_cap = float(correction_cap)
        self.stem = nn.Conv2d(INPUT_CHANNELS, self.width, 1)
        self.blocks = nn.Sequential(
            *[
                FrequencyFNOBlock(self.width, self.modes)
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


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def frequency_objective(
    prediction: torch.Tensor,
    residual: torch.Tensor,
    *,
    frequency_scale: torch.Tensor,
    target_square_total: torch.Tensor,
    frequency_weight: torch.Tensor,
    family_weight: torch.Tensor,
    frequency_count: int,
    tail_weight: float,
    hinge_weight: float,
    shape_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    error_square_norm = (prediction.float() - residual.float()).square().sum(
        dim=(1, 2, 3)
    )
    parent_square_norm = residual.float().square().sum(dim=(1, 2, 3))
    physical_scale = frequency_weight.float() * frequency_scale.float().square()
    denominator = target_square_total.float().clamp_min(1.0e-12)
    candidate = (
        float(frequency_count) * physical_scale * error_square_norm / denominator
    )
    parent = (
        float(frequency_count) * physical_scale * parent_square_norm / denominator
    )
    weighted_mean = (candidate * family_weight.float()).sum() / family_weight.float().sum().clamp_min(1.0e-8)
    tail_count = max(1, int(math.ceil(0.25 * candidate.numel())))
    tail = torch.topk(candidate, k=tail_count).values.mean()
    hinge = F.relu(
        torch.sqrt(candidate.clamp_min(1.0e-14))
        - torch.sqrt(parent.detach().clamp_min(1.0e-14))
    ).square().mean()
    shape = F.smooth_l1_loss(prediction.float(), residual.float(), beta=0.02)
    loss = (
        weighted_mean
        + float(tail_weight) * tail
        + float(hinge_weight) * hinge
        + float(shape_weight) * shape
    )
    return loss, {
        "physical_mean": weighted_mean.detach(),
        "physical_tail": tail.detach(),
        "hinge": hinge.detach(),
        "shape": shape.detach(),
        "parent_physical_mean": parent.mean().detach(),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    candidate = np.asarray([float(row["candidate_rel_l2"]) for row in rows])
    parent = np.asarray([float(row["parent_rel_l2"]) for row in rows])
    return {
        "count": len(rows),
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
def evaluate(
    model: nn.Module,
    collection: FrequencyCacheCollection,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    if collection.expected_subset != "holdout":
        raise ValueError("evaluation requires holdout caches")
    model.eval()
    rows: list[dict[str, Any]] = []
    family_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILIES}
    indices = collection.frequency_indices
    fft_weight = rfft_weights(TIME_COUNT)[indices]
    retained = int(collection.retained)

    for file_index, local_index in collection.records:
        handle = collection.handles[file_index]
        sample_id = str(handle["sample_id"].asstr()[local_index])
        group_id = str(handle["group_id"].asstr()[local_index])
        family = str(handle["family"].asstr()[local_index])
        base_norm = np.asarray(handle["base_dct_norm"][local_index], dtype=np.float32)
        static_norm = torch.from_numpy(
            np.asarray(handle["static_dct_norm"][local_index], dtype=np.float32)
        ).to(device)
        static_scale = torch.from_numpy(
            np.asarray(handle["static_dct_scale"][local_index], dtype=np.float32)
        ).to(device)
        frequencies = np.asarray(handle["frequency_hz"], dtype=np.float32)
        frequency_scale = np.asarray(
            handle["frequency_scale"][local_index], dtype=np.float32
        )
        f0 = float(handle["source_f0_hz"][local_index])
        t0 = float(handle["source_t0_s"][local_index])
        predictions: list[np.ndarray] = []
        for start in range(0, len(frequencies), int(batch_size)):
            stop = min(start + int(batch_size), len(frequencies))
            block = stop - start
            base = torch.from_numpy(base_norm[start:stop]).to(device)
            static = static_norm[None].expand(block, -1, -1, -1)
            scales = static_scale[None].expand(block, -1)
            frequency = torch.from_numpy(frequencies[start:stop]).to(device)
            frequency_scale_block = torch.from_numpy(
                frequency_scale[start:stop]
            ).to(device)
            f0_tensor = torch.full((block,), f0, device=device)
            t0_tensor = torch.full((block,), t0, device=device)
            features = make_features(
                base,
                static,
                scales,
                frequency_hz=frequency,
                frequency_scale=frequency_scale_block,
                source_f0_hz=f0_tensor,
                source_t0_s=t0_tensor,
            )
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp
                else nullcontext()
            )
            with context:
                prediction = model(features)
            predictions.append(prediction.float().cpu().numpy())
        correction_norm = np.concatenate(predictions, axis=0)
        correction_coeff = channels_to_complex(correction_norm)
        correction_coeff *= frequency_scale[:, None, None]

        base_selected = channels_to_complex(
            np.asarray(handle["base_spectrum_selected"][local_index], dtype=np.float32)
        )
        truth_selected = channels_to_complex(
            np.asarray(handle["truth_spectrum_selected"][local_index], dtype=np.float32)
        )
        grid = int(base_selected.shape[-1])
        padded = np.zeros((len(indices), grid, grid), dtype=np.complex64)
        padded[:, :retained, :retained] = correction_coeff
        correction_spatial = idctn(
            padded.real, type=2, norm="ortho", axes=(-2, -1)
        ) + 1j * idctn(
            padded.imag, type=2, norm="ortho", axes=(-2, -1)
        )
        selected_candidate_square = float(
            np.sum(
                fft_weight
                * np.sum(
                    np.abs(base_selected + correction_spatial - truth_selected) ** 2,
                    axis=(1, 2),
                    dtype=np.float64,
                )
            )
        )
        selected_parent_square = float(
            np.sum(
                fft_weight
                * np.sum(
                    np.abs(base_selected - truth_selected) ** 2,
                    axis=(1, 2),
                    dtype=np.float64,
                )
            )
        )
        unselected = float(handle["base_error_square_unselected"][local_index])
        target_square = float(handle["target_square_total"][local_index])
        candidate_square = unselected + selected_candidate_square
        parent_square = unselected + selected_parent_square
        candidate_rel = math.sqrt(candidate_square / max(target_square, 1.0e-30))
        parent_rel = math.sqrt(parent_square / max(target_square, 1.0e-30))
        row = {
            "sample_id": sample_id,
            "group_id": group_id,
            "family": family,
            "candidate_rel_l2": candidate_rel,
            "parent_rel_l2": parent_rel,
            "relative_improvement": 1.0
            - candidate_rel / max(parent_rel, 1.0e-30),
            "candidate_error_square": candidate_square,
            "parent_error_square": parent_square,
            "target_square": target_square,
            "unselected_error_square": unselected,
        }
        rows.append(row)
        family_rows[family].append(row)

    aggregate = summarize_rows(rows)
    passed = bool(
        float(aggregate["candidate_mean"]) <= 0.05
        and float(aggregate["candidate_max"]) <= 0.05
    )
    return {
        "aggregate": aggregate,
        "per_family": {
            family: summarize_rows(values) for family, values in family_rows.items()
        },
        "absolute_goal": {
            "mean_lte_0p05": float(aggregate["candidate_mean"]) <= 0.05,
            "max_lte_0p05": float(aggregate["candidate_max"]) <= 0.05,
            "passed": passed,
        },
        "records": rows,
    }


def reduce_scalar(value: torch.Tensor, *, distributed: bool) -> float:
    result = value.detach().double()
    if distributed:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        result /= torch.distributed.get_world_size()
    return float(result)


def checkpoint_payload(
    *,
    model: FrequencyResidualFNO,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    selection_sha256: str,
    metrics: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "selection_sha256": selection_sha256,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": {
            "input_channels": INPUT_CHANNELS,
            "width": int(args.width),
            "modes": int(args.modes),
            "blocks": int(args.blocks),
            "correction_cap": float(args.correction_cap),
        },
        "holdout_metrics": dict(metrics),
        "data_access": {
            "fit": "train_only",
            "holdout": "train_group_disjoint_opened_development",
            "r29b_opened": False,
            "final_validation_opened": False,
            "test_id_opened": False,
        },
    }


def save_checkpoint(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--correction-cap", type=float, default=1.5)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--hinge-weight", type=float, default=2.0)
    parser.add_argument("--shape-weight", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=400828)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--stop-on-pass", action="store_true")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("R40 training requires CUDA")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if distributed:
        torch.distributed.init_process_group(backend="nccl")

    seed = int(args.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    fit: FrequencyCacheCollection | None = None
    holdout: FrequencyCacheCollection | None = None
    try:
        fit = FrequencyCacheCollection(args.fit_cache, expected_subset="fit")
        holdout = FrequencyCacheCollection(
            args.holdout_cache, expected_subset="holdout"
        )
        if fit.selection_sha256 != holdout.selection_sha256:
            raise RuntimeError("fit/holdout selection digest mismatch")
        if set(fit.group_ids) & set(holdout.group_ids):
            raise RuntimeError("fit/holdout cache group leakage")
        if fit.frequency_count != holdout.frequency_count:
            raise RuntimeError("fit/holdout frequency counts disagree")
        for handle in holdout.handles:
            if "base_spectrum_selected" not in handle or "truth_spectrum_selected" not in handle:
                raise RuntimeError("holdout cache lacks full selected spectra")

        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            torch.distributed.barrier()

        dataset = FitFrequencyDataset(fit)
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(args.seed),
                drop_last=True,
            )
            if distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        model_for_save = FrequencyResidualFNO(
            width=int(args.width),
            modes=int(args.modes),
            blocks=int(args.blocks),
            correction_cap=float(args.correction_cap),
        ).to(device)
        model: nn.Module = model_for_save
        if distributed:
            model = DistributedDataParallel(
                model_for_save,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        warmup_epochs = min(
            max(int(args.warmup_epochs), 0), max(int(args.epochs) - 1, 0)
        )
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(args.epochs) - warmup_epochs, 1),
                eta_min=float(args.learning_rate) * 0.05,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(args.epochs), 1),
                eta_min=float(args.learning_rate) * 0.05,
            )

        if rank == 0:
            identity = {
                "schema": "r40_frequency_residual_run_identity_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "selection_sha256": fit.selection_sha256,
                "fit_cache_evidence": fit.cache_evidence,
                "holdout_cache_evidence": holdout.cache_evidence,
                "fit_record_count": len(fit.records),
                "holdout_record_count": len(holdout.records),
                "fit_group_count": len(set(fit.group_ids)),
                "holdout_group_count": len(set(holdout.group_ids)),
                "frequency_count": fit.frequency_count,
                "frequency_min_hz": float(fit.frequency_hz.min()),
                "frequency_max_hz": float(fit.frequency_hz.max()),
                "dct_retained": int(fit.retained),
                "input_channels": INPUT_CHANNELS,
                "parameter_count": parameter_count(model_for_save),
                "model": {
                    "name": "frequency_conditioned_spatial_dct_fno",
                    "width": int(args.width),
                    "modes": int(args.modes),
                    "blocks": int(args.blocks),
                    "correction_cap": float(args.correction_cap),
                    "output_channels": "complex_residual_real_imaginary",
                    "zero_initialized_output": True,
                    "medium_dct_scales_preserved": True,
                    "raw_medium_dct_reconstructed": True,
                    "mean_velocity_conditioning": True,
                },
                "optimization": {
                    "epochs": int(args.epochs),
                    "max_steps_per_epoch": int(args.max_steps_per_epoch),
                    "local_batch_size": int(args.batch_size),
                    "global_batch_size": int(args.batch_size) * world_size,
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "optimizer": "adam",
                    "warmup_epochs": warmup_epochs,
                    "tail_weight": float(args.tail_weight),
                    "hinge_weight": float(args.hinge_weight),
                    "shape_weight": float(args.shape_weight),
                    "amp_bfloat16": bool(args.amp),
                    "seed": int(args.seed),
                },
                "data_access": {
                    "fit": "train_only",
                    "holdout": "train_group_disjoint_opened_development",
                    "truth_deployment_input": False,
                    "r29b_opened": False,
                    "final_validation_opened": False,
                    "test_id_opened": False,
                },
                "absolute_goal": {
                    "record_rel_l2_mean_lte": 0.05,
                    "record_rel_l2_max_lte": 0.05,
                },
            }
            atomic_json(identity, output_dir / "run_identity.json")
        if distributed:
            torch.distributed.barrier()

        updates_path = output_dir / "updates.jsonl"
        metrics_path = output_dir / "holdout_metrics.jsonl"
        started = time.perf_counter()
        best_score = math.inf
        best_epoch = -1
        best_metrics: dict[str, Any] | None = None
        initial_passed = False

        if rank == 0:
            initial_metrics = evaluate(
                model_for_save,
                holdout,
                device=device,
                batch_size=int(args.eval_batch_size),
                amp=bool(args.amp),
            )
            initial_metrics.update(
                {
                    "event": "initial_identity_evaluation",
                    "epoch": 0,
                    "global_step": 0,
                    "train_loss": None,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            maximum_identity_difference = max(
                abs(float(row["candidate_rel_l2"]) - float(row["parent_rel_l2"]))
                for row in initial_metrics["records"]
            )
            if maximum_identity_difference > 1.0e-8:
                raise RuntimeError(
                    "zero-initialized R40 model is not an exact base-model identity: "
                    f"max metric difference={maximum_identity_difference}"
                )
            initial_aggregate = initial_metrics["aggregate"]
            best_score = float(initial_aggregate["candidate_max"]) + float(
                initial_aggregate["candidate_mean"]
            )
            best_epoch = 0
            best_metrics = initial_metrics
            initial_passed = bool(initial_metrics["absolute_goal"]["passed"])
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(initial_metrics, sort_keys=True) + "\n")
            atomic_json(initial_metrics, output_dir / "initial_holdout.json")
            save_checkpoint(
                checkpoint_payload(
                    model=model_for_save,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=0,
                    global_step=0,
                    selection_sha256=str(fit.selection_sha256),
                    metrics=initial_metrics,
                    args=args,
                ),
                output_dir / "best.pt",
            )
            atomic_json(
                {"epoch": best_epoch, "score": best_score, "metrics": best_metrics},
                output_dir / "best.json",
            )
            print(
                json.dumps(
                    {
                        "event": "initial_identity_evaluation",
                        "candidate_mean": initial_aggregate["candidate_mean"],
                        "candidate_max": initial_aggregate["candidate_max"],
                        "maximum_identity_metric_difference": maximum_identity_difference,
                        "absolute_goal_passed": initial_passed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if distributed:
            initial_tensor = torch.tensor(
                1 if initial_passed else 0, device=device, dtype=torch.int32
            )
            torch.distributed.broadcast(initial_tensor, src=0)
            initial_passed = bool(int(initial_tensor.item()))
            torch.distributed.barrier()

        global_step = 0
        stopped_on_pass = bool(initial_passed and args.stop_on_pass)
        for epoch in range(1, int(args.epochs) + 1):
            if stopped_on_pass:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            epoch_total = torch.zeros((), device=device, dtype=torch.float64)
            epoch_batches = 0
            for batch_index, batch in enumerate(loader):
                if (
                    int(args.max_steps_per_epoch) > 0
                    and batch_index >= int(args.max_steps_per_epoch)
                ):
                    break
                (
                    base,
                    residual,
                    static_norm,
                    static_scale,
                    frequency_hz,
                    source_f0_hz,
                    source_t0_s,
                    frequency_scale,
                    target_square_total,
                    frequency_weight,
                    family_weight,
                ) = batch
                base = base.to(device, non_blocking=True)
                residual = residual.to(device, non_blocking=True)
                static_norm = static_norm.to(device, non_blocking=True)
                static_scale = static_scale.to(device, non_blocking=True)
                frequency_hz = frequency_hz.to(device, non_blocking=True)
                source_f0_hz = source_f0_hz.to(device, non_blocking=True)
                source_t0_s = source_t0_s.to(device, non_blocking=True)
                frequency_scale = frequency_scale.to(device, non_blocking=True)
                target_square_total = target_square_total.to(device, non_blocking=True)
                frequency_weight = frequency_weight.to(device, non_blocking=True)
                family_weight = family_weight.to(device, non_blocking=True)
                features = make_features(
                    base,
                    static_norm,
                    static_scale,
                    frequency_hz=frequency_hz,
                    frequency_scale=frequency_scale,
                    source_f0_hz=source_f0_hz,
                    source_t0_s=source_t0_s,
                )
                optimizer.zero_grad(set_to_none=True)
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.amp
                    else nullcontext()
                )
                with context:
                    prediction = model(features)
                    loss, components = frequency_objective(
                        prediction.float(),
                        residual,
                        frequency_scale=frequency_scale,
                        target_square_total=target_square_total,
                        frequency_weight=frequency_weight,
                        family_weight=family_weight,
                        frequency_count=fit.frequency_count,
                        tail_weight=float(args.tail_weight),
                        hinge_weight=float(args.hinge_weight),
                        shape_weight=float(args.shape_weight),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"nonfinite R40 loss at global step {global_step}"
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                epoch_total += loss.detach().double()
                epoch_batches += 1
                if rank == 0 and (global_step == 1 or global_step % 100 == 0):
                    event = {
                        "event": "update",
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": float(loss.detach()),
                        "physical_mean": float(components["physical_mean"]),
                        "physical_tail": float(components["physical_tail"]),
                        "hinge": float(components["hinge"]),
                        "shape": float(components["shape"]),
                        "parent_physical_mean": float(
                            components["parent_physical_mean"]
                        ),
                        "gradient_norm": float(gradient_norm),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                    with updates_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)

            if epoch_batches == 0:
                raise RuntimeError("R40 epoch processed zero batches")
            scheduler.step()
            mean_loss = epoch_total / epoch_batches
            mean_loss_value = reduce_scalar(mean_loss, distributed=distributed)
            should_evaluate = (
                epoch == 1
                or epoch % int(args.eval_every) == 0
                or epoch == int(args.epochs)
            )
            if distributed:
                torch.distributed.barrier()
            passed = False
            if rank == 0 and should_evaluate:
                metrics = evaluate(
                    model_for_save,
                    holdout,
                    device=device,
                    batch_size=int(args.eval_batch_size),
                    amp=bool(args.amp),
                )
                metrics.update(
                    {
                        "event": "holdout_evaluation",
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": mean_loss_value,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                aggregate = metrics["aggregate"]
                score = float(aggregate["candidate_max"]) + float(
                    aggregate["candidate_mean"]
                )
                passed = bool(metrics["absolute_goal"]["passed"])
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                atomic_json(metrics, output_dir / "latest_holdout.json")
                print(
                    json.dumps(
                        {
                            "event": "holdout_evaluation",
                            "epoch": epoch,
                            "candidate_mean": aggregate["candidate_mean"],
                            "candidate_max": aggregate["candidate_max"],
                            "parent_mean": aggregate["parent_mean"],
                            "parent_max": aggregate["parent_max"],
                            "absolute_goal_passed": passed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    best_metrics = metrics
                    save_checkpoint(
                        checkpoint_payload(
                            model=model_for_save,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            selection_sha256=str(fit.selection_sha256),
                            metrics=metrics,
                            args=args,
                        ),
                        output_dir / "best.pt",
                    )
                    atomic_json(
                        {
                            "epoch": best_epoch,
                            "score": best_score,
                            "metrics": best_metrics,
                        },
                        output_dir / "best.json",
                    )
            if distributed:
                passed_tensor = torch.tensor(
                    1 if passed else 0, device=device, dtype=torch.int32
                )
                torch.distributed.broadcast(passed_tensor, src=0)
                passed = bool(int(passed_tensor.item()))
                torch.distributed.barrier()
            if passed and args.stop_on_pass:
                stopped_on_pass = True

        if rank == 0:
            if best_metrics is None:
                raise RuntimeError("R40 produced no holdout metrics")
            checkpoint_path = output_dir / "best.pt"
            final_summary = {
                "schema": "r40_frequency_residual_run_summary_v1",
                "status": "complete",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "selection_sha256": fit.selection_sha256,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_checkpoint": str(checkpoint_path),
                "best_checkpoint_sha256": sha256_file(checkpoint_path),
                "best_metrics": best_metrics,
                "stopped_on_pass": stopped_on_pass,
                "r29b_opened": False,
                "final_validation_opened": False,
                "test_id_opened": False,
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(final_summary, output_dir / "run_summary.json")
            print(
                json.dumps(
                    {
                        "event": "run_complete",
                        "best_epoch": best_epoch,
                        "candidate_mean": best_metrics["aggregate"][
                            "candidate_mean"
                        ],
                        "candidate_max": best_metrics["aggregate"]["candidate_max"],
                        "absolute_goal_passed": best_metrics["absolute_goal"][
                            "passed"
                        ],
                        "checkpoint_sha256": final_summary[
                            "best_checkpoint_sha256"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if distributed:
            torch.distributed.barrier()
        return 0
    finally:
        if fit is not None:
            fit.close()
        if holdout is not None:
            holdout.close()
        if distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
