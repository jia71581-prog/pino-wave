#!/usr/bin/env python3
"""Train and evaluate the R25 coarse-field residual neural operator.

Only train-split caches produced by ``build_r25_coarse_residual_cache.py`` are
accepted.  The model predicts a bounded correction to a live coarse LWC-84
field from deployment-available medium/source/query features.  Cached truth is
used solely on the loss/metric side.
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
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


SCHEMA_CACHE = "r25_coarse_residual_cache_v1"
FAMILIES = ("uniform", "layered", "marmousi")
INPUT_CHANNELS = 13


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


class CacheCollection:
    def __init__(self, paths: Sequence[Path], *, expected_subset: str):
        if not paths:
            raise ValueError(f"no {expected_subset} cache paths supplied")
        self.paths = tuple(path.expanduser().resolve() for path in paths)
        self.handles: list[h5py.File] = []
        self.records: list[tuple[int, int]] = []
        self.selection_sha256: str | None = None
        self.time_count: int | None = None
        self.cache_hashes: dict[str, str] = {}
        self.expected_subset = str(expected_subset)
        for file_index, path in enumerate(self.paths):
            if not path.is_file():
                self.close()
                raise FileNotFoundError(path)
            handle = h5py.File(path, "r", swmr=True)
            self.handles.append(handle)
            if str(handle.attrs.get("schema", "")) != SCHEMA_CACHE:
                self.close()
                raise RuntimeError(f"unexpected cache schema: {path}")
            if str(handle.attrs.get("status", "")) != "complete":
                self.close()
                raise RuntimeError(f"cache is not complete: {path}")
            if str(handle.attrs.get("subset", "")) != self.expected_subset:
                self.close()
                raise RuntimeError(f"cache subset mismatch: {path}")
            truth_policy = str(handle.attrs.get("truth_policy", ""))
            if truth_policy != "train_only_supervision_not_deployment_input":
                self.close()
                raise RuntimeError(f"truth policy mismatch: {path}")
            selection = str(handle.attrs.get("selection_sha256", ""))
            if self.selection_sha256 is None:
                self.selection_sha256 = selection
            elif selection != self.selection_sha256:
                self.close()
                raise RuntimeError("cache selection digests disagree")
            frames = int(handle["coarse_norm"].shape[1])
            if self.time_count is None:
                self.time_count = frames
            elif frames != self.time_count:
                self.close()
                raise RuntimeError("cache time counts disagree")
            if handle["coarse_norm"].shape != handle["truth_norm"].shape:
                self.close()
                raise RuntimeError("coarse/truth cache shapes disagree")
            count = int(handle["coarse_norm"].shape[0])
            self.records.extend((file_index, local) for local in range(count))
            self.cache_hashes[str(path)] = sha256_file(path)
        if self.selection_sha256 is None or self.time_count is None:
            self.close()
            raise RuntimeError("empty cache collection")
        sample_ids: list[str] = []
        group_ids: list[str] = []
        for handle in self.handles:
            sample_ids.extend(text_values(handle["sample_id"]))
            group_ids.extend(text_values(handle["group_id"]))
        if len(sample_ids) != len(set(sample_ids)):
            self.close()
            raise RuntimeError("cache collection has duplicate sample IDs")
        self.sample_ids = tuple(sample_ids)
        self.group_ids = tuple(group_ids)

    def close(self) -> None:
        for handle in getattr(self, "handles", []):
            try:
                handle.close()
            except Exception:
                pass
        self.handles = []

    def __del__(self):
        self.close()


def make_dynamic_features(
    coarse: torch.Tensor,
    static: torch.Tensor,
    *,
    time_s: torch.Tensor,
    source_f0_hz: torch.Tensor,
    source_t0_s: torch.Tensor,
) -> torch.Tensor:
    """Build deployment-only features for [B,H,W] coarse frames."""

    if coarse.ndim != 3 or static.ndim != 4 or static.shape[1] != 7:
        raise ValueError("unexpected coarse/static feature shape")
    batch, height, width = coarse.shape
    travel = static[:, 4]
    time_map = time_s[:, None, None].expand(batch, height, width)
    tau = time_map - source_t0_s[:, None, None] - travel
    phase = 2.0 * math.pi * source_f0_hz[:, None, None] * tau
    ricker_arg = math.pi * source_f0_hz[:, None, None] * tau
    ricker_square = ricker_arg.square().clamp_max(60.0)
    ricker = (1.0 - 2.0 * ricker_square) * torch.exp(-ricker_square)
    dynamic = torch.stack(
        [
            (2.0 * time_map - 1.0).clamp(-1.5, 1.5),
            (tau / 0.20).clamp(-3.0, 3.0),
            torch.sin(phase),
            torch.cos(phase),
            ricker,
        ],
        dim=1,
    )
    return torch.cat([coarse[:, None], static, dynamic], dim=1)


class FitFrameDataset(Dataset):
    def __init__(self, collection: CacheCollection):
        if collection.expected_subset != "fit":
            raise ValueError("FitFrameDataset requires fit caches")
        self.collection = collection
        self.time_count = int(collection.time_count)

    def __len__(self) -> int:
        return len(self.collection.records) * self.time_count

    def __getitem__(self, index: int):
        record_position, frame_position = divmod(int(index), self.time_count)
        file_index, local_index = self.collection.records[record_position]
        handle = self.collection.handles[file_index]
        coarse = torch.from_numpy(
            np.asarray(handle["coarse_norm"][local_index, frame_position], dtype=np.float32)
        )
        truth = torch.from_numpy(
            np.asarray(handle["truth_norm"][local_index, frame_position], dtype=np.float32)
        )
        static = torch.from_numpy(
            np.asarray(handle["static_features"][local_index], dtype=np.float32)
        )
        time_s = torch.tensor(
            float(handle["time_s"][frame_position]), dtype=torch.float32
        )
        f0 = torch.tensor(
            float(handle["source_f0_hz"][local_index]), dtype=torch.float32
        )
        t0 = torch.tensor(
            float(handle["source_t0_s"][local_index]), dtype=torch.float32
        )
        mean_energy = torch.tensor(
            float(handle["truth_frame_energy_mean_norm"][local_index]),
            dtype=torch.float32,
        )
        active = torch.tensor(float(time_s >= t0), dtype=torch.float32)
        features = make_dynamic_features(
            coarse[None],
            static[None],
            time_s=time_s[None],
            source_f0_hz=f0[None],
            source_t0_s=t0[None],
        )[0]
        return features, coarse, truth, mean_energy, active


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 4
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(value)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1):
        super().__init__()
        self.first = ConvNormAct(channels, channels, dilation=dilation)
        groups = 8 if channels % 8 == 0 else 4
        self.second = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.second(self.first(value)), inplace=True)


class CoarseResidualUNet(nn.Module):
    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.5):
        super().__init__()
        width = int(base_width)
        if width < 16 or width % 8:
            raise ValueError("base_width must be a multiple of 8 and at least 16")
        self.correction_cap = float(correction_cap)
        self.stem = ConvNormAct(INPUT_CHANNELS, width)
        self.enc0 = ResidualBlock(width)
        self.down1 = ConvNormAct(width, width * 2, stride=2)
        self.enc1 = ResidualBlock(width * 2)
        self.down2 = ConvNormAct(width * 2, width * 3, stride=2)
        self.enc2 = ResidualBlock(width * 3, dilation=2)
        self.down3 = ConvNormAct(width * 3, width * 4, stride=2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(width * 4, dilation=2),
            ResidualBlock(width * 4, dilation=3),
        )
        self.up2 = ConvNormAct(width * 4 + width * 3, width * 3)
        self.dec2 = ResidualBlock(width * 3)
        self.up1 = ConvNormAct(width * 3 + width * 2, width * 2)
        self.dec1 = ResidualBlock(width * 2)
        self.up0 = ConvNormAct(width * 2 + width, width)
        self.dec0 = ResidualBlock(width)
        self.output = nn.Conv2d(width, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, features: torch.Tensor, *, active: torch.Tensor | None = None
    ) -> torch.Tensor:
        x0 = self.enc0(self.stem(features))
        x1 = self.enc1(self.down1(x0))
        x2 = self.enc2(self.down2(x1))
        x3 = self.bottleneck(self.down3(x2))
        y2 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.dec2(self.up2(torch.cat([y2, x2], dim=1)))
        y1 = F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(self.up1(torch.cat([y1, x1], dim=1)))
        y0 = F.interpolate(y1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y0 = self.dec0(self.up0(torch.cat([y0, x0], dim=1)))
        correction = self.correction_cap * torch.tanh(self.output(y0)[:, 0])
        if active is not None:
            correction = correction * active[:, None, None]
        correction = correction.clone()
        correction[:, 0, :] = 0.0
        return correction


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def spatial_gradient_loss(
    prediction: torch.Tensor, target: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    pred_x = prediction[:, :, 1:] - prediction[:, :, :-1]
    truth_x = target[:, :, 1:] - target[:, :, :-1]
    pred_z = prediction[:, 1:, :] - prediction[:, :-1, :]
    truth_z = target[:, 1:, :] - target[:, :-1, :]
    error = (pred_x - truth_x).square().mean(dim=(1, 2))
    error = error + (pred_z - truth_z).square().mean(dim=(1, 2))
    return (error / denominator).mean()


def train_loss(
    correction: torch.Tensor,
    coarse: torch.Tensor,
    truth: torch.Tensor,
    mean_energy: torch.Tensor,
    *,
    hinge_weight: float,
    gradient_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = coarse + correction
    denominator = mean_energy.clamp_min(1.0e-7)
    candidate_mse = (prediction - truth).square().mean(dim=(1, 2))
    parent_mse = (coarse - truth).square().mean(dim=(1, 2))
    candidate_relative_square = candidate_mse / denominator
    parent_relative_square = parent_mse / denominator
    hinge = F.relu(
        torch.sqrt(candidate_relative_square.clamp_min(1.0e-12))
        - torch.sqrt(parent_relative_square.clamp_min(1.0e-12))
    ).square()
    gradient = spatial_gradient_loss(prediction, truth, denominator)
    correction_energy = correction.square().mean()
    total = (
        candidate_relative_square.mean()
        + float(hinge_weight) * hinge.mean()
        + float(gradient_weight) * gradient
        + 1.0e-5 * correction_energy
    )
    return total, {
        "relative_square": candidate_relative_square.mean().detach(),
        "parent_relative_square": parent_relative_square.mean().detach(),
        "hinge": hinge.mean().detach(),
        "gradient": gradient.detach(),
        "correction_energy": correction_energy.detach(),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    collection: CacheCollection,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    family_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILIES}
    for file_index, local_index in collection.records:
        handle = collection.handles[file_index]
        sample_id = str(handle["sample_id"].asstr()[local_index])
        family = str(handle["family"].asstr()[local_index])
        static = torch.from_numpy(
            np.asarray(handle["static_features"][local_index], dtype=np.float32)
        ).to(device)
        f0 = float(handle["source_f0_hz"][local_index])
        t0 = float(handle["source_t0_s"][local_index])
        time_s_all = np.asarray(handle["time_s"][:], dtype=np.float32)
        coarse_all = handle["coarse_norm"][local_index]
        truth_all = handle["truth_norm"][local_index]
        candidate_error = 0.0
        parent_error = 0.0
        target_square = 0.0
        correction_square = 0.0
        for start in range(0, len(time_s_all), int(batch_size)):
            stop = min(start + int(batch_size), len(time_s_all))
            coarse = torch.from_numpy(
                np.asarray(coarse_all[start:stop], dtype=np.float32)
            ).to(device)
            truth = torch.from_numpy(
                np.asarray(truth_all[start:stop], dtype=np.float32)
            ).to(device)
            block = stop - start
            static_block = static[None].expand(block, -1, -1, -1)
            time_s = torch.from_numpy(time_s_all[start:stop]).to(device)
            f0_tensor = torch.full((block,), f0, device=device)
            t0_tensor = torch.full((block,), t0, device=device)
            features = make_dynamic_features(
                coarse,
                static_block,
                time_s=time_s,
                source_f0_hz=f0_tensor,
                source_t0_s=t0_tensor,
            )
            active = (time_s >= t0_tensor).to(dtype=torch.float32)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp
                else nullcontext()
            )
            with context:
                correction = model(features, active=active)
            prediction = coarse + correction.float()
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

    def summarize(values: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        if not values:
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
        candidate = [float(row["candidate_rel_l2"]) for row in values]
        parent = [float(row["parent_rel_l2"]) for row in values]
        return {
            "count": len(values),
            "candidate_mean": float(np.mean(candidate)),
            "candidate_max": float(np.max(candidate)),
            "candidate_median": float(np.median(candidate)),
            "parent_mean": float(np.mean(parent)),
            "parent_max": float(np.max(parent)),
            "mean_relative_improvement": float(
                1.0 - np.mean(candidate) / max(float(np.mean(parent)), 1.0e-30)
            ),
            "max_relative_improvement": float(
                1.0 - np.max(candidate) / max(float(np.max(parent)), 1.0e-30)
            ),
        }

    aggregate = summarize(rows)
    return {
        "aggregate": aggregate,
        "per_family": {
            family: summarize(values) for family, values in family_rows.items()
        },
        "absolute_goal": {
            "mean_lte_0p05": aggregate["candidate_mean"] <= 0.05,
            "max_lte_0p05": aggregate["candidate_max"] <= 0.05,
            "passed": aggregate["candidate_mean"] <= 0.05
            and aggregate["candidate_max"] <= 0.05,
        },
        "records": rows,
    }


def reduce_scalar(value: torch.Tensor, *, distributed: bool) -> float:
    result = value.detach().double()
    if distributed:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        result /= torch.distributed.get_world_size()
    return float(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--correction-cap", type=float, default=0.5)
    parser.add_argument("--hinge-weight", type=float, default=0.5)
    parser.add_argument("--gradient-weight", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=250827)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("R25 training requires CUDA")
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

    fit = CacheCollection(args.fit_cache, expected_subset="fit")
    holdout = CacheCollection(args.holdout_cache, expected_subset="holdout")
    if fit.selection_sha256 != holdout.selection_sha256:
        raise RuntimeError("fit/holdout selection digest mismatch")
    if set(fit.group_ids) & set(holdout.group_ids):
        raise RuntimeError("fit/holdout cache group leakage")
    if int(holdout.time_count) != 401:
        raise RuntimeError("holdout cache must contain all 401 saved frames")

    dataset = FitFrameDataset(fit)
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
    model = CoarseResidualUNet(
        base_width=int(args.base_width), correction_cap=float(args.correction_cap)
    ).to(device)
    model_for_save = model
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        model_for_save = model.module
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(args.epochs), 1), eta_min=float(args.learning_rate) * 0.05
    )

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "schema": "r25_coarse_residual_run_identity_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": fit.selection_sha256,
            "fit_cache_sha256": fit.cache_hashes,
            "holdout_cache_sha256": holdout.cache_hashes,
            "fit_record_count": len(fit.records),
            "fit_frames_per_record": fit.time_count,
            "holdout_record_count": len(holdout.records),
            "holdout_frames_per_record": holdout.time_count,
            "input_channels": INPUT_CHANNELS,
            "parameter_count": parameter_count(model_for_save),
            "model": {
                "name": "coarse_residual_unet",
                "base_width": int(args.base_width),
                "correction_cap": float(args.correction_cap),
                "top_row_correction_zero": True,
                "pre_onset_correction_zero": True,
            },
            "optimization": {
                "epochs": int(args.epochs),
                "global_batch_size": int(args.batch_size) * world_size,
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "hinge_weight": float(args.hinge_weight),
                "gradient_weight": float(args.gradient_weight),
                "amp_bfloat16": bool(args.amp),
                "seed": int(args.seed),
            },
            "data_access": {
                "fit": "train",
                "holdout": "train_group_disjoint",
                "validation_opened": False,
                "test_id_opened": False,
                "truth_deployment_input": False,
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
                "zero-initialized R25 model is not an exact coarse-field identity: "
                f"max metric difference={maximum_identity_difference}"
            )
        initial_aggregate = initial_metrics["aggregate"]
        best_score = float(initial_aggregate["candidate_max"]) + float(
            initial_aggregate["candidate_mean"]
        )
        best_epoch = 0
        best_metrics = initial_metrics
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(initial_metrics, sort_keys=True) + "\n")
        atomic_json(initial_metrics, output_dir / "initial_holdout.json")
        initial_checkpoint = {
            "schema": "r25_coarse_residual_checkpoint_v1",
            "epoch": 0,
            "global_step": 0,
            "selection_sha256": fit.selection_sha256,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model_for_save.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": {
                "base_width": int(args.base_width),
                "correction_cap": float(args.correction_cap),
                "input_channels": INPUT_CHANNELS,
            },
            "holdout_metrics": initial_metrics,
        }
        initial_temporary = output_dir / f".best.pt.tmp-{os.getpid()}"
        torch.save(initial_checkpoint, initial_temporary)
        os.replace(initial_temporary, output_dir / "best.pt")
        atomic_json(
            {
                "epoch": best_epoch,
                "score": best_score,
                "metrics": best_metrics,
            },
            output_dir / "best.json",
        )
        print(
            json.dumps(
                {
                    "event": "initial_identity_evaluation",
                    "candidate_mean": initial_aggregate["candidate_mean"],
                    "candidate_max": initial_aggregate["candidate_max"],
                    "maximum_identity_metric_difference": maximum_identity_difference,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if distributed:
        torch.distributed.barrier()
    global_step = 0
    try:
        for epoch in range(1, int(args.epochs) + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            epoch_total = torch.zeros((), device=device, dtype=torch.float64)
            epoch_batches = 0
            for batch in loader:
                features, coarse, truth, mean_energy, active = batch
                features = features.to(device, non_blocking=True)
                coarse = coarse.to(device, non_blocking=True)
                truth = truth.to(device, non_blocking=True)
                mean_energy = mean_energy.to(device, non_blocking=True)
                active = active.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.amp
                    else nullcontext()
                )
                with context:
                    correction = model(features, active=active)
                    loss, components = train_loss(
                        correction.float(),
                        coarse,
                        truth,
                        mean_energy,
                        hinge_weight=float(args.hinge_weight),
                        gradient_weight=float(args.gradient_weight),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite R25 loss at step {global_step}")
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
                        "relative_square": float(components["relative_square"]),
                        "parent_relative_square": float(
                            components["parent_relative_square"]
                        ),
                        "hinge": float(components["hinge"]),
                        "gradient": float(components["gradient"]),
                        "correction_energy": float(components["correction_energy"]),
                        "gradient_norm": float(gradient_norm),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                    with updates_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)
            scheduler.step()
            mean_loss = epoch_total / max(epoch_batches, 1)
            mean_loss_value = reduce_scalar(mean_loss, distributed=distributed)

            should_evaluate = epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs)
            if distributed:
                torch.distributed.barrier()
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
                            "mean_relative_improvement": aggregate[
                                "mean_relative_improvement"
                            ],
                            "max_relative_improvement": aggregate[
                                "max_relative_improvement"
                            ],
                            "absolute_goal_passed": metrics["absolute_goal"]["passed"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    best_metrics = metrics
                    checkpoint = {
                        "schema": "r25_coarse_residual_checkpoint_v1",
                        "epoch": epoch,
                        "global_step": global_step,
                        "selection_sha256": fit.selection_sha256,
                        "model_state_dict": {
                            key: value.detach().cpu()
                            for key, value in model_for_save.state_dict().items()
                        },
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "model_config": {
                            "base_width": int(args.base_width),
                            "correction_cap": float(args.correction_cap),
                            "input_channels": INPUT_CHANNELS,
                        },
                        "holdout_metrics": metrics,
                    }
                    temporary = output_dir / f".best.pt.tmp-{os.getpid()}"
                    torch.save(checkpoint, temporary)
                    os.replace(temporary, output_dir / "best.pt")
                    atomic_json(
                        {
                            "epoch": best_epoch,
                            "score": best_score,
                            "metrics": best_metrics,
                        },
                        output_dir / "best.json",
                    )
            if distributed:
                torch.distributed.barrier()

        if rank == 0:
            terminal = {
                "schema": "r25_coarse_residual_terminal_v1",
                "status": "complete",
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "global_step": global_step,
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": str(output_dir / "best.pt"),
                "checkpoint_sha256": sha256_file(output_dir / "best.pt"),
                "validation_opened": False,
                "test_id_opened": False,
            }
            atomic_json(terminal, output_dir / "terminal.json")
            print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        if rank == 0:
            atomic_json(
                {
                    "schema": "r25_coarse_residual_terminal_v1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "global_step": global_step,
                    "elapsed_seconds": time.perf_counter() - started,
                    "validation_opened": False,
                    "test_id_opened": False,
                },
                output_dir / "terminal.json",
            )
        raise
    finally:
        fit.close()
        holdout.close()
        if distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
