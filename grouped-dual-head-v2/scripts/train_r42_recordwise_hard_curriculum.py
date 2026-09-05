#!/usr/bin/env python3
"""Train R42 with train-only hard-record sampling and recordwise physics loss."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler


def load_r40_module(path: Path):
    spec = importlib.util.spec_from_file_location("r40_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import R40 module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FitRecordDataset(Dataset):
    def __init__(self, r40, collection):
        if collection.expected_subset != "fit":
            raise ValueError("FitRecordDataset requires fit caches")
        self.r40 = r40
        self.collection = collection
        self.frequency_weights = r40.rfft_weights(r40.TIME_COUNT)[
            collection.frequency_indices
        ].astype(np.float32)

    def __len__(self) -> int:
        return len(self.collection.records)

    def __getitem__(self, position: int):
        file_index, local_index = self.collection.records[int(position)]
        handle = self.collection.handles[file_index]
        return (
            torch.from_numpy(
                np.asarray(handle["base_dct_norm"][local_index], dtype=np.float32)
            ),
            torch.from_numpy(
                np.asarray(handle["residual_dct_norm"][local_index], dtype=np.float32)
            ),
            torch.from_numpy(
                np.asarray(handle["static_dct_norm"][local_index], dtype=np.float32)
            ),
            torch.from_numpy(
                np.asarray(handle["static_dct_scale"][local_index], dtype=np.float32)
            ),
            torch.from_numpy(
                np.asarray(handle["frequency_hz"], dtype=np.float32)
            ),
            torch.tensor(float(handle["source_f0_hz"][local_index]), dtype=torch.float32),
            torch.tensor(float(handle["source_t0_s"][local_index]), dtype=torch.float32),
            torch.from_numpy(
                np.asarray(handle["frequency_scale"][local_index], dtype=np.float32)
            ),
            torch.tensor(
                float(handle["target_square_total"][local_index]), dtype=torch.float32
            ),
            torch.from_numpy(self.frequency_weights.copy()),
            torch.tensor(int(position), dtype=torch.int64),
        )


class DistributedWeightedRecordSampler(Sampler[int]):
    def __init__(
        self,
        weights: Sequence[float],
        *,
        num_replicas: int,
        rank: int,
        samples_per_rank: int,
        seed: int,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        if self.weights.ndim != 1 or bool(torch.any(self.weights <= 0)):
            raise ValueError("sampling weights must be a positive vector")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.samples_per_rank = int(samples_per_rank)
        self.total_size = self.samples_per_rank * self.num_replicas
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        global_draws = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        local = global_draws[self.rank : self.total_size : self.num_replicas]
        if local.numel() != self.samples_per_rank:
            raise RuntimeError("weighted sampler produced the wrong local draw count")
        return iter(local.tolist())

    def __len__(self) -> int:
        return self.samples_per_rank


def validate_curriculum(path: Path, fit) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "r42_train_only_hard_record_curriculum_v1":
        raise RuntimeError("unexpected R42 curriculum schema")
    if str(payload.get("selection_sha256")) != str(fit.selection_sha256):
        raise RuntimeError("curriculum/cache selection digest mismatch")
    if bool(payload.get("data_boundary", {}).get("r29b_opened")) or bool(
        payload.get("data_boundary", {}).get("final_validation_opened")
    ):
        raise RuntimeError("curriculum violates the frozen data boundary")
    records = payload.get("records", [])
    if len(records) != len(fit.records):
        raise RuntimeError("curriculum/cache record counts differ")
    weights = np.empty(len(records), dtype=np.float64)
    base_errors = np.empty(len(records), dtype=np.float64)
    for position, record in enumerate(records):
        file_index, local_index = fit.records[position]
        handle = fit.handles[file_index]
        expected = {
            "record_position": position,
            "cache_basename": fit.paths[file_index].name,
            "local_row": local_index,
            "sample_id": fit.sample_ids[position],
            "group_id": fit.group_ids[position],
            "family": fit.families[position],
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise RuntimeError(
                    f"curriculum identity mismatch at {position} for {key}: "
                    f"{record.get(key)} != {value}"
                )
        weights[position] = float(record["sampling_weight"])
        base_errors[position] = float(record["base_record_rel_l2"])
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("invalid curriculum weights")
    if not np.all(np.isfinite(base_errors)) or np.any(base_errors < 0):
        raise RuntimeError("invalid curriculum base errors")
    return payload, weights, base_errors


def make_record_features(
    r40,
    *,
    base: torch.Tensor,
    static_norm: torch.Tensor,
    static_scale: torch.Tensor,
    frequency_hz: torch.Tensor,
    frequency_scale: torch.Tensor,
    source_f0_hz: torch.Tensor,
    source_t0_s: torch.Tensor,
) -> torch.Tensor:
    count = int(base.shape[0])
    return r40.make_features(
        base,
        static_norm[None].expand(count, -1, -1, -1),
        static_scale[None].expand(count, -1),
        frequency_hz=frequency_hz,
        frequency_scale=frequency_scale,
        source_f0_hz=source_f0_hz.expand(count),
        source_t0_s=source_t0_s.expand(count),
    )


@torch.inference_mode()
def predict_cached_record(
    r40,
    model: torch.nn.Module,
    handle,
    local_index: int,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
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
    predictions = []
    for start in range(0, len(frequencies), int(batch_size)):
        stop = min(start + int(batch_size), len(frequencies))
        block = stop - start
        base = torch.from_numpy(base_norm[start:stop]).to(device)
        frequency = torch.from_numpy(frequencies[start:stop]).to(device)
        scale = torch.from_numpy(frequency_scale[start:stop]).to(device)
        features = r40.make_features(
            base,
            static_norm[None].expand(block, -1, -1, -1),
            static_scale[None].expand(block, -1),
            frequency_hz=frequency,
            frequency_scale=scale,
            source_f0_hz=torch.full((block,), f0, device=device),
            source_t0_s=torch.full((block,), t0, device=device),
        )
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp
            else nullcontext()
        )
        with context:
            predictions.append(model(features).float().cpu().numpy())
    return np.concatenate(predictions, axis=0)


@torch.inference_mode()
def evaluate_holdout_scales(
    r40,
    model: torch.nn.Module,
    collection,
    *,
    scales: Sequence[float],
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    if collection.expected_subset != "holdout":
        raise ValueError("holdout scale evaluation requires holdout caches")
    model.eval()
    indices = collection.frequency_indices
    fft_weight = r40.rfft_weights(r40.TIME_COUNT)[indices]
    retained = int(collection.retained)
    rows_by_scale: dict[float, list[dict[str, Any]]] = {
        float(scale): [] for scale in scales
    }
    family_by_scale = {
        float(scale): {name: [] for name in r40.FAMILIES} for scale in scales
    }
    for file_index, local_index in collection.records:
        handle = collection.handles[file_index]
        correction_norm = predict_cached_record(
            r40,
            model,
            handle,
            local_index,
            device=device,
            batch_size=batch_size,
            amp=amp,
        )
        frequency_scale = np.asarray(
            handle["frequency_scale"][local_index], dtype=np.float32
        )
        correction_coeff = r40.channels_to_complex(correction_norm)
        correction_coeff *= frequency_scale[:, None, None]
        base_selected = r40.channels_to_complex(
            np.asarray(handle["base_spectrum_selected"][local_index], dtype=np.float32)
        )
        truth_selected = r40.channels_to_complex(
            np.asarray(handle["truth_spectrum_selected"][local_index], dtype=np.float32)
        )
        grid = int(base_selected.shape[-1])
        padded = np.zeros((len(indices), grid, grid), dtype=np.complex64)
        padded[:, :retained, :retained] = correction_coeff
        correction_spatial = r40.idctn(
            padded.real, type=2, norm="ortho", axes=(-2, -1)
        ) + 1j * r40.idctn(padded.imag, type=2, norm="ortho", axes=(-2, -1))
        unselected = float(handle["base_error_square_unselected"][local_index])
        target_square = float(handle["target_square_total"][local_index])
        sample_id = str(handle["sample_id"].asstr()[local_index])
        group_id = str(handle["group_id"].asstr()[local_index])
        family = str(handle["family"].asstr()[local_index])
        parent_selected_square = float(
            np.sum(
                fft_weight
                * np.sum(
                    np.abs(base_selected - truth_selected) ** 2,
                    axis=(1, 2),
                    dtype=np.float64,
                )
            )
        )
        parent_rel = math.sqrt(
            (unselected + parent_selected_square) / max(target_square, 1.0e-30)
        )
        for scale in rows_by_scale:
            candidate_selected_square = float(
                np.sum(
                    fft_weight
                    * np.sum(
                        np.abs(
                            base_selected
                            + float(scale) * correction_spatial
                            - truth_selected
                        )
                        ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                )
            )
            candidate_rel = math.sqrt(
                (unselected + candidate_selected_square)
                / max(target_square, 1.0e-30)
            )
            row = {
                "sample_id": sample_id,
                "group_id": group_id,
                "family": family,
                "candidate_rel_l2": candidate_rel,
                "parent_rel_l2": parent_rel,
                "relative_improvement": 1.0
                - candidate_rel / max(parent_rel, 1.0e-30),
            }
            rows_by_scale[scale].append(row)
            family_by_scale[scale][family].append(row)
    evaluations = {}
    for scale, rows in rows_by_scale.items():
        aggregate = r40.summarize_rows(rows)
        passed = bool(
            float(aggregate["candidate_mean"]) <= 0.05
            and float(aggregate["candidate_max"]) <= 0.05
        )
        evaluations[str(scale)] = {
            "correction_scale": scale,
            "aggregate": aggregate,
            "per_family": {
                family: r40.summarize_rows(values)
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


@torch.inference_mode()
def evaluate_fit_probe(
    r40,
    model: torch.nn.Module,
    fit,
    positions: Sequence[int],
    base_errors: np.ndarray,
    *,
    correction_scale: float,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    frequency_weight = r40.rfft_weights(r40.TIME_COUNT)[fit.frequency_indices]
    rows = []
    for position in positions:
        file_index, local_index = fit.records[int(position)]
        handle = fit.handles[file_index]
        prediction = predict_cached_record(
            r40,
            model,
            handle,
            local_index,
            device=device,
            batch_size=batch_size,
            amp=amp,
        )
        residual = np.asarray(
            handle["residual_dct_norm"][local_index], dtype=np.float32
        )
        frequency_scale = np.asarray(
            handle["frequency_scale"][local_index], dtype=np.float64
        )
        target_square = float(handle["target_square_total"][local_index])
        parent_retained = float(
            np.sum(
                frequency_weight
                * frequency_scale**2
                * np.sum(residual.astype(np.float64) ** 2, axis=(1, 2, 3))
            )
        )
        parent_total = float(base_errors[position]) ** 2 * target_square
        irreducible = max(parent_total - parent_retained, 0.0)
        error = float(correction_scale) * prediction.astype(np.float64) - residual
        candidate_retained = float(
            np.sum(
                frequency_weight
                * frequency_scale**2
                * np.sum(error**2, axis=(1, 2, 3))
            )
        )
        candidate_rel = math.sqrt(
            (irreducible + candidate_retained) / max(target_square, 1.0e-30)
        )
        rows.append(
            {
                "record_position": int(position),
                "sample_id": fit.sample_ids[position],
                "family": fit.families[position],
                "candidate_rel_l2": candidate_rel,
                "parent_rel_l2": float(base_errors[position]),
                "relative_improvement": 1.0
                - candidate_rel / max(float(base_errors[position]), 1.0e-30),
            }
        )
    return {"aggregate": r40.summarize_rows(rows), "records": rows}


def choose_evaluation(evaluations: Mapping[str, Mapping[str, Any]]):
    candidates = list(evaluations.values())
    return min(
        candidates,
        key=lambda item: (
            float(item["aggregate"]["candidate_max"])
            + 0.1 * float(item["aggregate"]["candidate_mean"]),
            float(item["correction_scale"]),
        ),
    )


def reduce_mean(value: torch.Tensor, *, distributed: bool) -> float:
    result = value.detach().clone()
    if distributed:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        result /= torch.distributed.get_world_size()
    return float(result.cpu())


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--records-per-rank-epoch", type=int, default=256)
    parser.add_argument("--frequency-chunk", type=int, default=15)
    parser.add_argument("--eval-batch-size", type=int, default=15)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--correction-cap", type=float, default=1.5)
    parser.add_argument("--hinge-weight", type=float, default=2.0)
    parser.add_argument("--shape-weight", type=float, default=0.002)
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=420829)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--stop-on-pass", action="store_true")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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

    r40 = load_r40_module(args.r40_script.expanduser().resolve())
    fit = None
    holdout = None
    try:
        fit = r40.FrequencyCacheCollection(args.fit_cache, expected_subset="fit")
        holdout = r40.FrequencyCacheCollection(
            args.holdout_cache, expected_subset="holdout"
        )
        if fit.selection_sha256 != holdout.selection_sha256:
            raise RuntimeError("fit/holdout selection digest mismatch")
        if set(fit.group_ids) & set(holdout.group_ids):
            raise RuntimeError("fit/holdout group leakage")
        curriculum_path = args.curriculum_manifest.expanduser().resolve()
        curriculum, weights, base_errors = validate_curriculum(
            curriculum_path, fit
        )
        probe_positions = np.argsort(-base_errors)[: int(args.probe_count)].tolist()
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            torch.distributed.barrier()

        dataset = FitRecordDataset(r40, fit)
        sampler = DistributedWeightedRecordSampler(
            weights,
            num_replicas=world_size,
            rank=rank,
            samples_per_rank=int(args.records_per_rank_epoch),
            seed=int(args.seed),
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        model_for_save = r40.FrequencyResidualFNO(
            width=int(args.width),
            modes=int(args.modes),
            blocks=int(args.blocks),
            correction_cap=float(args.correction_cap),
        ).to(device)
        model: torch.nn.Module = model_for_save
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
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(int(args.epochs) - warmup_epochs, 1),
                        eta_min=float(args.learning_rate) * 0.05,
                    ),
                ],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(args.epochs), 1),
                eta_min=float(args.learning_rate) * 0.05,
            )

        scales = (0.0, 0.25, 0.5, 0.75, 1.0)
        manifest_sha256 = sha256_file(curriculum_path)
        if rank == 0:
            identity = {
                "schema": "r42_recordwise_hard_curriculum_identity_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "selection_sha256": fit.selection_sha256,
                "curriculum_manifest": str(curriculum_path),
                "curriculum_manifest_sha256": manifest_sha256,
                "fit_record_count": len(fit.records),
                "fit_group_count": len(set(fit.group_ids)),
                "holdout_record_count": len(holdout.records),
                "holdout_group_count": len(set(holdout.group_ids)),
                "model": {
                    "name": "frequency_conditioned_spatial_dct_fno",
                    "parameter_count": r40.parameter_count(model_for_save),
                    "width": int(args.width),
                    "modes": int(args.modes),
                    "blocks": int(args.blocks),
                    "correction_cap": float(args.correction_cap),
                    "zero_initialized_output": True,
                },
                "optimization": {
                    "objective": "recordwise_retained_physical_relative_l2_squared",
                    "hard_record_sampling": True,
                    "records_per_rank_epoch": int(args.records_per_rank_epoch),
                    "global_records_per_epoch": int(args.records_per_rank_epoch)
                    * world_size,
                    "frequency_chunk": int(args.frequency_chunk),
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "warmup_epochs": warmup_epochs,
                    "hinge_weight": float(args.hinge_weight),
                    "shape_weight": float(args.shape_weight),
                    "amp_bfloat16": bool(args.amp),
                    "correction_scale_candidates": list(scales),
                    "selection_score": "candidate_max_plus_0p1_candidate_mean",
                    "seed": int(args.seed),
                },
                "absolute_goal": {
                    "record_rel_l2_mean_lte": 0.05,
                    "record_rel_l2_max_lte": 0.05,
                },
                "data_boundary": {
                    "fit_truth_used": True,
                    "opened_development_holdout_used": True,
                    "r29b_opened": False,
                    "final_validation_opened": False,
                    "test_id_opened": False,
                    "paper_modified": False,
                },
            }
            atomic_json(identity, output_dir / "run_identity.json")
        if distributed:
            torch.distributed.barrier()

        started = time.perf_counter()
        global_step = 0
        best_score = math.inf
        best_epoch = -1
        best_metrics = None
        stopped_on_pass = False
        metrics_path = output_dir / "holdout_metrics.jsonl"
        updates_path = output_dir / "updates.jsonl"
        if rank == 0:
            initial_scales = evaluate_holdout_scales(
                r40,
                model_for_save,
                holdout,
                scales=scales,
                device=device,
                batch_size=int(args.eval_batch_size),
                amp=bool(args.amp),
            )
            initial = choose_evaluation(initial_scales)
            if float(initial["correction_scale"]) != 0.0:
                raise RuntimeError("zero-initialized R42 did not select identity scale")
            maximum_identity_difference = max(
                abs(float(row["candidate_rel_l2"]) - float(row["parent_rel_l2"]))
                for row in initial["records"]
            )
            if maximum_identity_difference > 1.0e-8:
                raise RuntimeError("R42 initial model is not an exact identity")
            initial_probe = evaluate_fit_probe(
                r40,
                model_for_save,
                fit,
                probe_positions,
                base_errors,
                correction_scale=0.0,
                device=device,
                batch_size=int(args.eval_batch_size),
                amp=bool(args.amp),
            )
            metrics = {
                "event": "initial_identity_evaluation",
                "epoch": 0,
                "global_step": 0,
                "train_loss": None,
                "selected": initial,
                "scales": initial_scales,
                "fit_hard_probe": initial_probe,
                "maximum_identity_metric_difference": maximum_identity_difference,
                "elapsed_seconds": time.perf_counter() - started,
            }
            best_score = float(initial["aggregate"]["candidate_max"]) + 0.1 * float(
                initial["aggregate"]["candidate_mean"]
            )
            best_epoch = 0
            best_metrics = metrics
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            atomic_json(metrics, output_dir / "initial_holdout.json")
            checkpoint = {
                "schema": "r42_recordwise_hard_curriculum_checkpoint_v1",
                "epoch": 0,
                "global_step": 0,
                "model_state_dict": model_for_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "selection_sha256": fit.selection_sha256,
                "curriculum_manifest_sha256": manifest_sha256,
                "selected_correction_scale": 0.0,
                "metrics": metrics,
                "model_config": {
                    "width": int(args.width),
                    "modes": int(args.modes),
                    "blocks": int(args.blocks),
                    "correction_cap": float(args.correction_cap),
                },
            }
            save_checkpoint(checkpoint, output_dir / "best.pt")
            save_checkpoint(checkpoint, output_dir / "latest.pt")
            atomic_json(
                {"epoch": 0, "score": best_score, "metrics": metrics},
                output_dir / "best.json",
            )
            print(
                json.dumps(
                    {
                        "event": "initial_identity_evaluation",
                        "candidate_mean": initial["aggregate"]["candidate_mean"],
                        "candidate_max": initial["aggregate"]["candidate_max"],
                        "fit_probe_max": initial_probe["aggregate"]["candidate_max"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if distributed:
            torch.distributed.barrier()

        for epoch in range(1, int(args.epochs) + 1):
            sampler.set_epoch(epoch)
            model.train()
            epoch_loss = torch.zeros((), device=device, dtype=torch.float64)
            epoch_records = 0
            for batch in loader:
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
                    record_position,
                ) = batch
                base = base[0].to(device, non_blocking=True)
                residual = residual[0].to(device, non_blocking=True)
                static_norm = static_norm[0].to(device, non_blocking=True)
                static_scale = static_scale[0].to(device, non_blocking=True)
                frequency_hz = frequency_hz[0].to(device, non_blocking=True)
                source_f0_hz = source_f0_hz.to(device, non_blocking=True)
                source_t0_s = source_t0_s.to(device, non_blocking=True)
                frequency_scale = frequency_scale[0].to(device, non_blocking=True)
                target_square = target_square_total[0].to(device, non_blocking=True)
                frequency_weight = frequency_weight[0].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                frequency_count = int(base.shape[0])
                chunk_starts = list(
                    range(0, frequency_count, int(args.frequency_chunk))
                )
                for chunk_number, start in enumerate(chunk_starts):
                    stop = min(start + int(args.frequency_chunk), frequency_count)
                    sl = slice(start, stop)
                    features = make_record_features(
                        r40,
                        base=base[sl],
                        static_norm=static_norm,
                        static_scale=static_scale,
                        frequency_hz=frequency_hz[sl],
                        frequency_scale=frequency_scale[sl],
                        source_f0_hz=source_f0_hz,
                        source_t0_s=source_t0_s,
                    )
                    should_sync = chunk_number == len(chunk_starts) - 1
                    sync_context = (
                        nullcontext()
                        if should_sync or not distributed
                        else model.no_sync()
                    )
                    amp_context = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if args.amp
                        else nullcontext()
                    )
                    with sync_context:
                        with amp_context:
                            prediction = model(features)
                            error_square = (
                                prediction.float() - residual[sl].float()
                            ).square().sum(dim=(1, 2, 3))
                            parent_square = residual[sl].float().square().sum(
                                dim=(1, 2, 3)
                            )
                            physical_scale = (
                                frequency_weight[sl].float()
                                * frequency_scale[sl].float().square()
                                / target_square.float().clamp_min(1.0e-12)
                            )
                            candidate_contribution = physical_scale * error_square
                            parent_contribution = physical_scale * parent_square
                            physical = candidate_contribution.sum()
                            hinge = F.relu(
                                torch.sqrt(candidate_contribution.clamp_min(1.0e-14))
                                - torch.sqrt(parent_contribution.detach().clamp_min(1.0e-14))
                            ).square().sum()
                            shape = F.smooth_l1_loss(
                                prediction.float(), residual[sl].float(), beta=0.02
                            ) * ((stop - start) / frequency_count)
                            loss = (
                                physical
                                + float(args.hinge_weight) * hinge
                                + float(args.shape_weight) * shape
                            )
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"nonfinite R42 loss at step {global_step}"
                            )
                        loss.backward()
                    accumulated_loss += float(loss.detach().cpu())
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                epoch_records += 1
                epoch_loss += accumulated_loss
                if rank == 0 and (global_step == 1 or global_step % 50 == 0):
                    event = {
                        "event": "update",
                        "epoch": epoch,
                        "global_step": global_step,
                        "record_position": int(record_position[0]),
                        "record_loss": accumulated_loss,
                        "gradient_norm": float(gradient_norm),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                    with updates_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)
            if epoch_records != int(args.records_per_rank_epoch):
                raise RuntimeError("R42 epoch record count mismatch")
            scheduler.step()
            mean_loss = epoch_loss / epoch_records
            mean_loss_value = reduce_mean(mean_loss, distributed=distributed)
            should_evaluate = (
                epoch == 1
                or epoch % int(args.eval_every) == 0
                or epoch == int(args.epochs)
            )
            if distributed:
                torch.distributed.barrier()
            passed = False
            if rank == 0 and should_evaluate:
                scale_evaluations = evaluate_holdout_scales(
                    r40,
                    model_for_save,
                    holdout,
                    scales=scales,
                    device=device,
                    batch_size=int(args.eval_batch_size),
                    amp=bool(args.amp),
                )
                selected = choose_evaluation(scale_evaluations)
                selected_scale = float(selected["correction_scale"])
                fit_probe = evaluate_fit_probe(
                    r40,
                    model_for_save,
                    fit,
                    probe_positions,
                    base_errors,
                    correction_scale=selected_scale,
                    device=device,
                    batch_size=int(args.eval_batch_size),
                    amp=bool(args.amp),
                )
                metrics = {
                    "event": "holdout_evaluation",
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": mean_loss_value,
                    "selected": selected,
                    "scales": scale_evaluations,
                    "fit_hard_probe": fit_probe,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                atomic_json(metrics, output_dir / "latest_holdout.json")
                checkpoint = {
                    "schema": "r42_recordwise_hard_curriculum_checkpoint_v1",
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model_for_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "selection_sha256": fit.selection_sha256,
                    "curriculum_manifest_sha256": manifest_sha256,
                    "selected_correction_scale": selected_scale,
                    "metrics": metrics,
                    "model_config": {
                        "width": int(args.width),
                        "modes": int(args.modes),
                        "blocks": int(args.blocks),
                        "correction_cap": float(args.correction_cap),
                    },
                }
                save_checkpoint(checkpoint, output_dir / "latest.pt")
                aggregate = selected["aggregate"]
                score = float(aggregate["candidate_max"]) + 0.1 * float(
                    aggregate["candidate_mean"]
                )
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    best_metrics = metrics
                    save_checkpoint(checkpoint, output_dir / "best.pt")
                    atomic_json(
                        {"epoch": epoch, "score": score, "metrics": metrics},
                        output_dir / "best.json",
                    )
                passed = bool(selected["absolute_goal"]["passed"])
                print(
                    json.dumps(
                        {
                            "event": "holdout_evaluation",
                            "epoch": epoch,
                            "candidate_mean": aggregate["candidate_mean"],
                            "candidate_max": aggregate["candidate_max"],
                            "correction_scale": selected_scale,
                            "fit_probe_max": fit_probe["aggregate"]["candidate_max"],
                            "train_loss": mean_loss_value,
                            "absolute_goal_passed": passed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
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
                break

        if rank == 0:
            summary = {
                "schema": "r42_recordwise_hard_curriculum_summary_v1",
                "status": "complete",
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "stopped_on_pass": stopped_on_pass,
                "absolute_goal_passed": bool(
                    best_metrics
                    and best_metrics["selected"]["absolute_goal"]["passed"]
                ),
                "r29b_opened": False,
                "final_validation_opened": False,
                "test_id_opened": False,
                "paper_modified": False,
            }
            atomic_json(summary, output_dir / "run_summary.json")
            print(
                json.dumps(
                    {
                        "event": "run_complete",
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "absolute_goal_passed": summary["absolute_goal_passed"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if distributed:
            torch.distributed.barrier()
    finally:
        if fit is not None:
            fit.close()
        if holdout is not None:
            holdout.close()
        if distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
