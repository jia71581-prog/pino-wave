#!/usr/bin/env python3
"""R29A development fine-tune for late-time complex-medium tail errors.

This experiment continues the R28 best checkpoint on the unchanged R28 fit
cache.  It oversamples late frames and difficult/Marmousi records, while the
already-opened R28 train holdout remains development-only.  Validation and
test data are never opened by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


SCRIPT_PATH = Path(__file__).resolve()
R28_PATH = SCRIPT_PATH.with_name("train_r28_expanded_tail_spectral.py")
SPEC = importlib.util.spec_from_file_location("r29a_r28_components", R28_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R28 components: {R28_PATH}")
r28 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r28
SPEC.loader.exec_module(r28)
r26 = r28.r26
r25 = r28.r25


class LateTailFitFrameDataset(r25.FitFrameDataset):
    """Oversample difficult records and the late multiple-scattering regime."""

    def __init__(self, collection: Any, *, late_start_s: float):
        super().__init__(collection)
        metadata: list[tuple[float, str]] = []
        for file_index, local_index in collection.records:
            handle = collection.handles[file_index]
            error_square = float(handle["baseline_error_square_norm"][local_index])
            target_square = float(handle["target_square_norm"][local_index])
            baseline = math.sqrt(error_square / max(target_square, 1.0e-30))
            family = str(handle["family"].asstr()[local_index])
            metadata.append((baseline, family))
        values = np.asarray([row[0] for row in metadata], dtype=np.float64)
        q75 = float(np.quantile(values, 0.75))
        q90 = float(np.quantile(values, 0.90))

        reference_times = np.asarray(collection.handles[0]["time_s"][:], dtype=np.float64)
        if len(reference_times) != self.time_count:
            raise RuntimeError("fit-cache time axis mismatch")
        mapping: list[int] = []
        record_repeats: list[int] = []
        for record_position, (baseline, family) in enumerate(metadata):
            repeats = (
                1
                + int(family == "marmousi")
                + int(baseline >= q75)
                + int(baseline >= q90)
            )
            record_repeats.append(repeats)
            start = record_position * self.time_count
            for frame_position, time_s in enumerate(reference_times):
                late_repeats = 1 + int(float(time_s) >= float(late_start_s))
                mapping.extend(
                    [start + frame_position] * (repeats * late_repeats)
                )
        self.mapping = tuple(mapping)
        self.baseline_q75 = q75
        self.baseline_q90 = q90
        self.record_repeats = tuple(record_repeats)
        self.late_start_s = float(late_start_s)

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int):
        return super().__getitem__(self.mapping[int(index)])


def set_seed(seed: int, rank: int) -> None:
    value = int(seed) + 100003 * int(rank)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    selection_sha256: str,
    base_width: int,
    correction_cap: float,
) -> str:
    payload = {
        "schema": "r29a_late_tail_finetune_checkpoint_v1",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "selection_sha256": selection_sha256,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": {
            "base_width": int(base_width),
            "correction_cap": float(correction_cap),
            "input_channels": int(r25.INPUT_CHANNELS),
        },
        "holdout_metrics": metrics,
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return r25.sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hinge-weight", type=float, default=3.0)
    parser.add_argument("--gradient-weight", type=float, default=0.05)
    parser.add_argument("--late-start-s", type=float, default=0.50)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=290828)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(int(args.seed), rank)

    fit = r25.CacheCollection(args.fit_cache, expected_subset="fit")
    holdout = r25.CacheCollection(args.holdout_cache, expected_subset="holdout")
    if fit.selection_sha256 != holdout.selection_sha256:
        raise RuntimeError("fit and holdout cache selection digests differ")
    dataset = LateTailFitFrameDataset(fit, late_start_s=float(args.late_start_s))
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

    init_path = args.init_checkpoint.expanduser().resolve()
    initial = torch.load(init_path, map_location="cpu", weights_only=False)
    config = initial.get("model_config", {})
    base_width = int(config.get("base_width", 32))
    correction_cap = float(config.get("correction_cap", 0.25))
    model = r26.TailSpectralResidualUNet(
        base_width=base_width, correction_cap=correction_cap
    )
    model.load_state_dict(initial["model_state_dict"], strict=True)
    model.to(device)
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
        optimizer,
        T_max=max(int(args.epochs), 1),
        eta_min=float(args.learning_rate) * 0.1,
    )

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        preregistration = {
            "schema": "r29a_late_tail_finetune_preregistration_v1",
            "status": "frozen_before_development_finetune",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "role": "R28_already_opened_train_holdout_development_only",
            "initial_checkpoint": str(init_path),
            "initial_checkpoint_sha256": r25.sha256_file(init_path),
            "fit_selection_sha256": fit.selection_sha256,
            "sampling": {
                "record_repeats": "1 + Marmousi + baseline_q75 + baseline_q90",
                "late_frame_repeats": f"2_at_or_after_{float(args.late_start_s):.6f}_s_else_1",
                "dataset_frame_instances": len(dataset),
                "baseline_q75": dataset.baseline_q75,
                "baseline_q90": dataset.baseline_q90,
            },
            "optimization": {
                "epochs": int(args.epochs),
                "global_batch_size": int(args.batch_size) * world_size,
                "learning_rate": float(args.learning_rate),
                "loss": "R26 frame tail CVaR proxy plus record/frame oversampling",
                "max_steps_per_epoch": int(args.max_steps_per_epoch),
            },
            "success_gate": {
                "development_mean_lte": 0.05,
                "development_max_lte": 0.05,
                "both_required": True,
            },
            "evidence_boundary": "A pass only authorizes a fresh group-disjoint train holdout experiment.",
            "validation_opened": False,
            "test_id_opened": False,
            "script_sha256": r25.sha256_file(SCRIPT_PATH),
            "r28_wrapper_sha256": r25.sha256_file(R28_PATH),
        }
        r25.atomic_json(preregistration, output_dir / "r29a_preregistration.json")
    if distributed:
        torch.distributed.barrier()

    started = time.perf_counter()
    global_step = 0
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    best_score = math.inf
    if rank == 0:
        initial_metrics = r25.evaluate(
            model_for_save,
            holdout,
            device=device,
            batch_size=int(args.eval_batch_size),
            amp=bool(args.amp),
        )
        initial_metrics.update(
            {
                "event": "initial_checkpoint_evaluation",
                "epoch": 0,
                "global_step": 0,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        aggregate = initial_metrics["aggregate"]
        best_score = float(aggregate["candidate_max"]) + float(
            aggregate["candidate_mean"]
        )
        best_metrics = initial_metrics
        sha = save_checkpoint(
            output_dir / "best.pt",
            model=model_for_save,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            global_step=0,
            metrics=initial_metrics,
            selection_sha256=str(fit.selection_sha256),
            base_width=base_width,
            correction_cap=correction_cap,
        )
        r25.atomic_json(initial_metrics, output_dir / "initial_holdout.json")
        print(
            json.dumps(
                {
                    "event": "initial_checkpoint_evaluation",
                    "candidate_mean": aggregate["candidate_mean"],
                    "candidate_max": aggregate["candidate_max"],
                    "checkpoint_sha256": sha,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if distributed:
        torch.distributed.barrier()

    updates_path = output_dir / "updates.jsonl"
    metrics_path = output_dir / "holdout_metrics.jsonl"
    for epoch in range(1, int(args.epochs) + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
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
                loss, components = r26.tail_risk_loss(
                    correction.float(),
                    coarse,
                    truth,
                    mean_energy,
                    hinge_weight=float(args.hinge_weight),
                    gradient_weight=float(args.gradient_weight),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite R29A loss at step {global_step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            if rank == 0 and (global_step == 1 or global_step % 100 == 0):
                event = {
                    "event": "update",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": float(loss.detach()),
                    "relative_square": float(components["relative_square"]),
                    "parent_relative_square": float(components["parent_relative_square"]),
                    "hinge": float(components["hinge"]),
                    "gradient": float(components["gradient"]),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                with updates_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event, sort_keys=True), flush=True)
            if int(args.max_steps_per_epoch) > 0 and global_step % int(args.max_steps_per_epoch) == 0:
                break
        scheduler.step()
        if distributed:
            torch.distributed.barrier()
        if rank == 0:
            metrics = r25.evaluate(
                model_for_save,
                holdout,
                device=device,
                batch_size=int(args.eval_batch_size),
                amp=bool(args.amp),
            )
            metrics.update(
                {
                    "event": "development_holdout_evaluation",
                    "epoch": epoch,
                    "global_step": global_step,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            aggregate = metrics["aggregate"]
            score = float(aggregate["candidate_max"]) + float(
                aggregate["candidate_mean"]
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            r25.atomic_json(metrics, output_dir / "latest_holdout.json")
            print(
                json.dumps(
                    {
                        "event": "development_holdout_evaluation",
                        "epoch": epoch,
                        "candidate_mean": aggregate["candidate_mean"],
                        "candidate_max": aggregate["candidate_max"],
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
                sha = save_checkpoint(
                    output_dir / "best.pt",
                    model=model_for_save,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=metrics,
                    selection_sha256=str(fit.selection_sha256),
                    base_width=base_width,
                    correction_cap=correction_cap,
                )
                r25.atomic_json(
                    {"epoch": best_epoch, "score": best_score, "checkpoint_sha256": sha, "metrics": best_metrics},
                    output_dir / "best.json",
                )
        if distributed:
            torch.distributed.barrier()

    if rank == 0:
        if best_metrics is None:
            raise RuntimeError("missing R29A best metrics")
        terminal = {
            "schema": "r29a_late_tail_finetune_terminal_v1",
            "status": "complete",
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "best_metrics": best_metrics,
            "checkpoint": str(output_dir / "best.pt"),
            "checkpoint_sha256": r25.sha256_file(output_dir / "best.pt"),
            "absolute_goal_passed": bool(best_metrics["absolute_goal"]["passed"]),
            "validation_opened": False,
            "test_id_opened": False,
            "elapsed_seconds": time.perf_counter() - started,
            "script_sha256": r25.sha256_file(SCRIPT_PATH),
        }
        r25.atomic_json(terminal, output_dir / "terminal.json")
        print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    fit.close()
    holdout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
