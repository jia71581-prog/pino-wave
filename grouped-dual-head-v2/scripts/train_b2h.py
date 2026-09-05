#!/usr/bin/env python3
"""Independent DDP trainer for the B2-H physical residual propagator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from saved_time_phase_operator_v4.b2h import PhysicalResidualPropagator
from saved_time_phase_operator_v4.b2h_training import (
    SequenceWindowDataset,
    metric_rows,
    relative_window_loss,
    summarize_metric_rows,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _save_checkpoint(
    path: Path,
    *,
    model: PhysicalResidualPropagator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config_digest: str,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "b2h-smoke-v1",
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config_digest": config_digest,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        },
        temporary,
    )
    os.replace(temporary, path)


def _distributed_setup() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world, torch.device("cuda", local_rank)


def _all_rows(local_rows: list[dict], world: int) -> list[dict]:
    if world == 1:
        return local_rows
    gathered: list[list[dict] | None] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_rows)
    return [row for block in gathered if block is not None for row in block]


@torch.no_grad()
def _validate(
    ddp_model,
    loader,
    *,
    device: torch.device,
    world: int,
    time_count: int,
    energy_floor_fraction: float,
) -> tuple[dict, dict]:
    module = ddp_model.module if isinstance(ddp_model, DistributedDataParallel) else ddp_model
    module.eval()
    learned_rows: list[dict] = []
    baseline_rows: list[dict] = []
    saved_gate = module.gate.detach().clone()
    for batch in loader:
        tensors = {
            name: batch[name].to(device, non_blocking=True)
            for name in (
                "history", "p0", "p1", "target", "velocity", "source_map",
                "source_series", "source_parameters", "initial_time_s",
            )
        }
        module.gate.zero_()
        baseline = module(
            tensors["p0"], tensors["p1"], tensors["velocity"],
            tensors["source_map"], tensors["source_series"],
            source_parameters=tensors["source_parameters"],
            initial_time_s=tensors["initial_time_s"],
            history=tensors["history"] if module.memory_steps else None,
        )
        module.gate.copy_(saved_gate)
        learned = module(
            tensors["p0"], tensors["p1"], tensors["velocity"],
            tensors["source_map"], tensors["source_series"],
            source_parameters=tensors["source_parameters"],
            initial_time_s=tensors["initial_time_s"],
            history=tensors["history"] if module.memory_steps else None,
        )
        common = {
            "target": tensors["target"],
            "families": list(batch["family"]),
            "records": batch["record"],
            "starts": batch["start"],
            "time_count": time_count,
            "energy_floor_fraction": energy_floor_fraction,
        }
        baseline_rows.extend(metric_rows(baseline, **common))
        learned_rows.extend(metric_rows(learned, **common))
    module.gate.copy_(saved_gate)
    return (
        summarize_metric_rows(_all_rows(learned_rows, world)),
        summarize_metric_rows(_all_rows(baseline_rows, world)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw)
    config_digest = hashlib.sha256(raw).hexdigest()
    rank, local_rank, world, device = _distributed_setup()
    is_main = rank == 0
    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = Path(config["run_dir"])
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        _atomic_json(
            run_dir / "run_identity.json",
            {
                "architecture": (
                    "B2-HM PhysicalResidualPropagator"
                    if int(config["model"].get("memory_steps", 0))
                    else "B2-H PhysicalResidualPropagator"
                ),
                "config": str(config_path),
                "config_digest": config_digest,
                "dataset": config["dataset"],
                "protocol_scope": "teacher_initialized_fixed_windows_smoke",
                "strict_goal_eligible": False,
                "world_size": world,
                "seed": seed,
                "a3_checkpoint_preserved": config.get("a3_checkpoint_preserved"),
                "warmstart_checkpoint": config.get("warmstart_checkpoint"),
                "memory_steps": int(config["model"].get("memory_steps", 0)),
            },
        )
        _atomic_json(
            run_dir / "terminal.json",
            {"status": "running", "started_at": time.time()},
        )
    if world > 1:
        dist.barrier()

    data = config["data"]
    train_data = SequenceWindowDataset(
        config["dataset"],
        split="train",
        rollout_steps=int(data["rollout_steps"]),
        seed=seed,
        records_per_family=int(data["train_records_per_family"]),
        history_steps=int(data.get("history_steps", 0)),
        samples_per_epoch=int(data["samples_per_epoch"]),
    )
    validation_data = SequenceWindowDataset(
        config["dataset"],
        split="validation",
        rollout_steps=int(data["rollout_steps"]),
        seed=seed + 17,
        records_per_family=int(data["validation_records_per_family"]),
        history_steps=int(data.get("history_steps", 0)),
        fixed_starts=tuple(int(v) for v in data["validation_starts"]),
    )
    train_sampler = DistributedSampler(
        train_data, num_replicas=world, rank=rank, shuffle=True, seed=seed
    )
    validation_sampler = DistributedSampler(
        validation_data, num_replicas=world, rank=rank, shuffle=False
    )
    loader_args = {
        "batch_size": int(data["batch_size_per_rank"]),
        "num_workers": int(data.get("workers", 0)),
        "pin_memory": True,
    }
    train_loader = DataLoader(train_data, sampler=train_sampler, **loader_args)
    validation_loader = DataLoader(
        validation_data, sampler=validation_sampler, **loader_args
    )

    model = PhysicalResidualPropagator(**config["model"]).to(device)
    warmstart_path = config.get("warmstart_checkpoint")
    if warmstart_path:
        parent = torch.load(
            warmstart_path, map_location=device, weights_only=False
        )
        parent_state = dict(parent["model_state"])
        if int(config["model"].get("memory_steps", 0)) == 1:
            old_lift = parent_state["lift.weight"]
            new_lift = model.state_dict()["lift.weight"].clone()
            if (
                new_lift.shape[1] != old_lift.shape[1] + 1
                or new_lift.shape[0] != old_lift.shape[0]
            ):
                raise ValueError("memory warmstart lift shape is incompatible")
            new_lift[:, : old_lift.shape[1]].copy_(old_lift)
            new_lift[:, old_lift.shape[1] :].zero_()
            parent_state["lift.weight"] = new_lift
        model.load_state_dict(parent_state, strict=True)
    uses_memory = model.memory_steps == 1
    ddp_model = (
        DistributedDataParallel(model, device_ids=[local_rank])
        if world > 1 else model
    )
    gate_parameters = [model.gate, model.log_residual_scale]
    gate_ids = {id(value) for value in gate_parameters}
    network_parameters = [
        value for value in model.parameters() if id(value) not in gate_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": network_parameters,
                "lr": float(config["optimizer"]["learning_rate"]),
            },
            {
                "params": gate_parameters,
                "lr": float(config["optimizer"]["gate_learning_rate"]),
                "weight_decay": 0.0,
            },
        ],
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    energy_floor = float(config["loss"]["energy_floor_fraction"])
    epochs = int(config["epochs"])
    global_step = 0
    started = time.time()
    best_score = float("inf")
    try:
        for epoch in range(1, epochs + 1):
            train_data.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            ddp_model.train()
            torch.cuda.reset_peak_memory_stats(device)
            epoch_loss = 0.0
            gate_grad_sum = 0.0
            residual_grad_sum = 0.0
            for batch in train_loader:
                tensors = {
                    name: batch[name].to(device, non_blocking=True)
                    for name in (
                        "history", "p0", "p1", "target", "velocity",
                        "source_map", "source_series", "source_parameters",
                        "initial_time_s",
                    )
                }
                optimizer.zero_grad(set_to_none=True)
                prediction = ddp_model(
                    tensors["p0"], tensors["p1"], tensors["velocity"],
                    tensors["source_map"], tensors["source_series"],
                    source_parameters=tensors["source_parameters"],
                    initial_time_s=tensors["initial_time_s"],
                    history=tensors["history"] if uses_memory else None,
                )
                loss = relative_window_loss(
                    prediction,
                    tensors["target"],
                    energy_floor_fraction=energy_floor,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                loss.backward()
                gate_grad_sum += float(
                    0.0 if model.gate.grad is None else model.gate.grad.detach().abs()
                )
                residual_grad_sum += float(
                    0.0
                    if model.project.weight.grad is None
                    else model.project.weight.grad.detach().norm()
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config["optimizer"]["gradient_clip"]),
                )
                optimizer.step()
                epoch_loss += float(loss.detach())
                global_step += 1

            learned, baseline = _validate(
                ddp_model,
                validation_loader,
                device=device,
                world=world,
                time_count=validation_data.time_count,
                energy_floor_fraction=energy_floor,
            )
            if world > 1:
                dist.barrier()
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            if is_main:
                event = {
                    "event": "epoch",
                    "epoch": epoch,
                    "global_step": global_step,
                    "elapsed_seconds": time.time() - started,
                    "train_loss": epoch_loss / max(len(train_loader), 1),
                    "gate_raw": float(model.gate.detach()),
                    "gate_effective": float(torch.tanh(model.gate.detach())),
                    "residual_scale": float(model.residual_scale.detach()),
                    "gate_gradient_sum": gate_grad_sum,
                    "project_gradient_norm_sum": residual_grad_sum,
                    "peak_cuda_gib_rank0": peak_gib,
                    "metrics": learned,
                    "physical_baseline_metrics": baseline,
                    "validation_to_train_boundaries_completed": max(epoch - 1, 0),
                }
                _append_jsonl(run_dir / "metrics.jsonl", event)
                checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
                _save_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    config_digest=config_digest,
                    metrics=learned,
                )
                shutil.copyfile(checkpoint, run_dir / "latest.pt")
                if float(learned["aggregate_relative_l2"]) < best_score:
                    best_score = float(learned["aggregate_relative_l2"])
                    shutil.copyfile(checkpoint, run_dir / "best.pt")
                    _atomic_json(
                        run_dir / "best.json",
                        {
                            "epoch": epoch,
                            "checkpoint": str(checkpoint),
                            "metrics": learned,
                        },
                    )
                print(json.dumps(event, sort_keys=True), flush=True)
            if world > 1:
                dist.barrier()
        if is_main:
            _atomic_json(
                run_dir / "terminal.json",
                {
                    "status": "success",
                    "epochs": epochs,
                    "global_step": global_step,
                    "elapsed_seconds": time.time() - started,
                    "strict_goal_eligible": False,
                },
            )
    except Exception as error:
        if is_main:
            _atomic_json(
                run_dir / "terminal.json",
                {
                    "status": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "global_step": global_step,
                },
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
