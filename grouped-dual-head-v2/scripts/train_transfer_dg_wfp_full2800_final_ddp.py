#!/usr/bin/env python3
"""Four-rank final phase-WFP pretraining on all 2,800 train records."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.train_transfer_dg_wfp_e1 import (  # noqa: E402
    CacheCollection,
    FeatureBuilder,
    atomic_checkpoint,
    atomic_json,
    sha256,
)
from scripts.train_transfer_dg_wfp_e1d import (  # noqa: E402
    FAMILIES,
    TravelCollection,
    apply_phase,
)
from saved_time_phase_operator_v4.wfp import (  # noqa: E402
    BackgroundFrequencyOperator,
    parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--travel", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=372)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run only this many optimizer steps and emit a smoke terminal.",
    )
    return parser.parse_args()


def reduced_mean(value: torch.Tensor, world_size: int) -> float:
    result = value.detach().float().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result.div_(world_size)
    return float(result)


def main() -> int:
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"expected four DDP ranks, observed {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_rank0 = rank == 0

    prereg = json.loads(args.preregistration.read_text())
    training = prereg["training"]
    bindings = prereg["bindings"]
    if sha256(Path(__file__)) != bindings["trainer_sha256"]:
        raise RuntimeError("final DDP trainer binding drift")
    if sha256(args.manifest) != bindings["manifest_sha256"]:
        raise RuntimeError("manifest binding drift")
    if args.epochs != int(training["epochs"]) or args.seed != int(training["seed"]):
        raise RuntimeError("training arguments differ from preregistration")
    expected_output = (ROOT / prereg["output_dir"]).resolve()
    if args.smoke_steps <= 0 and args.output_dir.resolve() != expected_output:
        raise RuntimeError("final output directory differs from preregistration")
    if args.smoke_steps < 0:
        raise ValueError("smoke steps must be nonnegative")

    if is_rank0:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    dist.barrier()

    manifest = json.loads(args.manifest.read_text())
    collection = CacheCollection(args.cache, manifest, expected_count=2800)
    travel = TravelCollection(args.travel, expected_count=2800)
    positions = list(range(len(collection.records)))
    if len(positions) != 2800:
        raise RuntimeError("full train census is not 2,800 records")
    if any(row[4] not in {"fit", "calibration", "confirmation"} for row in collection.records):
        raise RuntimeError("unexpected role in the all-train manifest")

    builder = FeatureBuilder(collection)
    by_family: dict[str, list[int]] = defaultdict(list)
    for position in positions:
        by_family[collection.records[position][3]].append(position)
    if set(by_family) != set(FAMILIES):
        raise RuntimeError(f"family census mismatch: {sorted(by_family)}")

    width = int(training["width"])
    spectral_rank = int(training["rank"])
    depth = int(training["depth"])
    radii = tuple(int(value) for value in training["radii"])
    global_per_family = int(training["global_per_family_batch"])
    if global_per_family != world_size:
        raise RuntimeError("exact global batch requires one item per family per rank")
    global_batch = global_per_family * len(FAMILIES)
    if global_batch != int(training["global_batch_size"]):
        raise RuntimeError("global batch binding mismatch")

    maximum_pairs = max(len(by_family[family]) * 64 for family in FAMILIES)
    steps_per_epoch = math.ceil(maximum_pairs / global_per_family)
    total_updates = args.epochs * steps_per_epoch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = BackgroundFrequencyOperator(
        medium_channels=12,
        source_channels=5,
        width=width,
        rank=spectral_rank,
        depth=depth,
        arm="wfp",
        radii=radii,
    ).to(device)
    distributed = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        distributed.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
        eta_min=float(training["eta_min"]),
    )
    active = torch.from_numpy(builder.active).to(device)
    rng = np.random.default_rng(args.seed + 91)
    identity = {
        "schema": "transfer_dg_wfp_full2800_final_ddp_identity_v1",
        "world_size": world_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_updates": total_updates,
        "global_batch_size": global_batch,
        "rank_batch_size": len(FAMILIES),
        "width": width,
        "rank": spectral_rank,
        "depth": depth,
        "parameter_count": parameter_count(model),
        "record_count": len(positions),
        "role_counts": {
            role: sum(row[4] == role for row in collection.records)
            for role in ("fit", "calibration", "confirmation")
        },
        "family_counts": {family: len(by_family[family]) for family in FAMILIES},
        "all_train_roles_used_for_gradients": True,
        "all_record_frequency_pairs_seen_each_epoch": True,
        "model_input_future_wavefield_frames": 0,
        "training_full_truth_allowed": True,
        "validation_opened": False,
        "test_id_opened": False,
        "manifest_sha256": sha256(args.manifest),
        "preregistration_sha256": sha256(args.preregistration),
        "trainer_sha256": sha256(Path(__file__)),
        "smoke_steps": args.smoke_steps,
    }
    if is_rank0:
        atomic_json(identity, args.output_dir / "run_identity.json")

    started = time.time()
    global_update = 0
    metrics_path = args.output_dir / "metrics.jsonl"
    stop = False
    last_physical = math.nan
    last_cpml = math.nan
    for epoch in range(1, args.epochs + 1):
        schedules: dict[str, np.ndarray] = {}
        for family in FAMILIES:
            values = np.asarray(
                [(position, frequency) for position in by_family[family] for frequency in range(64)],
                dtype=np.int64,
            )
            rng.shuffle(values)
            schedules[family] = values
        distributed.train()
        for step in range(steps_per_epoch):
            selected = []
            for family in FAMILIES:
                values = schedules[family]
                indices = (np.arange(global_per_family) + step * global_per_family) % len(values)
                selected.append(values[indices][rank].tolist())
            items = [builder.build(int(position), int(frequency), device) for position, frequency in selected]
            medium, source, scalars, target, aux_target = (
                torch.cat([item[index] for item in items]) for index in range(5)
            )
            sample_ids = [str(item[5]["sample_id"]) for item in items]
            frequencies = torch.tensor(
                [float(item[5]["frequency_hz"]) for item in items], device=device
            )
            travel_rows = [travel.read(sample_id) for sample_id in sample_ids]
            phase_time = torch.from_numpy(np.stack([row[0] for row in travel_rows])).to(device)
            envelope_time = torch.from_numpy(np.stack([row[1] for row in travel_rows])).to(device)
            physical_total = torch.tensor(
                [row[2] for row in travel_rows], device=device, dtype=torch.float32
            )
            auxiliary_total = torch.tensor(
                [row[3] for row in travel_rows], device=device, dtype=torch.float32
            )

            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary = distributed(medium, source, scalars)
            prediction, auxiliary = apply_phase(
                prediction,
                auxiliary,
                phase_time,
                envelope_time,
                frequencies,
                True,
            )
            physical_loss = (
                64
                * (prediction.float() - target.float()).square().sum((1, 2, 3))
                / physical_total.clamp_min(1e-8)
            ).mean()
            mask = active[None, None].expand_as(aux_target)
            auxiliary_error = (
                (auxiliary.float() - aux_target.float()).square() * mask
            ).sum((1, 2, 3))
            cpml_loss = (
                64 * auxiliary_error / auxiliary_total.clamp_min(1e-8)
            ).mean()
            loss = physical_loss + float(training["cpml_weight"]) * cpml_loss
            finite = torch.tensor(
                int(bool(torch.isfinite(loss))), device=device, dtype=torch.int32
            )
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                raise FloatingPointError(f"non-finite distributed loss at update {global_update + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(distributed.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_update += 1
            should_measure = global_update % 500 == 0 or (
                args.smoke_steps and global_update >= args.smoke_steps
            )
            if should_measure:
                last_physical = reduced_mean(physical_loss, world_size)
                last_cpml = reduced_mean(cpml_loss, world_size)
            if is_rank0 and global_update % 500 == 0:
                event = {
                    "event": "update",
                    "epoch": epoch,
                    "update": global_update,
                    "physical_loss": last_physical,
                    "cpml_loss": last_cpml,
                    "elapsed_s": time.time() - started,
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event), flush=True)
            if args.smoke_steps and global_update >= args.smoke_steps:
                stop = True
                break
        if stop:
            break
        dist.barrier()
        if is_rank0:
            checkpoint = {
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in distributed.module.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "update": global_update,
                "identity": identity,
            }
            atomic_checkpoint(checkpoint, args.output_dir / "latest.pt")
            print(json.dumps({"event": "checkpoint", "epoch": epoch, "update": global_update}), flush=True)
        dist.barrier()

    if args.smoke_steps:
        terminal = {
            "schema": "transfer_dg_wfp_full2800_final_ddp_smoke_terminal_v1",
            "status": "smoke_complete",
            "world_size": world_size,
            "updates": global_update,
            "last_physical_loss": last_physical,
            "last_cpml_loss": last_cpml,
            "elapsed_s": time.time() - started,
            "validation_opened": False,
            "test_id_opened": False,
        }
    else:
        checkpoint_path = args.output_dir / "latest.pt"
        terminal = {
            "schema": "transfer_dg_wfp_full2800_final_ddp_terminal_v1",
            "status": "complete",
            "world_size": world_size,
            "epochs": args.epochs,
            "updates": global_update,
            "record_count": len(positions),
            "latest_checkpoint": str(checkpoint_path.resolve()),
            "latest_checkpoint_sha256": sha256(checkpoint_path) if is_rank0 else None,
            "elapsed_s": time.time() - started,
            "all_train_roles_used_for_gradients": True,
            "model_input_future_wavefield_frames": 0,
            "validation_opened": False,
            "test_id_opened": False,
        }
    dist.barrier()
    if is_rank0:
        atomic_json(terminal, args.output_dir / "terminal.json")
    collection.close()
    travel.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
