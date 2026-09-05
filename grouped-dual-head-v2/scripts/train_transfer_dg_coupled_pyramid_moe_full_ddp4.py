#!/usr/bin/env python3
"""Four-GPU full-data pretraining for the mHC+Muon pyramid-MoE operator."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
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

from scripts.train_transfer_dg_coupled_mhc_muon_pilot import (  # noqa: E402
    BLOCK,
    FAMILIES,
    loss_terms,
    model_prediction,
    optimizer_for,
)
from scripts.train_transfer_dg_phase_scatter64_full_ddp import (  # noqa: E402
    FullCollection,
    atomic_checkpoint,
    atomic_json,
    sha256,
)
from scripts.train_transfer_dg_wfp_e1 import CacheCollection  # noqa: E402
from scripts.train_transfer_dg_wfp_e1d import TravelCollection  # noqa: E402
from saved_time_phase_operator_v4.coupled_pyramid_moe_wave import (  # noqa: E402
    PyramidMoECoupledWaveOperator,
    parameter_count,
)


def stable_block_offset(sample_id: str) -> int:
    digest = hashlib.sha256(f"pyramid-moe-full-v1:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % (64 // BLOCK)


def learning_rate_factor(update: int, total_updates: int, warmup: int, eta_ratio: float) -> float:
    if update <= warmup:
        return 0.1 + 0.9 * float(update) / max(float(warmup), 1.0)
    fraction = (float(update) - warmup) / max(float(total_updates - warmup), 1.0)
    fraction = min(max(fraction, 0.0), 1.0)
    return eta_ratio + (1.0 - eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * fraction))


def set_learning_rate(optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        initial = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = initial * float(factor)


class DirectFullPretrainData:
    """Read exact cached targets without repeatedly evaluating the frozen parent."""

    def __init__(
        self,
        residual: FullCollection,
        base: CacheCollection,
        travel: TravelCollection,
    ) -> None:
        self.residual = residual
        self.base = base
        self.travel = travel
        self.base_pos = {row[2]: index for index, row in enumerate(base.records)}
        x = np.arange(241, dtype=np.float32) * 10.0 - 200.0
        z = np.arange(221, dtype=np.float32) * 10.0
        self.xx, self.zz = np.meshgrid(x, z)

    def close(self) -> None:
        self.residual.close()
        self.base.close()
        self.travel.close()

    def block(self, position: int, start: int, device: torch.device) -> dict:
        file_index, local, sample_id, family = self.residual.records[position]
        handle = self.residual.handles[file_index]
        base_position = self.base_pos[sample_id]
        frequencies = tuple(range(int(start), int(start) + BLOCK))

        medium = np.asarray(handle["medium"][local], dtype=np.float32)
        source_map = np.asarray(handle["source_map"][local], dtype=np.float32)
        parameters = np.asarray(handle["source_parameters"][local], dtype=np.float32)
        source_wavelet = np.asarray(handle["source_wavelet"][local], dtype=np.float32)
        wavelet_fft = np.fft.rfft(source_wavelet, norm="ortho")
        wavelet_fft /= max(float(np.max(np.abs(wavelet_fft))), 1.0e-12)
        sx, sz, f0, t0 = (float(value) for value in parameters)
        extended = np.zeros((221, 241), dtype=np.float32)
        extended[:201, 20:221] = source_map

        source_values = []
        scalar_values = []
        physical_targets = []
        auxiliary_targets = []
        frequency_values = []
        for frequency in frequencies:
            wave = wavelet_fft[frequency]
            source_values.append(
                np.stack(
                    (
                        extended,
                        extended * float(wave.real),
                        extended * float(wave.imag),
                        np.clip((self.xx - sx) / 2000.0, -1.2, 1.2),
                        np.clip((self.zz - sz) / 2000.0, -0.2, 1.2),
                    ),
                    axis=0,
                ).astype(np.float32)
            )
            physical, auxiliary, frequency_hz = self.base.targets(base_position, frequency)
            physical_targets.append(physical)
            auxiliary_targets.append(auxiliary)
            frequency_values.append(frequency_hz)
            scalar_values.append(
                (
                    frequency_hz / 200.0,
                    f0 / 30.0,
                    t0 / 0.2,
                    float(wave.real),
                    float(wave.imag),
                )
            )

        travel_physical, travel_exterior, _, auxiliary_total = self.travel.read(sample_id)
        medium_tensor = torch.from_numpy(medium)[None, None].expand(
            1, BLOCK, -1, -1, -1
        ).to(device)
        return {
            "sample_id": sample_id,
            "family": family,
            "medium": medium_tensor,
            "source": torch.from_numpy(np.stack(source_values))[None].to(device),
            "scalars": torch.from_numpy(np.asarray(scalar_values, dtype=np.float32))[None].to(device),
            "travel_physical": torch.from_numpy(travel_physical)[None].expand(BLOCK, -1, -1).to(device),
            "travel_exterior": torch.from_numpy(travel_exterior)[None].expand(BLOCK, -1, -1).to(device),
            "frequency_hz": torch.from_numpy(np.asarray(frequency_values, dtype=np.float32)).to(device),
            "target": torch.from_numpy(np.stack(physical_targets)).to(device),
            "auxiliary_target": torch.from_numpy(np.stack(auxiliary_targets)).to(device),
            "target_total": float(handle["target_modeled_total_square"][local]),
            "auxiliary_total": float(auxiliary_total),
        }


def checkpoint_payload(
    model: PyramidMoECoupledWaveOperator,
    optimizer,
    identity: dict,
    *,
    epoch: int,
    next_step: int,
    update: int,
) -> dict:
    return {
        "schema": "transfer_dg_coupled_pyramid_moe_full_ddp4_checkpoint_v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "identity": identity,
        "epoch": int(epoch),
        "next_step": int(next_step),
        "update": int(update),
        "validation_opened": False,
        "test_id_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-cache", type=Path, action="append", required=True)
    parser.add_argument("--base-cache", type=Path, action="append", required=True)
    parser.add_argument("--travel", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError("full pyramid-MoE pretraining requires exactly four DDP ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_rank0 = rank == 0

    prereg = json.loads(args.preregistration.read_text())
    config = prereg["training"]
    bindings = prereg["bindings"]
    bound_paths = {
        "trainer_sha256": Path(__file__),
        "pilot_trainer_sha256": ROOT / "scripts/train_transfer_dg_coupled_mhc_muon_pilot.py",
        "model_sha256": ROOT / "saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py",
        "base_model_sha256": ROOT / "saved_time_phase_operator_v4/coupled_mhc_wave.py",
        "manifest_sha256": args.manifest,
        "init_checkpoint_sha256": args.init_checkpoint,
    }
    for key, path in bound_paths.items():
        if sha256(path) != bindings[key]:
            raise RuntimeError(f"full pretraining binding drift: {key}")
    if args.max_updates and int(args.max_updates) != int(prereg["smoke"]["max_updates"]):
        raise RuntimeError("unregistered max-updates override")

    if is_rank0:
        if args.resume is None and args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=args.resume is not None)
    dist.barrier()

    manifest = json.loads(args.manifest.read_text())
    residual = FullCollection(args.residual_cache, manifest)
    base = CacheCollection(args.base_cache, manifest, expected_count=2800)
    travel = TravelCollection(args.travel, expected_count=2800)
    data = DirectFullPretrainData(residual, base, travel)
    if len(residual.records) != 2800:
        raise RuntimeError("full pretraining did not bind all 2800 records")
    family_counts = {
        family: sum(row[3] == family for row in residual.records) for family in FAMILIES
    }
    if any(count <= 0 for count in family_counts.values()):
        raise RuntimeError("full pretraining is missing a medium family")
    family_weights = {
        family: len(residual.records) / (len(FAMILIES) * count)
        for family, count in family_counts.items()
    }

    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PyramidMoECoupledWaveOperator(use_mhc=True).to(device)
    initialization = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(initialization["model_state"], strict=True)
    if parameter_count(model) != int(prereg["model"]["parameter_count"]):
        raise RuntimeError("full pretraining parameter census drift")
    optimizer = optimizer_for(model, "muon", config)

    start_epoch = 1
    start_step = 0
    update = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch = int(resume["epoch"])
        start_step = int(resume["next_step"])
        update = int(resume["update"])

    distributed = DistributedDataParallel(model, device_ids=[local_rank])
    steps_per_epoch = len(residual.records) // world
    epochs = int(config["epochs"])
    total_updates = epochs * steps_per_epoch
    identity = {
        "schema": "transfer_dg_coupled_pyramid_moe_full_ddp4_identity_v1",
        "world_size": world,
        "model": "PyramidMoECoupledWaveOperator",
        "use_mhc": True,
        "optimizer": "Muon+AdamW",
        "parameter_count": parameter_count(model),
        "record_count": len(residual.records),
        "family_counts": family_counts,
        "family_weights": family_weights,
        "frequency_count": 64,
        "frequency_block": BLOCK,
        "records_per_epoch": len(residual.records),
        "frequency_blocks_per_record_per_epoch": 1,
        "complete_frequency_coverage_epochs": 8,
        "rank_batch_records": 1,
        "global_batch_records": world,
        "global_batch_frequencies": world * BLOCK,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_updates": total_updates,
        "checkpoint_interval_updates": int(config["checkpoint_interval_updates"]),
        "checkpoint_retention": "latest_only",
        "target_source": "direct cached train truth",
        "all_train_records_used": True,
        "init_checkpoint": str(args.init_checkpoint.resolve()),
        "init_checkpoint_sha256": bindings["init_checkpoint_sha256"],
        "manifest_sha256": bindings["manifest_sha256"],
        "trainer_sha256": bindings["trainer_sha256"],
        "model_sha256": bindings["model_sha256"],
        "resume": str(args.resume.resolve()) if args.resume else None,
        "max_updates": int(args.max_updates),
        "validation_opened": False,
        "test_id_opened": False,
    }
    if is_rank0:
        atomic_json(identity, args.output_dir / "run_identity.json")

    metrics_path = args.output_dir / "metrics.jsonl"
    latest_path = args.output_dir / "latest.pt"
    metric_names = ("total", "weighted_total", "energy", "balanced", "continuity", "cpml", "router")
    metric_sum = torch.zeros(len(metric_names), device=device, dtype=torch.float64)
    route_probability = torch.zeros(len(FAMILIES), 4, device=device, dtype=torch.float64)
    route_load = torch.zeros_like(route_probability)
    route_entropy = torch.zeros(len(FAMILIES), device=device, dtype=torch.float64)
    route_count = torch.zeros(len(FAMILIES), device=device, dtype=torch.float64)
    segment_count = 0
    last_metrics: dict[str, float] = {}
    started = time.time()
    stopped_early = False

    for epoch in range(start_epoch, epochs + 1):
        pairs = []
        for position, row in enumerate(residual.records):
            offset = stable_block_offset(row[2])
            block_index = (offset + epoch - 1) % (64 // BLOCK)
            pairs.append((position, block_index * BLOCK))
        rng = np.random.default_rng(seed + 1009 * epoch)
        rng.shuffle(pairs)
        rank_pairs = pairs[rank::world]
        if len(rank_pairs) != steps_per_epoch:
            raise RuntimeError("DDP rank schedule length drift")
        distributed.train()
        first_step = start_step if epoch == start_epoch else 0
        for step in range(first_step, steps_per_epoch):
            position, block_start = rank_pairs[step]
            batch = data.block(position, block_start, device)
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary = model_prediction(distributed, batch)
            terms = loss_terms(prediction, batch["target"], auxiliary, batch, config)
            terms["router"] = distributed.module.router_auxiliary_loss()
            terms["total"] = terms["total"] + float(config["router_weight"]) * terms["router"]
            weight = float(family_weights[batch["family"]])
            weighted_total = terms["total"] * weight
            weighted_total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            finite = torch.tensor(
                int(torch.isfinite(weighted_total) and torch.isfinite(gradient_norm)),
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite):
                raise FloatingPointError("non-finite full-pretraining loss or gradient norm")
            update += 1
            set_learning_rate(
                optimizer,
                learning_rate_factor(
                    update,
                    total_updates,
                    int(config["warmup_updates"]),
                    float(config["cosine_eta_ratio"]),
                ),
            )
            optimizer.step()

            values = {
                "total": terms["total"],
                "weighted_total": weighted_total,
                "energy": terms["energy"],
                "balanced": terms["balanced"],
                "continuity": terms["continuity"],
                "cpml": terms["cpml"],
                "router": terms["router"],
            }
            last_metrics = {
                name: float(value.detach()) for name, value in values.items()
            }
            last_metrics["gradient_norm"] = float(gradient_norm.detach())
            metric_sum += torch.stack([values[name].detach().double() for name in metric_names])
            family_index = FAMILIES.index(batch["family"])
            routes = distributed.module.route_statistics()
            for expert in range(4):
                route_probability[family_index, expert] += routes[f"route_probability_{expert}"].double()
                route_load[family_index, expert] += routes[f"route_load_{expert}"].double()
            route_entropy[family_index] += routes["route_entropy"].double()
            route_count[family_index] += 1.0
            segment_count += 1

            if update % int(config["log_interval_updates"]) == 0:
                packed = torch.cat(
                    (
                        metric_sum,
                        route_probability.flatten(),
                        route_load.flatten(),
                        route_entropy,
                        route_count,
                        torch.tensor([segment_count], device=device, dtype=torch.float64),
                    )
                )
                dist.all_reduce(packed)
                if is_rank0:
                    cursor = 0
                    count = float(packed[-1])
                    event = {
                        "event": "update",
                        "epoch": epoch,
                        "step": step + 1,
                        "update": update,
                        "elapsed_s": time.time() - started,
                    }
                    for name in metric_names:
                        event[name] = float(packed[cursor] / max(count, 1.0)); cursor += 1
                    probabilities = packed[cursor : cursor + 16].reshape(4, 4); cursor += 16
                    loads = packed[cursor : cursor + 16].reshape(4, 4); cursor += 16
                    entropies = packed[cursor : cursor + 4]; cursor += 4
                    counts = packed[cursor : cursor + 4]
                    event["routes"] = {
                        family: {
                            "probability": [float(value) for value in probabilities[index] / counts[index].clamp_min(1.0)],
                            "load": [float(value) for value in loads[index] / counts[index].clamp_min(1.0)],
                            "entropy": float(entropies[index] / counts[index].clamp_min(1.0)),
                        }
                        for index, family in enumerate(FAMILIES)
                    }
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)
                metric_sum.zero_(); route_probability.zero_(); route_load.zero_()
                route_entropy.zero_(); route_count.zero_(); segment_count = 0

            checkpoint_due = update % int(config["checkpoint_interval_updates"]) == 0
            if checkpoint_due:
                dist.barrier()
                if is_rank0:
                    atomic_checkpoint(
                        checkpoint_payload(
                            model,
                            optimizer,
                            identity,
                            epoch=epoch,
                            next_step=step + 1,
                            update=update,
                        ),
                        latest_path,
                    )
                dist.barrier()

            if args.max_updates and update >= int(args.max_updates):
                stopped_early = True
                break

        start_step = 0
        dist.barrier()
        if is_rank0:
            atomic_checkpoint(
                checkpoint_payload(
                    model,
                    optimizer,
                    identity,
                    epoch=epoch + (0 if stopped_early else 1),
                    next_step=0,
                    update=update,
                ),
                latest_path,
            )
            with metrics_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "epoch_complete" if not stopped_early else "smoke_stop",
                            "epoch": epoch,
                            "update": update,
                            "elapsed_s": time.time() - started,
                            "last_metrics": last_metrics,
                        },
                        sort_keys=True,
                    ) + "\n"
                )
        dist.barrier()
        if stopped_early:
            break

    if is_rank0:
        terminal = {
            "schema": "transfer_dg_coupled_pyramid_moe_full_ddp4_terminal_v1",
            "status": "smoke_complete" if stopped_early else "complete",
            "epoch": epoch,
            "update": update,
            "latest_checkpoint": str(latest_path.resolve()),
            "latest_checkpoint_sha256": sha256(latest_path),
            "elapsed_s": time.time() - started,
            "last_metrics": last_metrics,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(terminal, args.output_dir / "terminal.json")
    data.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
