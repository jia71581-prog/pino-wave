#!/usr/bin/env python3
"""Resume epoch 20 mid-boundary, then enable bounded train-only feedback control."""
from __future__ import annotations

import argparse
import json
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

from saved_time_phase_operator_v4.adaptive_pretraining import (  # noqa: E402
    initial_controller_state,
    probe_selection_sha256,
    stable_probe_positions,
    update_controller,
)
from saved_time_phase_operator_v4.coupled_pyramid_moe_wave import (  # noqa: E402
    PyramidMoECoupledWaveOperator,
    parameter_count,
)
from scripts.train_transfer_dg_coupled_mhc_muon_pilot import (  # noqa: E402
    BLOCK,
    FAMILIES,
    loss_terms,
    model_prediction,
    optimizer_for,
)
from scripts.train_transfer_dg_coupled_pyramid_moe_full_ddp4 import (  # noqa: E402
    DirectFullPretrainData,
    learning_rate_factor,
    set_learning_rate,
    stable_block_offset,
)
from scripts.train_transfer_dg_phase_scatter64_full_ddp import (  # noqa: E402
    FullCollection,
    atomic_checkpoint,
    atomic_json,
    sha256,
)
from scripts.train_transfer_dg_wfp_e1 import CacheCollection  # noqa: E402
from scripts.train_transfer_dg_wfp_e1d import TravelCollection  # noqa: E402


CONTROL_METRICS = ("total", "energy", "balanced", "continuity", "cpml")


@torch.inference_mode()
def evaluate_train_probe(
    model: PyramidMoECoupledWaveOperator,
    data: DirectFullPretrainData,
    selected: dict[str, list[int]],
    config: dict,
    device: torch.device,
    rank: int,
    world: int,
) -> dict:
    """Evaluate a fixed train-only probe, with equal work on all four ranks."""

    model.eval()
    family_count = len(FAMILIES)
    block_count = 64 // BLOCK
    sums = torch.zeros(
        family_count, block_count, len(CONTROL_METRICS) + 1,
        device=device, dtype=torch.float64,
    )
    ordered = [
        (family_index, position)
        for family_index, family in enumerate(FAMILIES)
        for position in selected[family]
    ]
    for item_index, (family_index, position) in enumerate(ordered):
        if item_index % world != rank:
            continue
        for block_index, block_start in enumerate(range(0, 64, BLOCK)):
            batch = data.block(position, block_start, device)
            prediction, auxiliary = model_prediction(model, batch)
            terms = loss_terms(prediction, batch["target"], auxiliary, batch, config)
            for metric_index, name in enumerate(CONTROL_METRICS):
                sums[family_index, block_index, metric_index] += terms[name].double()
            sums[family_index, block_index, -1] += 1.0
    dist.all_reduce(sums)
    counts = sums[..., -1]
    if torch.any(counts <= 0.0):
        raise RuntimeError("train-only controller probe has an empty family-frequency cell")
    means = sums[..., :-1] / counts[..., None]
    result = {
        name: means[..., metric_index].cpu().tolist()
        for metric_index, name in enumerate(CONTROL_METRICS)
    }
    result["counts"] = counts.cpu().to(torch.int64).tolist()
    result["validation_opened"] = False
    result["test_id_opened"] = False
    return result


def checkpoint_payload(
    model: PyramidMoECoupledWaveOperator,
    optimizer,
    identity: dict,
    controller_state: dict,
    *,
    epoch: int,
    next_step: int,
    update: int,
) -> dict:
    return {
        "schema": "transfer_dg_coupled_pyramid_moe_adaptive_ddp4_checkpoint_v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "identity": identity,
        "controller_state": controller_state,
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
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError("adaptive pyramid-MoE pretraining requires exactly four DDP ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_rank0 = rank == 0

    prereg = json.loads(args.preregistration.read_text())
    config = prereg["training"]
    controller_config = prereg["controller"]
    bindings = prereg["bindings"]
    bound_paths = {
        "trainer_sha256": Path(__file__),
        "controller_sha256": ROOT / "saved_time_phase_operator_v4/adaptive_pretraining.py",
        "source_trainer_sha256": ROOT / "scripts/train_transfer_dg_coupled_pyramid_moe_full_ddp4.py",
        "pilot_trainer_sha256": ROOT / "scripts/train_transfer_dg_coupled_mhc_muon_pilot.py",
        "model_sha256": ROOT / "saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py",
        "base_model_sha256": ROOT / "saved_time_phase_operator_v4/coupled_mhc_wave.py",
        "manifest_sha256": args.manifest,
        "init_checkpoint_sha256": args.init_checkpoint,
    }
    for key, path in bound_paths.items():
        observed = sha256(path)
        if observed != bindings[key]:
            raise RuntimeError(f"adaptive pretraining binding drift: {key} {observed}")
    if args.max_updates and int(args.max_updates) != int(prereg["smoke"]["max_updates"]):
        raise RuntimeError("unregistered max-updates override")

    if is_rank0:
        if args.output_dir.exists():
            same_run_resume = args.resume.parent.resolve() == args.output_dir.resolve()
            if not same_run_resume or not (args.output_dir / "run_identity.json").exists():
                raise FileExistsError(args.output_dir)
        else:
            args.output_dir.mkdir(parents=True)
    dist.barrier()

    manifest = json.loads(args.manifest.read_text())
    residual = FullCollection(args.residual_cache, manifest)
    base = CacheCollection(args.base_cache, manifest, expected_count=2800)
    travel = TravelCollection(args.travel, expected_count=2800)
    data = DirectFullPretrainData(residual, base, travel)
    if len(residual.records) != 2800:
        raise RuntimeError("adaptive pretraining did not bind all 2800 records")
    family_counts = {
        family: sum(row[3] == family for row in residual.records) for family in FAMILIES
    }
    if any(count <= 0 for count in family_counts.values()):
        raise RuntimeError("adaptive pretraining is missing a medium family")
    base_family_weights = {
        family: len(residual.records) / (len(FAMILIES) * count)
        for family, count in family_counts.items()
    }
    probe_positions = stable_probe_positions(
        residual.records,
        FAMILIES,
        per_family=int(controller_config["probe_records_per_family"]),
        namespace=str(controller_config["probe_namespace"]),
    )
    probe_sha256 = probe_selection_sha256(residual.records, probe_positions)
    if probe_sha256 != controller_config["probe_selection_sha256"]:
        raise RuntimeError("train-only control-probe selection drift")
    probe_ids = {
        family: [residual.records[position][2] for position in probe_positions[family]]
        for family in FAMILIES
    }

    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PyramidMoECoupledWaveOperator(use_mhc=True).to(device)
    initialization = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(initialization["model_state"], strict=True)
    if parameter_count(model) != int(prereg["model"]["parameter_count"]):
        raise RuntimeError("adaptive pretraining parameter census drift")
    optimizer = optimizer_for(model, "muon", config)

    resume = torch.load(args.resume, map_location=device, weights_only=False)
    if resume.get("validation_opened") or resume.get("test_id_opened"):
        raise RuntimeError("resume checkpoint has opened a sealed evaluation split")
    allowed_schemas = {
        "transfer_dg_coupled_pyramid_moe_full_ddp4_checkpoint_v1",
        "transfer_dg_coupled_pyramid_moe_adaptive_ddp4_checkpoint_v1",
    }
    if resume.get("schema") not in allowed_schemas:
        raise RuntimeError("unsupported adaptive-pretraining resume checkpoint")
    model.load_state_dict(resume["model_state"], strict=True)
    optimizer.load_state_dict(resume["optimizer_state"])
    start_epoch = int(resume["epoch"])
    start_step = int(resume["next_step"])
    update = int(resume["update"])
    expected_start_epoch = int(prereg["handoff"]["expected_start_epoch"])
    expected_next_step = int(prereg["handoff"]["expected_next_step"])
    expected_update = int(prereg["handoff"]["expected_update"])
    if start_epoch < expected_start_epoch:
        raise RuntimeError("handoff checkpoint predates the preregistered start epoch")
    if resume.get("schema") == "transfer_dg_coupled_pyramid_moe_adaptive_ddp4_checkpoint_v1":
        controller_state = dict(resume["controller_state"])
    else:
        if (
            start_epoch != expected_start_epoch
            or start_step != expected_next_step
            or update != expected_update
        ):
            raise RuntimeError("initial fixed-to-adaptive handoff cursor differs from preregistration")
        controller_state = initial_controller_state(len(FAMILIES), 64 // BLOCK)

    distributed = DistributedDataParallel(model, device_ids=[local_rank])
    steps_per_epoch = len(residual.records) // world
    epochs = int(config["epochs"])
    total_updates = epochs * steps_per_epoch
    resume_sha256 = sha256(args.resume) if is_rank0 else None
    resume_hashes = [resume_sha256]
    dist.broadcast_object_list(resume_hashes, src=0, device=device)
    resume_sha256 = resume_hashes[0]
    identity = {
        "schema": "transfer_dg_coupled_pyramid_moe_adaptive_epoch20_ddp4_identity_v1",
        "world_size": world,
        "model": "PyramidMoECoupledWaveOperator",
        "optimizer": "Muon+AdamW",
        "parameter_count": parameter_count(model),
        "record_count": len(residual.records),
        "family_counts": family_counts,
        "base_family_weights": base_family_weights,
        "frequency_count": 64,
        "frequency_block": BLOCK,
        "records_per_epoch": len(residual.records),
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "total_updates": total_updates,
        "controller": controller_config,
        "control_probe_sample_ids": probe_ids,
        "control_probe_selection_sha256": probe_sha256,
        "resume": str(args.resume.resolve()),
        "resume_sha256": resume_sha256,
        "resume_epoch": start_epoch,
        "resume_next_step": start_step,
        "resume_update": update,
        "controller_first_update_epoch": start_epoch if start_step == 0 else start_epoch + 1,
        "trainer_sha256": bindings["trainer_sha256"],
        "controller_sha256": bindings["controller_sha256"],
        "model_sha256": bindings["model_sha256"],
        "manifest_sha256": bindings["manifest_sha256"],
        "validation_opened": False,
        "test_id_opened": False,
    }
    if is_rank0:
        identity_path = args.output_dir / "run_identity.json"
        if identity_path.exists():
            existing = json.loads(identity_path.read_text())
            for key in ("trainer_sha256", "controller_sha256", "model_sha256", "manifest_sha256"):
                if existing.get(key) != identity[key]:
                    raise RuntimeError(f"existing adaptive run identity drift: {key}")
        else:
            atomic_json(identity, identity_path)

    metrics_path = args.output_dir / "metrics.jsonl"
    latest_path = args.output_dir / "latest.pt"
    metric_names = (
        "total", "weighted_total", "energy", "balanced", "continuity", "cpml", "router"
    )
    metric_sum = torch.zeros(len(metric_names), device=device, dtype=torch.float64)
    metric_count = 0
    last_metrics: dict[str, float] = {}
    started = time.time()
    stopped_early = False

    for epoch in range(start_epoch, epochs + 1):
        first_step = start_step if epoch == start_epoch else 0
        if first_step == 0:
            probe_metrics = evaluate_train_probe(
                model, data, probe_positions, config, device, rank, world
            )
            controller_state, controller_event = update_controller(
                probe_metrics, controller_state, controller_config, epoch=epoch
            )
            if is_rank0:
                event = {
                    "event": "controller_update",
                    "epoch": epoch,
                    "update": update,
                    "elapsed_s": time.time() - started,
                    "probe": probe_metrics,
                    **controller_event,
                    "validation_opened": False,
                    "test_id_opened": False,
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event, sort_keys=True), flush=True)

        adaptive_weights = np.asarray(controller_state["weights"], dtype=np.float64)
        pairs = []
        for position, row in enumerate(residual.records):
            offset = stable_block_offset(row[2])
            block_index = (offset + epoch - 1) % (64 // BLOCK)
            pairs.append((position, block_index * BLOCK))
        rng = np.random.default_rng(seed + 1009 * epoch)
        rng.shuffle(pairs)
        rank_pairs = pairs[rank::world]
        if len(rank_pairs) != steps_per_epoch:
            raise RuntimeError("adaptive DDP rank schedule length drift")
        distributed.train()
        for step in range(first_step, steps_per_epoch):
            position, block_start = rank_pairs[step]
            batch = data.block(position, block_start, device)
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary = model_prediction(distributed, batch)
            terms = loss_terms(prediction, batch["target"], auxiliary, batch, config)
            terms["router"] = distributed.module.router_auxiliary_loss()
            terms["total"] = terms["total"] + float(config["router_weight"]) * terms["router"]
            family_index = FAMILIES.index(batch["family"])
            block_index = block_start // BLOCK
            control_weight = float(adaptive_weights[family_index, block_index])
            weight = float(base_family_weights[batch["family"]]) * control_weight
            weighted_total = terms["total"] * weight
            weighted_total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            finite = torch.tensor(
                int(torch.isfinite(weighted_total) and torch.isfinite(gradient_norm)),
                device=device, dtype=torch.int32,
            )
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite):
                raise FloatingPointError("non-finite adaptive-pretraining loss or gradient norm")
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
            last_metrics = {name: float(value.detach()) for name, value in values.items()}
            last_metrics["gradient_norm"] = float(gradient_norm.detach())
            last_metrics["control_weight"] = control_weight
            metric_sum += torch.stack([values[name].detach().double() for name in metric_names])
            metric_count += 1

            if update % int(config["log_interval_updates"]) == 0:
                packed = torch.cat(
                    (metric_sum, torch.tensor([metric_count], device=device, dtype=torch.float64))
                )
                dist.all_reduce(packed)
                if is_rank0:
                    count = float(packed[-1])
                    event = {
                        "event": "update",
                        "epoch": epoch,
                        "step": step + 1,
                        "update": update,
                        "elapsed_s": time.time() - started,
                        "controller_weight_min": float(adaptive_weights.min()),
                        "controller_weight_max": float(adaptive_weights.max()),
                    }
                    for metric_index, name in enumerate(metric_names):
                        event[name] = float(packed[metric_index] / max(count, 1.0))
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)
                metric_sum.zero_()
                metric_count = 0

            if update % int(config["checkpoint_interval_updates"]) == 0:
                dist.barrier()
                if is_rank0:
                    atomic_checkpoint(
                        checkpoint_payload(
                            model, optimizer, identity, controller_state,
                            epoch=epoch, next_step=step + 1, update=update,
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
            checkpoint_epoch = epoch if stopped_early else epoch + 1
            checkpoint_next_step = step + 1 if stopped_early else 0
            atomic_checkpoint(
                checkpoint_payload(
                    model, optimizer, identity, controller_state,
                    epoch=checkpoint_epoch,
                    next_step=checkpoint_next_step,
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
                            "controller_state": controller_state,
                        },
                        sort_keys=True,
                    ) + "\n"
                )
        dist.barrier()
        if stopped_early:
            break

    if is_rank0:
        terminal = {
            "schema": "transfer_dg_coupled_pyramid_moe_adaptive_ddp4_terminal_v1",
            "status": "smoke_complete" if stopped_early else "complete",
            "epoch": epoch,
            "update": update,
            "latest_checkpoint": str(latest_path.resolve()),
            "latest_checkpoint_sha256": sha256(latest_path),
            "controller_state": controller_state,
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
