#!/usr/bin/env python3
"""Four-rank final phase/scatter-64 correction training on all train records."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.phase_carrier import travel_phase_carrier  # noqa: E402
from saved_time_phase_operator_v4.phase_scatter import (  # noqa: E402
    PhaseScatterCorrectionOperator,
    combine_phase_scatter,
    parameter_count,
)


FAMILIES = ("uniform", "layered", "anomaly", "marmousi")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class FullCollection:
    def __init__(self, paths: list[Path], manifest: dict) -> None:
        self.handles = []
        self.records = []
        selection = str(manifest["selection_sha256"])
        for file_index, path in enumerate(paths):
            summary = json.loads(path.with_suffix(path.suffix + ".summary.json").read_text())
            if summary.get("status") != "complete" or summary.get("smoke"):
                raise RuntimeError(f"invalid full correction cache: {path}")
            handle = h5py.File(path, "r", swmr=True)
            self.handles.append(handle)
            if handle.attrs.get("status") != "complete" or handle.attrs.get("manifest_selection_sha256") != selection:
                raise RuntimeError(f"full correction cache binding drift: {path}")
            for local in range(len(handle["sample_id"])):
                self.records.append((
                    file_index, local, str(handle["sample_id"].asstr()[local]),
                    str(handle["family"].asstr()[local]),
                ))
        if len(self.records) != 2800:
            raise RuntimeError("full correction cache census mismatch")
        self.frequency_hz = np.asarray(self.handles[0]["frequency_hz"], dtype=np.float32)
        self.static_cache = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.close()

    def static(self, position: int) -> dict:
        if position not in self.static_cache:
            file_index, local, sample_id, family = self.records[position]
            handle = self.handles[file_index]
            wavelet = np.asarray(handle["source_wavelet"][local], dtype=np.float32)
            wavelet_fft = np.fft.rfft(wavelet, norm="ortho")
            wavelet_fft /= max(float(np.max(np.abs(wavelet_fft))), 1e-12)
            self.static_cache[position] = {
                "sample_id": sample_id, "family": family,
                "medium": np.asarray(handle["medium"][local], dtype=np.float32),
                "source_map": np.asarray(handle["source_map"][local], dtype=np.float32),
                "wavelet_fft": wavelet_fft,
                "parameters": np.asarray(handle["source_parameters"][local], dtype=np.float32),
                "travel": np.asarray(handle["travel_physical_s"][local], dtype=np.float32),
                "target_total": float(handle["target_modeled_total_square"][local]),
            }
        return self.static_cache[position]

    def build(self, position: int, frequency: int, device: torch.device) -> dict:
        file_index, local, _, _ = self.records[position]
        handle = self.handles[file_index]
        row = self.static(position)
        source_map = row["source_map"]
        wave = row["wavelet_fft"][frequency]
        sx, sz, f0, t0 = (float(value) for value in row["parameters"])
        x = np.arange(241, dtype=np.float32) * 10.0 - 200.0
        z = np.arange(221, dtype=np.float32) * 10.0
        xx, zz = np.meshgrid(x, z)
        extended = np.zeros((221, 241), dtype=np.float32)
        extended[:201, 20:221] = source_map
        source = np.stack((
            extended, extended * float(wave.real), extended * float(wave.imag),
            np.clip((xx - sx) / 2000.0, -1.2, 1.2),
            np.clip((zz - sz) / 2000.0, -0.2, 1.2),
        ), axis=0).astype(np.float32)
        scalars = np.asarray((
            float(self.frequency_hz[frequency]) / 200.0, f0 / 30.0, t0 / 0.2,
            float(wave.real), float(wave.imag),
        ), dtype=np.float32)
        return {
            "medium": torch.from_numpy(row["medium"])[None].to(device),
            "source": torch.from_numpy(source)[None].to(device),
            "scalars": torch.from_numpy(scalars)[None].to(device),
            "travel": torch.from_numpy(row["travel"])[None].to(device),
            "frequency_hz": torch.tensor([self.frequency_hz[frequency]], device=device),
            "residual": torch.from_numpy(np.asarray(
                handle["residual_coeff_norm"][local, frequency], dtype=np.float32
            ))[None].to(device),
            "target_total": row["target_total"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=0)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError("full correction requires four DDP ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is0 = rank == 0
    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    if sha256(Path(__file__)) != bindings["trainer_sha256"]:
        raise RuntimeError("full correction trainer binding drift")
    manifest = json.loads(args.manifest.read_text())
    if sha256(args.manifest) != bindings["train_manifest_sha256"]:
        raise RuntimeError("train manifest binding drift")
    if is0:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
    dist.barrier()
    collection = FullCollection(args.cache, manifest)
    family = FAMILIES[rank]
    positions = [index for index, row in enumerate(collection.records) if row[3] == family]
    plan = prereg["training"]
    epochs = int(plan["epochs"])
    family_counts = {
        name: sum(int(value) for value in manifest["family_role_counts"][name].values())
        for name in FAMILIES
    }
    if len(positions) != family_counts[family]:
        raise RuntimeError(f"family census mismatch for {family}")
    rank_batch_size = int(plan["rank_batch_size"])
    if rank_batch_size <= 0:
        raise RuntimeError("rank batch size must be positive")
    maximum_pairs = max(family_counts.values()) * 64
    steps_per_epoch = math.ceil(maximum_pairs / rank_batch_size)
    total_updates = epochs * steps_per_epoch
    torch.manual_seed(int(plan["seed"]))
    np.random.seed(int(plan["seed"]))
    model = PhaseScatterCorrectionOperator().to(device)
    distributed = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        distributed.parameters(), lr=float(plan["learning_rate"]),
        weight_decay=float(plan["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(plan["eta_min"])
    )
    identity = {
        "schema": "transfer_dg_phase_scatter64_full_ddp_identity_v1",
        "world_size": world, "rank_family_mapping": dict(enumerate(FAMILIES)),
        "family_counts": family_counts,
        "record_count": 2800, "frequency_count": 64,
        "epochs": epochs, "steps_per_epoch": steps_per_epoch,
        "total_updates": total_updates,
        "rank_batch_size": rank_batch_size,
        "global_batch_size": rank_batch_size * world,
        "parameter_count": parameter_count(model),
        "parent_checkpoint_sha256": bindings["parent_checkpoint_sha256"],
        "trainer_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(args.preregistration),
        "all_train_records_used": True, "model_input_wavefield_frames": 0,
        "validation_opened": False, "test_id_opened": False,
        "max_updates": args.max_updates,
    }
    if is0:
        atomic_json(identity, args.output_dir / "run_identity.json")
    rng = np.random.default_rng(int(plan["seed"]) + 311 + rank)
    started = time.time()
    global_update = 0
    metrics_path = args.output_dir / "metrics.jsonl"
    stop = False
    for epoch in range(1, epochs + 1):
        schedule = np.asarray([
            (position, frequency) for position in positions for frequency in range(64)
        ], dtype=np.int64)
        rng.shuffle(schedule)
        distributed.train()
        for step in range(steps_per_epoch):
            indices = (
                np.arange(rank_batch_size, dtype=np.int64) + step * rank_batch_size
            ) % len(schedule)
            pairs = schedule[indices]
            items = [
                collection.build(int(position), int(frequency), device)
                for position, frequency in pairs
            ]
            medium = torch.cat([item["medium"] for item in items])
            source = torch.cat([item["source"] for item in items])
            scalars = torch.cat([item["scalars"] for item in items])
            travel = torch.cat([item["travel"] for item in items])
            frequency_hz = torch.cat([item["frequency_hz"] for item in items])
            residual = torch.cat([item["residual"] for item in items])
            target_total = torch.tensor(
                [item["target_total"] for item in items],
                device=device,
                dtype=torch.float32,
            )
            phase, scatter = distributed(medium, source, scalars)
            carrier = travel_phase_carrier(travel, frequency_hz)
            correction = combine_phase_scatter(
                torch.zeros_like(residual), phase, scatter, carrier
            )
            error = (correction.float() - residual.float()).square().sum((1, 2, 3))
            physical_loss = (
                64.0 * error / target_total.clamp_min(1e-8)
            ).mean()
            penalty = 1e-6 * (phase.float().square().mean() + scatter.float().square().mean())
            loss = physical_loss + penalty
            finite = torch.tensor(int(bool(torch.isfinite(loss))), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                raise FloatingPointError(f"non-finite loss at update {global_update + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(distributed.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_update += 1
            if global_update % 500 == 0:
                value = physical_loss.detach().float().clone()
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value.div_(world)
                if is0:
                    event = {"event": "update", "epoch": epoch, "update": global_update,
                             "physical_loss": float(value), "elapsed_s": time.time() - started}
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event), flush=True)
            if args.max_updates and global_update >= args.max_updates:
                stop = True
                break
        dist.barrier()
        if is0:
            atomic_checkpoint({
                "model_state": {key: value.detach().cpu() for key, value in distributed.module.state_dict().items()},
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "epoch": epoch, "update": global_update, "identity": identity,
            }, args.output_dir / "latest.pt")
            print(json.dumps({"event": "checkpoint", "epoch": epoch, "update": global_update}), flush=True)
        dist.barrier()
        if stop:
            break
    checkpoint_path = args.output_dir / "latest.pt"
    terminal = {
        "schema": "transfer_dg_phase_scatter64_full_ddp_terminal_v1",
        "status": "smoke_complete" if args.max_updates else "complete",
        "world_size": world, "epochs": epochs,
        "updates": global_update, "record_count": 2800,
        "latest_checkpoint": str(checkpoint_path.resolve()),
        "latest_checkpoint_sha256": sha256(checkpoint_path) if is0 else None,
        "elapsed_s": time.time() - started, "validation_opened": False,
        "test_id_opened": False,
    }
    dist.barrier()
    if is0:
        atomic_json(terminal, args.output_dir / "terminal.json")
    collection.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
