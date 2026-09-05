#!/usr/bin/env python3
"""R53: complete spatio-temporal tail operator on solver records.

Upgrades the per-frame R39 corrector to a complete operator over a whole
64-frame record: the audited 2D corrector runs on every frame, then a
zero-initialized 3D temporal mixer couples the correction sequence, so
loading an R39/R52 checkpoint is function-identical before training.
Training batches are whole records (record-level repeats follow R29A's
1 + Marmousi + q75 + q90 rule); the loss is the audited R26 tail risk loss
with the CVaR proxy now taken within the record.  Evaluation reuses
r25.evaluate with eval batch 64, which walks each record's frames in order.
Sealed R29B, validation, and test data are never read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import train_r29a_late_tail_finetune as r29a
import train_r39_hfs_tail_finetune as r39
import train_r52_spatial_hfs as r52b

r25 = r29a.r25
r26 = r29a.r26

SCRIPT_PATH = Path(__file__).resolve()
TEMPORAL_LR_MULTIPLIER = 10.0
FRAMES_PER_RECORD = 64


class TemporalMixer(nn.Module):
    """Zero-initialized dilated 3D stack coupling a correction sequence."""

    def __init__(self, channels: int = 24):
        super().__init__()
        ch = int(channels)

        def block(cin: int, dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(
                    cin,
                    ch,
                    kernel_size=(5, 3, 3),
                    padding=(2 * dilation, 1, 1),
                    dilation=(dilation, 1, 1),
                ),
                nn.GroupNorm(4, ch),
                nn.GELU(),
            )

        self.blocks = nn.Sequential(block(2, 1), block(ch, 2), block(ch, 4))
        self.out = nn.Conv3d(ch, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        for parameter in self.parameters():
            parameter._r53_temporal_parameter = True

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(sequence))


class R53SpatioTemporalTailNet(r39.HFSTailSpectralResidualUNet):
    """R39 network plus an identity-initialized temporal mixer.

    The batch dimension of ``forward`` must be the ordered time axis of one
    record (training feeds whole records; r25.evaluate with batch 64 does
    the same).
    """

    def __init__(self, *, base_width: int = 32, correction_cap: float = 0.25):
        super().__init__(base_width=base_width, correction_cap=correction_cap)
        # Spatial HFS modules (R52-B) so every frozen init candidate loads:
        # scalar checkpoints (r39v3 / R52-A) broadcast, R52-B loads directly.
        for name, module in list(self.named_children()):
            if name.startswith("hfs_"):
                setattr(
                    self,
                    name,
                    r52b.SpatialPatchHighFrequencyScaling(
                        module.channels, patch_size=module.patch_size
                    ),
                )
        self.temporal = TemporalMixer()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = dict(state_dict)
        for name, parameter in self.named_parameters():
            if not name.startswith("hfs_") or name not in state_dict:
                continue
            value = state_dict[name]
            if value.shape != parameter.shape and value.numel() == parameter.shape[1]:
                state_dict[name] = (
                    value.reshape(1, -1, 1, 1, 1, 1).expand(parameter.shape).contiguous()
                )
        result = nn.Module.load_state_dict(self, state_dict, strict=False, assign=assign)
        bad_missing = [
            key
            for key in result.missing_keys
            if not key.startswith(("hfs_", "temporal."))
        ]
        if strict and (bad_missing or result.unexpected_keys):
            raise RuntimeError(
                "incompatible R53 checkpoint: "
                f"missing={bad_missing}, unexpected={result.unexpected_keys}"
            )
        return result

    def forward(
        self, features: torch.Tensor, *, active: torch.Tensor | None = None
    ) -> torch.Tensor:
        base = super().forward(features, active=active)
        sequence = torch.stack([base, features[:, 0]], dim=0)[None].float()
        delta = self.temporal(sequence)[0, 0].to(dtype=base.dtype)
        if active is not None:
            delta = delta * active[:, None, None]
        correction = torch.clamp(
            base + delta, -self.correction_cap, self.correction_cap
        ).clone()
        correction[:, 0, :] = 0.0
        return correction


class RecordDataset(Dataset):
    """Whole-record samples with R29A record-level repeats and lazy handles."""

    def __init__(self, collection, *, paths: list[Path]):
        if collection.expected_subset != "fit":
            raise ValueError("RecordDataset requires fit caches")
        self.paths = [Path(path) for path in paths]
        self.records = tuple(collection.records)
        self.time_count = int(collection.time_count)
        if self.time_count != FRAMES_PER_RECORD:
            raise RuntimeError("R53 expects 64 stored frames per record")
        metadata = []
        for file_index, local_index in self.records:
            handle = collection.handles[file_index]
            error_square = float(handle["baseline_error_square_norm"][local_index])
            target_square = float(handle["target_square_norm"][local_index])
            baseline = math.sqrt(error_square / max(target_square, 1.0e-30))
            family = str(handle["family"].asstr()[local_index])
            metadata.append((baseline, family))
        values = np.asarray([row[0] for row in metadata], dtype=np.float64)
        self.baseline_q75 = float(np.quantile(values, 0.75))
        self.baseline_q90 = float(np.quantile(values, 0.90))
        mapping: list[int] = []
        for position, (baseline, family) in enumerate(metadata):
            repeats = (
                1
                + int(family == "marmousi")
                + int(baseline >= self.baseline_q75)
                + int(baseline >= self.baseline_q90)
            )
            mapping.extend([position] * repeats)
        self.mapping = tuple(mapping)
        self._handles: dict[int, list[h5py.File]] = {}

    def _handle(self, file_index: int) -> h5py.File:
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = [h5py.File(path, "r") for path in self.paths]
        return self._handles[pid][file_index]

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int):
        file_index, local_index = self.records[self.mapping[int(index)]]
        handle = self._handle(file_index)
        coarse = torch.from_numpy(
            np.asarray(handle["coarse_norm"][local_index], dtype=np.float32)
        )
        truth = torch.from_numpy(
            np.asarray(handle["truth_norm"][local_index], dtype=np.float32)
        )
        static = torch.from_numpy(
            np.asarray(handle["static_features"][local_index], dtype=np.float32)
        )
        time_s = torch.from_numpy(np.asarray(handle["time_s"][:], dtype=np.float32))
        f0 = float(handle["source_f0_hz"][local_index])
        t0 = float(handle["source_t0_s"][local_index])
        mean_energy = float(handle["truth_frame_energy_mean_norm"][local_index])
        frames = coarse.shape[0]
        features = r25.make_dynamic_features(
            coarse,
            static[None].expand(frames, -1, -1, -1),
            time_s=time_s,
            source_f0_hz=torch.full((frames,), f0),
            source_t0_s=torch.full((frames,), t0),
        )
        active = (time_s >= t0).to(dtype=torch.float32)
        energy = torch.full((frames,), mean_energy, dtype=torch.float32)
        return features, coarse, truth, energy, active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hinge-weight", type=float, default=3.0)
    parser.add_argument("--gradient-weight", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=530829)
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
    r29a.set_seed(int(args.seed), rank)

    fit = r25.CacheCollection(args.fit_cache, expected_subset="fit")
    holdout = r25.CacheCollection(args.holdout_cache, expected_subset="holdout")
    if fit.selection_sha256 != holdout.selection_sha256:
        raise RuntimeError("fit and holdout cache selection digests differ")
    dataset = RecordDataset(fit, paths=list(args.fit_cache))
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
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
    )

    init_path = args.init_checkpoint.expanduser().resolve()
    initial = torch.load(init_path, map_location="cpu", weights_only=False)
    config = initial.get("model_config", {})
    base_width = int(config.get("base_width", 32))
    correction_cap = float(config.get("correction_cap", 0.25))
    model = R53SpatioTemporalTailNet(
        base_width=base_width, correction_cap=correction_cap
    )
    model.load_state_dict(initial["model_state_dict"])
    model.to(device)
    model_for_save = model
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    temporal, hfs, base = [], [], []
    for parameter in model.parameters():
        if getattr(parameter, "_r53_temporal_parameter", False):
            temporal.append(parameter)
        elif getattr(parameter, "_r39_hfs_parameter", False):
            hfs.append(parameter)
        else:
            base.append(parameter)
    if not temporal or not hfs or not base:
        raise RuntimeError(
            f"R53 optimizer partition failed: base={len(base)}, hfs={len(hfs)}, "
            f"temporal={len(temporal)}"
        )
    lr = float(args.learning_rate)
    optimizer = torch.optim.AdamW(
        [
            {"params": base, "lr": lr, "weight_decay": float(args.weight_decay)},
            {"params": hfs, "lr": 1.0e-4, "weight_decay": 0.0},
            {
                "params": temporal,
                "lr": lr * TEMPORAL_LR_MULTIPLIER,
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        weight_decay=float(args.weight_decay),
    )
    steps_per_epoch = len(loader) if not args.max_steps_per_epoch else min(
        len(loader), int(args.max_steps_per_epoch)
    )
    total_steps = max(1, steps_per_epoch * int(args.epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        r25.atomic_json(
            {
                "schema": "r53_spatiotemporal_tail_preregistration_v1",
                "status": "frozen_before_development_finetune",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "role": "R28_already_opened_train_holdout_development_only",
                "initial_checkpoint": str(init_path),
                "initial_checkpoint_sha256": r25.sha256_file(init_path),
                "fit_selection_sha256": fit.selection_sha256,
                "architecture": "R39_HFS_net_plus_zero_init_dilated_temporal_mixer",
                "sampling": {
                    "unit": "whole_record_64_frames",
                    "record_repeats": "1 + Marmousi + baseline_q75 + baseline_q90",
                    "dataset_record_instances": len(dataset),
                    "baseline_q75": dataset.baseline_q75,
                    "baseline_q90": dataset.baseline_q90,
                },
                "optimization": {
                    "epochs": int(args.epochs),
                    "records_per_step": world_size,
                    "learning_rate": lr,
                    "hfs_learning_rate": 1.0e-4,
                    "temporal_learning_rate": lr * TEMPORAL_LR_MULTIPLIER,
                    "loss": "R26 tail risk loss with within-record CVaR",
                    "max_steps_per_epoch": int(args.max_steps_per_epoch),
                },
                "success_gate": {
                    "development_mean_lte": 0.05,
                    "development_max_lte": 0.05,
                    "both_required": True,
                },
                "validation_opened": False,
                "test_id_opened": False,
                "script_sha256": r25.sha256_file(SCRIPT_PATH),
            },
            output_dir / "r53_preregistration.json",
        )
    if distributed:
        torch.distributed.barrier()

    started = time.perf_counter()
    global_step = 0
    best_score = math.inf
    if rank == 0:
        initial_metrics = r25.evaluate(
            model_for_save,
            holdout,
            device=device,
            batch_size=FRAMES_PER_RECORD,
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
        r29a.save_checkpoint(
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
        for step, batch in enumerate(loader):
            if args.max_steps_per_epoch and step >= int(args.max_steps_per_epoch):
                break
            features, coarse, truth, energy, active = (
                tensor.squeeze(0).to(device, non_blocking=True) for tensor in batch
            )
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
                    energy,
                    hinge_weight=float(args.hinge_weight),
                    gradient_weight=float(args.gradient_weight),
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            if rank == 0 and global_step % 50 == 0:
                with open(updates_path, "a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "event": "update",
                                "epoch": epoch,
                                "global_step": global_step,
                                "loss": float(loss.detach()),
                                "relative_square": float(
                                    components["relative_square"]
                                ),
                                "parent_relative_square": float(
                                    components["parent_relative_square"]
                                ),
                                "hinge": float(components["hinge"]),
                                "learning_rate": float(
                                    optimizer.param_groups[0]["lr"]
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        if distributed:
            torch.distributed.barrier()
        if rank == 0:
            metrics = r25.evaluate(
                model_for_save,
                holdout,
                device=device,
                batch_size=FRAMES_PER_RECORD,
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
            with open(metrics_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            score = float(aggregate["candidate_max"]) + float(
                aggregate["candidate_mean"]
            )
            if score < best_score:
                best_score = score
                r29a.save_checkpoint(
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
            print(
                json.dumps(
                    {
                        "event": "development_holdout_evaluation",
                        "epoch": epoch,
                        "candidate_mean": aggregate["candidate_mean"],
                        "candidate_max": aggregate["candidate_max"],
                        "best_score": best_score,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if distributed:
            torch.distributed.barrier()
        model.train()

    if rank == 0:
        best_payload = torch.load(
            output_dir / "best.pt", map_location="cpu", weights_only=False
        )
        r25.atomic_json(
            {
                "schema": "r53_spatiotemporal_tail_terminal_v1",
                "status": "complete",
                "best_score": best_score,
                "best_epoch": int(best_payload["epoch"]),
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": str(output_dir / "best.pt"),
                "checkpoint_sha256": r25.sha256_file(output_dir / "best.pt"),
                "script_sha256": r25.sha256_file(SCRIPT_PATH),
                "validation_opened": False,
                "test_id_opened": False,
            },
            output_dir / "terminal.json",
        )
    if distributed:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
