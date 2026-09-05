#!/usr/bin/env python3
"""R46 physical-space residual FNO on the R40 plus Marmousi fit union.

The model is trained only with fit truth.  The already-opened group-disjoint
development split is used for checkpoint selection.  R29B, final validation,
test data, and manuscript files remain frozen.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_union_curriculum(path: Path, fit):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "r46_train_only_union_hard_curriculum_v1":
        raise RuntimeError("unexpected R46 union curriculum schema")
    if str(payload.get("selection_sha256")) != str(fit.selection_sha256):
        raise RuntimeError("R46 curriculum/cache union digest mismatch")
    if list(payload.get("component_selection_sha256", [])) != list(
        fit.component_selection_sha256
    ):
        raise RuntimeError("R46 curriculum component digests mismatch")
    if payload.get("union_identity") != fit.union_identity:
        raise RuntimeError("R46 curriculum union identity mismatch")
    boundary = payload.get("data_boundary", {})
    if any(
        bool(boundary.get(key))
        for key in (
            "r29b_opened",
            "final_validation_opened",
            "test_id_opened",
            "paper_modified",
        )
    ):
        raise RuntimeError("R46 curriculum violates the frozen data boundary")
    records = list(payload.get("records", []))
    if len(records) != len(fit.records):
        raise RuntimeError("R46 curriculum/cache record counts differ")
    weights = np.empty(len(records), dtype=np.float64)
    base_errors = np.empty(len(records), dtype=np.float64)
    for position, record in enumerate(records):
        file_index, local_index = fit.records[position]
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
                    f"R46 curriculum identity mismatch at {position} for {key}: "
                    f"{record.get(key)} != {value}"
                )
        weights[position] = float(record["sampling_weight"])
        base_errors[position] = float(record["base_record_rel_l2"])
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("invalid R46 curriculum weights")
    if not np.all(np.isfinite(base_errors)) or np.any(base_errors < 0):
        raise RuntimeError("invalid R46 curriculum base errors")
    return payload, weights, base_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--r42-script", type=Path, required=True)
    parser.add_argument("--r44-script", type=Path, required=True)
    parser.add_argument("--union-script", type=Path, required=True)
    parser.add_argument("--base-fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--supplement-fit-cache", type=Path, nargs="+", required=True
    )
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--records-per-rank-epoch", type=int, default=256)
    parser.add_argument("--frequency-batch", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--correction-cap", type=float, default=3.0)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--hinge-weight", type=float, default=2.0)
    parser.add_argument("--shape-weight", type=float, default=0.002)
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4501)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--stop-on-pass", action="store_true")
    args = parser.parse_args()

    if min(
        int(args.epochs),
        int(args.records_per_rank_epoch),
        int(args.frequency_batch),
        int(args.eval_batch_size),
        int(args.eval_every),
    ) < 1:
        raise ValueError("R46 count arguments must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("R46 requires CUDA")
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

    r40_path = args.r40_script.expanduser().resolve()
    r42_path = args.r42_script.expanduser().resolve()
    r44_path = args.r44_script.expanduser().resolve()
    union_path = args.union_script.expanduser().resolve()
    r40 = load_module(r40_path, "r46_r40")
    r42 = load_module(r42_path, "r46_r42")
    r44 = load_module(r44_path, "r46_r44")
    r46_union = load_module(union_path, "r46_union")
    fit = None
    holdout = None
    try:
        fit = r46_union.UnionFrequencyCacheCollection(
            r40,
            [
                [path.expanduser().resolve() for path in args.base_fit_cache],
                [
                    path.expanduser().resolve()
                    for path in args.supplement_fit_cache
                ],
            ],
            expected_subset="fit",
        )
        holdout = r40.FrequencyCacheCollection(
            [path.expanduser().resolve() for path in args.holdout_cache],
            expected_subset="holdout",
        )
        if str(fit.component_selection_sha256[0]) != str(
            holdout.selection_sha256
        ):
            raise RuntimeError("R46 base-fit/holdout source selection mismatch")
        if set(fit.group_ids) & set(holdout.group_ids):
            raise RuntimeError("R46 fit/holdout group leakage")
        if int(fit.retained) != int(holdout.retained):
            raise RuntimeError("R46 fit/holdout DCT geometry mismatch")
        if not np.array_equal(fit.frequency_indices, holdout.frequency_indices):
            raise RuntimeError("R46 fit/holdout frequency mismatch")

        curriculum_path = args.curriculum_manifest.expanduser().resolve()
        curriculum, sampling_weights, base_errors = validate_union_curriculum(
            curriculum_path, fit
        )
        manifest_records = list(curriculum["records"])
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing nonempty R46 output: {output_dir}")
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            torch.distributed.barrier()

        synthesis_array = r44.make_synthesis_matrix(
            r44.GRID_SIZE, int(fit.retained)
        )
        synthesis_error = r44.verify_synthesis_contract(
            synthesis_array, int(fit.retained)
        )
        synthesis_cpu = torch.from_numpy(synthesis_array)
        synthesis_device = synthesis_cpu.to(device)
        hard_records = None
        hard_positions = None
        if rank == 0:
            hard_records, hard_positions = r44.preload_hard_records(
                fit,
                manifest_records,
                base_errors,
                count=min(int(args.probe_count), len(fit.records)),
                synthesis_matrix_cpu=synthesis_cpu,
            )

        dataset = r42.FitRecordDataset(r40, fit)
        sampler = r42.DistributedWeightedRecordSampler(
            sampling_weights,
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
        model_for_save = r44.PhysicalFrequencyResidualFNO(
            r40,
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
        optimizer = torch.optim.AdamW(
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

        scales = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
        identity = {
            "schema": "r46_physical_space_union_curriculum_identity_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": fit.selection_sha256,
            "component_selection_sha256": list(
                fit.component_selection_sha256
            ),
            "fit_record_count": len(fit.records),
            "fit_group_count": len(set(fit.group_ids)),
            "base_fit_record_count": len(fit.components[0].records),
            "supplement_fit_record_count": len(fit.components[1].records),
            "holdout_record_count": len(holdout.records),
            "holdout_group_count": len(set(holdout.group_ids)),
            "world_size": world_size,
            "model": {
                "name": "physical_xz_frequency_residual_fno",
                "parameter_count": r40.parameter_count(model_for_save),
                "width": int(args.width),
                "modes": int(args.modes),
                "blocks": int(args.blocks),
                "correction_cap": float(args.correction_cap),
                "grid_size": int(r44.GRID_SIZE),
                "retained_dct": int(fit.retained),
                "zero_initialized_output": True,
            },
            "optimization": {
                "objective": "sampled_full_wavefield_relative_energy_with_tail_hinge",
                "hard_record_sampling": True,
                "records_per_rank_epoch": int(args.records_per_rank_epoch),
                "global_records_per_epoch": int(args.records_per_rank_epoch)
                * world_size,
                "frequency_batch": int(args.frequency_batch),
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "warmup_epochs": warmup_epochs,
                "tail_weight": float(args.tail_weight),
                "hinge_weight": float(args.hinge_weight),
                "shape_weight": float(args.shape_weight),
                "amp_bfloat16": bool(args.amp),
                "correction_scale_candidates": list(scales),
                "selection_score": "candidate_max_plus_0p1_candidate_mean",
                "seed": int(args.seed),
            },
            "contracts": {
                "dct_synthesis_max_abs_error": synthesis_error,
                "fit_holdout_group_overlap": 0,
                "base_supplement_group_overlap": 0,
            },
            "evidence": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": r42.sha256_file(Path(__file__).resolve()),
                "r40_script": str(r40_path),
                "r40_script_sha256": r42.sha256_file(r40_path),
                "r42_script": str(r42_path),
                "r42_script_sha256": r42.sha256_file(r42_path),
                "r44_script": str(r44_path),
                "r44_script_sha256": r42.sha256_file(r44_path),
                "union_script": str(union_path),
                "union_script_sha256": r42.sha256_file(union_path),
                "curriculum_manifest": str(curriculum_path),
                "curriculum_manifest_sha256": r42.sha256_file(curriculum_path),
            },
            "absolute_goal": {
                "record_relative_l2_mean_lte": 0.05,
                "record_relative_l2_max_lte": 0.05,
            },
            "data_boundary": {
                "fit_truth_used": True,
                "opened_group_disjoint_development_used": True,
                "r29b_opened": False,
                "final_validation_opened": False,
                "test_id_opened": False,
                "paper_modified": False,
            },
        }
        if rank == 0:
            r42.atomic_json(identity, output_dir / "run_identity.json")
        if distributed:
            torch.distributed.barrier()

        frequency_weight_all = torch.from_numpy(
            r40.rfft_weights(r40.TIME_COUNT)[fit.frequency_indices].astype(
                np.float32
            )
        )
        frequency_count = int(fit.frequency_count)
        frequency_generator = torch.Generator(device="cpu")
        frequency_generator.manual_seed(int(args.seed) * 1009 + rank)
        metrics_path = output_dir / "holdout_metrics.jsonl"
        updates_path = output_dir / "updates.jsonl"
        started = time.perf_counter()
        global_step = 0
        best_score = math.inf
        best_epoch = -1
        best_metrics: Mapping[str, Any] | None = None
        stopped_on_pass = False

        if rank == 0:
            initial_scales = r44.evaluate_holdout_scales(
                r40,
                model_for_save,
                holdout,
                scales=scales,
                synthesis_matrix=synthesis_device,
                device=device,
                batch_size=int(args.eval_batch_size),
                amp=bool(args.amp),
            )
            initial = r44.choose_scale(initial_scales)
            if float(initial["correction_scale"]) != 0.0:
                raise RuntimeError("R46 zero model did not select identity")
            identity_difference = max(
                abs(float(row["candidate_rel_l2"]) - float(row["parent_rel_l2"]))
                for row in initial["records"]
            )
            if identity_difference > 1.0e-8:
                raise RuntimeError("R46 zero model changed the parent metric")
            hard_scales = r44.evaluate_hard_fit_scales(
                model_for_save,
                hard_records,
                frequency_weight=frequency_weight_all.numpy(),
                scales=scales,
                synthesis_matrix=synthesis_device,
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
                "fit_hard_scales": hard_scales,
                "maximum_identity_metric_difference": identity_difference,
                "elapsed_seconds": time.perf_counter() - started,
            }
            best_score = float(initial["aggregate"]["candidate_max"]) + 0.1 * float(
                initial["aggregate"]["candidate_mean"]
            )
            best_epoch = 0
            best_metrics = metrics
            checkpoint = {
                "schema": "r46_physical_space_union_curriculum_checkpoint_v1",
                "epoch": 0,
                "global_step": 0,
                "model_state_dict": model_for_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "selection_sha256": fit.selection_sha256,
                "curriculum_manifest_sha256": r42.sha256_file(curriculum_path),
                "selected_correction_scale": 0.0,
                "metrics": metrics,
                "model_config": identity["model"],
            }
            r42.save_checkpoint(checkpoint, output_dir / "best.pt")
            r42.save_checkpoint(checkpoint, output_dir / "latest.pt")
            r42.atomic_json(metrics, output_dir / "initial_holdout.json")
            r42.atomic_json(
                {"epoch": 0, "score": best_score, "metrics": metrics},
                output_dir / "best.json",
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "event": "r46_initial_identity",
                        "candidate_mean": initial["aggregate"]["candidate_mean"],
                        "candidate_max": initial["aggregate"]["candidate_max"],
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
                    _frequency_weight,
                    record_position,
                ) = batch
                indices = torch.randperm(
                    frequency_count, generator=frequency_generator
                )[: min(int(args.frequency_batch), frequency_count)]
                base_selected = base[0].index_select(0, indices).to(
                    device, non_blocking=True
                )
                residual_selected = residual[0].index_select(0, indices).to(
                    device, non_blocking=True
                )
                frequency_selected = frequency_hz[0].index_select(0, indices).to(
                    device, non_blocking=True
                )
                scale_selected = frequency_scale[0].index_select(0, indices).to(
                    device, non_blocking=True
                )
                weight_selected = frequency_weight_all.index_select(0, indices).to(
                    device, non_blocking=True
                )
                static_coefficients = static_norm[0].to(
                    device, non_blocking=True
                ) * static_scale[0].to(device, non_blocking=True)[:, None, None]
                static_spatial = r44.synthesize(
                    static_coefficients[None], synthesis_device
                )[0]
                count = int(len(indices))
                f0_selected = source_f0_hz.to(device, non_blocking=True).expand(count)
                t0_selected = source_t0_s.to(device, non_blocking=True).expand(count)
                features = r44.physical_features(
                    base_selected,
                    static_spatial[None].expand(count, -1, -1, -1),
                    synthesis_matrix=synthesis_device,
                    frequency_hz=frequency_selected,
                    frequency_scale=scale_selected,
                    source_f0_hz=f0_selected,
                    source_t0_s=t0_selected,
                )
                target_spatial = r44.synthesize(
                    residual_selected, synthesis_device
                )
                optimizer.zero_grad(set_to_none=True)
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if bool(args.amp)
                    else nullcontext()
                )
                with amp_context:
                    prediction = model(features)
                error_square = (
                    prediction.float() - target_spatial.float()
                ).square().sum(dim=(1, 2, 3))
                parent_square = target_spatial.float().square().sum(
                    dim=(1, 2, 3)
                )
                target_total = target_square_total[0].to(
                    device, non_blocking=True
                ).float().clamp_min(1.0e-12)
                physical_scale = weight_selected.float() * scale_selected.float().square()
                candidate_contribution = (
                    float(frequency_count)
                    * physical_scale
                    * error_square
                    / target_total
                )
                parent_contribution = (
                    float(frequency_count)
                    * physical_scale
                    * parent_square
                    / target_total
                )
                physical_mean = candidate_contribution.mean()
                tail_count = max(1, int(math.ceil(0.25 * count)))
                physical_tail = torch.topk(
                    candidate_contribution, k=tail_count
                ).values.mean()
                hinge = F.relu(
                    torch.sqrt(candidate_contribution.clamp_min(1.0e-14))
                    - torch.sqrt(parent_contribution.detach().clamp_min(1.0e-14))
                ).square().mean()
                shape = F.smooth_l1_loss(
                    prediction.float(), target_spatial.float(), beta=0.02
                )
                loss = (
                    physical_mean
                    + float(args.tail_weight) * physical_tail
                    + float(args.hinge_weight) * hinge
                    + float(args.shape_weight) * shape
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"nonfinite R46 loss at rank {rank}, step {global_step + 1}"
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
                global_step += 1
                epoch_records += 1
                epoch_loss += float(loss.detach().cpu())
                if rank == 0 and (global_step == 1 or global_step % 50 == 0):
                    event = {
                        "event": "r46_update",
                        "epoch": epoch,
                        "global_step": global_step,
                        "record_position": int(record_position[0]),
                        "loss": float(loss.detach().cpu()),
                        "physical_mean": float(physical_mean.detach().cpu()),
                        "physical_tail": float(physical_tail.detach().cpu()),
                        "hinge": float(hinge.detach().cpu()),
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                    with updates_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)

            if epoch_records != int(args.records_per_rank_epoch):
                raise RuntimeError("R46 epoch record count mismatch")
            scheduler.step()
            mean_loss = r42.reduce_mean(
                epoch_loss / epoch_records, distributed=distributed
            )
            should_evaluate = (
                epoch == 1
                or epoch % int(args.eval_every) == 0
                or epoch == int(args.epochs)
            )
            if distributed:
                torch.distributed.barrier()
            passed = False
            if rank == 0 and should_evaluate:
                scale_evaluations = r44.evaluate_holdout_scales(
                    r40,
                    model_for_save,
                    holdout,
                    scales=scales,
                    synthesis_matrix=synthesis_device,
                    device=device,
                    batch_size=int(args.eval_batch_size),
                    amp=bool(args.amp),
                )
                selected = r44.choose_scale(scale_evaluations)
                selected_scale = float(selected["correction_scale"])
                hard_scales = r44.evaluate_hard_fit_scales(
                    model_for_save,
                    hard_records,
                    frequency_weight=frequency_weight_all.numpy(),
                    scales=scales,
                    synthesis_matrix=synthesis_device,
                    device=device,
                    batch_size=int(args.eval_batch_size),
                    amp=bool(args.amp),
                )
                metrics = {
                    "event": "r46_holdout_evaluation",
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": mean_loss,
                    "selected": selected,
                    "scales": scale_evaluations,
                    "fit_hard_scales": hard_scales,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                checkpoint = {
                    "schema": "r46_physical_space_union_curriculum_checkpoint_v1",
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model_for_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "selection_sha256": fit.selection_sha256,
                    "curriculum_manifest_sha256": r42.sha256_file(curriculum_path),
                    "selected_correction_scale": selected_scale,
                    "metrics": metrics,
                    "model_config": identity["model"],
                }
                r42.save_checkpoint(checkpoint, output_dir / "latest.pt")
                r42.atomic_json(metrics, output_dir / "latest_holdout.json")
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                aggregate = selected["aggregate"]
                score = float(aggregate["candidate_max"]) + 0.1 * float(
                    aggregate["candidate_mean"]
                )
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    best_metrics = metrics
                    r42.save_checkpoint(checkpoint, output_dir / "best.pt")
                    r42.atomic_json(
                        {"epoch": epoch, "score": score, "metrics": metrics},
                        output_dir / "best.json",
                    )
                passed = bool(selected["absolute_goal"]["passed"])
                print(
                    json.dumps(
                        {
                            "event": "r46_holdout_evaluation",
                            "epoch": epoch,
                            "candidate_mean": aggregate["candidate_mean"],
                            "candidate_max": aggregate["candidate_max"],
                            "correction_scale": selected_scale,
                            "fit_hard_max_at_selected_scale": hard_scales[
                                str(selected_scale)
                            ]["aggregate"]["candidate_max"],
                            "train_loss": mean_loss,
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
            if passed and bool(args.stop_on_pass):
                stopped_on_pass = True
                break

        if rank == 0:
            summary = {
                "schema": "r46_physical_space_union_curriculum_summary_v1",
                "status": "complete",
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "stopped_on_pass": stopped_on_pass,
                "absolute_goal_passed": bool(
                    best_metrics
                    and best_metrics["selected"]["absolute_goal"]["passed"]
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "global_step_per_rank": global_step,
                "world_size": world_size,
                "data_boundary": identity["data_boundary"],
                "evidence": {
                    **identity["evidence"],
                    "best_checkpoint": str(output_dir / "best.pt"),
                    "latest_checkpoint": str(output_dir / "latest.pt"),
                    "run_identity": str(output_dir / "run_identity.json"),
                },
            }
            r42.atomic_json(summary, output_dir / "run_summary.json")
            print(
                json.dumps(
                    {
                        "event": "r46_complete",
                        "best_epoch": best_epoch,
                        "best_mean": best_metrics["selected"]["aggregate"][
                            "candidate_mean"
                        ],
                        "best_max": best_metrics["selected"]["aggregate"][
                            "candidate_max"
                        ],
                        "absolute_goal_passed": summary["absolute_goal_passed"],
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
