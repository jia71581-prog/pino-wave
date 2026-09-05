#!/usr/bin/env python
"""Restart the best V4 dense decoder with deterministic batch-48 L-BFGS."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint, save_checkpoint_atomic
from saved_time_phase_operator_v4.data import (
    ExactStoredTimeBatchDataset,
    merge_pilot_batches,
    split_pilot_batch,
)
from saved_time_phase_operator_v4.lbfgs import (
    FixedClosureBatch,
    effective_batch_records,
    freeze_for_dense_lbfgs,
    refinement_gate,
)
from saved_time_phase_operator_v4.losses import band_limited_residual_loss
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_probe import (
    _atomic_hardlink,
    _atomic_json,
    _digest,
    _evaluate,
    _loss,
    _model,
    _schedule,
)


def _gpu_snapshot() -> dict[str, float | None]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().split(",")
        return {"utilization_percent": float(output[0]), "power_w": float(output[1]), "memory_mib": float(output[2])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"utilization_percent": None, "power_w": None, "memory_mib": None}


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _load_bound_model(config, device):
    parent_identity = json.loads(Path(config["parent_identity"]).read_text())
    base = V3Config.from_yaml(parent_identity["config"]["base_config"])
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(manifest, {"train": base.data.expected_train_records, "validation": base.data.expected_validation_records})
    if parent_identity["manifest_digest"] != manifest.digest:
        raise ValueError("L-BFGS parent manifest identity mismatch")
    if "variant_config" in parent_identity:
        variant = ProbeVariant(**parent_identity["variant_config"])
        model = _model(base, manifest, variant).to(device)
    else:
        # Full-support continuation identities record the effective config and
        # keep the original architecture identity one level below it.  Rebuild
        # that exact architecture before loading the requested immutable parent.
        run_config = parent_identity.get(
            "model_config", parent_identity.get("config")
        )
        if not isinstance(run_config, dict):
            raise ValueError("refinement parent identity has no effective config")
        architecture_identity_path = run_config.get("parent_identity")
        if not architecture_identity_path:
            raise ValueError(
                "full-support refinement parent has no architecture identity"
            )
        architecture_identity = json.loads(
            Path(str(architecture_identity_path)).read_text()
        )
        from scripts.train_saved_time_v4_full_support import _load_parent_model

        model = _load_parent_model(
            run_config,
            base,
            manifest,
            architecture_identity,
            device,
        )
    load_checkpoint(
        config["parent_checkpoint"], model=model,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=parent_identity["run_digest"], map_location=device,
    )
    return model, base, manifest, parent_identity


def _fixed_batches(config, base, manifest):
    seed = int(config["seed"])
    train_count = int(config["fixed_train_macros"])
    train_schedule = _schedule(manifest, split="train", pool_steps=train_count, total_steps=train_count, seed=seed)
    train_data = ExactStoredTimeBatchDataset(
        base.data.source_h5, manifest, split="train", schedule=train_schedule,
        query_points=int(config["query_points"]), seed=seed,
    )
    train_batches = tuple(train_data[index] for index in range(len(train_data)))
    validation_count = int(config["validation_macros"])
    validation_schedule = _schedule(
        manifest, split="validation", pool_steps=validation_count,
        total_steps=validation_count, seed=seed + 7919, offset=10_000,
    )
    validation_data = ExactStoredTimeBatchDataset(
        base.data.source_h5, manifest, split="validation", schedule=validation_schedule,
        query_points=int(config["query_points"]), seed=seed + 7919,
    )
    return FixedClosureBatch((merge_pilot_batches(train_batches),)), tuple(
        validation_data[index] for index in range(len(validation_data))
    )


def _closure_factory(model, optimizer, fixed, normalizer, device, config, trace):
    microbatch_records = int(config["microbatch_records"])
    micro_count = sum(len(split_pilot_batch(batch, microbatch_records=microbatch_records)) for batch in fixed.batches)

    def closure():
        fixed.verify(fixed.batches)
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for macro in fixed.batches:
            for micro in split_pilot_batch(macro, microbatch_records=microbatch_records):
                tensors = _to_device(micro, device); source = tensors["source_parameters"]
                prepared = model.prepare_sources(
                    model.encode_medium(tensors["velocity_mps"], normalizer), source,
                    tensors["source_map"], normalizer,
                    record_to_medium=tensors["record_to_medium"],
                )
                prediction = model.dense_normalized(
                    prepared, tensors["requested_time_s"], x_m=tensors["x_m"],
                    z_m=tensors["z_m"], time_block=1,
                )
                target = normalizer.encode_pressure(tensors["dense_target_physical"], source[:, 4])
                loss, _, _, _ = _loss(
                    prediction, target,
                    gradient_weight=float(config["loss"]["spatial_gradient"]),
                    spectrum_weight=float(config["loss"]["spectrum"]),
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite L-BFGS closure loss")
                (loss / micro_count).backward()
                total += loss.detach() / micro_count
        value = float(total)
        trace.append(value)
        return total

    return closure


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda")
    model, base, manifest, parent_identity = _load_bound_model(config, device)
    parameters = freeze_for_dense_lbfgs(model)
    normalizer = load_normalizer(base, manifest.digest)
    fixed, validation_batches = _fixed_batches(config, base, manifest)
    macro_records = len(fixed.batches[0].sample_id)
    effective_batch = effective_batch_records(macro_records=macro_records, accumulated_macros=len(fixed.batches))
    if effective_batch != 48:
        raise ValueError(f"L-BFGS effective batch must be 48, got {effective_batch}")
    parent_report = json.loads(Path(config["parent_best_report"]).read_text())
    parent_metrics = parent_report["metrics"]
    identity = {
        "schema": "saved_time_v4_lbfgs_refinement_v1", "config": config,
        "parent_run_digest": parent_identity["run_digest"],
        "parent_checkpoint": str(Path(config["parent_checkpoint"]).resolve()),
        "manifest_digest": manifest.digest, "effective_batch_records": effective_batch,
        "fixed_train_sample_ids": fixed.sample_signature,
    }
    identity["run_digest"] = _digest(identity)
    root = Path(config["artifact_dir"]) / ("smoke" if args.smoke_steps else "run")
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "run_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("L-BFGS run identity mismatch")
    if not identity_path.exists(): _atomic_json(identity, identity_path)
    opt = config["optimizer"]
    optimizer = torch.optim.LBFGS(
        parameters, lr=float(opt["learning_rate"]), max_iter=int(opt["max_iter"]),
        max_eval=int(opt["max_eval"]), tolerance_grad=float(opt["tolerance_grad"]),
        tolerance_change=float(opt["tolerance_change"]), history_size=int(opt["history_size"]),
        line_search_fn=str(opt["line_search_fn"]),
    )
    steps = int(args.smoke_steps) if args.smoke_steps else int(config["outer_steps"])
    torch.cuda.reset_peak_memory_stats(); best = None
    for step in range(1, steps + 1):
        model.train(); trace=[]
        closure = _closure_factory(model, optimizer, fixed, normalizer, device, config, trace)
        optimizer.step(closure)
        if not trace:
            raise RuntimeError("L-BFGS did not evaluate its closure")
        initial = trace[0]
        metrics = _evaluate(
            model, validation_batches, normalizer, device,
            int(config["validation_microbatch_records"]), float(config["energy_floor_fraction"]),
        )
        score = float(metrics["aggregate_floored_relative_l2"])
        gate = refinement_gate(
            parent_score=float(parent_metrics["aggregate_floored_relative_l2"]), candidate_score=score,
            parent_family=parent_metrics["family_floored_relative_l2"],
            candidate_family=metrics["family_floored_relative_l2"],
            minimum_relative_improvement=float(config["gate"]["minimum_relative_improvement"]),
            family_regression_tolerance=float(config["gate"]["family_regression_tolerance"]),
        )
        peak = int(torch.cuda.max_memory_allocated())
        if peak > float(config["gate"]["maximum_peak_cuda_gib"]) * 1024**3:
            raise RuntimeError("L-BFGS peak CUDA memory exceeded the registered limit")
        checkpoint = root / "checkpoints" / f"step_{step:03d}.pt"
        save_checkpoint_atomic(
            checkpoint, model=model, optimizer=None, epoch=step, global_step=step,
            manifest_digest=manifest.digest, config_digest=identity["run_digest"],
            metrics={"validation": score},
        )
        _atomic_hardlink(checkpoint, root / "latest.pt")
        report = {
            "event": "lbfgs_step", "step": step, "closure_evaluations": len(trace),
            "initial_closure_loss": initial, "final_closure_loss": trace[-1],
            "closure_loss_trace": trace, "metrics": metrics, "gate": gate,
            "peak_cuda_bytes": peak, "gpu": _gpu_snapshot(), "checkpoint": str(checkpoint),
        }
        _append_jsonl(root / "optimizer_steps.jsonl", report)
        if best is None or score < best["metrics"]["aggregate_floored_relative_l2"]:
            best = report; _atomic_hardlink(checkpoint, root / "best.pt"); _atomic_json(best, root / "best.json")
        print(json.dumps(report, sort_keys=True), flush=True)
    final_gate = best["gate"] if best else {"passed": False}
    terminal = {"status": "complete", "best": best, "gate": final_gate, "run_digest": identity["run_digest"]}
    _atomic_json(terminal, root / "terminal.json"); print(json.dumps(terminal, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
