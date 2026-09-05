#!/usr/bin/env python
"""Dense-decoder L-BFGS probe for a saved-time V6 full-support checkpoint.

This script is intentionally narrow: it loads an already completed V6 checkpoint,
freezes everything except the dense decoder, runs deterministic L-BFGS closure
steps on a fixed full-wavefield batch, and evaluates with the same V6 validation
path used by the long AdamW run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
import torch
import yaml
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.training.checkpoint import (  # noqa: E402
    load_checkpoint,
    save_checkpoint_atomic,
)
from saved_time_phase_operator_v4.data import (  # noqa: E402
    ExactStoredTimeBatchDataset,
    merge_pilot_batches,
    split_pilot_batch,
)
from saved_time_phase_operator_v4.full_support import (  # noqa: E402
    build_full_support_schedule,
)
from saved_time_phase_operator_v4.lbfgs import (  # noqa: E402
    FixedClosureBatch,
    freeze_for_dense_lbfgs,
)
from saved_time_phase_operator_v4.losses import (  # noqa: E402
    apply_hard_causality,
    residual_recovery_loss,
)
from scripts.refine_saved_time_v4_lbfgs import (  # noqa: E402
    _append_jsonl,
    _gpu_snapshot,
)
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer  # noqa: E402
from scripts.train_saved_time_v4_full_support import (  # noqa: E402
    _atomic_hardlink,
    _atomic_json,
    _digest,
    _evaluate,
    _load_context,
    _load_parent_model,
    recovery_time_indices,
    validation_panel_indices,
)


def _completed_run_digest(config: Mapping[str, object], mode: str) -> str:
    identity_path = Path(str(config["artifact_dir"])) / mode / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    return str(identity["run_digest"])


def _load_checkpoint_model(config, checkpoint_path: Path, device: torch.device):
    base, manifest, parent_identity = _load_context(config)
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    digest = _completed_run_digest(config, "run")
    metadata = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=digest,
        map_location=device,
    )
    return model, base, manifest, metadata, digest


def _fixed_closure_batch(config, base, manifest, *, train_macros: int, seed: int):
    schedule = build_full_support_schedule(
        int(base.data.expected_train_records),
        epochs=1,
        macro_records=int(config["macro_records"]),
        macros_per_update=max(int(train_macros), 1),
        seed=int(seed),
    )[: int(train_macros)]
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split="train",
        schedule=schedule,
        query_points=1,
        seed=int(seed),
        time_policy=str(config.get("time_policy", "appearance16")),
        frames_per_record=(
            16 if str(config.get("time_policy", "appearance16")) == "appearance16" else 4
        ),
    )
    batches = tuple(dataset[index] for index in range(len(dataset)))
    return FixedClosureBatch((merge_pilot_batches(batches),))


def _closure_factory(
    *,
    model,
    optimizer,
    fixed: FixedClosureBatch,
    normalizer,
    device: torch.device,
    config,
    microbatch_records: int,
    trace: list[float],
):
    pieces_per_closure = sum(
        len(split_pilot_batch(batch, microbatch_records=int(microbatch_records)))
        for batch in fixed.batches
    )
    if pieces_per_closure <= 0:
        raise ValueError("empty L-BFGS closure")

    def closure():
        fixed.verify(fixed.batches)
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for macro in fixed.batches:
            for micro in split_pilot_batch(
                macro, microbatch_records=int(microbatch_records)
            ):
                tensors = _to_device(micro, device)
                source = tensors["source_parameters"]

                def forward(velocity, source_parameters, source_map, record_to_medium, times, x_m, z_m):
                    prepared = model.prepare_sources(
                        model.encode_medium(velocity, normalizer),
                        source_parameters,
                        source_map,
                        normalizer,
                        record_to_medium=record_to_medium,
                    )
                    return model.dense_normalized_with_coarse(
                        prepared,
                        times,
                        x_m=x_m,
                        z_m=z_m,
                        time_block=1,
                    )

                inputs = (
                    tensors["velocity_mps"],
                    source,
                    tensors["source_map"],
                    tensors["record_to_medium"],
                    tensors["requested_time_s"],
                    tensors["x_m"],
                    tensors["z_m"],
                )
                prediction, coarse = (
                    checkpoint(forward, *inputs, use_reentrant=False)
                    if bool(config["optimizer"].get("full_forward_checkpointing", True))
                    else forward(*inputs)
                )
                target = normalizer.encode_pressure(
                    tensors["dense_target_physical"], source[:, 4]
                )
                if bool(config["loss"].get("hard_causality", False)):
                    prediction = apply_hard_causality(
                        prediction, tensors["requested_time_s"], source[:, 3]
                    )
                    coarse = apply_hard_causality(
                        coarse, tensors["requested_time_s"], source[:, 3]
                    )
                parts = residual_recovery_loss(
                    prediction,
                    coarse,
                    target,
                    time_indices=recovery_time_indices(micro, device),
                    delta_weight=float(config["loss"]["delta"]),
                    temporal_weight=float(config["loss"]["temporal_difference"]),
                    gradient_weight=float(config["loss"]["spatial_gradient"]),
                    spectrum_weight=float(config["loss"]["spectrum"]),
                    delta_energy_floor_fraction=float(
                        config["loss"].get("delta_energy_floor_fraction", 0.0)
                    ),
                )
                loss = parts.total
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite dense L-BFGS closure loss")
                (loss / pieces_per_closure).backward()
                total = total + loss.detach() / pieces_per_closure
        trace.append(float(total.detach().cpu()))
        return total

    return closure


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--train-macros", type=int, default=4)
    parser.add_argument("--microbatch-records", type=int, default=8)
    parser.add_argument("--validation-records", type=int, default=48)
    parser.add_argument("--validation-frames", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--max-eval", type=int, default=4)
    args = parser.parse_args(argv)

    if args.steps <= 0 or args.train_macros <= 0:
        raise ValueError("steps and train-macros must be positive")

    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("dense L-BFGS probe requires CUDA for comparable timing")
    torch.manual_seed(int(config["seed"]) + 17)
    torch.cuda.manual_seed_all(int(config["seed"]) + 17)

    checkpoint_path = Path(args.checkpoint).resolve()
    model, base, manifest, metadata, config_digest = _load_checkpoint_model(
        config, checkpoint_path, device
    )
    params = freeze_for_dense_lbfgs(model)
    normalizer = load_normalizer(base, manifest.digest)
    fixed = _fixed_closure_batch(
        config,
        base,
        manifest,
        train_macros=int(args.train_macros),
        seed=int(config["seed"]) + 17,
    )

    root = Path(args.artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "saved_time_v6_dense_lbfgs_probe_v1",
        "config": config,
        "parent_checkpoint": str(checkpoint_path),
        "parent_epoch": int(metadata.epoch),
        "parent_global_step": int(metadata.global_step),
        "parent_config_digest": config_digest,
        "manifest_digest": manifest.digest,
        "train_sample_signature": fixed.sample_signature,
        "optimizer": {
            "name": "torch.optim.LBFGS",
            "lr": float(args.lr),
            "history_size": int(args.history_size),
            "max_iter": int(args.max_iter),
            "max_eval": int(args.max_eval),
        },
    }
    identity["run_digest"] = _digest(identity)
    _atomic_json(identity, root / "run_identity.json")

    opt = torch.optim.LBFGS(
        params,
        lr=float(args.lr),
        history_size=int(args.history_size),
        max_iter=int(args.max_iter),
        max_eval=int(args.max_eval),
        line_search_fn="strong_wolfe",
    )
    selected_indices = validation_panel_indices(
        validation_records=base.data.expected_validation_records,
        panel_records=int(args.validation_records),
        epoch=1,
        seed=int(config["seed"]),
    )
    baseline = _evaluate(
        model,
        base,
        manifest,
        normalizer,
        device,
        config,
        selected_indices,
        epoch_offset=0,
        time_policy="validation_fixed",
        frames_per_record=int(args.validation_frames),
    )
    baseline_score = float(baseline["aggregate_relative_l2"])
    _append_jsonl(
        root / "optimizer_steps.jsonl",
        {
            "event": "baseline",
            "checkpoint": str(checkpoint_path),
            "metrics": baseline,
            "gpu": _gpu_snapshot(),
        },
    )

    best_score = baseline_score
    best_checkpoint = None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(args.steps) + 1):
        model.train()
        trace: list[float] = []
        closure = _closure_factory(
            model=model,
            optimizer=opt,
            fixed=fixed,
            normalizer=normalizer,
            device=device,
            config=config,
            microbatch_records=int(args.microbatch_records),
            trace=trace,
        )
        opt.step(closure)
        metrics = _evaluate(
            model,
            base,
            manifest,
            normalizer,
            device,
            config,
            selected_indices,
            epoch_offset=0,
            time_policy="validation_fixed",
            frames_per_record=int(args.validation_frames),
        )
        score = float(metrics["aggregate_relative_l2"])
        improvement = (baseline_score - score) / max(baseline_score, 1.0e-16)
        checkpoint = root / "checkpoints" / f"lbfgs_step_{step:03d}.pt"
        save_checkpoint_atomic(
            checkpoint,
            model=model,
            optimizer=None,
            epoch=int(metadata.epoch),
            global_step=int(metadata.global_step),
            manifest_digest=manifest.digest,
            config_digest=identity["run_digest"],
            metrics={"validation": score},
        )
        _atomic_hardlink(checkpoint, root / "latest.pt")
        if score <= best_score:
            best_score = score
            best_checkpoint = str(checkpoint)
            _atomic_hardlink(checkpoint, root / "best.pt")
        report = {
            "event": "lbfgs_step",
            "step": step,
            "closure_evaluations": len(trace),
            "initial_closure_loss": trace[0] if trace else math.nan,
            "final_closure_loss": trace[-1] if trace else math.nan,
            "closure_loss_trace": trace,
            "baseline_aggregate_relative_l2": baseline_score,
            "aggregate_relative_l2": score,
            "relative_improvement_vs_baseline": improvement,
            "metrics": metrics,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu": _gpu_snapshot(),
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint),
        }
        _append_jsonl(root / "optimizer_steps.jsonl", report)
        print(json.dumps(report, sort_keys=True), flush=True)

    terminal = {
        "status": "complete",
        "baseline_aggregate_relative_l2": baseline_score,
        "best_aggregate_relative_l2": best_score,
        "best_checkpoint": best_checkpoint,
        "relative_improvement_vs_baseline": (baseline_score - best_score)
        / max(baseline_score, 1.0e-16),
        "run_digest": identity["run_digest"],
    }
    _atomic_json(terminal, root / "terminal.json")
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
