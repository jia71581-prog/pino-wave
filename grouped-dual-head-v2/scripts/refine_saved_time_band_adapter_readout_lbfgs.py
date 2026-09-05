#!/usr/bin/env python
"""Calibrate only the three band-adapter linear readouts with fixed-batch L-BFGS."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.training.checkpoint import (  # noqa: E402
    load_checkpoint,
    save_checkpoint_atomic,
)
from saved_time_phase_operator_v4.lbfgs import (  # noqa: E402
    freeze_for_band_adapter_readout_lbfgs,
    refinement_gate,
)
from scripts.refine_saved_time_v6_dense_lbfgs import (  # noqa: E402
    _closure_factory,
    _fixed_closure_batch,
)
from scripts.train_grouped_v3_pilot import load_normalizer  # noqa: E402
from scripts.train_saved_time_v4_full_support import (  # noqa: E402
    _atomic_hardlink,
    _atomic_json,
    _digest,
    _evaluate,
    _load_context,
    _load_parent_model,
    validation_panel_indices,
)
from scripts.refine_saved_time_v4_lbfgs import (  # noqa: E402
    _append_jsonl,
    _gpu_snapshot,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-identity", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--train-macros", type=int, default=2)
    parser.add_argument("--microbatch-records", type=int, default=2)
    parser.add_argument("--validation-records", type=int, default=48)
    parser.add_argument("--validation-frames", type=int, default=32)
    parser.add_argument("--validation-seed", type=int, default=401)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--max-eval", type=int, default=5)
    args = parser.parse_args(argv)
    if min(
        args.steps,
        args.train_macros,
        args.microbatch_records,
        args.validation_records,
        args.validation_frames,
        args.max_iter,
        args.max_eval,
    ) <= 0:
        raise ValueError("readout calibration counts must be positive")

    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint_identity = json.loads(args.checkpoint_identity.read_text())
    device = torch.device("cuda")
    base, manifest, parent_identity = _load_context(config)
    if checkpoint_identity.get("manifest_digest") != manifest.digest:
        raise ValueError("readout checkpoint identity manifest mismatch")
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    metadata = load_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=None,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=str(checkpoint_identity["run_digest"]),
        map_location=device,
    )
    parameters = freeze_for_band_adapter_readout_lbfgs(model)
    normalizer = load_normalizer(base, manifest.digest)
    fixed = _fixed_closure_batch(
        config,
        base,
        manifest,
        train_macros=int(args.train_macros),
        seed=int(args.validation_seed) + 17,
    )

    root = args.artifact_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "saved_time_band_adapter_readout_lbfgs_v1",
        "config": config,
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_identity": str(args.checkpoint_identity.resolve()),
        "parent_checkpoint_digest": str(checkpoint_identity["run_digest"]),
        "parent_epoch": int(metadata.epoch),
        "parent_global_step": int(metadata.global_step),
        "manifest_digest": manifest.digest,
        "train_sample_signature": fixed.sample_signature,
        "validation_seed": int(args.validation_seed),
        "validation_records": int(args.validation_records),
        "validation_frames": int(args.validation_frames),
        "trainable_parameter_count": sum(value.numel() for value in parameters),
        "optimizer": {
            "name": "torch.optim.LBFGS",
            "lr": float(args.lr),
            "history_size": int(args.history_size),
            "max_iter": int(args.max_iter),
            "max_eval": int(args.max_eval),
        },
    }
    identity["run_digest"] = _digest(identity)
    identity_path = root / "run_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("readout calibration run identity mismatch")
    if not identity_path.exists():
        _atomic_json(identity, identity_path)

    selected_indices = validation_panel_indices(
        validation_records=base.data.expected_validation_records,
        panel_records=int(args.validation_records),
        epoch=1,
        seed=int(args.validation_seed),
    )
    model.eval()
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
        {"event": "baseline", "metrics": baseline, "checkpoint": str(args.checkpoint)},
    )

    optimizer = torch.optim.LBFGS(
        parameters,
        lr=float(args.lr),
        history_size=int(args.history_size),
        max_iter=int(args.max_iter),
        max_eval=int(args.max_eval),
        line_search_fn="strong_wolfe",
    )
    best_score = baseline_score
    best_metrics = baseline
    best_checkpoint: str | None = None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(args.steps) + 1):
        model.train()
        trace: list[float] = []
        closure = _closure_factory(
            model=model,
            optimizer=optimizer,
            fixed=fixed,
            normalizer=normalizer,
            device=device,
            config=config,
            microbatch_records=int(args.microbatch_records),
            trace=trace,
        )
        optimizer.step(closure)
        model.eval()
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
        gate = refinement_gate(
            parent_score=baseline_score,
            candidate_score=score,
            parent_family=baseline["family_relative_l2"],
            candidate_family=metrics["family_relative_l2"],
            minimum_relative_improvement=0.02,
            family_regression_tolerance=float(
                config["gate"]["family_regression_tolerance"]
            ),
        )
        checkpoint = root / "checkpoints" / f"lbfgs_step_{step:03d}.pt"
        save_checkpoint_atomic(
            checkpoint,
            model=model,
            optimizer=None,
            epoch=int(metadata.epoch),
            global_step=int(metadata.global_step),
            manifest_digest=manifest.digest,
            config_digest=str(identity["run_digest"]),
            metrics={"validation": score},
        )
        _atomic_hardlink(checkpoint, root / "latest.pt")
        if score < best_score:
            best_score = score
            best_metrics = metrics
            best_checkpoint = str(checkpoint)
            _atomic_hardlink(checkpoint, root / "best.pt")
        peak = int(torch.cuda.max_memory_allocated())
        if peak >= float(config["gate"]["maximum_peak_cuda_gib"]) * 1024**3:
            raise RuntimeError("readout L-BFGS exceeded the registered CUDA limit")
        report = {
            "event": "lbfgs_step",
            "step": step,
            "closure_evaluations": len(trace),
            "initial_closure_loss": trace[0] if trace else math.nan,
            "final_closure_loss": trace[-1] if trace else math.nan,
            "metrics": metrics,
            "gate": gate,
            "checkpoint": str(checkpoint),
            "peak_cuda_bytes": peak,
            "gpu": _gpu_snapshot(),
            "elapsed_seconds": time.monotonic() - started,
        }
        _append_jsonl(root / "optimizer_steps.jsonl", report)
        print(json.dumps(report, sort_keys=True), flush=True)

    best_gate = refinement_gate(
        parent_score=baseline_score,
        candidate_score=best_score,
        parent_family=baseline["family_relative_l2"],
        candidate_family=best_metrics["family_relative_l2"],
        minimum_relative_improvement=0.02,
        family_regression_tolerance=float(
            config["gate"]["family_regression_tolerance"]
        ),
    )
    terminal = {
        "status": "complete",
        "baseline_aggregate_relative_l2": baseline_score,
        "best_aggregate_relative_l2": best_score,
        "best_family_relative_l2": best_metrics["family_relative_l2"],
        "best_checkpoint": best_checkpoint,
        "selected_parent": best_checkpoint is None,
        "gate": best_gate,
        "run_digest": identity["run_digest"],
    }
    _atomic_json(terminal, root / "terminal.json")
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
