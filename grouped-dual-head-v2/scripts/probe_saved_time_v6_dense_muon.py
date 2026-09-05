#!/usr/bin/env python
"""Muon-hybrid optimizer probe for a saved-time V6 checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.training.checkpoint import (  # noqa: E402
    load_checkpoint,
    save_checkpoint_atomic,
)
from saved_time_phase_operator_v4.data import (  # noqa: E402
    ExactStoredTimeBatchDataset,
    merge_pilot_batches,
)
from saved_time_phase_operator_v4.full_support import (  # noqa: E402
    build_staged_adamw,
    build_full_support_schedule,
    configure_recovery_stage,
)
from saved_time_phase_operator_v4.muon import build_dense_muon_hybrid  # noqa: E402
from scripts.refine_saved_time_v4_lbfgs import _append_jsonl, _gpu_snapshot  # noqa: E402
from scripts.train_grouped_v3_pilot import load_normalizer  # noqa: E402
from scripts.train_saved_time_v4_full_support import (  # noqa: E402
    _atomic_hardlink,
    _atomic_json,
    _digest,
    _evaluate,
    _gradient_report,
    _load_context,
    _load_parent_model,
    _train_update,
    validation_panel_indices,
)


def _completed_run_digest(config, identity_path: str | None = None) -> str:
    identity_path = Path(identity_path) if identity_path else Path(str(config["artifact_dir"])) / "run" / "run_identity.json"
    return str(json.loads(identity_path.read_text())["run_digest"])


def _load_model(config, checkpoint: Path, device, *, checkpoint_identity: str | None = None):
    base, manifest, parent_identity = _load_context(config)
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    digest = _completed_run_digest(config, checkpoint_identity)
    metadata = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=digest,
        map_location=device,
    )
    return model, base, manifest, metadata, digest


def _update_batches(config, base, manifest, *, updates: int, seed: int):
    macro_records = int(config["macro_records"])
    macros_per_update = int(config["macros_per_update"])
    schedule = build_full_support_schedule(
        int(base.data.expected_train_records),
        epochs=1,
        macro_records=macro_records,
        macros_per_update=macros_per_update,
        seed=int(seed),
    )[: int(updates) * macros_per_update]
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
    batches = []
    for start in range(0, len(dataset), macros_per_update):
        batches.append(
            merge_pilot_batches(
                tuple(dataset[index] for index in range(start, start + macros_per_update))
            )
        )
    return tuple(batches)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-identity")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--optimizer", choices=("muon", "adamw"), default="muon")
    parser.add_argument("--muon-lr", type=float, default=0.002)
    parser.add_argument("--adamw-lr", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--microbatch-records", type=int, default=8)
    parser.add_argument("--validation-records", type=int, default=48)
    parser.add_argument("--validation-frames", type=int, default=32)
    parser.add_argument("--seed-offset", type=int, default=29)
    args = parser.parse_args(argv)

    if args.updates <= 0:
        raise ValueError("updates must be positive")
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Muon probe requires CUDA")
    seed = int(config["seed"]) + int(args.seed_offset)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    checkpoint = Path(args.checkpoint).resolve()
    model, base, manifest, metadata, parent_digest = _load_model(
        config,
        checkpoint,
        device,
        checkpoint_identity=args.checkpoint_identity,
    )
    stage = configure_recovery_stage(
        model,
        epoch=int(metadata.epoch) + 1,
        decoder_only_epochs=int(config["residual_recovery"].get("decoder_only_epochs", 2)),
    )
    if stage.trainable_prefixes != ("dense_decoder",):
        raise ValueError(f"Muon dense probe expected decoder-only stage, got {stage.trainable_prefixes}")

    if args.optimizer == "muon":
        optimizer = build_dense_muon_hybrid(
            model,
            muon_lr=float(args.muon_lr),
            adamw_lr=float(args.adamw_lr),
            weight_decay=float(args.weight_decay),
            momentum=float(args.momentum),
            ns_steps=int(args.ns_steps),
        )
    else:
        optimizer = build_staged_adamw(
            model,
            dense_lr=float(args.adamw_lr),
            geometry_lr=float(config["optimizer"]["geometry_learning_rate"]),
            backbone_lr=float(config["optimizer"]["backbone_learning_rate"]),
            weight_decay=float(args.weight_decay),
        )
    normalizer = load_normalizer(base, manifest.digest)
    batches = _update_batches(config, base, manifest, updates=int(args.updates), seed=seed)
    selected_indices = validation_panel_indices(
        validation_records=base.data.expected_validation_records,
        panel_records=int(args.validation_records),
        epoch=1,
        seed=int(config["seed"]),
    )

    root = Path(args.artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "saved_time_v6_dense_muon_probe_v1",
        "parent_checkpoint": str(checkpoint),
        "parent_epoch": int(metadata.epoch),
        "parent_global_step": int(metadata.global_step),
        "parent_config_digest": parent_digest,
        "manifest_digest": manifest.digest,
        "stage": stage.__dict__,
        "optimizer": {
            "name": str(args.optimizer),
            "muon_lr": float(args.muon_lr),
            "adamw_lr": float(args.adamw_lr),
            "weight_decay": float(args.weight_decay),
            "momentum": float(args.momentum),
            "ns_steps": int(args.ns_steps),
        },
        "updates": int(args.updates),
        "seed": seed,
    }
    identity["run_digest"] = _digest(identity)
    _atomic_json(identity, root / "run_identity.json")

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
            "metrics": baseline,
            "checkpoint": str(checkpoint),
            "gpu": _gpu_snapshot(),
        },
    )

    best_score = baseline_score
    best_checkpoint = None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for index, batch in enumerate(batches, start=1):
        model.train()
        components = _train_update(
            model,
            optimizer,
            batch,
            normalizer,
            device,
            config,
            microbatch_records=int(args.microbatch_records),
        )
        grad_norms = _gradient_report(model, ("dense_decoder",))
        total_norm = float(torch.linalg.vector_norm(torch.tensor(list(grad_norms.values()))))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=float(config["optimizer"]["gradient_clip"]),
        )
        optimizer.step()
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
        ckpt = root / "checkpoints" / f"{args.optimizer}_update_{index:03d}.pt"
        save_checkpoint_atomic(
            ckpt,
            model=model,
            optimizer=None,
            epoch=int(metadata.epoch),
            global_step=int(metadata.global_step) + index,
            manifest_digest=manifest.digest,
            config_digest=identity["run_digest"],
            metrics={"validation": score},
        )
        _atomic_hardlink(ckpt, root / "latest.pt")
        if score <= best_score:
            best_score = score
            best_checkpoint = str(ckpt)
            _atomic_hardlink(ckpt, root / "best.pt")
        report = {
            "event": f"{args.optimizer}_update",
            "update": index,
            "loss_components": components,
            "gradient_norm_before_clip": total_norm,
            "gradient_norms": grad_norms,
            "baseline_aggregate_relative_l2": baseline_score,
            "aggregate_relative_l2": score,
            "relative_improvement_vs_baseline": (baseline_score - score)
            / max(baseline_score, 1.0e-16),
            "metrics": metrics,
            "gpu": _gpu_snapshot(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(ckpt),
        }
        _append_jsonl(root / "optimizer_steps.jsonl", report)
        print(json.dumps(report, sort_keys=True), flush=True)

    terminal = {
        "status": "complete",
        "baseline_aggregate_relative_l2": baseline_score,
        "best_aggregate_relative_l2": best_score,
        "relative_improvement_vs_baseline": (baseline_score - best_score)
        / max(baseline_score, 1.0e-16),
        "best_checkpoint": best_checkpoint,
        "run_digest": identity["run_digest"],
    }
    _atomic_json(terminal, root / "terminal.json")
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
