#!/usr/bin/env python
"""Balanced V3 refinement with robust relative losses and rejection rollback."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES, V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.data.pilot import (
    PilotBatchDataset,
    build_pilot_schedule,
    make_pilot_loader,
)
from grouped_ufno_mionet_v3.training.checkpoint import CHECKPOINT_FORMAT, save_checkpoint_atomic
from grouped_ufno_mionet_v3.training.curriculum import StageDecision
from grouped_ufno_mionet_v3.training.refinement_control import (
    RefinementControlState,
    advance_refinement_control,
)
from scripts.train_grouped_v3 import build_model
from scripts.train_grouped_v3_curriculum import (
    accumulate_curriculum_update,
    family_validation_scores,
    rejection_recovery,
)
from scripts.train_grouped_v3_pilot import (
    _atomic_hardlink,
    _atomic_json,
    _prepare_validation_batches,
    _to_device,
    _validate_artifact_disk,
    _weights,
    combine_validation_metrics,
    evaluate_pilot_batch,
    load_normalizer,
    pilot_forward_loss,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def select_matching_validation(
    selected_checkpoint: Path,
    reports: list[Mapping[str, object]],
) -> Mapping[str, object]:
    selected = selected_checkpoint.resolve()
    matches = [
        report
        for report in reports
        if report.get("checkpoint")
        and Path(str(report["checkpoint"])).expanduser().resolve() == selected
    ]
    if len(matches) != 1:
        raise ValueError("selected curriculum checkpoint must match exactly one validation report")
    return matches[0]


def accept_refinement_checkpoint(
    *,
    anchor_family_scores: Mapping[str, float],
    candidate_family_scores: Mapping[str, float],
    anchor_aggregate: float,
    candidate_aggregate: float,
    family_regression_tolerance: float,
    anchor_head_consistency: float | None = None,
    candidate_head_consistency: float | None = None,
    head_consistency_regression_tolerance: float = 0.0,
) -> StageDecision:
    allowed = set(ALLOWED_MEDIUM_TYPES)
    if set(anchor_family_scores) != allowed or set(candidate_family_scores) != allowed:
        raise ValueError("refinement scores must contain exactly three medium families")
    if family_regression_tolerance < 0:
        raise ValueError("family regression tolerance must be nonnegative")
    values = [
        float(anchor_aggregate),
        float(candidate_aggregate),
        *(float(value) for value in anchor_family_scores.values()),
        *(float(value) for value in candidate_family_scores.values()),
    ]
    if not all(np.isfinite(value) and value >= 0 for value in values):
        return StageDecision(False, ("refinement metrics contain nonfinite values",))
    failures: list[str] = []
    if float(candidate_aggregate) >= float(anchor_aggregate):
        failures.append("balanced validation score did not improve")
    for family in ALLOWED_MEDIUM_TYPES:
        limit = float(anchor_family_scores[family]) * (1.0 + family_regression_tolerance)
        if float(candidate_family_scores[family]) > limit:
            failures.append(f"family regression tolerance exceeded for {family}")
    if (anchor_head_consistency is None) != (candidate_head_consistency is None):
        raise ValueError("both refinement head-consistency metrics must be provided")
    if anchor_head_consistency is not None:
        if head_consistency_regression_tolerance < 0:
            raise ValueError("head-consistency regression tolerance must be nonnegative")
        limit = float(anchor_head_consistency) * (
            1.0 + float(head_consistency_regression_tolerance)
        )
        if float(candidate_head_consistency) > limit:
            failures.append("dual-head consistency regression tolerance exceeded")
    return StageDecision(not failures, tuple(failures))


def _load_parent(parent_dir: Path, expected_manifest_digest: str) -> dict[str, object]:
    terminal = _load_json(parent_dir / "terminal_report.json")
    identity = _load_json(parent_dir / "run_identity.json")
    if terminal.get("status") != "complete":
        raise ValueError("stable refinement parent is not complete")
    if terminal.get("run_digest") != identity.get("run_digest"):
        raise ValueError("stable refinement parent run identity mismatch")
    if identity.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("stable refinement parent manifest mismatch")
    selected = Path(str(terminal.get("selected_checkpoint", ""))).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(selected)
    parent_root = parent_dir.resolve()
    if selected != parent_root and parent_root not in selected.parents:
        raise ValueError("selected refinement parent escapes its artifact directory")
    validation_paths = []
    if (parent_dir / "best_validation.json").is_file():
        validation_paths.append(parent_dir / "best_validation.json")
    validation_paths.extend(sorted(parent_dir.glob("*/best_validation.json")))
    reports = [_load_json(path) for path in validation_paths]
    validation_report = select_matching_validation(selected, reports)
    validation = validation_report.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("refinement parent validation metrics are missing")
    score = float(validation.get("score", float("nan")))
    if not np.isfinite(score) or not np.isclose(
        score, float(terminal.get("selected_validation_score", float("nan"))), rtol=0, atol=1e-10
    ):
        raise ValueError("refinement parent validation score mismatch")
    payload = torch.load(selected, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("refinement parent checkpoint format mismatch")
    if payload.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("refinement parent checkpoint manifest mismatch")
    if payload.get("config_digest") != terminal.get("run_digest"):
        raise ValueError("refinement parent checkpoint run mismatch")
    return {
        "artifact_dir": str(parent_root),
        "run_digest": str(terminal["run_digest"]),
        "checkpoint": str(selected),
        "checkpoint_sha256": _sha256(selected),
        "validation": dict(validation),
        "validation_score": score,
    }


def _load_config(path: Path) -> tuple[dict[str, object], str]:
    encoded = path.read_bytes()
    config = yaml.safe_load(encoded)
    if not isinstance(config, dict):
        raise ValueError("stable refinement config is invalid")
    return config, hashlib.sha256(encoded).hexdigest()


def _make_optimizer(model, values: Mapping[str, object], learning_rate: float):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(values["weight_decay"]),
    )


def refinement_control_values(
    config: Mapping[str, object],
) -> tuple[str, int, int]:
    values = config.get("training_control")
    if values is None:
        return "legacy_guarded", 1, 0
    if not isinstance(values, Mapping):
        raise ValueError("training_control must be a mapping")
    mode = str(values.get("mode", ""))
    if mode not in {"legacy_guarded", "continuous"}:
        raise ValueError("training_control mode is invalid")
    lr_patience = int(values.get("lr_patience_epochs", 1))
    early_patience = int(values.get("early_stopping_patience_epochs", 0))
    if lr_patience < 1 or early_patience < 0:
        raise ValueError("training_control patience values are invalid")
    return mode, lr_patience, early_patience


def reduce_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    factor: float,
    minimum: float,
) -> float:
    if not 0.0 < factor < 1.0 or minimum < 0.0:
        raise ValueError("learning-rate reduction values are invalid")
    if not optimizer.param_groups:
        raise ValueError("optimizer has no parameter groups")
    learning_rates: list[float] = []
    for group in optimizer.param_groups:
        reduced = max(float(minimum), float(group["lr"]) * float(factor))
        group["lr"] = reduced
        learning_rates.append(reduced)
    if not np.allclose(learning_rates, learning_rates[0], rtol=0.0, atol=0.0):
        raise ValueError("refinement optimizer parameter groups have different learning rates")
    return learning_rates[0]


def run_refinement(
    config_path: Path,
    *,
    artifact_dir: Path,
    device_name: str,
    benchmark_steps: int = 0,
) -> dict[str, object]:
    config, config_digest = _load_config(config_path)
    base_path = (config_path.parent.parent.parent / str(config["base_config"])).resolve()
    if not base_path.is_file():
        base_path = Path(str(config["base_config"])).resolve()
    base_config = V3Config.from_yaml(base_path)
    manifest = build_manifest(base_config.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": base_config.data.expected_train_records,
            "validation": base_config.data.expected_validation_records,
        },
    )
    parent = _load_parent(Path(str(config["parent_artifact_dir"])).resolve(), manifest.digest)
    run_digest = hashlib.sha256(
        json.dumps(
            {"config_digest": config_digest, "parent": parent},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf8")
    ).hexdigest()
    identity = {
        "schema": "grouped_v3_stable_refinement_v1",
        "manifest_digest": manifest.digest,
        "config_digest": config_digest,
        "run_digest": run_digest,
        "parent": parent,
        "benchmark_steps": int(benchmark_steps),
    }
    _validate_artifact_disk(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    identity_path = artifact_dir / "run_identity.json"
    if identity_path.exists():
        if _load_json(identity_path) != identity:
            raise ValueError("existing stable refinement identity mismatch")
        if (artifact_dir / "metrics.jsonl").exists():
            raise RuntimeError("stable refinement artifact already contains metrics")
    else:
        _atomic_json(identity, identity_path)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA stable refinement requested but unavailable")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(base_config).to(device)
    parent_checkpoint = Path(str(parent["checkpoint"])).resolve()
    parent_payload = torch.load(parent_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(parent_payload["model_state"], strict=True)
    normalizer = load_normalizer(base_config, manifest.digest)
    validation_exact, validation_interpolated = _prepare_validation_batches(base_config, manifest)
    weights = _weights(base_config)

    epochs = int(config["epochs"])
    configured_steps_per_epoch = int(config["steps_per_epoch"])
    total_steps = int(benchmark_steps) if benchmark_steps else epochs * configured_steps_per_epoch
    steps_per_epoch = total_steps if benchmark_steps else configured_steps_per_epoch
    schedule = build_pilot_schedule(manifest, split="train", steps=total_steps, seed=seed)
    dataset = PilotBatchDataset(
        base_config.data.source_h5,
        manifest,
        split="train",
        schedule=schedule,
        continuous_fraction=float(config["continuous_fraction"]),
        query_points=int(config["query_points_per_record"]),
        seed=seed,
    )
    loader = make_pilot_loader(
        dataset,
        workers=int(config["workers"]),
        prefetch_factor=int(config["prefetch_factor"]),
        pin_memory=device.type == "cuda",
    )
    optimizer_values = config["optimizer"]
    recovery_values = config["rejection_recovery"]
    acceptance_values = config["acceptance"]
    assert isinstance(optimizer_values, Mapping)
    assert isinstance(recovery_values, Mapping)
    assert isinstance(acceptance_values, Mapping)
    control_mode, lr_patience_epochs, early_stopping_patience_epochs = (
        refinement_control_values(config)
    )
    learning_rate = float(optimizer_values["learning_rate"])
    optimizer = _make_optimizer(model, optimizer_values, learning_rate)
    parameters = tuple(model.parameters())
    gradient_groups = model.required_gradient_groups()

    anchor_checkpoint = parent_checkpoint
    anchor_validation = combine_validation_metrics(
        evaluate_pilot_batch(model, validation_exact, normalizer, device=device),
        evaluate_pilot_batch(model, validation_interpolated, normalizer, device=device),
    )
    anchor_scores = family_validation_scores(anchor_validation)
    anchor_aggregate = float(anchor_validation["score"])
    _atomic_hardlink(anchor_checkpoint, artifact_dir / "best.pt")
    _atomic_json(
        {
            "epoch": 0,
            "global_step": 0,
            "checkpoint": str(anchor_checkpoint),
            "validation": anchor_validation,
            "family_scores": anchor_scores,
            "decision": {"accepted": True, "failures": []},
            "source": "parent",
        },
        artifact_dir / "best_validation.json",
    )

    metrics_path = artifact_dir / "metrics.jsonl"
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    started = time.monotonic()
    global_step = 0
    epoch = 0
    consecutive_rejections = 0
    control_state = RefinementControlState()
    latest_checkpoint = anchor_checkpoint
    stopped_early = False
    for _ in range(total_steps):
        wait_started = time.monotonic()
        batch = next(iterator)
        data_wait = time.monotonic() - wait_started
        if Counter(batch.medium_type) != {family: 4 for family in ALLOWED_MEDIUM_TYPES}:
            raise RuntimeError("stable refinement batch lost family balance")
        transfer_started = time.monotonic()
        tensors = _to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_seconds = time.monotonic() - transfer_started
        captured: dict[str, object] = {}

        def closure() -> torch.Tensor:
            result, _, _, _, _ = pilot_forward_loss(
                model,
                tensors,
                normalizer,
                weights,
                phase_energy_fraction=base_config.loss.phase_energy_fraction,
                relative_energy_floor_fraction=float(config["relative_energy_floor_fraction"]),
            )
            captured["result"] = result
            return result.total

        compute_started = time.monotonic()
        update = accumulate_curriculum_update(
            optimizer=optimizer,
            parameters=parameters,
            required_gradient_groups=gradient_groups,
            closures=(closure,),
            record_counts=(len(batch.sample_id),),
            loss_scales=(1.0,),
            effective_records=len(batch.sample_id),
            gradient_clip=float(optimizer_values["gradient_clip"]),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_seconds = time.monotonic() - compute_started
        result = captured["result"]
        global_step += 1
        row = {
            "event": "train_step",
            "global_step": global_step,
            "epoch": epoch,
            "loss": float(update["weighted_loss"]),
            "components": {
                name: float(value.detach()) for name, value in result.unweighted.items()
            },
            "per_frame_relative_l2_max": float(result.per_frame_relative_l2.max().detach()),
            "gradient_norm": float(update["gradient_norm"]),
            "learning_rate": learning_rate,
            "timing_seconds": {
                "data_wait": data_wait,
                "host_to_device": transfer_seconds,
                "forward_backward_update": compute_seconds,
                "total": data_wait + transfer_seconds + compute_seconds,
            },
            "records_per_second": len(batch.sample_id)
            / max(data_wait + transfer_seconds + compute_seconds, 1.0e-9),
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        with metrics_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(row, sort_keys=True), flush=True)
        if global_step % steps_per_epoch:
            continue

        epoch += 1
        validation = combine_validation_metrics(
            evaluate_pilot_batch(model, validation_exact, normalizer, device=device),
            evaluate_pilot_batch(model, validation_interpolated, normalizer, device=device),
        )
        candidate_scores = family_validation_scores(validation)
        decision = accept_refinement_checkpoint(
            anchor_family_scores=anchor_scores,
            candidate_family_scores=candidate_scores,
            anchor_aggregate=anchor_aggregate,
            candidate_aggregate=float(validation["score"]),
            family_regression_tolerance=float(
                acceptance_values["family_regression_tolerance"]
            ),
            anchor_head_consistency=sum(
                float(anchor_validation[mode]["aggregate_head_consistency_relative_l2"])
                for mode in ("exact", "interpolated")
            ),
            candidate_head_consistency=sum(
                float(validation[mode]["aggregate_head_consistency_relative_l2"])
                for mode in ("exact", "interpolated")
            ),
            head_consistency_regression_tolerance=float(
                acceptance_values.get("head_consistency_regression_tolerance", 0.02)
            ),
        )
        checkpoint = save_checkpoint_atomic(
            checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            manifest_digest=manifest.digest,
            config_digest=run_digest,
            metrics={
                "loss": float(update["weighted_loss"]),
                "validation_score": float(validation["score"]),
            },
        )
        validation_report = {
            "epoch": epoch,
            "global_step": global_step,
            "checkpoint": str(checkpoint.resolve()),
            "validation": validation,
            "family_scores": candidate_scores,
            "decision": decision.to_dict(),
        }
        _atomic_json(validation_report, artifact_dir / "validation_latest.json")
        recovery: dict[str, object] | None = None
        should_stop = False
        if control_mode == "continuous":
            action = advance_refinement_control(
                control_state,
                decision,
                lr_patience_epochs=lr_patience_epochs,
                early_stopping_patience_epochs=early_stopping_patience_epochs,
            )
            control_state = action.state
            latest_checkpoint = checkpoint
            rollback_path: Path | None = None
            if action.new_best:
                anchor_checkpoint = checkpoint
                anchor_validation = validation
                anchor_scores = candidate_scores
                anchor_aggregate = float(validation["score"])
                _atomic_hardlink(checkpoint, artifact_dir / "best.pt")
                _atomic_json(validation_report, artifact_dir / "best_validation.json")
            elif action.rollback:
                rollback_path = anchor_checkpoint.resolve()
                rollback_payload = torch.load(
                    rollback_path,
                    map_location=device,
                    weights_only=False,
                )
                model.load_state_dict(rollback_payload["model_state"], strict=True)
                restored_optimizer = _make_optimizer(
                    model,
                    optimizer_values,
                    learning_rate,
                )
                optimizer_state = rollback_payload.get("optimizer_state")
                if optimizer_state is None:
                    raise ValueError("rollback checkpoint has no optimizer state")
                restored_optimizer.load_state_dict(optimizer_state)
                for group in restored_optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer = restored_optimizer
                latest_checkpoint = rollback_path
            if action.reduce_learning_rate:
                learning_rate = reduce_optimizer_learning_rate(
                    optimizer,
                    factor=float(recovery_values["learning_rate_factor"]),
                    minimum=float(recovery_values["minimum_learning_rate"]),
                )
            recovery = {
                "mode": control_mode,
                "rollback_checkpoint": None
                if rollback_path is None
                else str(rollback_path),
                "learning_rate": learning_rate,
                "learning_rate_reduced": action.reduce_learning_rate,
                "plateau_epochs": control_state.plateau_epochs,
                "lr_wait_epochs": control_state.lr_wait_epochs,
                "safety_rollbacks": control_state.safety_rollbacks,
                "learning_rate_reductions": control_state.lr_reductions,
            }
            should_stop = action.stop
        else:
            if decision.accepted:
                anchor_checkpoint = checkpoint
                anchor_validation = validation
                anchor_scores = candidate_scores
                anchor_aggregate = float(validation["score"])
                consecutive_rejections = 0
                latest_checkpoint = checkpoint
                _atomic_hardlink(checkpoint, artifact_dir / "best.pt")
                _atomic_json(validation_report, artifact_dir / "best_validation.json")
            else:
                consecutive_rejections += 1
                rollback, reduced_lr = rejection_recovery(
                    stage_parent=str(parent_checkpoint),
                    accepted_checkpoint=str(anchor_checkpoint),
                    learning_rate=learning_rate,
                    factor=float(recovery_values["learning_rate_factor"]),
                    minimum=float(recovery_values["minimum_learning_rate"]),
                )
                rollback_path = Path(rollback).resolve()
                rollback_payload = torch.load(
                    rollback_path,
                    map_location=device,
                    weights_only=False,
                )
                model.load_state_dict(rollback_payload["model_state"], strict=True)
                learning_rate = reduced_lr
                optimizer = _make_optimizer(model, optimizer_values, learning_rate)
                latest_checkpoint = rollback_path
                recovery = {
                    "rollback_checkpoint": str(rollback_path),
                    "learning_rate": learning_rate,
                    "consecutive_rejections": consecutive_rejections,
                }
            should_stop = (
                not benchmark_steps
                and consecutive_rejections
                >= int(recovery_values["max_consecutive_rejections"])
            )
        _atomic_hardlink(latest_checkpoint, artifact_dir / "latest.pt")
        print(
            json.dumps(
                {"event": "epoch", **validation_report, "recovery": recovery},
                sort_keys=True,
            ),
            flush=True,
        )
        _validate_artifact_disk(artifact_dir)
        if not benchmark_steps and should_stop:
            stopped_early = True
            break

    terminal = {
        "status": "complete",
        "global_step": global_step,
        "epochs": epoch,
        "selected_checkpoint": str(anchor_checkpoint.resolve()),
        "latest_checkpoint": str(latest_checkpoint.resolve()),
        "selected_validation_score": anchor_aggregate,
        "run_digest": run_digest,
        "stopped_early": stopped_early,
        "final_learning_rate": learning_rate,
        "training_control_mode": control_mode,
        "plateau_epochs": control_state.plateau_epochs,
        "safety_rollbacks": control_state.safety_rollbacks,
        "learning_rate_reductions": control_state.lr_reductions,
    }
    _atomic_json(terminal, artifact_dir / "terminal_report.json")
    print(json.dumps({"event": "terminal", **terminal}, sort_keys=True), flush=True)
    return terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    args = parser.parse_args(argv)
    run_refinement(
        Path(args.config).resolve(),
        artifact_dir=Path(args.artifact_dir).resolve(),
        device_name=args.device,
        benchmark_steps=int(args.benchmark_steps),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
