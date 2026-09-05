#!/usr/bin/env python3
"""Train the A3-warm-started snapshot-only stable wave propagator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saved_time_phase_operator_v4.snapshot_propagator import (
    SnapshotOnlyWavePropagator,
    transfer_pretrained_decoder_stack,
)
from saved_time_phase_operator_v4.snapshot_training import (
    SnapshotWindowDataset,
    snapshot_rollout_loss,
    summarize_snapshot_rows,
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _horizon(schedule: list[list[int]], epoch: int) -> int:
    selected = int(schedule[0][1])
    for start, value in schedule:
        if int(epoch) >= int(start):
            selected = int(value)
    return selected


@torch.no_grad()
def _evaluate(
    model,
    loader,
    *,
    horizon: int,
    report_horizons: tuple[int, ...],
    modal_only: bool,
    device: torch.device,
):
    model.eval()
    horizons = tuple(sorted({int(value) for value in report_horizons} | {int(horizon)}))
    if horizons[0] <= 0 or horizons[-1] > int(horizon):
        raise ValueError("validation report horizons must lie in [1,horizon]")
    learned_rows = {value: [] for value in horizons}
    baseline_rows = {value: [] for value in horizons}
    for batch in loader:
        history = batch["wavefield_history"].to(device)
        target = batch["target"][:, :horizon].to(device)
        learned = (
            model.calibrated_modal_rollout(history, horizon)
            if modal_only
            else model(history, horizon)
        )
        baseline = model.modal_baseline(history, horizon)
        for report_horizon in horizons:
            for index, family in enumerate(batch["family"]):
                reference = (
                    target[index, :report_horizon].double().flatten().norm().clamp_min(1e-30)
                )
                learned_rows[report_horizon].append(
                    {
                        "family": family,
                        "relative_l2": float(
                            (
                                learned[index, :report_horizon].double()
                                - target[index, :report_horizon].double()
                            ).flatten().norm()
                            / reference
                        ),
                    }
                )
                baseline_rows[report_horizon].append(
                    {
                        "family": family,
                        "relative_l2": float(
                            (
                                baseline[index, :report_horizon].double()
                                - target[index, :report_horizon].double()
                            ).flatten().norm()
                            / reference
                        ),
                    }
                )
    learned_summary = {
        str(value): summarize_snapshot_rows(learned_rows[value]) for value in horizons
    }
    baseline_summary = {
        str(value): summarize_snapshot_rows(baseline_rows[value]) for value in horizons
    }
    learned_full = dict(learned_summary[str(horizon)])
    baseline_full = dict(baseline_summary[str(horizon)])
    learned_full["by_horizon"] = learned_summary
    baseline_full["by_horizon"] = baseline_summary
    return learned_full, baseline_full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device or config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(args.output_dir or config["artifact_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"non-empty output directory requires --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    model_config = dict(config["model"])
    training = dict(config["training"])
    validation = dict(config["validation"])
    if args.smoke:
        model_config.update(width=8, spectral_rank=4, modes=4, depth=1)
        training.update(epochs=1, samples_per_epoch=3, batch_size=1, workers=0)
        training["horizon_schedule"] = [[0, 1]]
        validation.update(
            records_per_family=1,
            context_ends=[80],
            horizon=1,
            report_horizons=[1],
        )
    model = SnapshotOnlyWavePropagator(**model_config).to(device)
    transfer_report: dict[str, object]
    if args.smoke:
        transfer_report = {"skipped_for_smoke": True, "modal_warmstart_preserved": True}
    else:
        parent_path = Path(config["parent_checkpoint"]).resolve()
        parent = torch.load(
            str(parent_path), map_location="cpu", weights_only=False, mmap=True
        )
        transfer_report = transfer_pretrained_decoder_stack(model, parent["model_state"])
        transfer_report.update(
            parent_checkpoint=str(parent_path), parent_epoch=int(parent["epoch"])
        )

    maximum_horizon = max(int(row[1]) for row in training["horizon_schedule"])
    train_data = SnapshotWindowDataset(
        config["dataset"], split="train",
        history_frames=int(model_config["minimum_history"]),
        rollout_steps=maximum_horizon, seed=seed,
        records_per_family=int(training["records_per_family"]),
        samples_per_epoch=int(training["samples_per_epoch"]),
        minimum_context_end=int(training["minimum_context_end"]),
        maximum_context_end=int(training["maximum_context_end"]),
        maximum_context_fraction=float(training["maximum_context_fraction"]),
        post_peak_cycles=float(training["post_peak_cycles"]),
    )
    validation_data = SnapshotWindowDataset(
        config["dataset"], split="validation",
        history_frames=int(model_config["minimum_history"]),
        rollout_steps=int(validation["horizon"]), seed=seed + 17,
        records_per_family=int(validation["records_per_family"]),
        fixed_context_ends=tuple(int(value) for value in validation["context_ends"]),
        minimum_context_end=int(training["minimum_context_end"]),
        maximum_context_end=int(training["maximum_context_end"]),
        maximum_context_fraction=float(training["maximum_context_fraction"]),
        post_peak_cycles=float(training["post_peak_cycles"]),
        record_selection=str(validation.get("record_selection", "random")),
    )
    workers = int(training["workers"])
    train_loader = DataLoader(
        train_data, batch_size=int(training["batch_size"]), shuffle=False,
        num_workers=workers, persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=1, shuffle=False, num_workers=0
    )
    closure_parameters = list(model.closure.parameters())
    closure_ids = {id(value) for value in closure_parameters}
    adapter_parameters = [
        value for value in model.parameters() if id(value) not in closure_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": float(training["adapter_learning_rate"])},
            {"params": closure_parameters, "lr": float(training["closure_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    latest = output_dir / "latest.pt"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_metric = float(state.get("best_metric", best_metric))

    _atomic_json(
        output_dir / "run_identity.json",
        {
            "schema": "snapshot_only_modal_closure_v2",
            "config": str(config_path),
            "dataset": str(Path(config["dataset"]).resolve()),
            "deployment_inputs": ["wavefield_history", "steps"],
            "observation_protocol": "strict_early_post_onset_only",
            "maximum_context_fraction": float(training["maximum_context_fraction"]),
            "validation_horizons": validation.get("report_horizons", [validation["horizon"]]),
            "modal_only_epochs": int(training.get("modal_only_epochs", 0)),
            "transfer": transfer_report,
            "smoke": bool(args.smoke),
        },
    )
    started = time.time()
    try:
        for epoch in range(start_epoch, int(training["epochs"])):
            train_data.set_epoch(epoch)
            horizon = _horizon(training["horizon_schedule"], epoch)
            modal_only = epoch < int(training.get("modal_only_epochs", 0))
            closure_trainable = (
                not modal_only and epoch >= int(training["freeze_closure_epochs"])
            )
            for parameter in closure_parameters:
                parameter.requires_grad_(closure_trainable)
            model.train()
            running = 0.0
            component_sums: dict[str, float] = {}
            for batch in train_loader:
                history = batch["wavefield_history"].to(device)
                target = batch["target"][:, :horizon].to(device)
                noise = float(training["context_noise_fraction"])
                if noise > 0.0:
                    scale = history.detach().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
                    history = history + noise * scale * torch.randn_like(history)
                optimizer.zero_grad(set_to_none=True)
                prediction = (
                    model.calibrated_modal_rollout(history, horizon)
                    if modal_only
                    else model(history, horizon)
                )
                loss, components = snapshot_rollout_loss(
                    prediction, target, **dict(config["loss"])
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite snapshot rollout loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                optimizer.step()
                global_step += 1
                running += float(loss.detach())
                for name, value in components.items():
                    component_sums[name] = component_sums.get(name, 0.0) + float(value.detach())
            learned, baseline = _evaluate(
                model, validation_loader,
                horizon=int(validation["horizon"]),
                report_horizons=tuple(
                    int(value)
                    for value in validation.get("report_horizons", [validation["horizon"]])
                ),
                modal_only=modal_only,
                device=device,
            )
            metric = float(learned["aggregate_record_mean_relative_l2"])
            payload = {
                "format": "snapshot_only_modal_closure_v2",
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": min(best_metric, metric),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": model_config,
                "transfer": transfer_report,
            }
            epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            _save_checkpoint(epoch_path, payload)
            _save_checkpoint(latest, payload)
            if metric < best_metric:
                best_metric = metric
                _save_checkpoint(output_dir / "best.pt", payload)
                _atomic_json(
                    output_dir / "best.json",
                    {"epoch": epoch, "checkpoint": str(epoch_path), "metrics": learned},
                )
            event = {
                "epoch": epoch,
                "global_step": global_step,
                "horizon": horizon,
                "closure_trainable": closure_trainable,
                "modal_only": modal_only,
                "train_loss": running / max(len(train_loader), 1),
                "train_components": {
                    name: value / max(len(train_loader), 1)
                    for name, value in component_sums.items()
                },
                "learned": learned,
                "modal_baseline": baseline,
                "elapsed_seconds": time.time() - started,
                "checkpoint": str(epoch_path),
            }
            with (output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
        _atomic_json(
            output_dir / "terminal.json",
            {"status": "success", "global_step": global_step, "best_metric": best_metric},
        )
    except Exception as error:
        _atomic_json(
            output_dir / "terminal.json",
            {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
