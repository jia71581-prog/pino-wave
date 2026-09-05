#!/usr/bin/env python3
"""Evaluate the stable snapshot-only modal propagator on a leak-free prefix.

The propagator is run before future targets are read. Future wavefield frames are
opened only for scoring and never enter symbol estimation or rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saved_time_phase_operator_v4.snapshot_propagator import SnapshotOnlyWavePropagator


FAMILIES = ("uniform", "layered", "marmousi")


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _integers(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (prediction.double() - target.double()).flatten().norm()
        / target.double().flatten().norm().clamp_min(1.0e-30)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-frames", type=_integers, default=(4, 8, 16))
    parser.add_argument("--context-ends", type=_integers, default=(80, 112, 120))
    parser.add_argument("--horizons", type=_integers, default=(1, 8, 32))
    parser.add_argument("--records-per-family", type=int, default=2)
    parser.add_argument("--record-offset-per-family", type=int, default=0)
    parser.add_argument(
        "--split-name", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-context-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--modal-radius", type=float, default=1.0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--causal-backtest-prefix", type=int, default=None)
    args = parser.parse_args()

    if int(args.record_offset_per_family) < 0:
        raise ValueError("record-offset-per-family must be nonnegative")

    dataset = Path(args.dataset).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    rows: list[dict[str, object]] = []
    input_digests: dict[str, str] = {}
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    checkpoint_model = None
    checkpoint_epoch = None
    closure_projection_nonzero = None
    if checkpoint_path is not None:
        state = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
        )
        checkpoint_model_config = dict(state["model_config"])
        if args.causal_backtest_prefix is not None:
            checkpoint_model_config["causal_backtest_prefix"] = int(
                args.causal_backtest_prefix
            )
        checkpoint_model = SnapshotOnlyWavePropagator(**checkpoint_model_config)
        checkpoint_model.load_state_dict(state["model_state"], strict=True)
        checkpoint_model = checkpoint_model.to(device).eval()
        checkpoint_epoch = int(state["epoch"])
        closure_projection_nonzero = bool(
            torch.count_nonzero(checkpoint_model.project.weight)
            or torch.count_nonzero(checkpoint_model.project.bias)
        )
    with h5py.File(dataset, "r", swmr=True) as h5:
        time_count = int(h5["wavefield"].shape[1])
        maximum_context_fraction = float(args.maximum_context_fraction)
        if not 0.0 < maximum_context_fraction <= 1.0:
            raise ValueError("maximum-context-fraction must lie in (0,1]")
        invalid_ends = [
            int(value)
            for value in args.context_ends
            if int(value) / max(time_count - 1, 1) >= maximum_context_fraction
        ]
        if invalid_ends:
            raise ValueError(
                "context ends enter the forbidden middle/late interval: "
                f"{invalid_ends}"
            )
        split = np.asarray([_decode(value) for value in h5["split"][:]])
        family = np.asarray([_decode(value) for value in h5["medium_type"][:]])
        selected: list[int] = []
        for name in FAMILIES:
            candidates = np.flatnonzero(
                (split == str(args.split_name)) & (family == name)
            )
            start_record = int(args.record_offset_per_family)
            chosen = candidates[
                start_record : start_record + int(args.records_per_family)
            ]
            if len(chosen) != int(args.records_per_family):
                raise ValueError(
                    f"not enough {args.split_name} records for family {name} "
                    f"at offset {start_record}"
                )
            selected.extend(chosen.tolist())
        maximum_horizon = max(args.horizons)
        for record in selected:
            sample_id = _decode(h5["sample_id"][record])
            for context_end in args.context_ends:
                for history_frames in args.history_frames:
                    start = int(context_end) - int(history_frames) + 1
                    stop = int(context_end) + 1
                    if start < 0 or stop + maximum_horizon > h5["wavefield"].shape[1]:
                        raise ValueError("history/horizon selection is outside the stored time axis")
                    observed = torch.from_numpy(
                        np.asarray(h5["wavefield"][record, start:stop], dtype=np.float32)
                    )[None].to(device)
                    digest = hashlib.sha256(
                        observed.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    input_digests[
                        f"{sample_id}:end{int(context_end)}:k{int(history_frames)}"
                    ] = digest
                    with torch.no_grad():
                        if checkpoint_model is None:
                            model = SnapshotOnlyWavePropagator(
                                minimum_history=int(history_frames),
                                memory_frames=min(4, int(history_frames)),
                                width=8,
                                spectral_rank=4,
                                modes=4,
                                depth=1,
                                activation_checkpointing=False,
                                modal_radius=float(args.modal_radius),
                            ).to(device).eval()
                            prediction = model.modal_baseline(
                                observed, maximum_horizon
                            )[0].cpu()
                        else:
                            calibration_selected = bool(
                                checkpoint_model.causal_backtest_decision(observed)[0]
                            )
                            prediction = checkpoint_model(
                                observed, maximum_horizon
                            )[0].cpu()
                    # Future truth is deliberately opened only after the rollout completes.
                    target = torch.from_numpy(
                        np.asarray(
                            h5["wavefield"][record, stop : stop + maximum_horizon],
                            dtype=np.float32,
                        )
                    )
                    for horizon in args.horizons:
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "family": str(family[record]),
                                "record_index": int(record),
                                "context_end": int(context_end),
                                "history_frames": int(history_frames),
                                "horizon": int(horizon),
                                "calibration_selected": (
                                    calibration_selected
                                    if checkpoint_model is not None
                                    else None
                                ),
                                "relative_l2": _relative_l2(
                                    prediction[: int(horizon)], target[: int(horizon)]
                                ),
                            }
                        )
        manifest_sha256 = str(h5.attrs.get("manifest_sha256", ""))
        wavefield_shape = tuple(int(value) for value in h5["wavefield"].shape)

    summary: dict[str, object] = {}
    for context_end in args.context_ends:
        for history_frames in args.history_frames:
            for horizon in args.horizons:
                key = f"end{int(context_end)}_k{int(history_frames)}_h{int(horizon)}"
                chosen = [
                    row
                    for row in rows
                    if row["context_end"] == int(context_end)
                    and row["history_frames"] == int(history_frames)
                    and row["horizon"] == int(horizon)
                ]
                summary[key] = {
                    "aggregate_record_mean_relative_l2": float(
                        np.mean([float(row["relative_l2"]) for row in chosen])
                    ),
                    "family_record_mean_relative_l2": {
                        name: float(
                            np.mean(
                                [
                                    float(row["relative_l2"])
                                    for row in chosen
                                    if row["family"] == name
                                ]
                            )
                        )
                        for name in FAMILIES
                    },
                }
    report = {
        "scope": (
            "snapshot_only_checkpoint_small_panel"
            if checkpoint_path is not None
            else "snapshot_only_modal_baseline_small_panel"
        ),
        "strict_goal_eligible": False,
        "dataset": str(dataset),
        "dataset_manifest_sha256": manifest_sha256,
        "wavefield_shape": wavefield_shape,
        "split_name": str(args.split_name),
        "records_per_family": int(args.records_per_family),
        "record_offset_per_family": int(args.record_offset_per_family),
        "deployment_contract": {
            "inference_signature": str(inspect.signature(SnapshotOnlyWavePropagator.forward)),
            "input_keys": ["wavefield_history", "steps"],
            "velocity_used": False,
            "source_used": False,
            "travel_time_used": False,
            "numerical_solver_used": bool(
                checkpoint_model is not None
                and checkpoint_model.local_wave_blend_weight > 0.0
            ),
            "external_numerical_solver_used": False,
            "snapshot_inferred_local_wave_recurrence_used": bool(
                checkpoint_model is not None
                and checkpoint_model.local_wave_blend_weight > 0.0
            ),
            "future_truth_used_for_propagation": False,
            "middle_or_late_snapshots_used": False,
            "maximum_context_fraction": maximum_context_fraction,
            "fixed_modal_radius": float(
                checkpoint_model.modal_radius
                if checkpoint_model is not None
                else args.modal_radius
            ),
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_epoch": checkpoint_epoch,
            "closure_projection_nonzero": closure_projection_nonzero,
            "causal_backtest_prefix": (
                checkpoint_model.causal_backtest_prefix
                if checkpoint_model is not None
                else 0
            ),
            "observed_defect_memory": (
                checkpoint_model.observed_defect_memory
                if checkpoint_model is not None
                else False
            ),
            "defect_memory_decay": (
                checkpoint_model.defect_memory_decay
                if checkpoint_model is not None
                else None
            ),
            "defect_trend_scale": (
                checkpoint_model.defect_trend_scale
                if checkpoint_model is not None
                else None
            ),
            "defect_memory_mode": (
                checkpoint_model.defect_memory_mode
                if checkpoint_model is not None
                else None
            ),
            "defect_modal_radius": (
                checkpoint_model.defect_modal_radius
                if checkpoint_model is not None
                else None
            ),
            "defect_high_frequency_radius": (
                checkpoint_model.defect_high_frequency_radius
                if checkpoint_model is not None
                else None
            ),
            "defect_high_frequency_cutoff": (
                checkpoint_model.defect_high_frequency_cutoff
                if checkpoint_model is not None
                else None
            ),
            "local_wave_blend_weight": (
                checkpoint_model.local_wave_blend_weight
                if checkpoint_model is not None
                else None
            ),
            "local_wave_blend_ramp_steps": (
                checkpoint_model.local_wave_blend_ramp_steps
                if checkpoint_model is not None
                else None
            ),
            "local_wave_substeps": (
                checkpoint_model.local_wave_substeps
                if checkpoint_model is not None
                else None
            ),
            "local_wave_pool_size": (
                checkpoint_model.local_wave_pool_size
                if checkpoint_model is not None
                else None
            ),
            "local_wave_regularization": (
                checkpoint_model.local_wave_regularization
                if checkpoint_model is not None
                else None
            ),
            "local_wave_coefficient_bounds": (
                [
                    checkpoint_model.local_wave_minimum_coefficient,
                    checkpoint_model.local_wave_maximum_coefficient,
                ]
                if checkpoint_model is not None
                else None
            ),
            "local_wave_instance_backtest": (
                checkpoint_model.local_wave_instance_backtest
                if checkpoint_model is not None
                else None
            ),
            "local_wave_adaptive_boost": (
                checkpoint_model.local_wave_adaptive_boost
                if checkpoint_model is not None
                else None
            ),
            "local_wave_adaptive_boost_threshold": (
                checkpoint_model.local_wave_adaptive_boost_threshold
                if checkpoint_model is not None
                else None
            ),
            "local_wave_adaptive_boost_width": (
                checkpoint_model.local_wave_adaptive_boost_width
                if checkpoint_model is not None
                else None
            ),
            "local_wave_adaptive_boost_maximum_weight": (
                checkpoint_model.local_wave_adaptive_boost_maximum_weight
                if checkpoint_model is not None
                else None
            ),
            "local_wave_adaptive_boost_start_step": (
                checkpoint_model.local_wave_adaptive_boost_start_step
                if checkpoint_model is not None
                else None
            ),
            "local_wave_adaptive_boost_ramp_steps": (
                checkpoint_model.local_wave_adaptive_boost_ramp_steps
                if checkpoint_model is not None
                else None
            ),
        },
        "observed_input_sha256": input_digests,
        "summary": summary,
        "rows": rows,
    }
    _atomic_json(Path(args.output).resolve(), report)
    print(json.dumps({"output": str(Path(args.output).resolve()), "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
