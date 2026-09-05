#!/usr/bin/env python3
"""Promote a calibrated snapshot checkpoint with an inference-only causal gate."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saved_time_phase_operator_v4.snapshot_propagator import (
    SnapshotOnlyWavePropagator,
    transfer_pretrained_decoder_stack,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--causal-backtest-prefix", type=int, default=5)
    parser.add_argument("--observed-defect-memory", action="store_true")
    parser.add_argument("--defect-memory-decay", type=float, default=0.55)
    parser.add_argument("--defect-trend-scale", type=float, default=0.75)
    parser.add_argument(
        "--defect-memory-mode",
        choices=("polynomial", "stable_modal"),
        default="polynomial",
    )
    parser.add_argument("--defect-modal-radius", type=float, default=0.88)
    parser.add_argument("--defect-high-frequency-radius", type=float, default=None)
    parser.add_argument("--defect-high-frequency-cutoff", type=float, default=0.75)
    parser.add_argument("--local-wave-blend-weight", type=float, default=0.0)
    parser.add_argument("--local-wave-blend-ramp-steps", type=int, default=8)
    parser.add_argument("--local-wave-substeps", type=int, default=8)
    parser.add_argument("--local-wave-pool-size", type=int, default=15)
    parser.add_argument("--local-wave-regularization", type=float, default=0.1)
    parser.add_argument("--local-wave-minimum-coefficient", type=float, default=0.02)
    parser.add_argument("--local-wave-maximum-coefficient", type=float, default=1.5)
    parser.add_argument("--local-wave-instance-backtest", action="store_true")
    parser.add_argument("--local-wave-adaptive-boost", action="store_true")
    parser.add_argument(
        "--local-wave-adaptive-boost-threshold", type=float, default=0.1
    )
    parser.add_argument("--local-wave-adaptive-boost-width", type=float, default=0.2)
    parser.add_argument(
        "--local-wave-adaptive-boost-maximum-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--local-wave-adaptive-boost-start-step", type=int, default=8
    )
    parser.add_argument(
        "--local-wave-adaptive-boost-ramp-steps", type=int, default=4
    )
    parser.add_argument("--pretrained-parent-checkpoint", default=None)
    args = parser.parse_args()

    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    state = torch.load(str(source), map_location="cpu", weights_only=False, mmap=True)
    model_config = dict(state["model_config"])
    if int(model_config["minimum_history"]) <= int(args.causal_backtest_prefix):
        raise ValueError("causal backtest prefix must leave held-out history frames")
    model_config["causal_backtest_prefix"] = int(args.causal_backtest_prefix)
    model_config["observed_defect_memory"] = bool(args.observed_defect_memory)
    model_config["defect_memory_decay"] = float(args.defect_memory_decay)
    model_config["defect_trend_scale"] = float(args.defect_trend_scale)
    model_config["defect_memory_mode"] = str(args.defect_memory_mode)
    model_config["defect_modal_radius"] = float(args.defect_modal_radius)
    model_config["defect_high_frequency_radius"] = (
        None
        if args.defect_high_frequency_radius is None
        else float(args.defect_high_frequency_radius)
    )
    model_config["defect_high_frequency_cutoff"] = float(
        args.defect_high_frequency_cutoff
    )
    model_config["local_wave_blend_weight"] = float(args.local_wave_blend_weight)
    model_config["local_wave_blend_ramp_steps"] = int(
        args.local_wave_blend_ramp_steps
    )
    model_config["local_wave_substeps"] = int(args.local_wave_substeps)
    model_config["local_wave_pool_size"] = int(args.local_wave_pool_size)
    model_config["local_wave_regularization"] = float(
        args.local_wave_regularization
    )
    model_config["local_wave_minimum_coefficient"] = float(
        args.local_wave_minimum_coefficient
    )
    model_config["local_wave_maximum_coefficient"] = float(
        args.local_wave_maximum_coefficient
    )
    model_config["local_wave_instance_backtest"] = bool(
        args.local_wave_instance_backtest
    )
    model_config["local_wave_adaptive_boost"] = bool(
        args.local_wave_adaptive_boost
    )
    model_config["local_wave_adaptive_boost_threshold"] = float(
        args.local_wave_adaptive_boost_threshold
    )
    model_config["local_wave_adaptive_boost_width"] = float(
        args.local_wave_adaptive_boost_width
    )
    model_config["local_wave_adaptive_boost_maximum_weight"] = float(
        args.local_wave_adaptive_boost_maximum_weight
    )
    model_config["local_wave_adaptive_boost_start_step"] = int(
        args.local_wave_adaptive_boost_start_step
    )
    model_config["local_wave_adaptive_boost_ramp_steps"] = int(
        args.local_wave_adaptive_boost_ramp_steps
    )
    model = SnapshotOnlyWavePropagator(**model_config)
    model.load_state_dict(state["model_state"], strict=True)

    transfer = None
    parent_path = None
    if args.pretrained_parent_checkpoint is not None:
        parent_path = Path(args.pretrained_parent_checkpoint).resolve()
        parent = torch.load(
            str(parent_path), map_location="cpu", weights_only=False, mmap=True
        )
        transfer = transfer_pretrained_decoder_stack(model, parent["model_state"])
        transfer.update(
            {
                "parent_checkpoint": str(parent_path),
                "parent_epoch": int(parent["epoch"]),
            }
        )

    promoted = dict(state)
    if (
        args.observed_defect_memory
        and args.local_wave_blend_weight > 0.0
        and args.local_wave_instance_backtest
        and args.local_wave_adaptive_boost
    ):
        promoted["format"] = "snapshot_only_modal_causal_adaptive_local_wave_v9"
    elif (
        args.observed_defect_memory
        and args.local_wave_blend_weight > 0.0
        and args.local_wave_instance_backtest
    ):
        promoted["format"] = "snapshot_only_modal_causal_instance_local_wave_v8"
    elif args.observed_defect_memory and args.local_wave_blend_weight > 0.0:
        promoted["format"] = "snapshot_only_modal_causal_local_wave_blend_v7"
    elif (
        args.observed_defect_memory
        and args.defect_memory_mode == "stable_modal"
        and args.defect_high_frequency_radius is not None
    ):
        promoted["format"] = "snapshot_only_modal_causal_multiscale_defect_v6"
    elif args.observed_defect_memory and args.defect_memory_mode == "stable_modal":
        promoted["format"] = "snapshot_only_modal_causal_defect_dynamics_v5"
    elif args.observed_defect_memory:
        promoted["format"] = "snapshot_only_modal_causal_defect_v4"
    else:
        promoted["format"] = "snapshot_only_modal_causal_gate_v3"
    promoted["model_config"] = model_config
    promoted["model_state"] = model.state_dict()
    if transfer is not None:
        promoted["transfer"] = transfer
    promoted["promotion"] = {
        "source_checkpoint": str(source),
        "causal_backtest_prefix": int(args.causal_backtest_prefix),
        "observed_history_frames": int(model_config["minimum_history"]),
        "future_truth_used_for_gate": False,
        "observed_defect_memory": bool(args.observed_defect_memory),
        "defect_memory_decay": float(args.defect_memory_decay),
        "defect_trend_scale": float(args.defect_trend_scale),
        "defect_memory_mode": str(args.defect_memory_mode),
        "defect_modal_radius": float(args.defect_modal_radius),
        "defect_high_frequency_radius": (
            None
            if args.defect_high_frequency_radius is None
            else float(args.defect_high_frequency_radius)
        ),
        "defect_high_frequency_cutoff": float(args.defect_high_frequency_cutoff),
        "local_wave_blend_weight": float(args.local_wave_blend_weight),
        "local_wave_blend_ramp_steps": int(args.local_wave_blend_ramp_steps),
        "local_wave_substeps": int(args.local_wave_substeps),
        "local_wave_pool_size": int(args.local_wave_pool_size),
        "local_wave_regularization": float(args.local_wave_regularization),
        "local_wave_minimum_coefficient": float(
            args.local_wave_minimum_coefficient
        ),
        "local_wave_maximum_coefficient": float(
            args.local_wave_maximum_coefficient
        ),
        "local_wave_instance_backtest": bool(args.local_wave_instance_backtest),
        "local_wave_adaptive_boost": bool(args.local_wave_adaptive_boost),
        "local_wave_adaptive_boost_threshold": float(
            args.local_wave_adaptive_boost_threshold
        ),
        "local_wave_adaptive_boost_width": float(
            args.local_wave_adaptive_boost_width
        ),
        "local_wave_adaptive_boost_maximum_weight": float(
            args.local_wave_adaptive_boost_maximum_weight
        ),
        "local_wave_adaptive_boost_start_step": int(
            args.local_wave_adaptive_boost_start_step
        ),
        "local_wave_adaptive_boost_ramp_steps": int(
            args.local_wave_adaptive_boost_ramp_steps
        ),
        "adaptive_boost_parameters_selected_on_training_split": bool(
            args.local_wave_adaptive_boost
        ),
        "pretrained_parent_checkpoint": (
            None if parent_path is None else str(parent_path)
        ),
        "local_wave_parameters_selected_on_training_split": bool(
            args.local_wave_blend_weight > 0.0
        ),
        "defect_parameters_selected_on_training_split": bool(
            args.observed_defect_memory
        ),
        "model_parameters_changed": bool(transfer is not None),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(promoted, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(str(output))


if __name__ == "__main__":
    main()
