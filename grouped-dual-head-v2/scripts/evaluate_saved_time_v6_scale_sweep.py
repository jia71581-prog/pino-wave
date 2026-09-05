#!/usr/bin/env python
"""Evaluate residual correction scales with one shared model forward pass."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint  # noqa: E402
from saved_time_phase_operator_v4.data import (  # noqa: E402
    ExactStoredTimeBatchDataset,
    split_pilot_batch,
)
from saved_time_phase_operator_v4.losses import apply_hard_causality  # noqa: E402
from saved_time_phase_operator_v4.streaming_metrics import (  # noqa: E402
    ExactWavefieldMetricAccumulator,
)
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer  # noqa: E402
from scripts.train_saved_time_v4_full_support import (  # noqa: E402
    _atomic_json,
    _load_context,
    _load_parent_model,
    _validation_schedule,
    validation_panel_indices,
)


def scaled_correction_prediction(
    prediction: torch.Tensor,
    coarse: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Keep the transferred field fixed and rescale only the learned correction."""

    value = float(scale)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("correction scale must be finite and nonnegative")
    if prediction.shape != coarse.shape:
        raise ValueError("prediction and coarse field shapes must match")
    return coarse + value * (prediction - coarse)


def prepare_dense_grid_for_micro(model, prepared, tensors, micro, device):
    """Build the dense grid with the same travel-time source used by training."""

    cached = micro.dense_travel_time_s
    travel_time_s = (
        None
        if cached is None
        else cached.to(device, non_blocking=device.type == "cuda")
    )
    return model.prepare_dense_grid(
        prepared,
        x_m=tensors["x_m"],
        z_m=tensors["z_m"],
        travel_time_s=travel_time_s,
    )


def update_scale_metric_accumulators(
    accumulators,
    prediction: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    *,
    micro,
    onset_indices,
) -> None:
    """Update the official exact-wavefield metric for every correction scale."""

    for scale, accumulator in accumulators.items():
        candidate = scaled_correction_prediction(prediction, coarse, scale=float(scale))
        accumulator.update(
            candidate,
            target,
            families=micro.medium_type,
            group_ids=micro.group_id,
            sample_ids=micro.sample_id,
            time_indices=micro.left_index,
            source_onset_indices=onset_indices,
        )


def _parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(item) for item in str(value).split(","))
    if not scales or any(not math.isfinite(item) or item < 0.0 for item in scales):
        raise ValueError("scales must be a comma-separated nonnegative finite list")
    if len(set(scales)) != len(scales):
        raise ValueError("scales must be unique")
    return scales


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-identity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-records", type=int, default=48)
    parser.add_argument("--validation-frames", type=int, default=32)
    parser.add_argument("--panel-epoch", type=int, default=1)
    parser.add_argument("--scales", default="0,0.5,1,2,4,8")
    args = parser.parse_args(argv)

    scales = _parse_scales(args.scales)
    config = yaml.safe_load(Path(args.config).read_text())
    identity = json.loads(Path(args.checkpoint_identity).read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("scale sweep requires CUDA")
    device = torch.device("cuda")
    base, manifest, parent_identity = _load_context(config)
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    metadata = load_checkpoint(
        Path(args.checkpoint),
        model=model,
        optimizer=None,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=str(identity["run_digest"]),
        map_location=device,
    )
    normalizer = load_normalizer(base, manifest.digest)
    indices = validation_panel_indices(
        validation_records=base.data.expected_validation_records,
        panel_records=int(args.validation_records),
        epoch=int(args.panel_epoch),
        seed=int(config["seed"]),
    )
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split="validation",
        schedule=_validation_schedule(indices, epoch_offset=int(args.panel_epoch) - 1),
        query_points=1,
        seed=int(config["seed"]) + 7919,
        time_policy="validation_fixed",
        frames_per_record=int(args.validation_frames),
        travel_time_h5=config.get("travel_time_h5"),
    )

    accumulators = {
        scale: ExactWavefieldMetricAccumulator(
            energy_floor_fraction=float(config.get("energy_floor_fraction", 0.01)),
            require_unique=True,
            stored_time_count=len(manifest.time_s),
        )
        for scale in scales
    }
    base_correction_square = 0.0
    coarse_square = 0.0
    model.eval()
    for batch_index in range(len(dataset)):
        batch = dataset[batch_index]
        for micro in split_pilot_batch(
            batch,
            microbatch_records=int(config["validation"]["microbatch_records"]),
        ):
            tensors = _to_device(micro, device)
            source = tensors["source_parameters"]
            prepared = model.prepare_sources(
                model.encode_medium(tensors["velocity_mps"], normalizer),
                source,
                tensors["source_map"],
                normalizer,
                record_to_medium=tensors["record_to_medium"],
            )
            dense_grid = prepare_dense_grid_for_micro(
                model, prepared, tensors, micro, device
            )
            prediction, coarse = model.dense_normalized_with_coarse(
                prepared,
                tensors["requested_time_s"],
                dense_grid=dense_grid,
                time_block=1,
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
            metric_onset_s = source[:, 3]
            if config.get("residual_recovery"):
                metric_onset_s = torch.clamp(
                    source[:, 3] - source[:, 2].reciprocal(),
                    min=float(manifest.time_s[0]),
                )
            onset_indices = torch.searchsorted(
                torch.as_tensor(manifest.time_s, device=device),
                metric_onset_s.contiguous(),
            ).cpu().tolist()
            correction = prediction.float() - coarse.float()
            base_correction_square += float(correction.double().square().sum())
            coarse_square += float(coarse.double().square().sum())
            update_scale_metric_accumulators(
                accumulators,
                prediction,
                coarse,
                target,
                micro=micro,
                onset_indices=onset_indices,
            )

    scale_reports: dict[str, object] = {}
    coarse_score = None
    base_ratio = math.sqrt(base_correction_square / max(coarse_square, 1.0e-16))
    for scale in scales:
        metrics = accumulators[scale].finalize()
        aggregate = float(metrics["aggregate_relative_l2"])
        if scale == 0.0:
            coarse_score = aggregate
        scale_reports[str(scale)] = {
            "aggregate_relative_l2": aggregate,
            "family_relative_l2": metrics["family_relative_l2"],
            "phase_correlation": metrics["phase_correlation"],
            "time_bin_relative_l2": metrics["time_bin_relative_l2"],
            "frame_count": metrics["frame_count"],
            "correction_to_coarse_l2_ratio": float(scale * base_ratio),
        }
    if coarse_score is None:
        raise ValueError("scale sweep must include scale 0 for its coarse baseline")
    for report in scale_reports.values():
        report["relative_improvement_vs_coarse"] = (
            coarse_score - float(report["aggregate_relative_l2"])
        ) / max(coarse_score, 1.0e-16)
    best_scale = min(
        scales,
        key=lambda value: float(scale_reports[str(value)]["aggregate_relative_l2"]),
    )
    report = {
        "status": "complete",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(metadata.epoch),
        "checkpoint_global_step": int(metadata.global_step),
        "panel_epoch": int(args.panel_epoch),
        "validation_records": int(args.validation_records),
        "validation_frames": int(args.validation_frames),
        "best_scale": float(best_scale),
        "best_metrics": scale_reports[str(best_scale)],
        "scales": scale_reports,
    }
    _atomic_json(report, Path(args.output))
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
