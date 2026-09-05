#!/usr/bin/env python3
"""Evaluate a V49-high/V59-low-mid splice on one registered fixed panel."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.band_adapter import registered_high_band_mask
from saved_time_phase_operator_v4.data import (
    ExactStoredTimeBatchDataset,
    split_pilot_batch,
)
from saved_time_phase_operator_v4.evaluation import sha256_file, time_axis_sha256
from saved_time_phase_operator_v4.losses import (
    apply_hard_causality,
    source_causality_onset_s,
)
from saved_time_phase_operator_v4.streaming_metrics import (
    ExactWavefieldMetricAccumulator,
)
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_full_support import (
    _atomic_json,
    _load_context,
    _load_parent_model,
    _validation_schedule,
    validation_panel_indices,
)


def _validated_fields(
    low_mid_candidate: torch.Tensor,
    high_anchor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate = torch.as_tensor(low_mid_candidate)
    anchor = torch.as_tensor(high_anchor, device=candidate.device)
    if candidate.ndim != 4 or anchor.ndim != 4:
        raise ValueError("splice inputs must be [record,time,z,x]")
    if candidate.shape != anchor.shape:
        raise ValueError("splice inputs must have the same shape")
    if not bool(torch.isfinite(candidate).all()) or not bool(torch.isfinite(anchor).all()):
        raise ValueError("splice inputs must be finite")
    return candidate, anchor


def splice_low_mid_with_high_anchor(
    low_mid_candidate: torch.Tensor,
    high_anchor: torch.Tensor,
) -> torch.Tensor:
    """Take low/middle Fourier modes from candidate and high modes from anchor."""

    candidate, anchor = _validated_fields(low_mid_candidate, high_anchor)
    height, width = candidate.shape[-2:]
    candidate_fft = torch.fft.rfft2(candidate.float(), norm="ortho")
    anchor_fft = torch.fft.rfft2(anchor.float(), norm="ortho")
    high = registered_high_band_mask(height, width, candidate_fft.device)
    spliced_fft = torch.where(high, anchor_fft, candidate_fft)
    return torch.fft.irfft2(
        spliced_fft,
        s=(height, width),
        norm="ortho",
    )


def high_band_anchor_delta(
    spliced: torch.Tensor,
    high_anchor: torch.Tensor,
) -> float:
    """Return relative high-band distance from the registered anchor."""

    candidate, anchor = _validated_fields(spliced, high_anchor)
    height, width = candidate.shape[-2:]
    candidate_fft = torch.fft.rfft2(candidate.float(), norm="ortho")
    anchor_fft = torch.fft.rfft2(anchor.float(), norm="ortho")
    high = registered_high_band_mask(height, width, candidate_fft.device)
    numerator = (candidate_fft[..., high] - anchor_fft[..., high]).abs().square().sum()
    denominator = anchor_fft[..., high].abs().square().sum().clamp_min(1.0e-16)
    return float(torch.sqrt(numerator / denominator))


def checkpoint_run_identity_path(checkpoint: str | Path) -> Path:
    """Resolve the immutable run identity adjacent to an epoch checkpoint."""

    path = Path(checkpoint).resolve()
    if path.parent.name != "checkpoints":
        raise ValueError("checkpoint must live in a checkpoints directory")
    identity = path.parent.parent / "run_identity.json"
    if not identity.is_file():
        raise FileNotFoundError(f"checkpoint run identity is unavailable: {identity}")
    return identity.resolve()


def _read_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"splice config must be a mapping: {path}")
    return payload


def _validation_contract(config: Mapping[str, object]) -> tuple[int, int, int, int]:
    validation = config.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("splice config is missing validation settings")
    values = (
        int(config["seed"]),
        int(validation["panel_records"]),
        int(validation["frames_per_record"]),
        int(validation["microbatch_records"]),
    )
    if any(value <= 0 for value in values[1:]):
        raise ValueError("splice validation settings must be positive")
    return values


def _load_bound_checkpoint(
    config_path: Path,
    checkpoint: Path,
    device: torch.device,
) -> dict[str, object]:
    config = _read_config(config_path)
    base, manifest, parent_identity = _load_context(config)
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    identity_path = checkpoint_run_identity_path(checkpoint)
    identity = json.loads(identity_path.read_text())
    if not isinstance(identity, dict) or not identity.get("run_digest"):
        raise ValueError("checkpoint run identity is malformed")
    metadata = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=str(identity["run_digest"]),
        map_location=device,
    )
    return {
        "config": config,
        "config_path": config_path,
        "base": base,
        "manifest": manifest,
        "model": model,
        "checkpoint": checkpoint,
        "identity": identity,
        "identity_path": identity_path,
        "checkpoint_epoch": int(metadata.epoch),
        "checkpoint_global_step": int(metadata.global_step),
    }


def _assert_same_panel(anchor: Mapping[str, object], candidate: Mapping[str, object]):
    anchor_manifest = anchor["manifest"]
    candidate_manifest = candidate["manifest"]
    if anchor_manifest.digest != candidate_manifest.digest:
        raise ValueError("splice checkpoints do not share one manifest")
    if time_axis_sha256(anchor_manifest.time_s) != time_axis_sha256(
        candidate_manifest.time_s
    ):
        raise ValueError("splice checkpoints do not share one stored-time axis")
    anchor_normalizer = load_normalizer(anchor["base"], anchor_manifest.digest)
    candidate_normalizer = load_normalizer(candidate["base"], candidate_manifest.digest)
    if asdict(anchor_normalizer.metadata) != asdict(candidate_normalizer.metadata):
        raise ValueError("splice checkpoints do not share one normalization")
    anchor_contract = _validation_contract(anchor["config"])
    candidate_contract = _validation_contract(candidate["config"])
    if anchor_contract[:3] != candidate_contract[:3]:
        raise ValueError("splice checkpoints do not share one fixed validation panel")
    seed, records, frames, _ = anchor_contract
    anchor_indices = validation_panel_indices(
        validation_records=anchor["base"].data.expected_validation_records,
        panel_records=records,
        epoch=1,
        seed=seed,
    )
    candidate_indices = validation_panel_indices(
        validation_records=candidate["base"].data.expected_validation_records,
        panel_records=records,
        epoch=1,
        seed=seed,
    )
    if anchor_indices != candidate_indices:
        raise ValueError("splice validation indices are not identical")
    anchor_records = tuple(
        record for record in anchor_manifest.records if record.split == "validation"
    )
    candidate_records = tuple(
        record for record in candidate_manifest.records if record.split == "validation"
    )
    anchor_ids = tuple(anchor_records[index].sample_id for index in anchor_indices)
    candidate_ids = tuple(candidate_records[index].sample_id for index in candidate_indices)
    if anchor_ids != candidate_ids:
        raise ValueError("splice validation source IDs are not identical")
    return anchor_normalizer, anchor_indices, anchor_ids, frames


def _prediction(
    model,
    normalizer,
    tensors: Mapping[str, torch.Tensor],
    micro,
    config: Mapping[str, object],
) -> torch.Tensor:
    source = tensors["source_parameters"]
    prepared = model.prepare_sources(
        model.encode_medium(tensors["velocity_mps"], normalizer),
        source,
        tensors["source_map"],
        normalizer,
        record_to_medium=tensors["record_to_medium"],
    )
    dense_grid = model.prepare_dense_grid(
        prepared,
        x_m=tensors["x_m"],
        z_m=tensors["z_m"],
        travel_time_s=(
            None
            if micro.dense_travel_time_s is None
            else micro.dense_travel_time_s.to(
                tensors["velocity_mps"].device,
                non_blocking=tensors["velocity_mps"].device.type == "cuda",
            )
        ),
    )
    prediction = model.dense_normalized(
        prepared,
        tensors["requested_time_s"],
        dense_grid=dense_grid,
        time_block=1,
    )
    loss = config.get("loss", {})
    if isinstance(loss, Mapping) and bool(loss.get("hard_causality", False)):
        onset = source_causality_onset_s(
            source,
            lead_cycles=float(loss.get("hard_causality_lead_cycles", 0.0)),
        )
        prediction = apply_hard_causality(
            prediction,
            tensors["requested_time_s"],
            onset,
        )
    return prediction


@torch.inference_mode()
def evaluate_splice(
    anchor: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    device: torch.device,
) -> dict[str, object]:
    normalizer, indices, sample_ids, frames = _assert_same_panel(anchor, candidate)
    config = anchor["config"]
    manifest = anchor["manifest"]
    dataset = ExactStoredTimeBatchDataset(
        anchor["base"].data.source_h5,
        manifest,
        split="validation",
        schedule=_validation_schedule(indices, epoch_offset=0),
        query_points=1,
        seed=int(config["seed"]) + 7919,
        time_policy="validation_fixed",
        frames_per_record=int(frames),
        travel_time_h5=config.get("travel_time_h5"),
    )
    energy_floor = float(config.get("energy_floor_fraction", 0.01))
    accumulators = {
        name: ExactWavefieldMetricAccumulator(
            energy_floor_fraction=energy_floor,
            require_unique=True,
            stored_time_count=len(manifest.time_s),
        )
        for name in ("anchor", "candidate", "splice")
    }
    anchor["model"].eval()
    candidate["model"].eval()
    torch.cuda.reset_peak_memory_stats(device)
    delta_numerator = 0.0
    delta_denominator = 0.0
    microbatch = min(
        _validation_contract(anchor["config"])[3],
        _validation_contract(candidate["config"])[3],
    )
    for batch_index in range(len(dataset)):
        batch = dataset[batch_index]
        for micro in split_pilot_batch(batch, microbatch_records=microbatch):
            tensors = _to_device(micro, device)
            anchor_prediction = _prediction(
                anchor["model"], normalizer, tensors, micro, anchor["config"]
            )
            candidate_prediction = _prediction(
                candidate["model"], normalizer, tensors, micro, candidate["config"]
            )
            if anchor_prediction.shape != candidate_prediction.shape:
                raise RuntimeError("splice model output shapes differ")
            if anchor_prediction.shape[-2:] != (201, 201):
                raise RuntimeError("splice evaluation requires 201x201 full wavefields")
            spliced = splice_low_mid_with_high_anchor(
                candidate_prediction,
                anchor_prediction,
            )
            target = normalizer.encode_pressure(
                tensors["dense_target_physical"],
                tensors["source_parameters"][:, 4],
            )
            metric_onset_s = tensors["source_parameters"][:, 3]
            if config.get("residual_recovery"):
                metric_onset_s = torch.clamp(
                    tensors["source_parameters"][:, 3]
                    - tensors["source_parameters"][:, 2].reciprocal(),
                    min=float(manifest.time_s[0]),
                )
            onset_indices = torch.searchsorted(
                torch.as_tensor(manifest.time_s, device=device),
                metric_onset_s.contiguous(),
            ).cpu().tolist()
            for name, prediction in (
                ("anchor", anchor_prediction),
                ("candidate", candidate_prediction),
                ("splice", spliced),
            ):
                accumulators[name].update(
                    prediction,
                    target,
                    families=micro.medium_type,
                    group_ids=micro.group_id,
                    sample_ids=micro.sample_id,
                    time_indices=micro.left_index,
                    source_onset_indices=onset_indices,
                )
            height, width = spliced.shape[-2:]
            high = registered_high_band_mask(height, width, device)
            spliced_fft = torch.fft.rfft2(spliced.float(), norm="ortho")
            anchor_fft = torch.fft.rfft2(anchor_prediction.float(), norm="ortho")
            delta_numerator += float(
                (spliced_fft[..., high] - anchor_fft[..., high])
                .abs()
                .square()
                .sum()
            )
            delta_denominator += float(anchor_fft[..., high].abs().square().sum())
    metrics = {name: value.finalize() for name, value in accumulators.items()}
    if any(value["record_count"] != len(indices) for value in metrics.values()):
        raise RuntimeError("splice evaluation did not cover the fixed record panel")
    return {
        "metrics": metrics,
        "validation_indices": list(indices),
        "sample_ids": list(sample_ids),
        "high_band_anchor_delta": math.sqrt(delta_numerator)
        / math.sqrt(max(delta_denominator, 1.0e-16)),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _checkpoint_binding(loaded: Mapping[str, object]) -> dict[str, object]:
    return {
        "config": str(loaded["config_path"]),
        "config_sha256": sha256_file(loaded["config_path"]),
        "checkpoint": str(loaded["checkpoint"]),
        "checkpoint_sha256": sha256_file(loaded["checkpoint"]),
        "checkpoint_epoch": int(loaded["checkpoint_epoch"]),
        "checkpoint_global_step": int(loaded["checkpoint_global_step"]),
        "run_identity": str(loaded["identity_path"]),
        "run_identity_sha256": sha256_file(loaded["identity_path"]),
        "run_digest": str(loaded["identity"]["run_digest"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-config", required=True)
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("same-panel splice evaluation requires CUDA")
    anchor_config = Path(args.anchor_config).resolve()
    anchor_checkpoint = Path(args.anchor_checkpoint).resolve()
    candidate_config = Path(args.candidate_config).resolve()
    candidate_checkpoint = Path(args.candidate_checkpoint).resolve()
    output = Path(args.output).resolve()
    anchor = _load_bound_checkpoint(anchor_config, anchor_checkpoint, device)
    candidate = _load_bound_checkpoint(
        candidate_config,
        candidate_checkpoint,
        device,
    )
    result = evaluate_splice(anchor, candidate, device=device)
    manifest = anchor["manifest"]
    report = {
        "schema": "saved_time_high_band_splice_v1",
        "status": "complete",
        "binding": {
            "anchor": _checkpoint_binding(anchor),
            "candidate": _checkpoint_binding(candidate),
            "manifest_digest": str(manifest.digest),
            "time_axis_sha256": time_axis_sha256(manifest.time_s),
            "validation_seed": int(anchor["config"]["seed"]),
            "validation_indices": result["validation_indices"],
            "sample_ids": result["sample_ids"],
            "wavefield_shape": [201, 201],
            "exact_stored_times_only": True,
            "one_source_per_record": True,
            "receiver_input": False,
        },
        "metrics": result["metrics"],
        "high_band_anchor_delta": float(result["high_band_anchor_delta"]),
        "peak_cuda_bytes": int(result["peak_cuda_bytes"]),
    }
    _atomic_json(report, output)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "checkpoint_run_identity_path",
    "evaluate_splice",
    "high_band_anchor_delta",
    "splice_low_mid_with_high_anchor",
]
