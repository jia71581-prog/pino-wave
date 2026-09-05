#!/usr/bin/env python3
"""Measure the best direct-frequency output projection on frozen Born features.

This diagnostic separates three ceilings for the background-conditioned Helmholtz
reflection branch:

1. the temporal Fourier truncation ceiling;
2. the current trained direct-frequency output;
3. the least-squares-optimal shared 1x1 output projection on the current propagated
   spatial features.

It never changes a checkpoint.  The optional JSON report records all source paths and
sample IDs so the result can be reproduced independently of a training run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
from saved_time_phase_operator_v4.local_field import _BackgroundBornConditioner
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer


CONDITIONER_PREFIX = "local_field.helmholtz_background_conditioner."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--background-cache", type=Path, action="append", required=True)
    parser.add_argument("--travel-time-h5", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--direct-frequencies", type=int, default=32)
    parser.add_argument("--propagation-modes", type=int, default=48)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _decode(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _load_conditioner(
    checkpoint: Path,
    *,
    direct_frequencies: int,
    propagation_modes: int,
    device: torch.device,
) -> tuple[_BackgroundBornConditioner, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state mapping")
    selected = {
        key[len(CONDITIONER_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(CONDITIONER_PREFIX)
    }
    first_weight = selected.get("input_projection.0.weight")
    if not isinstance(first_weight, torch.Tensor):
        raise ValueError("checkpoint has no background-conditioner input projection")
    width = int(first_weight.shape[0])
    conditioner = _BackgroundBornConditioner(
        width,
        num_frequencies=max(int(direct_frequencies), 1),
        background_sigma_cells=2.0,
        global_propagation=True,
        propagation_modes=int(propagation_modes),
        direct_frequency_output=True,
        direct_frequency_count=int(direct_frequencies),
    )
    incompatible = conditioner.load_state_dict(selected, strict=False)
    # Checkpoints created before the coupled x-z propagation branch do not contain
    # its zero-initialised gates.  That is the only intentional compatibility gap;
    # reject every other mismatch so this diagnostic cannot silently analyse the
    # wrong architecture.
    allowed_missing = {"propagation_coupled_gates"}
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if not missing.issubset(allowed_missing) or unexpected:
        raise RuntimeError(
            "background-conditioner checkpoint mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    conditioner.to(device).eval()
    metadata = {
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "width": width,
        "conditioner_tensor_count": len(selected),
    }
    return conditioner, metadata


def _target_coefficients(target: torch.Tensor, count: int) -> torch.Tensor:
    """Return [record,2*count,z,x] coefficients used by direct synthesis."""

    spectrum = torch.fft.rfft(target.float(), dim=1, norm="forward")[:, :count]
    cosine = 2.0 * spectrum.real
    sine = -2.0 * spectrum.imag
    cosine[:, 0] = spectrum[:, 0].real
    sine[:, 0] = 0.0
    return torch.cat((cosine, sine), dim=1)


def _field_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction64 = prediction.double().reshape(-1)
    target64 = target.double().reshape(-1)
    target_square = target64.square().sum().clamp_min(1.0e-30)
    prediction_square = prediction64.square().sum()
    error = (prediction64 - target64).square().sum().sqrt() / target_square.sqrt()
    energy_ratio = prediction_square.sqrt() / target_square.sqrt()
    cosine = torch.dot(prediction64, target64) / (
        prediction_square.sqrt() * target_square.sqrt()
    ).clamp_min(1.0e-30)
    return {
        "relative_l2": float(error),
        "prediction_to_target_l2_ratio": float(energy_ratio),
        "cosine_similarity": float(cosine),
    }


def main() -> int:
    args = parse_args()
    if args.direct_frequencies <= 0:
        raise ValueError("direct frequency count must be positive")
    if args.propagation_modes <= 0:
        raise ValueError("propagation mode count must be positive")
    if not math.isfinite(args.ridge_relative) or args.ridge_relative < 0.0:
        raise ValueError("relative ridge must be finite and nonnegative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    conditioner, checkpoint_metadata = _load_conditioner(
        args.checkpoint.resolve(),
        direct_frequencies=int(args.direct_frequencies),
        propagation_modes=int(args.propagation_modes),
        device=device,
    )
    with args.normalization_json.open(encoding="utf8") as handle:
        normalizer = PhysicalNormalizer.from_dict(json.load(handle))

    provider = BackgroundFieldProvider([path.resolve() for path in args.background_cache])
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    try:
        with h5py.File(args.source_h5.resolve(), "r", swmr=True) as source, h5py.File(
            args.travel_time_h5.resolve(), "r", swmr=True
        ) as travel:
            source_ids = _decode(source["sample_id"][:])
            travel_ids = _decode(travel["sample_id"][:])
            source_positions = {sample_id: index for index, sample_id in enumerate(source_ids)}
            travel_positions = {sample_id: index for index, sample_id in enumerate(travel_ids)}
            saved_time_values = torch.as_tensor(
                source["time_s"][:], dtype=torch.float32, device=device
            )
            dt = (saved_time_values[-1] - saved_time_values[0]) / float(
                saved_time_values.numel() - 1
            )
            omega = (
                2.0
                * math.pi
                * torch.arange(
                    int(args.direct_frequencies), dtype=torch.float32, device=device
                )
                / (float(saved_time_values.numel()) * dt)
            )

            for sample_id in args.sample_id:
                if sample_id not in source_positions:
                    raise KeyError(f"source dataset has no sample {sample_id!r}")
                if sample_id not in travel_positions:
                    raise KeyError(f"travel cache has no sample {sample_id!r}")
                source_index = source_positions[sample_id]
                travel_index = travel_positions[sample_id]
                target_physical = torch.from_numpy(
                    np.asarray(source["wavefield"][source_index], dtype=np.float32)
                )[None]
                background_physical = provider.full_physical([sample_id])
                scattering_physical = target_physical - background_physical
                amplitude = torch.as_tensor(
                    [float(source["source_amplitude"][source_index])], dtype=torch.float32
                )
                target = normalizer.encode_pressure(scattering_physical, amplitude).to(device)
                background = normalizer.encode_pressure(background_physical, amplitude).to(device)
                velocity = torch.from_numpy(
                    np.asarray(source["velocity_mps"][source_index], dtype=np.float32)
                )[None].to(device)
                arrival = torch.from_numpy(
                    np.asarray(travel["travel_time_s"][travel_index], dtype=np.float32)
                )[None].to(device)
                mapping = torch.zeros(1, dtype=torch.long, device=device)

                captured: list[torch.Tensor] = []

                def capture_input(_module, inputs):
                    captured.append(inputs[0].detach())

                hook = conditioner.output_projection.register_forward_pre_hook(capture_input)
                with torch.inference_mode():
                    current_coefficients = conditioner(
                        background, velocity, mapping, arrival, omega
                    )
                hook.remove()
                if len(captured) != 1:
                    raise RuntimeError("output projection hook did not capture one feature map")

                target_coefficients = _target_coefficients(
                    target, int(args.direct_frequencies)
                )
                features = captured[0].permute(0, 2, 3, 1).reshape(
                    -1, captured[0].shape[1]
                )
                coefficients = target_coefficients.permute(0, 2, 3, 1).reshape(
                    -1, target_coefficients.shape[1]
                )
                feature_rows.append(features.cpu())
                target_rows.append(coefficients.cpu())

                with torch.inference_mode():
                    projected_target = conditioner.synthesize_direct_frequency_output(
                        target_coefficients,
                        saved_time_values[None],
                        saved_time_values,
                    )
                    current_field = conditioner.synthesize_direct_frequency_output(
                        current_coefficients,
                        saved_time_values[None],
                        saved_time_values,
                    )
                records.append(
                    {
                        "sample_id": sample_id,
                        "medium_type": _decode([source["medium_type"][source_index]])[0],
                        "source_f0_hz": float(source["source_f0_hz"][source_index]),
                        "dt_used_s": float(source["dt_used_s"][source_index]),
                        "temporal_projection": _field_metrics(projected_target, target),
                        "current_direct_output": _field_metrics(current_field, target),
                        "target": target.cpu(),
                    }
                )

        features = torch.cat(feature_rows, dim=0).double()
        targets = torch.cat(target_rows, dim=0).double()
        ones = torch.ones((features.shape[0], 1), dtype=features.dtype)
        design = torch.cat((features, ones), dim=1)
        gram = design.T @ design
        cross = design.T @ targets
        scale = float(torch.trace(gram) / max(gram.shape[0], 1))
        ridge = float(args.ridge_relative) * max(scale, 1.0e-30)
        regularized = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
        weights = torch.linalg.solve(regularized, cross)
        oracle_coefficients = (design @ weights).float()

        start = 0
        for record, target_row in zip(records, target_rows, strict=True):
            count = target_row.shape[0]
            height = width = int(round(math.sqrt(count)))
            if height * width != count:
                raise ValueError("oracle assumes a square saved spatial grid")
            coefficient_field = oracle_coefficients[start : start + count].reshape(
                1, height, width, -1
            ).permute(0, 3, 1, 2).to(device)
            with torch.inference_mode():
                oracle_field = conditioner.synthesize_direct_frequency_output(
                    coefficient_field,
                    saved_time_values[None],
                    saved_time_values,
                )
            record["output_projection_oracle"] = _field_metrics(
                oracle_field, record.pop("target").to(device)
            )
            start += count

        coefficient_error = (design @ weights - targets).square().sum().sqrt()
        coefficient_scale = targets.square().sum().sqrt().clamp_min(1.0e-30)
        report = {
            "schema": "helmholtz_output_projection_oracle_v1",
            "source_h5": str(args.source_h5.resolve()),
            "background_caches": [str(path.resolve()) for path in args.background_cache],
            "travel_time_h5": str(args.travel_time_h5.resolve()),
            "normalization_json": str(args.normalization_json.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "direct_frequencies": int(args.direct_frequencies),
            "propagation_modes": int(args.propagation_modes),
            "ridge_relative": float(args.ridge_relative),
            "ridge_absolute": ridge,
            "design_rows": int(design.shape[0]),
            "design_columns": int(design.shape[1]),
            "coefficient_relative_l2": float(coefficient_error / coefficient_scale),
            "checkpoint_metadata": checkpoint_metadata,
            "records": records,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n", encoding="utf8")
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
