#!/usr/bin/env python3
"""Train-only capacity probe for one-scalar parent correction rescaling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.pretrained_subspace import (
    train_only_scalar_multiplier_oracle,
)
from saved_time_phase_operator_v4.losses import apply_hard_causality, source_causality_onset_s
from scripts.run_pretrained_temporal_subspace_adaptation import build_train_pilot_manifest
from scripts.run_v5_instance_adaptation import _load_parent


def time_dilation_direction(
    field: torch.Tensor, time_s: torch.Tensor, source_t0_s: torch.Tensor
) -> torch.Tensor:
    """Linearized source-centered time dilation: ``(t-t0) * dp/dt``."""

    value = torch.as_tensor(field)
    times = torch.as_tensor(time_s, dtype=value.dtype, device=value.device).flatten()
    onset = torch.as_tensor(source_t0_s, dtype=value.dtype, device=value.device).flatten()
    if value.ndim != 4 or times.shape != (value.shape[1],) or onset.shape != (value.shape[0],):
        raise ValueError("time-dilation inputs have incompatible shapes")
    if value.shape[1] < 3 or not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("time dilation requires at least three increasing times")
    derivative = torch.empty_like(value)
    derivative[:, 0] = (value[:, 1] - value[:, 0]) / (times[1] - times[0])
    derivative[:, -1] = (value[:, -1] - value[:, -2]) / (times[-1] - times[-2])
    denominator = (times[2:] - times[:-2])[None, :, None, None]
    derivative[:, 1:-1] = (value[:, 2:] - value[:, :-2]) / denominator
    offset = (times[None, :] - onset[:, None])[:, :, None, None]
    return derivative * offset


def travel_time_warp_direction(
    field: torch.Tensor, time_s: torch.Tensor, travel_time_s: torch.Tensor
) -> torch.Tensor:
    """Linearized spatial arrival-time warp: ``travel_time(x,z) * dp/dt``."""

    value = torch.as_tensor(field)
    times = torch.as_tensor(time_s, dtype=value.dtype, device=value.device).flatten()
    travel = torch.as_tensor(
        travel_time_s, dtype=value.dtype, device=value.device
    )
    if travel.ndim == 2:
        travel = travel[None]
    if value.ndim != 4 or times.shape != (value.shape[1],) or travel.shape != (
        value.shape[0], value.shape[2], value.shape[3]
    ):
        raise ValueError("travel-time warp inputs have incompatible shapes")
    if value.shape[1] < 3 or not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("travel-time warp requires at least three increasing times")
    derivative = torch.empty_like(value)
    derivative[:, 0] = (value[:, 1] - value[:, 0]) / (times[1] - times[0])
    derivative[:, -1] = (value[:, -1] - value[:, -2]) / (times[-1] - times[-2])
    derivative[:, 1:-1] = (value[:, 2:] - value[:, :-2]) / (
        times[2:] - times[:-2]
    )[None, :, None, None]
    return derivative * travel[:, None]


@torch.inference_mode()
def run(
    config_path: str | Path,
    *,
    output: str | Path,
    per_family: int = 2,
    seed: int = 30_372,
    device_name: str = "cuda",
    minimum_multiplier: float = -1.0,
    maximum_multiplier: float = 1.0,
    steps: int = 81,
    direction: str = "correction_scale",
) -> dict[str, object]:
    config_file = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_file.read_text())
    manifest = build_manifest(config["source_h5"])
    prior_per_family = int((config.get("cpadc", {}) or {}).get("per_family", 32))
    rows, excluded = build_train_pilot_manifest(
        manifest,
        per_family=int(per_family),
        seed=int(seed),
        excluded_sample_ids=set(),
        prior_basis_per_family=prior_per_family,
        prior_basis_seed=int(config.get("seed", 372)),
    )
    device = torch.device(device_name)
    parent, normalizer = _load_parent(config, manifest, device)
    dataset = GuardedOnsetDataset(
        config["source_h5"],
        manifest,
        split="train",
        sample_ids=tuple(row.sample_id for row in rows),
        travel_time_h5=config.get("travel_time_h5"),
    )
    reports: list[dict[str, object]] = []
    try:
        for record in dataset:
            velocity = record.velocity_mps.to(device).unsqueeze(0)
            source = record.source_parameters.to(device).unsqueeze(0)
            source_map = record.source_map.to(device).unsqueeze(0)
            prepared = parent.prepare_sources(
                parent.encode_medium(velocity, normalizer),
                source,
                source_map,
                normalizer,
                record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
            )
            dense_grid = parent.prepare_dense_grid(
                prepared,
                x_m=record.x_m.to(device),
                z_m=record.z_m.to(device),
                travel_time_s=(
                    None
                    if record.dense_travel_time_s is None
                    else record.dense_travel_time_s.to(device).unsqueeze(0)
                ),
            )
            predicted, coarse = parent.dense_normalized_with_coarse(
                prepared,
                record.time_s.to(device),
                dense_grid=dense_grid,
                time_block=1,
            )
            onset = source_causality_onset_s(source, lead_cycles=1.0)
            times = record.time_s.to(device)[None, :]
            predicted = apply_hard_causality(predicted, times, onset)
            coarse = apply_hard_causality(coarse, times, onset)
            parent_field = normalizer.decode_pressure(predicted, source[:, 4]).cpu()
            normalized_direction = (
                predicted - coarse
                if str(direction) == "correction_scale"
                else time_dilation_direction(
                    predicted, record.time_s.to(device), source[:, 3]
                )
                if str(direction) == "time_dilation"
                else travel_time_warp_direction(
                    predicted,
                    record.time_s.to(device),
                    record.dense_travel_time_s.to(device),
                )
                if str(direction) == "travel_time_warp"
                else None
            )
            if normalized_direction is None:
                raise ValueError("unknown oracle direction")
            directed_field = normalizer.decode_pressure(
                predicted + normalized_direction, source[:, 4]
            ).cpu()
            with h5py.File(config["source_h5"], "r", swmr=True) as handle:
                truth = torch.from_numpy(
                    np.asarray(handle["wavefield"][record.source_index], dtype=np.float32)
                ).unsqueeze(0)
            oracle = train_only_scalar_multiplier_oracle(
                parent_field,
                directed_field,
                truth,
                record.observed_indices,
                minimum_multiplier=float(minimum_multiplier),
                maximum_multiplier=float(maximum_multiplier),
                steps=int(steps),
            )
            parent_error = float(oracle.parent_relative_l2[0])
            best_error = float(oracle.best_relative_l2[0])
            reports.append(
                {
                    "sample_id": record.sample_id,
                    "family": record.medium_type,
                    "source_index": int(record.source_index),
                    "parent_relative_l2": parent_error,
                    "best_relative_l2": best_error,
                    "direction": str(direction),
                    "best_multiplier": float(oracle.best_multiplier[0]),
                    "effective_parent_correction_scale": (
                        1.0 + float(oracle.best_multiplier[0])
                        if str(direction) == "correction_scale"
                        else None
                    ),
                    "effective_time_dilation": (
                        1.0 + float(oracle.best_multiplier[0])
                        if str(direction) == "time_dilation"
                        else None
                    ),
                    "effective_travel_time_scale": (
                        float(oracle.best_multiplier[0])
                        if str(direction) == "travel_time_warp"
                        else None
                    ),
                    "relative_improvement": (parent_error - best_error)
                    / max(parent_error, 1.0e-12),
                }
            )
    finally:
        dataset.close()
    by_family: dict[str, list[float]] = {}
    for row in reports:
        by_family.setdefault(str(row["family"]), []).append(
            float(row["relative_improvement"])
        )
    payload = {
        "schema": "parent_correction_scale_oracle_v1",
        "scope": "train_only_future_truth_capacity_diagnostic",
        "config": str(config_file),
        "parent_checkpoint": str(Path(config["parent_checkpoint"]).resolve()),
        "manifest_digest": manifest.digest,
        "selection_seed": int(seed),
        "prior_basis_exclusion_count": len(excluded),
        "record_count": len(reports),
        "multiplier_range": [float(minimum_multiplier), float(maximum_multiplier)],
        "steps": int(steps),
        "direction": str(direction),
        "mean_relative_improvement": float(
            np.mean([float(row["relative_improvement"]) for row in reports])
        ),
        "nonworse_fraction": float(
            np.mean([float(row["relative_improvement"]) >= -1.0e-9 for row in reports])
        ),
        "family_mean_relative_improvement": {
            family: float(np.mean(values)) for family, values in sorted(by_family.items())
        },
        "records": reports,
        "validation_opened": False,
        "test_id_opened": False,
    }
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite correction-scale oracle output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument("--seed", type=int, default=30372)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-multiplier", type=float, default=-1.0)
    parser.add_argument("--maximum-multiplier", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=81)
    parser.add_argument(
        "--direction", choices=(
            "correction_scale", "time_dilation", "travel_time_warp"
        ),
        default="correction_scale",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                args.config,
                output=args.output,
                per_family=args.per_family,
                seed=args.seed,
                device_name=args.device,
                minimum_multiplier=args.minimum_multiplier,
                maximum_multiplier=args.maximum_multiplier,
                steps=args.steps,
                direction=args.direction,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
