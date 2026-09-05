#!/usr/bin/env python
"""End-to-end CPU/CUDA shape, loss, and gradient smoke for V3."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.losses import V3LossWeights, compute_v3_losses
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer, ScaleMetadata
from grouped_ufno_mionet_v3.training.trainer import GuardedV3Trainer
from scripts.train_grouped_v3 import build_model


def _normalizer() -> PhysicalNormalizer:
    return PhysicalNormalizer(
        ScaleMetadata(
            velocity_center_mps=2200.0,
            velocity_scale_mps=500.0,
            pressure_scale_pa=1.0,
            source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
            train_manifest_sha256="synthetic-structural-smoke",
            allowed_medium_types=("uniform", "layered", "marmousi"),
            record_count=2,
            algorithm="synthetic_structural_smoke",
        )
    )


def _synthetic_batch(device: torch.device):
    size = 17
    x = torch.linspace(0.0, 2000.0, size, device=device)
    z = torch.linspace(0.0, 2000.0, size, device=device)
    velocity = torch.full((2, 1, size, size), 2000.0, device=device)
    velocity[1, :, size // 2 :] = 2500.0
    source = torch.tensor(
        [[500.0, 250.0, 10.0, 0.10, 1.0], [1375.0, 375.0, 15.0, 0.12, 1.5]],
        device=device,
    )
    source_map = torch.zeros(2, 1, size, size, device=device)
    source_map[0, 0, 2, 4] = 1.0
    source_map[1, 0, 3, 11] = 1.0
    times = torch.tensor([0.18, 0.42, 0.74], device=device)
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    targets = []
    for record in range(2):
        radius = torch.sqrt(
            (xx - source[record, 0]).square() + (zz - source[record, 1]).square() + 1.0e-8
        )
        effective_velocity = 2000.0 if record == 0 else 2250.0
        tau = times[:, None, None] - source[record, 3] - radius[None] / effective_velocity
        phase = 2.0 * math.pi * source[record, 2] * tau
        envelope = torch.exp(-0.5 * (tau / 0.12).square())
        surface = torch.tanh(z / 20.0).square()[None, :, None]
        targets.append(torch.sin(phase) * envelope * surface)
    dense_target = torch.stack(targets)
    time_index = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    z_index = torch.tensor([3, 7, 5, 10, 8, 14], device=device)
    x_index = torch.tensor([2, 9, 13, 4, 11, 15], device=device)
    query_coords = torch.stack(
        (x[x_index], z[z_index], times[time_index]), dim=-1
    )[None].expand(2, -1, -1)
    query_target = torch.stack(
        [dense_target[record, time_index, z_index, x_index] for record in range(2)]
    )
    probability = torch.full_like(query_target, 1.0 / query_target.shape[1])
    return velocity, source, source_map, times, x, z, dense_target, query_coords, query_target, probability, time_index, z_index, x_index


def run_structural_smoke(
    config: V3Config,
    *,
    device: str,
    checkpoint_dir: str | Path,
) -> dict[str, object]:
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA structural smoke requested but CUDA is unavailable")
    torch.manual_seed(config.train.seed)
    if selected.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)
        torch.cuda.reset_peak_memory_stats(selected)
    model = build_model(config).to(selected)
    normalizer = _normalizer()
    batch = _synthetic_batch(selected)
    (
        velocity,
        source,
        source_map,
        times,
        x,
        z,
        dense_target,
        query_coords,
        query_target,
        probability,
        time_index,
        z_index,
        x_index,
    ) = batch
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    trainer = GuardedV3Trainer(
        model,
        optimizer,
        checkpoint_dir=checkpoint_dir,
        manifest_digest="synthetic-structural-smoke",
        config_digest=config.digest(),
        gradient_clip=config.train.gradient_clip,
    )
    weights = V3LossWeights(
        point=config.loss.point,
        frame=config.loss.frame,
        complex_spectrum=config.loss.complex_spectrum,
        spectral_phase=config.loss.spectral_phase,
        spatial_gradient=config.loss.spatial_gradient,
        time_difference=config.loss.time_difference,
        consistency=config.loss.consistency,
    )
    captured: dict[str, object] = {}

    def closure() -> torch.Tensor:
        prepared = model.prepare_sources(
            model.encode_medium(velocity, normalizer),
            source,
            source_map,
            normalizer,
            record_to_medium=torch.tensor([0, 1], device=selected),
        )
        prediction_query = model.query_normalized(prepared, query_coords)
        prediction_dense = model.dense_normalized(prepared, times, x_m=x, z_m=z)
        dense_at_query = torch.stack(
            [
                prediction_dense[record, time_index, z_index, x_index]
                for record in range(prediction_dense.shape[0])
            ]
        )
        result = compute_v3_losses(
            prediction_query=prediction_query,
            target_query=query_target,
            query_probability=probability,
            prediction_dense=prediction_dense,
            target_dense=dense_target,
            dense_at_query=dense_at_query,
            weights=weights,
            phase_energy_fraction=config.loss.phase_energy_fraction,
        )
        captured["query_shape"] = list(prediction_query.shape)
        captured["dense_shape"] = list(prediction_dense.shape)
        captured["phase_mask_count"] = result.phase_mask_count
        return result.total

    loss = trainer.train_step(closure)
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
        peak = torch.cuda.max_memory_allocated(selected) / (1024.0**2)
    else:
        peak = 0.0
    gradient_groups = {
        name: len(parameters) for name, parameters in model.required_gradient_groups().items()
    }
    return {
        "device": selected.type,
        "loss": float(loss),
        "global_step": trainer.global_step,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "gradient_groups": gradient_groups,
        "peak_memory_mib": peak,
        **captured,
    }


def _write_json_atomic(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial.{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--output",
        default="artifacts/grouped_ufno_mionet_v3/smoke/report.json",
    )
    args = parser.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    output = Path(args.output)
    report = run_structural_smoke(config, device=args.device, checkpoint_dir=output.parent / "checkpoints")
    _write_json_atomic(report, output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
