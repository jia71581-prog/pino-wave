from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from ..config import DomainConfig, ModelConfig
from ..data.dataset import ContinuousWaveDataset
from ..data.query_sampling import QuerySampler
from ..model import ContinuousWaveOperator
from ..training.validation import validation_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a continuous wave operator checkpoint")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_values = dict(raw["model"])
    model_values["spectral_modes"] = tuple(model_values["spectral_modes"])
    model = ContinuousWaveOperator(ModelConfig(**model_values), DomainConfig(**raw["domain"]))
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    dataset = ContinuousWaveDataset(raw["dataset"]["path"], split=args.split)
    count = min(args.samples, len(dataset))
    sampler = QuerySampler(seed=2026, time_frames=16, points_per_frame=256)
    batch = dataset.sample_queries(list(range(count)), sampler)
    with torch.no_grad():
        prediction = model(
            batch.velocity_mps.to(args.device),
            batch.source_map.to(args.device),
            batch.source_parameters.to(args.device),
            batch.query_coords.to(args.device),
            chunk_size=4096,
        )
    metrics = validation_metrics(
        prediction[:, 0].cpu(), batch.target_pressure[:, 0], medium_types=batch.medium_types
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()

