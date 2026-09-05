#!/usr/bin/env python3
"""Evaluate per-instance Elastic VTI PINO errors on saved validation splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fno_acoustic.checkpoint import load_checkpoint
from fno_acoustic.config import load_config
from fno_acoustic.data_elastic_vti import PinoHDF5Dataset
from fno_acoustic.model_elastic_vti import ElasticVTIFNO3D
from fno_acoustic.normalization import decode_standard
from reports.no_asvgd_group_meeting_20260710.make_elastic_vti_section import (
    MODEL_SPECS,
    relative_l2_per_component,
    select_validation_indices,
    validate_distribution_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATHS = {
    "uniform": ROOT / "artifacts/elastic_vti_pino_uniform/splits.json",
    "layered": ROOT / "artifacts/elastic_vti_pino_layered/splits.json",
    "marmousi": ROOT / "artifacts/elastic_vti_pino_marmousi_component_balanced_short/splits.json",
    "marmousi_pretrain": ROOT / "artifacts/elastic_vti_pino_marmousi/splits.json",
}
EVALUATION_SPECS = {
    **MODEL_SPECS,
    "marmousi_pretrain": {
        "config": ROOT / "configs/pino_elastic_vti_marmousi_gpu.yaml",
        "checkpoint": ROOT / "artifacts/elastic_vti_pino_marmousi/checkpoints/best.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(EVALUATION_SPECS), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    started = time.perf_counter()
    result = {
        "protocol": {
            "instances_per_model": int(args.count),
            "split": "val",
            "selection": "first instances in each saved validation split order",
            "metric": "physical-space full-field relative L2 per displacement component",
            "device": str(args.device),
        },
        "models": {},
    }
    selected_models = [args.model] if args.model else list(EVALUATION_SPECS)
    for model_name in selected_models:
        spec = EVALUATION_SPECS[model_name]
        indices = select_validation_indices(SPLIT_PATHS[model_name], args.count)
        config = load_config(spec["config"])
        checkpoint = load_checkpoint(spec["checkpoint"], map_location="cpu")
        model_config = {key: value for key, value in checkpoint["model_config"].items() if key != "name"}
        model = ElasticVTIFNO3D(**model_config).to(args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        dataset = PinoHDF5Dataset(
            config,
            indices,
            normalization_stats=checkpoint["normalization_stats"],
            return_normalized=True,
        )
        ux_errors: list[float] = []
        uz_errors: list[float] = []
        model_started = time.perf_counter()
        for offset in range(len(dataset)):
            sample = dataset[offset]
            with torch.inference_mode():
                prediction_normalized = model(sample["input"].unsqueeze(0).to(args.device))[0].cpu()
            target = decode_standard(
                sample["target"],
                checkpoint["normalization_stats"]["wavefield"],
                eps=config["normalization"]["eps"],
            ).cpu()
            prediction = decode_standard(
                prediction_normalized,
                checkpoint["normalization_stats"]["wavefield"],
                eps=config["normalization"]["eps"],
            ).cpu()
            errors = relative_l2_per_component(prediction.numpy(), target.numpy())
            ux_errors.append(errors["ux"])
            uz_errors.append(errors["uz"])
            print(
                f"[{model_name}] {offset + 1:02d}/{len(dataset)} index={indices[offset]} "
                f"ux={errors['ux']:.6f} uz={errors['uz']:.6f}",
                flush=True,
            )
        result["models"][model_name] = {
            "indices": indices,
            "ux_relative_l2": ux_errors,
            "uz_relative_l2": uz_errors,
            "checkpoint": str(spec["checkpoint"]),
            "config": str(spec["config"]),
            "split_path": str(SPLIT_PATHS[model_name]),
            "elapsed_s": float(time.perf_counter() - model_started),
        }
        del model, dataset, checkpoint
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    result["protocol"]["total_elapsed_s"] = float(time.perf_counter() - started)
    validate_distribution_metrics(result, expected_count=args.count, required_models=set(selected_models))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["protocol"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
