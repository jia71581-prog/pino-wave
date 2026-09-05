from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..config import DomainConfig, ModelConfig, TrainingConfig
from ..data.dataset import ContinuousWaveDataset
from ..model import ContinuousWaveOperator
from ..training.trainer import Trainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the continuous multi-shot wave operator")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-workers", type=int, help="runtime HDF5 reader processes")
    parser.add_argument("--prefetch-batches", type=int, help="runtime CPU batches prepared ahead")
    parser.add_argument("--time-frames", type=int, help="runtime frames sampled per example")
    parser.add_argument("--points-per-frame", type=int, help="runtime receivers sampled per frame")
    parser.add_argument("--validate-every", type=int, help="runtime validation interval")
    parser.add_argument("--checkpoint-every", type=int, help="runtime checkpoint interval")
    return parser


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def main() -> None:
    args = _parser().parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    model_values = dict(raw["model"])
    model_values["spectral_modes"] = tuple(model_values["spectral_modes"])
    training_values = dict(raw["training"])
    training_values["spatial_ais_shape"] = tuple(training_values["spatial_ais_shape"])
    domain = DomainConfig(**raw["domain"])
    model_config = ModelConfig(**model_values)
    training_config = TrainingConfig(**training_values)
    runtime_overrides = {
        name: value
        for name, value in (
            ("data_workers", args.data_workers),
            ("prefetch_batches", args.prefetch_batches),
            ("time_frames", args.time_frames),
            ("points_per_frame", args.points_per_frame),
            ("validate_every", args.validate_every),
            ("checkpoint_every", args.checkpoint_every),
        )
        if value is not None
    }
    if runtime_overrides:
        training_config = replace(training_config, **runtime_overrides)
    dataset_path = Path(raw["dataset"]["path"]).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    train_dataset = ContinuousWaveDataset(dataset_path, split="train")
    validation_dataset = ContinuousWaveDataset(dataset_path, split="validation")
    dataset_binding = {
        "path": str(dataset_path),
        "size": dataset_path.stat().st_size,
        "mtime_ns": dataset_path.stat().st_mtime_ns,
    }
    trainer = Trainer(
        model=ContinuousWaveOperator(model_config, domain),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=training_config,
        output_dir=args.output,
        device=device,
        config_digest=_digest(raw),
        dataset_digest=_digest(dataset_binding),
    )
    if args.resume is not None:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
