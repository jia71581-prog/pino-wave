#!/usr/bin/env python
"""Gated pilot and production training for grouped dual-head V2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v2.config import V2Config
from grouped_ufno_mionet_v2.data.batch import pack_v2_groups
from grouped_ufno_mionet_v2.data.cache import StructuredCacheDataset
from grouped_ufno_mionet_v2.model.operator import DualHeadWaveOperator
from grouped_ufno_mionet_v2.normalization import PhysicalNormalizer
from grouped_ufno_mionet_v2.training.checkpoint import AtomicCheckpointManager
from grouped_ufno_mionet_v2.training.trainer import JointDualHeadTrainer


def validate_launch_gate(path, *, require_production_authorized=False):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"required overfit gate is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"overfit gate is unreadable: {path}") from error
    if payload.get("passed") is not True:
        raise RuntimeError(f"overfit gate did not pass: {path}")
    if require_production_authorized and payload.get("production_authorized") is not True:
        raise RuntimeError(f"pilot did not authorize production: {path}")
    return payload


def _atomic_json(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf8")
    os.replace(temporary, path)


def _loader(dataset, config, *, train):
    workers = config.train.workers
    kwargs = dict(dataset=dataset, batch_size=config.train.batch_records, shuffle=train,
                  num_workers=workers, collate_fn=pack_v2_groups, pin_memory=torch.cuda.is_available(),
                  persistent_workers=workers > 0, drop_last=train,
                  generator=torch.Generator().manual_seed(config.train.seed + (0 if train else 1)))
    if workers > 0:
        kwargs["prefetch_factor"] = config.train.prefetch_factor
    return DataLoader(**kwargs)


def _validate(trainer, loader, limit):
    aggregate = []
    for index, batch in enumerate(loader):
        aggregate.append(trainer.run_step(batch, train=False))
        if index + 1 >= limit: break
    if not aggregate: raise RuntimeError("validation loader produced no batches")
    return {key: float(np.mean([row[key] for row in aggregate])) for key in aggregate[0]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--gate", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config_path = Path(args.config); config_bytes = config_path.read_bytes(); config = V2Config.from_yaml(config_path)
    if config.train.run_kind not in {"pilot", "production"}:
        raise ValueError("train.run_kind must be pilot or production")
    gate = validate_launch_gate(args.gate, require_production_authorized=config.train.run_kind == "production")
    if config.data.train_cache == config.data.validation_cache:
        raise RuntimeError("train and validation caches must be distinct")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise RuntimeError(f"fresh V2 output directory required: {output}")
    seed = config.train.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    normalizer_payload = json.loads(Path(config.data.normalization_json).read_text(encoding="utf8"))
    normalizer = PhysicalNormalizer.from_dict(normalizer_payload)
    train_dataset = StructuredCacheDataset(config.data.train_cache, expected_split="train")
    validation_dataset = StructuredCacheDataset(config.data.validation_cache, expected_split="validation")
    train_loader = _loader(train_dataset, config, train=True); validation_loader = _loader(validation_dataset, config, train=False)
    model = DualHeadWaveOperator(width=config.model.width, rank=config.model.rank, modes=config.model.modes,
                                 heads=config.model.heads, dense_time_block=config.model.dense_time_block).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate,
                                  weight_decay=config.train.weight_decay)
    trainer = JointDualHeadTrainer(model, normalizer, optimizer, config.loss, device=device,
                                   audit_every=config.train.validation_every)
    checkpoint = AtomicCheckpointManager(output); start_epoch = 0
    if args.resume:
        state = checkpoint.restore(args.resume, model, optimizer, map_location=device)
        trainer.step = int(state["step"]); start_epoch = int(state["epoch"])
    metadata = train_dataset[0].metadata
    dataset_digests = {"train": str(metadata["source_dataset_sha256"]),
                       "normalization": str(metadata["normalization_sha256"])}
    config_digest = hashlib.sha256(config_bytes).hexdigest(); zero = {"query": 1.0, "dense": 1.0}
    last_validation = None
    for epoch in range(start_epoch, config.train.epochs):
        for batch in train_loader:
            metrics = trainer.run_step(batch, train=True)
            if trainer.step % 10 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": trainer.step, **metrics}, sort_keys=True), flush=True)
            if trainer.step >= config.train.max_steps: break
        last_validation = _validate(trainer, validation_loader, config.train.validation_batches)
        score = last_validation["query_relative_l2"] + last_validation["dense_relative_l2"]
        checkpoint.save_epoch(model=model, optimizer=optimizer, epoch=epoch + 1, step=trainer.step,
                              validation_score=score, validation_metrics=last_validation,
                              normalizer=normalizer.metadata.to_dict(), dataset_digests=dataset_digests,
                              config_digest=config_digest, zero_baseline=zero)
        print(json.dumps({"event": "validation", "epoch": epoch + 1, "step": trainer.step, **last_validation}, sort_keys=True), flush=True)
        if trainer.step >= config.train.max_steps: break
    if last_validation is None: raise RuntimeError("training completed without validation")
    if config.train.run_kind == "pilot":
        query = last_validation["query_relative_l2"]; dense = last_validation["dense_relative_l2"]
        authorized = all(math.isfinite(value) and value < 1.0 for value in (query, dense))
        pilot_gate = {"passed": authorized, "production_authorized": authorized,
                      "query_relative_l2": query, "dense_relative_l2": dense,
                      "zero_query_relative_l2": 1.0, "zero_dense_relative_l2": 1.0,
                      "step": trainer.step, "source_gate": str(Path(args.gate).resolve()),
                      "config_sha256": config_digest}
        _atomic_json(pilot_gate, output / "pilot_gate.json")
        if not authorized: raise SystemExit(3)


if __name__ == "__main__":
    main()
