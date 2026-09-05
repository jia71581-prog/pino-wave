#!/usr/bin/env python
"""Mandatory source-isolated eight-record overfit gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v2.config import V2Config
from grouped_ufno_mionet_v2.data.batch import pack_v2_groups
from grouped_ufno_mionet_v2.data.cache import StructuredCacheDataset, V2CacheRecord
from grouped_ufno_mionet_v2.model.operator import DualHeadWaveOperator
from grouped_ufno_mionet_v2.normalization import PhysicalNormalizer, ScaleMetadata
from grouped_ufno_mionet_v2.training.checkpoint import AtomicCheckpointManager
from grouped_ufno_mionet_v2.training.trainer import JointDualHeadTrainer


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]
    query_rel_l2: float
    dense_rel_l2: float
    zero_query_rel_l2: float
    zero_dense_rel_l2: float
    missing_gradients: tuple[str, ...]


def evaluate_overfit_gate(query_rel_l2, dense_rel_l2, zero_query_rel_l2,
                          zero_dense_rel_l2, missing_gradients, threshold=.10):
    reasons = []
    metrics = (query_rel_l2, dense_rel_l2, zero_query_rel_l2, zero_dense_rel_l2)
    if not all(math.isfinite(float(value)) for value in metrics):
        reasons.append("nonfinite metric")
    if float(query_rel_l2) >= min(float(threshold), float(zero_query_rel_l2)):
        reasons.append("query head did not pass threshold and zero baseline")
    if float(dense_rel_l2) >= min(float(threshold), float(zero_dense_rel_l2)):
        reasons.append("dense head did not pass threshold and zero baseline")
    if missing_gradients:
        reasons.append("required parameters are missing gradients")
    return GateDecision(not reasons, tuple(reasons), float(query_rel_l2), float(dense_rel_l2),
                        float(zero_query_rel_l2), float(zero_dense_rel_l2), tuple(missing_gradients))


def _atomic_json(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf8")
    os.replace(temporary, path)


def _seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _synthetic_records(count=2):
    records = []
    time = torch.linspace(.1, .5, 5)
    for index in range(count):
        velocity = torch.full((1, 17, 17), 2100. + 100 * (index % 2))
        source = torch.tensor([500. + index * 100, 600., 10., .05, 1.])
        source_map = torch.zeros(1, 17, 17); source_map[0, 5, 5 + index] = 1
        dense = torch.randn(2, 17, 17) * 1e-8
        receivers = torch.randn(2, 5) * 1e-8
        coords = torch.tensor([[500., 500., .1], [900., 700., .2], [1200., 800., .4], [1600., 900., .5]])
        records.append(V2CacheRecord(
            velocity, source, source_map, torch.tensor([0, 3]), dense,
            torch.tensor([[2, 3], [2, 12]]), receivers, coords, torch.randn(4) * 1e-8,
            torch.full((4,), .25), time, f"synthetic-{index}", f"g-{index}",
            "uniform" if index == 0 else "layered", {"split": "train", "source_dataset_sha256": "synthetic", "normalization_sha256": "synthetic"},
        ))
    return records


def _select_records(dataset, count=8):
    if len(dataset) < count:
        raise ValueError(f"overfit gate needs {count} records, cache has {len(dataset)}")
    candidates = []
    for index in np.linspace(0, len(dataset) - 1, min(len(dataset), 256), dtype=int):
        record = dataset[int(index)]
        energy = float(record.dense_target.float().square().mean().sqrt())
        if energy > 0 and torch.isfinite(record.dense_target).all():
            candidates.append((record.medium_type, int(index), energy))
    chosen, used = [], set()
    families = sorted({family for family, _, _ in candidates})
    for pass_index in range(count):
        family = families[pass_index % len(families)]
        available = [item for item in candidates if item[0] == family and item[1] not in used]
        if not available:
            available = [item for item in candidates if item[1] not in used]
        if not available: break
        _, index, _ = max(available, key=lambda item: item[2])
        used.add(index); chosen.append(dataset[index])
    if len(chosen) != count or len({item.medium_type for item in chosen}) < 2:
        raise ValueError("could not select eight energetic records from at least two medium families")
    return chosen


def _batches(records, batch_size):
    return [pack_v2_groups(records[start:start + batch_size]) for start in range(0, len(records), batch_size)]


def _mean_metrics(trainer, batches):
    metrics = [trainer.run_step(batch, train=False) for batch in batches]
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def restore_for_finetune(checkpoint, path, model, optimizer, trainer, *, learning_rate,
                         restore_optimizer=True):
    """Restore weights and AdamW moments, then start a fresh low-LR schedule."""
    state = checkpoint.restore(
        path, model, optimizer if restore_optimizer else None,
        map_location=trainer.device,
    )
    trainer.step = int(state["step"])
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
        group["initial_lr"] = float(learning_rate)
    return state


def build_optimizer(config, model):
    if config.train.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    return torch.optim.LBFGS(
        model.parameters(), lr=config.train.learning_rate,
        max_iter=config.train.lbfgs_max_iter,
        history_size=config.train.lbfgs_history_size,
        line_search_fn=config.train.lbfgs_line_search_fn,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config_path = Path(args.config); config_bytes = config_path.read_bytes(); config = V2Config.from_yaml(config_path)
    _seed_everything(config.train.seed)
    if config.train.smoke:
        normalizer = PhysicalNormalizer(ScaleMetadata(2500, 1000, 1e-8, (2000, 2000, 50, 1.2, 1), "synthetic"))
        records = _synthetic_records(2)
    else:
        normalization_payload = json.loads(Path(config.data.normalization_json).read_text(encoding="utf8"))
        normalizer = PhysicalNormalizer.from_dict(normalization_payload)
        dataset = StructuredCacheDataset(config.data.train_cache, expected_split="train")
        records = _select_records(dataset, 8)
    output = Path(args.output or ("artifacts/grouped_ufno_mionet_v2/smoke" if config.train.smoke else "artifacts/grouped_ufno_mionet_v2/overfit"))
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA overfit gate requested but CUDA is unavailable")
    model = DualHeadWaveOperator(
        width=config.model.width, rank=config.model.rank, modes=config.model.modes,
        heads=config.model.heads, dense_time_block=config.model.dense_time_block,
    ).to(device)
    optimizer = build_optimizer(config, model)
    trainer = JointDualHeadTrainer(model, normalizer, optimizer, config.loss, device=device,
                                   audit_every=config.train.validation_every)
    batches = _batches(records, config.train.batch_records)
    history = []; last = None
    checkpoint = AtomicCheckpointManager(output)
    if args.resume:
        restored = restore_for_finetune(
            checkpoint, args.resume, model, optimizer, trainer,
            learning_rate=config.train.learning_rate,
            restore_optimizer=config.train.optimizer == "adamw",
        )
        print(json.dumps({"resumed_from": str(args.resume), "restored_step": restored["step"],
                          "learning_rate": config.train.learning_rate}), flush=True)
    scheduler = None
    if config.train.optimizer == "adamw":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.train.max_steps), eta_min=config.train.learning_rate * .05
        )
    zero = {"query": 1.0, "dense": 1.0}
    dataset_digests = {"train": records[0].metadata["source_dataset_sha256"],
                       "normalization": records[0].metadata["normalization_sha256"]}
    max_steps = max(1, config.train.max_steps)
    for local_step in range(max_steps):
        batch = batches[local_step % len(batches)]
        train_metrics = (trainer.run_lbfgs_step(batches) if config.train.optimizer == "lbfgs"
                         else trainer.run_step(batch, train=True))
        if scheduler is not None:
            scheduler.step()
        if (local_step + 1) % config.train.validation_every == 0 or local_step + 1 == max_steps:
            last = _mean_metrics(trainer, batches); last["step"] = trainer.step; history.append(last)
            print(json.dumps(last, sort_keys=True), flush=True)
            score = last["query_relative_l2"] + last["dense_relative_l2"]
            checkpoint.save_epoch(
                model=model, optimizer=optimizer, epoch=0, step=trainer.step,
                validation_score=score, validation_metrics=last,
                normalizer=normalizer.metadata.to_dict(), dataset_digests=dataset_digests,
                config_digest=hashlib.sha256(config_bytes).hexdigest(), zero_baseline=zero,
                scheduler=scheduler,
            )
            decision = evaluate_overfit_gate(last["query_relative_l2"], last["dense_relative_l2"], 1., 1., [])
            if decision.passed and not config.train.smoke:
                break
    assert last is not None
    decision = evaluate_overfit_gate(last["query_relative_l2"], last["dense_relative_l2"], 1., 1., [])
    payload = asdict(decision) | {
        "structural_smoke": bool(config.train.smoke), "steps": int(last["step"]),
        "sample_ids": [record.sample_id for record in records],
        "medium_families": sorted({record.medium_type for record in records}),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "history": history,
    }
    if config.train.smoke:
        payload["passed"] = True; payload["reasons"] = ["structural smoke only; accuracy gate not evaluated"]
    _atomic_json(payload, output / "gate.json")
    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(8, 3))
        steps = [item["step"] for item in history]
        axes[0].plot(steps, [item["query_relative_l2"] for item in history]); axes[0].set_title("query relative L2")
        axes[1].plot(steps, [item["dense_relative_l2"] for item in history]); axes[1].set_title("dense relative L2")
        for axis in axes: axis.set_xlabel("step"); axis.grid(alpha=.2)
        figure.tight_layout(); figure.savefig(output / "loss_history.png", dpi=160); plt.close(figure)
    except ImportError:
        pass
    if not payload["passed"]:
        raise SystemExit(2)
    print(json.dumps({"gate": str(output / "gate.json"), "passed": True}), flush=True)


if __name__ == "__main__":
    main()
