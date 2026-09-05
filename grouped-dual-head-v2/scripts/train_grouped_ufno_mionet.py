#!/usr/bin/env python
"""Train the new grouped single-source operator from the unified HDF5 VDS."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet import OperatorConfig, GroupedSingleSourceUFNOMIONetOperator
from grouped_ufno_mionet.data import (
    GroupedWavefieldDataset, GroupedBatchSampler, SparseCacheDataset, pack_groups,
)
from grouped_ufno_mionet.training import GroupedTrainer
from grouped_ufno_mionet.training.checkpoint import load_checkpoint, save_checkpoint


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="grouped_ufno_mionet/configs/production.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--output", default="artifacts/grouped_ufno_mionet/production")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--sparse-cache", default=None,
                    help="precomputed query cache; bypasses source wavefield VDS reads")
    args = ap.parse_args(argv)
    cfg = OperatorConfig.from_yaml(args.config)
    if args.dataset:
        cfg.data.dataset = args.dataset
    steps = args.max_steps or getattr(cfg.train, "max_steps", 1000)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    dataset_class = SparseCacheDataset if args.sparse_cache else GroupedWavefieldDataset
    dataset_path = args.sparse_cache or cfg.data.dataset
    dataset = dataset_class(dataset_path, split=args.split,
                            frames=cfg.data.frames_per_record,
                            blocks=cfg.data.query_blocks,
                            receivers=cfg.data.receivers_per_frame,
                            return_full_field=False, seed=cfg.train.seed)
    sampler = GroupedBatchSampler(dataset, batch_size=cfg.train.batch_records,
                                  shuffle=True, seed=cfg.train.seed)
    loader_kwargs = dict(batch_sampler=sampler, num_workers=cfg.train.num_workers,
                         pin_memory=(device == "cuda"),
                         collate_fn=lambda xs: pack_groups(xs, cfg.data.max_records_per_macro_batch))
    if cfg.train.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(dataset, **loader_kwargs)
    model = GroupedSingleSourceUFNOMIONetOperator(cfg).to(device)
    trainer = GroupedTrainer(model, cfg, device=device)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        payload = load_checkpoint(args.resume, model, trainer.optimizer, map_location=device)
        saved = payload.get("state", {})
        trainer.state.global_step = int(saved.get("global_step", 0))
        trainer.state.optimizer_steps = int(saved.get("optimizer_steps", trainer.state.global_step))
    else:
        saved = {}
    log_path = out / "train.log"
    epoch = int(saved.get("epoch", 0))
    while trainer.state.global_step < steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            result = trainer.train_macro_batch(batch)
            result["step"] = trainer.state.global_step
            with log_path.open("a", encoding="utf8") as handle:
                handle.write(str(result) + "\n")
            print(result, flush=True)
            if trainer.state.global_step >= steps:
                break
        epoch += 1
        if len(loader) == 0:
            raise RuntimeError("training DataLoader yielded no batches")
        # A complete data pass is the durable recovery boundary.  Persist the
        # epoch together with optimizer/model state so resume retains shuffle.
        save_checkpoint(out / "checkpoints" / "last.pt", model, trainer.optimizer,
                        state={**vars(trainer.state), "epoch": epoch},
                        extra={"config": str(args.config), "sparse_cache": args.sparse_cache})
    save_checkpoint(out / "checkpoints" / "last.pt", model, trainer.optimizer,
                    state={**vars(trainer.state), "epoch": epoch},
                    extra={"config": str(args.config), "sparse_cache": args.sparse_cache})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
