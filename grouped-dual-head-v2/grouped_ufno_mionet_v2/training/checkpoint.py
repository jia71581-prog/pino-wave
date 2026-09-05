"""Atomic, self-describing best/last V2 checkpoints."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_state(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(state, temporary)
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class AtomicCheckpointManager:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.best_score = float("inf")
        best = self.output_dir / "best.pt"
        if best.exists():
            self.best_score = float(load_state(best)["validation_score"])

    def _state(self, *, model, optimizer, epoch: int, step: int, validation_score: float,
               validation_metrics: dict[str, float], normalizer: dict[str, Any],
               dataset_digests: dict[str, str], config_digest: str,
               zero_baseline: dict[str, float], scaler=None, scheduler=None,
               sampler_state=None) -> dict[str, Any]:
        state = {
            "format": "grouped_dual_head_v2", "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": int(epoch), "step": int(step),
            "validation_score": float(validation_score), "validation_metrics": dict(validation_metrics),
            "normalizer": dict(normalizer), "dataset_digests": dict(dataset_digests),
            "config_digest": str(config_digest), "zero_baseline": dict(zero_baseline),
            "sampler_state": sampler_state, "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        return state

    def save_epoch(self, **kwargs) -> dict[str, Any]:
        state = self._state(**kwargs)
        _atomic_torch_save(state, self.output_dir / "last.pt")
        if state["validation_score"] < self.best_score:
            self.best_score = state["validation_score"]
            _atomic_torch_save(state, self.output_dir / "best.pt")
        return state

    def restore(self, path: str | Path, model, optimizer=None, scaler=None, scheduler=None,
                *, map_location="cpu") -> dict[str, Any]:
        state = load_state(path, map_location=map_location)
        if state.get("format") != "grouped_dual_head_v2":
            raise ValueError("checkpoint is not grouped dual-head V2")
        model.load_state_dict(state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        return state
