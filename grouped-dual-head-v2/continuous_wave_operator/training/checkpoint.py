from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adaptive_sampling import HierarchicalAIS


@dataclass(frozen=True)
class RestoredTrainingState:
    epoch: int
    global_step: int
    normalization: dict[str, Any]
    ais: HierarchicalAIS


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ais: HierarchicalAIS,
    normalization: dict[str, Any],
    config_digest: str,
    dataset_digest: str,
    epoch: int,
    global_step: int,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    state = {
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "ais": ais.state_dict(),
        "normalization": normalization,
        "config_digest": str(config_digest),
        "dataset_digest": str(dataset_digest),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_config_digest: str,
    expected_dataset_digest: str,
    map_location: str | torch.device = "cpu",
) -> RestoredTrainingState:
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    if state.get("config_digest") != expected_config_digest:
        raise ValueError("checkpoint config digest does not match")
    if state.get("dataset_digest") != expected_dataset_digest:
        raise ValueError("checkpoint dataset digest does not match")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state["scheduler"] is None:
            raise ValueError("checkpoint is missing scheduler state")
        scheduler.load_state_dict(state["scheduler"])
    ais = HierarchicalAIS.from_state_dict(state["ais"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if torch.cuda.is_available() and state["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
    return RestoredTrainingState(
        epoch=int(state["epoch"]),
        global_step=int(state["global_step"]),
        normalization=dict(state["normalization"]),
        ais=ais,
    )
