"""Atomic, identity-bound V3 checkpoints with deterministic RNG restoration."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT = "phase_aligned_complex_fno_mionet_v3"


@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    global_step: int
    metrics: Mapping[str, float]
    manifest_digest: str
    config_digest: str


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        saved_cuda_states = [value.cpu() for value in state["torch_cuda"]]
        visible_devices = torch.cuda.device_count()
        if len(saved_cuda_states) < visible_devices:
            raise ValueError(
                "checkpoint has fewer CUDA RNG states than visible devices"
            )
        # A resume may use fewer DDP ranks to reduce runtime host-memory pressure.
        # Preserve the RNG streams for the still-visible device prefix; surplus
        # states belong to devices that cannot be addressed in this process.
        torch.cuda.set_rng_state_all(saved_cuda_states[:visible_devices])


def save_checkpoint_atomic(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    manifest_digest: str,
    config_digest: str,
    metrics: Mapping[str, float],
) -> Path:
    if not manifest_digest or not config_digest:
        raise ValueError("checkpoint manifest and config digests are required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "manifest_digest": str(manifest_digest),
        "config_digest": str(config_digest),
        "metrics": {str(key): float(value) for key, value in metrics.items()},
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "rng_state": _rng_state(),
    }
    try:
        with partial.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected_manifest_digest: str,
    expected_config_digest: str,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
    allowed_missing_prefixes: Sequence[str] = (),
) -> CheckpointMetadata:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    checkpoint_format = str(payload.get("format", ""))
    if checkpoint_format != CHECKPOINT_FORMAT:
        if "v2" in checkpoint_format.lower():
            raise ValueError("V2 checkpoint format is incompatible with V3")
        raise ValueError(f"unexpected V3 checkpoint format: {checkpoint_format!r}")
    if payload.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("checkpoint manifest digest does not match the active dataset")
    if payload.get("config_digest") != expected_config_digest:
        raise ValueError("checkpoint config digest does not match the active experiment")
    prefixes = tuple(str(value) for value in allowed_missing_prefixes)
    if any(not value for value in prefixes):
        raise ValueError("allowed missing model key prefixes must be nonempty")
    if prefixes:
        if optimizer is not None:
            raise ValueError(
                "optimizer state cannot be restored across an expanded model schema"
            )
        incompatible = model.load_state_dict(payload["model_state"], strict=False)
        unexpected = tuple(incompatible.unexpected_keys)
        missing = tuple(incompatible.missing_keys)
        if unexpected:
            raise ValueError(f"unexpected model keys in checkpoint: {unexpected}")
        forbidden = tuple(
            key for key in missing if not any(key.startswith(prefix) for prefix in prefixes)
        )
        if forbidden:
            raise ValueError(f"missing model keys are not allow-listed: {forbidden}")
    else:
        model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is None:
            raise ValueError("checkpoint has no optimizer state to restore")
        optimizer.load_state_dict(optimizer_state)
    if restore_rng:
        if "rng_state" not in payload:
            raise ValueError("checkpoint has no RNG state")
        _restore_rng(payload["rng_state"])
    return CheckpointMetadata(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        metrics=dict(payload.get("metrics", {})),
        manifest_digest=str(payload["manifest_digest"]),
        config_digest=str(payload["config_digest"]),
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointMetadata",
    "load_checkpoint",
    "save_checkpoint_atomic",
]
