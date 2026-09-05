from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch


def fixed_validation_indices(
    sample_id: str,
    *,
    seed: int,
    shape: tuple[int, int, int],
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(*shape, count) <= 0:
        raise ValueError("validation dimensions must be positive")
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little", signed=False)
    generator = np.random.default_rng(local_seed)
    nt, nz, nx = shape
    return (
        generator.integers(0, nt, count, dtype=np.int64),
        generator.integers(0, nz, count, dtype=np.int64),
        generator.integers(0, nx, count, dtype=np.int64),
    )


def _core_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    relative = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(target).clamp_min(1.0e-12)
    nrmse = error.square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1.0e-12)
    pred_centered = prediction - prediction.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    correlation = (pred_centered * target_centered).sum(dim=-1) / (
        torch.linalg.vector_norm(pred_centered, dim=-1)
        * torch.linalg.vector_norm(target_centered, dim=-1)
    ).clamp_min(1.0e-12)
    pred_spectrum = torch.fft.rfft(prediction, dim=-1)
    target_spectrum = torch.fft.rfft(target, dim=-1)
    spectral = torch.linalg.vector_norm(pred_spectrum - target_spectrum) / torch.linalg.vector_norm(
        target_spectrum
    ).clamp_min(1.0e-12)
    return {
        "relative_l2": float(relative),
        "normalized_rmse": float(nrmse),
        "trace_correlation": float(correlation.mean()),
        "spectral_error": float(spectral),
    }


def validation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    medium_types: tuple[str, ...],
) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.shape[0] != len(medium_types):
        raise ValueError("validation prediction/metadata shapes differ")
    result: dict[str, Any] = _core_metrics(prediction, target)
    per_medium: dict[str, dict[str, float]] = {}
    for medium in sorted(set(medium_types)):
        indices = [index for index, value in enumerate(medium_types) if value == medium]
        per_medium[medium] = _core_metrics(prediction[indices], target[indices])
    result["per_medium"] = per_medium
    return result
