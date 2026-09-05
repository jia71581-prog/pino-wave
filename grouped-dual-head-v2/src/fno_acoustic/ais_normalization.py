"""Train-only normalization contract for AIS-MQFNO."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .query_data import QueryScene


@dataclass(frozen=True)
class AISNormalizationBinding:
    """Immutable statistics and exact-byte identity for AIS normalization v2."""

    path: Path
    stats_sha256: str
    velocity_mean: float
    velocity_std: float
    wavefield_mean: float
    wavefield_std: float
    eps: float
    contract_id: str = "ais_normalization_v2"

    def encode_wavefield(self, value: Any) -> Any:
        scale = _effective_scale(value, self.wavefield_std, self.eps)
        return (value - self.wavefield_mean) / scale

    def decode_wavefield(self, value: Any) -> Any:
        scale = _effective_scale(value, self.wavefield_std, self.eps)
        return value * scale + self.wavefield_mean


def _effective_scale(reference: Any, scale: Any, eps: float) -> Any:
    """Represent a finite positive scale safely in the reference's precision."""
    if isinstance(reference, torch.Tensor):
        dtype = (
            reference.dtype
            if torch.is_floating_point(reference)
            else torch.get_default_dtype()
        )
        limits = torch.finfo(dtype)
        scale_tensor = torch.as_tensor(scale, dtype=dtype, device=reference.device)
        eps_tensor = torch.as_tensor(eps, dtype=dtype, device=reference.device)
        return torch.maximum(scale_tensor, eps_tensor).clamp(
            min=limits.tiny, max=limits.max
        )

    if isinstance(reference, (np.ndarray, np.generic)):
        reference_dtype = np.asarray(reference).dtype
        dtype = (
            reference_dtype
            if np.issubdtype(reference_dtype, np.floating)
            else np.dtype(np.float64)
        )
        limits = np.finfo(dtype)
        return np.clip(
            np.maximum(np.asarray(scale, dtype=dtype), np.asarray(eps, dtype=dtype)),
            limits.tiny,
            limits.max,
        )

    return max(float(scale), float(eps), sys.float_info.min)


def _required_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _stats_section(payload: dict[str, object], name: str) -> dict[str, object]:
    section = payload.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"normalization stats missing {name} section")
    return section


def load_ais_normalization(path: str | Path) -> AISNormalizationBinding:
    """Load validated train-only statistics and hash their exact file bytes."""
    stats_path = Path(path)
    raw = stats_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("normalization stats must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("normalization stats must be a JSON object")
    if payload.get("computed_from_split") != "train":
        raise ValueError("normalization stats must be computed from the train split")

    velocity = _stats_section(payload, "velocity")
    wavefield = _stats_section(payload, "wavefield")
    return AISNormalizationBinding(
        path=stats_path,
        stats_sha256=hashlib.sha256(raw).hexdigest(),
        velocity_mean=_required_real(velocity.get("mean"), "velocity.mean"),
        velocity_std=_required_real(
            velocity.get("std"), "velocity.std", positive=True
        ),
        wavefield_mean=_required_real(wavefield.get("mean"), "wavefield.mean"),
        wavefield_std=_required_real(
            wavefield.get("std"), "wavefield.std", positive=True
        ),
        eps=_required_real(payload.get("eps"), "eps", positive=True),
    )


def build_normalized_static_features(
    scene: QueryScene, binding: AISNormalizationBinding
) -> torch.Tensor:
    """Build dimensionless velocity, source, gradient, and slowness channels."""
    velocity = scene.velocity_cpu
    source_raw = scene.source_cpu
    if not isinstance(velocity, torch.Tensor) or not isinstance(
        source_raw, torch.Tensor
    ):
        raise TypeError("scene velocity and source must be tensors")
    if velocity.ndim != 2 or source_raw.shape != velocity.shape:
        raise ValueError("scene velocity and source must have the same two-dimensional shape")
    if not torch.is_floating_point(velocity) or not torch.is_floating_point(source_raw):
        raise TypeError("scene velocity and source must be floating-point tensors")
    if min(velocity.shape) < 3:
        raise ValueError("scene must be at least 3 by 3 for normalized gradients")
    if not bool(torch.isfinite(velocity).all()) or not bool(
        torch.isfinite(source_raw).all()
    ):
        raise ValueError("scene velocity and source must be finite")
    if not bool((velocity > 0).all()):
        raise ValueError("scene velocity must be strictly positive")

    source_raw = source_raw.to(dtype=velocity.dtype, device=velocity.device)
    if not bool(torch.isfinite(source_raw).all()):
        raise ValueError("scene source must remain finite after dtype conversion")

    velocity_scale = _effective_scale(
        velocity, binding.velocity_std, binding.eps
    )
    velocity_hat = (velocity - binding.velocity_mean) / velocity_scale
    source_scale = _effective_scale(
        velocity, source_raw.abs().max(), binding.eps
    )
    source = source_raw / source_scale
    xi = torch.linspace(
        -1.0, 1.0, velocity.shape[0], dtype=velocity.dtype, device=velocity.device
    )
    zeta = torch.linspace(
        -1.0, 1.0, velocity.shape[1], dtype=velocity.dtype, device=velocity.device
    )
    grad_x, grad_z = torch.gradient(
        velocity_hat, spacing=(xi, zeta), edge_order=2
    )
    slow_contrast = (binding.velocity_mean / velocity).square() - 1.0
    features = torch.stack((velocity_hat, source, grad_x, grad_z, slow_contrast))[None]
    if not bool(torch.isfinite(features).all()):
        raise ValueError("normalized static features must all be finite")
    return features
