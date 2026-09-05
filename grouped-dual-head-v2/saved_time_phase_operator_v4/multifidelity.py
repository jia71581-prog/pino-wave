"""Numerical-teacher cache and additive multi-fidelity distillation losses."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from .losses import (
    RelativeEnergySquaredReference,
    relative_energy_squared_block_loss,
    relative_energy_squared_reference,
)


TEACHER_CACHE_SCHEMA = "lwc84_multifidelity_teacher_v1"


def fixed_teacher_time_indices(
    *, stored_time_count: int = 401, count: int = 64
) -> tuple[int, ...]:
    """Return a deterministic exact-time pool spanning the complete saved axis."""

    times = int(stored_time_count)
    selected = int(count)
    if times < 2 or selected < 2 or selected > times:
        raise ValueError("teacher time count must lie in [2, stored_time_count]")
    indices = np.rint(np.linspace(0, times - 1, selected)).astype(np.int64)
    values = tuple(int(value) for value in indices)
    if (
        len(values) != selected
        or len(set(values)) != selected
        or values[0] != 0
        or values[-1] != times - 1
        or any(left >= right for left, right in zip(values, values[1:]))
    ):
        raise RuntimeError("teacher time construction did not produce a strict span")
    return values


def numerical_teacher_pool_indices(
    pool: Sequence[int],
    *,
    sample_id: str,
    appearance: int,
    seed: int,
    count: int,
) -> tuple[int, ...]:
    """Select a stable rotating window whose first cycle covers the pool once."""

    values = tuple(int(value) for value in pool)
    selected = int(count)
    current = int(appearance)
    if (
        not values
        or len(set(values)) != len(values)
        or any(left >= right for left, right in zip(values, values[1:]))
        or selected <= 0
        or selected > len(values)
        or current < 0
        or not str(sample_id)
    ):
        raise ValueError("numerical teacher pool selection is invalid")
    digest = hashlib.sha256(f"{int(seed)}:{sample_id}".encode("utf8")).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    order = generator.permutation(np.asarray(values, dtype=np.int64))
    start = (current * selected) % len(values)
    positions = np.arange(start, start + selected, dtype=np.int64) % len(values)
    return tuple(sorted(int(value) for value in order[positions]))


def residual_adaptive_teacher_pool_indices(
    pool: Sequence[int],
    *,
    sample_id: str,
    appearance: int,
    seed: int,
    count: int,
    probabilities: Sequence[float],
) -> tuple[int, ...]:
    """Deterministically sample unique exact times from a residual-adaptive PDF."""

    values = tuple(int(value) for value in pool)
    selected = int(count)
    current = int(appearance)
    probability = np.asarray(tuple(float(value) for value in probabilities), dtype=np.float64)
    if (
        not values
        or len(set(values)) != len(values)
        or any(left >= right for left, right in zip(values, values[1:]))
        or selected <= 0
        or selected > len(values)
        or current < 0
        or not str(sample_id)
        or probability.shape != (len(values),)
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or float(probability.sum()) <= 0.0
    ):
        raise ValueError("residual-adaptive teacher pool selection is invalid")
    probability = probability / probability.sum()
    digest = hashlib.sha256(
        f"{int(seed)}:{sample_id}:{current}:time-rad".encode("utf8")
    ).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    chosen = generator.choice(
        np.asarray(values, dtype=np.int64),
        size=selected,
        replace=False,
        p=probability,
    )
    return tuple(sorted(int(value) for value in chosen))


def _decode_text(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    )


class NumericalTeacherCache:
    """Lazy process-local lookup for identity-bound exact LWC-84 frames."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_source_manifest_sha256: str | None = None,
        expected_sample_ids: Sequence[str] | None = None,
    ) -> None:
        self._handle: h5py.File | None = None
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r", swmr=True) as handle:
            schema = str(handle.attrs.get("schema", ""))
            status = str(handle.attrs.get("status", ""))
            manifest = str(handle.attrs.get("source_manifest_sha256", ""))
            required = {
                "sample_id",
                "source_index",
                "time_indices",
                "time_s",
                "wavefield",
            }
            if schema != TEACHER_CACHE_SCHEMA or status != "complete":
                raise ValueError("numerical teacher cache schema or status is invalid")
            if not required.issubset(handle):
                raise ValueError("numerical teacher cache is incomplete")
            sample_ids = _decode_text(np.asarray(handle["sample_id"][:]))
            source_indices = np.asarray(handle["source_index"][:], dtype=np.int64)
            time_indices = np.asarray(handle["time_indices"][:], dtype=np.int64)
            time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
            shape = tuple(int(value) for value in handle["wavefield"].shape)
            dtype = handle["wavefield"].dtype
        if (
            not sample_ids
            or len(sample_ids) != len(set(sample_ids))
            or source_indices.shape != (len(sample_ids),)
            or time_indices.ndim != 1
            or time_s.shape != time_indices.shape
            or len(set(time_indices.tolist())) != len(time_indices)
            or bool(np.any(np.diff(time_indices) <= 0))
            or shape[:2] != (len(sample_ids), len(time_indices))
            or len(shape) != 4
            or dtype != np.dtype(np.float32)
        ):
            raise ValueError("numerical teacher cache dimensions are invalid")
        expected_manifest = (
            None
            if expected_source_manifest_sha256 is None
            else str(expected_source_manifest_sha256)
        )
        if expected_manifest is not None and manifest != expected_manifest:
            raise ValueError("numerical teacher source manifest identity mismatch")
        if expected_sample_ids is not None and sample_ids != tuple(
            str(value) for value in expected_sample_ids
        ):
            raise ValueError("numerical teacher sample identity mismatch")
        self.source_manifest_sha256 = manifest
        self.sample_ids = sample_ids
        self.source_indices = tuple(int(value) for value in source_indices)
        self.time_indices = tuple(int(value) for value in time_indices)
        self.time_s = tuple(float(value) for value in time_s)
        self.spatial_shape = shape[-2:]
        self._sample_rows = {value: index for index, value in enumerate(sample_ids)}
        self._time_columns = {
            int(value): index for index, value in enumerate(time_indices.tolist())
        }

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)
        return self._handle

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def read(
        self, sample_ids: Sequence[str], time_indices: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """Read physical float32 fields as `[record,time,z,x]`."""

        samples = tuple(str(value) for value in sample_ids)
        requested = np.asarray(torch.as_tensor(time_indices, device="cpu"), dtype=np.int64)
        if requested.ndim != 2 or requested.shape[0] != len(samples) or not samples:
            raise ValueError("numerical teacher lookup dimensions are invalid")
        output: list[np.ndarray] = []
        dataset = self._file()["wavefield"]
        for sample, row_times in zip(samples, requested, strict=True):
            try:
                row = self._sample_rows[sample]
            except KeyError as error:
                raise KeyError(f"unregistered numerical teacher sample: {sample}") from error
            if len(set(int(value) for value in row_times)) != len(row_times) or bool(
                np.any(np.diff(row_times) <= 0)
            ):
                raise ValueError("numerical teacher time requests must be strictly increasing")
            try:
                columns = [self._time_columns[int(value)] for value in row_times]
            except KeyError as error:
                raise KeyError(f"unregistered numerical teacher time: {error.args[0]}") from error
            values = np.asarray(dataset[row, columns], dtype=np.float32)
            if not np.isfinite(values).all():
                raise FloatingPointError("numerical teacher cache contains non-finite values")
            output.append(values)
        return torch.from_numpy(np.stack(output))


@dataclass(frozen=True)
class MultifidelityEnergyReference:
    """Full selected-time denominators shared by additive time blocks."""

    high_reference: RelativeEnergySquaredReference
    high_energy: torch.Tensor
    low_energy: torch.Tensor
    residual_energy: torch.Tensor
    residual_energy_floor_fraction: float


@dataclass(frozen=True)
class MultifidelityDistillationBlockLoss:
    """One additive multi-fidelity objective block."""

    total: torch.Tensor
    high: torch.Tensor
    low: torch.Tensor
    residual: torch.Tensor
    spectrum: torch.Tensor


def multifidelity_energy_reference(
    high_target: torch.Tensor,
    low_target: torch.Tensor,
    *,
    residual_energy_floor_fraction: float = 0.1,
    spectrum_energy_floor_fraction: float = 0.05,
) -> MultifidelityEnergyReference:
    """Build detached per-record energies for exact block decomposition."""

    high = torch.as_tensor(high_target).detach()
    low = torch.as_tensor(low_target, device=high.device).detach()
    residual_floor = float(residual_energy_floor_fraction)
    if high.shape != low.shape or high.ndim != 4:
        raise ValueError("multi-fidelity targets must match [record,time,z,x]")
    if not math.isfinite(residual_floor) or not 0.0 < residual_floor <= 1.0:
        raise ValueError("residual energy floor fraction must lie in (0, 1]")
    high_reference = relative_energy_squared_reference(
        high, energy_floor_fraction=float(spectrum_energy_floor_fraction)
    )
    high_energy = high_reference.target_square
    low_energy = low.float().square().flatten(1).sum(dim=-1).detach().clamp_min(1.0e-16)
    residual_energy_raw = (
        (high.float() - low.float()).square().flatten(1).sum(dim=-1).detach()
    )
    residual_energy = torch.maximum(
        residual_energy_raw,
        residual_floor * high_energy,
    ).clamp_min(1.0e-16)
    return MultifidelityEnergyReference(
        high_reference=high_reference,
        high_energy=high_energy,
        low_energy=low_energy,
        residual_energy=residual_energy,
        residual_energy_floor_fraction=residual_floor,
    )


def multifidelity_distillation_block_loss(
    prediction: torch.Tensor,
    coarse: torch.Tensor,
    high_target: torch.Tensor,
    low_target: torch.Tensor,
    *,
    reference: MultifidelityEnergyReference,
    low_fidelity_weight: float,
    residual_weight: float,
    spectrum_weight: float = 0.0,
) -> MultifidelityDistillationBlockLoss:
    """Supervise neural low/high fields and their high-minus-low residual."""

    predicted = torch.as_tensor(prediction)
    base = torch.as_tensor(coarse, device=predicted.device)
    high = torch.as_tensor(high_target, device=predicted.device).detach()
    low = torch.as_tensor(low_target, device=predicted.device).detach()
    if (
        predicted.shape != base.shape
        or predicted.shape != high.shape
        or predicted.shape != low.shape
        or predicted.ndim != 4
    ):
        raise ValueError("multi-fidelity block fields must match [record,time,z,x]")
    records = int(predicted.shape[0])
    if (
        reference.high_energy.shape != (records,)
        or reference.low_energy.shape != (records,)
        or reference.residual_energy.shape != (records,)
    ):
        raise ValueError("multi-fidelity energy reference does not match the block")
    weights = tuple(
        float(value) for value in (low_fidelity_weight, residual_weight, spectrum_weight)
    )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("multi-fidelity loss weights must be finite and nonnegative")

    high_parts = relative_energy_squared_block_loss(
        predicted,
        high,
        reference=reference.high_reference,
        spectrum_weight=weights[2],
    )
    low_error = (base.float() - low.float()).square().flatten(1).sum(dim=-1)
    low_loss = (low_error / reference.low_energy).mean()
    predicted_residual = predicted.float() - base.float()
    target_residual = high.float() - low.float()
    residual_error = (
        (predicted_residual - target_residual).square().flatten(1).sum(dim=-1)
    )
    residual_loss = (residual_error / reference.residual_energy).mean()
    return MultifidelityDistillationBlockLoss(
        total=(
            high_parts.total
            + weights[0] * low_loss
            + weights[1] * residual_loss
        ),
        high=high_parts.frame,
        low=low_loss,
        residual=residual_loss,
        spectrum=high_parts.spectrum,
    )


__all__ = [
    "MultifidelityDistillationBlockLoss",
    "MultifidelityEnergyReference",
    "NumericalTeacherCache",
    "TEACHER_CACHE_SCHEMA",
    "fixed_teacher_time_indices",
    "residual_adaptive_teacher_pool_indices",
    "multifidelity_distillation_block_loss",
    "multifidelity_energy_reference",
    "numerical_teacher_pool_indices",
]
