"""Mask-confined rank-16 correction (v15).

The r4e9 probe (``results/r4e9_mask_confined_oracle_20260826/``) showed that the
convention (iii) blow-up of the masked arm lives entirely in the frames the
energy mask drops: for uniform at rank 16 the outside-mask share of the error
increase was 1.0002, while the inside-mask error actually fell.  The masked arm
fitted coefficients on retained frames only but applied the correction to every
frame, so nothing bounded it where it was never fitted.

This module confines the correction to the mask.  The mask is derived from
**parent** frame energy, never from truth, so it is computable at deployment
time; r4e9 measured parent-vs-truth mask agreement at 0.9948 / 0.9895 / 1.0000.
Outside the mask the output is bit-identical to the parent.
"""
from __future__ import annotations

import torch

from .r16_dscp import TIME_COUNT, c1_causal_mask


ENERGY_TAU = 1.0e-3


class MaskConfinementError(RuntimeError):
    """A mask-confinement contract was violated."""


def frame_energy(field: torch.Tensor) -> torch.Tensor:
    """Per-frame squared L2 energy in float64, shape [time].

    Matches ``scripts/probe_r4e8_masked_oracle.py:frame_energy`` so v15 and the
    probes measure the same quantity.
    """
    value = torch.as_tensor(field)
    if value.ndim < 2:
        raise ValueError("field must be [time, ...space]")
    return value.reshape(value.shape[0], -1).double().square().sum(dim=1)


def parent_energy_mask(
    parent_wavefield: torch.Tensor, *, tau: float = ENERGY_TAU
) -> tuple[torch.Tensor, float]:
    """Keep-mask and absolute floor ``tau * max_s E_s``, from the parent only.

    Truth is not an argument and must never become one: a truth-derived mask is
    not computable at deployment time and is vetoed by the v15 preregistration.
    """
    energy = frame_energy(parent_wavefield)
    if not 0.0 < float(tau) < 1.0:
        raise ValueError("tau must lie strictly inside (0, 1)")
    if not bool(torch.isfinite(energy).all()):
        raise MaskConfinementError("parent frame energy is not finite")
    peak = float(energy.max().item())
    if peak <= 0.0:
        raise MaskConfinementError("parent energy profile has no positive peak")
    floor = float(tau) * peak
    return energy >= floor, floor


def confine_correction(
    correction: torch.Tensor,
    parent_wavefield: torch.Tensor,
    *,
    tau: float = ENERGY_TAU,
) -> torch.Tensor:
    """Zero the correction on every frame the parent-energy mask drops."""
    field = torch.as_tensor(correction)
    keep, _ = parent_energy_mask(parent_wavefield, tau=tau)
    if keep.shape[0] != field.shape[0]:
        raise MaskConfinementError("mask and correction disagree on the time axis")
    gate = keep.to(device=field.device, dtype=field.dtype)
    return field * gate.reshape(-1, *([1] * (field.ndim - 1)))


def confined_causal_gate(
    parent_wavefield: torch.Tensor,
    k1: int,
    *,
    tau: float = ENERGY_TAU,
    time_count: int = TIME_COUNT,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C1 causal ramp times the parent-energy mask, shape [time].

    The ramp alone is what v14 applied; the energy mask is an additional layer,
    not a replacement, so C1 causality at ``k1`` is preserved exactly.
    """
    parent = torch.as_tensor(parent_wavefield)
    ramp = c1_causal_mask(int(time_count), int(k1), device=parent.device, dtype=dtype)
    keep, _ = parent_energy_mask(parent, tau=tau)
    return ramp * keep.to(device=ramp.device, dtype=ramp.dtype)


def dimensionless_coefficient_energy(
    coefficients: torch.Tensor, parent_wavefield: torch.Tensor
) -> torch.Tensor:
    """Coefficient energy normalized by parent energy, so the ratio is unitless.

    v14 summed ``coefficients.square().mean()`` (units of amplitude squared)
    directly with three dimensionless ratios.  Scaling parent and truth by the
    same factor moved that term quadratically while leaving the others fixed,
    which silently reweighted the loss.  Dividing by the parent's mean squared
    amplitude cancels the units: the basis is orthonormal in time, so a
    coefficient carries the amplitude of the field it reconstructs.
    """
    values = torch.as_tensor(coefficients).double()
    scale = torch.as_tensor(parent_wavefield, device=values.device).double()
    denominator = scale.square().mean().clamp_min(1.0e-30)
    return (values.square().mean() / denominator).to(
        dtype=torch.as_tensor(coefficients).dtype
    )


def per_record_loss_reduction(
    losses_by_record: dict[str, list[float]],
) -> dict[str, float]:
    """Loss reduction computed within each record, never across records.

    v14 divided ``losses[0]`` by ``losses[-1]`` over a round-robin stream of
    three records, so the first value came from uniform and the last from
    marmousi.  That yielded 0.99967 while the model had barely moved.  Comparing
    a quantity from one record against a quantity from another is void.
    """
    output: dict[str, float] = {}
    for record, series in losses_by_record.items():
        if not series:
            raise MaskConfinementError(f"record {record} has no losses")
        first, last = float(series[0]), float(series[-1])
        output[record] = (first - last) / max(abs(first), 1.0e-30)
    return output


__all__ = [
    "ENERGY_TAU",
    "MaskConfinementError",
    "confine_correction",
    "confined_causal_gate",
    "dimensionless_coefficient_energy",
    "frame_energy",
    "parent_energy_mask",
    "per_record_loss_reduction",
]


def measured_peak_vram_bytes(device: torch.device | None) -> int:
    """Peak reserved VRAM, or a refusal marker rather than a silent zero.

    v14 wrote ``peak_vram_bytes = 0`` into the smoke terminal, so the 8 GiB gate
    passed with value 0 while nvidia-smi showed 6941 MiB in flight.  A gate that
    cannot fail is not a gate.  Returning -1 for "not measured" keeps the
    distinction between "measured nothing" and "did not measure".
    """
    if device is None or torch.device(device).type != "cuda":
        return -1
    return int(torch.cuda.max_memory_reserved(device))


def vram_gate(peak_bytes: int, *, threshold_bytes: int) -> dict[str, object]:
    """VRAM gate that refuses to pass on an unmeasured or absurd value."""
    value = int(peak_bytes)
    measured = value > 0
    return {
        "value": value,
        "threshold": int(threshold_bytes),
        "measured": measured,
        "passed": measured and value <= int(threshold_bytes),
        "refusal_reason": None if measured else "peak vram was never measured",
    }


def per_record_loss_gate(
    losses_by_record: dict[str, list[float]], *, threshold: float
) -> dict[str, object]:
    """Loss gate scored inside each record; every record must clear it.

    Replaces the v14 gate, which took the first and last loss of a round-robin
    stream over three records and reported 0.99967.
    """
    reductions = per_record_loss_reduction(losses_by_record)
    return {
        "per_record": reductions,
        "threshold": float(threshold),
        "worst_record": min(reductions, key=reductions.get) if reductions else None,
        "worst_value": min(reductions.values()) if reductions else None,
        "passed": bool(reductions) and all(v >= float(threshold) for v in reductions.values()),
        "cross_record_comparison_used": False,
    }


__all__ += [
    "measured_peak_vram_bytes",
    "per_record_loss_gate",
    "vram_gate",
]
