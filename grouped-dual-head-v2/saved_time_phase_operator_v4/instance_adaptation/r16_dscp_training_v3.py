"""V15 numerical core: parent-energy mask confinement and the four v14 fixes.

Scope statement, restated here so it cannot be lost downstream.

What this is:  an output-field data fit plus a feature-driven ansatz.  A small
predictor emits temporal-POD coefficients from deployment-computable features;
the materialized correction is added to the parent wavefield.

What this is NOT:  it is not a physics residual, not a PDE residual, and not
deployment-time truth supervision.  No PDE residual is evaluated anywhere in
this pipeline.  (inherited veto (e))

The keep mask ``m_t`` is derived from PARENT frame energy only::

    m_t = 1[E_t^parent >= tau * max_s E_s^parent],  tau = 1e-3

Truth frame energy must never enter the mask (inherited veto (b)).  Every
mask-producing function in this module takes the parent field and nothing else,
so the violation is structurally impossible rather than merely discouraged.

Four v14 defects are addressed here and in ``r16_dscp_engine_v15``:

1. the correction is multiplied by ``m_t`` before it is applied, so it is
   exactly zero on dropped frames and the output there is bit-identical to the
   parent (this module: :func:`confine_correction`, :func:`apply_confined_correction`);
2. the smoke loss-reduction gate is computed per record, never by taking the
   first and last loss of a round-robin stream (engine v15);
3. the oracle upper bound is recomputed on the record being scored; the three
   v14 hardcoded constants are void and appear nowhere in v15
   (this module: :func:`confined_oracle_upper_bound`);
4. ``normalized_coefficient_energy`` is made dimensionless by dividing by the
   mean squared PARENT amplitude on the retained future frames
   (this module: :func:`dimensionless_coefficient_energy`).

Nothing in this module is evidence of learning or convergence.  The oracle is
an upper bound, not an achieved result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------

#: mask threshold fraction, transcribed from the frozen v15 preregistration
#: objective.mask.tau; not tunable and never moved toward a measured value.
TAU = 1.0e-3

#: denominator floor used by every ratio in this module
EPS_SQUARED = 1.0e-30

#: metric conventions, named exactly as in the r4e9 evidence chain
CONVENTIONS = (
    "unmasked_per_frame_unsquared_mean",
    "masked_energy_floored_per_frame",
    "global_energy_rel_l2",
)

#: the acceptance convention inherited unchanged from the v14 preregistration
ACCEPTANCE_CONVENTION = "global_energy_rel_l2"

#: convention (ii) is a training proxy only; no acceptance claim may use it
TRAINING_PROXY_CONVENTION = "masked_energy_floored_per_frame"


class V15ContractError(RuntimeError):
    """A frozen v15 numerical contract is violated."""


class MaskCausalityRefusal(V15ContractError):
    """A mask was asked to consume deployment-time truth."""


class ConfinementLeak(V15ContractError):
    """A confined correction is not identically zero outside the keep mask."""


# ---------------------------------------------------------------------------
# frame energy and the parent-energy keep mask
# ---------------------------------------------------------------------------


def frame_energy(field: torch.Tensor) -> torch.Tensor:
    """Per-frame squared L2 energy in float64, shape ``[time]``.

    Same definition as the r4e8/r4e9 evidence chain: sum of squares over every
    non-time axis.
    """
    value = torch.as_tensor(field)
    if value.ndim < 2:
        raise ValueError("field must be [time, ...space]")
    return value.reshape(value.shape[0], -1).double().square().sum(dim=1)


def parent_energy_keep_mask(
    parent_field: torch.Tensor, *, tau: float = TAU
) -> tuple[torch.Tensor, float]:
    """The v15 keep mask, derived from the parent field and nothing else.

    ``m_t = 1[E_t^parent >= tau * max_s E_s^parent]``.

    This signature deliberately accepts no truth argument.  Passing truth here
    would be an inherited-veto (b) violation, and there is no parameter through
    which it could be supplied.

    Returns the boolean keep mask and the absolute energy floor.
    """
    energy = frame_energy(parent_field)
    return energy_keep_mask(energy, tau=tau)


def energy_keep_mask(
    energy: torch.Tensor, *, tau: float = TAU
) -> tuple[torch.Tensor, float]:
    """Threshold a nonnegative ``[time]`` energy profile at ``tau * max``."""
    values = torch.as_tensor(energy, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("energy must be a nonempty [time] vector")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("energy must be finite and nonnegative")
    if not 0.0 < float(tau) < 1.0:
        raise ValueError("tau must lie strictly inside (0, 1)")
    peak = float(values.max().item())
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("energy profile has no positive peak")
    floor = float(tau) * peak
    return values >= floor, floor


def full_time_keep_mask(
    parent_wavefield: torch.Tensor, *, k1: int, tau: float = TAU
) -> tuple[torch.Tensor, float]:
    """Assemble the full ``[time]`` keep mask for one record.

    The energy profile and its peak are taken over the C1-causal future window
    ``t > k1`` only, matching the r4e9 confined arm.  Frames at or before ``k1``
    are dropped, which is already what the C1 causal ramp enforces; the parent
    energy mask is an ADDITIONAL layer on top of that ramp, not a replacement
    for it.
    """
    parent = torch.as_tensor(parent_wavefield)
    if parent.ndim < 2:
        raise ValueError("parent must be [time, ...space]")
    count = int(parent.shape[0])
    start = int(k1) + 1
    if not 0 < start < count:
        raise ValueError("C1 causal start must lie strictly inside the time axis")
    keep, floor = parent_energy_keep_mask(parent[start:], tau=tau)
    full = torch.zeros(count, dtype=torch.bool, device=keep.device)
    full[start:] = keep
    return full, floor


def mask_provenance(*, tau: float = TAU) -> dict[str, Any]:
    """Machine-readable attestation of how the deployed mask is derived."""
    return {
        "symbol": "m_t",
        "formula": "m_t = 1[E_t^parent >= tau * max_s E_s^parent]",
        "driven_by": "parent_frame_energy",
        "tau": float(tau),
        "deployment_computable": True,
        "uses_truth_frame_energy": False,
        "uses_validation_or_test_data": False,
        "layered_on_top_of_c1_causal_ramp": True,
        "replaces_c1_causal_ramp": False,
    }


def mask_agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    """Fraction of frames on which two boolean masks agree (diagnostic only)."""
    a = torch.as_tensor(left).bool()
    b = torch.as_tensor(right).bool()
    if a.shape != b.shape or a.ndim != 1 or a.numel() == 0:
        raise ValueError("masks must be matching nonempty [time] vectors")
    return float((a == b).double().mean().item())


# ---------------------------------------------------------------------------
# mask-confined correction
# ---------------------------------------------------------------------------


def _time_gate(keep_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    keep = torch.as_tensor(keep_mask)
    if keep.dtype != torch.bool:
        raise ValueError("keep mask must be a boolean tensor")
    if keep.ndim != 1:
        raise ValueError("keep mask must be a [time] vector")
    if int(keep.shape[0]) != int(reference.shape[-3]):
        raise ValueError("keep mask length must equal the time axis length")
    view = keep.to(device=reference.device).reshape(
        *(1,) * (reference.ndim - 3), int(keep.shape[0]), 1, 1
    )
    return view


def confine_correction(
    correction: torch.Tensor, keep_mask: torch.Tensor
) -> torch.Tensor:
    """Multiply a correction by the keep mask; outside it the result is +0.0.

    ``correction`` is ``[..., time, height, width]``.  On dropped frames the
    returned tensor is exactly zero -- not small, not signed-zero-ambiguous --
    because the masked entries are written from a zero tensor rather than
    obtained by multiplying by ``0.0``.
    """
    value = torch.as_tensor(correction)
    if value.ndim < 3:
        raise ValueError("correction must be [..., time, height, width]")
    gate = _time_gate(keep_mask, value)
    confined = torch.where(gate, value, torch.zeros((), dtype=value.dtype, device=value.device))
    outside = confined[(~gate).expand_as(confined)]
    if outside.numel() and bool((outside != 0).any()):
        raise ConfinementLeak("confined correction is nonzero outside the keep mask")
    return confined


def apply_confined_correction(
    parent: torch.Tensor, correction: torch.Tensor, keep_mask: torch.Tensor
) -> torch.Tensor:
    """Add a mask-confined correction to the parent.

    Outside the keep mask the output is bit-identical to the parent: the parent
    value is selected, never summed with a zero, so even signed zeros and
    denormals survive unchanged.
    """
    base = torch.as_tensor(parent)
    delta = torch.as_tensor(correction, device=base.device)
    if base.shape != delta.shape:
        raise ValueError("parent and correction must have identical shapes")
    if base.ndim < 3:
        raise ValueError("parent must be [..., time, height, width]")
    gate = _time_gate(keep_mask, base)
    confined = confine_correction(delta, keep_mask)
    return torch.where(gate, base + confined, base)


def bitwise_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    """True iff two tensors have identical dtype, shape and raw bytes."""
    a = torch.as_tensor(left).detach().cpu().contiguous()
    b = torch.as_tensor(right).detach().cpu().contiguous()
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
        a.element_size()
    ]
    return bool(
        torch.equal(a.reshape(-1).view(width), b.reshape(-1).view(width))
    )


def confinement_invariants(
    parent: torch.Tensor,
    correction: torch.Tensor,
    keep_mask: torch.Tensor,
) -> dict[str, Any]:
    """Measure, on real tensors, that confinement is exact rather than close."""
    confined = confine_correction(correction, keep_mask)
    corrected = apply_confined_correction(parent, correction, keep_mask)
    gate = _time_gate(keep_mask, torch.as_tensor(parent))
    outside = (~gate).expand_as(confined)
    outside_correction = confined[outside]
    parent_outside = torch.as_tensor(parent)[outside]
    corrected_outside = corrected[outside]
    return {
        "correction_identically_zero_outside_mask": bool(
            outside_correction.numel() == 0 or bool((outside_correction == 0).all())
        ),
        "max_abs_applied_correction_outside_mask": float(
            outside_correction.abs().max().item()
        )
        if outside_correction.numel()
        else 0.0,
        "output_bitwise_identical_to_parent_outside_mask": bitwise_identical(
            corrected_outside, parent_outside
        ),
        "kept_frame_count": int(torch.as_tensor(keep_mask).bool().sum().item()),
        "dropped_frame_count": int((~torch.as_tensor(keep_mask).bool()).sum().item()),
    }


# ---------------------------------------------------------------------------
# fix 4: dimensionless coefficient energy
# ---------------------------------------------------------------------------


def parent_reference_energy(
    parent_future: torch.Tensor, keep_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean squared parent amplitude on the frames the correction can touch.

    Units: amplitude squared.  This is the denominator that makes the
    coefficient-energy loss term dimensionless.  It is computed from the parent
    only, so it is deployment-computable and truth-free.
    """
    value = torch.as_tensor(parent_future)
    if not value.is_floating_point():
        value = value.float()
    if value.ndim < 3:
        raise ValueError("parent future must be [..., time, height, width]")
    if keep_mask is None:
        return value.square().mean()
    gate = _time_gate(keep_mask, value)
    selected = value[gate.expand_as(value)]
    if selected.numel() == 0:
        raise V15ContractError("parent reference energy has no retained frame")
    return selected.square().mean()


def dimensionless_coefficient_energy(
    coefficients: torch.Tensor,
    parent_future: torch.Tensor,
    keep_mask: torch.Tensor | None = None,
    *,
    eps: float = EPS_SQUARED,
) -> torch.Tensor:
    """The v14 ``normalized_coefficient_energy`` term, made dimensionless.

    v14 used ``coefficients.square().mean()``.  The temporal POD basis is
    orthonormal in time, so a coefficient carries the units of the field
    (pressure amplitude) and its square carries amplitude squared, which was
    then summed directly with three dimensionless ratios.  This function
    divides by the mean squared parent amplitude on the retained future frames,
    which carries the same amplitude-squared units, so the quotient is a pure
    number.

    Invariance:  scaling the parent, the truth and hence the fitted
    coefficients by a common factor ``c`` scales the numerator by ``c**2`` and
    the denominator by ``c**2``, leaving this term unchanged.
    """
    value = torch.as_tensor(coefficients)
    if not value.is_floating_point():
        value = value.float()
    numerator = value.square().mean()
    denominator = parent_reference_energy(parent_future, keep_mask).clamp_min(eps)
    return numerator / denominator.to(numerator.dtype)


def coefficient_energy_definition() -> dict[str, Any]:
    """The recorded choice and rationale for the fix-4 normalization."""
    return {
        "term": "normalized_coefficient_energy",
        "v14_definition": "coefficients.square().mean()",
        "v14_units": "amplitude_squared",
        "v14_defect": (
            "an amplitude-squared quantity was summed directly with three "
            "dimensionless ratios under weight 1e-4, so the term's effective "
            "strength moved with the absolute pressure scale of the record"
        ),
        "v15_definition": (
            "coefficients.square().mean() / "
            "parent_future[keep_mask].square().mean()"
        ),
        "v15_units": "dimensionless",
        "normalizer": "mean_squared_parent_amplitude_on_retained_future_frames",
        "normalizer_is_deployment_computable": True,
        "normalizer_uses_truth": False,
        "weight_unchanged_from_v14": 1.0e-4,
        "alternative_considered": "residual_energy_normalization",
        "why_parent_energy_was_chosen": [
            "the parent field is deployment-computable and truth-free, so the "
            "regularizer's own denominator never depends on a label",
            "residual energy shrinks as the fit improves, so a residual "
            "normalizer would inflate the penalty exactly as training succeeds "
            "and would be a self-amplifying, unstable term",
            "the pilot gate correction_energy_ratio already normalizes "
            "correction energy by parent energy, so this choice makes the loss "
            "term and that gate dimensionally consistent",
            "the term is invariant under a common rescaling of parent, truth "
            "and fitted coefficients, which is the property the fix requires",
        ],
        "in_preregistration_scope": (
            "results/r16_dscp_v15_preregistration_20260826.json "
            "v14_known_defects_to_fix.normalized_coefficient_energy_dimensional_error"
        ),
    }


# ---------------------------------------------------------------------------
# masked training loss (convention (ii) style flooring, parent-driven mask)
# ---------------------------------------------------------------------------


def _late_third_selector(keep_mask: torch.Tensor) -> torch.Tensor:
    keep = torch.as_tensor(keep_mask).bool()
    indices = keep.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        raise V15ContractError("masked loss has no retained frame")
    start = (2 * int(indices.numel())) // 3
    return indices[start:]


def masked_confined_loss(
    corrected_future: torch.Tensor,
    truth_future: torch.Tensor,
    coefficients: torch.Tensor,
    parent_future: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    tau: float = TAU,
) -> Mapping[str, torch.Tensor]:
    """Train-only supervised loss on the mask-retained frames.

    Tensors are ``[B, T, H, W]``; ``keep_mask`` is the parent-driven ``[T]``
    boolean mask restricted to the same future window.  The four term weights
    are exactly the v14 weights; only the coefficient term's definition changed.

    This function is train-only.  It is never used in deployment, and it
    evaluates no PDE residual.
    """
    prediction = torch.as_tensor(corrected_future).float()
    truth = torch.as_tensor(truth_future, device=prediction.device).float()
    parent = torch.as_tensor(parent_future, device=prediction.device).float()
    if prediction.ndim != 4 or prediction.shape != truth.shape or prediction.shape != parent.shape:
        raise ValueError("corrected, truth and parent futures must match [B,T,H,W]")
    keep = torch.as_tensor(keep_mask, device=prediction.device).bool()
    if keep.ndim != 1 or int(keep.shape[0]) != int(prediction.shape[1]):
        raise ValueError("keep mask must be a [T] boolean vector over the future window")
    if not bool(keep.any()):
        raise V15ContractError("masked loss has no retained frame")

    error = prediction - truth
    truth_energy = truth.square().sum(dim=(2, 3))
    floor = (float(tau) * truth_energy.amax(dim=1, keepdim=True)).clamp_min(EPS_SQUARED)
    denominator = truth_energy.clamp_min(floor)
    frame_values = error.square().sum(dim=(2, 3)) / denominator

    frame_term = frame_values[:, keep].mean()
    late_index = _late_third_selector(keep)
    late_term = frame_values[:, late_index].mean()

    pair = keep[:-1] & keep[1:]
    if int(prediction.shape[1]) < 2 or not bool(pair.any()):
        temporal_term = prediction.new_zeros(())
    else:
        delta_error = (error[:, 1:] - error[:, :-1])[:, pair]
        delta_truth = (truth[:, 1:] - truth[:, :-1])[:, pair]
        temporal_term = (
            delta_error.square().sum() / delta_truth.square().sum().clamp_min(EPS_SQUARED)
        )

    coefficient_term = dimensionless_coefficient_energy(coefficients, parent, keep)
    total = frame_term + 0.25 * late_term + 0.10 * temporal_term + 1.0e-4 * coefficient_term
    return {
        "total": total,
        "frame_relative_l2_squared": frame_term,
        "late_third_relative_l2_squared": late_term,
        "temporal_difference": temporal_term,
        "normalized_coefficient_energy": coefficient_term,
    }


def loss_specification() -> dict[str, Any]:
    """The frozen v15 loss description, for the terminal artifact."""
    return {
        "name": "masked_loss_plus_mask_confined_correction",
        "weights": {
            "frame_relative_l2_squared": 1.0,
            "late_third_relative_l2_squared": 0.25,
            "temporal_difference": 0.10,
            "normalized_coefficient_energy": 1.0e-4,
        },
        "weights_unchanged_from_v14": True,
        "frame_restriction": "parent_energy_keep_mask_retained_frames_only",
        "denominator": "truth_frame_energy_floored_at_tau_times_peak_truth_energy",
        "denominator_role": (
            "convention (ii) style flooring; convention (ii) is a TRAINING PROXY "
            "and must not carry an acceptance claim"
        ),
        "acceptance_convention": ACCEPTANCE_CONVENTION,
        "mask": mask_provenance(),
        "coefficient_energy": coefficient_energy_definition(),
        "pde_residual_used": False,
        "deployment_time_truth_supervision": False,
        "is_learning_or_convergence_evidence": False,
    }


# ---------------------------------------------------------------------------
# metric conventions and fix 3: oracle recomputed on the scored record
# ---------------------------------------------------------------------------


def convention_metrics(
    error_squared_by_frame: torch.Tensor,
    truth_energy_by_frame: torch.Tensor,
    *,
    tau: float = TAU,
) -> dict[str, float]:
    """The three named conventions from one per-frame squared error vector.

    (i)   ``unmasked_per_frame_unsquared_mean``
    (ii)  ``masked_energy_floored_per_frame``  -- training proxy only
    (iii) ``global_energy_rel_l2``             -- the acceptance convention

    Convention (ii)'s frame selection is a property of that scoring convention
    and is defined on truth energy.  It is a metric definition, never a gate on
    the applied correction; the applied correction is gated by ``m_t``, which is
    parent-only.
    """
    error = torch.as_tensor(error_squared_by_frame, dtype=torch.float64)
    energy = torch.as_tensor(truth_energy_by_frame, dtype=torch.float64, device=error.device)
    if error.ndim != 1 or error.shape != energy.shape:
        raise ValueError("error and truth energy must be matching [time] vectors")
    if not bool(torch.isfinite(error).all()) or not bool(torch.isfinite(energy).all()):
        raise FloatingPointError("per-frame error or truth energy is non-finite")
    unmasked = float(
        (error / energy.clamp_min(EPS_SQUARED)).sqrt().mean().item()
    )
    metric_keep, metric_floor = energy_keep_mask(energy, tau=tau)
    floored = energy.clamp_min(metric_floor)
    masked = float((error[metric_keep] / floored[metric_keep]).sqrt().mean().item())
    global_value = float(
        math.sqrt(float(error.sum().item()) / max(float(energy.sum().item()), EPS_SQUARED))
    )
    return {
        "unmasked_per_frame_unsquared_mean": unmasked,
        "masked_energy_floored_per_frame": masked,
        "global_energy_rel_l2": global_value,
    }


def convention_gains(
    parent_metrics: Mapping[str, float], corrected_metrics: Mapping[str, float]
) -> dict[str, float]:
    """``(parent - corrected) / max(parent, 1e-30)`` per convention."""
    return {
        name: (float(parent_metrics[name]) - float(corrected_metrics[name]))
        / max(float(parent_metrics[name]), EPS_SQUARED)
        for name in CONVENTIONS
    }


def weighted_coefficient_map(design: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted-LS coefficient map: pinv of the weight-rooted design, un-rooted.

    Same algebraic form as the published estimator used throughout the r4e
    evidence chain.
    """
    matrix = torch.as_tensor(design, dtype=torch.float64)
    weight = torch.as_tensor(weights, dtype=torch.float64, device=matrix.device)
    if matrix.ndim != 2 or weight.shape != (matrix.shape[0],):
        raise ValueError("design must be [time, rank] with one weight per frame")
    if not bool((weight > 0).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("weights must be finite and positive")
    root = weight.sqrt()[:, None]
    mapping = torch.linalg.pinv(root * matrix, rtol=1.0e-12)
    return mapping * root[:, 0][None, :]


@torch.no_grad()
def confined_oracle_upper_bound(
    parent_future: torch.Tensor,
    truth_future: torch.Tensor,
    basis_future: torch.Tensor,
    *,
    tau: float = TAU,
    spatial_chunk: int = 4096,
) -> dict[str, Any]:
    """Recompute the mask-confined oracle upper bound ON THE GIVEN RECORD.

    This replaces the three voided v14 constants.  Those constants were
    measured on ``train_uniform_00102``, ``train_layered_01032`` and a
    SYNTHETIC marmousi record, then used to score
    ``train_uniform_00321`` / ``train_layered_00564`` / ``train_marmousi_00385``.
    Carrying them across records, or across a panel containing a synthetic
    sample, is an inherited-veto (c) violation.  Nothing here is transplanted:
    every number is a function of the arguments only.

    The oracle reads train truth -- that is what makes it an oracle, and it is
    why its output is an UPPER BOUND and never an achieved result.  The MASK,
    however, is parent-only, both for the fit restriction and for the
    confinement gate, because the deployed candidate can only ever see the
    parent mask.  This differs from the r4e9 probe, which restricted the fit to
    the truth mask; the deviation is recorded in the returned payload and no
    r4e9 reproduction is claimed.

    ``parent_future``/``truth_future`` are ``[T, H, W]``; ``basis_future`` is
    ``[T, R]`` already restricted to the same future window.
    """
    parent = torch.as_tensor(parent_future)
    truth = torch.as_tensor(truth_future, device=parent.device)
    if parent.ndim != 3 or parent.shape != truth.shape:
        raise ValueError("parent and truth futures must both be [T,H,W]")
    basis = torch.as_tensor(basis_future, dtype=torch.float64, device=parent.device)
    if basis.ndim != 2 or int(basis.shape[0]) != int(parent.shape[0]):
        raise ValueError("basis must be [T,R] over the same future window")
    frames = int(parent.shape[0])
    points = int(parent.shape[1]) * int(parent.shape[2])

    flat_parent = parent.reshape(frames, points).double()
    flat_truth = truth.reshape(frames, points).double()
    residual = flat_truth - flat_parent

    truth_energy = flat_truth.square().sum(dim=1)
    parent_error = residual.square().sum(dim=1)
    residual_energy = float(parent_error.sum().item())
    if not math.isfinite(residual_energy) or residual_energy <= 0.0:
        raise V15ContractError("future residual energy is invalid")

    keep, floor = parent_energy_keep_mask(parent, tau=tau)
    keep = keep.to(device=parent.device)
    keep_index = keep.nonzero(as_tuple=True)[0]
    if int(keep_index.numel()) < int(basis.shape[1]):
        raise V15ContractError("fewer retained frames than basis columns")

    _metric_keep, metric_floor = energy_keep_mask(truth_energy, tau=tau)
    weights = truth_energy.clamp_min(metric_floor).reciprocal()
    mapping = weighted_coefficient_map(basis[keep_index], weights[keep_index])

    parent_metrics = convention_metrics(parent_error, truth_energy, tau=tau)
    confined_error = torch.zeros(frames, dtype=torch.float64, device=parent.device)
    coefficient_square_sum = 0.0
    correction_energy = 0.0
    parent_energy_total = float(flat_parent.square().sum().item())
    max_outside = 0.0
    coefficient_count = 0

    for low in range(0, points, int(spatial_chunk)):
        high = min(low + int(spatial_chunk), points)
        block = residual[:, low:high]
        coefficients = mapping @ block[keep_index]
        raw = basis @ coefficients
        confined = confine_correction(
            raw.reshape(frames, 1, high - low), keep
        ).reshape(frames, high - low)
        outside = confined[~keep]
        if outside.numel():
            max_outside = max(max_outside, float(outside.abs().max().item()))
        confined_error += (block - confined).square().sum(dim=1)
        coefficient_square_sum += float(coefficients.square().sum().item())
        coefficient_count += int(coefficients.numel())
        correction_energy += float(confined.square().sum().item())
        del block, coefficients, raw, confined, outside

    if max_outside != 0.0:
        raise ConfinementLeak(
            f"oracle correction is nonzero outside the parent mask: {max_outside}"
        )
    corrected_metrics = convention_metrics(confined_error, truth_energy, tau=tau)
    gains = convention_gains(parent_metrics, corrected_metrics)
    payload = {
        "kind": "train_only_confined_oracle_upper_bound",
        "is_achieved_result": False,
        "recomputed_on_the_scored_record": True,
        "transplanted_constants_used": False,
        "synthetic_records_used": False,
        "voided_v14_constants_referenced": False,
        "future_frame_count": frames,
        "spatial_point_count": points,
        "fit_frame_count": int(keep_index.numel()),
        "parent_energy_floor": floor,
        "tau": float(tau),
        "mask": mask_provenance(tau=tau),
        "fit_objective": (
            "weighted LS restricted to PARENT-mask retained frames with "
            "truth-energy weights floored at tau times the peak truth energy"
        ),
        "deviation_from_r4e9_probe": (
            "r4e9 restricted the fit to the truth mask; v15 restricts it to the "
            "parent mask so the bound matches what the deployment-causal gate "
            "can express. No r4e9 reproduction is claimed."
        ),
        "parent_metrics": parent_metrics,
        "corrected_metrics": corrected_metrics,
        "gain": gains,
        "acceptance_convention": ACCEPTANCE_CONVENTION,
        "acceptance_convention_gain": float(gains[ACCEPTANCE_CONVENTION]),
        "correction_energy": correction_energy,
        "correction_energy_ratio": correction_energy
        / max(parent_energy_total, EPS_SQUARED),
        "mean_squared_coefficient": coefficient_square_sum / max(coefficient_count, 1),
        "dimensionless_coefficient_energy": (
            coefficient_square_sum / max(coefficient_count, 1)
        )
        / max(float(parent_reference_energy(parent[None], keep).item()), EPS_SQUARED),
        "confinement_invariants": {
            "correction_identically_zero_outside_mask": True,
            "max_abs_applied_correction_outside_mask": max_outside,
        },
    }
    if not all(
        math.isfinite(float(value))
        for group in (parent_metrics, corrected_metrics, gains)
        for value in group.values()
    ):
        raise FloatingPointError("confined oracle produced a non-finite metric")
    return payload


# ---------------------------------------------------------------------------
# fix 5: real CUDA peak accounting
# ---------------------------------------------------------------------------


@dataclass
class CudaPeakVram:
    """Real peak VRAM accounting.

    v14 reported ``peak_vram_bytes = 0`` while nvidia-smi measured 6941 MiB,
    because ``V4ProductionBackend.prepare`` was the only place that sampled
    CUDA memory and every V5+ backend overrides ``prepare`` and takes the
    cached update/score/backward paths instead.  This monitor is sampled
    explicitly on each of those paths, and :func:`vram_gate` refuses to pass on
    an unmeasured value.
    """

    device: torch.device
    peak_reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    samples: int = 0

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def _cuda_ready(self) -> bool:
        """True only when this process can legally query CUDA memory stats."""
        return (
            self.is_cuda
            and torch.cuda.is_available()
            and torch.cuda.is_initialized()
        )

    def reset(self) -> "CudaPeakVram":
        if self._cuda_ready():
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_reserved_bytes = 0
        self.peak_allocated_bytes = 0
        self.samples = 0
        return self

    def sample(self) -> int:
        if self._cuda_ready():
            self.peak_reserved_bytes = max(
                self.peak_reserved_bytes,
                int(torch.cuda.max_memory_reserved(self.device)),
            )
            self.peak_allocated_bytes = max(
                self.peak_allocated_bytes,
                int(torch.cuda.max_memory_allocated(self.device)),
            )
        self.samples += 1
        return self.peak_reserved_bytes

    @property
    def peak_bytes(self) -> int:
        return int(self.peak_reserved_bytes)

    def payload(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "device_type": self.device.type,
            "source": "torch.cuda.max_memory_reserved",
            "samples": int(self.samples),
            "peak_reserved_bytes": int(self.peak_reserved_bytes),
            "peak_allocated_bytes": int(self.peak_allocated_bytes),
            "measured": bool(self.is_cuda and self.peak_reserved_bytes > 0),
        }


TRAIN_VRAM_LIMIT_BYTES = 8 * 1024**3
INFERENCE_VRAM_LIMIT_BYTES = int(6.5 * 1024**3)


def vram_gate(
    payload: Mapping[str, Any], *, limit_bytes: int = TRAIN_VRAM_LIMIT_BYTES
) -> dict[str, Any]:
    """A non-trivial VRAM gate: an unmeasured or zero peak cannot pass.

    The threshold is the inherited v14 value and is not moved.
    """
    peak = int(payload.get("peak_reserved_bytes", 0))
    measured = bool(payload.get("measured"))
    cuda = str(payload.get("device_type")) == "cuda"
    if not cuda:
        return {
            "value": peak,
            "threshold": int(limit_bytes),
            "measured": False,
            "nontrivial": False,
            "passed": False,
            "reason": "vram gate requires a cuda device; a non-cuda run cannot pass it",
        }
    if not measured or peak <= 0:
        return {
            "value": peak,
            "threshold": int(limit_bytes),
            "measured": False,
            "nontrivial": False,
            "passed": False,
            "reason": (
                "peak vram was not measured; a zero peak is the v14 defect and "
                "must never pass trivially"
            ),
        }
    return {
        "value": peak,
        "threshold": int(limit_bytes),
        "measured": True,
        "nontrivial": True,
        "passed": peak <= int(limit_bytes),
        "reason": "measured peak torch.cuda.max_memory_reserved within the inherited limit",
    }


# ---------------------------------------------------------------------------
# fix 2: per-record smoke loss accounting
# ---------------------------------------------------------------------------


@dataclass
class PerRecordLossLedger:
    """Per-record loss traces.

    v14 computed ``(losses[0] - losses[-1]) / |losses[0]|`` over a round-robin
    stream, so ``losses[0]`` and ``losses[-1]`` came from DIFFERENT records
    (192 updates over 3 records, 192 % 3 == 0, so the first observation was
    uniform and the last was marmousi).  That produced 0.99967 from
    1767.36 -> 0.5876 while almost nothing had been learned.

    This ledger is keyed by record, so a cross-record first/last is not
    expressible.  There is no method that returns a flat ordered list of every
    loss, and :meth:`reductions` is per record by construction.

    Nothing this class reports is evidence of learning or convergence
    (inherited veto (d)); it is a wiring sanity check only.
    """

    order: list[str] = field(default_factory=list)
    traces: dict[str, list[float]] = field(default_factory=dict)

    def observe(self, record_key: str, loss: float) -> float:
        key = str(record_key)
        value = float(loss)
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite smoke loss on record {key}")
        if key not in self.traces:
            self.traces[key] = []
            self.order.append(key)
        self.traces[key].append(value)
        return value

    @property
    def update_count(self) -> int:
        return sum(len(values) for values in self.traces.values())

    def per_record(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for key in self.order:
            values = self.traces[key]
            initial = float(values[0])
            final = float(values[-1])
            reduction = (
                (initial - final) / max(abs(initial), EPS_SQUARED)
                if len(values) >= 2
                else None
            )
            result[key] = {
                "record": key,
                "observations": len(values),
                "initial_loss": initial,
                "final_loss": final,
                "loss_reduction": reduction,
                "sufficient_observations": len(values) >= 2,
            }
        return result

    def reductions(self) -> dict[str, float | None]:
        return {key: row["loss_reduction"] for key, row in self.per_record().items()}

    def gate(self, *, threshold: float) -> dict[str, Any]:
        """Per-record loss-reduction gate.  Every record must clear it alone."""
        rows = self.per_record()
        if not rows:
            return {
                "per_record": {},
                "threshold": float(threshold),
                "passed": False,
                "reason": "no record observed",
                "computed_per_record": True,
                "cross_record_first_last_used": False,
                "is_learning_or_convergence_evidence": False,
            }
        insufficient = sorted(
            key for key, row in rows.items() if not row["sufficient_observations"]
        )
        failing = sorted(
            key
            for key, row in rows.items()
            if row["loss_reduction"] is None
            or float(row["loss_reduction"]) < float(threshold)
        )
        return {
            "per_record": rows,
            "threshold": float(threshold),
            "records_with_fewer_than_two_observations": insufficient,
            "failing_records": failing,
            "passed": not insufficient and not failing,
            "computed_per_record": True,
            "cross_record_first_last_used": False,
            "v14_defect_replaced": (
                "first and last loss of the whole round-robin stream came from "
                "different records"
            ),
            "is_learning_or_convergence_evidence": False,
            "may_promote_to_pilot_or_long": False,
        }


__all__ = [
    "ACCEPTANCE_CONVENTION",
    "CONVENTIONS",
    "ConfinementLeak",
    "CudaPeakVram",
    "EPS_SQUARED",
    "INFERENCE_VRAM_LIMIT_BYTES",
    "MaskCausalityRefusal",
    "PerRecordLossLedger",
    "TAU",
    "TRAINING_PROXY_CONVENTION",
    "TRAIN_VRAM_LIMIT_BYTES",
    "V15ContractError",
    "apply_confined_correction",
    "bitwise_identical",
    "coefficient_energy_definition",
    "confine_correction",
    "confined_oracle_upper_bound",
    "confinement_invariants",
    "convention_gains",
    "convention_metrics",
    "dimensionless_coefficient_energy",
    "energy_keep_mask",
    "frame_energy",
    "full_time_keep_mask",
    "loss_specification",
    "mask_agreement",
    "mask_provenance",
    "masked_confined_loss",
    "parent_energy_keep_mask",
    "parent_reference_energy",
    "vram_gate",
    "weighted_coefficient_map",
]
