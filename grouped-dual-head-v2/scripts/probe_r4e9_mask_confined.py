#!/usr/bin/env python3
"""Mask-confined oracle probe (r4e9), train-only, zero training.

Frozen spec: results/r4e9_mask_confined_oracle_spec_20260826.md

Why this exists.  r4e8 found that the convention (ii) matched oracle arm
(``masked_matched``) destroys convention (iii) ``global_energy_rel_l2`` on the
uniform and layered families.  Reading
``scripts/probe_r4e8_masked_oracle.py:470-499``: that arm fits coefficients on
the retained frames only but then applies the correction to *every* frame, and
convention (iii) sums over every frame.

Hypothesis H: the (iii) blow-up is located in the mask-dropped frames, and
confining the correction to the mask removes the conflict.

This probe reports, per family and per rank, two variants:

- ``unconfined`` -- byte-identical arithmetic to the r4e8 ``masked_matched`` arm.
  Its numbers are cross-checked against the stored r4e8 terminal artifact; a
  mismatch is a stop condition.
- ``confined``   -- identical, except the correction is multiplied by the
  **parent-energy** mask before it is applied, so the correction is exactly zero
  on dropped frames.  The parent mask is deployment-causal (parent frame energy
  is available at inference); r4e8 measured its agreement with the truth mask at
  0.9948 / 0.9895 / 1.0000.

The truth-mask-confined variant is computed only as an explicitly labelled
diagnostic and is never the headline.

Both variants additionally report the in-mask / out-of-mask error energy split,
which tests H directly instead of by elimination.

This is a diagnostic probe.  It trains nothing, writes no checkpoint, and
serializes no field array.  It reads the train split only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from saved_time_phase_operator_v4.instance_adaptation.contracts import onset_indices
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4_family_temporal_pod_capacity as legacy_pod
from scripts import probe_r4e8_masked_oracle as r4e8
from scripts import train_r16_dscp as v1prep


CANDIDATE = "r4e9_mask_confined_oracle"
SCRIPT_PATH = Path(__file__).resolve()
TEST_PATH = (
    PROJECT_ROOT / "tests/saved_time_phase_operator_v4/test_r4e9_mask_confined.py"
)
SPEC_PATH = PROJECT_ROOT / "results/r4e9_mask_confined_oracle_spec_20260826.md"
SPEC_SHA256 = "6846636d61f3a346d3f213bafe8056bc535da0bcaf9a3b28c4c01f84204a2923"

PARENT_SPEC_PATH = PROJECT_ROOT / "results/r4e8_masked_oracle_probe_spec_20260826.md"
PARENT_SPEC_SHA256 = "ae977f2d9211f002782944cceeaa0a972d719d24f80871c23b8cbf2edf37eee6"
PARENT_SCRIPT_PATH = PROJECT_ROOT / "scripts/probe_r4e8_masked_oracle.py"
PARENT_SCRIPT_SHA256 = (
    "3373b5838e49da81e5a8e2a3bada3e236aa7855d4bcdcd149cd4e1233900b485"
)
# Read-only reference artifact.  Never written, never modified.
R4E8_TERMINAL_PATH = (
    PROJECT_ROOT / "results/r4e8_masked_oracle_probe_20260826/terminal.json"
)

RESULT_DIR = PROJECT_ROOT / "results/r4e9_mask_confined_oracle_20260826"
TERMINAL_PATH = RESULT_DIR / "terminal.json"
SPLIT_PATH = RESULT_DIR / "error_energy_splits.json"

# Inherited, unchanged, from the reused r4e8 code path.
FAMILIES = r4e8.FAMILIES
RANKS = r4e8.RANKS
TIME_COUNT = r4e8.TIME_COUNT
TIME_BLOCK = r4e8.TIME_BLOCK
SPATIAL_CHUNK = r4e8.SPATIAL_CHUNK
EPS_SQUARED = r4e8.EPS_SQUARED
TAU = r4e8.TAU
PANEL_IDS = r4e8.PANEL_IDS

# ---------------------------------------------------------------------------
# Pre-stated decision rule, transcribed from the frozen spec lines 59-69.
# These constants are fixed before the probe runs and are not tuned on its
# outcome.
# ---------------------------------------------------------------------------

# spec (a): confined (ii) gain must not fall more than this below unconfined
CONFINED_CONVENTION_II_TOLERANCE = 0.02
# spec (a): confined (iii) gain must be at least this
CONFINED_CONVENTION_III_MINIMUM = 0.0
# Operationalization of "the blow-up is located in the mask-dropped frames".
# The spec states the location claim qualitatively; this is the majority reading
# fixed before observing any r4e9 number: of the total error-energy increase the
# unconfined arm inflicts relative to the parent, at least this fraction must sit
# outside the mask.  The raw fraction is reported for every point regardless of
# the threshold, so the reader can re-decide independently.
OUTSIDE_MASK_BLOWUP_SHARE_MINIMUM = 0.5
# A (family, rank) point only has a blow-up to explain if the unconfined arm
# actually lost ground under convention (iii).
BLOWUP_CONVENTION_III_GAIN_CEILING = 0.0
DECISION_RANK = 16

GPU_SECONDS_MAXIMUM = 2400.0
MIN_FREE_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 32 * 1024**2
MAX_RESULT_DIR_BYTES = 500 * 1024**2
PEAK_CUDA_BYTES_MAXIMUM = int(23.5 * 1024**3)

ALLOWED_WRITE_ROOTS = (RESULT_DIR,)

VARIANTS = ("unconfined", "confined")
CONVENTIONS = (
    "unmasked_per_frame_unsquared_mean",
    "masked_energy_floored_per_frame",
    "global_energy_rel_l2",
)


class ProbeContractError(RuntimeError):
    """The frozen r4e9 probe contract is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# binding and scope guards
# --------------------------------------------------------------------------


def _verify_one(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ProbeContractError(f"{label} is missing: {path}")
    observed = parent_runtime.sha256_file(path)
    if observed != expected:
        raise parent_runtime.BindingDriftError(
            f"{label} sha256 drift: expected {expected}, observed {observed}"
        )
    return {
        "path": str(path),
        "sha256": observed,
        "size_bytes": int(path.stat().st_size),
    }


def verify_bindings() -> dict[str, Any]:
    """The r4e9 spec, the r4e8 spec and the reused r4e8 code path must all hold."""
    return {
        "spec": _verify_one(SPEC_PATH, SPEC_SHA256, "frozen r4e9 spec"),
        "parent_spec": _verify_one(
            PARENT_SPEC_PATH, PARENT_SPEC_SHA256, "frozen r4e8 spec"
        ),
        "reused_code_path": _verify_one(
            PARENT_SCRIPT_PATH, PARENT_SCRIPT_SHA256, "reused r4e8 probe script"
        ),
    }


def assert_write_allowed(path: Path) -> Path:
    """Refuse any write outside the authorized r4e9 result directory."""
    resolved = path.resolve()
    for root in ALLOWED_WRITE_ROOTS:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return resolved
    raise ProbeContractError(f"write outside the authorized scope: {resolved}")


def load_r4e8_reference() -> dict[str, Any]:
    """Read the stored r4e8 masked_matched gains.  Read-only."""
    if not R4E8_TERMINAL_PATH.exists():
        raise ProbeContractError(f"r4e8 reference artifact is missing: {R4E8_TERMINAL_PATH}")
    payload = json.loads(R4E8_TERMINAL_PATH.read_text(encoding="utf8"))
    reference: dict[str, Any] = {
        "path": str(R4E8_TERMINAL_PATH),
        "sha256": parent_runtime.sha256_file(R4E8_TERMINAL_PATH),
        "candidate": str(payload["candidate"]),
        "families": {},
    }
    for family, item in payload["panel"].items():
        ranks = {}
        for rank in RANKS:
            arm = item["ranks"][str(rank)]["masked_matched_fit"]
            ranks[str(rank)] = {
                "gain": {name: float(arm["gain"][name]) for name in CONVENTIONS},
                "corrected_metrics": {
                    name: float(arm["corrected_metrics"][name]) for name in CONVENTIONS
                },
            }
        reference["families"][family] = {
            "sample_id": str(item["sample_id"]),
            "parent_metrics": {
                name: float(item["parent_metrics"][name]) for name in CONVENTIONS
            },
            "ranks": ranks,
        }
    return reference


# --------------------------------------------------------------------------
# error energy split
# --------------------------------------------------------------------------


def error_energy_split(
    error_squared_by_frame: torch.Tensor, keep_mask: torch.Tensor
) -> dict[str, float]:
    """Split a per-frame squared error vector inside vs outside a keep-mask.

    ``inside`` is ``sum_{t in mask} err_t``, ``outside`` is ``sum_{t not in mask}
    err_t``.  This is the quantity that tests hypothesis H directly.
    """
    error = torch.as_tensor(error_squared_by_frame, dtype=torch.float64)
    keep = torch.as_tensor(keep_mask, device=error.device).bool()
    if error.ndim != 1 or keep.shape != error.shape:
        raise ValueError("error and mask must be matching [time] vectors")
    if not bool(torch.isfinite(error).all()):
        raise FloatingPointError("per-frame squared error is non-finite")
    inside = float(error[keep].sum().item())
    outside = float(error[~keep].sum().item())
    total = inside + outside
    return {
        "inside_mask": inside,
        "outside_mask": outside,
        "total": total,
        "outside_share_of_total": outside / max(total, EPS_SQUARED),
        "inside_frame_count": int(keep.sum().item()),
        "outside_frame_count": int((~keep).sum().item()),
    }


def blowup_attribution(
    parent_split: Mapping[str, float], variant_split: Mapping[str, float]
) -> dict[str, Any]:
    """How much of a variant's total error-energy increase sits outside the mask.

    ``delta_total`` is positive exactly when the variant made the global error
    energy worse than the parent, which is what convention (iii) scores.
    """
    delta_inside = float(variant_split["inside_mask"]) - float(parent_split["inside_mask"])
    delta_outside = float(variant_split["outside_mask"]) - float(
        parent_split["outside_mask"]
    )
    delta_total = delta_inside + delta_outside
    increased = delta_total > 0.0
    return {
        "delta_inside_mask": delta_inside,
        "delta_outside_mask": delta_outside,
        "delta_total": delta_total,
        "total_error_energy_increased_vs_parent": bool(increased),
        "outside_share_of_increase": (
            delta_outside / delta_total if increased else None
        ),
    }


# --------------------------------------------------------------------------
# per-record confined oracle
# --------------------------------------------------------------------------


@torch.inference_mode()
def mask_confined_record(
    parent: torch.Tensor,
    truth: torch.Tensor,
    temporal_basis: torch.Tensor,
    *,
    k1: int,
    ranks: Sequence[int] = RANKS,
    tau: float = TAU,
    spatial_chunk: int = SPATIAL_CHUNK,
) -> dict[str, Any]:
    """Score the unconfined and mask-confined oracle arms on one train record.

    The unconfined arm reproduces ``r4e8.masked_oracle_record``'s
    ``masked_matched_fit`` arithmetic expression for expression: the coefficients
    are fitted on the truth-mask retained frames with floored denominators, and
    the correction is applied to every frame.

    The confined arm differs in exactly one multiplication: the correction is
    multiplied by the parent-energy keep mask before it is subtracted, so on
    dropped frames the correction is exactly zero and the error is bit-identical
    to the parent error.
    """
    predicted = torch.as_tensor(parent)
    target = torch.as_tensor(truth, device=predicted.device)
    if predicted.shape != target.shape or predicted.shape[0] != TIME_COUNT:
        raise ValueError("parent and truth must both be [401, H, W]")
    start = int(k1) + 1
    if not 0 < start < TIME_COUNT:
        raise ValueError("C1 causal start must lie strictly inside the time axis")

    future_parent = predicted[start:].reshape(TIME_COUNT - start, -1)
    future_truth = target[start:].reshape(TIME_COUNT - start, -1)
    future_residual = future_truth - future_parent
    frames, points = future_parent.shape

    truth_energy = r4e8.frame_energy(target[start:])
    parent_energy_by_frame = r4e8.frame_energy(predicted[start:])
    parent_error_squared = future_residual.double().square().sum(dim=1)
    residual_energy = float(parent_error_squared.sum().item())
    parent_energy_total = float(future_parent.double().square().sum().item())
    if not math.isfinite(residual_energy) or residual_energy <= 0.0:
        raise ProbeContractError("future residual energy is invalid")

    truth_keep, truth_floor = r4e8.energy_mask(truth_energy, tau=tau)
    parent_keep, parent_floor = r4e8.energy_mask(parent_energy_by_frame, tau=tau)
    agreement = r4e8.mask_agreement(truth_keep, parent_keep)

    parent_view = r4e8.convention_metrics(parent_error_squared, truth_energy, tau=tau)
    masked_weights = truth_energy.clamp_min(truth_floor).reciprocal()
    keep_index = truth_keep.nonzero(as_tuple=True)[0]
    basis_cpu = torch.as_tensor(temporal_basis, device="cpu", dtype=torch.float64)

    device = future_parent.device
    keep_device = keep_index.to(device=device)
    # headline confinement gate: parent energy mask, available at deployment
    parent_gate = parent_keep.to(device=device, dtype=torch.float64)[:, None]
    # diagnostic only: the oracle truth mask
    truth_gate = truth_keep.to(device=device, dtype=torch.float64)[:, None]

    parent_split_parent_mask = error_energy_split(parent_error_squared, parent_keep)
    parent_split_truth_mask = error_energy_split(parent_error_squared, truth_keep)

    output: dict[str, Any] = {
        "future_start_index": start,
        "c1_k1": int(k1),
        "future_frame_count": int(frames),
        "spatial_point_count": int(points),
        "future_residual_energy": residual_energy,
        "future_parent_energy": parent_energy_total,
        "fit_objective": (
            "weighted LS restricted to truth-mask retained frames with floored "
            "denominators; identical to the r4e8 masked_matched arm"
        ),
        "confinement_gate": "parent_energy_mask",
        "mask": {
            "tau": float(tau),
            "truth_energy_floor": truth_floor,
            "parent_energy_floor": parent_floor,
            "truth_masked_frame_count": int(truth_keep.sum().item()),
            "parent_masked_frame_count": int(parent_keep.sum().item()),
            "agreement_rate": agreement,
            "agreement_passes_minimum": bool(
                agreement >= r4e8.MASK_AGREEMENT_MINIMUM
            ),
            "minimum": r4e8.MASK_AGREEMENT_MINIMUM,
        },
        "parent_metrics": parent_view["metrics"],
        "parent_metric_diagnostics": parent_view["diagnostics"],
        "parent_error_energy_split": {
            "parent_mask": parent_split_parent_mask,
            "truth_mask_diagnostic": parent_split_truth_mask,
        },
        "ranks": {},
    }

    for rank_value in ranks:
        rank = int(rank_value)
        if rank <= 0 or rank > basis_cpu.shape[1]:
            raise ValueError("POD rank is outside the available basis")
        design = basis_cpu[start:, :rank]
        # exactly the r4e8 masked arm's coefficient map
        masked_map = r4e8.weighted_coefficient_map(
            design[keep_index.cpu()], masked_weights.detach().cpu()[keep_index.cpu()]
        ).to(device=device)
        design_device = design.to(device=device)

        unconfined_error = torch.zeros(frames, dtype=torch.float64, device=device)
        confined_error = torch.zeros_like(unconfined_error)
        truth_confined_error = torch.zeros_like(unconfined_error)
        # parent error re-accumulated in the identical chunk order, so the
        # confinement invariant below is compared summation-order for
        # summation-order rather than against the whole-row reduction
        chunked_parent_error = torch.zeros_like(unconfined_error)
        max_abs_correction_outside_mask = 0.0
        unconfined_correction_energy = 0.0
        confined_correction_energy = 0.0
        unconfined_unexplained = 0.0
        confined_unexplained = 0.0
        for lo in range(0, points, int(spatial_chunk)):
            hi = min(lo + int(spatial_chunk), points)
            residual_block = future_residual[:, lo:hi].double()
            chunked_parent_error += residual_block.square().sum(dim=1)

            matched_coefficients = masked_map @ residual_block[keep_device]
            matched_correction = design_device @ matched_coefficients

            unconfined_squared = (residual_block - matched_correction).square()
            unconfined_error += unconfined_squared.sum(dim=1)
            unconfined_unexplained += float(unconfined_squared.sum().item())
            unconfined_correction_energy += float(
                matched_correction.square().sum().item()
            )

            confined_correction = parent_gate * matched_correction
            confined_squared = (residual_block - confined_correction).square()
            confined_error += confined_squared.sum(dim=1)
            confined_unexplained += float(confined_squared.sum().item())
            confined_correction_energy += float(
                confined_correction.square().sum().item()
            )
            outside_block = confined_correction[~parent_keep.to(device=device)]
            if outside_block.numel():
                max_abs_correction_outside_mask = max(
                    max_abs_correction_outside_mask,
                    float(outside_block.abs().max().item()),
                )

            truth_confined_correction = truth_gate * matched_correction
            truth_confined_error += (
                (residual_block - truth_confined_correction).square().sum(dim=1)
            )

            del residual_block, matched_coefficients, matched_correction
            del unconfined_squared, confined_correction, confined_squared
            del truth_confined_correction, outside_block

        # Structural invariants of confinement, asserted on the real data.
        #
        # The primary claim is that the applied correction is *identically zero*
        # outside the parent mask, so the confined error there is the parent
        # error computed by the same arithmetic.  Both are checked exactly.
        #
        # The confined error is additionally compared against the whole-row
        # parent reduction; that comparison is only reported, not gated, because
        # chunked and whole-row float64 summation differ in association order and
        # legitimately disagree in the last ulps.
        outside = ~parent_keep.to(device=device)
        inside = parent_keep.to(device=device)
        if max_abs_correction_outside_mask != 0.0:
            raise ProbeContractError(
                "confined correction is not identically zero outside the parent mask: "
                f"max |correction| = {max_abs_correction_outside_mask}"
            )
        whole_row_deviation = 0.0
        if bool(outside.any()):
            deviation = float(
                (confined_error[outside] - chunked_parent_error[outside])
                .abs()
                .max()
                .item()
            )
            if deviation != 0.0:
                raise ProbeContractError(
                    "confined error differs from the identically-chunked parent error "
                    f"outside the mask by {deviation}; confinement is not exact"
                )
            whole_row_deviation = float(
                (confined_error[outside] - parent_error_squared[outside])
                .abs()
                .max()
                .item()
            )

        views = {
            "unconfined": r4e8.convention_metrics(unconfined_error, truth_energy, tau=tau),
            "confined": r4e8.convention_metrics(confined_error, truth_energy, tau=tau),
        }
        truth_confined_view = r4e8.convention_metrics(
            truth_confined_error, truth_energy, tau=tau
        )

        splits = {
            "unconfined": {
                "parent_mask": error_energy_split(unconfined_error, parent_keep),
                "truth_mask_diagnostic": error_energy_split(unconfined_error, truth_keep),
            },
            "confined": {
                "parent_mask": error_energy_split(confined_error, parent_keep),
                "truth_mask_diagnostic": error_energy_split(confined_error, truth_keep),
            },
        }

        item: dict[str, Any] = {
            "rank": rank,
            "fit_frame_count": int(keep_index.numel()),
            "variants": {},
            "error_energy_split": {
                "parent": {
                    "parent_mask": parent_split_parent_mask,
                    "truth_mask_diagnostic": parent_split_truth_mask,
                },
                **splits,
            },
            "blowup_attribution": {
                name: blowup_attribution(
                    parent_split_parent_mask, splits[name]["parent_mask"]
                )
                for name in VARIANTS
            },
            "confinement_invariants": {
                "correction_identically_zero_outside_parent_mask": True,
                "max_abs_applied_correction_outside_parent_mask": (
                    max_abs_correction_outside_mask
                ),
                "outside_parent_mask_error_equals_chunk_matched_parent_exactly": True,
                "outside_parent_mask_max_abs_deviation_vs_whole_row_parent": (
                    whole_row_deviation
                ),
                "whole_row_deviation_note": (
                    "reported, not gated: the chunked and whole-row float64 reductions "
                    "differ in association order, so a last-ulp disagreement here is "
                    "floating-point summation order and not a confinement leak; the "
                    "exact claims are the two booleans above"
                ),
                "inside_parent_mask_max_abs_deviation_vs_unconfined": float(
                    (confined_error[inside] - unconfined_error[inside]).abs().max().item()
                )
                if bool(inside.any())
                else 0.0,
            },
            "truth_mask_confined_diagnostic": {
                "note": (
                    "NOT the headline variant; the truth mask is not available at "
                    "deployment and this arm is reported only to bound the cost of "
                    "using the parent mask as the proxy gate"
                ),
                "corrected_metrics": truth_confined_view["metrics"],
                "gain": r4e8.convention_gains(
                    parent_view["metrics"], truth_confined_view["metrics"]
                ),
            },
            "finite": True,
        }
        for name in VARIANTS:
            unexplained = (
                unconfined_unexplained if name == "unconfined" else confined_unexplained
            )
            correction_energy = (
                unconfined_correction_energy
                if name == "unconfined"
                else confined_correction_energy
            )
            item["variants"][name] = {
                "corrected_metrics": views[name]["metrics"],
                "gain": r4e8.convention_gains(
                    parent_view["metrics"], views[name]["metrics"]
                ),
                "raw_residual_energy_capture": 1.0 - unexplained / residual_energy,
                "unexplained_raw_residual_energy": unexplained,
                "correction_energy": correction_energy,
                "correction_energy_ratio": correction_energy
                / max(parent_energy_total, EPS_SQUARED),
            }
        if not _all_finite(item):
            raise FloatingPointError(f"rank-{rank} confined oracle output is non-finite")
        output["ranks"][str(rank)] = item
        del masked_map, design_device, unconfined_error, confined_error
        del truth_confined_error, chunked_parent_error

    return output


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


# --------------------------------------------------------------------------
# gate 1: the unconfined arm must reproduce r4e8
# --------------------------------------------------------------------------


def compare_against_r4e8(
    panel: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Every (family, rank) unconfined gain must equal the stored r4e8 value."""
    points: dict[str, Any] = {}
    exact = True
    worst = 0.0
    for family, item in panel.items():
        family_reference = reference["families"][family]
        if str(item["sample_id"]) != str(family_reference["sample_id"]):
            raise ProbeContractError(
                f"r4e8 comparison record drift for {family}: "
                f"{item['sample_id']} vs {family_reference['sample_id']}"
            )
        for rank in RANKS:
            mine = item["ranks"][str(rank)]["variants"]["unconfined"]["gain"]
            theirs = family_reference["ranks"][str(rank)]["gain"]
            differences = {
                name: float(mine[name]) - float(theirs[name]) for name in CONVENTIONS
            }
            point_exact = all(
                float(mine[name]) == float(theirs[name]) for name in CONVENTIONS
            )
            largest = max(abs(v) for v in differences.values())
            exact = exact and point_exact
            worst = max(worst, largest)
            points[f"{family}_rank{rank}"] = {
                "this_probe_unconfined_gain": {
                    name: float(mine[name]) for name in CONVENTIONS
                },
                "r4e8_masked_matched_gain": {
                    name: float(theirs[name]) for name in CONVENTIONS
                },
                "absolute_difference": differences,
                "bitwise_identical": bool(point_exact),
                "max_absolute_difference": largest,
            }
    return {
        "reference_path": reference["path"],
        "reference_sha256": reference["sha256"],
        "point_count": len(points),
        "all_bitwise_identical": bool(exact),
        "max_absolute_difference_over_all_points": worst,
        "points": points,
    }


# --------------------------------------------------------------------------
# pre-stated decision rule (spec lines 59-69)
# --------------------------------------------------------------------------


def apply_decision_rule(panel: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen three-branch rule.  Nothing here is tuned on the outcome."""
    per_family: dict[str, Any] = {}
    for family, item in panel.items():
        ranks: dict[str, Any] = {}
        for rank in RANKS:
            entry = item["ranks"][str(rank)]
            unconfined = entry["variants"]["unconfined"]["gain"]
            confined = entry["variants"]["confined"]["gain"]
            attribution = entry["blowup_attribution"]["unconfined"]
            unconfined_iii = float(unconfined["global_energy_rel_l2"])
            has_blowup = unconfined_iii < BLOWUP_CONVENTION_III_GAIN_CEILING
            share = attribution["outside_share_of_increase"]
            explained = (
                None
                if not has_blowup
                else bool(
                    share is not None
                    and float(share) >= OUTSIDE_MASK_BLOWUP_SHARE_MINIMUM
                )
            )
            ranks[str(rank)] = {
                "unconfined_gain": {name: float(unconfined[name]) for name in CONVENTIONS},
                "confined_gain": {name: float(confined[name]) for name in CONVENTIONS},
                "convention_ii_confined_minus_unconfined": float(
                    confined["masked_energy_floored_per_frame"]
                )
                - float(unconfined["masked_energy_floored_per_frame"]),
                "convention_ii_within_tolerance": bool(
                    float(confined["masked_energy_floored_per_frame"])
                    >= float(unconfined["masked_energy_floored_per_frame"])
                    - CONFINED_CONVENTION_II_TOLERANCE
                ),
                "convention_iii_confined_nonnegative": bool(
                    float(confined["global_energy_rel_l2"])
                    >= CONFINED_CONVENTION_III_MINIMUM
                ),
                "unconfined_has_convention_iii_blowup": bool(has_blowup),
                "outside_mask_share_of_unconfined_increase": (
                    float(share) if share is not None else None
                ),
                "blowup_located_outside_mask": explained,
            }
        decision_entry = ranks[str(DECISION_RANK)]
        per_family[family] = {
            "sample_id": str(item["sample_id"]),
            "mask_agreement_rate": float(item["mask"]["agreement_rate"]),
            "ranks": ranks,
            "rank16_convention_ii_within_tolerance": decision_entry[
                "convention_ii_within_tolerance"
            ],
            "rank16_convention_iii_nonnegative": decision_entry[
                "convention_iii_confined_nonnegative"
            ],
            "rank16_passes_branch_a_conditions": bool(
                decision_entry["convention_ii_within_tolerance"]
                and decision_entry["convention_iii_confined_nonnegative"]
            ),
            "rank16_blowup_located_outside_mask": decision_entry[
                "blowup_located_outside_mask"
            ],
            "confined_convention_iii_negative_at_every_measured_rank": bool(
                all(
                    not ranks[str(rank)]["convention_iii_confined_nonnegative"]
                    for rank in RANKS
                )
            ),
        }

    branch_a_families = {
        family: bool(value["rank16_passes_branch_a_conditions"])
        for family, value in per_family.items()
    }
    all_branch_a = all(branch_a_families.values())

    # H is evaluated at the decision rank on the points that actually blew up.
    blowup_points = {
        family: value["ranks"][str(DECISION_RANK)]
        for family, value in per_family.items()
        if value["ranks"][str(DECISION_RANK)]["unconfined_has_convention_iii_blowup"]
    }
    h_explained = {
        family: bool(entry["blowup_located_outside_mask"])
        for family, entry in blowup_points.items()
    }
    h_confirmed = bool(blowup_points) and all(h_explained.values())
    h_refuted = bool(blowup_points) and not all(h_explained.values())

    failing_iii = sorted(
        family
        for family, value in per_family.items()
        if not value["rank16_convention_iii_nonnegative"]
    )

    if h_refuted:
        branch = "c_hypothesis_H_refuted_stop_do_not_freeze_v15"
        verdict = (
            "the outside-mask error split does not account for the convention (iii) "
            "blow-up at rank 16; hypothesis H is refuted and v15 must not be frozen"
        )
        abstention = []
    elif all_branch_a:
        branch = "a_H_confirmed_and_confinement_works_v15_may_be_frozen"
        verdict = (
            "every family at rank 16 keeps its convention (ii) gain within "
            f"{CONFINED_CONVENTION_II_TOLERANCE} of the unconfined arm and reaches a "
            "nonnegative convention (iii) gain; the v15 objective is masked loss plus "
            "mask-confined correction"
        )
        abstention = []
    elif failing_iii:
        branch = "b_H_confirmed_but_confinement_insufficient_route_family_to_abstention"
        located = (
            "the outside-mask split locates the blow-up"
            if h_confirmed
            else "no family had a convention (iii) blow-up to locate at the decision rank"
        )
        verdict = (
            located
            + ", but confinement does not lift convention (iii) to nonnegative for "
            + ", ".join(failing_iii)
            + "; that family cannot be helped by this ansatz at any measured rank and "
            "v15 must route it to abstention with output bit-identical to the parent"
        )
        abstention = [
            family
            for family in failing_iii
            if per_family[family][
                "confined_convention_iii_negative_at_every_measured_rank"
            ]
        ]
    else:
        # The measured outcome is outside the three pre-stated branches: every
        # family clears convention (iii), so branch (b)'s own precondition
        # ("some family confined (iii) gain is still < 0") is false, yet at least
        # one family loses more than the frozen 0.02 of convention (ii) gain, so
        # branch (a) is false too.  The spec did not price this case, and the
        # honest report is to say so rather than to widen either threshold.
        losing_ii = sorted(
            family
            for family, value in per_family.items()
            if not value["rank16_convention_ii_within_tolerance"]
        )
        branch = "unclassified_outside_the_three_pre_stated_branches_escalate_to_lead"
        verdict = (
            "every family clears convention (iii) at rank 16, so branch (b)'s "
            "precondition (some family still negative under (iii)) does not hold, but "
            + ", ".join(losing_ii)
            + " loses more than the frozen "
            + f"{CONFINED_CONVENTION_II_TOLERANCE} of convention (ii) gain, so branch "
            "(a) does not hold either; the frozen rule does not cover this outcome and "
            "v15 must not be frozen on this probe alone without a new authorization"
        )
        abstention = []

    return {
        "rule_source": "results/r4e9_mask_confined_oracle_spec_20260826.md lines 59-69",
        "decision_rank": DECISION_RANK,
        "thresholds": {
            "confined_convention_ii_tolerance": CONFINED_CONVENTION_II_TOLERANCE,
            "confined_convention_iii_minimum": CONFINED_CONVENTION_III_MINIMUM,
            "outside_mask_blowup_share_minimum": OUTSIDE_MASK_BLOWUP_SHARE_MINIMUM,
            "threshold_provenance": (
                "the two convention thresholds are transcribed verbatim from the frozen "
                "spec; the outside-mask share threshold is the majority reading of the "
                "spec's qualitative location claim, fixed before the probe ran, and the "
                "raw share is reported for every point"
            ),
        },
        "families": per_family,
        "hypothesis_H": {
            "statement": (
                "the convention (iii) blow-up of the unconfined masked arm is located in "
                "the mask-dropped frames"
            ),
            "evaluated_at_rank": DECISION_RANK,
            "families_with_blowup": sorted(blowup_points),
            "families_without_blowup_nothing_to_explain": sorted(
                set(per_family) - set(blowup_points)
            ),
            "blowup_located_outside_mask_by_family": h_explained,
            "confirmed": h_confirmed,
            "refuted": h_refuted,
        },
        "branch_a_conditions_by_family": branch_a_families,
        "families_failing_convention_iii_at_rank16": failing_iii,
        "families_routed_to_abstention": abstention,
        "branch": branch,
        "verdict": verdict,
        "per_family_per_rank_reported": True,
        "collapsed_to_single_threshold": False,
    }


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------


def directory_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += int((Path(root) / name).stat().st_size)
    return total


@torch.inference_mode()
def run_probe(
    *, device: torch.device, physical_gpu_index: int, write: bool
) -> dict[str, Any]:
    started = time.monotonic()
    bindings = verify_bindings()
    v14 = r4e8.verify_v14_bindings()
    reference = load_r4e8_reference()
    disk_before = parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    checkpoint_before = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    if checkpoint_before != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError(
            "parent checkpoint hash mismatch before probe"
        )
    gpu = parent_runtime.gpu_identity(physical_gpu_index, device)

    model, normalizer, manifest, run_identity = parent_runtime.load_model_context(device)
    time_axis = torch.tensor(manifest["time_s"], dtype=torch.float64)
    basis_records = v1prep.basis_records(manifest)
    torch.cuda.reset_peak_memory_stats(device)

    basis_by_family: dict[str, torch.Tensor] = {}
    basis_meta: dict[str, Any] = {}
    for family in FAMILIES:
        eigenvalues, signed, meta = r4e8.build_family_basis(
            model, normalizer, manifest, basis_records[family], device=device
        )
        basis_by_family[family] = signed
        basis_meta[family] = meta
        del eigenvalues

    basis_cross_check = r4e8.cross_check_frozen_basis(basis_by_family)

    panel: dict[str, Any] = {}
    for family, sample_id in PANEL_IDS:
        record = r4e8.record_from_manifest(manifest, sample_id)
        if record.family != family:
            raise ProbeContractError(f"panel family drift: {sample_id}")
        loaded = parent_runtime.load_record_input(record)
        predicted = legacy_pod.generate_parent_full401(
            model, normalizer, loaded, manifest, device=device, time_block=TIME_BLOCK
        )
        truth, truth_hash = r4e8.load_authorized_train_truth(record, device=device)
        observed = onset_indices(
            time_axis,
            t0_s=float(loaded.source_parameters[3]),
            f0_hz=float(loaded.source_parameters[2]),
        )
        scored = mask_confined_record(
            predicted, truth, basis_by_family[family], k1=int(observed[1]), ranks=RANKS
        )
        # in-run cross-check: run the untouched r4e8 code path on the identical
        # tensors and require the unconfined arm to match it exactly
        r4e8_scored = r4e8.masked_oracle_record(
            predicted, truth, basis_by_family[family], k1=int(observed[1]), ranks=RANKS
        )
        r4e8_scored.pop("profiles", None)
        in_run: dict[str, Any] = {}
        for rank in RANKS:
            mine = scored["ranks"][str(rank)]["variants"]["unconfined"]["gain"]
            theirs = r4e8_scored["ranks"][str(rank)]["masked_matched_fit"]["gain"]
            differences = {
                name: float(mine[name]) - float(theirs[name]) for name in CONVENTIONS
            }
            identical = all(
                float(mine[name]) == float(theirs[name]) for name in CONVENTIONS
            )
            in_run[str(rank)] = {
                "absolute_difference": differences,
                "bitwise_identical": bool(identical),
            }
            if not identical:
                raise ProbeContractError(
                    "in-run unconfined arm diverges from the untouched r4e8 code path "
                    f"for {family} rank {rank}: {differences}"
                )
        panel[family] = {
            **scored,
            "sample_id": record.sample_id,
            "group_id": record.group_id,
            "source_index": int(record.source_index),
            "family": family,
            "split": "train",
            "manifest_sample_sha256": record.manifest_sample_sha256,
            "nontruth_input_sha256": loaded.nontruth_input_sha256,
            "train_truth_sha256": truth_hash,
            "observed_indices": [int(v) for v in observed],
            "in_run_r4e8_code_path_cross_check": in_run,
        }
        del predicted, truth, r4e8_scored
        torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    reproduction = compare_against_r4e8(panel, reference)
    if not bool(reproduction["all_bitwise_identical"]):
        raise ProbeContractError(
            "gate 1 failed: the unconfined variant does not reproduce the stored r4e8 "
            "masked_matched gains at every (family, rank) point; max absolute "
            f"difference {reproduction['max_absolute_difference_over_all_points']}"
        )

    elapsed = float(time.monotonic() - started)
    if elapsed > GPU_SECONDS_MAXIMUM:
        raise parent_runtime.BudgetExceededError("probe exceeded its GPU-second budget")
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if peak_reserved > PEAK_CUDA_BYTES_MAXIMUM:
        raise ProbeContractError("peak CUDA reserved memory exceeds the probe gate")

    decision = apply_decision_rule(panel)

    splits = {}
    for family in panel:
        splits[family] = {
            "sample_id": panel[family]["sample_id"],
            "future_start_index": panel[family]["future_start_index"],
            "mask": panel[family]["mask"],
            "parent_error_energy_split": panel[family]["parent_error_energy_split"],
            "ranks": {
                str(rank): {
                    "error_energy_split": panel[family]["ranks"][str(rank)][
                        "error_energy_split"
                    ],
                    "blowup_attribution": panel[family]["ranks"][str(rank)][
                        "blowup_attribution"
                    ],
                }
                for rank in RANKS
            },
        }

    checkpoint_after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    if checkpoint_after != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError("parent checkpoint changed during the probe")
    v14_after = r4e8.verify_v14_bindings()
    bindings_after = verify_bindings()

    payload = {
        "schema": "r4e9_mask_confined_oracle_result_v1",
        "candidate": CANDIDATE,
        "status": "success",
        "kind": "diagnostic_probe_not_a_promotion_candidate",
        "started_utc": utc_now(),
        "completed_utc": utc_now(),
        "exact_blocker": "none",
        "claim_scope": (
            "offline_train_only_mask_confined_oracle_upper_bound_under_three_metric_"
            "conventions_not_online_adaptation_not_validation_not_test_id"
        ),
        "why": (
            "the r4e8 masked_matched arm fits on retained frames but applies the "
            "correction to every frame; hypothesis H says the convention (iii) blow-up "
            "lives in the dropped frames"
        ),
        "bindings": bindings,
        "bindings_after": bindings_after,
        "sealed_data_attestation": {
            "train_truth_opened_for_offline_oracle": True,
            "validation_opened": False,
            "test_id_opened": False,
            "field_arrays_written_to_disk": False,
            "authorized_train_sample_ids": sorted(r4e8.ALLOWED_TRUTH_SAMPLE_IDS),
        },
        "checkpoint_attestation": {
            "writes": 0,
            "rollback_path": str(parent_runtime.CHECKPOINT_PATH),
            "rollback_sha256_before": checkpoint_before,
            "rollback_sha256_after": checkpoint_after,
            "unchanged": checkpoint_after == parent_runtime.CHECKPOINT_SHA256,
        },
        "frozen_v14_bindings_before": v14,
        "frozen_v14_bindings_after": v14_after,
        "training": {
            "optimizer_used": False,
            "backward_used": False,
            "gradient_steps": 0,
            "checkpoints_written": 0,
        },
        "protocol": {
            "split": "train",
            "time_count": TIME_COUNT,
            "time_block": TIME_BLOCK,
            "ranks": list(RANKS),
            "tau": TAU,
            "metric_eps_squared": EPS_SQUARED,
            "variants": {
                "unconfined": (
                    "the r4e8 masked_matched arm verbatim: weighted LS on truth-mask "
                    "retained frames with floored denominators, correction applied to "
                    "every future frame"
                ),
                "confined": (
                    "identical fit, correction multiplied by the parent-energy keep mask "
                    "before it is applied, so it is exactly zero on dropped frames"
                ),
                "truth_mask_confined": (
                    "labelled diagnostic only, not deployment-causal, never the headline"
                ),
            },
            "conventions": {
                "i_unmasked_per_frame_unsquared_mean": "mean_t sqrt(||err_t||^2 / max(||truth_t||^2, 1e-30))",
                "ii_masked_energy_floored_per_frame": (
                    "mean over frames with ||truth_t||^2 >= tau*max_s||truth_s||^2 of "
                    "sqrt(||err_t||^2 / max(||truth_t||^2, tau*max_s||truth_s||^2))"
                ),
                "iii_global_energy_rel_l2": "sqrt(sum_t ||err_t||^2 / sum_t ||truth_t||^2)",
            },
            "gain": "(parent_metric - corrected_metric) / max(parent_metric, 1e-30) per convention",
            "error_energy_split": (
                "sum of per-frame squared error inside and outside the keep mask, "
                "reported for parent, unconfined and confined; the headline split uses "
                "the parent-energy mask"
            ),
            "mask_causality": (
                "the confinement gate uses only parent frame energy, which is available "
                "at deployment; the truth mask appears only in the labelled diagnostic"
            ),
            "basis": (
                "per-family raw residual temporal POD over the two frozen v1 basis "
                "records, nested truncation to ranks 8/16/32, v1 sign rule"
            ),
        },
        "basis": basis_meta,
        "basis_cross_check_vs_frozen_v1_rank16": basis_cross_check,
        "r4e8_reproduction_gate": reproduction,
        "panel": panel,
        "decision_rule": decision,
        "budget": {
            "one_gpu_seconds_used": elapsed,
            "one_gpu_seconds_maximum": GPU_SECONDS_MAXIMUM,
            "within_budget": elapsed <= GPU_SECONDS_MAXIMUM,
        },
        "gpu": gpu,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "run_binding": {
            "run_digest": str(run_identity["run_digest"]),
            "manifest_digest": str(run_identity["manifest_digest"]),
            "time_axis_sha256": str(run_identity["time_axis_sha256"]),
            "checkpoint_sha256": parent_runtime.CHECKPOINT_SHA256,
            "script": parent_runtime.file_binding(SCRIPT_PATH),
            "reused_script": parent_runtime.file_binding(PARENT_SCRIPT_PATH),
            "test": parent_runtime.file_binding(TEST_PATH)
            if TEST_PATH.exists()
            else {"path": str(TEST_PATH), "sha256": "absent", "size_bytes": 0},
            "software": parent_runtime.software_identity(),
        },
        "disk_free_bytes_before": disk_before,
        "disk_free_bytes_after": parent_runtime.free_disk_bytes(),
    }

    if write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
        parent_runtime.atomic_json_exclusive(
            {
                "schema": "r4e9_mask_confined_oracle_error_energy_splits_v1",
                "candidate": CANDIDATE,
                "tau": TAU,
                "note": (
                    "inside/outside splits of the per-frame squared error energy; the "
                    "headline mask is the parent-energy mask"
                ),
                "families": splits,
            },
            assert_write_allowed(SPLIT_PATH),
            limit=MAX_OUTPUT_BYTES,
        )
        payload["error_energy_splits_path"] = str(SPLIT_PATH)
        parent_runtime.atomic_json_exclusive(
            payload, assert_write_allowed(TERMINAL_PATH), limit=MAX_OUTPUT_BYTES
        )
        total = directory_bytes(RESULT_DIR)
        if total > MAX_RESULT_DIR_BYTES:
            raise ProbeContractError("result directory exceeds the 500 MB cap")
        payload["result_dir_bytes"] = total
    return payload


@torch.inference_mode()
def run_smoke(*, device: torch.device, physical_gpu_index: int) -> dict[str, Any]:
    """Unscored one-family, one-record, one-rank sanity pass.  Writes nothing."""
    started = time.monotonic()
    bindings = verify_bindings()
    r4e8.verify_v14_bindings()
    parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    before = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    if before != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError("parent checkpoint hash mismatch before smoke")
    model, normalizer, manifest, _ = parent_runtime.load_model_context(device)
    torch.cuda.reset_peak_memory_stats(device)
    records = v1prep.basis_records(manifest)["layered"]
    _eigenvalues, signed, meta = r4e8.build_family_basis(
        model, normalizer, manifest, records, device=device
    )
    record = r4e8.record_from_manifest(manifest, "train_layered_00564")
    loaded = parent_runtime.load_record_input(record)
    predicted = legacy_pod.generate_parent_full401(
        model, normalizer, loaded, manifest, device=device, time_block=TIME_BLOCK
    )
    truth, truth_hash = r4e8.load_authorized_train_truth(record, device=device)
    observed = onset_indices(
        torch.tensor(manifest["time_s"], dtype=torch.float64),
        t0_s=float(loaded.source_parameters[3]),
        f0_hz=float(loaded.source_parameters[2]),
    )
    scored = mask_confined_record(predicted, truth, signed, k1=int(observed[1]), ranks=(16,))
    reference = r4e8.masked_oracle_record(
        predicted, truth, signed, k1=int(observed[1]), ranks=(16,)
    )
    mine = scored["ranks"]["16"]["variants"]["unconfined"]["gain"]
    theirs = reference["ranks"]["16"]["masked_matched_fit"]["gain"]
    identical = all(float(mine[name]) == float(theirs[name]) for name in CONVENTIONS)
    torch.cuda.synchronize(device)
    after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    return {
        "status": "passed" if identical else "failed",
        "unscored": True,
        "utc": utc_now(),
        "bindings": bindings,
        "sample_id": record.sample_id,
        "train_truth_sha256": truth_hash,
        "layered_basis_covariance_trace": meta["covariance_trace"],
        "observed_indices": [int(v) for v in observed],
        "unconfined_matches_r4e8_code_path_bitwise": bool(identical),
        "unscored_rank16": scored,
        "seconds": float(time.monotonic() - started),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "checkpoint_unchanged": after == parent_runtime.CHECKPOINT_SHA256,
        "arrays_serialized": False,
        "writes": 0,
        "validation_opened": False,
        "test_id_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "dry", "measured"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.mode == "smoke":
        payload = run_smoke(device=device, physical_gpu_index=args.physical_gpu_index)
    else:
        payload = run_probe(
            device=device,
            physical_gpu_index=args.physical_gpu_index,
            write=args.mode == "measured",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"passed", "success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOWUP_CONVENTION_III_GAIN_CEILING",
    "CANDIDATE",
    "CONFINED_CONVENTION_II_TOLERANCE",
    "CONFINED_CONVENTION_III_MINIMUM",
    "CONVENTIONS",
    "DECISION_RANK",
    "OUTSIDE_MASK_BLOWUP_SHARE_MINIMUM",
    "PARENT_SCRIPT_SHA256",
    "PARENT_SPEC_SHA256",
    "RANKS",
    "SPEC_SHA256",
    "VARIANTS",
    "ProbeContractError",
    "apply_decision_rule",
    "assert_write_allowed",
    "blowup_attribution",
    "compare_against_r4e8",
    "error_energy_split",
    "load_r4e8_reference",
    "mask_confined_record",
    "verify_bindings",
]
