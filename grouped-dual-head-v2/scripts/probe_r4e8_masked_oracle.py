#!/usr/bin/env python3
"""Train-only masked-oracle recompute probe (r4e8), zero training.

Frozen spec: results/r4e8_masked_oracle_probe_spec_20260826.md

Why this exists.  The v14 smoke gate compared an ``oracle_gain`` measured on
``train_uniform_00321``/``train_layered_00564``/``train_marmousi_00385`` against
three hardcoded constants
(``0.2224565778108675``/``0.37334545199603053``/``0.29789271567937464`` in
``r16_dscp_engine_v3.smoke_gates``) that were measured on *different* records
(``train_uniform_00102``/``train_layered_01032``/``synthetic_train_marmousi_fresh_v1``,
see ``results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_20260826.json``,
``paired_changes_vs_parent.mean_per_frame_relative_l2_reduction`` at rank 16).
The threshold was therefore transplanted across records and the gate is not
decidable.  This probe recomputes the achievable-gain reference on the records
that are actually scored, under three explicitly named metric conventions, and
checks whether a deployment-causal parent-energy mask can stand in for the
truth-energy mask.

This is a diagnostic probe.  It fits nothing, trains nothing, writes no
checkpoint, and serializes no field array.  It reads train split only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import c1_causal_mask
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4_family_temporal_pod_capacity as legacy_pod
from scripts import probe_r4_pod_metric_balance_factorial as factorial
from scripts import train_r16_dscp as v1prep


CANDIDATE = "r4e8_masked_oracle_probe_20260826"
SCRIPT_PATH = Path(__file__).resolve()
TEST_PATH = (
    PROJECT_ROOT / "tests/saved_time_phase_operator_v4/test_r4e8_masked_oracle.py"
)
SPEC_PATH = PROJECT_ROOT / "results/r4e8_masked_oracle_probe_spec_20260826.md"
SPEC_SHA256 = "ae977f2d9211f002782944cceeaa0a972d719d24f80871c23b8cbf2edf37eee6"
RESULT_DIR = PROJECT_ROOT / "results/r4e8_masked_oracle_probe_20260826"
TERMINAL_PATH = RESULT_DIR / "terminal.json"
PROFILE_PATH = RESULT_DIR / "energy_profiles.json"

V14_PREFLIGHT_PATH = PROJECT_ROOT / "results/r16_dscp_v14/design_preflight.json"
V1_BASIS_PATH = PROJECT_ROOT / "results/r16_dscp_v1/basis_rank16.pt"

FAMILIES = ("uniform", "layered", "marmousi")
RANKS = (8, 16, 32)
TIME_COUNT = 401
TIME_BLOCK = 16
SPATIAL_CHUNK = 4096
EPS_SQUARED = 1.0e-30
TAU = 1.0e-3

# The exact v14 smoke panel.
PANEL_IDS = (
    ("uniform", "train_uniform_00321"),
    ("layered", "train_layered_00564"),
    ("marmousi", "train_marmousi_00385"),
)

# The v14 hardcoded oracle constants and the foreign records they came from.
V14_TRANSPLANTED_ORACLE = {
    "uniform": {"value": 0.2224565778108675, "measured_on": "train_uniform_00102"},
    "layered": {"value": 0.37334545199603053, "measured_on": "train_layered_01032"},
    "marmousi": {
        "value": 0.29789271567937464,
        "measured_on": "synthetic_train_marmousi_fresh_v1",
    },
}

# Pre-stated decision rule from the frozen spec.  Not a promotion gate.
RANK16_MASKED_GAIN_MINIMUM = 0.20
RANK32_MARGINAL_GAIN_MAXIMUM = 0.05
MASK_AGREEMENT_MINIMUM = 0.90

# Self-consistency reproduction targets, in the OLD UNMASKED convention, on the
# published files' own records.
SELF_CHECK = {
    "uniform_rank16_unmasked_per_frame_reduction": {
        "sample_id": "train_uniform_00002",
        "family": "uniform",
        "published_value": -1.639005450056983,
        "published_file": "results/r4e7_family_temporal_pod_capacity_train9_v1_20260825.json",
        "published_pointer": "family_results.uniform.confirmation_record.ranks.16.relative_error_reduction",
        "estimator": "legacy_pod.oracle_projection_metrics (ordinary per-spatial-point LS, unmasked per-frame unsquared mean)",
        "absolute_tolerance": 1.0e-2,
    },
    "layered_rank16_raw_residual_energy_capture": {
        "sample_id": "train_layered_01032",
        "family": "layered",
        "published_value": 0.6430370502360689,
        "published_file": "results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_20260826.json",
        "published_pointer": "family_results.layered.metrics.ranks.16.raw_residual_energy_capture",
        "estimator": "factorial.fit_projection_metrics(fit_mode='weighted')",
        "absolute_tolerance": 5.0e-3,
    },
}
# Tolerance rationale, fixed before running and not tuned on the outcome: the
# float32 residual covariance and the parent forward pass are not bitwise
# reproducible across runs on this stack.  Two historical runs of the *identical*
# layered covariance trace disagree by 1.75e-4 relative
# (4.892363548211026e-13 in the r4e7 capacity result versus
# 4.893220105265188e-13 in results/r16_dscp_v1/basis_rank16.pt metadata).  The
# tolerances above are set at the precision to which the frozen spec quotes the
# two figures (-1.639 and 0.643).
SELF_CHECK_TOLERANCE_RATIONALE = (
    "float32 covariance and parent forward are not bitwise reproducible; two "
    "historical runs of the identical layered covariance trace disagree by "
    "1.75e-4 relative (4.892363548211026e-13 vs 4.893220105265188e-13); "
    "tolerances match the precision the frozen spec quotes (-1.639, 0.643)"
)

GPU_SECONDS_MAXIMUM = 1800.0
MIN_FREE_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 32 * 1024**2
MAX_RESULT_DIR_BYTES = 500 * 1024**2
PEAK_CUDA_BYTES_MAXIMUM = int(23.5 * 1024**3)

ALLOWED_WRITE_ROOTS = (RESULT_DIR,)
ALLOWED_TRUTH_SAMPLE_IDS = frozenset(
    {
        "train_uniform_00000",
        "train_uniform_00001",
        "train_uniform_00002",
        "train_layered_00000",
        "train_layered_00004",
        "train_layered_01032",
        "train_marmousi_00000",
        "train_marmousi_00005",
        "train_uniform_00321",
        "train_layered_00564",
        "train_marmousi_00385",
    }
)


class ProbeContractError(RuntimeError):
    """The frozen r4e8 probe contract is invalid."""


class TruthScopeError(ProbeContractError):
    """A non-train or unauthorized truth access was attempted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# binding and scope guards
# --------------------------------------------------------------------------


def verify_spec() -> dict[str, Any]:
    """The frozen spec must hash to the value the lead authorized."""
    if not SPEC_PATH.exists():
        raise ProbeContractError(f"frozen spec is missing: {SPEC_PATH}")
    observed = parent_runtime.sha256_file(SPEC_PATH)
    if observed != SPEC_SHA256:
        raise parent_runtime.BindingDriftError(
            f"spec sha256 drift: expected {SPEC_SHA256}, observed {observed}"
        )
    return {
        "path": str(SPEC_PATH),
        "sha256": observed,
        "size_bytes": int(SPEC_PATH.stat().st_size),
    }


def verify_v14_bindings() -> dict[str, Any]:
    """All 16 frozen v14 preregistration bindings must be byte-identical."""
    payload = json.loads(V14_PREFLIGHT_PATH.read_text(encoding="utf8"))
    bindings = payload["bindings"]
    observed: dict[str, Any] = {}
    drift: list[str] = []
    for name in sorted(bindings):
        expected = bindings[name]
        path = Path(str(expected["path"]))
        if not path.exists():
            drift.append(f"{name}:missing")
            continue
        digest = parent_runtime.sha256_file(path)
        size = int(path.stat().st_size)
        ok = digest == str(expected["sha256"]) and size == int(expected["size_bytes"])
        observed[name] = {"sha256": digest, "size_bytes": size, "unchanged": bool(ok)}
        if not ok:
            drift.append(name)
    if drift:
        raise parent_runtime.BindingDriftError(
            f"frozen v14 binding drift: {sorted(drift)}"
        )
    return {"count": len(observed), "all_unchanged": True, "bindings": observed}


def assert_write_allowed(path: Path) -> Path:
    """Refuse any write outside the authorized result directory."""
    resolved = path.resolve()
    for root in ALLOWED_WRITE_ROOTS:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return resolved
    raise ProbeContractError(f"write outside the authorized scope: {resolved}")


def record_from_manifest(manifest: Mapping[str, Any], sample_id: str) -> Any:
    """Resolve exactly one train manifest row by sample id."""
    matches = [row for row in manifest["records"] if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ProbeContractError(f"sample id is not unique in manifest: {sample_id}")
    row = matches[0]
    if str(row["split"]) != "train":
        raise TruthScopeError(f"refusing a non-train record: {sample_id}")
    if str(row["medium_type"]) not in FAMILIES:
        raise ProbeContractError(f"unknown medium family: {sample_id}")
    return parent_runtime.SelectedRecord(
        source_index=int(row["source_index"]),
        sample_id=str(row["sample_id"]),
        group_id=str(row["group_id"]),
        family=str(row["medium_type"]),
        split="train",
        split_id=int(row["split_id"]),
        manifest_sample_sha256=str(row["sample_sha256"]),
    )


def load_authorized_train_truth(record: Any, *, device: torch.device):
    """Every truth read in this probe goes through this allowlist."""
    if str(record.split) != "train":
        raise TruthScopeError("only train truth may be opened")
    if str(record.sample_id) not in ALLOWED_TRUTH_SAMPLE_IDS:
        raise TruthScopeError(f"sample id is not on the probe allowlist: {record.sample_id}")
    return legacy_pod.load_train_truth(record, device=device)


# --------------------------------------------------------------------------
# masked metric conventions
# --------------------------------------------------------------------------


def frame_energy(field: torch.Tensor) -> torch.Tensor:
    """Per-frame squared L2 energy in float64, shape [time]."""
    value = torch.as_tensor(field)
    if value.ndim < 2:
        raise ValueError("field must be [time, ...space]")
    return value.reshape(value.shape[0], -1).double().square().sum(dim=1)


def energy_mask(energy: torch.Tensor, *, tau: float = TAU) -> tuple[torch.Tensor, float]:
    """Return the boolean keep-mask and the absolute energy floor tau*max_s E_s."""
    values = torch.as_tensor(energy, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("energy must be a nonempty [time] vector")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("energy must be finite and nonnegative")
    if float(tau) <= 0.0 or float(tau) >= 1.0:
        raise ValueError("tau must lie strictly inside (0, 1)")
    peak = float(values.max().item())
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("energy profile has no positive peak")
    floor = float(tau) * peak
    return values >= floor, floor


def mask_agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    """Fraction of frames on which two boolean masks agree."""
    a = torch.as_tensor(left).bool()
    b = torch.as_tensor(right).bool()
    if a.shape != b.shape or a.ndim != 1 or a.numel() == 0:
        raise ValueError("masks must be matching nonempty [time] vectors")
    return float((a == b).double().mean().item())


def convention_metrics(
    error_squared_by_frame: torch.Tensor,
    truth_energy_by_frame: torch.Tensor,
    *,
    tau: float = TAU,
) -> dict[str, Any]:
    """The three named error conventions from one per-frame squared error vector.

    (i)   unmasked_per_frame_unsquared_mean  - what the v14 gate used
    (ii)  masked_energy_floored_per_frame    - the proposed v15 convention
    (iii) global_energy_rel_l2
    """
    error = torch.as_tensor(error_squared_by_frame, dtype=torch.float64)
    energy = torch.as_tensor(truth_energy_by_frame, dtype=torch.float64, device=error.device)
    if error.shape != energy.shape or error.ndim != 1:
        raise ValueError("error and energy must be matching [time] vectors")
    if not bool(torch.isfinite(error).all()):
        raise FloatingPointError("per-frame squared error is non-finite")
    keep, floor = energy_mask(energy, tau=tau)
    if not bool(keep.any()):
        raise ProbeContractError("energy mask retained no frames")

    unmasked = torch.sqrt(error / energy.clamp_min(EPS_SQUARED))
    floored = torch.sqrt(error / energy.clamp_min(floor))
    masked = floored[keep]
    values = {
        "unmasked_per_frame_unsquared_mean": float(unmasked.mean().item()),
        "masked_energy_floored_per_frame": float(masked.mean().item()),
        "global_energy_rel_l2": float(
            math.sqrt(
                float(error.sum().item())
                / max(float(energy.sum().item()), EPS_SQUARED)
            )
        ),
    }
    diagnostics = {
        "floored_unmasked_per_frame": float(floored.mean().item()),
        "energy_floor": floor,
        "tau": float(tau),
        "frame_count": int(error.numel()),
        "masked_frame_count": int(keep.sum().item()),
        "dropped_frame_count": int(error.numel() - int(keep.sum().item())),
    }
    if not all(math.isfinite(v) for v in values.values()):
        raise FloatingPointError("a convention metric is non-finite")
    return {"metrics": values, "diagnostics": diagnostics, "keep_mask": keep}


def convention_gains(
    parent_metrics: Mapping[str, float], corrected_metrics: Mapping[str, float]
) -> dict[str, float]:
    """Relative error reduction per convention, the same form the v14 gate used."""
    if set(parent_metrics) != set(corrected_metrics):
        raise ValueError("parent and corrected conventions disagree")
    output = {}
    for name in sorted(parent_metrics):
        base = float(parent_metrics[name])
        corrected = float(corrected_metrics[name])
        output[name] = (base - corrected) / max(base, EPS_SQUARED)
    return output


def weighted_coefficient_map(
    design: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Weighted-LS coefficient map, identical in form to the published estimator.

    This is exactly ``factorial.fit_projection_metrics`` fit_mode='weighted':
    pinv of the weight-rooted design, then un-rooted on the right.  It is also
    algebraically the normal-equation solve used by
    ``r16_dscp_training_v2.weighted_coefficient_target``; the unit test asserts
    the two agree.
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


# --------------------------------------------------------------------------
# per-record oracle
# --------------------------------------------------------------------------


@torch.inference_mode()
def masked_oracle_record(
    parent: torch.Tensor,
    truth: torch.Tensor,
    temporal_basis: torch.Tensor,
    *,
    k1: int,
    ranks: Sequence[int] = RANKS,
    tau: float = TAU,
    spatial_chunk: int = SPATIAL_CHUNK,
) -> dict[str, Any]:
    """Score one train record's weighted oracle under all three conventions."""
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

    truth_energy = frame_energy(target[start:])
    parent_energy_by_frame = frame_energy(predicted[start:])
    parent_error_squared = future_residual.double().square().sum(dim=1)
    residual_energy = float(parent_error_squared.sum().item())
    parent_energy_total = float(future_parent.double().square().sum().item())
    if not math.isfinite(residual_energy) or residual_energy <= 0.0:
        raise ProbeContractError("future residual energy is invalid")

    truth_keep, truth_floor = energy_mask(truth_energy, tau=tau)
    parent_keep, parent_floor = energy_mask(parent_energy_by_frame, tau=tau)
    agreement = mask_agreement(truth_keep, parent_keep)

    parent_view = convention_metrics(parent_error_squared, truth_energy, tau=tau)
    weights = truth_energy.clamp_min(EPS_SQUARED).reciprocal()
    # Convention (ii) matched weights: the published estimator minimizes the
    # convention (i) objective (w_t = 1/||truth_t||^2 on every frame, including
    # the decayed ones), so it is NOT an upper bound under convention (ii).  The
    # second arm below is weighted and restricted to exactly the masked,
    # energy-floored objective, which is the upper bound convention (ii) needs.
    masked_weights = truth_energy.clamp_min(truth_floor).reciprocal()
    keep_index = truth_keep.nonzero(as_tuple=True)[0]
    basis_cpu = torch.as_tensor(temporal_basis, device="cpu", dtype=torch.float64)
    ramp = c1_causal_mask(TIME_COUNT, int(k1)).to(dtype=torch.float64)[start:]

    output: dict[str, Any] = {
        "future_start_index": start,
        "c1_k1": int(k1),
        "future_frame_count": int(frames),
        "spatial_point_count": int(points),
        "future_residual_energy": residual_energy,
        "future_parent_energy": parent_energy_total,
        "weight_formula": "w_t=1/max(||truth_t||_2^2,1e-30)",
        "coefficient_fit": "weighted",
        "mask": {
            "tau": float(tau),
            "truth_energy_floor": truth_floor,
            "parent_energy_floor": parent_floor,
            "truth_masked_frame_count": int(truth_keep.sum().item()),
            "parent_masked_frame_count": int(parent_keep.sum().item()),
            "agreement_rate": agreement,
            "agreement_passes_minimum": bool(agreement >= MASK_AGREEMENT_MINIMUM),
            "minimum": MASK_AGREEMENT_MINIMUM,
        },
        "parent_metrics": parent_view["metrics"],
        "parent_metric_diagnostics": parent_view["diagnostics"],
        "ranks": {},
    }

    for rank_value in ranks:
        rank = int(rank_value)
        if rank <= 0 or rank > basis_cpu.shape[1]:
            raise ValueError("POD rank is outside the available basis")
        design = basis_cpu[start:, :rank]
        coefficient_map = weighted_coefficient_map(design, weights.detach().cpu())
        cmap = coefficient_map.to(device=future_parent.device)
        design_device = design.to(device=future_parent.device)
        ramp_device = ramp.to(device=future_parent.device)[:, None]

        # convention (ii) matched arm: fit only on retained frames, with the
        # floored denominators that convention (ii) actually scores
        masked_map = weighted_coefficient_map(
            design[keep_index.cpu()], masked_weights.detach().cpu()[keep_index.cpu()]
        ).to(device=future_parent.device)
        keep_device = keep_index.to(device=future_parent.device)

        corrected_error = torch.zeros(frames, dtype=torch.float64, device=future_parent.device)
        gated_error = torch.zeros_like(corrected_error)
        matched_error = torch.zeros_like(corrected_error)
        correction_energy = 0.0
        unexplained_energy = 0.0
        matched_correction_energy = 0.0
        matched_unexplained_energy = 0.0
        for lo in range(0, points, int(spatial_chunk)):
            hi = min(lo + int(spatial_chunk), points)
            residual_block = future_residual[:, lo:hi].double()
            coefficients = cmap @ residual_block
            correction = design_device @ coefficients
            remaining = residual_block - correction
            remaining_squared = remaining.square()
            corrected_error += remaining_squared.sum(dim=1)
            unexplained_energy += float(remaining_squared.sum().item())
            correction_energy += float(correction.square().sum().item())
            gated_error += (residual_block - ramp_device * correction).square().sum(dim=1)

            matched_coefficients = masked_map @ residual_block[keep_device]
            matched_correction = design_device @ matched_coefficients
            matched_remaining_squared = (residual_block - matched_correction).square()
            matched_error += matched_remaining_squared.sum(dim=1)
            matched_unexplained_energy += float(matched_remaining_squared.sum().item())
            matched_correction_energy += float(matched_correction.square().sum().item())
            del residual_block, coefficients, correction, remaining, remaining_squared
            del matched_coefficients, matched_correction, matched_remaining_squared

        corrected_view = convention_metrics(corrected_error, truth_energy, tau=tau)
        gains = convention_gains(parent_view["metrics"], corrected_view["metrics"])
        gated_view = convention_metrics(gated_error, truth_energy, tau=tau)
        gated_gains = convention_gains(parent_view["metrics"], gated_view["metrics"])
        matched_view = convention_metrics(matched_error, truth_energy, tau=tau)
        matched_gains = convention_gains(parent_view["metrics"], matched_view["metrics"])
        item = {
            "corrected_metrics": corrected_view["metrics"],
            "gain": gains,
            "raw_residual_energy_capture": 1.0 - unexplained_energy / residual_energy,
            "unexplained_raw_residual_energy": unexplained_energy,
            "correction_energy": correction_energy,
            "correction_energy_ratio": correction_energy
            / max(parent_energy_total, EPS_SQUARED),
            "basis_design_condition_number": factorial.condition_number(design),
            "weighted_design_condition_number": factorial.condition_number(
                weights.detach().cpu().sqrt()[:, None] * design
            ),
            "masked_frame_count": corrected_view["diagnostics"]["masked_frame_count"],
            "masked_matched_fit": {
                "fit_objective": "weighted LS restricted to retained frames with floored denominators; the convention (ii) upper bound",
                "corrected_metrics": matched_view["metrics"],
                "gain": matched_gains,
                "raw_residual_energy_capture": 1.0
                - matched_unexplained_energy / residual_energy,
                "unexplained_raw_residual_energy": matched_unexplained_energy,
                "correction_energy": matched_correction_energy,
                "correction_energy_ratio": matched_correction_energy
                / max(parent_energy_total, EPS_SQUARED),
                "fit_frame_count": int(keep_index.numel()),
                "weighted_design_condition_number": factorial.condition_number(
                    masked_weights.detach().cpu()[keep_index.cpu()].sqrt()[:, None]
                    * design[keep_index.cpu()]
                ),
            },
            "diagnostics": {
                "floored_unmasked_per_frame": corrected_view["diagnostics"][
                    "floored_unmasked_per_frame"
                ],
                "c1_ramp_gated_metrics": gated_view["metrics"],
                "c1_ramp_gated_gain": gated_gains,
            },
            "finite": True,
        }
        item["finite"] = factorial._all_finite(item)
        if not item["finite"]:
            raise FloatingPointError(f"rank-{rank} masked oracle output is non-finite")
        output["ranks"][str(rank)] = item
        del cmap, design_device, coefficient_map, masked_map

    output["profiles"] = {
        "truth_energy": [float(v) for v in truth_energy.tolist()],
        "truth_energy_normalized": [
            float(v) for v in (truth_energy / truth_energy.max()).tolist()
        ],
        "parent_energy": [float(v) for v in parent_energy_by_frame.tolist()],
        "parent_energy_normalized": [
            float(v) for v in (parent_energy_by_frame / parent_energy_by_frame.max()).tolist()
        ],
        "truth_mask": [int(v) for v in truth_keep.tolist()],
        "parent_mask": [int(v) for v in parent_keep.tolist()],
    }
    return output


# --------------------------------------------------------------------------
# family POD basis, rebuilt on the exact v1 path
# --------------------------------------------------------------------------


def build_family_basis(
    model: Any,
    normalizer: Any,
    manifest: Mapping[str, Any],
    records: Sequence[Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Raw residual temporal POD over the two frozen v1 basis records."""
    covariance = torch.zeros((TIME_COUNT, TIME_COUNT), dtype=torch.float32, device=device)
    rows = []
    for record in records:
        loaded = parent_runtime.load_record_input(record)
        predicted = legacy_pod.generate_parent_full401(
            model, normalizer, loaded, manifest, device=device, time_block=TIME_BLOCK
        )
        truth, truth_hash = load_authorized_train_truth(record, device=device)
        residual = (truth - predicted).contiguous()
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError("basis residual contains non-finite values")
        legacy_pod.stream_accumulate_temporal_covariance(covariance, residual)
        rows.append(
            {
                **asdict(record),
                "nontruth_input_sha256": loaded.nontruth_input_sha256,
                "train_truth_sha256": truth_hash,
                "residual_energy": float(residual.double().square().sum().item()),
            }
        )
        del predicted, truth, residual
        torch.cuda.empty_cache()
    eigenvalues, eigenvectors = legacy_pod.temporal_pod(covariance)
    total = float(eigenvalues.sum().item())
    if not math.isfinite(total) or total <= 0.0:
        raise ProbeContractError("family covariance energy is invalid")
    signed = v1prep._fix_eigenvector_signs(eigenvectors)
    meta = {
        "records": rows,
        "covariance_trace": float(covariance.diagonal().double().sum().item()),
        "covariance_dtype": "float32",
        "eigendecomposition_dtype": "float64",
        "energy_capture": {
            str(rank): float(eigenvalues[:rank].sum().item() / total) for rank in RANKS
        },
    }
    del covariance
    torch.cuda.empty_cache()
    return eigenvalues, signed, meta


def cross_check_frozen_basis(signed_basis: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Compare the rebuilt rank-16 basis against the frozen v1 artifact."""
    payload = torch.load(V1_BASIS_PATH, map_location="cpu")
    order = tuple(str(v) for v in payload["family_order"])
    frozen = payload["basis"]
    output: dict[str, Any] = {
        "path": str(V1_BASIS_PATH),
        "sha256": parent_runtime.sha256_file(V1_BASIS_PATH),
        "family_order": list(order),
        "families": {},
    }
    for index, family in enumerate(order):
        if family not in signed_basis:
            continue
        rebuilt = torch.as_tensor(signed_basis[family], dtype=torch.float64)[:, :16]
        reference = frozen[index].double()
        deviation = float((rebuilt - reference).abs().max().item())
        # subspace agreement is sign- and rotation-robust; report both
        gram = rebuilt.T @ reference
        principal = float(
            torch.linalg.svdvals(gram).min().item()
        )
        output["families"][family] = {
            "max_abs_column_deviation": deviation,
            "min_principal_cosine": principal,
        }
    return output


# --------------------------------------------------------------------------
# stage 1: self-consistency in the old unmasked convention
# --------------------------------------------------------------------------


@torch.inference_mode()
def run_self_consistency(
    model: Any,
    normalizer: Any,
    manifest: Mapping[str, Any],
    basis_by_family: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Reproduce the two published unmasked-regime numbers on their own records."""
    time_axis = torch.tensor(manifest["time_s"], dtype=torch.float64)
    results: dict[str, Any] = {}
    for key, spec in SELF_CHECK.items():
        record = record_from_manifest(manifest, str(spec["sample_id"]))
        if record.family != str(spec["family"]):
            raise ProbeContractError(f"self-check family drift: {key}")
        loaded = parent_runtime.load_record_input(record)
        predicted = legacy_pod.generate_parent_full401(
            model, normalizer, loaded, manifest, device=device, time_block=TIME_BLOCK
        )
        truth, truth_hash = load_authorized_train_truth(record, device=device)
        observed = onset_indices(
            time_axis,
            t0_s=float(loaded.source_parameters[3]),
            f0_hz=float(loaded.source_parameters[2]),
        )
        future_start = int(observed[1] + 1)
        basis = basis_by_family[record.family]
        if key.startswith("uniform"):
            metrics = legacy_pod.oracle_projection_metrics(
                predicted, truth, basis, future_start=future_start, ranks=(16,)
            )
            reproduced = float(metrics["ranks"]["16"]["relative_error_reduction"])
            extra = {
                "parent_mean_per_frame_relative_l2": float(
                    metrics["parent_mean_per_frame_relative_l2"]
                ),
                "corrected_mean_per_frame_relative_l2": float(
                    metrics["ranks"]["16"]["corrected_mean_per_frame_relative_l2"]
                ),
                "residual_energy_capture": float(
                    metrics["ranks"]["16"]["residual_energy_capture"]
                ),
            }
        else:
            shared = factorial.prepare_future_projection(
                predicted, truth, future_start=future_start
            )
            metrics = factorial.fit_projection_metrics(
                shared, basis, fit_mode="weighted", ranks=(16,)
            )
            reproduced = float(metrics["ranks"]["16"]["raw_residual_energy_capture"])
            extra = {
                "parent_metrics": dict(metrics["parent_metrics"]),
                "corrected_metrics": dict(metrics["ranks"]["16"]["corrected_metrics"]),
                "mean_per_frame_relative_l2_reduction": float(
                    metrics["ranks"]["16"]["paired_changes_vs_parent"][
                        "mean_per_frame_relative_l2_reduction"
                    ]
                ),
            }
        published = float(spec["published_value"])
        tolerance = float(spec["absolute_tolerance"])
        difference = reproduced - published
        results[key] = {
            **{name: spec[name] for name in ("sample_id", "family", "published_file", "published_pointer", "estimator")},
            "group_id": record.group_id,
            "source_index": int(record.source_index),
            "train_truth_sha256": truth_hash,
            "observed_indices": [int(v) for v in observed],
            "future_start_index": future_start,
            "future_frame_count": int(TIME_COUNT - future_start),
            "published_value": published,
            "reproduced_value": reproduced,
            "absolute_difference": difference,
            "absolute_tolerance": tolerance,
            "matched": bool(abs(difference) <= tolerance),
            "supporting": extra,
        }
        del predicted, truth
        torch.cuda.empty_cache()
    results["tolerance_rationale"] = SELF_CHECK_TOLERANCE_RATIONALE
    results["all_matched"] = all(
        bool(value["matched"])
        for value in results.values()
        if isinstance(value, dict) and "matched" in value
    )
    return results


# --------------------------------------------------------------------------
# decision rule
# --------------------------------------------------------------------------


FIT_ARMS = ("published_weighted", "masked_matched")


def _arm_gain(rank_item: Mapping[str, Any], arm: str) -> float:
    """Convention (ii) gain for one fit arm."""
    if arm == "published_weighted":
        return float(rank_item["gain"]["masked_energy_floored_per_frame"])
    if arm == "masked_matched":
        return float(
            rank_item["masked_matched_fit"]["gain"]["masked_energy_floored_per_frame"]
        )
    raise ValueError(f"unknown fit arm: {arm}")


def apply_decision_rule(
    panel: Mapping[str, Any], *, arm: str = "published_weighted"
) -> dict[str, Any]:
    """The pre-stated rule from the frozen spec.  Not a promotion gate.

    ``arm`` selects which oracle the convention (ii) gain is read from.  The
    thresholds are the frozen ones in both cases; nothing is tuned per arm.
    """
    per_family: dict[str, Any] = {}
    for family, item in panel.items():
        ranks = item["ranks"]
        gain16 = _arm_gain(ranks["16"], arm)
        gain32 = _arm_gain(ranks["32"], arm)
        gain8 = _arm_gain(ranks["8"], arm)
        marginal = gain32 - gain16
        per_family[family] = {
            "sample_id": item["sample_id"],
            "masked_gain_rank8": gain8,
            "masked_gain_rank16": gain16,
            "masked_gain_rank32": gain32,
            "rank32_marginal_gain_over_rank16": marginal,
            "rank16_meets_minimum": bool(gain16 >= RANK16_MASKED_GAIN_MINIMUM),
            "rank32_marginal_below_maximum": bool(marginal < RANK32_MARGINAL_GAIN_MAXIMUM),
            "mask_agreement_rate": float(item["mask"]["agreement_rate"]),
            "mask_agreement_passes": bool(
                float(item["mask"]["agreement_rate"]) >= MASK_AGREEMENT_MINIMUM
            ),
        }
    all_rank16 = all(bool(v["rank16_meets_minimum"]) for v in per_family.values())
    all_marginal = all(bool(v["rank32_marginal_below_maximum"]) for v in per_family.values())
    all_masks = all(bool(v["mask_agreement_passes"]) for v in per_family.values())
    if all_rank16 and all_marginal:
        rank_verdict = "rank_is_not_the_bottleneck_v15_may_proceed_at_rank16_under_convention_ii"
    elif all_rank16:
        rank_verdict = "rank16_clears_the_minimum_everywhere_but_rank32_still_adds_material_gain"
    else:
        rank_verdict = "rank16_misses_the_minimum_for_at_least_one_family_report_per_family_achievable_gain"
    return {
        "fit_arm": str(arm),
        "thresholds": {
            "rank16_masked_gain_minimum": RANK16_MASKED_GAIN_MINIMUM,
            "rank32_marginal_gain_maximum": RANK32_MARGINAL_GAIN_MAXIMUM,
            "mask_agreement_minimum": MASK_AGREEMENT_MINIMUM,
        },
        "families": per_family,
        "all_families_rank16_meet_minimum": all_rank16,
        "all_families_rank32_marginal_below_maximum": all_marginal,
        "all_families_mask_agreement_pass": all_masks,
        "rank_verdict": rank_verdict,
        "mask_verdict": (
            "parent_energy_mask_is_a_valid_deployment_causal_proxy"
            if all_masks
            else "parent_energy_mask_is_not_a_valid_proxy_masking_design_must_change_before_v15"
        ),
        "uniform_single_threshold_permitted": False,
        "note": "per-family achievable gain is reported; a single uniform 0.5 threshold is not set",
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
    spec_binding = verify_spec()
    v14 = verify_v14_bindings()
    disk_before = parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    checkpoint_before = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    if checkpoint_before != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError("parent checkpoint hash mismatch before probe")
    gpu = parent_runtime.gpu_identity(physical_gpu_index, device)

    model, normalizer, manifest, run_identity = parent_runtime.load_model_context(device)
    time_axis = torch.tensor(manifest["time_s"], dtype=torch.float64)
    basis_records = v1prep.basis_records(manifest)
    torch.cuda.reset_peak_memory_stats(device)

    basis_by_family: dict[str, torch.Tensor] = {}
    basis_meta: dict[str, Any] = {}
    for family in FAMILIES:
        eigenvalues, signed, meta = build_family_basis(
            model, normalizer, manifest, basis_records[family], device=device
        )
        basis_by_family[family] = signed
        basis_meta[family] = meta
        del eigenvalues

    basis_cross_check = cross_check_frozen_basis(basis_by_family)

    self_consistency = run_self_consistency(
        model, normalizer, manifest, basis_by_family, device=device
    )
    if not bool(self_consistency["all_matched"]):
        raise ProbeContractError(
            "self-consistency reproduction failed; refusing to report masked results: "
            + json.dumps(
                {
                    key: {
                        "published": value["published_value"],
                        "reproduced": value["reproduced_value"],
                        "difference": value["absolute_difference"],
                        "tolerance": value["absolute_tolerance"],
                    }
                    for key, value in self_consistency.items()
                    if isinstance(value, dict) and "matched" in value
                },
                sort_keys=True,
            )
        )

    panel: dict[str, Any] = {}
    for family, sample_id in PANEL_IDS:
        record = record_from_manifest(manifest, sample_id)
        if record.family != family:
            raise ProbeContractError(f"panel family drift: {sample_id}")
        loaded = parent_runtime.load_record_input(record)
        predicted = legacy_pod.generate_parent_full401(
            model, normalizer, loaded, manifest, device=device, time_block=TIME_BLOCK
        )
        truth, truth_hash = load_authorized_train_truth(record, device=device)
        observed = onset_indices(
            time_axis,
            t0_s=float(loaded.source_parameters[3]),
            f0_hz=float(loaded.source_parameters[2]),
        )
        scored = masked_oracle_record(
            predicted, truth, basis_by_family[family], k1=int(observed[1]), ranks=RANKS
        )
        # cross-check convention (i) against the published estimator on this record
        shared = factorial.prepare_future_projection(
            predicted, truth, future_start=int(observed[1] + 1)
        )
        reference = factorial.fit_projection_metrics(
            shared, basis_by_family[family], fit_mode="weighted", ranks=RANKS
        )
        cross = {}
        for rank in RANKS:
            mine = float(scored["ranks"][str(rank)]["gain"]["unmasked_per_frame_unsquared_mean"])
            theirs = float(
                reference["ranks"][str(rank)]["paired_changes_vs_parent"][
                    "mean_per_frame_relative_l2_reduction"
                ]
            )
            cross[str(rank)] = {
                "this_probe_convention_i_gain": mine,
                "published_estimator_gain": theirs,
                "absolute_difference": mine - theirs,
                "agrees": bool(abs(mine - theirs) <= 1.0e-9),
            }
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
            "convention_i_cross_check_vs_published_estimator": cross,
            "v14_transplanted_oracle_constant": V14_TRANSPLANTED_ORACLE[family],
        }
        del predicted, truth, shared
        torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    elapsed = float(time.monotonic() - started)
    if elapsed > GPU_SECONDS_MAXIMUM:
        raise parent_runtime.BudgetExceededError("probe exceeded its GPU-second budget")
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if peak_reserved > PEAK_CUDA_BYTES_MAXIMUM:
        raise ProbeContractError("peak CUDA reserved memory exceeds the probe gate")

    decision = {
        "note": "the same frozen thresholds are applied to both fit arms; nothing is tuned per arm",
        "published_weighted_arm_is_convention_i_matched": (
            "the published estimator weights every frame by 1/||truth_t||^2, which is the "
            "convention (i) objective, so it is not an upper bound under convention (ii); "
            "the masked_matched arm is the convention (ii) upper bound"
        ),
        "by_fit_arm": {
            arm: apply_decision_rule(panel, arm=arm) for arm in FIT_ARMS
        },
    }

    # split the long per-frame arrays out of the terminal
    profiles = {}
    for family in panel:
        profiles[family] = {
            "sample_id": panel[family]["sample_id"],
            "future_start_index": panel[family]["future_start_index"],
            **panel[family].pop("profiles"),
        }

    checkpoint_after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    if checkpoint_after != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError("parent checkpoint changed during the probe")
    v14_after = verify_v14_bindings()

    payload = {
        "schema": "r4e8_masked_oracle_probe_result_v1",
        "candidate": CANDIDATE,
        "status": "success",
        "kind": "diagnostic_probe_not_a_promotion_candidate",
        "started_utc": utc_now(),
        "completed_utc": utc_now(),
        "exact_blocker": "none",
        "claim_scope": "offline_train_only_oracle_upper_bound_under_three_metric_conventions_not_online_adaptation_not_validation_not_test_id",
        "spec": spec_binding,
        "why": "v14 oracle_gain compared records 00321/00564/00385 against constants measured on 00102/01032/synthetic; the threshold was transplanted and the gate was undecidable",
        "sealed_data_attestation": {
            "train_truth_opened_for_offline_oracle": True,
            "validation_opened": False,
            "test_id_opened": False,
            "field_arrays_written_to_disk": False,
            "authorized_train_sample_ids": sorted(ALLOWED_TRUTH_SAMPLE_IDS),
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
            "coefficient_fit": "weighted least squares on frames strictly after the second registered onset (C1 causal start), pinv of the weight-rooted design, identical in form to the published r4e7 weighted arm",
            "conventions": {
                "i_unmasked_per_frame_unsquared_mean": "mean_t sqrt(||err_t||^2 / max(||truth_t||^2, 1e-30)); the convention the v14 gate used",
                "ii_masked_energy_floored_per_frame": "mean over frames with ||truth_t||^2 >= tau*max_s||truth_s||^2 of sqrt(||err_t||^2 / max(||truth_t||^2, tau*max_s||truth_s||^2)); proposed v15 convention",
                "iii_global_energy_rel_l2": "sqrt(sum_t ||err_t||^2 / sum_t ||truth_t||^2)",
            },
            "gain": "(parent_metric - corrected_metric) / max(parent_metric, 1e-30) per convention",
            "mask_causality": "the parent mask uses only parent frame energy, which is available at deployment; the truth mask is the oracle reference",
            "basis": "per-family raw residual temporal POD over the two frozen v1 basis records, nested truncation to ranks 8/16/32, v1 sign rule",
        },
        "basis": basis_meta,
        "basis_cross_check_vs_frozen_v1_rank16": basis_cross_check,
        "self_consistency": self_consistency,
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
                "schema": "r4e8_masked_oracle_probe_energy_profiles_v1",
                "candidate": CANDIDATE,
                "tau": TAU,
                "families": profiles,
            },
            assert_write_allowed(PROFILE_PATH),
            limit=MAX_OUTPUT_BYTES,
        )
        payload["energy_profiles_path"] = str(PROFILE_PATH)
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
    spec_binding = verify_spec()
    verify_v14_bindings()
    parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    before = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    if before != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError("parent checkpoint hash mismatch before smoke")
    model, normalizer, manifest, _ = parent_runtime.load_model_context(device)
    torch.cuda.reset_peak_memory_stats(device)
    records = v1prep.basis_records(manifest)["layered"]
    _eigenvalues, signed, meta = build_family_basis(
        model, normalizer, manifest, records, device=device
    )
    record = record_from_manifest(manifest, "train_layered_00564")
    loaded = parent_runtime.load_record_input(record)
    predicted = legacy_pod.generate_parent_full401(
        model, normalizer, loaded, manifest, device=device, time_block=TIME_BLOCK
    )
    truth, truth_hash = load_authorized_train_truth(record, device=device)
    observed = onset_indices(
        torch.tensor(manifest["time_s"], dtype=torch.float64),
        t0_s=float(loaded.source_parameters[3]),
        f0_hz=float(loaded.source_parameters[2]),
    )
    scored = masked_oracle_record(
        predicted, truth, signed, k1=int(observed[1]), ranks=(16,)
    )
    scored.pop("profiles")
    torch.cuda.synchronize(device)
    after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    return {
        "status": "passed",
        "unscored": True,
        "utc": utc_now(),
        "spec": spec_binding,
        "sample_id": record.sample_id,
        "train_truth_sha256": truth_hash,
        "layered_basis_covariance_trace": meta["covariance_trace"],
        "observed_indices": [int(v) for v in observed],
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
    parser.add_argument("--mode", choices=("smoke", "selfcheck", "measured"), required=True)
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
    "CANDIDATE",
    "EPS_SQUARED",
    "MASK_AGREEMENT_MINIMUM",
    "PANEL_IDS",
    "RANK16_MASKED_GAIN_MINIMUM",
    "RANK32_MARGINAL_GAIN_MAXIMUM",
    "RANKS",
    "SELF_CHECK",
    "SPEC_SHA256",
    "TAU",
    "ProbeContractError",
    "TruthScopeError",
    "apply_decision_rule",
    "assert_write_allowed",
    "build_family_basis",
    "convention_gains",
    "convention_metrics",
    "energy_mask",
    "frame_energy",
    "mask_agreement",
    "masked_oracle_record",
    "record_from_manifest",
    "verify_spec",
    "verify_v14_bindings",
    "weighted_coefficient_map",
]
