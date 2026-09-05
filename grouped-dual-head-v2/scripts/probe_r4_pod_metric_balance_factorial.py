#!/usr/bin/env python3
"""Train-only oracle calibration of POD covariance and coefficient metrics.

The nine records are deliberately reused from the historically exposed
``r4e7_family_temporal_pod_capacity_train9_v1`` panel.  Consequently this is a
calibration-only factorial, never an independent confirmation.  Parent, truth,
residual, covariance inputs, coefficients, corrections, and corrected fields
remain in memory; only the preregistration and terminal scalar report may be
serialized as JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
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
from scripts import probe_r4_family_temporal_pod_capacity as legacy_probe


CANDIDATE = "r4e7_pod_metric_balance_factorial_calibration9_v1"
SCRIPT_PATH = Path(__file__).resolve()
TEST_PATH = (
    PROJECT_ROOT
    / "tests/saved_time_phase_operator_v4/test_r4_pod_metric_balance_factorial.py"
)
PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "results/r4e7_pod_metric_balance_factorial_calibration9_v1_preregistration_20260825.json"
)
RESULT_PATH = (
    PROJECT_ROOT
    / "results/r4e7_pod_metric_balance_factorial_calibration9_v1_20260825.json"
)
HISTORICAL_POD_PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "results/r4e7_family_temporal_pod_capacity_train9_v1_preregistration_20260825.json"
)
HISTORICAL_POD_RESULT_PATH = (
    PROJECT_ROOT / "results/r4e7_family_temporal_pod_capacity_train9_v1_20260825.json"
)
HISTORICAL_RUNTIME_PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "results/r4e7_parent_e2e_runtime_train9_v1_preregistration_20260825.json"
)
HISTORICAL_RUNTIME_RESULT_PATH = (
    PROJECT_ROOT / "results/r4e7_parent_e2e_runtime_train9_v1_20260825.json"
)

FAMILIES = ("uniform", "layered", "marmousi")
RANKS = (8, 16, 32)
TIME_COUNT = 401
TIME_BLOCK = 16
SPATIAL_CHUNK = 4096
WEIGHT_EPS_SQUARED = 1.0e-30
GPU_SECONDS_MAXIMUM = 360.0
MIN_FREE_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 10 * 1024**2
PEAK_CUDA_GIB_MAXIMUM = 23.5
PEAK_CUDA_BYTES_MAXIMUM = int(PEAK_CUDA_GIB_MAXIMUM * 1024**3)
RANK32_REDUCTION_MINIMUM = 0.20

COVARIANCE_MODES = ("raw", "equaltrace")
FIT_MODES = ("ordinary", "weighted")
ARM_KEYS = tuple(
    f"{covariance}+{fit}"
    for covariance in COVARIANCE_MODES
    for fit in FIT_MODES
)
TIE_ORDER = (
    "raw+weighted",
    "equaltrace+weighted",
    "equaltrace+ordinary",
    "raw+ordinary",
)
EXPECTED_SAMPLE_IDS = (
    "train_uniform_00000",
    "train_uniform_00001",
    "train_uniform_00002",
    "train_layered_00000",
    "train_layered_00004",
    "train_layered_00008",
    "train_marmousi_00000",
    "train_marmousi_00005",
    "train_marmousi_00010",
)
EXPECTED_GROUP_IDS = (
    "train:uniform:00000",
    "train:uniform:00001",
    "train:uniform:00002",
    "train:layered:00000",
    "train:layered:00001",
    "train:layered:00002",
    "train:marmousi:x0.0:z0.0",
    "train:marmousi:x150.0:z0.0",
    "train:marmousi:x300.0:z0.0",
)

CODE_BINDING_PATHS = tuple(
    dict.fromkeys(
        (
            SCRIPT_PATH,
            TEST_PATH,
            Path(legacy_probe.__file__).resolve(),
            *legacy_probe.CODE_BINDING_PATHS,
        )
    )
)


class FactorialContractError(RuntimeError):
    """The frozen factorial calibration contract is invalid."""


TruthScopeError = legacy_probe.TruthScopeError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _binding_map(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {str(path.resolve()): parent_runtime.file_binding(path) for path in paths}


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def covariance_pair_update(
    raw_sum: torch.Tensor,
    equal_trace_sum: torch.Tensor,
    residual: torch.Tensor,
    *,
    spatial_chunk: int = SPATIAL_CHUNK,
) -> float:
    """Update raw-sum and per-record equal-trace covariances from one residual."""
    if raw_sum.shape != equal_trace_sum.shape:
        raise ValueError("covariance arm shapes disagree")
    record_covariance = torch.zeros_like(raw_sum)
    legacy_probe.stream_accumulate_temporal_covariance(
        record_covariance, residual, spatial_chunk=int(spatial_chunk)
    )
    trace = float(record_covariance.diagonal().double().sum().item())
    if not math.isfinite(trace) or trace <= 0.0:
        raise FactorialContractError("per-record residual covariance trace is invalid")
    raw_sum.add_(record_covariance)
    equal_trace_sum.add_(record_covariance / trace)
    return trace


def temporal_pod(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return legacy_probe.temporal_pod(covariance)


def condition_number(design: torch.Tensor) -> float:
    value = torch.as_tensor(design, device="cpu", dtype=torch.float64)
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError("condition-number input must be a nonempty matrix")
    singular = torch.linalg.svdvals(value)
    smallest = float(singular[-1].item())
    largest = float(singular[0].item())
    if not math.isfinite(smallest) or not math.isfinite(largest) or smallest <= 0.0:
        return float("inf")
    return largest / smallest


def fit_coefficients(
    design: torch.Tensor,
    residual: torch.Tensor,
    *,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    """Fit coefficients by ordinary or frame-metric-weighted least squares."""
    matrix = torch.as_tensor(design, dtype=torch.float64)
    target = torch.as_tensor(residual, device=matrix.device, dtype=torch.float64)
    if matrix.ndim != 2 or target.ndim != 2 or matrix.shape[0] != target.shape[0]:
        raise ValueError("design and residual must be [time, rank] and [time, point]")
    if weights is None:
        weighted_design = matrix
        weighted_target = target
    else:
        weight = torch.as_tensor(weights, device=matrix.device, dtype=torch.float64)
        if weight.shape != (matrix.shape[0],) or not bool((weight > 0).all()):
            raise ValueError("weights must be finite positive per-frame scalars")
        root = weight.sqrt()[:, None]
        weighted_design = root * matrix
        weighted_target = root * target
    return torch.linalg.pinv(weighted_design, rtol=1.0e-12) @ weighted_target


def mean_frame_metrics_from_squared_errors(
    error_squared_by_frame: torch.Tensor,
    truth_squared_by_frame: torch.Tensor,
) -> dict[str, float]:
    error = torch.as_tensor(error_squared_by_frame, dtype=torch.float64)
    truth = torch.as_tensor(truth_squared_by_frame, device=error.device, dtype=torch.float64)
    ratios_squared = error / truth.clamp_min(WEIGHT_EPS_SQUARED)
    if not bool(torch.isfinite(ratios_squared).all()):
        raise FloatingPointError("per-frame relative metrics are non-finite")
    return {
        "mean_per_frame_relative_l2": float(ratios_squared.sqrt().mean().item()),
        "mean_squared_per_frame_relative_l2": float(ratios_squared.mean().item()),
    }


@torch.inference_mode()
def prepare_future_projection(
    parent: torch.Tensor,
    truth: torch.Tensor,
    *,
    future_start: int,
) -> dict[str, Any]:
    """Construct one future parent/truth/residual context shared by all arms."""
    predicted = torch.as_tensor(parent)
    target = torch.as_tensor(truth, device=predicted.device)
    if predicted.shape != target.shape:
        raise ValueError("parent and truth shapes disagree")
    start = int(future_start)
    if not 0 < start < predicted.shape[0]:
        raise ValueError("future start must lie strictly inside the time axis")
    future_parent = predicted[start:].reshape(predicted.shape[0] - start, -1)
    future_truth = target[start:].reshape(target.shape[0] - start, -1)
    future_residual = future_truth - future_parent
    truth_squared = future_truth.double().square().sum(dim=1)
    parent_error_squared = future_residual.double().square().sum(dim=1)
    residual_energy = float(parent_error_squared.sum().item())
    if residual_energy <= 0.0 or not math.isfinite(residual_energy):
        raise FactorialContractError("calibration future residual energy is invalid")
    return {
        "future_parent": future_parent,
        "future_truth": future_truth,
        "future_residual": future_residual,
        "truth_squared": truth_squared,
        "weights": truth_squared.clamp_min(WEIGHT_EPS_SQUARED).reciprocal(),
        "parent_metrics": mean_frame_metrics_from_squared_errors(
            parent_error_squared, truth_squared
        ),
        "residual_energy": residual_energy,
        "parent_energy": float(future_parent.double().square().sum().item()),
    }


@torch.inference_mode()
def fit_projection_metrics(
    shared_future: Mapping[str, Any],
    temporal_basis: torch.Tensor,
    *,
    fit_mode: str,
    ranks: Sequence[int] = RANKS,
    spatial_chunk: int = SPATIAL_CHUNK,
) -> dict[str, Any]:
    """Fit and score one covariance/fit arm without materializing corrections."""
    future_parent = torch.as_tensor(shared_future["future_parent"])
    future_truth = torch.as_tensor(
        shared_future["future_truth"], device=future_parent.device
    )
    future_residual = torch.as_tensor(
        shared_future["future_residual"], device=future_parent.device
    )
    if future_parent.shape != future_truth.shape or future_parent.shape != future_residual.shape:
        raise ValueError("shared future parent, truth, and residual shapes disagree")
    if fit_mode not in FIT_MODES:
        raise ValueError(f"unknown coefficient fit: {fit_mode}")
    truth_squared = torch.as_tensor(
        shared_future["truth_squared"], device=future_parent.device, dtype=torch.float64
    )
    weights = torch.as_tensor(
        shared_future["weights"], device=future_parent.device, dtype=torch.float64
    )
    parent_metrics = dict(shared_future["parent_metrics"])
    residual_energy = float(shared_future["residual_energy"])
    parent_energy = float(shared_future["parent_energy"])
    future_start = int(temporal_basis.shape[0] - future_parent.shape[0])
    if future_start <= 0:
        raise ValueError("shared future context and temporal basis disagree")

    basis_cpu = torch.as_tensor(temporal_basis, device="cpu", dtype=torch.float64)
    output: dict[str, Any] = {
        "future_frame_count": int(future_parent.shape[0]),
        "parent_metrics": parent_metrics,
        "future_residual_energy": residual_energy,
        "future_parent_energy": parent_energy,
        "weight_formula": "w_t=1/max(||truth_t||_2^2,1e-30)",
        "coefficient_fit": fit_mode,
        "ranks": {},
    }
    for rank_value in ranks:
        rank = int(rank_value)
        if rank <= 0 or rank > basis_cpu.shape[1]:
            raise ValueError("POD rank is outside the available basis")
        design = basis_cpu[future_start:, :rank]
        root_weight = weights.detach().cpu().sqrt()[:, None]
        ordinary_condition = condition_number(design)
        weighted_condition = condition_number(root_weight * design)
        chosen_weights = weights.detach().cpu() if fit_mode == "weighted" else None
        if fit_mode == "weighted":
            solver_design = root_weight * design
            coefficient_map = torch.linalg.pinv(solver_design, rtol=1.0e-12)
            coefficient_map = coefficient_map * root_weight[:, 0][None, :]
        else:
            coefficient_map = torch.linalg.pinv(design, rtol=1.0e-12)
        del chosen_weights
        coefficient_map_device = coefficient_map.to(device=future_parent.device)
        design_device = design.to(device=future_parent.device)
        corrected_error_squared = torch.zeros(
            future_parent.shape[0], dtype=torch.float64, device=future_parent.device
        )
        correction_energy = 0.0
        unexplained_energy = 0.0
        for spatial_start in range(0, future_parent.shape[1], int(spatial_chunk)):
            spatial_stop = min(spatial_start + int(spatial_chunk), future_parent.shape[1])
            residual_block = future_residual[:, spatial_start:spatial_stop].double()
            coefficients = coefficient_map_device @ residual_block
            correction = design_device @ coefficients
            remaining = residual_block - correction
            remaining_squared = remaining.square()
            corrected_error_squared += remaining_squared.sum(dim=1)
            unexplained_energy += float(remaining_squared.sum().item())
            correction_energy += float(correction.square().sum().item())
        corrected_metrics = mean_frame_metrics_from_squared_errors(
            corrected_error_squared, truth_squared
        )
        parent_unsquared = parent_metrics["mean_per_frame_relative_l2"]
        corrected_unsquared = corrected_metrics["mean_per_frame_relative_l2"]
        parent_squared = parent_metrics["mean_squared_per_frame_relative_l2"]
        corrected_squared = corrected_metrics["mean_squared_per_frame_relative_l2"]
        item = {
            "corrected_metrics": corrected_metrics,
            "raw_residual_energy_capture": 1.0 - unexplained_energy / residual_energy,
            "unexplained_raw_residual_energy": unexplained_energy,
            "correction_energy": correction_energy,
            "correction_energy_ratio_to_parent": correction_energy
            / max(parent_energy, WEIGHT_EPS_SQUARED),
            "basis_design_condition_number": ordinary_condition,
            "weighted_design_condition_number": weighted_condition,
            "paired_changes_vs_parent": {
                "mean_per_frame_relative_l2_absolute": corrected_unsquared
                - parent_unsquared,
                "mean_per_frame_relative_l2_reduction": (
                    parent_unsquared - corrected_unsquared
                )
                / max(parent_unsquared, WEIGHT_EPS_SQUARED),
                "mean_squared_per_frame_relative_l2_absolute": corrected_squared
                - parent_squared,
                "mean_squared_per_frame_relative_l2_reduction": (
                    parent_squared - corrected_squared
                )
                / max(parent_squared, WEIGHT_EPS_SQUARED),
            },
            "finite": True,
        }
        item["finite"] = _all_finite(item)
        output["ranks"][str(rank)] = item
    return output


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        "corrected_mean_per_frame_relative_l2": float(
            left["corrected_metrics"]["mean_per_frame_relative_l2"]
            - right["corrected_metrics"]["mean_per_frame_relative_l2"]
        ),
        "corrected_mean_squared_per_frame_relative_l2": float(
            left["corrected_metrics"]["mean_squared_per_frame_relative_l2"]
            - right["corrected_metrics"]["mean_squared_per_frame_relative_l2"]
        ),
        "unsquared_reduction": float(
            left["paired_changes_vs_parent"]["mean_per_frame_relative_l2_reduction"]
            - right["paired_changes_vs_parent"]["mean_per_frame_relative_l2_reduction"]
        ),
        "raw_residual_energy_capture": float(
            left["raw_residual_energy_capture"]
            - right["raw_residual_energy_capture"]
        ),
        "correction_energy_ratio_to_parent": float(
            left["correction_energy_ratio_to_parent"]
            - right["correction_energy_ratio_to_parent"]
        ),
    }


def factorial_paired_changes(arms: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for rank in RANKS:
        key = str(rank)
        output[key] = {
            "weighted_minus_ordinary": {
                covariance: _metric_delta(
                    arms[f"{covariance}+weighted"]["ranks"][key],
                    arms[f"{covariance}+ordinary"]["ranks"][key],
                )
                for covariance in COVARIANCE_MODES
            },
            "equaltrace_minus_raw": {
                fit: _metric_delta(
                    arms[f"equaltrace+{fit}"]["ranks"][key],
                    arms[f"raw+{fit}"]["ranks"][key],
                )
                for fit in FIT_MODES
            },
        }
    return output


def _historical_payload() -> dict[str, Any]:
    return json.loads(HISTORICAL_POD_PREREGISTRATION_PATH.read_text(encoding="utf8"))


def sample_census(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind the exact prior train9 panel, relabeling third records as calibration."""
    current = legacy_probe.sample_census(manifest)
    historical = _historical_payload()["sample_selection"]["records"]
    if len(current) != 9 or len(historical) != 9:
        raise FactorialContractError("historical train9 census length changed")
    output = []
    for current_row, historical_row in zip(current, historical, strict=True):
        comparable = dict(current_row)
        if comparable != historical_row:
            raise parent_runtime.BindingDriftError(
                f"historical train9 binding drift: {current_row['sample_id']}"
            )
        position = int(current_row["family_position"])
        row = dict(current_row)
        row["historical_role"] = str(row.pop("role"))
        row["role"] = "basis" if position < 2 else "exposed_calibration"
        row["independent_confirmation"] = False
        output.append(row)
    if tuple(row["sample_id"] for row in output) != EXPECTED_SAMPLE_IDS:
        raise FactorialContractError("sample ids differ from the exact historical panel")
    if tuple(row["group_id"] for row in output) != EXPECTED_GROUP_IDS:
        raise FactorialContractError("group ids differ from the exact historical panel")
    if Counter(row["role"] for row in output) != Counter(
        {"basis": 6, "exposed_calibration": 3}
    ):
        raise FactorialContractError("calibration roles are invalid")
    return output


def _measured_command() -> str:
    return (
        "env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python "
        "scripts/probe_r4_pod_metric_balance_factorial.py --mode measured "
        "--physical-gpu-index 0 --device cuda:0 --time-block 16 "
        "--preregistration results/r4e7_pod_metric_balance_factorial_calibration9_v1_preregistration_20260825.json "
        "--result results/r4e7_pod_metric_balance_factorial_calibration9_v1_20260825.json"
    )


def run_smoke(
    *, physical_gpu_index: int, device: torch.device, time_block: int
) -> dict[str, Any]:
    disk_before = parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    gpu = parent_runtime.gpu_identity(physical_gpu_index, device)
    if parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH) != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError("parent checkpoint hash mismatch before smoke")
    if int(time_block) != TIME_BLOCK:
        raise FactorialContractError("smoke requires parent full401 time_block16")
    started = time.monotonic()
    model, normalizer, manifest, _ = parent_runtime.load_model_context(device)
    record = parent_runtime.select_train_records(manifest)[0]
    loaded = parent_runtime.load_record_input(record)
    torch.cuda.reset_peak_memory_stats(device)
    parent = legacy_probe.generate_parent_full401(
        model, normalizer, loaded, manifest, device=device, time_block=time_block
    )
    truth, truth_hash = legacy_probe.load_train_truth(record, device=device)
    residual = truth - parent
    raw_covariance = torch.zeros(
        (TIME_COUNT, TIME_COUNT), dtype=torch.float32, device=device
    )
    equal_covariance = torch.zeros_like(raw_covariance)
    record_trace = covariance_pair_update(raw_covariance, equal_covariance, residual)
    raw_eigenvalues, raw_basis = temporal_pod(raw_covariance)
    equal_eigenvalues, equal_basis = temporal_pod(equal_covariance)
    observed = onset_indices(
        torch.tensor(manifest["time_s"], dtype=torch.float64),
        t0_s=float(loaded.source_parameters[3]),
        f0_hz=float(loaded.source_parameters[2]),
    )
    shared_future = prepare_future_projection(
        parent, truth, future_start=int(observed[1] + 1)
    )
    arms = {
        "raw+ordinary": fit_projection_metrics(
            shared_future, raw_basis, fit_mode="ordinary", ranks=(8,)
        ),
        "raw+weighted": fit_projection_metrics(
            shared_future, raw_basis, fit_mode="weighted", ranks=(8,)
        ),
        "equaltrace+ordinary": fit_projection_metrics(
            shared_future, equal_basis, fit_mode="ordinary", ranks=(8,)
        ),
        "equaltrace+weighted": fit_projection_metrics(
            shared_future, equal_basis, fit_mode="weighted", ranks=(8,)
        ),
    }
    torch.cuda.synchronize(device)
    elapsed = float(time.monotonic() - started)
    if elapsed > GPU_SECONDS_MAXIMUM:
        raise parent_runtime.BudgetExceededError("smoke exceeded complete GPU budget")
    payload = {
        "status": "passed",
        "unscored": True,
        "utc": utc_now(),
        "record": asdict(record),
        "record_role": "historically_exposed_smoke_only",
        "nontruth_input_sha256": loaded.nontruth_input_sha256,
        "train_truth_sha256": truth_hash,
        "observed_indices": list(observed),
        "future_start_index": int(observed[1] + 1),
        "time_count": TIME_COUNT,
        "time_block": int(time_block),
        "parent_shape": list(parent.shape),
        "same_in_memory_parent_truth_residual_for_all_four_arms": True,
        "record_raw_covariance_trace": record_trace,
        "raw_covariance_trace": float(raw_covariance.diagonal().double().sum().item()),
        "equaltrace_covariance_trace": float(equal_covariance.diagonal().double().sum().item()),
        "raw_largest_eigenvalue": float(raw_eigenvalues[0].item()),
        "equaltrace_largest_eigenvalue": float(equal_eigenvalues[0].item()),
        "unscored_rank8_arm_finite": {
            key: _all_finite(value) for key, value in arms.items()
        },
        "gpu_seconds_used": elapsed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "disk_free_bytes_before": disk_before,
        "disk_free_bytes_after": parent_runtime.free_disk_bytes(),
        "gpu": gpu,
        "validation_opened": False,
        "test_id_opened": False,
        "arrays_serialized": False,
        "checkpoint_writes": 0,
        "script_sha256": parent_runtime.sha256_file(SCRIPT_PATH),
    }
    if not all(payload["unscored_rank8_arm_finite"].values()):
        raise FloatingPointError("one-record factorial smoke produced non-finite output")
    if payload["peak_cuda_reserved_bytes"] > PEAK_CUDA_BYTES_MAXIMUM:
        raise FactorialContractError("smoke peak CUDA reserved memory exceeds gate")
    return payload


def build_preregistration(
    *,
    physical_gpu_index: int,
    device: torch.device,
    time_block: int,
    smoke_evidence: Mapping[str, Any],
    focused_test_command: str,
    focused_test_result: str,
) -> dict[str, Any]:
    free = parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    if int(physical_gpu_index) != 0 or device != torch.device("cuda:0"):
        raise FactorialContractError("frozen measured run requires physical GPU 0 as cuda:0")
    if int(time_block) != TIME_BLOCK:
        raise FactorialContractError("frozen measured run requires full401 time_block16")
    if (
        smoke_evidence.get("status") != "passed"
        or smoke_evidence.get("unscored") is not True
        or smoke_evidence.get("time_block") != TIME_BLOCK
        or smoke_evidence.get("same_in_memory_parent_truth_residual_for_all_four_arms")
        is not True
        or smoke_evidence.get("arrays_serialized") is not False
        or smoke_evidence.get("script_sha256")
        != parent_runtime.sha256_file(SCRIPT_PATH)
    ):
        raise FactorialContractError("one-record full401 smoke evidence is invalid or stale")
    if not focused_test_command or not focused_test_result:
        raise FactorialContractError("focused test evidence is required")
    manifest = parent_runtime.load_manifest_payload()
    run_identity = json.loads(parent_runtime.RUN_IDENTITY_PATH.read_text(encoding="utf8"))
    checkpoint = parent_runtime.file_binding(parent_runtime.CHECKPOINT_PATH)
    if checkpoint["sha256"] != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError("parent checkpoint hash mismatch")
    census = sample_census(manifest)
    gpu = parent_runtime.gpu_identity(physical_gpu_index, device)
    historical_artifacts = _binding_map(
        (
            HISTORICAL_POD_PREREGISTRATION_PATH,
            HISTORICAL_POD_RESULT_PATH,
            HISTORICAL_RUNTIME_PREREGISTRATION_PATH,
            HISTORICAL_RUNTIME_RESULT_PATH,
        )
    )
    payload = {
        "schema": "r4_pod_metric_balance_factorial_preregistration_v1",
        "candidate": CANDIDATE,
        "status": "frozen",
        "created_utc": utc_now(),
        "hypothesis": "At rank 32 on the already exposed train calibration panel, equalizing each basis record's covariance trace and/or fitting coefficients in the exact squared frame-relative metric will improve the worst-family unsquared mean-per-frame relative-L2 reduction enough for every family to reach at least 20% without worsening any family.",
        "mechanism": "A controlled 2x2 factorial separates basis-record energy dominance (raw covariance sum versus sum of C_i/trace(C_i)) from coefficient-objective mismatch (ordinary least squares versus truth-energy-weighted least squares).",
        "claim_scope": "offline_train_future_truth_oracle_calibration_only_not_independent_confirmation_not_online_adaptation_not_validation_not_test_id",
        "acceptance": {
            "primary_rank": 32,
            "each_family_unsquared_mean_per_frame_relative_l2_reduction_minimum": RANK32_REDUCTION_MINIMUM,
            "every_family_nonworse_required": True,
            "finite_rank32_required": True,
            "raw_residual_energy_capture_is_diagnostic_only": True,
            "peak_cuda_reserved_bytes_maximum": PEAK_CUDA_BYTES_MAXIMUM,
            "peak_cuda_gib_maximum": PEAK_CUDA_GIB_MAXIMUM,
        },
        "selection": {
            "eligible_arm_definition": "At rank32, every family has unsquared mean-per-frame relative-L2 reduction >=0.20, corrected metric <= same-parent metric, and all reported rank32 scalars are finite. Raw residual-energy capture is not an eligibility gate.",
            "ranking": "Among eligible arms, maximize minimum family reduction, then maximize mean family reduction, then apply the frozen tie order.",
            "tie_order": list(TIE_ORDER),
            "no_eligible_arm_decision": "no_winner",
        },
        "failure_signal": "Any binding, disk, split/truth-scope, full401/time_block16, finite, peak-memory, output-size, or 0.10 GPU-hour contract failure blocks or fails the measured run. Passing calibration selects at most one arm and never authorizes validation.",
        "budget": {
            "physical_gpu_count": 1,
            "gpu_hours_maximum": 0.10,
            "gpu_seconds_maximum": GPU_SECONDS_MAXIMUM,
            "minimum_free_disk_bytes": MIN_FREE_BYTES,
            "free_disk_bytes_at_freeze": free,
            "output_size_bytes_maximum": MAX_OUTPUT_BYTES,
        },
        "rollback": {
            **checkpoint,
            "expected_sha256": parent_runtime.CHECKPOINT_SHA256,
            "read_only": True,
            "checkpoint_writes_permitted": False,
            "on_failure": "Preserve the frozen parent and all A3, B2-H, Helmholtz, ASAM, and CPADC checkpoints; write only the allowed scalar terminal result and do not open validation/test_id.",
        },
        "measured_command": _measured_command(),
        "protocol": {
            "split": "train",
            "record_count": 9,
            "families": {family: 3 for family in FAMILIES},
            "basis_records_per_family": 2,
            "exposed_calibration_records_per_family": 1,
            "independent_confirmation_records": 0,
            "time_count": TIME_COUNT,
            "time_block": TIME_BLOCK,
            "ranks": list(RANKS),
            "factorial_arms": list(ARM_KEYS),
            "raw_covariance": "C_raw=sum_i C_i, where C_i=sum over spatial chunks of R_i R_i^T and R_i=truth_i-parent_i for a basis record",
            "equaltrace_covariance": "C_equaltrace=sum_i C_i/trace(C_i), with each trace finite and strictly positive",
            "ordinary_ls": "For each rank and spatial point, argmin_a sum_t (r_t-B_t a)^2 on exposed train future frames.",
            "metric_weighted_ls": "For each rank and spatial point, argmin_a sum_t w_t (r_t-B_t a)^2 with w_t=1/max(||truth_t||_2^2,1e-30).",
            "weighted_objective_identity": "Summing the weighted LS objective over spatial points and dividing by frame count exactly equals mean squared per-frame relative L2 under the same denominator floor.",
            "weighted_objective_limitation": "Metric-weighted LS does not exactly minimize mean unsquared per-frame relative L2.",
            "future_window": "All stored frames strictly after the second registered onset observation.",
            "same_generation_contract": "Within a family, each record's parent/truth/residual is generated once in memory; each basis residual updates both covariance arms and the exposed calibration tensors are reused by all four arms and all ranks.",
            "streaming_and_serialization": "Covariances are accumulated by spatial chunks. Parent, truth, residual, coefficient, correction, and corrected fields are never serialized.",
            "metrics": {
                "mean_per_frame_relative_l2": "mean_t sqrt(||error_t||_2^2/max(||truth_t||_2^2,1e-30))",
                "mean_squared_per_frame_relative_l2": "mean_t ||error_t||_2^2/max(||truth_t||_2^2,1e-30)",
                "raw_residual_energy_capture": "1-||residual-correction||_2^2/||residual||_2^2 using unweighted future-field energy; diagnostic only",
                "correction_energy_ratio_to_parent": "||correction||_2^2/max(||parent||_2^2,1e-30) over future frames",
                "basis_design_condition_number": "sigma_max(B_future)/sigma_min(B_future)",
                "weighted_design_condition_number": "sigma_max(diag(sqrt(w))B_future)/sigma_min(diag(sqrt(w))B_future)",
                "paired_changes": "Each arm versus the same parent plus weighted-minus-ordinary and equaltrace-minus-raw within rank and family.",
            },
        },
        "truth_scope": {
            "train_future_truth_opened": True,
            "basis_truth_use": "all 401 frames only for the two covariance arms",
            "calibration_truth_use": "future frames for oracle coefficient fit, weights, and scoring",
            "validation_opened": False,
            "test_id_opened": False,
            "online_deployment_compatible": False,
            "validation_authorized": False,
        },
        "historical_exposure_scope": {
            "all_nine_records_are_historically_exposed": True,
            "group_ids": list(EXPECTED_GROUP_IDS),
            "third_per_family_historical_label": "disjoint_confirmation in r4e7_family_temporal_pod_capacity_train9_v1",
            "third_per_family_current_label": "exposed_calibration",
            "independent_confirmation_claim_permitted": False,
            "bound_source_artifacts": historical_artifacts,
            "winner_authorization": "A winner authorizes only a future separately preregistered fresh train confirmation. That preregistration must freeze a complete historical gate/calibration/confirmation exposure registry and exclude every group in its union, including all nine group_ids listed here.",
        },
        "sample_selection": {
            "algorithm": "Reuse exactly the ordered nine records and hashes frozen by r4e7_family_temporal_pod_capacity_train9_v1; first two per family are basis and third is exposed calibration, never independent confirmation.",
            "records": census,
        },
        "bindings": {
            "parent_checkpoint": checkpoint,
            "run_identity": parent_runtime.file_binding(parent_runtime.RUN_IDENTITY_PATH),
            "run_digest": str(run_identity["run_digest"]),
            "manifest_digest": str(run_identity["manifest_digest"]),
            "time_axis_sha256": str(run_identity["time_axis_sha256"]),
            "config": parent_runtime.file_binding(parent_runtime.CONFIG_PATH),
            "base_config": parent_runtime.file_binding(parent_runtime.BASE_CONFIG_PATH),
            "parent_identity": parent_runtime.file_binding(parent_runtime.PARENT_IDENTITY_PATH),
            "manifest_json": parent_runtime.file_binding(parent_runtime.MANIFEST_PATH),
            "normalization_json": parent_runtime.file_binding(parent_runtime.NORMALIZATION_PATH),
            "source_data": parent_runtime.source_data_binding(manifest),
            "code": _binding_map(CODE_BINDING_PATHS),
            "gpu": gpu,
            "software": parent_runtime.software_identity(),
        },
        "preflight_evidence": {
            "focused_test_command": str(focused_test_command),
            "focused_test_result": str(focused_test_result),
            "focused_tests_passed": True,
            "one_record_full401_smoke": dict(smoke_evidence),
        },
        "promotion_boundary": {
            "calibration_can_authorize_validation": False,
            "winner_can_authorize_only": "future_preregistered_fresh_train_confirmation_excluding_every_historically_exposed_gate_group",
            "no_winner_action": "stop_without_validation",
        },
    }
    parent_runtime.assert_no_placeholders_or_nulls(payload)
    return payload


def verify_frozen_preregistration(payload: Mapping[str, Any]) -> None:
    parent_runtime.assert_no_placeholders_or_nulls(payload)
    if payload.get("candidate") != CANDIDATE or payload.get("status") != "frozen":
        raise parent_runtime.BindingDriftError("preregistration identity is not frozen")
    if payload.get("measured_command") != _measured_command():
        raise parent_runtime.BindingDriftError("measured command binding drift")
    if payload["selection"]["tie_order"] != list(TIE_ORDER):
        raise parent_runtime.BindingDriftError("selection tie order drift")
    bindings = payload["bindings"]
    for name in (
        "parent_checkpoint", "run_identity", "config", "base_config",
        "parent_identity", "manifest_json", "normalization_json",
    ):
        parent_runtime.verify_binding(bindings[name])
    for binding in bindings["code"].values():
        parent_runtime.verify_binding(binding)
    for binding in payload["historical_exposure_scope"]["bound_source_artifacts"].values():
        parent_runtime.verify_binding(binding)
    source = bindings["source_data"]
    stat = Path(str(source["path"])).stat()
    for name, value in {
        "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev), "inode": int(stat.st_ino),
    }.items():
        if source[name] != value:
            raise parent_runtime.BindingDriftError(f"source VDS stat binding drift: {name}")
    manifest = parent_runtime.load_manifest_payload()
    if str(manifest["digest"]) != str(bindings["manifest_digest"]):
        raise parent_runtime.BindingDriftError("manifest digest binding drift")
    frozen = payload["sample_selection"]["records"]
    current = parent_runtime.select_train_records(manifest)
    if [record.sample_id for record in current] != [row["sample_id"] for row in frozen]:
        raise parent_runtime.BindingDriftError("exact historical sample selection drift")
    if [row["role"] for row in frozen] != [
        "basis", "basis", "exposed_calibration",
        "basis", "basis", "exposed_calibration",
        "basis", "basis", "exposed_calibration",
    ]:
        raise parent_runtime.BindingDriftError("exposed calibration labels drift")


def _verify_record_hashes(
    record: Any, frozen: Mapping[str, Any], loaded: Any, truth_hash: str
) -> None:
    if asdict(record) != {name: frozen[name] for name in asdict(record)}:
        raise parent_runtime.BindingDriftError(f"manifest sample drift: {record.sample_id}")
    if loaded.nontruth_input_sha256 != frozen["nontruth_input_sha256"]:
        raise parent_runtime.BindingDriftError(f"nontruth input drift: {record.sample_id}")
    if truth_hash != frozen["train_truth_sha256"]:
        raise parent_runtime.BindingDriftError(f"train truth drift: {record.sample_id}")


def select_winner(family_results: Mapping[str, Any]) -> dict[str, Any]:
    arm_summaries: dict[str, Any] = {}
    for arm in ARM_KEYS:
        reductions = {
            family: float(
                family_results[family]["arms"][arm]["ranks"]["32"]
                ["paired_changes_vs_parent"]["mean_per_frame_relative_l2_reduction"]
            )
            for family in FAMILIES
        }
        nonworse = {
            family: float(
                family_results[family]["arms"][arm]["ranks"]["32"]
                ["corrected_metrics"]["mean_per_frame_relative_l2"]
            )
            <= float(
                family_results[family]["arms"][arm]["parent_metrics"]
                ["mean_per_frame_relative_l2"]
            )
            for family in FAMILIES
        }
        finite = {
            family: bool(
                family_results[family]["arms"][arm]["ranks"]["32"]["finite"]
            )
            for family in FAMILIES
        }
        eligible = (
            all(value >= RANK32_REDUCTION_MINIMUM for value in reductions.values())
            and all(nonworse.values())
            and all(finite.values())
        )
        arm_summaries[arm] = {
            "family_unsquared_reductions": reductions,
            "family_nonworse": nonworse,
            "family_finite": finite,
            "minimum_family_reduction": min(reductions.values()),
            "mean_family_reduction": sum(reductions.values()) / len(reductions),
            "eligible": eligible,
            "raw_residual_energy_capture_used_for_eligibility": False,
        }
    eligible_arms = [arm for arm in ARM_KEYS if arm_summaries[arm]["eligible"]]
    ranking = sorted(
        eligible_arms,
        key=lambda arm: (
            -float(arm_summaries[arm]["minimum_family_reduction"]),
            -float(arm_summaries[arm]["mean_family_reduction"]),
            TIE_ORDER.index(arm),
        ),
    )
    winner = ranking[0] if ranking else None
    return {
        "decision": "winner" if winner is not None else "no_winner",
        "winner": winner,
        "eligible_ranking": ranking,
        "arm_summaries": arm_summaries,
        "validation_authorized": False,
        "winner_authorizes_only": (
            "future_preregistered_fresh_train_confirmation_excluding_every_"
            "historically_exposed_gate_group"
            if winner is not None
            else "nothing"
        ),
    }


def run_measured(
    *,
    preregistration_path: Path,
    result_path: Path,
    physical_gpu_index: int,
    device: torch.device,
    time_block: int,
) -> dict[str, Any]:
    prereg_sha = parent_runtime.sha256_file(preregistration_path)
    prereg = json.loads(preregistration_path.read_text(encoding="utf8"))
    base = {
        "schema": "r4_pod_metric_balance_factorial_result_v1",
        "candidate": CANDIDATE,
        "preregistration_path": str(preregistration_path.resolve()),
        "preregistration_sha256": prereg_sha,
        "started_utc": utc_now(),
        "claim_scope": prereg.get("claim_scope", "calibration_only"),
        "sealed_data_attestation": {
            "train_future_truth_opened_for_oracle_calibration": True,
            "independent_confirmation_opened": False,
            "validation_opened": False,
            "test_id_opened": False,
            "fields_serialized": False,
        },
        "checkpoint_attestation": {
            "writes": 0,
            "rollback_path": str(parent_runtime.CHECKPOINT_PATH),
            "rollback_sha256_before": parent_runtime.CHECKPOINT_SHA256,
        },
        "disk_free_bytes_before": parent_runtime.free_disk_bytes(),
        "output_size_bytes_maximum": MAX_OUTPUT_BYTES,
    }
    budget: parent_runtime.BudgetGuard | None = None
    try:
        if result_path.exists():
            raise FileExistsError(f"result terminal already exists: {result_path}")
        if int(time_block) != TIME_BLOCK:
            raise parent_runtime.BindingDriftError("measured time_block differs from 16")
        verify_frozen_preregistration(prereg)
        disk_gate = parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
        gpu = parent_runtime.gpu_identity(physical_gpu_index, device)
        if gpu != prereg["bindings"]["gpu"]:
            raise parent_runtime.BindingDriftError("measured GPU differs from freeze")
        budget = parent_runtime.BudgetGuard(maximum_seconds=GPU_SECONDS_MAXIMUM)
        model, normalizer, manifest, run_identity = parent_runtime.load_model_context(device)
        budget.check("model_load")
        records = parent_runtime.select_train_records(manifest)
        frozen_rows = prereg["sample_selection"]["records"]
        frozen_by_id = {str(row["sample_id"]): row for row in frozen_rows}
        time_axis = torch.tensor(manifest["time_s"], dtype=torch.float64)
        torch.cuda.reset_peak_memory_stats(device)
        family_results: dict[str, Any] = {}
        record_execution: list[dict[str, Any]] = []

        for family in FAMILIES:
            family_records = [record for record in records if record.family == family]
            if len(family_records) != 3:
                raise FactorialContractError(f"family record count changed: {family}")
            raw_covariance = torch.zeros(
                (TIME_COUNT, TIME_COUNT), dtype=torch.float32, device=device
            )
            equal_covariance = torch.zeros_like(raw_covariance)
            basis_rows = []
            for ordinal, record in enumerate(family_records[:2], start=1):
                budget.check(f"before_{family}_basis_{ordinal}")
                parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
                loaded = parent_runtime.load_record_input(record)
                frozen = frozen_by_id[record.sample_id]
                predicted = legacy_probe.generate_parent_full401(
                    model, normalizer, loaded, manifest,
                    device=device, time_block=time_block,
                )
                truth, truth_hash = legacy_probe.load_train_truth(record, device=device)
                _verify_record_hashes(record, frozen, loaded, truth_hash)
                residual = truth - predicted
                record_trace = covariance_pair_update(
                    raw_covariance, equal_covariance, residual
                )
                basis_rows.append({
                    "sample_id": record.sample_id,
                    "group_id": record.group_id,
                    "role": "basis",
                    "manifest_sample_sha256": record.manifest_sample_sha256,
                    "nontruth_input_sha256": loaded.nontruth_input_sha256,
                    "train_truth_sha256": truth_hash,
                    "full401_residual_energy_and_covariance_trace": record_trace,
                })
                record_execution.append({
                    "sample_id": record.sample_id, "family": family, "role": "basis",
                    "single_parent_truth_residual_generation": True,
                })
                del residual, truth, predicted
                budget.check(f"after_{family}_basis_{ordinal}")

            covariance_payload: dict[str, Any] = {}
            bases: dict[str, torch.Tensor] = {}
            for covariance_mode, covariance in (
                ("raw", raw_covariance), ("equaltrace", equal_covariance)
            ):
                eigenvalues, basis = temporal_pod(covariance)
                total = float(eigenvalues.sum().item())
                if not math.isfinite(total) or total <= 0.0:
                    raise FactorialContractError(
                        f"invalid {family} {covariance_mode} covariance energy"
                    )
                bases[covariance_mode] = basis
                covariance_payload[covariance_mode] = {
                    "trace": float(covariance.diagonal().double().sum().item()),
                    "dtype": "float32",
                    "eigendecomposition_dtype": "float64",
                    "eigenvalue_energy_capture_diagnostic": {
                        str(rank): float(eigenvalues[:rank].sum().item() / total)
                        for rank in RANKS
                    },
                }

            calibration = family_records[2]
            budget.check(f"before_{family}_exposed_calibration")
            parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
            loaded = parent_runtime.load_record_input(calibration)
            frozen = frozen_by_id[calibration.sample_id]
            predicted = legacy_probe.generate_parent_full401(
                model, normalizer, loaded, manifest,
                device=device, time_block=time_block,
            )
            truth, truth_hash = legacy_probe.load_train_truth(calibration, device=device)
            _verify_record_hashes(calibration, frozen, loaded, truth_hash)
            observed = onset_indices(
                time_axis,
                t0_s=float(loaded.source_parameters[3]),
                f0_hz=float(loaded.source_parameters[2]),
            )
            if list(observed) != frozen["observed_indices"]:
                raise parent_runtime.BindingDriftError(
                    f"registered onset drift: {calibration.sample_id}"
                )
            future_start = int(observed[1] + 1)
            shared_future = prepare_future_projection(
                predicted, truth, future_start=future_start
            )
            arms: dict[str, Any] = {}
            for covariance_mode in COVARIANCE_MODES:
                for fit_mode in FIT_MODES:
                    arm = f"{covariance_mode}+{fit_mode}"
                    arms[arm] = fit_projection_metrics(
                        shared_future, bases[covariance_mode],
                        fit_mode=fit_mode, ranks=RANKS,
                    )
                    budget.check(f"after_{family}_{arm}")
            family_results[family] = {
                "basis_records": basis_rows,
                "covariances": covariance_payload,
                "exposed_calibration_record": {
                    "sample_id": calibration.sample_id,
                    "group_id": calibration.group_id,
                    "role": "exposed_calibration",
                    "historical_role": frozen["historical_role"],
                    "independent_confirmation": False,
                    "manifest_sample_sha256": calibration.manifest_sample_sha256,
                    "nontruth_input_sha256": loaded.nontruth_input_sha256,
                    "train_truth_sha256": truth_hash,
                    "observed_indices": list(observed),
                    "future_start_index": future_start,
                },
                "same_in_memory_parent_truth_residual_for_all_four_arms": True,
                "arms": arms,
                "factorial_paired_changes": factorial_paired_changes(arms),
            }
            record_execution.append({
                "sample_id": calibration.sample_id,
                "family": family,
                "role": "exposed_calibration",
                "independent_confirmation": False,
                "single_parent_truth_residual_generation_shared_by_all_arms": True,
            })
            del truth, predicted, raw_covariance, equal_covariance, bases
            budget.check(f"after_{family}_exposed_calibration")

        torch.cuda.synchronize(device)
        elapsed = budget.check("final_aggregation")
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if elapsed > GPU_SECONDS_MAXIMUM:
            raise parent_runtime.BudgetExceededError("measured factorial exceeded budget")
        if peak_reserved > PEAK_CUDA_BYTES_MAXIMUM:
            raise FactorialContractError("measured factorial exceeded peak CUDA gate")
        selection = select_winner(family_results)
        checkpoint_after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
        if checkpoint_after != parent_runtime.CHECKPOINT_SHA256:
            raise parent_runtime.BindingDriftError("parent checkpoint changed during probe")
        payload = {
            **base,
            "status": "success",
            "decision": selection["decision"],
            "completed_utc": utc_now(),
            "exact_blocker": "none",
            "protocol": prereg["protocol"],
            "truth_scope": prereg["truth_scope"],
            "historical_exposure_scope": prereg["historical_exposure_scope"],
            "promotion_boundary": prereg["promotion_boundary"],
            "disk_free_bytes_at_binding_gate": disk_gate,
            "disk_free_bytes_after": parent_runtime.free_disk_bytes(),
            "budget": {
                "one_gpu_seconds_used": elapsed,
                "one_gpu_seconds_maximum": GPU_SECONDS_MAXIMUM,
                "one_gpu_hours_maximum": 0.10,
                "within_budget": True,
            },
            "gpu": gpu,
            "run_binding": {
                "run_digest": str(run_identity["run_digest"]),
                "manifest_digest": str(run_identity["manifest_digest"]),
                "time_axis_sha256": str(run_identity["time_axis_sha256"]),
                "checkpoint_sha256": parent_runtime.CHECKPOINT_SHA256,
            },
            "sample_census": {
                "count": len(record_execution),
                "all_train": True,
                "unique_group_count": len({record.group_id for record in records}),
                "roles": dict(Counter(row["role"] for row in record_execution)),
                "execution": record_execution,
            },
            "family_results": family_results,
            "selection": selection,
            "finite_checks": {
                "all_family_results_finite": _all_finite(family_results),
                "selection_finite": _all_finite(selection),
            },
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        }
        if not all(payload["finite_checks"].values()):
            raise FloatingPointError("measured scalar report contains non-finite values")
    except (parent_runtime.BindingDriftError, parent_runtime.DiskRiskError, FileNotFoundError) as error:
        payload = {
            **base,
            "status": "blocked",
            "decision": "no_winner",
            "completed_utc": utc_now(),
            "exact_blocker": f"{type(error).__name__}: {error}",
            "disk_free_bytes_after": parent_runtime.free_disk_bytes(),
            "budget": {
                "one_gpu_seconds_used": 0.0 if budget is None else budget.elapsed(),
                "one_gpu_seconds_maximum": GPU_SECONDS_MAXIMUM,
            },
        }
    except Exception as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        payload = {
            **base,
            "status": "failed",
            "decision": "no_winner",
            "completed_utc": utc_now(),
            "exact_blocker": f"{type(error).__name__}: {error}",
            "disk_free_bytes_after": parent_runtime.free_disk_bytes(),
            "budget": {
                "one_gpu_seconds_used": 0.0 if budget is None else budget.elapsed(),
                "one_gpu_seconds_maximum": GPU_SECONDS_MAXIMUM,
            },
        }
    after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    payload["checkpoint_attestation"]["rollback_sha256_after"] = after
    payload["checkpoint_attestation"]["unchanged"] = (
        after == parent_runtime.CHECKPOINT_SHA256
    )
    parent_runtime.atomic_json_exclusive(payload, result_path, limit=MAX_OUTPUT_BYTES)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "preregister", "measured"), required=True)
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-block", type=int, default=TIME_BLOCK)
    parser.add_argument("--preregistration", type=Path, default=PREREGISTRATION_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--smoke-evidence-json", default="")
    parser.add_argument("--focused-test-command", default="")
    parser.add_argument("--focused-test-result", default="")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.mode == "smoke":
        payload = run_smoke(
            physical_gpu_index=args.physical_gpu_index,
            device=device,
            time_block=args.time_block,
        )
    elif args.mode == "preregister":
        if not args.smoke_evidence_json:
            raise FactorialContractError("preregistration requires smoke evidence JSON")
        payload = build_preregistration(
            physical_gpu_index=args.physical_gpu_index,
            device=device,
            time_block=args.time_block,
            smoke_evidence=json.loads(args.smoke_evidence_json),
            focused_test_command=args.focused_test_command,
            focused_test_result=args.focused_test_result,
        )
        parent_runtime.atomic_json_exclusive(
            payload, args.preregistration, limit=MAX_OUTPUT_BYTES
        )
    else:
        payload = run_measured(
            preregistration_path=args.preregistration,
            result_path=args.result,
            physical_gpu_index=args.physical_gpu_index,
            device=device,
            time_block=args.time_block,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"passed", "frozen", "success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_KEYS",
    "COVARIANCE_MODES",
    "EXPECTED_GROUP_IDS",
    "EXPECTED_SAMPLE_IDS",
    "FIT_MODES",
    "FactorialContractError",
    "GPU_SECONDS_MAXIMUM",
    "MAX_OUTPUT_BYTES",
    "MIN_FREE_BYTES",
    "PEAK_CUDA_BYTES_MAXIMUM",
    "RANK32_REDUCTION_MINIMUM",
    "RANKS",
    "TIE_ORDER",
    "TIME_BLOCK",
    "TruthScopeError",
    "condition_number",
    "covariance_pair_update",
    "fit_coefficients",
    "fit_projection_metrics",
    "prepare_future_projection",
    "sample_census",
    "select_winner",
    "temporal_pod",
]
