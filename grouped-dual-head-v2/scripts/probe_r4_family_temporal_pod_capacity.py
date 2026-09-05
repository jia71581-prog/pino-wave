#!/usr/bin/env python3
"""Offline train-oracle temporal POD capacity probe for the frozen R4 parent.

This program is deliberately not an online adaptation method.  It uses future
truth from nine group-disjoint *training* records.  Two records per family form
a temporal residual covariance and the third is a disjoint confirmation record
whose future residual is used to fit one oracle coefficient vector per spatial
point.  Parent, truth, residual, and corrected fields are never serialized.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from saved_time_phase_operator_v4.instance_adaptation.contracts import onset_indices
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime


CANDIDATE = "r4e7_family_temporal_pod_capacity_train9_v1"
SCRIPT_PATH = Path(__file__).resolve()
TEST_PATH = (
    PROJECT_ROOT
    / "tests/saved_time_phase_operator_v4/test_r4_family_temporal_pod_capacity.py"
)
PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "results/r4e7_family_temporal_pod_capacity_train9_v1_preregistration_20260825.json"
)
RESULT_PATH = (
    PROJECT_ROOT
    / "results/r4e7_family_temporal_pod_capacity_train9_v1_20260825.json"
)

FAMILIES = ("uniform", "layered", "marmousi")
RANKS = (8, 16, 32)
TIME_COUNT = 401
HEIGHT = 201
WIDTH = 201
TIME_BLOCK = 16
TRUTH_HASH_BLOCK = 16
SPATIAL_CHUNK = 4096
GPU_SECONDS_MAXIMUM = 360.0
MIN_FREE_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 10 * 1024**2
PEAK_CUDA_GIB_MAXIMUM = 23.5
PEAK_CUDA_BYTES_MAXIMUM = int(PEAK_CUDA_GIB_MAXIMUM * 1024**3)
RANK32_ERROR_REDUCTION_MINIMUM = 0.20
RANK32_RESIDUAL_CAPTURE_MINIMUM = 0.30
MEAN_PER_FRAME_EPS_SQUARED = 1.0e-30

CODE_BINDING_PATHS = (
    SCRIPT_PATH,
    TEST_PATH,
    PROJECT_ROOT / "scripts/benchmark_r4_parent_e2e_trainonly.py",
    PROJECT_ROOT / "saved_time_phase_operator_v4/instance_adaptation/contracts.py",
    PROJECT_ROOT / "saved_time_phase_operator_v4/evaluation.py",
    PROJECT_ROOT / "scripts/train_saved_time_v4_probe.py",
    PROJECT_ROOT / "scripts/train_saved_time_v4_full_support.py",
    PROJECT_ROOT / "saved_time_phase_operator_v4/operator.py",
    PROJECT_ROOT / "saved_time_phase_operator_v4/decoder.py",
    PROJECT_ROOT / "saved_time_phase_operator_v4/local_field.py",
    PROJECT_ROOT / "saved_time_phase_operator_v4/spectral.py",
    PROJECT_ROOT / "saved_time_phase_operator_v4/features.py",
    PROJECT_ROOT / "saved_time_phase_operator_v4/probe.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/config.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/normalization.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/training/checkpoint.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/model/operator.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/model/medium.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/model/source.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/model/travel_time.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/model/fusion.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/model/dense.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/model/features.py",
    PROJECT_ROOT / "grouped_ufno_mionet_v3/model/spectral.py",
)


class CapacityContractError(RuntimeError):
    """The frozen capacity-probe contract is invalid."""


class TruthScopeError(CapacityContractError):
    """A non-train truth access was attempted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _array_digest_start(name: str, shape: Sequence[int], dtype: np.dtype) -> Any:
    digest = hashlib.sha256()
    digest.update(str(name).encode("ascii"))
    digest.update(json.dumps([int(v) for v in shape], separators=(",", ":")).encode("ascii"))
    digest.update(str(np.dtype(dtype)).encode("ascii"))
    return digest


def _update_digest(digest: Any, value: np.ndarray) -> None:
    digest.update(np.ascontiguousarray(value).tobytes(order="C"))


def _assert_train_binding(handle: h5py.File, record: Any) -> None:
    index = int(record.source_index)
    observed = {
        "sample_id": _decode(handle["sample_id"][index]),
        "group_id": _decode(handle["group_id"][index]),
        "split": _decode(handle["split"][index]),
        "medium_type": _decode(handle["medium_type"][index]),
        "sample_sha256": _decode(handle["sample_sha256"][index]),
    }
    expected = {
        "sample_id": str(record.sample_id),
        "group_id": str(record.group_id),
        "split": "train",
        "medium_type": str(record.family),
        "sample_sha256": str(record.manifest_sample_sha256),
    }
    if observed != expected:
        raise TruthScopeError(
            f"train record binding mismatch for source index {index}: {observed}"
        )


def stream_train_truth_sha256(record: Any, *, block: int = TRUTH_HASH_BLOCK) -> str:
    """Hash one selected train wavefield in time blocks without retaining it."""
    if str(record.split) != "train":
        raise TruthScopeError("only train truth may be hashed")
    if int(block) <= 0:
        raise ValueError("truth hash block must be positive")
    with h5py.File(parent_runtime.SOURCE_H5_PATH, "r", swmr=True) as handle:
        _assert_train_binding(handle, record)
        dataset = handle["wavefield"]
        expected_shape = (TIME_COUNT, HEIGHT, WIDTH)
        if tuple(dataset.shape[1:]) != expected_shape or dataset.dtype != np.float32:
            raise CapacityContractError("train wavefield storage contract changed")
        digest = _array_digest_start("wavefield", expected_shape, np.dtype("float32"))
        for start in range(0, TIME_COUNT, int(block)):
            stop = min(start + int(block), TIME_COUNT)
            values = np.asarray(
                dataset[int(record.source_index), start:stop], dtype=np.float32
            )
            if values.shape != (stop - start, HEIGHT, WIDTH):
                raise CapacityContractError("streamed train truth block shape changed")
            if not np.isfinite(values).all():
                raise FloatingPointError("streamed train truth contains non-finite values")
            _update_digest(digest, values)
    return digest.hexdigest()


def load_train_truth(record: Any, *, device: torch.device) -> tuple[torch.Tensor, str]:
    """Open exactly one full train truth record in memory, never on disk output."""
    if str(record.split) != "train":
        raise TruthScopeError("only train truth may be opened")
    with h5py.File(parent_runtime.SOURCE_H5_PATH, "r", swmr=True) as handle:
        _assert_train_binding(handle, record)
        values = np.asarray(
            handle["wavefield"][int(record.source_index)], dtype=np.float32
        )
    if values.shape != (TIME_COUNT, HEIGHT, WIDTH):
        raise CapacityContractError("full train truth shape changed")
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    if not np.isfinite(values).all():
        raise FloatingPointError("full train truth contains non-finite values")
    digest = _array_digest_start("wavefield", values.shape, values.dtype)
    _update_digest(digest, values)
    tensor = torch.from_numpy(values).to(device=device, non_blocking=False)
    return tensor, digest.hexdigest()


@torch.inference_mode()
def generate_parent_full401(
    model: Any,
    normalizer: Any,
    loaded: Any,
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
    time_block: int,
) -> torch.Tensor:
    """Generate all 401 physical frames on one GPU with the frozen parent."""
    if str(loaded.record.split) != "train":
        raise TruthScopeError("parent generation is restricted to selected train records")
    if int(time_block) != TIME_BLOCK or len(manifest["time_s"]) != TIME_COUNT:
        raise CapacityContractError("parent generation requires full-401 time_block=16")
    velocity = torch.from_numpy(loaded.velocity_mps)[None, None].to(device)
    source = torch.from_numpy(loaded.source_parameters)[None].to(device)
    source_map = torch.from_numpy(loaded.source_map)[None, None].to(device)
    requested_times = torch.tensor(manifest["time_s"], dtype=torch.float32, device=device)
    x_m = torch.tensor(manifest["x_m"], dtype=torch.float32, device=device)
    z_m = torch.tensor(manifest["z_m"], dtype=torch.float32, device=device)
    record_to_medium = torch.zeros(1, dtype=torch.long, device=device)
    medium = model.encode_medium(velocity, normalizer)
    prepared = model.prepare_sources(
        medium, source, source_map, normalizer, record_to_medium=record_to_medium
    )
    normalized = model.dense_normalized(
        prepared,
        requested_times,
        x_m=x_m,
        z_m=z_m,
        time_block=int(time_block),
    )
    physical = normalizer.decode_pressure(normalized.float(), source[:, 4])[0].contiguous()
    if physical.shape != (TIME_COUNT, HEIGHT, WIDTH) or physical.dtype != torch.float32:
        raise CapacityContractError("parent full-401 output shape or dtype changed")
    if not bool(torch.isfinite(physical).all()):
        raise FloatingPointError("parent full-401 output contains non-finite values")
    return physical


def stream_accumulate_temporal_covariance(
    covariance: torch.Tensor,
    residual: torch.Tensor,
    *,
    spatial_chunk: int = SPATIAL_CHUNK,
) -> torch.Tensor:
    """Accumulate R R^T over spatial chunks for R=[time, spatial point]."""
    value = torch.as_tensor(residual)
    if value.ndim < 2 or value.shape[0] != covariance.shape[0]:
        raise ValueError("residual and temporal covariance shapes disagree")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("temporal covariance must be square")
    if covariance.device != value.device or covariance.dtype != value.dtype:
        raise ValueError("covariance and residual must share device and dtype")
    if int(spatial_chunk) <= 0:
        raise ValueError("spatial chunk must be positive")
    matrix = value.reshape(value.shape[0], -1)
    for start in range(0, matrix.shape[1], int(spatial_chunk)):
        block = matrix[:, start : start + int(spatial_chunk)]
        covariance.addmm_(block, block.transpose(0, 1))
    return covariance


def temporal_pod(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return descending nonnegative eigenvalues and temporal eigenvectors."""
    value = torch.as_tensor(covariance).detach().to(device="cpu", dtype=torch.float64)
    if value.shape != (TIME_COUNT, TIME_COUNT) and value.shape[0] != value.shape[1]:
        raise ValueError("POD covariance must be square")
    value = 0.5 * (value + value.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    order = torch.arange(eigenvalues.numel() - 1, -1, -1)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    if not bool(torch.isfinite(eigenvalues).all() and torch.isfinite(eigenvectors).all()):
        raise FloatingPointError("POD eigendecomposition is non-finite")
    return eigenvalues, eigenvectors


def mean_per_frame_relative_l2(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    eps_squared: float = MEAN_PER_FRAME_EPS_SQUARED,
) -> float:
    """Mean of framewise spatial relative L2 values, computed in float64."""
    predicted = torch.as_tensor(prediction)
    target = torch.as_tensor(truth, device=predicted.device)
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError("prediction and truth must match [time, ...space]")
    difference = (predicted - target).reshape(predicted.shape[0], -1).double()
    target_flat = target.reshape(target.shape[0], -1).double()
    numerator = difference.square().sum(dim=1).sqrt()
    denominator = target_flat.square().sum(dim=1).clamp_min(float(eps_squared)).sqrt()
    values = numerator / denominator
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("mean-per-frame relative L2 is non-finite")
    return float(values.mean().item())


@torch.inference_mode()
def oracle_projection_metrics(
    parent: torch.Tensor,
    truth: torch.Tensor,
    temporal_basis: torch.Tensor,
    *,
    future_start: int,
    ranks: Sequence[int] = RANKS,
    spatial_chunk: int = SPATIAL_CHUNK,
) -> dict[str, Any]:
    """Fit future-truth oracle coefficients independently at each spatial point."""
    predicted = torch.as_tensor(parent)
    target = torch.as_tensor(truth, device=predicted.device)
    if predicted.shape != target.shape or predicted.shape[0] != temporal_basis.shape[0]:
        raise ValueError("parent, truth, and temporal basis time axes disagree")
    start = int(future_start)
    if not 0 < start < predicted.shape[0]:
        raise ValueError("future start must lie inside the time axis")
    future_parent = predicted[start:].reshape(predicted.shape[0] - start, -1)
    future_truth = target[start:].reshape(target.shape[0] - start, -1)
    future_residual = future_truth - future_parent
    residual_energy = float(future_residual.double().square().sum().item())
    if not math.isfinite(residual_energy) or residual_energy <= 0.0:
        raise CapacityContractError("confirmation future residual energy is invalid")
    parent_metric = mean_per_frame_relative_l2(predicted[start:], target[start:])
    output: dict[str, Any] = {
        "parent_mean_per_frame_relative_l2": parent_metric,
        "future_residual_energy": residual_energy,
        "future_frame_count": int(predicted.shape[0] - start),
        "ranks": {},
    }
    basis_cpu = torch.as_tensor(temporal_basis, device="cpu", dtype=torch.float64)
    for rank_value in ranks:
        rank = int(rank_value)
        if rank <= 0 or rank > basis_cpu.shape[1]:
            raise ValueError("oracle POD rank is outside the available basis")
        restricted = basis_cpu[start:, :rank]
        coefficient_map = torch.linalg.pinv(restricted, rtol=1.0e-12).to(
            device=predicted.device, dtype=predicted.dtype
        )
        restricted_device = restricted.to(device=predicted.device, dtype=predicted.dtype)
        error_by_frame = torch.zeros(
            restricted.shape[0], dtype=torch.float64, device=predicted.device
        )
        truth_by_frame = torch.zeros_like(error_by_frame)
        remaining_energy = 0.0
        for spatial_start in range(0, future_parent.shape[1], int(spatial_chunk)):
            spatial_stop = min(spatial_start + int(spatial_chunk), future_parent.shape[1])
            residual_block = future_residual[:, spatial_start:spatial_stop]
            coefficients = coefficient_map @ residual_block
            correction = restricted_device @ coefficients
            remaining = residual_block - correction
            remaining64 = remaining.double()
            truth64 = future_truth[:, spatial_start:spatial_stop].double()
            error_by_frame += remaining64.square().sum(dim=1)
            truth_by_frame += truth64.square().sum(dim=1)
            remaining_energy += float(remaining64.square().sum().item())
        corrected_frame_values = torch.sqrt(
            error_by_frame / truth_by_frame.clamp_min(MEAN_PER_FRAME_EPS_SQUARED)
        )
        corrected_metric = float(corrected_frame_values.mean().item())
        reduction = (parent_metric - corrected_metric) / max(parent_metric, 1.0e-30)
        capture = 1.0 - remaining_energy / residual_energy
        values = (corrected_metric, reduction, capture, remaining_energy)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(f"rank-{rank} oracle output is non-finite")
        output["ranks"][str(rank)] = {
            "corrected_mean_per_frame_relative_l2": corrected_metric,
            "relative_error_reduction": float(reduction),
            "residual_energy_capture": float(capture),
            "unexplained_residual_energy": float(remaining_energy),
            "coefficient_fit": "independent_least_squares_at_each_spatial_point_using_reserved_train_future_truth",
        }
    return output


def _role(record_position: int) -> str:
    return "basis" if int(record_position) < 2 else "disjoint_confirmation"


def sample_census(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = parent_runtime.select_train_records(manifest)
    counts = Counter(record.family for record in records)
    if len(records) != 9 or counts != Counter({family: 3 for family in FAMILIES}):
        raise CapacityContractError("sample census is not exactly three per family")
    if len({record.group_id for record in records}) != 9:
        raise CapacityContractError("sample census is not globally group-disjoint")
    time_axis = torch.tensor(manifest["time_s"], dtype=torch.float64)
    output = []
    family_positions: Counter[str] = Counter()
    for record in records:
        loaded = parent_runtime.load_record_input(record)
        position = int(family_positions[record.family])
        family_positions[record.family] += 1
        observed = onset_indices(
            time_axis,
            t0_s=float(loaded.source_parameters[3]),
            f0_hz=float(loaded.source_parameters[2]),
        )
        output.append(
            {
                **asdict(record),
                "family_position": position,
                "role": _role(position),
                "nontruth_input_sha256": loaded.nontruth_input_sha256,
                "train_truth_sha256": stream_train_truth_sha256(record),
                "observed_indices": list(observed),
                "future_start_index": int(observed[1] + 1),
                "future_frame_count": int(TIME_COUNT - observed[1] - 1),
            }
        )
    return output


def _binding_map(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {str(path.resolve()): parent_runtime.file_binding(path) for path in paths}


def _measured_command() -> str:
    return (
        "env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python "
        "scripts/probe_r4_family_temporal_pod_capacity.py --mode measured "
        "--physical-gpu-index 0 --device cuda:0 --time-block 16 "
        "--preregistration results/r4e7_family_temporal_pod_capacity_train9_v1_preregistration_20260825.json "
        "--result results/r4e7_family_temporal_pod_capacity_train9_v1_20260825.json"
    )


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
        raise CapacityContractError("frozen measured run requires physical GPU 0 as cuda:0")
    if int(time_block) != TIME_BLOCK:
        raise CapacityContractError("frozen measured run requires time_block=16")
    if (
        smoke_evidence.get("status") != "passed"
        or smoke_evidence.get("unscored") is not True
        or smoke_evidence.get("time_block") != TIME_BLOCK
        or smoke_evidence.get("script_sha256") != parent_runtime.sha256_file(SCRIPT_PATH)
    ):
        raise CapacityContractError("one-record full-time smoke evidence is invalid or stale")
    manifest = parent_runtime.load_manifest_payload()
    run_identity = json.loads(parent_runtime.RUN_IDENTITY_PATH.read_text(encoding="utf8"))
    checkpoint = parent_runtime.file_binding(parent_runtime.CHECKPOINT_PATH)
    if checkpoint["sha256"] != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError("parent checkpoint hash mismatch")
    census = sample_census(manifest)
    gpu = parent_runtime.gpu_identity(physical_gpu_index, device)
    payload = {
        "schema": "r4_family_temporal_pod_capacity_preregistration_v1",
        "candidate": CANDIDATE,
        "status": "frozen",
        "created_utc": utc_now(),
        "hypothesis": "Within each medium family, the frozen epoch-7 R4 parent's full-401 train residual has a transferable low-rank temporal subspace: a rank-32 basis learned from two group-disjoint train records will reduce the reserved third train record's post-observation mean-per-frame relative L2 by at least 20% and capture at least 30% of its future residual energy.",
        "mechanism": "stream a family 401x401 covariance of truth-minus-parent over the two basis records, eigendecompose it, then use reserved future train truth to fit POD coefficients independently at every spatial point",
        "claim_scope": "offline_train_oracle_accuracy_capacity_only_not_online_adaptation_not_validation_not_test_id",
        "acceptance": {
            "rank32_relative_error_reduction_minimum_each_family": RANK32_ERROR_REDUCTION_MINIMUM,
            "rank32_residual_energy_capture_minimum_each_family": RANK32_RESIDUAL_CAPTURE_MINIMUM,
            "finite_output_required": True,
            "peak_cuda_reserved_bytes_maximum": PEAK_CUDA_BYTES_MAXIMUM,
            "peak_cuda_gib_maximum": PEAK_CUDA_GIB_MAXIMUM,
            "joint_gate": "every listed condition must pass",
        },
        "failure_signal": "Any family rank-32 gate miss yields success/fail_gate; any non-finite value, OOM, binding drift, split/truth-scope violation, disk shortfall, output overflow, multi-GPU exposure, time-axis/time-block mismatch, or budget overrun yields blocked or failed.",
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
        },
        "measured_command": _measured_command(),
        "protocol": {
            "split": "train",
            "record_count": 9,
            "families": {family: 3 for family in FAMILIES},
            "globally_group_disjoint": True,
            "basis_records_per_family": 2,
            "confirmation_records_per_family": 1,
            "time_count": TIME_COUNT,
            "time_block": TIME_BLOCK,
            "covariance": "sum over basis records and spatial chunks of R @ R.T where R=truth-parent has shape [401, spatial_point]",
            "covariance_accumulation_dtype": "float32",
            "eigendecomposition_dtype": "float64",
            "ranks": list(RANKS),
            "coefficient_fit": "for each rank and reserved record, independently at each spatial point by least squares on all train future frames strictly after the second registered onset observation",
            "metric": "mean over future frames of spatial relative L2, with numerator and denominator reductions in float64",
            "metric_eps_squared": MEAN_PER_FRAME_EPS_SQUARED,
            "residual_energy_capture": "1 - squared norm of oracle-unexplained future residual / squared norm of parent future residual",
            "parent_truth_residual_corrected_arrays_written_to_disk": False,
        },
        "truth_scope": {
            "train_future_truth_opened": True,
            "basis_truth_use": "all 401 frames only for family covariance",
            "confirmation_truth_use": "future frames for offline oracle coefficient fit and scoring; full residual may be resident only in memory",
            "validation_opened": False,
            "test_id_opened": False,
            "online_deployment_compatible": False,
            "online_adaptation_claim_permitted": False,
        },
        "sample_selection": {
            "algorithm": "for each family in uniform, layered, marmousi order, select the first three train manifest rows whose group has not appeared globally; first two are basis and third is confirmation",
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
            "one_record_full_time_smoke": dict(smoke_evidence),
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
    bindings = payload["bindings"]
    for name in (
        "parent_checkpoint",
        "run_identity",
        "config",
        "base_config",
        "parent_identity",
        "manifest_json",
        "normalization_json",
    ):
        parent_runtime.verify_binding(bindings[name])
    for binding in bindings["code"].values():
        parent_runtime.verify_binding(binding)
    source = bindings["source_data"]
    stat = Path(str(source["path"])).stat()
    for name, value in {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }.items():
        if source[name] != value:
            raise parent_runtime.BindingDriftError(f"source VDS stat binding drift: {name}")
    manifest = parent_runtime.load_manifest_payload()
    if str(manifest["digest"]) != str(bindings["manifest_digest"]):
        raise parent_runtime.BindingDriftError("manifest digest binding drift")
    current = parent_runtime.select_train_records(manifest)
    frozen = payload["sample_selection"]["records"]
    if [record.sample_id for record in current] != [row["sample_id"] for row in frozen]:
        raise parent_runtime.BindingDriftError("sample selection binding drift")


def _verify_record_hashes(record: Any, frozen: Mapping[str, Any], loaded: Any, truth_hash: str) -> None:
    if asdict(record) != {name: frozen[name] for name in asdict(record)}:
        raise parent_runtime.BindingDriftError(f"manifest sample binding drift: {record.sample_id}")
    if loaded.nontruth_input_sha256 != frozen["nontruth_input_sha256"]:
        raise parent_runtime.BindingDriftError(f"nontruth input hash drift: {record.sample_id}")
    if truth_hash != frozen["train_truth_sha256"]:
        raise parent_runtime.BindingDriftError(f"train truth hash drift: {record.sample_id}")


def run_smoke(
    *, physical_gpu_index: int, device: torch.device, time_block: int
) -> dict[str, Any]:
    disk_before = parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    gpu = parent_runtime.gpu_identity(physical_gpu_index, device)
    if parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH) != parent_runtime.CHECKPOINT_SHA256:
        raise parent_runtime.BindingDriftError("parent checkpoint hash mismatch before smoke")
    started = time.monotonic()
    model, normalizer, manifest, _ = parent_runtime.load_model_context(device)
    record = parent_runtime.select_train_records(manifest)[0]
    loaded = parent_runtime.load_record_input(record)
    torch.cuda.reset_peak_memory_stats(device)
    parent = generate_parent_full401(
        model, normalizer, loaded, manifest, device=device, time_block=time_block
    )
    truth, truth_hash = load_train_truth(record, device=device)
    covariance = torch.zeros((TIME_COUNT, TIME_COUNT), dtype=torch.float32, device=device)
    stream_accumulate_temporal_covariance(covariance, truth - parent)
    eigenvalues, eigenvectors = temporal_pod(covariance)
    observed = onset_indices(
        torch.tensor(manifest["time_s"], dtype=torch.float64),
        t0_s=float(loaded.source_parameters[3]),
        f0_hz=float(loaded.source_parameters[2]),
    )
    diagnostic = oracle_projection_metrics(
        parent,
        truth,
        eigenvectors,
        future_start=int(observed[1] + 1),
        ranks=(8,),
    )
    torch.cuda.synchronize(device)
    elapsed = float(time.monotonic() - started)
    if elapsed > GPU_SECONDS_MAXIMUM:
        raise parent_runtime.BudgetExceededError("smoke exceeded the complete GPU budget")
    payload = {
        "status": "passed",
        "unscored": True,
        "utc": utc_now(),
        "record": asdict(record),
        "nontruth_input_sha256": loaded.nontruth_input_sha256,
        "train_truth_sha256": truth_hash,
        "observed_indices": list(observed),
        "future_start_index": int(observed[1] + 1),
        "time_count": TIME_COUNT,
        "time_block": int(time_block),
        "parent_shape": list(parent.shape),
        "parent_dtype": str(parent.dtype).replace("torch.", ""),
        "parent_finite": True,
        "covariance_shape": list(covariance.shape),
        "covariance_trace": float(covariance.diagonal().double().sum().item()),
        "largest_eigenvalue": float(eigenvalues[0].item()),
        "unscored_rank8_self_projection_diagnostic": diagnostic["ranks"]["8"],
        "gpu_seconds_used": elapsed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "disk_free_bytes_before": disk_before,
        "disk_free_bytes_after": parent_runtime.free_disk_bytes(),
        "gpu": gpu,
        "truth_scope": "one selected train record only",
        "validation_opened": False,
        "test_id_opened": False,
        "arrays_serialized": False,
        "checkpoint_writes": 0,
        "script_sha256": parent_runtime.sha256_file(SCRIPT_PATH),
    }
    if payload["peak_cuda_reserved_bytes"] > PEAK_CUDA_BYTES_MAXIMUM:
        raise CapacityContractError("smoke peak CUDA reserved memory exceeds gate")
    return payload


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
        "schema": "r4_family_temporal_pod_capacity_result_v1",
        "candidate": CANDIDATE,
        "preregistration_path": str(preregistration_path.resolve()),
        "preregistration_sha256": prereg_sha,
        "started_utc": utc_now(),
        "claim_scope": "offline_train_oracle_accuracy_capacity_only_not_online_adaptation_not_validation_not_test_id",
        "sealed_data_attestation": {
            "train_future_truth_opened_for_offline_oracle": True,
            "validation_opened": False,
            "test_id_opened": False,
            "parent_truth_residual_corrected_arrays_written_to_disk": False,
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
            raise parent_runtime.BindingDriftError("measured GPU differs from preregistration")
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
                raise CapacityContractError(f"family record count changed: {family}")
            covariance = torch.zeros(
                (TIME_COUNT, TIME_COUNT), dtype=torch.float32, device=device
            )
            basis_rows = []
            for ordinal, record in enumerate(family_records[:2], start=1):
                budget.check(f"before_{family}_basis_{ordinal}")
                parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
                loaded = parent_runtime.load_record_input(record)
                frozen = frozen_by_id[record.sample_id]
                if loaded.nontruth_input_sha256 != frozen["nontruth_input_sha256"]:
                    raise parent_runtime.BindingDriftError(
                        f"nontruth input hash drift before parent generation: {record.sample_id}"
                    )
                predicted = generate_parent_full401(
                    model,
                    normalizer,
                    loaded,
                    manifest,
                    device=device,
                    time_block=time_block,
                )
                truth, truth_hash = load_train_truth(record, device=device)
                _verify_record_hashes(record, frozen, loaded, truth_hash)
                residual = truth - predicted
                residual_energy = float(residual.double().square().sum().item())
                stream_accumulate_temporal_covariance(covariance, residual)
                basis_rows.append(
                    {
                        "sample_id": record.sample_id,
                        "group_id": record.group_id,
                        "manifest_sample_sha256": record.manifest_sample_sha256,
                        "nontruth_input_sha256": loaded.nontruth_input_sha256,
                        "train_truth_sha256": truth_hash,
                        "full401_residual_energy": residual_energy,
                    }
                )
                record_execution.append(
                    {"sample_id": record.sample_id, "family": family, "role": "basis"}
                )
                del residual, truth, predicted
                budget.check(f"after_{family}_basis_{ordinal}")

            eigenvalues, eigenvectors = temporal_pod(covariance)
            eigen_total = float(eigenvalues.sum().item())
            if not math.isfinite(eigen_total) or eigen_total <= 0.0:
                raise CapacityContractError(f"family covariance energy is invalid: {family}")
            basis_captures = {
                str(rank): float(eigenvalues[:rank].sum().item() / eigen_total)
                for rank in RANKS
            }

            confirmation = family_records[2]
            budget.check(f"before_{family}_confirmation")
            parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
            loaded = parent_runtime.load_record_input(confirmation)
            frozen = frozen_by_id[confirmation.sample_id]
            if loaded.nontruth_input_sha256 != frozen["nontruth_input_sha256"]:
                raise parent_runtime.BindingDriftError(
                    f"nontruth input hash drift before parent generation: {confirmation.sample_id}"
                )
            predicted = generate_parent_full401(
                model,
                normalizer,
                loaded,
                manifest,
                device=device,
                time_block=time_block,
            )
            truth, truth_hash = load_train_truth(confirmation, device=device)
            _verify_record_hashes(confirmation, frozen, loaded, truth_hash)
            observed = onset_indices(
                time_axis,
                t0_s=float(loaded.source_parameters[3]),
                f0_hz=float(loaded.source_parameters[2]),
            )
            if list(observed) != frozen["observed_indices"]:
                raise parent_runtime.BindingDriftError(
                    f"registered onset binding drift: {confirmation.sample_id}"
                )
            metrics = oracle_projection_metrics(
                predicted,
                truth,
                eigenvectors,
                future_start=int(observed[1] + 1),
                ranks=RANKS,
            )
            family_results[family] = {
                "basis_records": basis_rows,
                "covariance_shape": [TIME_COUNT, TIME_COUNT],
                "covariance_dtype": "float32",
                "covariance_trace": float(covariance.diagonal().double().sum().item()),
                "eigendecomposition_dtype": "float64",
                "basis_covariance_energy_capture": basis_captures,
                "confirmation_record": {
                    "sample_id": confirmation.sample_id,
                    "group_id": confirmation.group_id,
                    "manifest_sample_sha256": confirmation.manifest_sample_sha256,
                    "nontruth_input_sha256": loaded.nontruth_input_sha256,
                    "train_truth_sha256": truth_hash,
                    "observed_indices": list(observed),
                    "future_start_index": int(observed[1] + 1),
                    **metrics,
                },
            }
            record_execution.append(
                {
                    "sample_id": confirmation.sample_id,
                    "family": family,
                    "role": "disjoint_confirmation",
                }
            )
            del truth, predicted, covariance
            budget.check(f"after_{family}_confirmation")

        torch.cuda.synchronize(device)
        elapsed = budget.check("final_aggregation")
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        finite_output = all(
            math.isfinite(float(value))
            for family in FAMILIES
            for value in (
                family_results[family]["confirmation_record"][
                    "parent_mean_per_frame_relative_l2"
                ],
                family_results[family]["confirmation_record"]["ranks"]["32"][
                    "corrected_mean_per_frame_relative_l2"
                ],
                family_results[family]["confirmation_record"]["ranks"]["32"][
                    "relative_error_reduction"
                ],
                family_results[family]["confirmation_record"]["ranks"]["32"][
                    "residual_energy_capture"
                ],
            )
        )
        family_gates = {}
        for family in FAMILIES:
            rank32 = family_results[family]["confirmation_record"]["ranks"]["32"]
            family_gates[family] = {
                "relative_error_reduction": {
                    "value": float(rank32["relative_error_reduction"]),
                    "minimum": RANK32_ERROR_REDUCTION_MINIMUM,
                    "passed": float(rank32["relative_error_reduction"])
                    >= RANK32_ERROR_REDUCTION_MINIMUM,
                },
                "residual_energy_capture": {
                    "value": float(rank32["residual_energy_capture"]),
                    "minimum": RANK32_RESIDUAL_CAPTURE_MINIMUM,
                    "passed": float(rank32["residual_energy_capture"])
                    >= RANK32_RESIDUAL_CAPTURE_MINIMUM,
                },
            }
            family_gates[family]["joint_passed"] = all(
                bool(item["passed"])
                for key, item in family_gates[family].items()
                if key != "joint_passed"
            )
        memory_gate = {
            "value_bytes": peak_reserved,
            "maximum_bytes": PEAK_CUDA_BYTES_MAXIMUM,
            "passed": peak_reserved <= PEAK_CUDA_BYTES_MAXIMUM,
            "metric": "peak_cuda_reserved",
        }
        joint_pass = (
            finite_output
            and memory_gate["passed"]
            and all(bool(family_gates[family]["joint_passed"]) for family in FAMILIES)
        )
        checkpoint_after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
        if checkpoint_after != parent_runtime.CHECKPOINT_SHA256:
            raise parent_runtime.BindingDriftError("parent checkpoint changed during probe")
        payload = {
            **base,
            "status": "success",
            "decision": "pass" if joint_pass else "fail_gate",
            "completed_utc": utc_now(),
            "exact_blocker": "none" if joint_pass else "one or more preregistered family, finite-output, or peak-memory gates failed",
            "protocol": prereg["protocol"],
            "truth_scope": prereg["truth_scope"],
            "disk_free_bytes_at_binding_gate": disk_gate,
            "disk_free_bytes_after": parent_runtime.free_disk_bytes(),
            "budget": {
                "one_gpu_seconds_used": elapsed,
                "one_gpu_seconds_maximum": GPU_SECONDS_MAXIMUM,
                "one_gpu_hours_maximum": 0.10,
                "within_budget": elapsed <= GPU_SECONDS_MAXIMUM,
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
                "by_family": dict(Counter(row["family"] for row in record_execution)),
                "roles": dict(Counter(row["role"] for row in record_execution)),
                "execution": record_execution,
            },
            "family_results": family_results,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "gate": {
                "families": family_gates,
                "finite_output": {"value": finite_output, "required": True, "passed": finite_output},
                "peak_cuda_reserved": memory_gate,
                "joint_passed": joint_pass,
            },
        }
    except (parent_runtime.BindingDriftError, parent_runtime.DiskRiskError, FileNotFoundError) as error:
        payload = {
            **base,
            "status": "blocked",
            "decision": "not_run_or_incomplete",
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
            "decision": "incomplete",
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
            raise CapacityContractError("preregistration requires smoke evidence JSON")
        smoke = json.loads(args.smoke_evidence_json)
        payload = build_preregistration(
            physical_gpu_index=args.physical_gpu_index,
            device=device,
            time_block=args.time_block,
            smoke_evidence=smoke,
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
    if payload.get("status") in {"passed", "frozen", "success"}:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CapacityContractError",
    "FAMILIES",
    "GPU_SECONDS_MAXIMUM",
    "MAX_OUTPUT_BYTES",
    "MIN_FREE_BYTES",
    "PEAK_CUDA_BYTES_MAXIMUM",
    "RANKS",
    "RANK32_ERROR_REDUCTION_MINIMUM",
    "RANK32_RESIDUAL_CAPTURE_MINIMUM",
    "TIME_BLOCK",
    "TruthScopeError",
    "mean_per_frame_relative_l2",
    "oracle_projection_metrics",
    "sample_census",
    "stream_accumulate_temporal_covariance",
    "stream_train_truth_sha256",
    "temporal_pod",
]
