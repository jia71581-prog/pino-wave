#!/usr/bin/env python3
"""No-truth runtime/identity gate for a fresh R26 head anchored to R6."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if value not in sys.path:
        sys.path.insert(0, value)

from gate_lwc84_cuda_graph_fine_grid_trainonly import _fine_velocity, _read_manifest_rows  # noqa: E402
from audit_target5 import audit as audit_target5  # noqa: E402
from reattest_frozen_fine_grid_r6_train import (  # noqa: E402
    atomic_write_bytes,
    report_bytes_fixed_point,
    sha256,
)
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig  # noqa: E402
from fno_acoustic.data_generation.restriction import restrict_nodal_2x  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84_fused import FusedLWC84CPMLSolver  # noqa: E402
from fno_acoustic.data_generation.source import bilinear_point_source  # noqa: E402


CANDIDATE = "r6_anchored_r54_residual_r1_20260905"
SELECTION_SCHEMA = "frozen_fine_grid_r6_current_environment_reattest_selection_v1"
SELECTION_NAMESPACE = "R6_current_environment_reattest_v1"
FAMILIES = ("uniform", "layered", "marmousi")
SMOKE_INDICES = (311, 780, 2165)
LWC_QMAX_LIMIT = 9.6
LWC_QMAX_LIMIT_RELATIVE_TOLERANCE = 1.0e-12
LWC_RELATION_RELATIVE_TOLERANCE = 1.0e-6
LWC_RELATION_ABSOLUTE_TOLERANCE = 1.0e-9
CHUNKS = tuple((start, min(start + 16, 401)) for start in range(0, 401, 16))
EXPECTED_SELECTION_DIGESTS = {
    "indices": "fd5cac5d541dd65bbe15dbe79e4334992492346ff4e5eba9e9ea0869b2241664",
    "sample_ids": "024fb03cdb2f685c2a52b8cbbdb603fafdf89180723b31e2801144913bff43a8",
    "groups_first_occurrence": "08172ad3cb404858412ee708337eb72eb0b30b9cf1821cbc12774253f7141e3d",
    "sample_hashes": "cced838665a127c477d304e54d453fd230b3d62022baf675e49f54b1bb031b49",
    "records": "2083c95b9742c9eb4b0c42efb2f340377a8a554016fa2153c562d01c1b342b98",
}
ALLOWED_DATASETS = {
    "time_s", "split", "medium_type", "sample_id", "group_id", "sample_sha256",
    "velocity_mps", "source_x_m", "source_z_m", "source_f0_hz", "source_t0_s",
    "source_amplitude",
}


class AllowlistedHDF:
    """Allow public deployment data only and audit both allowed and denied reads."""

    def __init__(self, handle: h5py.File) -> None:
        self.handle = handle
        self.access_counts: dict[str, int] = {}
        self.denied_counts: dict[str, int] = {}

    def __getitem__(self, key: str):
        name = str(key)
        if name not in ALLOWED_DATASETS:
            self.denied_counts[name] = self.denied_counts.get(name, 0) + 1
            raise PermissionError(f"HDF dataset denied: {name}")
        self.access_counts[name] = self.access_counts.get(name, 0) + 1
        return self.handle[name]

    def ledger(self) -> dict[str, Any]:
        return {
            "allowed_datasets": sorted(ALLOWED_DATASETS),
            "access_counts": dict(sorted(self.access_counts.items())),
            "denied_attempts": dict(sorted(self.denied_counts.items())),
            "denied_attempt_count": int(sum(self.denied_counts.values())),
            "wavefield_access_count": 0,
            "future_truth_accessed": False,
        }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _first_occurrence(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def selection_digests(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        "indices": canonical_sha256([int(row["source_index"]) for row in records]),
        "sample_ids": canonical_sha256([str(row["sample_id"]) for row in records]),
        "groups_first_occurrence": canonical_sha256(
            _first_occurrence([str(row["group_id"]) for row in records])
        ),
        "sample_hashes": canonical_sha256([str(row["sample_sha256"]) for row in records]),
        "records": canonical_sha256(list(records)),
    }


def validate_selection_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema") != SELECTION_SCHEMA or manifest.get("namespace") != SELECTION_NAMESPACE:
        raise RuntimeError("R6 selection schema/namespace drift")
    if manifest.get("split") != "train" or manifest.get("fresh") is not False:
        raise RuntimeError("R6 selection exposure/split drift")
    if manifest.get("independent_confirmation") is not False:
        raise RuntimeError("R6 selection independent-confirmation drift")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 60:
        raise RuntimeError("R6 selection record count drift")
    expected_fields = {"source_index", "sample_id", "group_id", "family", "sample_sha256"}
    if any(set(row) != expected_fields for row in records):
        raise RuntimeError("R6 selection record schema drift")
    if selection_digests(records) != EXPECTED_SELECTION_DIGESTS:
        raise RuntimeError("R6 selection canonical digest drift")
    if manifest.get("digests") != EXPECTED_SELECTION_DIGESTS:
        raise RuntimeError("R6 selection declared digest drift")
    counts = {family: sum(row["family"] == family for row in records) for family in FAMILIES}
    groups = len({str(row["group_id"]) for row in records})
    indices = [int(row["source_index"]) for row in records]
    if counts != {family: 20 for family in FAMILIES} or groups != 29:
        raise RuntimeError("R6 selection family/group census drift")
    if tuple(indices[position] for position in (0, 20, 40)) != SMOKE_INDICES:
        raise RuntimeError("R6 smoke indices drift")
    if len(set(indices)) != 60:
        raise RuntimeError("R6 selection source indices are not unique")
    return [dict(row) for row in records]


def parameter_manifest(model: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "numel": int(parameter.numel()),
        }
        for name, parameter in model.named_parameters()
    ]


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical_bytes({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fresh_head_cpu(seed: int = 540829):
    r25 = importlib.import_module("scripts.train_r25_coarse_residual_operator")
    r26 = importlib.import_module("scripts.train_r26_tail_spectral_pilot")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = r26.TailSpectralResidualUNet(base_width=32, correction_cap=0.25).cpu().eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != 2791762 or r25.INPUT_CHANNELS != 13 or model.spectral_cap != 0.12:
        raise RuntimeError("fresh head architecture drift")
    manifest = hashlib.sha256(canonical_bytes(parameter_manifest(model))).hexdigest()
    return model, r25, state_digest(model), manifest


def validate_zero_head(model: torch.nn.Module) -> dict[str, Any]:
    tensors = {
        "local_weight": model.output.weight.detach(),
        "local_bias": model.output.bias.detach(),
        "spectral_weight": model.spectral_output.weight.detach(),
        "spectral_bias": model.spectral_output.bias.detach(),
    }
    passed = all(torch.count_nonzero(value).item() == 0 for value in tensors.values())
    return {
        "passed": passed,
        "zero_tensor_count": sum(torch.count_nonzero(value).item() == 0 for value in tensors.values()),
        "tensor_count": 4,
    }


def validate_chunks(chunks: Sequence[tuple[int, int]] = CHUNKS) -> None:
    flattened = [index for start, stop in chunks for index in range(start, stop)]
    if len(chunks) != 26 or flattened != list(range(401)) or chunks[-1] != (400, 401):
        raise RuntimeError("26-chunk coverage drift")


def validate_time_axis(times: np.ndarray) -> dict[str, Any]:
    if not isinstance(times, np.ndarray) or times.dtype != np.dtype(np.float64):
        raise RuntimeError("time_s native dtype drift")
    differences = np.diff(times)
    passed = (
        times.shape == (401,)
        and float(times[0]) == 0.0
        and float(times[-1]) == 1.0
        and np.allclose(differences, 0.0025, rtol=0.0, atol=1.0e-15)
    )
    if not passed:
        raise RuntimeError("time_s exact contract failure")
    return {
        "passed": True, "shape": [401], "first_s": 0.0, "last_s": 1.0,
        "expected_difference_s": 0.0025,
        "maximum_difference_error_s": float(np.max(np.abs(differences - 0.0025))),
    }


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def preload_public_record(proxy: AllowlistedHDF, row: Mapping[str, Any], manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    index = int(row["source_index"])
    observed = {
        "split": _decode(proxy["split"][index]),
        "family": _decode(proxy["medium_type"][index]),
        "sample_id": _decode(proxy["sample_id"][index]),
        "group_id": _decode(proxy["group_id"][index]),
        "sample_sha256": _decode(proxy["sample_sha256"][index]),
    }
    expected = {
        "split": "train", "family": str(row["family"]), "sample_id": str(row["sample_id"]),
        "group_id": str(row["group_id"]), "sample_sha256": str(row["sample_sha256"]),
    }
    if observed != expected:
        raise RuntimeError("HDF public metadata differs from frozen R6 selection")
    manifest_identity = {
        "split": str(manifest_row.get("split")), "family": str(manifest_row.get("medium_type")),
        "sample_id": str(manifest_row.get("sample_id")), "group_id": str(manifest_row.get("group_id")),
    }
    if manifest_identity != {key: expected[key] for key in ("split", "family", "sample_id", "group_id")}:
        raise RuntimeError("source manifest row identity drift")
    metadata = {
        key: float(proxy[key][index])
        for key in ("source_x_m", "source_z_m", "source_f0_hz", "source_t0_s", "source_amplitude")
    }
    for key, value in metadata.items():
        if not math.isfinite(value) or float(manifest_row[key]) != value:
            raise RuntimeError(f"source manifest/HDF numeric identity drift: {key}")
    stored = np.asarray(proxy["velocity_mps"][index])
    if stored.dtype != np.dtype(np.float32) or stored.shape != (201, 201) or not stored.flags.c_contiguous or not np.isfinite(stored).all():
        raise RuntimeError("stored public velocity native QC failure")
    return {"row": dict(row), "manifest_row": dict(manifest_row), "metadata": metadata,
            "stored_velocity": stored, "public_identity": observed}


def parent_stability_qc(cfl_2d: float, lwc_qmax: float) -> dict[str, Any]:
    cfl = float(cfl_2d)
    qmax = float(lwc_qmax)
    relation_expected = (2048.0 / 315.0) * cfl**2
    finite_nonnegative = bool(math.isfinite(cfl) and math.isfinite(qmax) and cfl >= 0.0 and qmax >= 0.0)
    qmax_within_limit = bool(
        finite_nonnegative
        and qmax <= LWC_QMAX_LIMIT * (1.0 + LWC_QMAX_LIMIT_RELATIVE_TOLERANCE)
    )
    relation_passed = bool(
        finite_nonnegative
        and math.isclose(
            qmax, relation_expected,
            rel_tol=LWC_RELATION_RELATIVE_TOLERANCE,
            abs_tol=LWC_RELATION_ABSOLUTE_TOLERANCE,
        )
    )
    return {
        "cfl_2d": cfl, "cfl_is_gate": False, "cfl_finite_nonnegative": bool(math.isfinite(cfl) and cfl >= 0.0),
        "lwc_qmax": qmax, "lwc_qmax_limit": LWC_QMAX_LIMIT,
        "lwc_qmax_limit_relative_tolerance": LWC_QMAX_LIMIT_RELATIVE_TOLERANCE,
        "lwc_qmax_within_limit": qmax_within_limit,
        "relation_formula": "lwc_qmax ~= (2048/315) * cfl_2d^2",
        "relation_expected": relation_expected,
        "relation_relative_tolerance": LWC_RELATION_RELATIVE_TOLERANCE,
        "relation_absolute_tolerance": LWC_RELATION_ABSOLUTE_TOLERANCE,
        "relation_passed": relation_passed,
        "passed": bool(finite_nonnegative and qmax_within_limit and relation_passed),
    }


def validate_parent_result(result: Any, *, fine_velocity: np.ndarray, stored_velocity: np.ndarray,
                           metadata: Mapping[str, float], observation_identity: Mapping[str, Any] | None = None,
                           partial_observations: list[dict[str, Any]] | None = None
                           ) -> tuple[np.ndarray, dict[str, Any]]:
    wavefield = result.wavefield
    if (not isinstance(wavefield, np.ndarray) or wavefield.dtype != np.dtype(np.float32)
            or wavefield.shape != (1, 401, 201, 201) or not wavefield.flags.c_contiguous
            or not np.isfinite(wavefield).all()):
        raise RuntimeError("R6 parent batch native QC failure")
    parent = wavefield[0]
    if not parent.flags.c_contiguous or not np.array_equal(parent[:, 0, :], np.zeros((401, 201), np.float32)):
        raise RuntimeError("R6 parent top/contiguity QC failure")
    restricted = restrict_nodal_2x(fine_velocity)
    restriction_difference = float(np.max(np.abs(restricted - stored_velocity)))
    if restriction_difference != 0.0:
        raise RuntimeError("fine-to-stored restriction is not exact")
    point = bilinear_point_source(metadata["source_x_m"], metadata["source_z_m"], nx=401, nz=401,
                                  dx_m=5.0, dz_m=5.0, centering="node")
    source_saved_sum = float(result.source_map_saved[0].sum(dtype=np.float64))
    source_solver_sum = float(result.source_map_solver[0].sum(dtype=np.float64))
    delta_integral = float(point.delta_h.sum(dtype=np.float64) * 25.0)
    if (abs(source_saved_sum - 1.0) > 2.0e-7 or abs(source_solver_sum - 1.0) > 2.0e-7
            or abs(delta_integral - 1.0) > 2.0e-7):
        raise RuntimeError("source map/delta integral QC failure")
    if len(result.metrics) != 1:
        raise RuntimeError("solver metric batch count drift")
    metrics = result.metrics[0]
    cfl, qmax, dt_used = float(metrics["cfl_2d"]), float(metrics["lwc_qmax"]), float(metrics["dt_used_s"])
    stability = parent_stability_qc(cfl, qmax)
    if partial_observations is not None:
        partial_observations.append({
            **dict(observation_identity or {}), "stability_qc": dict(stability),
            "observation_stage": "parent_stability_before_identity",
        })
    if not stability["passed"]:
        raise RuntimeError("R6 parent stability QC failure")
    if dt_used != 0.000625 or parent.shape[0] != 401:
        raise RuntimeError("R6 solver dt/output-count drift")
    return parent, {
        "passed": True, "parent_native_shape": [401, 201, 201], "parent_native_dtype": "float32",
        "parent_native_C_contiguous": True, "parent_finite": True, "parent_top_row_exact_zero": True,
        "restriction_max_absolute_difference": restriction_difference, "source_map_saved_sum": source_saved_sum,
        "source_map_solver_sum": source_solver_sum, "delta_solver_integral": delta_integral,
        "cfl_2d": cfl, "lwc_qmax": qmax, "stability_qc": stability, "dt_used_s": dt_used,
        "output_frame_count": 401, "internal_step_count_to_1s": 1600,
    }


def bit_preserving_residual_add(parent: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    """Inference-only residual add preserving parent bits wherever correction is ±0."""
    if not isinstance(parent, torch.Tensor) or not isinstance(correction, torch.Tensor):
        raise TypeError("parent and correction must be torch tensors")
    if parent.shape != correction.shape:
        raise RuntimeError("residual add shape mismatch")
    if parent.dtype != correction.dtype:
        raise RuntimeError("residual add dtype mismatch")
    if parent.device != correction.device:
        raise RuntimeError("residual add device mismatch")
    if not torch.is_floating_point(parent):
        raise RuntimeError("residual add requires floating tensors")
    torch._assert_async(torch.isfinite(parent).all(), "residual add parent must be finite")
    torch._assert_async(torch.isfinite(correction).all(), "residual add correction must be finite")
    summed = parent + correction
    return torch.where(correction == 0, parent, summed)


def residual_add_identity_metrics(parent: np.ndarray, correction: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    zero_mask = correction == 0
    nonzero_mask = ~zero_mask
    parent_bits = parent.view(np.uint32)
    correction_bits = correction.view(np.uint32)
    candidate_bits = candidate.view(np.uint32)
    ordinary = np.add(parent, correction, dtype=np.float32)
    ordinary_bits = ordinary.view(np.uint32)
    byte_mismatch_count = int(np.count_nonzero(candidate_bits != parent_bits))
    numeric_mismatch_count = int(np.count_nonzero(candidate != parent))
    zero_position_count = int(np.count_nonzero(zero_mask))
    zero_position_preserved_count = int(np.count_nonzero(candidate_bits[zero_mask] == parent_bits[zero_mask]))
    nonzero_position_count = int(np.count_nonzero(nonzero_mask))
    nonzero_numeric_equal = bool(np.array_equal(candidate[nonzero_mask], ordinary[nonzero_mask]))
    nonzero_bit_equal = bool(np.array_equal(candidate_bits[nonzero_mask], ordinary_bits[nonzero_mask]))
    correction_all_zero = bool(nonzero_position_count == 0)
    return {
        "parent_signed_zero_count": int(np.count_nonzero((parent == 0) & np.signbit(parent))),
        "correction_signed_zero_count": int(np.count_nonzero(zero_mask & np.signbit(correction))),
        "candidate_signed_zero_count": int(np.count_nonzero((candidate == 0) & np.signbit(candidate))),
        "numeric_mismatch_count": numeric_mismatch_count,
        "byte_mismatch_count": byte_mismatch_count,
        "zero_position_count": zero_position_count,
        "zero_position_bit_preserved_count": zero_position_preserved_count,
        "zero_position_bits_preserved": bool(zero_position_preserved_count == zero_position_count),
        "nonzero_position_count": nonzero_position_count,
        "nonzero_additive_numeric_equal": nonzero_numeric_equal,
        "nonzero_additive_bit_equal": nonzero_bit_equal,
        "correction_all_zero": correction_all_zero,
        "full_identity": bool(correction_all_zero and byte_mismatch_count == 0),
    }


def output_identity(parent: np.ndarray, correction: np.ndarray, candidate: np.ndarray, *, t0: float,
                    times: np.ndarray) -> dict[str, Any]:
    for name, value in (("parent", parent), ("correction", correction), ("candidate", candidate)):
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32)
                or value.shape != (401, 201, 201) or not value.flags.c_contiguous
                or not np.isfinite(value).all()):
            raise RuntimeError(f"{name} native QC failure")
    active = times >= float(t0)
    if (np.count_nonzero(correction) != 0 or np.count_nonzero(correction[~active]) != 0
            or np.count_nonzero(correction[:, 0, :]) != 0):
        raise RuntimeError("fresh head correction is not exact zero")
    metrics = residual_add_identity_metrics(parent, correction, candidate)
    if not metrics["full_identity"] or candidate.tobytes() != parent.tobytes():
        raise RuntimeError("candidate is not byte-identical to parent")
    return {
        "parent_sha256": hashlib.sha256(parent.tobytes()).hexdigest(),
        "correction_sha256": hashlib.sha256(correction.tobytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
        **metrics,
        "candidate_equals_parent_bytes": True,
        "correction_preonset_zero": True, "correction_top_zero": True,
    }


def inference_head(model: torch.nn.Module, r25: Any, parent: np.ndarray, stored_velocity: np.ndarray,
                   times: np.ndarray, metadata: Mapping[str, float], *, device: torch.device
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    validate_chunks()
    static_builder = importlib.import_module("scripts.build_r25_coarse_residual_cache")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    x = np.linspace(0.0, 2000.0, 201, dtype=np.float32)
    static = static_builder.static_features(stored_velocity, x_m=x, z_m=x,
                                             source_x_m=metadata["source_x_m"], source_z_m=metadata["source_z_m"])
    scale = max(float(np.max(np.abs(parent))), 1.0e-12)
    static_gpu = torch.from_numpy(static).to(device)
    corrections: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for start, stop in CHUNKS:
            parent_raw_gpu = torch.from_numpy(parent[start:stop]).to(device)
            time_gpu = torch.from_numpy(times[start:stop].astype(np.float32)).to(device)
            block = stop - start
            features = r25.make_dynamic_features(
                parent_raw_gpu / scale, static_gpu[None].expand(block, -1, -1, -1), time_s=time_gpu,
                source_f0_hz=torch.full((block,), metadata["source_f0_hz"], device=device),
                source_t0_s=torch.full((block,), metadata["source_t0_s"], device=device),
            )
            if features.shape[1] != 13:
                raise RuntimeError("dynamic feature channel/order drift")
            active = (time_gpu >= metadata["source_t0_s"]).float()
            correction_gpu = model(features, active=active).float() * scale
            candidate_gpu = bit_preserving_residual_add(parent_raw_gpu, correction_gpu)
            corrections.append(correction_gpu.cpu().numpy())
            candidates.append(candidate_gpu.cpu().numpy())
    correction = np.ascontiguousarray(np.concatenate(corrections), dtype=np.float32)
    candidate = np.ascontiguousarray(np.concatenate(candidates), dtype=np.float32)
    torch.cuda.synchronize(device)
    return correction, candidate, float(time.perf_counter() - started)


def execute_repeat(*, prepared: Mapping[str, Any], grid: AcousticGrid, solver: FusedLWC84CPMLSolver,
                   model: torch.nn.Module, r25: Any, times: np.ndarray, marmousi: Path,
                   device: torch.device, partial_observations: list[dict[str, Any]] | None = None
                   ) -> dict[str, Any]:
    """Rebuild the complete public-input pipeline once; callers may discard warmup output."""
    torch.cuda.synchronize(device)
    outer_started = time.perf_counter()
    fine_velocity, fine_metadata = _fine_velocity(family=prepared["row"]["family"],
                                                   manifest_row=prepared["manifest_row"], grid=grid,
                                                   marmousi_npy=marmousi)
    result = solver.simulate(fine_velocity, **prepared["metadata"])
    parent, parent_qc = validate_parent_result(result, fine_velocity=fine_velocity,
                                                stored_velocity=prepared["stored_velocity"],
                                                metadata=prepared["metadata"],
                                                observation_identity={
                                                    "source_index": int(prepared["row"]["source_index"]),
                                                    "sample_id": str(prepared["row"]["sample_id"]),
                                                    "family": str(prepared["row"]["family"]),
                                                }, partial_observations=partial_observations)
    correction, candidate, head_runtime = inference_head(model, r25, parent, prepared["stored_velocity"],
                                                          times, prepared["metadata"], device=device)
    identity = output_identity(parent, correction, candidate,
                               t0=prepared["metadata"]["source_t0_s"], times=times)
    torch.cuda.synchronize(device)
    outer_runtime = float(time.perf_counter() - outer_started)
    del fine_velocity, result, parent, correction, candidate
    return {"identity": identity, "parent_qc": parent_qc, "fine_velocity_metadata": fine_metadata,
            "outer_runtime_s": outer_runtime, "head_runtime_s": float(head_runtime)}


def nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[max(1, math.ceil(probability * len(ordered))) - 1]


def runtime_gates(outer: Sequence[float], head: Sequence[float], *, smoke: bool) -> dict[str, Any]:
    if smoke:
        if len(outer) != 3 or len(head) != 3:
            raise RuntimeError("smoke runtime count drift")
        return {"applied": False, "raw_outer_count": 3, "raw_head_count": 3}
    if len(outer) != 180 or len(head) != 180:
        raise RuntimeError("full runtime count drift")
    metrics = {"outer_mean": float(np.mean(outer)), "outer_p95": nearest_rank(outer, 0.95),
               "outer_p95_rank": 171, "head_p95": nearest_rank(head, 0.95),
               "raw_outer_count": 180, "raw_head_count": 180}
    gates = {"outer_mean_le_1_90": metrics["outer_mean"] <= 1.90,
             "outer_p95_le_1_95": metrics["outer_p95"] <= 1.95,
             "outer_mean_le_target5": metrics["outer_mean"] <= 2.034137312322855,
             "outer_p95_le_target5": metrics["outer_p95"] <= 2.034137312322855,
             "head_p95_le_1_10": metrics["head_p95"] <= 1.10}
    return {"applied": True, "metrics": metrics, "gates": gates, "passed": all(gates.values())}


def collect_algorithm_flags() -> dict[str, bool]:
    return {
        "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def configure_algorithm_flags(expected_environment: Mapping[str, Any]) -> dict[str, Any]:
    """Disable only TF32, while requiring the frozen non-deterministic defaults."""
    keys = ("deterministic", "cudnn_benchmark", "cudnn_deterministic", "tf32_matmul", "tf32_cudnn")
    expected = {key: expected_environment.get(key) for key in keys}
    if expected != {
        "deterministic": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
    }:
        raise RuntimeError("preregistered algorithm flag contract drift")
    observed_before = collect_algorithm_flags()
    if any(observed_before[key] for key in ("deterministic", "cudnn_benchmark", "cudnn_deterministic")):
        raise RuntimeError("non-TF32 algorithm defaults were not preserved as false")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configured_after = collect_algorithm_flags()
    if configured_after != expected:
        raise RuntimeError("configured algorithm flags differ from preregistration")
    return {"observed_before": observed_before, "configured_after": configured_after, "expected": expected}


def target5_preflight(prereg: Mapping[str, Any], *, audit_function: Any = audit_target5) -> dict[str, Any]:
    result = audit_function(ROOT, _resolved(prereg["paths"]["reference"]))
    if result.get("passed") is not True or result.get("status") != "target_met":
        raise RuntimeError("Target-5 project audit did not pass")
    return dict(result)


def validate_runtime_completeness(records: Sequence[Mapping[str, Any]], *, mode: str) -> dict[str, Any]:
    expected_records = 3 if mode == "smoke" else 60
    expected_identities = 3 if mode == "smoke" else 180
    if len(records) != expected_records:
        raise RuntimeError("runtime record count drift")
    identity_count = sum(len(row["repeats"]) for row in records)
    if identity_count != expected_identities:
        raise RuntimeError("runtime identity count drift")
    expected_repeats = 1 if mode == "smoke" else 3
    for row in records:
        repeats = row["repeats"]
        if len(repeats) != expected_repeats:
            raise RuntimeError("per-record repeat count drift")
        for key in ("parent_sha256", "correction_sha256", "candidate_sha256"):
            if len({repeat["identity"][key] for repeat in repeats}) != 1:
                raise RuntimeError(f"per-record repeat {key} drift")
        if not all(repeat["parent_qc"]["passed"] for repeat in repeats):
            raise RuntimeError("per-repeat parent QC failure")
    return {"passed": True, "record_count": expected_records, "identity_count": expected_identities,
            "parent_hash_count": expected_identities, "correction_hash_count": expected_identities,
            "candidate_hash_count": expected_identities, "raw_outer_runtime_count": expected_identities,
            "raw_head_runtime_count": expected_identities, "repeats_per_record": expected_repeats,
            "repeat_hashes_consistent": True}


def environment_contract() -> dict[str, Any]:
    query = __import__("subprocess").run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                                            check=True, capture_output=True, text=True).stdout.splitlines()[0]
    driver, gpu = [value.strip() for value in query.split(",", 1)]
    return {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "python": platform.python_version(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__, "h5py": h5py.__version__, "scipy": scipy.__version__, "driver": driver,
            "gpu": gpu, "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}


def _resolved(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _local_module_path(module: str) -> Path | None:
    candidates: list[Path] = []
    if "." not in module:
        candidates.append(ROOT / "scripts" / module)
    if module == "scripts" or module.startswith("scripts."):
        candidates.append(ROOT / "scripts" / Path(*module.split(".")[1:]))
    if module == "fno_acoustic" or module.startswith("fno_acoustic."):
        candidates.append(ROOT / "src/fno_acoustic" / Path(*module.split(".")[1:]))
    if module == "saved_time_phase_operator_v4" or module.startswith("saved_time_phase_operator_v4."):
        candidates.append(ROOT / "saved_time_phase_operator_v4" / Path(*module.split(".")[1:]))
    if module == "grouped_ufno_mionet_v3" or module.startswith("grouped_ufno_mionet_v3."):
        candidates.append(ROOT / "grouped_ufno_mionet_v3" / Path(*module.split(".")[1:]))
    for base in candidates:
        file_path = base.with_suffix(".py")
        package_path = base / "__init__.py"
        if file_path.is_file():
            return file_path.resolve()
        if package_path.is_file():
            return package_path.resolve()
    return None


def _local_module_name(path: Path) -> str | None:
    path = path.resolve()
    for base, prefix in ((ROOT / "scripts", "scripts"), (ROOT / "src/fno_acoustic", "fno_acoustic"),
                         (ROOT / "saved_time_phase_operator_v4", "saved_time_phase_operator_v4"),
                         (ROOT / "grouped_ufno_mionet_v3", "grouped_ufno_mionet_v3")):
        try:
            relative = path.relative_to(base)
        except ValueError:
            continue
        parts = list(relative.with_suffix("").parts)
        return ".".join((prefix, *parts))
    return None


def _local_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current = _local_module_name(path)
    result: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level and current:
                package = current.split(".")[:-1]
                keep = len(package) - node.level + 1
                prefix = package[:max(keep, 0)]
                base = ".".join([*prefix, base] if base else prefix)
            if base:
                modules.append(base)
                modules.extend(f"{base}.{alias.name}" for alias in node.names)
        for module in modules:
            dependency = _local_module_path(module)
            if dependency is not None:
                result.add(dependency)
    return result


def runtime_dependency_closure(roots: Sequence[Path]) -> list[Path]:
    pending = [path.resolve() for path in roots]
    closure: set[Path] = set()
    permitted = ((ROOT / "scripts").resolve(), (ROOT / "src/fno_acoustic").resolve(),
                 (ROOT / "saved_time_phase_operator_v4").resolve(),
                 (ROOT / "grouped_ufno_mionet_v3").resolve())
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        if not path.is_file() or not any(path == base or base in path.parents for base in permitted):
            raise RuntimeError(f"dependency outside local runtime roots: {path}")
        closure.add(path)
        pending.extend(sorted(_local_imports(path)))
        for base in permitted:
            if path == base or base in path.parents:
                parent = path.parent
                while parent == base or base in parent.parents:
                    initializer = parent / "__init__.py"
                    if initializer.is_file():
                        pending.append(initializer.resolve())
                    if parent == base:
                        break
                    parent = parent.parent
                break
    return sorted(closure, key=lambda path: path.relative_to(ROOT).as_posix())


def verify_dependency_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    closure = runtime_dependency_closure([_resolved(path) for path in manifest["roots"]])
    observed = {path.relative_to(ROOT).as_posix(): sha256(path) for path in closure}
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest["files"]}
    if observed != expected:
        raise RuntimeError("recursive runtime dependency closure drift")
    for row in manifest.get("explicit_bindings", []):
        path = _resolved(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"explicit dependency binding drift: {row['path']}")
    return observed


def bootstrap_canonical_output(args: argparse.Namespace, prereg: Mapping[str, Any]) -> tuple[str, Path]:
    if prereg.get("schema") != "r6_anchored_r54_runtime_preregistration_v1" or prereg.get("candidate") != CANDIDATE:
        raise RuntimeError("preregistration schema/candidate drift")
    mode = "smoke" if args.smoke else "full"
    if args.preregistration.resolve() != _resolved(prereg["paths"]["preregistration"]):
        raise RuntimeError("preregistration path override")
    output = _resolved(prereg["paths"][f"{mode}_output"])
    if args.output.resolve() != output:
        raise RuntimeError("canonical output override rejected")
    if output.exists():
        raise FileExistsError("fixed output exists")
    return mode, output


def validate_remaining_invocation(args: argparse.Namespace, prereg: Mapping[str, Any], mode: str) -> None:
    expected_status = "static_passed_smoke_pending_audit" if mode == "smoke" else "smoke_passed_full_pending_audit"
    if prereg.get("status") != expected_status:
        raise RuntimeError("stage status rejected")
    for key, supplied in {"source_h5": args.source_h5, "manifest": args.manifest, "marmousi": args.marmousi}.items():
        if supplied.resolve() != _resolved(prereg["paths"][key]):
            raise RuntimeError(f"fixed path override: {key}")
    if mode == "full":
        prerequisite = prereg["prerequisites"]["smoke"]
        prerequisite_path = _resolved(prerequisite["path"])
        if prerequisite.get("status") != "passed" or sha256(prerequisite_path) != prerequisite.get("sha256"):
            raise RuntimeError("smoke prerequisite is not frozen/passed")


def verify_all_inputs(prereg: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    dependency_path = _resolved(prereg["paths"]["dependency_manifest"])
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    closure = verify_dependency_manifest(dependency)
    paths = {name: _resolved(path) for name, path in prereg["binding_paths"].items()}
    explicit = {name: sha256(path) for name, path in paths.items()}
    if explicit != prereg["bindings"]:
        drift = sorted(name for name, value in explicit.items() if prereg["bindings"].get(name) != value)
        raise RuntimeError(f"explicit input binding drift: {drift}")
    return {"dependency_closure": closure, "explicit_bindings": explicit}


def sanitize_for_json(value: Any, *, path: str = "$", nonfinite: list[str] | None = None) -> tuple[Any, list[str]]:
    paths = [] if nonfinite is None else nonfinite
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(path)
        return None, paths
    if isinstance(value, Path):
        return str(value), paths
    if isinstance(value, Mapping):
        return {str(key): sanitize_for_json(item, path=f"{path}.{key}", nonfinite=paths)[0]
                for key, item in value.items()}, paths
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item, path=f"{path}[{index}]", nonfinite=paths)[0]
                for index, item in enumerate(value)], paths
    return value, paths


def minimal_failure_payload(*, report: Mapping[str, Any], error: Exception, started: float,
                            proxy: AllowlistedHDF | None) -> dict[str, Any]:
    safe = {"schema": report.get("schema"), "candidate": report.get("candidate"), "mode": report.get("mode"),
            "argv": report.get("argv"), "status": "invalid", "error": repr(error),
            "claim": "no-truth contract failure",
            "preregistration_sha256_before": report.get("preregistration_sha256_before"),
            "preregistration_sha256_after_if_available": report.get("preregistration_sha256_after"),
            "input_hashes_before": report.get("input_hashes_before"),
            "input_hashes_after_if_available": report.get("input_hashes_after"),
            "expected_environment": report.get("expected_environment"),
            "observed_environment": report.get("observed_environment"),
            "algorithm_flags": report.get("algorithm_flags"),
            "target5_audit": report.get("target5_audit"),
            "partial_runtime_observations": report.get("partial_runtime_observations"),
            "hdf_ledger": proxy.ledger() if proxy is not None else None,
            "resources": {"elapsed_seconds": float(time.time() - started),
                          "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0,
                          "output_bytes": 0},
            "weights_loaded": False, "cache_written": False, "predictions_persisted": False,
            "promotion_authorized": False, "validation_truth_opened": False, "test_id_truth_opened": False}
    sanitized, paths = sanitize_for_json(safe)
    sanitized["nonfinite_paths_sanitized_to_null"] = paths
    return sanitized


def write_failure_atomic(*, report: Mapping[str, Any], error: Exception, started: float,
                         proxy: AllowlistedHDF | None, output: Path) -> None:
    payload = minimal_failure_payload(report=report, error=error, started=started, proxy=proxy)
    try:
        data = report_bytes_fixed_point(payload)
        if len(data) >= 64 * 2**20:
            raise RuntimeError("failure report output budget")
        atomic_write_bytes(data, output)
    except Exception as serializer_error:
        fallback = {"schema": "r6_anchored_r54_runtime_report_v1", "candidate": CANDIDATE,
                    "status": "invalid", "error": "failure_report_serializer_error",
                    "serializer_error": repr(serializer_error), "no_truth": True,
                    "preregistration_sha256_before": (
                        report.get("preregistration_sha256_before")
                        if isinstance(report.get("preregistration_sha256_before"), str) else None
                    ),
                    "input_hashes_before": None,
                    "observed_environment": None,
                    "hdf_ledger": None,
                    "fallback_fields_unavailable_due_serializer_failure": [
                        "input_hashes_before", "observed_environment", "hdf_ledger"
                    ],
                    "resources": {"peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
                    ), "output_bytes": 0}, "promotion_authorized": False,
                    "validation_truth_opened": False, "test_id_truth_opened": False}
        for _ in range(16):
            data = (json.dumps(fallback, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            if fallback["resources"]["output_bytes"] == len(data):
                break
            fallback["resources"]["output_bytes"] = len(data)
        atomic_write_bytes(data, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--marmousi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    mode, output = bootstrap_canonical_output(args, prereg)
    started = time.time()
    proxy: AllowlistedHDF | None = None
    report: dict[str, Any] = {
        "schema": "r6_anchored_r54_runtime_report_v1", "candidate": CANDIDATE, "mode": mode,
        "status": "running", "argv": list(sys.argv),
        "claim": "R6-anchored R54/R26 residual runtime identity recipe; no component causality claim",
        "preregistration_sha256_before": sha256(args.preregistration),
        "expected_environment": None, "observed_environment": None, "algorithm_flags": None,
        "target5_audit": None, "partial_runtime_observations": [],
        "no_truth": True,
        "weights_loaded": False, "cache_written": False, "predictions_persisted": False,
        "promotion_authorized": False, "validation_truth_opened": False, "test_id_truth_opened": False,
    }
    try:
        validate_remaining_invocation(args, prereg, mode)
        report["expected_environment"] = dict(prereg["environment"])
        report["algorithm_flags"] = configure_algorithm_flags(prereg["environment"])
        observed = environment_contract()
        report["observed_environment"] = observed
        if observed != prereg["environment"]:
            raise RuntimeError("environment drift")
        inputs_before = verify_all_inputs(prereg)
        report["input_hashes_before"] = inputs_before
        report["target5_audit"] = target5_preflight(prereg)
        selection_manifest = json.loads(_resolved(prereg["paths"]["selection"]).read_text(encoding="utf-8"))
        selection = validate_selection_manifest(selection_manifest)
        active = [selection[0], selection[20], selection[40]] if mode == "smoke" else selection
        model, r25, initial_digest, parameter_digest = fresh_head_cpu()
        zero = validate_zero_head(model)
        if (initial_digest != prereg["fresh_head"]["initial_state_digest"]
                or parameter_digest != prereg["fresh_head"]["parameter_manifest_digest"] or not zero["passed"]):
            raise RuntimeError("fresh head identity drift")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        model = model.to(device).eval()
        grid = AcousticGrid(nx=401, nz=401, dx_m=5.0, dz_m=5.0, lx_m=2000.0, lz_m=2000.0,
                            centering="node")
        with h5py.File(args.source_h5, "r", swmr=True) as raw:
            proxy = AllowlistedHDF(raw)
            times = np.asarray(proxy["time_s"][:], dtype=np.float64)
            time_qc = validate_time_axis(times)
            manifest_rows = _read_manifest_rows(args.manifest, {row["sample_id"] for row in active})
            prepared = [preload_public_record(proxy, row, manifest_rows[row["sample_id"]]) for row in active]
            solver = FusedLWC84CPMLSolver(grid=grid, boundaries=BoundaryConfig(npml=40), dt_s=0.000625,
                                          output_times_s=times, c_ref_mps=6750.0, device=device,
                                          dtype=torch.float32, kappa_max=3.0, minimum_frequency_hz=8.0,
                                          output_restriction_factor=2, cuda_graphs=True)
            solver_warmup_seconds = float(solver.warmup(batch=1))
            complete_warmup = execute_repeat(prepared=prepared[0], grid=grid, solver=solver, model=model, r25=r25,
                                             times=times, marmousi=args.marmousi, device=device,
                                             partial_observations=report["partial_runtime_observations"])
            del complete_warmup
            records: list[dict[str, Any]] = []
            outer_all: list[float] = []
            head_all: list[float] = []
            for item in prepared:
                repeat_count = 1 if mode == "smoke" else 3
                repeats = [execute_repeat(prepared=item, grid=grid, solver=solver, model=model, r25=r25,
                                          times=times, marmousi=args.marmousi, device=device,
                                          partial_observations=report["partial_runtime_observations"])
                           for _ in range(repeat_count)]
                outer_all.extend(float(repeat["outer_runtime_s"]) for repeat in repeats)
                head_all.extend(float(repeat["head_runtime_s"]) for repeat in repeats)
                records.append({**item["row"], "public_identity": item["public_identity"], "repeats": repeats})
        completeness = validate_runtime_completeness(records, mode=mode)
        ledger = proxy.ledger()
        if ledger["denied_attempt_count"] != 0 or ledger["wavefield_access_count"] != 0:
            raise RuntimeError("HDF proxy ledger is not sealed")
        gates = runtime_gates(outer_all, head_all, smoke=mode == "smoke")
        status = ("smoke_complete" if mode == "smoke" else
                  ("runtime_identity_passed_cache_prereg_pending_audit" if gates["passed"]
                   else "runtime_identity_rejected"))
        inputs_after = verify_all_inputs(prereg)
        report["input_hashes_after"] = inputs_after
        algorithm_flags_after = collect_algorithm_flags()
        if algorithm_flags_after != report["algorithm_flags"]["configured_after"]:
            raise RuntimeError("algorithm flags changed during runtime gate")
        report["algorithm_flags"]["observed_after"] = algorithm_flags_after
        prereg_after = sha256(args.preregistration)
        report["preregistration_sha256_after"] = prereg_after
        if inputs_after != inputs_before or prereg_after != report["preregistration_sha256_before"]:
            raise RuntimeError("input/preregistration drift during runtime gate")
        resources = {"elapsed_seconds": float(time.time() - started),
                     "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "output_bytes": 0}
        wall_limit = 180.0 if mode == "smoke" else 900.0
        if resources["elapsed_seconds"] > wall_limit or resources["peak_allocated_bytes"] >= 8 * 2**30:
            raise RuntimeError("runtime resource budget exceeded")
        report.update({"status": status,
                       "claim": "implementation_only; no runtime acceptance claim" if mode == "smoke" else report["claim"],
                       "selection_validation": {"schema": SELECTION_SCHEMA, "record_count": 60, "group_count": 29,
                                                "family_counts": {family: 20 for family in FAMILIES},
                                                "smoke_indices": list(SMOKE_INDICES),
                                                "digests": EXPECTED_SELECTION_DIGESTS, "fresh": False,
                                                "independent_confirmation": False},
                       "record_count": len(records), "records": records,
                       "identity_completeness": completeness, "runtime_gate": gates, "time_axis_qc": time_qc,
                       "warmup": {"excluded_from_formal_timing": True,
                                  "solver_warmup_seconds": solver_warmup_seconds,
                                  "complete_pipeline_warmup_discarded": True,
                                  "solver_and_head_construction_excluded": True},
                       "timing_contract": {"outer_pre_and_post_cuda_synchronize": True,
                                           "head_pre_and_post_cuda_synchronize": True,
                                           "each_formal_repeat_rebuilds_fine_static_parent_correction_candidate": True,
                                           "only_public_metadata_stored_velocity_manifest_preloaded": True,
                                           "disk_and_warmup_excluded": True},
                       "fresh_head": {"parameter_count": 2791762, "initial_state_digest": initial_digest,
                                      "parameter_manifest_digest": parameter_digest, "zero_heads": zero,
                                      "weights_loaded": False},
                       "hdf_ledger": ledger, "resources": resources})
        data = report_bytes_fixed_point(report)
        if len(data) != report["resources"]["output_bytes"] or len(data) >= 64 * 2**20:
            raise RuntimeError("fixed-point JSON output budget failure")
        atomic_write_bytes(data, output)
        return 0 if status in {"smoke_complete", "runtime_identity_passed_cache_prereg_pending_audit"} else 2
    except Exception as error:
        try:
            report["preregistration_sha256_after"] = sha256(args.preregistration)
            report["input_hashes_after"] = verify_all_inputs(prereg)
        except Exception:
            pass
        write_failure_atomic(report=report, error=error, started=started, proxy=proxy, output=output)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
