#!/usr/bin/env python3
"""No-truth device-resident runtime/identity gate for the fresh R26 head."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if value not in sys.path:
        sys.path.insert(0, value)

from gate_lwc84_cuda_graph_fine_grid_trainonly import _fine_velocity, _read_manifest_rows  # noqa: E402
from gate_r6_anchored_r54_runtime import (  # noqa: E402
    AllowlistedHDF,
    CHUNKS,
    bit_preserving_residual_add,
    collect_algorithm_flags,
    configure_algorithm_flags,
    environment_contract,
    fresh_head_cpu,
    parent_stability_qc,
    preload_public_record,
    runtime_dependency_closure,
    sanitize_for_json,
    target5_preflight,
    validate_chunks,
    validate_selection_manifest,
    validate_time_axis,
    validate_zero_head,
)
from reattest_frozen_fine_grid_r6_train import atomic_write_bytes, report_bytes_fixed_point, sha256  # noqa: E402
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig  # noqa: E402
from fno_acoustic.data_generation.restriction import restrict_nodal_2x  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84_fused import FusedLWC84CPMLSolver  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84_fused_device import (  # noqa: E402
    DeviceResidentFusedLWC84CPMLSolver,
)


CANDIDATE = "r6_anchored_r54_device_resident_r1_20260905"
FAMILIES = ("uniform", "layered", "marmousi")
ZERO_CORRECTION_SHA256 = "ddc85b421224868704e104182225ebd4992edae08d7ff114a0087188d7cbb3ef"
PARENT_SHAPE = (401, 201, 201)
PARENT_BYTES = int(np.prod(PARENT_SHAPE) * np.dtype(np.float32).itemsize)
STATIC_BYTES = 7 * 201 * 201 * 4
TIME_BYTES = 401 * 4


class TransferLedger:
    def __init__(self) -> None:
        self.h2d_calls: list[dict[str, Any]] = []
        self.d2h_calls: list[dict[str, Any]] = []

    def h2d(self, label: str, byte_count: int, calls: int = 1) -> None:
        self.h2d_calls.append({"label": str(label), "calls": int(calls), "bytes": int(byte_count)})

    def d2h(self, label: str, byte_count: int, calls: int = 1) -> None:
        self.d2h_calls.append({"label": str(label), "calls": int(calls), "bytes": int(byte_count)})

    def summary(self) -> dict[str, Any]:
        return {
            "h2d": list(self.h2d_calls), "d2h": list(self.d2h_calls),
            "h2d_call_count": sum(row["calls"] for row in self.h2d_calls),
            "h2d_bytes": sum(row["bytes"] for row in self.h2d_calls),
            "d2h_call_count": sum(row["calls"] for row in self.d2h_calls),
            "d2h_bytes": sum(row["bytes"] for row in self.d2h_calls),
            "parent_d2h_call_count": sum(row["calls"] for row in self.d2h_calls if row["label"] == "parent"),
            "correction_d2h_call_count": sum(row["calls"] for row in self.d2h_calls if row["label"] == "correction"),
            "candidate_d2h_call_count": sum(row["calls"] for row in self.d2h_calls if row["label"] == "candidate"),
        }


def validate_owned_device_tensor(value: Any, *, finite_asserted: bool) -> dict[str, Any]:
    device_type = getattr(getattr(value, "device", None), "type", None)
    dtype = getattr(value, "dtype", None)
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    contiguous = bool(value.is_contiguous())
    owns_storage = getattr(value, "_base", None) is None
    passed = bool(
        device_type == "cuda" and dtype == torch.float32 and shape == (1, *PARENT_SHAPE)
        and contiguous and owns_storage and finite_asserted
    )
    if not passed:
        raise RuntimeError("device-resident parent ownership/native contract failure")
    return {"passed": True, "device": "cuda", "dtype": "float32", "shape": [1, *PARENT_SHAPE],
            "C_contiguous": True, "owns_storage": True, "finite_asserted_on_device": True}


def validate_transfer_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("h2d_call_count") != 32:
        raise RuntimeError("formal H2D call count drift")
    if summary.get("d2h_call_count") != 1 or summary.get("d2h_bytes") != PARENT_BYTES:
        raise RuntimeError("formal D2H count/bytes drift")
    if summary.get("candidate_d2h_call_count") != 1:
        raise RuntimeError("final candidate D2H missing")
    if summary.get("parent_d2h_call_count") != 0 or summary.get("correction_d2h_call_count") != 0:
        raise RuntimeError("intermediate parent/correction D2H is forbidden")


def device_result_qc(result: Any, *, fine: np.ndarray, stored: np.ndarray) -> dict[str, Any]:
    tensor = result.wavefield_device
    native = validate_owned_device_tensor(tensor, finite_asserted=bool(result.metrics[0]["finite_asserted_on_device"]))
    restriction_difference = float(np.max(np.abs(restrict_nodal_2x(fine) - stored)))
    if restriction_difference != 0.0:
        raise RuntimeError("fine/stored restriction drift")
    source_saved_sum = float(result.source_map_saved[0].sum(dtype=np.float64))
    source_solver_sum = float(result.source_map_solver[0].sum(dtype=np.float64))
    if abs(source_saved_sum - 1.0) > 2e-7 or abs(source_solver_sum - 1.0) > 2e-7:
        raise RuntimeError("device solver source QC failure")
    metrics = result.metrics[0]
    stability = parent_stability_qc(float(metrics["cfl_2d"]), float(metrics["lwc_qmax"]))
    if not stability["passed"] or float(metrics["dt_used_s"]) != 0.000625 or int(metrics["output_frame_count"]) != 401:
        raise RuntimeError("device solver stability/time QC failure")
    return {"passed": True, "native": native, "restriction_max_absolute_difference": restriction_difference,
            "source_map_saved_sum": source_saved_sum, "source_map_solver_sum": source_solver_sum,
            "stability_qc": stability, "dt_used_s": 0.000625, "output_frame_count": 401}


def device_head(
    model: torch.nn.Module,
    r25: Any,
    parent: torch.Tensor,
    stored_velocity: np.ndarray,
    times: np.ndarray,
    metadata: Mapping[str, float],
    *, device: torch.device,
    solver_transfer: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], float, dict[str, Any]]:
    validate_chunks()
    ledger = TransferLedger()
    ledger.h2d("solver_internal_public_inputs", int(solver_transfer["solver_h2d_bytes"]),
               int(solver_transfer["solver_h2d_call_count"]))
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    builder = importlib.import_module("scripts.build_r25_coarse_residual_cache")
    axis = np.linspace(0.0, 2000.0, 201, dtype=np.float32)
    static = builder.static_features(stored_velocity, x_m=axis, z_m=axis,
                                     source_x_m=metadata["source_x_m"], source_z_m=metadata["source_z_m"])
    static_gpu = torch.from_numpy(static).to(device)
    ledger.h2d("static7", STATIC_BYTES)
    scale = torch.clamp(parent.abs().amax(), min=1.0e-12)
    correction_chunks: list[torch.Tensor] = []
    candidate_chunks: list[torch.Tensor] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for start, stop in CHUNKS:
            time_gpu = torch.from_numpy(times[start:stop].astype(np.float32)).to(device)
            ledger.h2d("time_chunk", (stop - start) * 4)
            block = stop - start
            parent_chunk = parent[start:stop]
            features = r25.make_dynamic_features(
                parent_chunk / scale, static_gpu[None].expand(block, -1, -1, -1), time_s=time_gpu,
                source_f0_hz=torch.full((block,), metadata["source_f0_hz"], device=device),
                source_t0_s=torch.full((block,), metadata["source_t0_s"], device=device),
            )
            if features.shape[1] != 13:
                raise RuntimeError("device head feature channel/order drift")
            active = (time_gpu >= metadata["source_t0_s"]).float()
            correction = model(features, active=active).float() * scale
            candidate = bit_preserving_residual_add(parent_chunk, correction)
            correction_chunks.append(correction)
            candidate_chunks.append(candidate)
    correction_gpu = torch.cat(correction_chunks, dim=0).contiguous()
    candidate_gpu = torch.cat(candidate_chunks, dim=0).contiguous()
    zero_mask = correction_gpu == 0
    parent_bits = parent.view(torch.int32)
    candidate_bits = candidate_gpu.view(torch.int32)
    torch._assert_async(torch.count_nonzero(~zero_mask) == 0, "device correction is not exact numeric zero")
    torch._assert_async(torch.count_nonzero(parent_bits != candidate_bits) == 0, "device candidate bits differ from parent")
    torch._assert_async(torch.count_nonzero(parent != candidate_gpu) == 0, "device candidate values differ from parent")
    torch._assert_async(
        torch.count_nonzero(candidate_bits[zero_mask] != parent_bits[zero_mask]) == 0,
        "zero-correction positions did not preserve parent bits",
    )
    candidate_cpu = candidate_gpu.detach().cpu().numpy()
    ledger.d2h("candidate", PARENT_BYTES)
    torch.cuda.synchronize(device)
    head_seconds = float(time.perf_counter() - started)
    if candidate_cpu.dtype != np.float32 or candidate_cpu.shape != PARENT_SHAPE or not candidate_cpu.flags.c_contiguous:
        raise RuntimeError("final candidate native CPU contract failure")
    candidate_sha = hashlib.sha256(candidate_cpu.tobytes()).hexdigest()
    candidate_signed_zero = int(np.count_nonzero((candidate_cpu == 0) & np.signbit(candidate_cpu)))
    identity = {
        "parent_sha256_inferred_after_device_bit_identity": candidate_sha,
        "correction_sha256_of_exact_zero_float32_shape": ZERO_CORRECTION_SHA256,
        "candidate_sha256": candidate_sha,
        "parent_signed_zero_count_inferred_from_device_bit_identity": candidate_signed_zero,
        "correction_signed_zero_count": None,
        "candidate_signed_zero_count": candidate_signed_zero,
        "numeric_mismatch_count": 0, "byte_mismatch_count": 0,
        "zero_position_count": int(candidate_cpu.size),
        "zero_position_bit_preserved_count": int(candidate_cpu.size),
        "zero_position_bits_preserved": True, "nonzero_position_count": 0,
        "nonzero_additive_equality": True, "correction_all_zero": True, "full_identity": True,
        "identity_assertions_executed_on_device_without_scalar_D2H": True,
    }
    transfer = ledger.summary()
    validate_transfer_summary(transfer)
    return candidate_cpu, identity, head_seconds, transfer


def execute_device_repeat(*, prepared: Mapping[str, Any], grid: AcousticGrid,
                          solver: DeviceResidentFusedLWC84CPMLSolver, model: torch.nn.Module,
                          r25: Any, times: np.ndarray, marmousi: Path, device: torch.device
                          ) -> tuple[np.ndarray, dict[str, Any]]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    fine, fine_metadata = _fine_velocity(family=prepared["row"]["family"],
                                         manifest_row=prepared["manifest_row"], grid=grid,
                                         marmousi_npy=marmousi)
    result = solver.simulate_device(fine, **prepared["metadata"])
    parent_qc = device_result_qc(result, fine=fine, stored=prepared["stored_velocity"])
    candidate, identity, head_seconds, transfer = device_head(
        model, r25, result.wavefield_device[0], prepared["stored_velocity"], times,
        prepared["metadata"], device=device, solver_transfer=result.transfer_ledger,
    )
    torch.cuda.synchronize(device)
    outer_seconds = float(time.perf_counter() - started)
    del fine, result
    return candidate, {"identity": identity, "parent_qc": parent_qc,
                       "fine_velocity_metadata": fine_metadata, "outer_runtime_s": outer_seconds,
                       "head_runtime_s": head_seconds, "transfer_ledger": transfer}


def nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[max(1, math.ceil(probability * len(ordered))) - 1]


def runtime_gate(outer: Sequence[float], head: Sequence[float], *, smoke: bool) -> dict[str, Any]:
    expected = 3 if smoke else 180
    if len(outer) != expected or len(head) != expected:
        raise RuntimeError("raw runtime count drift")
    if smoke:
        return {"applied": False, "raw_outer_count": 3, "raw_head_count": 3}
    metrics = {"outer_mean": float(np.mean(outer)), "outer_p95": nearest_rank(outer, .95),
               "outer_p95_rank": 171, "head_p95": nearest_rank(head, .95),
               "raw_outer_count": 180, "raw_head_count": 180}
    gates = {"outer_mean_le_1_90": metrics["outer_mean"] <= 1.90,
             "outer_p95_le_1_95": metrics["outer_p95"] <= 1.95,
             "outer_mean_le_2_034137312322855": metrics["outer_mean"] <= 2.034137312322855,
             "outer_p95_le_2_034137312322855": metrics["outer_p95"] <= 2.034137312322855,
             "head_p95_le_1_10": metrics["head_p95"] <= 1.10}
    return {"applied": True, "metrics": metrics, "gates": gates, "passed": all(gates.values())}


def verify_dependency_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    closure = runtime_dependency_closure([_resolved(path) for path in manifest["roots"]])
    observed = {path.relative_to(ROOT).as_posix(): sha256(path) for path in closure}
    expected = {row["path"]: row["sha256"] for row in manifest["files"]}
    if observed != expected:
        raise RuntimeError("device runtime dependency closure drift")
    for row in manifest.get("explicit_bindings", []):
        path = _resolved(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"device runtime explicit binding drift: {row['path']}")
    return observed


def _resolved(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bootstrap(args: argparse.Namespace, prereg: Mapping[str, Any]) -> tuple[str, Path]:
    if prereg.get("schema") != "r6_anchored_r54_device_runtime_preregistration_v1" or prereg.get("candidate") != CANDIDATE:
        raise RuntimeError("device runtime preregistration schema/candidate drift")
    mode = "smoke" if args.smoke else "full"
    if args.preregistration.resolve() != _resolved(prereg["paths"]["preregistration"]):
        raise RuntimeError("preregistration path override")
    output = _resolved(prereg["paths"][f"{mode}_output"])
    if args.output.resolve() != output:
        raise RuntimeError("canonical output override")
    if output.exists():
        raise FileExistsError("fixed output exists")
    return mode, output


def validate_remaining(args: argparse.Namespace, prereg: Mapping[str, Any], mode: str) -> None:
    expected = "static_passed_smoke_pending_audit" if mode == "smoke" else "smoke_passed_full_pending_audit"
    if prereg.get("status") != expected:
        raise RuntimeError("device runtime stage status rejected")
    for key, supplied in {"source_h5": args.source_h5, "manifest": args.manifest, "marmousi": args.marmousi}.items():
        if supplied.resolve() != _resolved(prereg["paths"][key]):
            raise RuntimeError(f"fixed path override: {key}")
    if mode == "full":
        item = prereg["prerequisites"]["smoke"]
        if item.get("status") != "passed" or sha256(_resolved(item["path"])) != item.get("sha256"):
            raise RuntimeError("device runtime smoke prerequisite failure")


def verify_inputs(prereg: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    dependency = json.loads(_resolved(prereg["paths"]["dependency_manifest"]).read_text())
    closure = verify_dependency_manifest(dependency)
    explicit = {name: sha256(_resolved(path)) for name, path in prereg["binding_paths"].items()}
    if explicit != prereg["bindings"]:
        raise RuntimeError("device runtime explicit input drift")
    return {"dependency_closure": closure, "explicit_bindings": explicit}


def write_failure(report: Mapping[str, Any], error: Exception, output: Path, started: float,
                  proxy: AllowlistedHDF | None) -> None:
    payload = {"schema": "r6_anchored_r54_device_runtime_report_v1", "candidate": CANDIDATE,
               "mode": report.get("mode"), "status": "invalid", "error": repr(error),
               "argv": report.get("argv"), "target5_audit": report.get("target5_audit"),
               "environment": report.get("environment"), "algorithm_flags": report.get("algorithm_flags"),
               "preregistration_sha256_before": report.get("preregistration_sha256_before"),
               "preregistration_sha256_after_if_available": report.get("preregistration_sha256_after"),
               "input_hashes_before": report.get("input_hashes_before"),
               "input_hashes_after_if_available": report.get("input_hashes_after"),
               "partial_transfer_ledgers": report.get("partial_transfer_ledgers"),
               "hdf_ledger": proxy.ledger() if proxy else None,
               "resources": {"elapsed_seconds": float(time.time() - started),
                             "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0,
                             "output_bytes": 0},
               "no_truth": True, "weights_loaded": False, "cache_written": False,
               "predictions_persisted": False, "promotion_authorized": False,
               "validation_truth_opened": False, "test_id_truth_opened": False}
    sanitized, nonfinite = sanitize_for_json(payload)
    sanitized["nonfinite_paths_sanitized_to_null"] = nonfinite
    try:
        data = report_bytes_fixed_point(sanitized)
        if len(data) >= 64 * 2**20:
            raise RuntimeError("failure report budget")
        atomic_write_bytes(data, output)
    except Exception as serializer_error:
        fallback = {"schema": "r6_anchored_r54_device_runtime_report_v1", "candidate": CANDIDATE,
                    "status": "invalid", "error": "failure_report_serializer_error",
                    "serializer_error": repr(serializer_error), "no_truth": True,
                    "resources": {"peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
                    ), "output_bytes": 0}, "promotion_authorized": False,
                    "validation_truth_opened": False, "test_id_truth_opened": False}
        for _ in range(16):
            data = (json.dumps(fallback, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            if fallback["resources"]["output_bytes"] == len(data):
                break
            fallback["resources"]["output_bytes"] = len(data)
        atomic_write_bytes(data, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true"); modes.add_argument("--full", action="store_true")
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--source-h5", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--marmousi", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    mode, output = bootstrap(args, prereg)
    started = time.time(); proxy = None
    report: dict[str, Any] = {"schema": "r6_anchored_r54_device_runtime_report_v1", "candidate": CANDIDATE,
                              "mode": mode, "argv": list(sys.argv), "status": "running",
                              "preregistration_sha256_before": sha256(args.preregistration),
                              "promotion_authorized": False, "no_truth": True,
                              "partial_transfer_ledgers": []}
    try:
        validate_remaining(args, prereg, mode)
        report["algorithm_flags"] = configure_algorithm_flags(prereg["environment"])
        report["environment"] = environment_contract()
        if report["environment"] != prereg["environment"]:
            raise RuntimeError("device runtime environment drift")
        before = verify_inputs(prereg); report["input_hashes_before"] = before
        report["target5_audit"] = target5_preflight(prereg)
        selection = validate_selection_manifest(json.loads(_resolved(prereg["paths"]["selection"]).read_text()))
        active = [selection[0], selection[20], selection[40]] if mode == "smoke" else selection
        model, r25, initial_digest, parameter_digest = fresh_head_cpu(); zero = validate_zero_head(model)
        if initial_digest != prereg["fresh_head"]["initial_state_digest"] or parameter_digest != prereg["fresh_head"]["parameter_manifest_digest"] or not zero["passed"]:
            raise RuntimeError("fresh head identity drift")
        device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
        model = model.to(device).eval()
        grid = AcousticGrid(nx=401, nz=401, dx_m=5, dz_m=5, lx_m=2000, lz_m=2000, centering="node")
        with h5py.File(args.source_h5, "r", swmr=True) as raw:
            proxy = AllowlistedHDF(raw); times = np.asarray(proxy["time_s"][:], np.float64); time_qc = validate_time_axis(times)
            manifest_rows = _read_manifest_rows(args.manifest, {row["sample_id"] for row in active})
            prepared = [preload_public_record(proxy, row, manifest_rows[row["sample_id"]]) for row in active]
            common = dict(grid=grid, boundaries=BoundaryConfig(npml=40), dt_s=.000625, output_times_s=times,
                          c_ref_mps=6750, device=device, dtype=torch.float32, kappa_max=3,
                          minimum_frequency_hz=8, output_restriction_factor=2, cuda_graphs=True)
            solver = DeviceResidentFusedLWC84CPMLSolver(**common); solver_warmup = float(solver.warmup(batch=1))
            legacy = FusedLWC84CPMLSolver(**common) if mode == "smoke" else None
            if legacy is not None: legacy.warmup(batch=1)
            warm_candidate, _ = execute_device_repeat(prepared=prepared[0], grid=grid, solver=solver, model=model,
                                                       r25=r25, times=times, marmousi=args.marmousi, device=device)
            del warm_candidate
            records = []; outer = []; head = []
            for item in prepared:
                parity = None
                if legacy is not None:
                    fine, _ = _fine_velocity(family=item["row"]["family"], manifest_row=item["manifest_row"],
                                             grid=grid, marmousi_npy=args.marmousi)
                    legacy_result = legacy.simulate(fine, **item["metadata"])
                    legacy_parent = legacy_result.wavefield[0]
                    legacy_sha = hashlib.sha256(legacy_parent.tobytes()).hexdigest()
                repeats = []
                for _ in range(1 if mode == "smoke" else 3):
                    candidate, repeat = execute_device_repeat(prepared=item, grid=grid, solver=solver, model=model,
                                                              r25=r25, times=times, marmousi=args.marmousi, device=device)
                    report["partial_transfer_ledgers"].append(repeat["transfer_ledger"])
                    if legacy is not None:
                        parity = {"legacy_sha256": legacy_sha, "device_final_sha256": repeat["identity"]["candidate_sha256"],
                                  "numeric_equal": bool(np.array_equal(legacy_parent, candidate)),
                                  "byte_equal": bool(legacy_parent.tobytes() == candidate.tobytes())}
                        if not parity["numeric_equal"] or not parity["byte_equal"]:
                            raise RuntimeError("legacy/device solver parity failure")
                    repeats.append(repeat); outer.append(repeat["outer_runtime_s"]); head.append(repeat["head_runtime_s"])
                    del candidate
                records.append({**item["row"], "repeats": repeats, "smoke_legacy_device_parity": parity})
        expected_identity = 3 if mode == "smoke" else 180
        all_repeats = [repeat for row in records for repeat in row["repeats"]]
        if len(all_repeats) != expected_identity or not all(repeat["identity"]["full_identity"] and repeat["parent_qc"]["passed"] for repeat in all_repeats):
            raise RuntimeError("device identity/QC count failure")
        if not all(len({repeat["identity"]["candidate_sha256"] for repeat in row["repeats"]}) == 1 for row in records):
            raise RuntimeError("device repeat hash drift")
        aggregate_transfer = {"h2d_call_count": sum(r["transfer_ledger"]["h2d_call_count"] for r in all_repeats),
                              "h2d_bytes": sum(r["transfer_ledger"]["h2d_bytes"] for r in all_repeats),
                              "d2h_call_count": sum(r["transfer_ledger"]["d2h_call_count"] for r in all_repeats),
                              "d2h_bytes": sum(r["transfer_ledger"]["d2h_bytes"] for r in all_repeats),
                              "parent_d2h_call_count": 0, "correction_d2h_call_count": 0,
                              "candidate_d2h_call_count": expected_identity}
        gate = runtime_gate(outer, head, smoke=mode == "smoke")
        status = "smoke_complete" if mode == "smoke" else ("runtime_identity_passed_cache_prereg_pending_audit" if gate["passed"] else "runtime_identity_rejected")
        after = verify_inputs(prereg); prereg_after = sha256(args.preregistration)
        flags_after = collect_algorithm_flags(); report["algorithm_flags"]["observed_after"] = flags_after
        if after != before or prereg_after != report["preregistration_sha256_before"] or flags_after != report["algorithm_flags"]["configured_after"]:
            raise RuntimeError("device runtime input/flag drift")
        resources = {"elapsed_seconds": float(time.time() - started),
                     "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "output_bytes": 0}
        if resources["elapsed_seconds"] > (180 if mode == "smoke" else 900) or resources["peak_allocated_bytes"] >= 8 * 2**30:
            raise RuntimeError("device runtime resource budget")
        ledger = proxy.ledger()
        if ledger["denied_attempt_count"] or ledger["wavefield_access_count"]:
            raise RuntimeError("device runtime HDF ledger failure")
        report.update({"status": status, "claim": "implementation_only" if mode == "smoke" else "device-resident R6-anchored runtime identity gate; no accuracy or causality claim",
                       "record_count": len(records), "identity_count": expected_identity, "records": records,
                       "runtime_gate": gate, "aggregate_transfer_ledger": aggregate_transfer,
                       "time_axis_qc": time_qc, "solver_warmup_seconds_excluded": solver_warmup,
                       "fresh_head": {"initial_state_digest": initial_digest, "parameter_manifest_digest": parameter_digest,
                                      "parameter_count": 2791762, "zero_heads": zero, "weights_loaded": False},
                       "target5_audit": report["target5_audit"], "hdf_ledger": ledger,
                       "input_hashes_after": after, "preregistration_sha256_after": prereg_after,
                       "resources": resources, "weights_loaded": False, "cache_written": False,
                       "predictions_persisted": False, "validation_truth_opened": False, "test_id_truth_opened": False})
        data = report_bytes_fixed_point(report)
        if len(data) >= 64 * 2**20 or len(data) != report["resources"]["output_bytes"]:
            raise RuntimeError("device runtime JSON budget")
        atomic_write_bytes(data, output)
        return 0 if status in {"smoke_complete", "runtime_identity_passed_cache_prereg_pending_audit"} else 2
    except Exception as error:
        try:
            report["preregistration_sha256_after"] = sha256(args.preregistration)
            report["input_hashes_after"] = verify_inputs(prereg)
        except Exception:
            pass
        write_failure(report, error, output, started, proxy)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
