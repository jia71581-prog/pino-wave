#!/usr/bin/env python3
"""Gate an accelerated numerical coarse LWC84 parent on exposed train records."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if value not in sys.path:
        sys.path.insert(0, value)

from gate_lwc84_cuda_graph_fine_grid_trainonly import (  # noqa: E402
    _add_terms,
    _error_terms,
    _fine_velocity,
    _nearest_rank,
    _read_manifest_rows,
    _relative_l2,
)
from reattest_frozen_fine_grid_r6_train import (  # noqa: E402
    atomic_write_bytes,
    dependency_closure,
    report_bytes_fixed_point,
    sha256,
    verify_dependency_manifest,
)
from saved_time_phase_operator_v4.streaming_metrics import (  # noqa: E402
    ExactWavefieldMetricAccumulator,
)
from fno_acoustic.data_generation.cpml import build_cfs_cpml_profiles  # noqa: E402
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84_fused import (  # noqa: E402
    FusedLWC84CPMLSolver,
)
from fno_acoustic.data_generation.source import bilinear_point_source  # noqa: E402


CANDIDATE = "accelerated_coarse_lwc84_residual_r1_20260905"
NAMESPACE = "accelerated_coarse_lwc84_residual_r1_parent_gate_v1"
FAMILIES = ("uniform", "layered", "marmousi")
DT_A = 0.000125
DT_B = 0.00025
EXPECTED_SELECTION_DIGESTS = {
    "indices": "442da7a5f0d66fd66e1a733480baff225b612f4dc90f5d9d15e516a6ae7fb4a4",
    "sample_ids": "6e2d638d1ea50334771a0f7372e00cbe25b759dff948adf6f8d43de210da304f",
    "groups": "bfb97ec57df9c845bcf2668ea862eab4f992db75ba72c1949f128d0f1670f2ac",
    "sample_hashes": "0f1874d31596e137e0810759ef1dad15dba10a38ed9ee52c15ab3d72ab8e6be8",
    "records": "21655affc0b82233cbc046872810c23c08b1d45f61cc09bda0ee8700834745f4",
}
TEMPORAL_RANGES = {"early": (0, 134), "middle": (134, 267), "late": (267, 401)}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def selection_digests(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        "indices": canonical_sha256([int(row["source_index"]) for row in records]),
        "sample_ids": canonical_sha256([str(row["sample_id"]) for row in records]),
        "groups": canonical_sha256([str(row["group_id"]) for row in records]),
        "sample_hashes": canonical_sha256([str(row["sample_sha256"]) for row in records]),
        "records": canonical_sha256(list(records)),
    }


def select_metadata(handle: h5py.File) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, list[int]]] = {family: defaultdict(list) for family in FAMILIES}
    for index in range(int(handle["split"].shape[0])):
        if decode(handle["split"][index]) != "train":
            continue
        family = decode(handle["medium_type"][index])
        if family in groups:
            groups[family][decode(handle["group_id"][index])].append(index)
    records = []
    for family in FAMILIES:
        ranked = sorted(
            groups[family],
            key=lambda group: (
                hashlib.sha256((NAMESPACE + "\0" + family + "\0" + group).encode()).digest(),
                group,
            ),
        )[:8]
        for group in ranked:
            index = min(groups[family][group], key=lambda i: (int(i), decode(handle["sample_id"][i])))
            records.append(
                {
                    "source_index": int(index),
                    "sample_id": decode(handle["sample_id"][index]),
                    "group_id": group,
                    "family": family,
                    "sample_sha256": decode(handle["sample_sha256"][index]),
                }
            )
    if len(records) != 24 or len({row["group_id"] for row in records}) != 24:
        raise RuntimeError("selection count/group drift")
    if selection_digests(records) != EXPECTED_SELECTION_DIGESTS:
        raise RuntimeError("selection digest drift")
    return records


def raw_sequence(position: int) -> tuple[str, ...]:
    return ("A", "B", "A", "B", "A", "B") if position % 2 == 0 else ("B", "A", "B", "A", "B", "A")


def formal_p95(values: Sequence[float]) -> float:
    return _nearest_rank(list(values), 0.95)


class ABTruthGuard:
    def __init__(self, records: Sequence[Mapping[str, Any]], *, mode: str) -> None:
        self.expected = {int(row["source_index"]): dict(row) for row in records}
        self.required_tags = (
            {"A_hash1", "A_hash2", "A_hash3", "B_hash1", "B_hash2", "B_hash3"}
            if mode == "full"
            else {"A_graph", "A_nongraph", "B_graph", "B_nongraph"}
        )
        self.tags = {index: {} for index in self.expected}
        self.tag_qc = {index: {} for index in self.expected}
        self.truth_counts = {index: 0 for index in self.expected}
        self.states = {index: "metadata" for index in self.expected}
        self.transitions: list[dict[str, Any]] = []

    def register_hash(self, *, index: int, sample_id: str, family: str, tag: str, digest: str, native_qc: Mapping[str, Any] | None = None) -> None:
        row = self.expected.get(index)
        if row is None or row["sample_id"] != sample_id or row["family"] != family:
            raise PermissionError("prediction identity mismatch")
        if tag not in self.required_tags or tag in self.tags[index] or len(digest) != 64:
            raise RuntimeError("prediction tag/hash contract failure")
        int(digest, 16)
        self.tags[index][tag] = digest
        self.tag_qc[index][tag] = dict(native_qc or {})
        self.transitions.append({"source_index": index, "to": tag})
        if set(self.tags[index]) == self.required_tags:
            self.states[index] = "both_arms_hashed"
            self.transitions.append({"source_index": index, "to": "both_arms_hashed"})

    def read_truth(self, handle: h5py.File, *, index: int, sample_id: str, split: str, family: str) -> np.ndarray:
        row = self.expected.get(index)
        if self.states.get(index) != "both_arms_hashed":
            raise PermissionError("truth forbidden until both arms are fully hashed")
        if row is None or row["sample_id"] != sample_id or row["family"] != family:
            raise PermissionError("truth identity mismatch")
        if split != "train" or family not in FAMILIES or family == "anomaly":
            raise PermissionError("truth restricted to selected train target families")
        if self.truth_counts[index]:
            raise PermissionError("truth second read forbidden")
        self.transitions.append({"source_index": index, "to": "truth_authorized"})
        truth = np.asarray(handle["wavefield"][index, 0:401], dtype=np.float32)
        self.truth_counts[index] = 1
        self.states[index] = "truth_read"
        self.transitions.append({"source_index": index, "to": "truth_read_once"})
        return truth

    def mark_complete(self, index: int) -> None:
        if self.states.get(index) != "truth_read":
            raise RuntimeError("metrics require one truth read")
        self.states[index] = "metrics_complete"
        self.transitions.append({"source_index": index, "to": "metrics_complete"})

    def summary(self) -> dict[str, Any]:
        return {
            "selected_count": len(self.expected),
            "prediction_hash_count": sum(len(value) for value in self.tags.values()),
            "truth_read_count": sum(self.truth_counts.values()),
            "metrics_complete_count": sum(state == "metrics_complete" for state in self.states.values()),
            "tag_to_sha256": {str(index): dict(sorted(tags.items())) for index, tags in self.tags.items()},
            "per_tag_native_qc": {str(index): dict(sorted(tags.items())) for index, tags in self.tag_qc.items()},
            "each_truth_once": all(count == 1 for count in self.truth_counts.values()),
            "all_truth_counts_at_most_one": all(count <= 1 for count in self.truth_counts.values()),
            "transitions": list(self.transitions),
            "validation_truth_reopened_this_stage": False,
            "test_id_truth_reopened_this_stage": False,
        }


def prediction_qc(value: np.ndarray) -> tuple[np.ndarray, str, dict[str, Any]]:
    if not isinstance(value, np.ndarray):
        raise TypeError("prediction must be a native numpy ndarray")
    if value.dtype != np.dtype(np.float32):
        raise RuntimeError("prediction native dtype must be exact float32")
    if value.shape != (401, 201, 201):
        raise RuntimeError("prediction native shape drift")
    if not value.flags.c_contiguous:
        raise RuntimeError("prediction native array must be C-contiguous")
    if not np.isfinite(value).all():
        raise RuntimeError("prediction contains non-finite values")
    if not np.array_equal(value[:, 0, :], np.zeros((401, 201), dtype=np.float32)):
        raise RuntimeError("prediction top row not exact zero")
    array = np.ascontiguousarray(value)
    qc = {"native_ndarray": True, "native_dtype": "float32", "native_shape": [401, 201, 201], "native_C_contiguous": True, "finite": True, "top_row_exact_zero": True}
    return array, hashlib.sha256(array.tobytes()).hexdigest(), qc


def profile_relation_qc() -> dict[str, Any]:
    grid = AcousticGrid(nx=201, nz=201, dx_m=10.0, dz_m=10.0, lx_m=2000.0, lz_m=2000.0, centering="node")
    boundary = BoundaryConfig(npml=20)
    common = dict(grid=grid, boundaries=boundary, c_ref_mps=6750.0, target_reflection=1e-8, polynomial_order=3, kappa_max=3.0, minimum_frequency_hz=8.0, device="cpu", dtype=torch.float32)
    a = build_cfs_cpml_profiles(dt_s=DT_A, **common)
    b = build_cfs_cpml_profiles(dt_s=DT_B, **common)
    fields = ("sigma_x", "sigma_z", "kappa_x", "kappa_z", "alpha_x", "alpha_z")
    static_max = max(float((getattr(a, name) - getattr(b, name)).abs().max()) for name in fields)
    active_equal = torch.equal(a.active_x, b.active_x) and torch.equal(a.active_z, b.active_z)
    b_relation = max(float((b.b_x - a.b_x.square()).abs().max()), float((b.b_z - a.b_z.square()).abs().max()))
    a_relation = max(float((b.a_x - a.a_x * (1.0 + a.b_x)).abs().max()), float((b.a_z - a.a_z * (1.0 + a.b_z)).abs().max()))
    return {"static_profile_max_abs": static_max, "active_exact": active_equal, "b_relation_max_abs": b_relation, "a_relation_max_abs": a_relation, "passed": static_max <= 1e-6 and active_equal and b_relation <= 1e-6 and a_relation <= 1e-6}


def source_qc(source_x: float, source_z: float) -> dict[str, Any]:
    point = bilinear_point_source(source_x, source_z, nx=201, nz=201, dx_m=10.0, dz_m=10.0, centering="node")
    return {
        "source_map_saved_sum": float(point.source_map.sum(dtype=np.float64)),
        "delta_solver_integral": float(point.delta_h.sum() * 10.0 * 10.0),
        "passed": abs(float(point.source_map.sum(dtype=np.float64)) - 1.0) <= 2e-7 and abs(float(point.delta_h.sum() * 100.0) - 1.0) <= 2e-7,
    }


def stability_qc(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    cfl_ratio = float(b["cfl_2d"]) / float(a["cfl_2d"])
    qmax_ratio = float(b["lwc_qmax"]) / float(a["lwc_qmax"])
    return {"A_cfl": float(a["cfl_2d"]), "B_cfl": float(b["cfl_2d"]), "A_qmax": float(a["lwc_qmax"]), "B_qmax": float(b["lwc_qmax"]), "cfl_ratio": cfl_ratio, "qmax_ratio": qmax_ratio, "passed": max(float(a["cfl_2d"]), float(b["cfl_2d"]), float(a["lwc_qmax"]), float(b["lwc_qmax"])) < 1.0 and math.isclose(cfl_ratio, 2.0, rel_tol=1e-6) and math.isclose(qmax_ratio, 4.0, rel_tol=1e-6)}


def pretruth_physics_qc(
    *,
    mode: str,
    profile: Mapping[str, Any],
    source: Mapping[str, Any],
    result_metrics: Mapping[str, Mapping[str, Any]],
    graph_nongraph: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, Any]:
    if mode == "smoke":
        stability = {
            "graph": stability_qc(result_metrics["A_graph"], result_metrics["B_graph"]),
            "nongraph": stability_qc(result_metrics["A_nongraph"], result_metrics["B_nongraph"]),
        }
        graph_checks = bool(
            graph_nongraph
            and all(
                graph_nongraph[arm]["relative_l2"] <= 1e-6
                and graph_nongraph[arm]["max_abs"] <= 1e-10
                for arm in ("A", "B")
            )
        )
    else:
        stability = {
            f"repeat_{repeat}": stability_qc(
                result_metrics[f"A_hash{repeat}"], result_metrics[f"B_hash{repeat}"]
            )
            for repeat in (1, 2, 3)
        }
        graph_checks = True
    result = {
        "profile_passed": bool(profile.get("passed")),
        "source_passed": bool(source.get("passed")),
        "stability": stability,
        "graph_nongraph_passed": graph_checks,
    }
    result["passed"] = (
        result["profile_passed"]
        and result["source_passed"]
        and all(row["passed"] for row in stability.values())
        and graph_checks
    )
    return result


def truth_after_pretruth_qc(*, qc: Mapping[str, Any], truth_callback):
    if not qc.get("passed"):
        raise RuntimeError("pre-truth physics QC failure")
    return truth_callback()


def update_accumulator(accumulator: ExactWavefieldMetricAccumulator, prediction: np.ndarray, target: np.ndarray, row: Mapping[str, Any]) -> None:
    for start in range(0, 401, 32):
        stop = min(start + 32, 401)
        accumulator.update(
            torch.from_numpy(prediction[start:stop])[None],
            torch.from_numpy(target[start:stop])[None],
            families=[str(row["family"])],
            group_ids=[str(row["group_id"])],
            sample_ids=[str(row["sample_id"])],
            time_indices=torch.arange(start, stop, dtype=torch.long)[None],
            source_onset_indices=None,
        )


def energy_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    total = (0.0, 0.0); families = {family: (0.0, 0.0) for family in FAMILIES}
    values = []
    for row in rows:
        terms = tuple(row[key]); total = _add_terms(total, terms); families[row["family"]] = _add_terms(families[row["family"]], terms); values.append(_relative_l2(terms))
    counts = {family: sum(row["family"] == family for row in rows) for family in FAMILIES}
    return {"energy_aggregate": _relative_l2(total), "family_count": counts, "family_energy": {family: (_relative_l2(families[family]) if counts[family] else None) for family in FAMILIES}, "record_mean": float(np.mean(values)), "family_record_mean": {family: (float(np.mean([value for value, row in zip(values, rows) if row["family"] == family])) if counts[family] else None) for family in FAMILIES}, "maximum_record": max(values), "record_values": values}


def assert_full_family_counts(*summaries: Mapping[str, Any]) -> None:
    expected = {"uniform": 8, "layered": 8, "marmousi": 8}
    for summary in summaries:
        if summary.get("family_count") != expected:
            raise RuntimeError("full family count contract failure before family arithmetic")


def assemble_truth_delta(
    A: Mapping[str, Any], B: Mapping[str, Any], *, require_full: bool
) -> dict[str, Any]:
    if require_full:
        assert_full_family_counts(A, B)
    family = {}
    for name in FAMILIES:
        a_value, b_value = A["family_energy"].get(name), B["family_energy"].get(name)
        present = int(A["family_count"].get(name, 0)) > 0 and int(B["family_count"].get(name, 0)) > 0
        family[name] = (
            float(b_value) - float(a_value)
            if present and a_value is not None and b_value is not None and math.isfinite(float(a_value)) and math.isfinite(float(b_value))
            else None
        )
    deltas = [float(b) - float(a) for a, b in zip(A["record_values"], B["record_values"])]
    if not all(math.isfinite(value) for value in deltas):
        raise FloatingPointError("non-finite per-record truth delta")
    aggregate = float(B["energy_aggregate"]) - float(A["energy_aggregate"])
    if not math.isfinite(aggregate):
        raise FloatingPointError("non-finite aggregate truth delta")
    return {
        "aggregate": aggregate,
        "family": family,
        "maximum_record": max(deltas) if deltas else None,
        "count_le_0_005": sum(value <= 0.005 for value in deltas),
        "record_values": deltas,
    }


def present_aware_streaming(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    observed = dict(metrics.get("family_relative_l2", {}))
    result["family_relative_l2"] = {
        family: (float(observed[family]) if family in observed and math.isfinite(float(observed[family])) else None)
        for family in FAMILIES
    }
    return result


def evaluate_gates(metrics: Mapping[str, Any], qc_passed: bool) -> dict[str, bool]:
    runtime = metrics["runtime"]; ba = metrics["B_vs_A"]; bt = metrics["B_vs_truth"]; at = metrics["A_vs_truth"]; delta = metrics["B_minus_A_truth"]
    gates = {
        "B_runtime_mean": runtime["B_mean"] <= 0.90,
        "B_runtime_p95": runtime["B_p95"] <= 1.0,
        "runtime_mean_ratio": runtime["B_mean"] <= 0.70 * runtime["A_mean"],
        "runtime_p95_ratio": runtime["B_p95"] <= 0.75 * runtime["A_p95"],
        "B_vs_A_aggregate": ba["energy_aggregate"] <= 0.01,
        "B_vs_A_family": max(ba["family_energy"].values()) <= 0.015,
        "B_vs_A_max": ba["maximum_record"] <= 0.025,
        "B_truth_mean": bt["record_mean"] <= 0.10,
        "B_truth_family_mean": max(bt["family_record_mean"].values()) <= 0.12,
        "B_truth_max": bt["maximum_record"] <= 0.25,
        "truth_energy_delta": delta["aggregate"] <= 0.0025,
        "truth_family_delta": max(delta["family"].values()) <= 0.005,
        "truth_max_record_delta": delta["maximum_record"] <= 0.01,
        "truth_record_delta_count": delta["count_le_0_005"] >= 20,
        "temporal": all(metrics["temporal_B"][name] <= max(1.10 * metrics["temporal_A"][name], metrics["temporal_A"][name] + 0.005) for name in TEMPORAL_RANGES),
        "spectrum_high": metrics["streaming_B_vs_A"]["spectrum_relative_l2"]["high"] <= 0.02,
        "phase_B_vs_A": metrics["streaming_B_vs_A"]["phase_correlation"] >= 0.9999,
        "phase_B_truth": metrics["streaming_B_truth"]["phase_correlation"] >= 0.995 and metrics["streaming_B_truth"]["phase_correlation"] >= metrics["streaming_A_truth"]["phase_correlation"] - 0.0005,
        "xcorr": metrics["streaming_B_vs_A"]["xcorr_peak_shift_cells"] <= 0.05,
        "centroid": metrics["streaming_B_vs_A"]["centroid_shift_cells"] <= 0.25,
        "all_QC": bool(qc_passed),
    }
    return gates


def sanitize_for_json(value: Any, *, path: str = "$", nonfinite_paths: list[str] | None = None) -> tuple[Any, list[str]]:
    paths = [] if nonfinite_paths is None else nonfinite_paths
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(path)
        return None, paths
    if isinstance(value, Mapping):
        return {str(key): sanitize_for_json(item, path=f"{path}.{key}", nonfinite_paths=paths)[0] for key, item in value.items()}, paths
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item, path=f"{path}[{index}]", nonfinite_paths=paths)[0] for index, item in enumerate(value)], paths
    return value, paths


def minimal_failure_payload(*, report: Mapping[str, Any], error: Exception, started: float, guard: ABTruthGuard | None) -> dict[str, Any]:
    safe_source = {
        "schema": report.get("schema"), "candidate": report.get("candidate"), "mode": report.get("mode"), "argv": report.get("argv"),
        "expected_environment": report.get("expected_environment"), "expected_algorithm_flags": report.get("expected_algorithm_flags"), "observed_environment": report.get("observed_environment"),
        "preregistration_sha256_before": report.get("preregistration_sha256_before"), "input_hashes_before": report.get("input_hashes_before"),
        "partial_truth_ledger": guard.summary() if guard else None,
    }
    sanitized, nonfinite = sanitize_for_json(safe_source)
    return {**sanitized, "status":"invalid", "error":repr(error), "nonfinite_paths_sanitized_to_null":nonfinite, "resources":{"elapsed_seconds":float(time.time()-started),"peak_allocated_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0,"output_bytes":0}, "promotion_authorized":False,"validation_truth_reopened_this_stage":False,"test_id_truth_reopened_this_stage":False}


def write_failure_atomic(*, report: Mapping[str, Any], error: Exception, started: float, guard: ABTruthGuard | None, output: Path) -> None:
    payload = minimal_failure_payload(report=report, error=error, started=started, guard=guard)
    try:
        atomic_write_bytes(report_bytes_fixed_point(payload), output)
    except Exception as serializer_error:
        fallback = {"schema":"accelerated_coarse_lwc84_parent_gate_report_v1","status":"invalid","error":"failure_report_serializer_error","serializer_error":repr(serializer_error),"resources":{"output_bytes":0},"promotion_authorized":False,"validation_truth_reopened_this_stage":False,"test_id_truth_reopened_this_stage":False}
        for _ in range(16):
            data=(json.dumps(fallback,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
            if fallback["resources"]["output_bytes"]==len(data): break
            fallback["resources"]["output_bytes"]=len(data)
        atomic_write_bytes(data, output)


def environment_whitelist() -> dict[str, Any]:
    result = {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "h5py": h5py.__version__, "scipy": scipy.__version__, "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"cudnn_benchmark":torch.backends.cudnn.benchmark,"cudnn_deterministic":torch.backends.cudnn.deterministic}
    query = __import__("subprocess").run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    result["nvidia_driver"], result["gpu_model"] = [value.strip() for value in query.split(",", 1)]
    return result


def _resolved(value: str | Path) -> Path:
    path = Path(value); return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bootstrap_canonical_output(args: argparse.Namespace, prereg: Mapping[str, Any]) -> tuple[str, Path]:
    if prereg.get("schema") != "accelerated_coarse_lwc84_parent_gate_preregistration_v1" or prereg.get("candidate") != CANDIDATE:
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
    paths = prereg["paths"]
    for key, value in {"selection": args.selection, "source_h5": args.source_h5, "manifest": args.manifest, "marmousi": args.marmousi}.items():
        if value.resolve() != _resolved(paths[key]): raise RuntimeError(f"path override: {key}")
    if mode == "full":
        smoke = prereg["prerequisites"]["smoke"]; path = _resolved(smoke["path"])
        if smoke.get("status") != "passed" or sha256(path) != smoke.get("sha256"): raise RuntimeError("smoke prerequisite failure")


def validate_environment_schema(prereg: Mapping[str, Any]) -> None:
    required_environment={"CUBLAS_WORKSPACE_CONFIG","CUDA_VISIBLE_DEVICES","python","torch","torch_cuda","cudnn","numpy","h5py","scipy","nvidia_driver","gpu_model"}
    flags=prereg.get("algorithm_flags")
    if not isinstance(prereg.get("environment"),Mapping) or set(prereg["environment"])!=required_environment:
        raise RuntimeError("environment schema incomplete")
    if not isinstance(flags,Mapping) or flags!={"deterministic_algorithms":False,"cudnn_benchmark":False,"cudnn_deterministic":False}:
        raise RuntimeError("algorithm_flags schema/value incomplete")


def validate_environment(observed: Mapping[str, Any], prereg: Mapping[str, Any]) -> None:
    validate_environment_schema(prereg)
    expected={**prereg["environment"],**prereg["algorithm_flags"]}
    expected.pop("contract",None)
    for key,value in expected.items():
        if observed.get(key)!=value: raise RuntimeError(f"environment contract failure: {key}")


def validate_invocation(args: argparse.Namespace, prereg: Mapping[str, Any]) -> tuple[str, Path]:
    mode, output = bootstrap_canonical_output(args, prereg)
    validate_remaining_invocation(args, prereg, mode)
    return mode, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=True); modes.add_argument("--smoke", action="store_true"); modes.add_argument("--full", action="store_true")
    parser.add_argument("--preregistration", type=Path, required=True); parser.add_argument("--selection", type=Path, required=True); parser.add_argument("--source-h5", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--marmousi", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); prereg = json.loads(args.preregistration.read_text()); mode, output = bootstrap_canonical_output(args, prereg); started = time.time(); guard = None
    report: dict[str, Any] = {"schema":"accelerated_coarse_lwc84_parent_gate_report_v1","candidate":CANDIDATE,"mode":mode,"status":"running","argv":list(sys.argv),"claim":"numeric coarse-parent diagnostic; no cache, neural training, validation/test_id, or promotion claim","expected_environment":None,"expected_algorithm_flags":None,"observed_environment":None,"preregistration_sha256_before":sha256(args.preregistration),"promotion_authorized":False,"validation_truth_reopened_this_stage":False,"test_id_truth_reopened_this_stage":False}
    try:
        validate_environment_schema(prereg); report["expected_environment"]=dict(prereg["environment"]); report["expected_algorithm_flags"]=dict(prereg["algorithm_flags"]); observed_environment=environment_whitelist(); report["observed_environment"]=observed_environment; validate_environment(observed_environment,prereg); validate_remaining_invocation(args,prereg,mode)
        dependency = json.loads(_resolved(prereg["paths"]["dependency_manifest"]).read_text()); closure = verify_dependency_manifest(dependency)
        input_paths = {name:_resolved(path) for name,path in prereg["binding_paths"].items()}; hashes_before = {name:sha256(path) for name,path in input_paths.items()};
        if any(hashes_before[name] != prereg["bindings"][name] for name in hashes_before): raise RuntimeError("binding drift")
        report["input_hashes_before"] = {**hashes_before, **{f"dependency:{k}":v for k,v in closure.items()}}
        with h5py.File(args.source_h5,"r",swmr=True) as source:
            records = select_metadata(source); frozen = json.loads(args.selection.read_text())["records"]
            if records != frozen: raise RuntimeError("selection manifest drift")
            active = records[:1] if mode == "smoke" else records; guard = ABTruthGuard(active, mode=mode)
            times = np.asarray(source["time_s"][:],dtype=np.float64)
            if times.shape != (401,) or times[0] != 0 or times[-1] != 1 or not np.allclose(np.diff(times),.0025,rtol=0,atol=1e-15): raise RuntimeError("time contract failure")
            manifest_rows = _read_manifest_rows(args.manifest,{r["sample_id"] for r in active}); grid=AcousticGrid(nx=201,nz=201,dx_m=10,dz_m=10,lx_m=2000,lz_m=2000,centering="node"); boundary=BoundaryConfig(npml=20); device=torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
            def solver(dt,graphs): return FusedLWC84CPMLSolver(grid=grid,boundaries=boundary,dt_s=dt,output_times_s=times,c_ref_mps=6750,device=device,dtype=torch.float32,kappa_max=3,minimum_frequency_hz=8,output_restriction_factor=1,cuda_graphs=graphs,fuse_saved_interval=True)
            profile_qc=profile_relation_qc(); measurements=[]; warmups={}
            if mode=="full": solvers={"A":solver(DT_A,True),"B":solver(DT_B,True)}; warmups={k:float(v.warmup(batch=1)) for k,v in solvers.items()}
            else: solvers={"A_graph":solver(DT_A,True),"A_nongraph":solver(DT_A,False),"B_graph":solver(DT_B,True),"B_nongraph":solver(DT_B,False)}; warmups={k:float(v.warmup(batch=1)) for k,v in solvers.items()}
            accumulators={name:ExactWavefieldMetricAccumulator(energy_floor_fraction=.01,require_unique=True,stored_time_count=401) for name in ("B_vs_A","A_truth","B_truth")}
            for position,row in enumerate(active):
                i=row["source_index"]; meta={"source_x_m":float(source["source_x_m"][i]),"source_z_m":float(source["source_z_m"][i]),"source_f0_hz":float(source["source_f0_hz"][i]),"source_t0_s":float(source["source_t0_s"][i]),"source_amplitude":float(source["source_amplitude"][i]),"split":decode(source["split"][i])}; fine,_=_fine_velocity(family=row["family"],manifest_row=manifest_rows[row["sample_id"]],grid=grid,marmousi_npy=args.marmousi); sqc=source_qc(meta["source_x_m"],meta["source_z_m"])
                outputs={}; runtimes={"A":[],"B":[]}; result_metrics={}; tag_hashes={}; tag_qc={}
                sequence=raw_sequence(position) if mode=="full" else ("A_graph","A_nongraph","B_graph","B_nongraph")
                counts={"A":0,"B":0}
                for tag in sequence:
                    arm=tag[0]; torch.cuda.synchronize(device); tick=time.perf_counter(); result=solvers[tag if mode=="smoke" else arm].simulate(fine,source_x_m=meta["source_x_m"],source_z_m=meta["source_z_m"],source_f0_hz=meta["source_f0_hz"],source_t0_s=meta["source_t0_s"],source_amplitude=meta["source_amplitude"]); torch.cuda.synchronize(device); outer=time.perf_counter()-tick; pred,digest,native_qc=prediction_qc(result.wavefield[0]); counts[arm]+=1; ledger_tag=(tag if mode=="smoke" else f"{arm}_hash{counts[arm]}"); guard.register_hash(index=i,sample_id=row["sample_id"],family=row["family"],tag=ledger_tag,digest=digest,native_qc=native_qc); tag_hashes[ledger_tag]=digest; tag_qc[ledger_tag]=native_qc; outputs[tag if mode=="smoke" else f"{arm}{counts[arm]}"]=pred; runtimes[arm].append(float(outer)); result_metrics[ledger_tag]=result.metrics[0]
                    if not sqc["passed"] or abs(float(result.source_map_saved[0].sum(dtype=np.float64))-1)>2e-7: raise RuntimeError("source QC failure")
                if mode=="smoke":
                    comparisons={};
                    for arm in ("A","B"):
                        g=outputs[f"{arm}_graph"]; n=outputs[f"{arm}_nongraph"]; comparisons[arm]={"relative_l2":_relative_l2(_error_terms(g,n)),"max_abs":float(np.max(np.abs(g.astype(np.float64)-n.astype(np.float64))))}
                        if comparisons[arm]["relative_l2"]>1e-6 or comparisons[arm]["max_abs"]>1e-10: raise RuntimeError("graph/nongraph QC failure")
                    A=outputs["A_graph"]; B=outputs["B_graph"]
                else:
                    if len({tag_hashes[f"A_hash{k}"] for k in (1,2,3)})!=1 or len({tag_hashes[f"B_hash{k}"] for k in (1,2,3)})!=1: raise RuntimeError("repeat output hash drift")
                    A=outputs["A1"]; B=outputs["B1"]; comparisons=None
                physics_before_truth=pretruth_physics_qc(mode=mode,profile=profile_qc,source=sqc,result_metrics=result_metrics,graph_nongraph=comparisons)
                truth=truth_after_pretruth_qc(qc=physics_before_truth,truth_callback=lambda:guard.read_truth(source,index=i,sample_id=row["sample_id"],split=meta["split"],family=row["family"]))
                update_accumulator(accumulators["B_vs_A"],B,A,row); update_accumulator(accumulators["A_truth"],A,truth,row); update_accumulator(accumulators["B_truth"],B,truth,row)
                temporal={name:{"B_vs_A":list(_error_terms(B[s:e],A[s:e])),"A_truth":list(_error_terms(A[s:e],truth[s:e])),"B_truth":list(_error_terms(B[s:e],truth[s:e]))} for name,(s,e) in TEMPORAL_RANGES.items()}; guard.mark_complete(i)
                measurements.append({**row,"runtimes":runtimes,"tag_to_sha256":dict(sorted(tag_hashes.items())),"per_tag_native_qc":dict(sorted(tag_qc.items())),"repeat_hashes_equal":({"A":len({v for k,v in tag_hashes.items() if k.startswith('A_')})==1,"B":len({v for k,v in tag_hashes.items() if k.startswith('B_')})==1} if mode=="full" else None),"B_vs_A_terms":list(_error_terms(B,A)),"A_truth_terms":list(_error_terms(A,truth)),"B_truth_terms":list(_error_terms(B,truth)),"temporal_terms":temporal,"pretruth_physics_qc":physics_before_truth,"stability_qc":physics_before_truth["stability"],"source_qc":sqc,"graph_nongraph":comparisons})
                outputs.clear()
        stream={k:present_aware_streaming(v.finalize()) for k,v in accumulators.items()}; ba=energy_summary(measurements,"B_vs_A_terms"); at=energy_summary(measurements,"A_truth_terms"); bt=energy_summary(measurements,"B_truth_terms")
        def temporal(key):
            out={}
            for name in TEMPORAL_RANGES:
                terms=(0.,0.)
                for row in measurements: terms=_add_terms(terms,tuple(row["temporal_terms"][name][key]))
                out[name]=_relative_l2(terms)
            return out
        Araw=[v for row in measurements for v in row["runtimes"]["A"]]; Braw=[v for row in measurements for v in row["runtimes"]["B"]]; deltas=[b-a for a,b in zip(at["record_values"],bt["record_values"])]
        metrics={"runtime":{"A_mean":float(np.mean(Araw)),"A_p95":formal_p95(Araw),"B_mean":float(np.mean(Braw)),"B_p95":formal_p95(Braw),"raw_count_each":len(Araw),"p95_rank":69 if len(Araw)==72 else None},"B_vs_A":ba,"A_vs_truth":at,"B_vs_truth":bt,"B_minus_A_truth":assemble_truth_delta(at,bt,require_full=mode=="full"),"temporal_A":temporal("A_truth"),"temporal_B":temporal("B_truth"),"streaming_B_vs_A":stream["B_vs_A"],"streaming_A_truth":stream["A_truth"],"streaming_B_truth":stream["B_truth"]}
        ledger_summary=guard.summary(); expected_records=1 if mode=="smoke" else 24; expected_hashes=4 if mode=="smoke" else 144; expected_transitions=8 if mode=="smoke" else 240
        if not (ledger_summary["selected_count"]==expected_records and ledger_summary["truth_read_count"]==expected_records and ledger_summary["metrics_complete_count"]==expected_records and ledger_summary["prediction_hash_count"]==expected_hashes and len(ledger_summary["transitions"])==expected_transitions and ledger_summary["each_truth_once"]): raise RuntimeError("exact ledger count contract failure")
        qc=profile_qc["passed"] and all(row["pretruth_physics_qc"]["passed"] for row in measurements); gates=evaluate_gates(metrics,qc) if mode=="full" else {}; status=("smoke_complete" if mode=="smoke" else ("parent_gate_passed" if all(gates.values()) else "parent_gate_rejected"))
        hashes_after={name:sha256(path) for name,path in input_paths.items()}; prereg_after=sha256(args.preregistration)
        if hashes_after!=hashes_before or prereg_after!=report["preregistration_sha256_before"]: raise RuntimeError("input/prereg drift")
        resources={"elapsed_seconds":time.time()-started,"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),"output_bytes":0}; limit=180 if mode=="smoke" else 900
        if resources["elapsed_seconds"]>limit or resources["peak_allocated_bytes"]>=8*2**30: raise RuntimeError("resource budget")
        report.update({"status":status,"claim":("implementation_only" if mode=="smoke" else report["claim"]),"accuracy_gate_applied":mode=="full","records":measurements,"record_count":len(measurements),"metrics":metrics,"gates":gates,"profile_qc":profile_qc,"truth_ledger":ledger_summary,"warmups_excluded":warmups,"input_hashes_after":{**hashes_after,**{f"dependency:{k}":v for k,v in closure.items()}},"preregistration_sha256_after":prereg_after,"resources":resources})
        data=report_bytes_fixed_point(report)
        if len(data)>=32*2**20: raise RuntimeError("output budget")
        atomic_write_bytes(data,output); return 0 if status in {"smoke_complete","parent_gate_passed"} else 2
    except Exception as error:
        write_failure_atomic(report=report,error=error,started=started,guard=guard,output=output); raise


if __name__ == "__main__": raise SystemExit(main())
