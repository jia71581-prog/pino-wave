#!/usr/bin/env python3
"""Build train-only R6 device-parent residual caches with a sealed truth guard."""
from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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

from build_r25_coarse_residual_cache import static_features  # noqa: E402
from gate_lwc84_cuda_graph_fine_grid_trainonly import _fine_velocity, _read_manifest_rows  # noqa: E402
from gate_r6_anchored_r54_runtime import (  # noqa: E402
    configure_algorithm_flags,
    environment_contract,
    parent_stability_qc,
    target5_preflight,
)
from gate_r6_anchored_r54_device_runtime import device_result_qc  # noqa: E402
from reattest_frozen_fine_grid_r6_train import atomic_write_bytes, dependency_closure, report_bytes_fixed_point, sha256  # noqa: E402
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig  # noqa: E402
from fno_acoustic.data_generation.restriction import restrict_nodal_2x  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84_fused import FusedLWC84CPMLSolver  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84_fused_device import DeviceResidentFusedLWC84CPMLSolver  # noqa: E402


CANDIDATE = "r6_anchored_r54_device_resident_r1_20260905"
CACHE_SCHEMA = "r25_coarse_residual_cache_v1"
MANIFEST_SCHEMA = "r25_coarse_residual_manifest_v1"
SUMMARY_SCHEMA = "r6_device_residual_cache_summary_v1"
FAMILIES = ("uniform", "layered", "marmousi")
FIT_TIME_INDICES = (0, 6, 13, 19, 25, 32, 38, 44, 51, 57, 63, 70, 76, 83, 89, 95, 102, 108, 114, 121, 127, 133, 140, 146, 152, 159, 165, 171, 178, 184, 190, 197, 203, 210, 216, 222, 229, 235, 241, 248, 254, 260, 267, 273, 279, 286, 292, 298, 305, 311, 317, 324, 330, 337, 343, 349, 356, 362, 368, 375, 381, 387, 394, 400)
DEVELOPMENT_TIME_INDICES = tuple(range(401))
TIME_DIGESTS = {"fit": "680b8aa0e5b1fd613f8f5b25205c8d4d32e0a39b07679f46cb217185b3a29a82",
                "development": "9dfdf6adf0a9f5a5596566bf52c2ced0dc5ff6af5ca3eecec44f8904f71b4002"}
ROLE_DIGESTS = {
    "fit": {"sample_ids": "6a6edfb9149e16559ba7b8d9ab5cb060c80f690083ec658db29b1d8fe234d7bd",
            "groups": "953cbb2ed5e242b613d0dcf8b7196f11cd0e4365d20a0b095e4fe843cdaf53bb",
            "sample_hashes": "7f9018789449d8c075d6ee4c9d5f5166bde0830c450c527ff03392af99f9af15",
            "source_indices": "4c9904a3f5f0ce476be8778e54d5582b7bd5dc8c51b560925e07805fe9301055",
            "records": "f0da53220f8d6638661511e4abef8343f90e28eb62d27821c656374ee1863b14"},
    "development": {"sample_ids": "d5b4d0dc57eef029a1025bde1fa74f1b79f216ef94ca311eae43b9beb77d424e",
                    "groups": "471c86904490aefcdbdf8db1bfe807c046098712aabd6dcd31a1fbb11269ea5a",
                    "sample_hashes": "4987c3e81f88d56b53730151f6939bddef8a20bd75b9ae046a0adb9dabbb6c5b",
                    "source_indices": "862b5718e70208c4c603b8fa96ff533955896f729ac4f9724e6caac8ac2c1968",
                    "records": "c045e599fddf6d25f83bfbb102a4dc2012715902fd9a034f462c4e0cefe7d009"},
}
EXPECTED_COUNTS = {"fit": 2128, "development": 56}
EXPECTED_GROUPS = {"fit": 790, "development": 25}
EXPECTED_FAMILY = {"fit": {"uniform": 388, "layered": 1080, "marmousi": 660},
                   "development": {"uniform": 16, "layered": 20, "marmousi": 20}}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def role_digests(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {"sample_ids": canonical_sha256(sorted(str(row["sample_id"]) for row in records)),
            "groups": canonical_sha256(sorted({str(row["group_id"]) for row in records})),
            "sample_hashes": canonical_sha256(sorted(str(row["sample_sha256"]) for row in records)),
            "source_indices": canonical_sha256(sorted(int(row["source_index"]) for row in records)),
            "records": canonical_sha256(list(records))}


def load_roles(manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "frozen_before_cache_generation":
        raise RuntimeError("R38 manifest schema/status drift")
    payload = dict(manifest); declared = payload.pop("selection_sha256")
    if canonical_sha256(payload) != declared or declared != "5005f614d1ec7cbd77ddad440ac6132d0b79b5953aea70c566a8713d1433b937":
        raise RuntimeError("R38 selection digest drift")
    roles = {"fit": [dict(row) for row in manifest["fit_records"]],
             "development": [dict(row) for row in manifest["holdout_records"]]}
    for role, records in roles.items():
        counts = {family: sum(row["family"] == family for row in records) for family in FAMILIES}
        if len(records) != EXPECTED_COUNTS[role] or len({row["group_id"] for row in records}) != EXPECTED_GROUPS[role]:
            raise RuntimeError(f"{role} count/group drift")
        if counts != EXPECTED_FAMILY[role] or role_digests(records) != ROLE_DIGESTS[role]:
            raise RuntimeError(f"{role} family/digest drift")
        if any(row["split"] != "train" for row in records):
            raise RuntimeError("non-train record in cache role")
    fit_groups = {row["group_id"] for row in roles["fit"]}; dev_groups = {row["group_id"] for row in roles["development"]}
    fresh = json.loads(Path(manifest["fresh_manifest"]).read_text())
    fresh_groups = {row["group_id"] for row in fresh["holdout_records"]}
    if fit_groups & dev_groups or fit_groups & fresh_groups or dev_groups & fresh_groups:
        raise RuntimeError("fit/development/R29B group overlap")
    if sha256(Path(manifest["fresh_manifest"])) != "a350a5d04c32dfffba358577eb28dd96c456892ab696399029aa15f3cf3d42a1" or manifest["fresh_selection_sha256"] != "6ba657675616fda446801fc231914aa3a4cffeceade8a17a082822a2cbf31e6d":
        raise RuntimeError("R29B exclusion binding drift")
    return roles


def role_time_indices(role: str) -> tuple[int, ...]:
    values = FIT_TIME_INDICES if role == "fit" else DEVELOPMENT_TIME_INDICES
    if canonical_sha256(list(values)) != TIME_DIGESTS[role]:
        raise RuntimeError("time index digest drift")
    return values


def shard_records(records: Sequence[Mapping[str, Any]], shard_index: int, shard_count: int = 4) -> list[dict[str, Any]]:
    if shard_count != 4 or shard_index not in range(4):
        raise RuntimeError("cache shard contract requires exactly four workers")
    selected = [dict(row) for row in records[shard_index::shard_count]]
    expected = 532 if len(records) == 2128 else 14
    if len(selected) != expected:
        raise RuntimeError("cache shard ownership/count drift")
    return selected


class TruthGuard:
    def __init__(self, records: Sequence[Mapping[str, Any]], role: str) -> None:
        self.role = role; self.expected = {int(row["source_index"]): dict(row) for row in records}
        self.prediction_hashes: dict[int, str] = {}; self.truth_counts = {index: 0 for index in self.expected}
        self.frames = 0; self.raw_bytes = 0; self.transitions: list[dict[str, Any]] = []

    def register_parent(self, *, index: int, sample_id: str, digest: str) -> None:
        row = self.expected.get(int(index))
        if row is None or row["sample_id"] != sample_id or row["split"] != "train":
            raise PermissionError("parent identity outside authorized train role")
        if index in self.prediction_hashes or len(digest) != 64:
            raise RuntimeError("parent hash state drift")
        int(digest, 16); self.prediction_hashes[index] = digest
        self.transitions.append({"source_index": index, "to": "parent_hashed"})

    def read_truth(self, source: h5py.File, *, index: int, sample_id: str,
                   split: str, time_indices: Sequence[int]) -> np.ndarray:
        row = self.expected.get(int(index))
        if row is None or row["sample_id"] != sample_id or split != "train" or row["split"] != "train":
            raise PermissionError("truth identity/split rejected")
        if index not in self.prediction_hashes or self.truth_counts[index] != 0:
            raise PermissionError("truth requires hashed parent and exactly one read")
        truth = np.asarray(source["wavefield"][index, list(time_indices)], dtype=np.float32)
        self.truth_counts[index] += 1; self.frames += int(truth.shape[0]); self.raw_bytes += int(truth.nbytes)
        self.transitions.append({"source_index": index, "to": "truth_read_once", "frames": int(truth.shape[0])})
        return truth

    def summary(self) -> dict[str, Any]:
        ordered_hashes = [self.prediction_hashes.get(index) for index in self.expected]
        return {"role": self.role, "authorized_record_count": len(self.expected),
                "parent_hashed_count": len(self.prediction_hashes), "truth_read_count": sum(self.truth_counts.values()),
                "truth_frame_count": self.frames, "truth_raw_bytes": self.raw_bytes,
                "parent_full_sha256_digest": canonical_sha256(ordered_hashes),
                "all_parent_hashes_present_and_64hex": all(
                    isinstance(value, str) and len(value) == 64 for value in ordered_hashes
                ),
                "each_truth_once": all(value == 1 for value in self.truth_counts.values()),
                "transitions": list(self.transitions), "confirmation_opened": False,
                "validation_opened": False, "test_id_opened": False}


def normalized_cache_values(parent_full: np.ndarray, truth: np.ndarray, selected: Sequence[int]) -> dict[str, Any]:
    if parent_full.dtype != np.float32 or parent_full.shape != (401, 201, 201) or truth.dtype != np.float32:
        raise RuntimeError("cache normalization native contract")
    scale = max(float(np.max(np.abs(parent_full))), 1e-12)
    parent_norm = parent_full[list(selected)] / scale; truth_norm = truth / scale
    difference = parent_norm.astype(np.float64) - truth_norm.astype(np.float64)
    error_square = float(np.square(difference).sum()); target_square = float(np.square(truth_norm.astype(np.float64)).sum())
    energies = np.square(truth_norm.astype(np.float64)).mean(axis=(1, 2))
    return {"scale": scale, "parent_norm": parent_norm, "truth_norm": truth_norm,
            "baseline_error_square_norm": error_square, "target_square_norm": target_square,
            "truth_frame_energy_max_norm": float(energies.max()),
            "truth_frame_energy_mean_norm": float(energies.mean())}


def quantization_diagnostic(parent: np.ndarray, truth: np.ndarray, cached_parent: np.ndarray,
                            cached_truth: np.ndarray, static: np.ndarray, cached_static: np.ndarray,
                            time_indices: Sequence[int]) -> dict[str, Any]:
    """Measure smoke-only f16 storage error from actual HDF5 readback, without another truth read.

    P/T are the pre-storage normalized f32 arrays. qP/qT are stored f16 arrays
    converted to f32; differences, sums, dot products and norms then use f64.
    The record gate is deliberately separate from diagnostic temporal bands.
    """
    if parent.dtype != np.float32 or truth.dtype != np.float32 or parent.shape != truth.shape:
        raise RuntimeError("quantization diagnostic native field contract")
    if (cached_parent.dtype != np.float16 or cached_truth.dtype != np.float16
            or cached_parent.shape != parent.shape or cached_truth.shape != truth.shape
            or len(time_indices) != parent.shape[0]):
        raise RuntimeError("quantization diagnostic stored field contract")
    if any(not np.isfinite(value).all() for value in (parent, truth, cached_parent, cached_truth, static, cached_static)):
        raise RuntimeError("nonfinite quantization diagnostic input")
    qparent = cached_parent.astype(np.float32); qtruth = cached_truth.astype(np.float32)
    indices = np.asarray(time_indices, np.int64)
    frame_sums = {name: [] for name in ("truth", "parent", "native", "cached", "quantization",
                                      "parent_error", "truth_error", "residual_dot")}
    zero_truth_preserved = []
    for frame in range(len(indices)):
        p = parent[frame].astype(np.float64); t = truth[frame].astype(np.float64)
        qp = qparent[frame].astype(np.float64); qt = qtruth[frame].astype(np.float64)
        native = t - p; cached = qt - qp
        for name, value in (("truth", t), ("parent", p), ("native", native), ("cached", cached),
                            ("quantization", cached - native), ("parent_error", qp - p), ("truth_error", qt - t)):
            frame_sums[name].append(float(np.square(value).sum(dtype=np.float64)))
        frame_sums["residual_dot"].append(float(np.multiply(native, cached).sum(dtype=np.float64)))
        zero_truth_preserved.append(bool(np.all(qt[t == 0] == 0)))
    sums = {name: np.asarray(values, np.float64) for name, values in frame_sums.items()}

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        totals = {name: float(values[mask].sum(dtype=np.float64)) for name, values in sums.items()}
        def relative(numerator: str, denominator: str) -> float | None:
            return math.sqrt(totals[numerator] / totals[denominator]) if totals[denominator] > 0 else None
        native = relative("native", "truth"); error = relative("quantization", "truth")
        ratio = relative("quantization", "native")
        return {"frame_count": int(mask.sum()), "time_indices": indices[mask].tolist(),
                "R_native": native, "R_cached": relative("cached", "truth"), "E_q": error,
                "E_q_over_R_native": ratio,
                "residual_cosine": (totals["residual_dot"] / math.sqrt(totals["native"] * totals["cached"])
                                    if totals["native"] > 0 and totals["cached"] > 0 else None),
                "qP_field_relative_error": relative("parent_error", "parent"),
                "qT_field_relative_error": relative("truth_error", "truth"),
                "qP_error_over_truth_norm": relative("parent_error", "truth"),
                "square_norms_float64": totals,
                "truth_norm_zero": totals["truth"] == 0,
                "zero_truth_values_preserved": bool(all(np.asarray(zero_truth_preserved)[mask])),
                "near_zero_native_residual": native is None or native < 1e-5}

    record = metrics(np.ones(len(indices), dtype=bool))
    static_exact = (cached_static.dtype == np.float16 and cached_static.shape == static.shape
                    and cached_static.tobytes() == static.astype(np.float16).tobytes())
    ratio_applies = record["R_native"] is not None and record["R_native"] >= 1e-5
    absolute_passed = record["E_q"] is None or record["E_q"] < 1e-3
    ratio_passed = not ratio_applies or record["E_q_over_R_native"] <= .25
    passed = bool(absolute_passed and ratio_passed and record["zero_truth_values_preserved"] and static_exact)
    return {"schema": "r6_cache_quantization_diagnostic_v1", "record": record,
            "bands": {"early": metrics(indices < 134), "mid": metrics((indices >= 134) & (indices < 267)),
                      "late": metrics(indices >= 267)},
            "static_features_stored_float16_bytes_exact": bool(static_exact),
            "gate": {"passed": passed, "scope": "record_only_bands_diagnostic",
                     "E_q_lt_1e_minus_3": bool(absolute_passed), "residual_ratio_gate_applies": bool(ratio_applies),
                     "E_q_over_R_native_lte_0p25": bool(ratio_passed),
                     "zero_truth_checked_explicitly": True,
                     "failure_classification": None if passed else "cache_representation_rejected"}}


def dataset_contract(role: str, count: int) -> dict[str, dict[str, Any]]:
    frames = len(role_time_indices(role))
    return {"source_index": {"shape": [count], "dtype": "int64"},
            "sample_id": {"shape": [count], "dtype": "utf8"}, "group_id": {"shape": [count], "dtype": "utf8"},
            "family": {"shape": [count], "dtype": "utf8"}, "sample_sha256": {"shape": [count], "dtype": "utf8"},
            "parent_full_sha256": {"shape": [count], "dtype": "utf8"},
            "time_indices": {"shape": [frames], "dtype": "int64"}, "time_s": {"shape": [frames], "dtype": "float64"},
            "field_scale": {"shape": [count], "dtype": "float32"}, "source_f0_hz": {"shape": [count], "dtype": "float32"},
            "source_t0_s": {"shape": [count], "dtype": "float32"},
            "coarse_norm": {"shape": [count, frames, 201, 201], "dtype": "float16", "chunks": [1, 1, 201, 201], "compression": "lzf", "shuffle": True},
            "truth_norm": {"shape": [count, frames, 201, 201], "dtype": "float16", "chunks": [1, 1, 201, 201], "compression": "lzf", "shuffle": True},
            "static_features": {"shape": [count, 7, 201, 201], "dtype": "float16", "chunks": [1, 7, 201, 201], "compression": "lzf", "shuffle": True},
            "truth_frame_energy_max_norm": {"shape": [count], "dtype": "float32"},
            "truth_frame_energy_mean_norm": {"shape": [count], "dtype": "float32"},
            "baseline_error_square_norm": {"shape": [count], "dtype": "float64"},
            "target_square_norm": {"shape": [count], "dtype": "float64"}}


def raw_schema_bytes() -> int:
    total = 0
    for role, count in EXPECTED_COUNTS.items():
        frames = len(role_time_indices(role)); total += count * frames * 201 * 201 * 2 * 2
        total += count * 7 * 201 * 201 * 2
        total += count * (8 + 4 * 4 + 8 * 2) + frames * 16
    return int(total)


def predicted_cache_bytes() -> int:
    # HDF5 metadata, chunk-index, LZF framing, summaries, progress and atomic-final overhead.
    return raw_schema_bytes() + 603_979_776


def verify_bindings(prereg: Mapping[str, Any]) -> dict[str, str]:
    observed = {}
    for name, value in prereg["binding_paths"].items():
        path = Path(value); path = path if path.is_absolute() else ROOT / path
        observed[name] = sha256(path)
    if observed != prereg["bindings"]:
        raise RuntimeError("cache stage input binding drift")
    return observed


def verify_stage_inputs(prereg: Mapping[str, Any]) -> dict[str, Any]:
    explicit = verify_bindings(prereg)
    manifest_path = ROOT / prereg["paths"]["dependency_manifest"]
    manifest = json.loads(manifest_path.read_text())
    closure = dependency_closure([ROOT / path for path in manifest["roots"]])
    observed = {path.relative_to(ROOT).as_posix(): sha256(path) for path in closure}
    expected = {row["path"]: row["sha256"] for row in manifest["files"]}
    if observed != expected:
        raise RuntimeError("recursive cache dependency closure drift")
    for row in manifest["explicit_bindings"]:
        path = Path(row["path"]); path = path if path.is_absolute() else ROOT / path
        if sha256(path) != row["sha256"]: raise RuntimeError(f"cache dependency binding drift: {row['path']}")
    return {"explicit_bindings": explicit, "dependency_closure": observed}


def validate_environment(prereg: Mapping[str, Any], expected_visible: str) -> dict[str, Any]:
    configured = configure_algorithm_flags(prereg["environment"])
    observed = environment_contract(); expected = dict(prereg["environment"])
    if expected.pop("CUDA_VISIBLE_DEVICES") != "worker_shard_index":
        raise RuntimeError("cache CUDA visibility policy drift")
    if observed.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError("cache CUDA visibility differs from worker shard")
    comparable = dict(observed); comparable.pop("CUDA_VISIBLE_DEVICES")
    if comparable != expected:
        raise RuntimeError("cache environment drift")
    return {"algorithm_flags": configured, "observed": observed, "expected_visible": expected_visible}


def validate_invocation(args: argparse.Namespace, prereg: Mapping[str, Any], shard: int) -> None:
    if prereg.get("schema") != "r6_device_residual_cache_preregistration_v1" or prereg.get("candidate") != CANDIDATE:
        raise RuntimeError("cache preregistration schema/candidate drift")
    expected_path = (ROOT / prereg["paths"]["preregistration"]).resolve()
    if args.preregistration.resolve() != expected_path:
        raise RuntimeError("cache preregistration path override")
    if (ROOT / prereg["paths"]["stage_dir"]).resolve() != expected_path.parent:
        raise RuntimeError("cache stage directory drift")
    if args.device != "cuda:0": raise RuntimeError("cache logical device override")
    expected_status = "runtime_passed_cache_smoke_pending_audit" if args.smoke else "cache_build_authorized"
    if prereg.get("status") != expected_status: raise RuntimeError("cache stage status rejected")
    runtime = prereg["prerequisites"]["runtime"]; runtime_path = ROOT / runtime["path"]
    if runtime.get("status") != "passed" or sha256(runtime_path) != runtime.get("sha256"):
        raise RuntimeError("passed runtime prerequisite missing")
    if not args.smoke:
        smoke = prereg["prerequisites"]["smoke"]; smoke_path = ROOT / smoke["path"]
        if smoke.get("status") != "passed" or sha256(smoke_path) != smoke.get("sha256"):
            raise RuntimeError("passed cache smoke prerequisite missing")
    if shard not in range(4): raise RuntimeError("cache shard index drift")


def cache_output(stage: Path, role: str, shard: int, *, smoke: bool) -> Path:
    prefix = "smoke_" if smoke else ""
    return stage / ("smoke" if smoke else "shards") / f"{prefix}{role}_shard_{shard}.h5"


def validate_source_identity(source: h5py.File, row: Mapping[str, Any], manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    index = int(row["source_index"])
    observed = {"split": str(source["split"].asstr()[index]),
                "family": str(source["medium_type"].asstr()[index]),
                "sample_id": str(source["sample_id"].asstr()[index]),
                "group_id": str(source["group_id"].asstr()[index]),
                "sample_sha256": str(source["sample_sha256"].asstr()[index])}
    expected = {"split": "train", "family": str(row["family"]), "sample_id": str(row["sample_id"]),
                "group_id": str(row["group_id"]), "sample_sha256": str(row["sample_sha256"])}
    if observed != expected: raise RuntimeError("source metadata identity drift")
    manifest_identity = {"split": str(manifest_row.get("split")), "family": str(manifest_row.get("medium_type")),
                         "sample_id": str(manifest_row.get("sample_id")), "group_id": str(manifest_row.get("group_id"))}
    if manifest_identity != {key: expected[key] for key in ("split", "family", "sample_id", "group_id")}:
        raise RuntimeError("source manifest identity drift")
    metadata = {name: float(source[name][index]) for name in
                ("source_x_m", "source_z_m", "source_f0_hz", "source_t0_s", "source_amplitude")}
    for name, value in metadata.items():
        if not math.isfinite(value) or float(manifest_row[name]) != value:
            raise RuntimeError(f"source manifest numeric drift: {name}")
    stored = np.asarray(source["velocity_mps"][index])
    if stored.dtype != np.float32 or stored.shape != (201, 201) or not stored.flags.c_contiguous or not np.isfinite(stored).all():
        raise RuntimeError("stored velocity native contract drift")
    return {"metadata": metadata, "stored_velocity": stored, "identity": observed}


def _create_datasets(cache: h5py.File, records: Sequence[Mapping[str, Any]], role: str,
                     time_indices: Sequence[int], time_s: np.ndarray) -> dict[str, h5py.Dataset]:
    count = len(records); frames = len(time_indices); text = h5py.string_dtype("utf-8")
    for name in ("sample_id", "group_id", "family", "sample_sha256"):
        cache.create_dataset(name, data=np.asarray([row[name] for row in records], dtype=object), dtype=text)
    cache.create_dataset("source_index", data=np.asarray([row["source_index"] for row in records], np.int64))
    cache.create_dataset("time_indices", data=np.asarray(time_indices, np.int64)); cache.create_dataset("time_s", data=time_s[list(time_indices)])
    datasets = {"parent_full_sha256": cache.create_dataset("parent_full_sha256", shape=(count,), dtype=text)}
    datasets.update({name: cache.create_dataset(name, shape=(count,), dtype=dtype) for name, dtype in
                (("field_scale", np.float32), ("source_f0_hz", np.float32), ("source_t0_s", np.float32),
                 ("truth_frame_energy_max_norm", np.float32), ("truth_frame_energy_mean_norm", np.float32),
                 ("baseline_error_square_norm", np.float64), ("target_square_norm", np.float64))})
    datasets["coarse_norm"] = cache.create_dataset("coarse_norm", shape=(count, frames, 201, 201), dtype=np.float16,
                                                    chunks=(1, 1, 201, 201), compression="lzf", shuffle=True)
    datasets["truth_norm"] = cache.create_dataset("truth_norm", shape=(count, frames, 201, 201), dtype=np.float16,
                                                   chunks=(1, 1, 201, 201), compression="lzf", shuffle=True)
    datasets["static_features"] = cache.create_dataset("static_features", shape=(count, 7, 201, 201), dtype=np.float16,
                                                        chunks=(1, 7, 201, 201), compression="lzf", shuffle=True)
    return datasets


def cleanup_role_temporary(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        output = Path(kwargs["output"])
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        try:
            return function(*args, **kwargs)
        finally:
            if temporary.exists(): temporary.unlink()
    return wrapped


@cleanup_role_temporary
def build_role_cache(*, records: Sequence[Mapping[str, Any]], role: str, output: Path, source_h5: Path,
                     source_manifest: Path, marmousi: Path, device: torch.device, smoke: bool,
                     shard_index: int = 0) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    time_indices = role_time_indices(role); guard = TruthGuard(records, role); started = time.time()
    quantization_records = []
    grid = AcousticGrid(nx=401, nz=401, dx_m=5, dz_m=5, lx_m=2000, lz_m=2000, centering="node")
    with h5py.File(source_h5, "r", swmr=True) as source:
        time_s = np.asarray(source["time_s"][:], np.float64); manifest_rows = _read_manifest_rows(source_manifest, {r["sample_id"] for r in records})
        solver = DeviceResidentFusedLWC84CPMLSolver(grid=grid, boundaries=BoundaryConfig(npml=40), dt_s=.000625,
                                                     output_times_s=time_s, c_ref_mps=6750, device=device,
                                                     dtype=torch.float32, kappa_max=3, minimum_frequency_hz=8,
                                                     output_restriction_factor=2, cuda_graphs=True)
        solver.warmup(batch=1)
        legacy = FusedLWC84CPMLSolver(grid=grid, boundaries=BoundaryConfig(npml=40), dt_s=.000625,
                                      output_times_s=time_s, c_ref_mps=6750, device=device, dtype=torch.float32,
                                      kappa_max=3, minimum_frequency_hz=8, output_restriction_factor=2,
                                      cuda_graphs=True) if smoke else None
        if legacy: legacy.warmup(batch=1)
        with h5py.File(temporary, "x") as cache:
            cache.attrs.update({"schema": CACHE_SCHEMA, "status": "building", "subset": role,
                                "role": role, "parent_kind": "R6_device_resident_dt625us_restricted2x",
                                "shard_index": int(shard_index), "shard_count": 4,
                                "parent_internal_dt_s": .000625, "parent_output_restriction_factor": 2,
                                "truth_policy": "train_only_supervision_not_deployment_input",
                                "selection_sha256": "5005f614d1ec7cbd77ddad440ac6132d0b79b5953aea70c566a8713d1433b937"})
            datasets = _create_datasets(cache, records, role, time_indices, time_s)
            for position, row in enumerate(records):
                index = int(row["source_index"])
                identity = validate_source_identity(source, row, manifest_rows[row["sample_id"]])
                stored = identity["stored_velocity"]; metadata = identity["metadata"]
                fine, _ = _fine_velocity(family=row["family"], manifest_row=manifest_rows[row["sample_id"]], grid=grid, marmousi_npy=marmousi)
                if float(np.max(np.abs(restrict_nodal_2x(fine) - stored))) != 0: raise RuntimeError("velocity restriction drift")
                result = solver.simulate_device(fine, **metadata)
                parent_qc = device_result_qc(result, fine=fine, stored=stored)
                if not parent_qc["passed"]: raise RuntimeError("device parent QC failure")
                parent = result.wavefield_device.detach().cpu().numpy()[0]
                digest = hashlib.sha256(parent.tobytes()).hexdigest(); guard.register_parent(index=index, sample_id=row["sample_id"], digest=digest)
                datasets["parent_full_sha256"][position] = digest
                if not parent_stability_qc(result.metrics[0]["cfl_2d"], result.metrics[0]["lwc_qmax"])["passed"]: raise RuntimeError("parent stability drift")
                if smoke and legacy is not None:
                    legacy_parent = legacy.simulate(fine, **metadata).wavefield[0]
                    if legacy_parent.tobytes() != parent.tobytes(): raise RuntimeError("smoke legacy/device parity failure")
                truth = guard.read_truth(source, index=index, sample_id=row["sample_id"], split="train", time_indices=time_indices)
                values = normalized_cache_values(parent, truth, time_indices)
                datasets["field_scale"][position] = values["scale"]; datasets["source_f0_hz"][position] = metadata["source_f0_hz"]; datasets["source_t0_s"][position] = metadata["source_t0_s"]
                datasets["coarse_norm"][position] = values["parent_norm"].astype(np.float16); datasets["truth_norm"][position] = values["truth_norm"].astype(np.float16)
                axis = np.linspace(0, 2000, 201, dtype=np.float32)
                static = static_features(stored, x_m=axis, z_m=axis, source_x_m=metadata["source_x_m"], source_z_m=metadata["source_z_m"])
                datasets["static_features"][position] = static.astype(np.float16)
                for name in ("truth_frame_energy_max_norm", "truth_frame_energy_mean_norm", "baseline_error_square_norm", "target_square_norm"): datasets[name][position] = values[name]
                cache.flush()
                if smoke:
                    cached_parent = datasets["coarse_norm"][position]; cached_truth = datasets["truth_norm"][position]
                    if (cached_parent.tobytes() != values["parent_norm"].astype(np.float16).tobytes()
                            or cached_truth.tobytes() != values["truth_norm"].astype(np.float16).tobytes()):
                        raise RuntimeError("smoke stored float16 byte reconstruction failure")
                    diagnostic = quantization_diagnostic(values["parent_norm"], values["truth_norm"],
                                                         cached_parent, cached_truth, static,
                                                         datasets["static_features"][position], time_indices)
                    diagnostic.update({"sample_id": row["sample_id"], "source_index": index,
                                       "family": row["family"], "role": role, "additional_truth_reads": 0})
                    quantization_records.append(diagnostic)
                progress = {"schema": "r6_device_residual_cache_progress_v1", "role": role,
                            "shard_index": int(shard_index), "completed_records": position + 1,
                            "record_count": len(records), "last_sample_id": row["sample_id"],
                            "parent_hashed_before_truth": True, "resources": {"output_bytes": 0}}
                atomic_write_bytes(report_bytes_fixed_point(progress), output.with_suffix(".progress.json"))
            cache.attrs["status"] = "complete"; cache.attrs["truth_ledger_json"] = json.dumps(guard.summary(), sort_keys=True); cache.flush()
    os.replace(temporary, output)
    summary = {"schema": SUMMARY_SCHEMA, "status": "complete", "role": role, "record_count": len(records),
               "frame_count_per_record": len(time_indices), "output": str(output), "output_sha256": sha256(output),
               "output_bytes": output.stat().st_size, "truth_ledger": guard.summary(),
               "elapsed_seconds": time.time() - started, "smoke": smoke,
               "legacy_device_parity_passed": bool(smoke), "stored_float16_bytes_reconstructed": bool(smoke),
               "validation_opened": False, "test_id_opened": False}
    if smoke:
        summary["quantization_diagnostics"] = quantization_records
        summary["cache_representation_gate_passed"] = all(row["gate"]["passed"] for row in quantization_records)
    atomic_write_bytes(report_bytes_fixed_point(summary), output.with_suffix(".summary.json")); return summary


def aggregate_worker_terminals(stage: Path) -> bool:
    terminal_dir = stage / "terminals"
    expected = [terminal_dir / f"worker_{index}.json" for index in range(4)]
    present = sorted(terminal_dir.glob("worker_*.json")) if terminal_dir.exists() else []
    if any(path not in expected for path in present): raise RuntimeError("unexpected worker terminal")
    if not all(path.is_file() for path in expected): return False
    lock = stage / ".build_terminal_aggregator.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(descriptor)
    except FileExistsError:
        return False
    try:
        workers = []
        for index, path in enumerate(expected):
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete" or payload.get("shard_index") != index:
                raise RuntimeError("worker terminal status/shard drift")
            if len(payload.get("summaries", [])) != 2 or payload.get("input_hashes_unchanged") is not True:
                raise RuntimeError("worker terminal completeness drift")
            for summary in payload["summaries"]:
                output = Path(summary["output"])
                if summary.get("status") != "complete" or not output.is_file() or sha256(output) != summary["output_sha256"]:
                    raise RuntimeError("worker terminal output hash drift")
            workers.append({"shard_index": index, "path": str(path), "sha256": sha256(path)})
        final = {"schema": "r6_device_residual_cache_build_terminal_v1", "status": "complete",
                 "worker_count": 4, "worker_terminals": workers,
                 "confirmation_opened": False, "validation_opened": False, "test_id_opened": False,
                 "resources": {"output_bytes": 0}}
        output = stage / "build_terminal.json"
        if output.exists(): raise FileExistsError(output)
        atomic_write_bytes(report_bytes_fixed_point(final), output); return True
    finally:
        if lock.exists(): lock.unlink()


def write_worker_terminal(stage: Path, shard: int, summaries: Sequence[Mapping[str, Any]],
                          input_hashes_before: Mapping[str, Any], input_hashes_after: Mapping[str, Any],
                          peak_allocated_bytes: int) -> None:
    terminal = {"schema": "r6_device_residual_cache_worker_terminal_v1", "status": "complete",
                "shard_index": shard, "summaries": list(summaries),
                "input_hashes_before": dict(input_hashes_before), "input_hashes_after": dict(input_hashes_after),
                "input_hashes_unchanged": input_hashes_before == input_hashes_after,
                "resources": {"peak_allocated_bytes": peak_allocated_bytes, "output_bytes": 0},
                "promotion_authorized": False, "confirmation_opened": False,
                "validation_opened": False, "test_id_opened": False}
    atomic_write_bytes(report_bytes_fixed_point(terminal), stage / "terminals" / f"worker_{shard}.json")
    aggregate_worker_terminals(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true"); modes.add_argument("--worker", action="store_true")
    parser.add_argument("--preregistration", type=Path, required=True); parser.add_argument("--shard-index", type=int)
    parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text()); stage = ROOT / "results/r6_anchored_r54_device_resident_r1_20260905/cache_stage"
    shard = 0 if args.smoke else args.shard_index
    if shard is None or shard not in range(4): raise RuntimeError("worker shard index required")
    terminal = stage / ("smoke_terminal.json" if args.smoke else f"terminals/worker_{shard}.json")
    if terminal.exists(): raise FileExistsError(terminal)
    started = time.time(); before = None
    try:
        validate_invocation(args, prereg, shard)
        environment = validate_environment(prereg, str(shard)); before = verify_stage_inputs(prereg)
        target = target5_preflight(prereg)
        roles = load_roles(ROOT / prereg["paths"]["R38_manifest"]); device = torch.device(args.device)
        torch.cuda.set_device(device); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps({"event": "R6_DEVICE_CACHE_WORKER_START", "candidate": CANDIDATE,
                          "mode": "smoke" if args.smoke else "worker", "shard_index": shard}), flush=True)
        summaries = []
        if args.smoke:
            for role in ("fit", "development"):
                records = [next(row for row in roles[role] if row["family"] == family) for family in FAMILIES]
                summaries.append(build_role_cache(records=records, role=role, output=cache_output(stage, role, 0, smoke=True),
                                                   source_h5=Path(prereg["paths"]["source_h5"]), source_manifest=Path(prereg["paths"]["source_manifest"]),
                                                   marmousi=Path(prereg["paths"]["marmousi"]), device=device, smoke=True, shard_index=0))
            after = verify_stage_inputs(prereg)
            elapsed = time.time() - started; peak = int(torch.cuda.max_memory_allocated(device))
            if elapsed > 180 or peak >= 8 * 2**30: raise RuntimeError("cache smoke resource budget")
            representation_passed = all(summary["cache_representation_gate_passed"] for summary in summaries)
            payload = {"schema": "r6_device_residual_cache_smoke_terminal_v1",
                       "status": "complete" if representation_passed else "cache_representation_rejected",
                       "cache_representation_gate_passed": representation_passed,
                       "operator_failure_claimed": False, "smoke_artifacts_preserved": True,
                       "record_count": 6, "summaries": summaries, "legacy_device_parity": True,
                       "input_hashes_before": before, "input_hashes_after": after, "input_hashes_unchanged": before == after,
                       "environment": environment, "target5_audit": target,
                       "resources": {"elapsed_seconds": elapsed, "peak_allocated_bytes": peak, "output_bytes": 0},
                       "promotion_authorized": False, "confirmation_opened": False,
                       "validation_opened": False, "test_id_opened": False}
            atomic_write_bytes(report_bytes_fixed_point(payload), terminal)
            if not representation_passed: return 2
        else:
            for role in ("fit", "development"):
                summaries.append(build_role_cache(records=shard_records(roles[role], shard), role=role,
                                                   output=cache_output(stage, role, shard, smoke=False),
                                                   source_h5=Path(prereg["paths"]["source_h5"]), source_manifest=Path(prereg["paths"]["source_manifest"]),
                                                   marmousi=Path(prereg["paths"]["marmousi"]), device=device, smoke=False, shard_index=shard))
            after = verify_stage_inputs(prereg)
            elapsed = time.time() - started; peak = int(torch.cuda.max_memory_allocated(device))
            if elapsed > 1800 or peak >= 8 * 2**30: raise RuntimeError("cache worker resource budget")
            write_worker_terminal(stage, shard, summaries, before, after, peak)
        return 0
    except Exception as error:
        for role in ("fit", "development"):
            path = cache_output(stage, role, shard, smoke=args.smoke)
            temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            if temporary.exists(): temporary.unlink()
        if not terminal.exists():
            failure = {"schema": "r6_device_residual_cache_terminal_v1", "status": "failed",
                       "mode": "smoke" if args.smoke else "worker", "shard_index": shard,
                       "error": repr(error), "input_hashes_before": before,
                       "resources": {"elapsed_seconds": time.time() - started, "output_bytes": 0},
                       "promotion_authorized": False, "confirmation_opened": False,
                       "validation_opened": False, "test_id_opened": False}
            atomic_write_bytes(report_bytes_fixed_point(failure), terminal)
        raise


if __name__ == "__main__": raise SystemExit(main())
