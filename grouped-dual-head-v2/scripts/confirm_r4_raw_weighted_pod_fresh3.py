#!/usr/bin/env python3
"""Fresh train-distribution confirmation for the frozen raw+weighted POD arm.

Uniform and layered confirmations are existing train records absent from every
discoverable historical experiment artifact.  Marmousi is generated after
preregistration from a new deterministic crop/source specification with the
exact dataset LWC-84/free-surface/three-sided-CFS-CPML solver.  No field array
is serialized.  Future truth is used only for this offline oracle-capacity
confirmation; this is not an online adaptation method.
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
import re
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from fno_acoustic.data_generation.config import (
    boundaries_from_config,
    grid_from_config,
    time_from_config,
)
from fno_acoustic.data_generation.pipeline_lwc84 import validated_lwc84_time_plan
from fno_acoustic.data_generation.restriction import restrict_nodal_2x
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from fno_acoustic.data_generation.source import bilinear_point_source
from fno_acoustic.data_generation.velocity_models_lwc84 import load_marmousi_crop
from saved_time_phase_operator_v4.instance_adaptation.contracts import onset_indices
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4_family_temporal_pod_capacity as legacy_probe
from scripts import probe_r4_pod_metric_balance_factorial as factorial


CANDIDATE = "r4e7_raw_weighted_pod_fresh3_confirmation_v1"
SCRIPT_PATH = Path(__file__).resolve()
TEST_PATH = (
    PROJECT_ROOT
    / "tests/saved_time_phase_operator_v4/test_r4_raw_weighted_pod_fresh3.py"
)
PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_preregistration_20260826.json"
)
RESULT_PATH = (
    PROJECT_ROOT
    / "results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_20260826.json"
)
CALIBRATION_PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "results/r4e7_pod_metric_balance_factorial_calibration9_v1_preregistration_20260825.json"
)
CALIBRATION_RESULT_PATH = (
    PROJECT_ROOT
    / "results/r4e7_pod_metric_balance_factorial_calibration9_v1_20260825.json"
)
GENERATOR_CONFIG_PATH = Path(
    "/root/autodl-tmp/data/jiayh/data/"
    "acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/frozen_config.yaml"
)
MARM_SOURCE_PATH = Path(
    "/root/autodl-tmp/home/jiayh/Data/data/"
    "marmousi_zenodo_16114161/prepared/marmousi1_vp_zx_751x2301_4m.npy"
)

FAMILIES = ("uniform", "layered", "marmousi")
RANKS = (8, 16, 32)
TIME_COUNT = 401
TIME_BLOCK = 16
GPU_SECONDS_MAXIMUM = 360.0
MIN_FREE_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 10 * 1024**2
PEAK_CUDA_BYTES_MAXIMUM = int(23.5 * 1024**3)
RANK32_REDUCTION_MINIMUM = 0.20
WINNER_ARM = "raw+weighted"
STORED_CONFIRMATION_IDS = {
    "uniform": "train_uniform_00102",
    "layered": "train_layered_01032",
}
EXPECTED_STORED_GROUPS = {
    "uniform": "train:uniform:00102",
    "layered": "train:layered:00258",
}
SYNTHETIC_SPEC = {
    "family": "marmousi",
    "seed": 9102026082603,
    "crop_x0_m": 1425.0,
    "crop_z0_m": 375.0,
    "source_x_m": 1462.5,
    "source_z_m": 187.5,
    "source_f0_hz": 19.25,
    "source_t0_s": 1.5 / 19.25,
    "source_amplitude": 1.0,
    "sample_id": "synthetic_train_marmousi_fresh_v1",
    "group_id": "synthetic_train:marmousi:crop_x1425_z375:v1",
}
EXPECTED_SYNTHETIC_SPEC_SHA256 = (
    "7fc6ea09296fc764f9004ffc93cde57a369e3520a0e0844b9d1780ad7ce98588"
)
EXPECTED_SYNTHETIC_CROP_SHA256 = (
    "286c728b0bf8d184f65a4e26d62ff9ffd53a85a630d3cba4a3739f411d5cfb40"
)
EXPECTED_SYNTHETIC_NONTRUTH_SHA256 = (
    "9285101678195984ac10f773a8580b8e4239b0ebf1a98ed3c652a146ce5b44d2"
)
EXPECTED_GENERATOR_CONFIG_DIGEST = (
    "7afe021042c75360a132d345849f96d0f7ab418c9f8e59ec35170515feaee13d"
)
EXPECTED_MARM_SOURCE_SHA256 = (
    "004086fec3cb918d24b9144163bfde384883c2517d866c1a8f499f829aec0deb"
)

EXPOSURE_ROOTS = (
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "reports",
    PROJECT_ROOT / "configs",
    PROJECT_ROOT / "paper",
    Path(
        "/root/autodl-tmp/home/jiayh/Data/"
        "FNO-Acoustic-Wave-Simulation/1/pretraining"
    ),
)
EXPOSURE_SUFFIXES = frozenset(
    {".json", ".jsonl", ".md", ".yaml", ".yml", ".log", ".txt", ".csv"}
)
EXPOSURE_EXCLUDED_TOKENS = (
    "manifest.json",
    "manifest.jsonl",
    "split_snapshot",
    "dataset_inventory",
    "source_inventory",
)
SAMPLE_PATTERN = re.compile(r"train_(?:uniform|layered|marmousi)_\d{5}")
GROUP_PATTERN = re.compile(
    r"train:(?:uniform|layered):\d{5}|"
    r"train:marmousi:x-?[0-9.]+:z-?[0-9.]+"
)

CODE_BINDING_PATHS = tuple(
    dict.fromkeys(
        (
            SCRIPT_PATH,
            TEST_PATH,
            Path(factorial.__file__).resolve(),
            Path(legacy_probe.__file__).resolve(),
            Path(parent_runtime.__file__).resolve(),
            PROJECT_ROOT / "src/fno_acoustic/data_generation/solver_lwc84.py",
            PROJECT_ROOT / "src/fno_acoustic/data_generation/cpml.py",
            PROJECT_ROOT / "src/fno_acoustic/data_generation/lwc84.py",
            PROJECT_ROOT / "src/fno_acoustic/data_generation/restriction.py",
            PROJECT_ROOT / "src/fno_acoustic/data_generation/ricker.py",
            PROJECT_ROOT / "src/fno_acoustic/data_generation/source.py",
            PROJECT_ROOT / "src/fno_acoustic/data_generation/velocity_models_lwc84.py",
            PROJECT_ROOT / "src/fno_acoustic/data_generation/grid.py",
            PROJECT_ROOT / "src/fno_acoustic/data_generation/config.py",
            PROJECT_ROOT / "src/fno_acoustic/data_generation/pipeline_lwc84.py",
            *legacy_probe.CODE_BINDING_PATHS,
        )
    )
)


class FreshConfirmationError(RuntimeError):
    """The frozen fresh-confirmation contract is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    ).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def field_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(b"wavefield")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"float32")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def generator_config() -> dict[str, Any]:
    payload = yaml.safe_load(GENERATOR_CONFIG_PATH.read_text(encoding="utf8"))
    if str(payload.get("config_sha256")) != EXPECTED_GENERATOR_CONFIG_DIGEST:
        raise FreshConfirmationError("generator config digest field changed")
    required = {
        "top": "free_surface_dirichlet",
        "left": "cpml",
        "right": "cpml",
        "bottom": "cpml",
    }
    if {key: payload["boundaries"][key] for key in required} != required:
        raise FreshConfirmationError("generator boundary contract changed")
    if (
        int(payload["time"]["nt_out"]) != TIME_COUNT
        or float(payload["time"]["dt_used_s"]) != 0.000125
        or float(payload["time"]["dt_out_s"]) != 0.0025
    ):
        raise FreshConfirmationError("generator time contract changed")
    return payload


def solver_from_config(config: Mapping[str, Any], device: torch.device) -> LWC84CPMLSolver:
    plan = validated_lwc84_time_plan(dict(config))
    return LWC84CPMLSolver(
        grid=grid_from_config(dict(config)),
        boundaries=boundaries_from_config(dict(config)),
        dt_s=plan.dt_used_s,
        output_times_s=time_from_config(dict(config)).t_s,
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float32,
        kappa_max=float(config["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(config["boundaries"]["minimum_frequency_hz"]),
        output_restriction_factor=2,
    )


def record_from_manifest(manifest: Mapping[str, Any], sample_id: str) -> Any:
    matches = [row for row in manifest["records"] if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise FreshConfirmationError(f"stored confirmation is not unique: {sample_id}")
    row = matches[0]
    if row["split"] != "train" or row["medium_type"] not in FAMILIES:
        raise FreshConfirmationError(f"stored confirmation scope changed: {sample_id}")
    return parent_runtime.SelectedRecord(
        source_index=int(row["source_index"]),
        sample_id=str(row["sample_id"]),
        group_id=str(row["group_id"]),
        family=str(row["medium_type"]),
        split="train",
        split_id=int(row["split_id"]),
        manifest_sample_sha256=str(row["sample_sha256"]),
    )


def synthetic_input() -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    spec = dict(SYNTHETIC_SPEC)
    spec_sha = canonical_sha256(spec)
    if spec_sha != EXPECTED_SYNTHETIC_SPEC_SHA256:
        raise FreshConfirmationError("synthetic specification digest changed")
    config = generator_config()
    grid = grid_from_config(config)
    fine_velocity, crop_metadata = load_marmousi_crop(
        MARM_SOURCE_PATH,
        grid=grid,
        source_dx_m=4.0,
        source_dz_m=4.0,
        source_unit="m/s",
        crop_x0_m=float(spec["crop_x0_m"]),
        crop_z0_m=float(spec["crop_z0_m"]),
        interpolation="scipy_regular_grid_linear",
    )
    if crop_metadata["crop_sha256"] != EXPECTED_SYNTHETIC_CROP_SHA256:
        raise FreshConfirmationError("synthetic Marmousi crop changed")
    saved_velocity = (
        restrict_nodal_2x(torch.from_numpy(fine_velocity)[None])
        .numpy()[0]
        .astype(np.float32, copy=False)
    )
    source_map = bilinear_point_source(
        float(spec["source_x_m"]),
        float(spec["source_z_m"]),
        nx=201,
        nz=201,
        dx_m=10.0,
        dz_m=10.0,
        centering="node",
    ).source_map.astype(np.float32, copy=False)
    source_parameters = np.asarray(
        [
            spec["source_x_m"],
            spec["source_z_m"],
            spec["source_f0_hz"],
            spec["source_t0_s"],
            spec["source_amplitude"],
        ],
        dtype=np.float32,
    )
    record = parent_runtime.SelectedRecord(
        source_index=-1,
        sample_id=str(spec["sample_id"]),
        group_id=str(spec["group_id"]),
        family="marmousi",
        split="train",
        split_id=-1,
        manifest_sample_sha256=spec_sha,
    )
    digest = parent_runtime.nontruth_input_digest(
        record, saved_velocity, source_parameters, source_map
    )
    if digest != EXPECTED_SYNTHETIC_NONTRUTH_SHA256:
        raise FreshConfirmationError("synthetic parent input digest changed")
    loaded = parent_runtime.RecordInput(
        record, saved_velocity, source_parameters, source_map, digest
    )
    return record, loaded, fine_velocity, crop_metadata


def exposure_registry(manifest: Mapping[str, Any]) -> dict[str, Any]:
    train_rows = [row for row in manifest["records"] if row["split"] == "train"]
    sample_to_group = {str(row["sample_id"]): str(row["group_id"]) for row in train_rows}
    group_family = {str(row["group_id"]): str(row["medium_type"]) for row in train_rows}
    excluded_paths = {
        SCRIPT_PATH.resolve(),
        TEST_PATH.resolve(),
        PREREGISTRATION_PATH.resolve(),
        RESULT_PATH.resolve(),
    }
    groups: set[str] = set()
    scanned_files = 0
    scanned_bytes = 0
    files_with_hits = 0
    for root in EXPOSURE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.resolve() in excluded_paths
                or path.suffix.lower() not in EXPOSURE_SUFFIXES
            ):
                continue
            lowered = str(path).lower()
            if any(token in lowered for token in EXPOSURE_EXCLUDED_TOKENS):
                continue
            try:
                size = int(path.stat().st_size)
                if size > 100 * 1024**2:
                    continue
                text = path.read_text(encoding="utf8", errors="ignore")
            except OSError:
                continue
            scanned_files += 1
            scanned_bytes += size
            found = set(GROUP_PATTERN.findall(text))
            found.update(
                sample_to_group[sample]
                for sample in SAMPLE_PATTERN.findall(text)
                if sample in sample_to_group
            )
            found.intersection_update(group_family)
            if found:
                files_with_hits += 1
                groups.update(found)
    ordered = sorted(groups)
    return {
        "schema": "historical_train_group_exposure_registry_v1",
        "group_count": len(ordered),
        "group_sha256": canonical_sha256(ordered),
        "by_family": {
            family: sum(group_family[group] == family for group in ordered)
            for family in FAMILIES
        },
        "scanned_file_count": scanned_files,
        "scanned_bytes": scanned_bytes,
        "files_with_train_group_hits": files_with_hits,
        "selected_stored_group_hits": {
            family: EXPECTED_STORED_GROUPS[family] in groups
            for family in STORED_CONFIRMATION_IDS
        },
        "synthetic_group_present_in_manifest": str(SYNTHETIC_SPEC["group_id"])
        in group_family,
        "synthetic_sample_present_in_manifest": str(SYNTHETIC_SPEC["sample_id"])
        in sample_to_group,
    }


def fresh_confirmation_census(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for family in ("uniform", "layered"):
        record = record_from_manifest(manifest, STORED_CONFIRMATION_IDS[family])
        if record.group_id != EXPECTED_STORED_GROUPS[family]:
            raise FreshConfirmationError(f"fresh stored group changed: {family}")
        loaded = parent_runtime.load_record_input(record)
        rows.append(
            {
                **asdict(record),
                "role": "fresh_stored_train_confirmation",
                "future_truth_opened_before_freeze": False,
                "nontruth_input_sha256": loaded.nontruth_input_sha256,
            }
        )
    record, loaded, _, crop = synthetic_input()
    rows.append(
        {
            **asdict(record),
            "role": "fresh_synthetic_train_distribution_confirmation",
            "future_truth_opened_before_freeze": False,
            "nontruth_input_sha256": loaded.nontruth_input_sha256,
            "synthetic_spec": dict(SYNTHETIC_SPEC),
            "crop_sha256": crop["crop_sha256"],
        }
    )
    if Counter(row["family"] for row in rows) != Counter({family: 1 for family in FAMILIES}):
        raise FreshConfirmationError("fresh confirmation is not balanced")
    return rows


def _binding_map(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {str(path.resolve()): parent_runtime.file_binding(path) for path in paths}


def measured_command() -> str:
    return (
        "env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python "
        "scripts/confirm_r4_raw_weighted_pod_fresh3.py --mode measured "
        "--physical-gpu-index 0 --device cuda:0 --time-block 16 "
        "--preregistration results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_preregistration_20260826.json "
        "--result results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_20260826.json"
    )


def reproduce_exposed_marmousi(device: torch.device) -> dict[str, Any]:
    config = generator_config()
    manifest = parent_runtime.load_manifest_payload()
    record = record_from_manifest(manifest, "train_marmousi_00000")
    with h5py.File(parent_runtime.SOURCE_H5_PATH, "r", swmr=True) as handle:
        index = int(record.source_index)
        parameters = {
            "source_x_m": float(handle["source_x_m"][index]),
            "source_z_m": float(handle["source_z_m"][index]),
            "source_f0_hz": float(handle["source_f0_hz"][index]),
            "source_t0_s": float(handle["source_t0_s"][index]),
            "source_amplitude": float(handle["source_amplitude"][index]),
        }
        crop_x = float(handle["crop_x0_m"][index])
        crop_z = float(handle["crop_z0_m"][index])
        stored_velocity = np.asarray(handle["velocity_mps"][index], dtype=np.float32)
        stored_source = np.asarray(handle["source_map"][index], dtype=np.float32)
        stored_truth = np.asarray(handle["wavefield"][index], dtype=np.float32)
    fine, metadata = load_marmousi_crop(
        MARM_SOURCE_PATH,
        grid=grid_from_config(config),
        source_dx_m=4.0,
        source_dz_m=4.0,
        source_unit="m/s",
        crop_x0_m=crop_x,
        crop_z0_m=crop_z,
        interpolation="scipy_regular_grid_linear",
    )
    started = time.monotonic()
    output = solver_from_config(config, device).simulate(fine, **parameters)
    elapsed = float(time.monotonic() - started)
    actual = output.wavefield[0]
    difference = actual.astype(np.float64) - stored_truth.astype(np.float64)
    relative = float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(stored_truth.astype(np.float64).ravel()), 1.0e-30)
    )
    return {
        "sample_id": record.sample_id,
        "historically_exposed_smoke_only": True,
        "relative_l2_reproduction": relative,
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "velocity_exact": bool(np.array_equal(output.velocity_saved_mps[0], stored_velocity)),
        "source_map_exact": bool(np.array_equal(output.source_map_saved[0], stored_source)),
        "top_surface_maximum_absolute_pressure": float(np.max(np.abs(actual[:, 0, :]))),
        "elapsed_s": elapsed,
        "solver_metrics": output.metrics[0],
        "crop_sha256": metadata["crop_sha256"],
        "validation_opened": False,
        "test_id_opened": False,
    }


def run_smoke(*, physical_gpu_index: int, device: torch.device) -> dict[str, Any]:
    disk_before = parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    gpu = parent_runtime.gpu_identity(physical_gpu_index, device)
    torch.cuda.reset_peak_memory_stats(device)
    evidence = reproduce_exposed_marmousi(device)
    if (
        evidence["relative_l2_reproduction"] != 0.0
        or evidence["maximum_absolute_difference"] != 0.0
        or evidence["velocity_exact"] is not True
        or evidence["source_map_exact"] is not True
        or evidence["top_surface_maximum_absolute_pressure"] != 0.0
    ):
        raise FreshConfirmationError("exact generator reproduction smoke failed")
    return {
        "status": "passed",
        "unscored": True,
        "utc": utc_now(),
        "generator_reproduction": evidence,
        "gpu": gpu,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "disk_free_bytes_before": disk_before,
        "disk_free_bytes_after": parent_runtime.free_disk_bytes(),
        "arrays_serialized": False,
        "checkpoint_writes": 0,
        "script_sha256": parent_runtime.sha256_file(SCRIPT_PATH),
    }


def build_preregistration(
    *,
    physical_gpu_index: int,
    device: torch.device,
    smoke_evidence: Mapping[str, Any],
    focused_test_command: str,
    focused_test_result: str,
) -> dict[str, Any]:
    free = parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    if device != torch.device("cuda:0") or int(physical_gpu_index) != 0:
        raise FreshConfirmationError("frozen run requires physical GPU0 as cuda:0")
    if (
        smoke_evidence.get("status") != "passed"
        or smoke_evidence.get("unscored") is not True
        or smoke_evidence.get("script_sha256") != parent_runtime.sha256_file(SCRIPT_PATH)
    ):
        raise FreshConfirmationError("smoke evidence is invalid or stale")
    calibration = json.loads(CALIBRATION_RESULT_PATH.read_text(encoding="utf8"))
    if (
        calibration.get("status") != "success"
        or calibration.get("decision") != "winner"
        or calibration["selection"]["winner"] != WINNER_ARM
    ):
        raise FreshConfirmationError("calibration winner binding changed")
    manifest = parent_runtime.load_manifest_payload()
    registry = exposure_registry(manifest)
    if any(registry["selected_stored_group_hits"].values()):
        raise FreshConfirmationError("stored confirmation appears in historical registry")
    if registry["synthetic_group_present_in_manifest"] or registry["synthetic_sample_present_in_manifest"]:
        raise FreshConfirmationError("synthetic confirmation is not novel")
    census = fresh_confirmation_census(manifest)
    config = generator_config()
    gpu = parent_runtime.gpu_identity(physical_gpu_index, device)
    payload = {
        "schema": "r4_raw_weighted_pod_fresh3_preregistration_v1",
        "candidate": CANDIDATE,
        "status": "frozen",
        "created_utc": utc_now(),
        "hypothesis": "The raw family residual POD selected on exposed calibration retains at least 20% metric-weighted rank32 oracle relative-L2 capacity in every family on fresh train-distribution confirmations.",
        "mechanism": "Reuse only the frozen raw covariance basis and truth-energy-weighted coefficient fit. Open two historically absent train records plus one post-preregistration deterministic Marmousi simulation; change no algorithm parameter.",
        "claim_scope": "offline_fresh_train_distribution_oracle_capacity_confirmation_only_not_online_adaptation_not_validation_not_test_id",
        "winner_binding": {
            "arm": WINNER_ARM,
            "calibration_preregistration": parent_runtime.file_binding(CALIBRATION_PREREGISTRATION_PATH),
            "calibration_result": parent_runtime.file_binding(CALIBRATION_RESULT_PATH),
            "calibration_result_sha256": parent_runtime.sha256_file(CALIBRATION_RESULT_PATH),
        },
        "protocol": {
            "basis": "same two historically exposed basis records per family used by the frozen calibration; C_raw=sum_i R_i R_i^T",
            "fit": "rank32 primary; per-spatial-point weighted LS with w_t=1/max(||truth_t||_2^2,1e-30)",
            "metric": "mean over future frames after the second onset observation of spatial relative L2",
            "ranks": list(RANKS),
            "time_count": TIME_COUNT,
            "time_block": TIME_BLOCK,
            "field_serialization": False,
            "synthetic_truth_generation": "exact current 401x401 LWC-84 core, top pressure-free surface, left/right/bottom unsplit CFS-CPML, binomial5 restriction to 201x201",
        },
        "acceptance": {
            "primary_rank": 32,
            "each_family_relative_l2_reduction_minimum": RANK32_REDUCTION_MINIMUM,
            "every_record_nonworse": True,
            "all_scalars_finite": True,
            "peak_cuda_reserved_bytes_maximum": PEAK_CUDA_BYTES_MAXIMUM,
        },
        "failure_signal": "Any family below 20%, any regression, non-finite scalar, binding drift, exposure-registry drift, generator reproduction failure, validation/test access, OOM, budget, disk, output-size, or checkpoint-hash failure rejects the candidate.",
        "confirmation_census": census,
        "historical_exposure_registry": registry,
        "truth_scope": {
            "stored_future_truth_opened_before_freeze": False,
            "synthetic_future_truth_generated_before_freeze": False,
            "stored_future_truth_authorized_after_freeze": [
                STORED_CONFIRMATION_IDS["uniform"],
                STORED_CONFIRMATION_IDS["layered"],
            ],
            "synthetic_truth_authorized_after_freeze": SYNTHETIC_SPEC["sample_id"],
            "validation_opened": False,
            "test_id_opened": False,
            "online_deployment_compatible": False,
        },
        "generator_contract": {
            "config": parent_runtime.file_binding(GENERATOR_CONFIG_PATH),
            "registered_config_digest": config["config_sha256"],
            "marmousi_source": parent_runtime.file_binding(MARM_SOURCE_PATH),
            "synthetic_spec": dict(SYNTHETIC_SPEC),
            "synthetic_spec_sha256": EXPECTED_SYNTHETIC_SPEC_SHA256,
            "synthetic_crop_sha256": EXPECTED_SYNTHETIC_CROP_SHA256,
            "synthetic_nontruth_input_sha256": EXPECTED_SYNTHETIC_NONTRUTH_SHA256,
        },
        "bindings": {
            "parent_checkpoint": parent_runtime.file_binding(parent_runtime.CHECKPOINT_PATH),
            "run_identity": parent_runtime.file_binding(parent_runtime.RUN_IDENTITY_PATH),
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
            "exact_generator_reproduction_smoke": dict(smoke_evidence),
        },
        "budget": {
            "gpu_seconds_maximum": GPU_SECONDS_MAXIMUM,
            "gpu_hours_maximum": 0.10,
            "peak_cuda_reserved_bytes_maximum": PEAK_CUDA_BYTES_MAXIMUM,
            "minimum_free_disk_bytes": MIN_FREE_BYTES,
            "free_disk_bytes_at_freeze": free,
            "output_size_bytes_maximum": MAX_OUTPUT_BYTES,
        },
        "rollback": {
            "checkpoint_writes_permitted": False,
            "path": str(parent_runtime.CHECKPOINT_PATH),
            "expected_sha256": parent_runtime.CHECKPOINT_SHA256,
            "on_failure": "Write only the scalar terminal result, preserve all protected checkpoints, and do not open validation/test_id.",
        },
        "promotion_boundary": {
            "confirmation_can_authorize_validation": False,
            "pass_can_authorize_only": "design of a deployment-compatible low-dimensional coefficient predictor followed by a separate train confirmation",
            "failure_action": "reject raw+weighted temporal POD transfer without opening validation/test_id",
        },
        "measured_command": measured_command(),
    }
    parent_runtime.assert_no_placeholders_or_nulls(payload)
    return payload


def verify_frozen_preregistration(payload: Mapping[str, Any]) -> None:
    parent_runtime.assert_no_placeholders_or_nulls(payload)
    if payload.get("candidate") != CANDIDATE or payload.get("status") != "frozen":
        raise parent_runtime.BindingDriftError("fresh preregistration identity drift")
    if payload.get("measured_command") != measured_command():
        raise parent_runtime.BindingDriftError("fresh measured command drift")
    for name in (
        "parent_checkpoint",
        "run_identity",
        "config",
        "base_config",
        "parent_identity",
        "manifest_json",
        "normalization_json",
    ):
        parent_runtime.verify_binding(payload["bindings"][name])
    for binding in payload["bindings"]["code"].values():
        parent_runtime.verify_binding(binding)
    parent_runtime.verify_binding(payload["winner_binding"]["calibration_preregistration"])
    parent_runtime.verify_binding(payload["winner_binding"]["calibration_result"])
    parent_runtime.verify_binding(payload["generator_contract"]["config"])
    parent_runtime.verify_binding(payload["generator_contract"]["marmousi_source"])
    manifest = parent_runtime.load_manifest_payload()
    registry = exposure_registry(manifest)
    if registry != payload["historical_exposure_registry"]:
        raise parent_runtime.BindingDriftError("historical exposure registry drift")
    census = fresh_confirmation_census(manifest)
    if census != payload["confirmation_census"]:
        raise parent_runtime.BindingDriftError("fresh confirmation census drift")


def _old_basis_records(manifest: Mapping[str, Any], family: str) -> tuple[Any, Any]:
    records = [record for record in parent_runtime.select_train_records(manifest) if record.family == family]
    if len(records) != 3:
        raise FreshConfirmationError(f"historical basis count changed: {family}")
    return records[0], records[1]


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
        "schema": "r4_raw_weighted_pod_fresh3_result_v1",
        "candidate": CANDIDATE,
        "preregistration_path": str(preregistration_path.resolve()),
        "preregistration_sha256": prereg_sha,
        "started_utc": utc_now(),
        "claim_scope": prereg.get("claim_scope", "fresh_train_oracle_only"),
        "sealed_data_attestation": {
            "stored_future_truth_opened_after_freeze_only": True,
            "synthetic_truth_generated_after_freeze_only": True,
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
            raise parent_runtime.BindingDriftError("fresh measured time_block drift")
        verify_frozen_preregistration(prereg)
        disk_gate = parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
        gpu = parent_runtime.gpu_identity(physical_gpu_index, device)
        if gpu != prereg["bindings"]["gpu"]:
            raise parent_runtime.BindingDriftError("fresh measured GPU drift")
        budget = parent_runtime.BudgetGuard(maximum_seconds=GPU_SECONDS_MAXIMUM)
        torch.cuda.reset_peak_memory_stats(device)

        synthetic_record, synthetic_loaded, synthetic_fine, synthetic_crop = synthetic_input()
        synthetic_output = solver_from_config(generator_config(), device).simulate(
            synthetic_fine,
            source_x_m=float(SYNTHETIC_SPEC["source_x_m"]),
            source_z_m=float(SYNTHETIC_SPEC["source_z_m"]),
            source_f0_hz=float(SYNTHETIC_SPEC["source_f0_hz"]),
            source_t0_s=float(SYNTHETIC_SPEC["source_t0_s"]),
            source_amplitude=float(SYNTHETIC_SPEC["source_amplitude"]),
        )
        synthetic_truth_np = np.ascontiguousarray(synthetic_output.wavefield[0], dtype=np.float32)
        if not np.array_equal(synthetic_output.velocity_saved_mps[0], synthetic_loaded.velocity_mps):
            raise FreshConfirmationError("synthetic saved velocity differs from parent input")
        if not np.array_equal(synthetic_output.source_map_saved[0], synthetic_loaded.source_map):
            raise FreshConfirmationError("synthetic source map differs from parent input")
        if float(np.max(np.abs(synthetic_truth_np[:, 0, :]))) != 0.0:
            raise FreshConfirmationError("synthetic free surface is not exactly zero")
        synthetic_truth_hash = field_sha256(synthetic_truth_np)
        budget.check("after_synthetic_truth_generation")
        torch.cuda.empty_cache()

        model, normalizer, manifest, run_identity = parent_runtime.load_model_context(device)
        confirmations = {
            family: record_from_manifest(manifest, STORED_CONFIRMATION_IDS[family])
            for family in ("uniform", "layered")
        }
        confirmations["marmousi"] = synthetic_record
        frozen_by_family = {
            row["family"]: row for row in prereg["confirmation_census"]
        }
        time_axis = torch.tensor(manifest["time_s"], dtype=torch.float64)
        family_results: dict[str, Any] = {}
        execution = []

        for family in FAMILIES:
            covariance = torch.zeros(
                (TIME_COUNT, TIME_COUNT), dtype=torch.float32, device=device
            )
            basis_rows = []
            for basis_record in _old_basis_records(manifest, family):
                parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
                loaded = parent_runtime.load_record_input(basis_record)
                predicted = legacy_probe.generate_parent_full401(
                    model,
                    normalizer,
                    loaded,
                    manifest,
                    device=device,
                    time_block=time_block,
                )
                truth, truth_hash = legacy_probe.load_train_truth(basis_record, device=device)
                residual = truth - predicted
                legacy_probe.stream_accumulate_temporal_covariance(covariance, residual)
                basis_rows.append(
                    {
                        "sample_id": basis_record.sample_id,
                        "group_id": basis_record.group_id,
                        "train_truth_sha256": truth_hash,
                        "residual_energy": float(residual.double().square().sum().item()),
                    }
                )
                del residual, truth, predicted
                budget.check(f"after_{family}_basis_{basis_record.sample_id}")
            eigenvalues, basis = legacy_probe.temporal_pod(covariance)
            total = float(eigenvalues.sum().item())
            if not math.isfinite(total) or total <= 0.0:
                raise FreshConfirmationError(f"invalid fresh basis covariance: {family}")

            confirmation = confirmations[family]
            if family == "marmousi":
                loaded = synthetic_loaded
                truth = torch.from_numpy(synthetic_truth_np).to(device=device)
                truth_hash = synthetic_truth_hash
                truth_origin = "post_preregistration_exact_lwc84_synthetic"
                generator_qc = {
                    "crop_sha256": synthetic_crop["crop_sha256"],
                    "top_surface_maximum_absolute_pressure": float(
                        np.max(np.abs(synthetic_truth_np[:, 0, :]))
                    ),
                    "solver_metrics": synthetic_output.metrics[0],
                }
            else:
                loaded = parent_runtime.load_record_input(confirmation)
                truth, truth_hash = legacy_probe.load_train_truth(confirmation, device=device)
                truth_origin = "post_preregistration_fresh_stored_train_truth"
                generator_qc = None
            frozen = frozen_by_family[family]
            if loaded.nontruth_input_sha256 != frozen["nontruth_input_sha256"]:
                raise parent_runtime.BindingDriftError(f"fresh input drift: {family}")
            predicted = legacy_probe.generate_parent_full401(
                model,
                normalizer,
                loaded,
                manifest,
                device=device,
                time_block=time_block,
            )
            observed = onset_indices(
                time_axis,
                t0_s=float(loaded.source_parameters[3]),
                f0_hz=float(loaded.source_parameters[2]),
            )
            shared = factorial.prepare_future_projection(
                predicted, truth, future_start=int(observed[1] + 1)
            )
            metrics = factorial.fit_projection_metrics(
                shared, basis, fit_mode="weighted", ranks=RANKS
            )
            rank32 = metrics["ranks"]["32"]
            reduction = float(
                rank32["paired_changes_vs_parent"][
                    "mean_per_frame_relative_l2_reduction"
                ]
            )
            nonworse = (
                float(rank32["corrected_metrics"]["mean_per_frame_relative_l2"])
                <= float(metrics["parent_metrics"]["mean_per_frame_relative_l2"])
            )
            family_results[family] = {
                "basis_records": basis_rows,
                "basis_covariance_trace": float(
                    covariance.diagonal().double().sum().item()
                ),
                "basis_rank_energy_capture_diagnostic": {
                    str(rank): float(eigenvalues[:rank].sum().item() / total)
                    for rank in RANKS
                },
                "confirmation": {
                    **asdict(confirmation),
                    "role": frozen["role"],
                    "truth_origin": truth_origin,
                    "truth_sha256": truth_hash,
                    "observed_indices": list(observed),
                    "future_start_index": int(observed[1] + 1),
                    "generator_qc": generator_qc,
                },
                "arm": WINNER_ARM,
                "metrics": metrics,
                "rank32_gate": {
                    "relative_l2_reduction": reduction,
                    "minimum": RANK32_REDUCTION_MINIMUM,
                    "reduction_passed": reduction >= RANK32_REDUCTION_MINIMUM,
                    "nonworse": nonworse,
                    "finite": bool(rank32["finite"]),
                },
            }
            execution.append(
                {
                    "family": family,
                    "sample_id": confirmation.sample_id,
                    "group_id": confirmation.group_id,
                    "role": frozen["role"],
                    "truth_opened_or_generated_after_freeze": True,
                }
            )
            del truth, predicted, covariance, basis, shared
            budget.check(f"after_{family}_fresh_confirmation")

        torch.cuda.synchronize(device)
        elapsed = budget.check("final_aggregation")
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        gates = [family_results[family]["rank32_gate"] for family in FAMILIES]
        passed = (
            all(item["reduction_passed"] for item in gates)
            and all(item["nonworse"] for item in gates)
            and all(item["finite"] for item in gates)
            and peak_reserved <= PEAK_CUDA_BYTES_MAXIMUM
        )
        checkpoint_after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
        if checkpoint_after != parent_runtime.CHECKPOINT_SHA256:
            raise parent_runtime.BindingDriftError("parent checkpoint changed")
        payload = {
            **base,
            "status": "success",
            "decision": "pass" if passed else "fail_gate",
            "completed_utc": utc_now(),
            "exact_blocker": "none" if passed else "one or more frozen family gates failed",
            "winner_binding": prereg["winner_binding"],
            "protocol": prereg["protocol"],
            "truth_scope": prereg["truth_scope"],
            "promotion_boundary": prereg["promotion_boundary"],
            "historical_exposure_registry": prereg["historical_exposure_registry"],
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
            "execution": execution,
            "family_results": family_results,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
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
        )
    elif args.mode == "preregister":
        if not args.smoke_evidence_json:
            raise FreshConfirmationError("preregistration requires smoke evidence")
        payload = build_preregistration(
            physical_gpu_index=args.physical_gpu_index,
            device=device,
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
