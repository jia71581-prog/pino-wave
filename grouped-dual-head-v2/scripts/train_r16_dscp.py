#!/usr/bin/env python3
"""Preparation CLI for frozen ``r16_dscp_v1``.

This revision implements immutable preflight, train-only panel construction,
deterministic train-only POD basis construction, static evidence, and final
preregistration.  It intentionally has no predictor training mode.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from saved_time_phase_operator_v4.instance_adaptation.contracts import onset_indices
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (
    CONDITION_MAXIMUM,
    FAMILIES,
    MODEL_MACS_201,
    R16DSCP,
    RANK,
    RidgePointwiseBaseline,
    TEMPORAL_MATERIALIZATION_MACS_201,
    TEMPORAL_PROJECTION_MACS_201,
    analytic_model_macs,
    basis_condition,
    predictor_parameter_count,
    velocity_route,
)
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4_family_temporal_pod_capacity as legacy_pod


CANDIDATE = "r16_dscp_v1"
SEED = 372
SCRIPT_PATH = Path(__file__).resolve()
MODULE_PATH = PROJECT_ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py"
TEST_PATH = PROJECT_ROOT / "tests/saved_time_phase_operator_v4/test_r16_dscp.py"
CONFIG_PATH = PROJECT_ROOT / "configs/r16_dscp_v1.yaml"
RESULT_DIR = PROJECT_ROOT / "results/r16_dscp_v1"
PREFLIGHT_PATH = RESULT_DIR / "design_freeze_preflight.json"
PREFLIGHT_CORRECTION_PATH = RESULT_DIR / "preflight_correction_v2.json"
PREFLIGHT_CORRECTION_V3_PATH = RESULT_DIR / "preflight_correction_v3.json"
PANELS_PATH = RESULT_DIR / "panels.json"
BASIS_PATH = RESULT_DIR / "basis_rank16.pt"
STATIC_PATH = RESULT_DIR / "static_evidence.json"
PREREG_PATH = PROJECT_ROOT / "results/r16_dscp_v1_preregistration_20260826.json"
ORACLE_RESULT_PATH = PROJECT_ROOT / "results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_20260826.json"
ORACLE_PREREG_PATH = PROJECT_ROOT / "results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_preregistration_20260826.json"
POD_RESULT_PATH = PROJECT_ROOT / "results/r4e7_family_temporal_pod_capacity_train9_v1_20260825.json"
POD_PREREG_PATH = PROJECT_ROOT / "results/r4e7_family_temporal_pod_capacity_train9_v1_preregistration_20260825.json"
PARENT_RUNTIME_RESULT = PROJECT_ROOT / "results/r4e7_parent_e2e_runtime_train9_v1_20260825.json"

PANEL_COUNTS = (
    ("smoke", 1),
    ("pilot_fit", 8),
    ("pilot_confirm", 8),
    ("long_fit", 64),
    ("long_calibration", 8),
    ("final_train_confirm", 8),
)
PANEL_COUNT_PER_FAMILY = sum(value for _, value in PANEL_COUNTS)
MAX_SMALL_ARTIFACT_BYTES = 64 * 1024**2
MIN_PREPARATION_FREE_BYTES = 2 * 1024**3
PROTECTED_PARENT_SHA256 = parent_runtime.CHECKPOINT_SHA256


class R16PreparationError(RuntimeError):
    """The frozen preparation contract failed."""


class SealedTruthError(R16PreparationError):
    """A non-train or unregistered truth access was attempted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf8"
    )
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def file_binding(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    stat = target.stat()
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def atomic_json_exclusive(payload: Mapping[str, Any], path: Path) -> int:
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"refuse overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf8"
    )
    if len(encoded) > MAX_SMALL_ARTIFACT_BYTES:
        raise R16PreparationError("small-artifact size budget exceeded")
    partial = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return len(encoded)


def atomic_torch_exclusive(payload: Mapping[str, Any], path: Path) -> int:
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"refuse overwrite: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"artifact directory absent: {destination.parent}")
    partial = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if partial.stat().st_size > MAX_SMALL_ARTIFACT_BYTES:
            raise R16PreparationError("basis artifact exceeds 64 MiB")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return int(destination.stat().st_size)


def require_disk(minimum: int = MIN_PREPARATION_FREE_BYTES) -> int:
    free = int(shutil.disk_usage(PROJECT_ROOT).free)
    if free < int(minimum):
        raise R16PreparationError(f"free disk {free} is below required {int(minimum)}")
    return free


def _selected_record(row: Mapping[str, Any]) -> parent_runtime.SelectedRecord:
    family = row.get("medium_type", row.get("family"))
    sample_sha256 = row.get("sample_sha256", row.get("manifest_sample_sha256"))
    if row.get("split") != "train" or family not in FAMILIES:
        raise SealedTruthError("record is outside the train-only family scope")
    if sample_sha256 is None:
        raise R16PreparationError("record is missing sample hash binding")
    return parent_runtime.SelectedRecord(
        source_index=int(row["source_index"]),
        sample_id=str(row["sample_id"]),
        group_id=str(row["group_id"]),
        family=str(family),
        split="train",
        split_id=int(row["split_id"]),
        manifest_sample_sha256=str(sample_sha256),
    )


def _manifest_rows_by_sample(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {str(row["sample_id"]): row for row in manifest["records"]}
    if len(rows) != len(manifest["records"]):
        raise R16PreparationError("manifest sample IDs are not unique")
    return rows


def basis_records(manifest: Mapping[str, Any]) -> dict[str, list[parent_runtime.SelectedRecord]]:
    pod_result = json.loads(POD_RESULT_PATH.read_text(encoding="utf8"))
    rows_by_sample = _manifest_rows_by_sample(manifest)
    output: dict[str, list[parent_runtime.SelectedRecord]] = {family: [] for family in FAMILIES}
    for row in pod_result["sample_census"]["execution"]:
        if row["role"] != "basis":
            continue
        sample_id = str(row["sample_id"])
        record = _selected_record(rows_by_sample[sample_id])
        if record.family != str(row["family"]):
            raise R16PreparationError("POD basis family binding changed")
        output[record.family].append(record)
    if any(len(output[family]) != 2 for family in FAMILIES):
        raise R16PreparationError("exactly two historical POD basis records are required")
    return output


def excluded_groups(manifest: Mapping[str, Any]) -> dict[str, set[str]]:
    rows_by_sample = _manifest_rows_by_sample(manifest)
    pod_result = json.loads(POD_RESULT_PATH.read_text(encoding="utf8"))
    fresh_result = json.loads(ORACLE_RESULT_PATH.read_text(encoding="utf8"))
    output = {family: set() for family in FAMILIES}
    for row in pod_result["sample_census"]["execution"]:
        manifest_row = rows_by_sample[str(row["sample_id"])]
        output[str(row["family"])].add(str(manifest_row["group_id"]))
    for row in fresh_result["execution"]:
        output[str(row["family"])].add(str(row["group_id"]))
    return output


def _route_one_velocity(velocity: np.ndarray) -> dict[str, Any]:
    decision = velocity_route(torch.from_numpy(np.asarray(velocity, dtype=np.float32))[None, None])[0]
    return {
        "route": decision.name,
        "route_index": decision.index,
        "frac_dx": decision.frac_dx,
        "frac_dz": decision.frac_dz,
        "abstain": decision.abstain,
    }


def build_panels(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    exclusions = excluded_groups(manifest)
    rows_by_family_group: dict[str, dict[str, Mapping[str, Any]]] = {
        family: {} for family in FAMILIES
    }
    for row in manifest["records"]:
        family = str(row.get("medium_type"))
        if row.get("split") != "train" or family not in FAMILIES:
            continue
        group = str(row["group_id"])
        current = rows_by_family_group[family].get(group)
        if current is None or (int(row["source_index"]), str(row["sample_id"])) < (
            int(current["source_index"]),
            str(current["sample_id"]),
        ):
            rows_by_family_group[family][group] = row

    selected: list[dict[str, Any]] = []
    router_counts: dict[str, Counter[str]] = {family: Counter() for family in FAMILIES}
    router_extrema: dict[str, dict[str, float]] = {
        family: {
            "frac_dx_minimum": math.inf,
            "frac_dx_maximum": -math.inf,
            "frac_dz_minimum": math.inf,
            "frac_dz_maximum": -math.inf,
        }
        for family in FAMILIES
    }
    h5_path = parent_runtime.SOURCE_H5_PATH
    with h5py.File(h5_path, "r", swmr=True) as handle:
        for family in FAMILIES:
            groups = rows_by_family_group[family]
            eligible = [group for group in groups if group not in exclusions[family]]
            eligible.sort(
                key=lambda group: hashlib.sha256(
                    f"r16-dscp-v1|372|{group}".encode("utf8")
                ).hexdigest()
            )
            if len(eligible) < PANEL_COUNT_PER_FAMILY:
                raise R16PreparationError(f"insufficient train groups for {family}")
            role_by_position: list[str] = []
            for role, count in PANEL_COUNTS:
                role_by_position.extend([role] * count)
            for position, group in enumerate(eligible):
                row = groups[group]
                velocity = np.asarray(handle["velocity_mps"][int(row["source_index"])], dtype=np.float32)
                route = _route_one_velocity(velocity)
                router_counts[family][route["route"]] += 1
                extrema = router_extrema[family]
                for axis in ("dx", "dz"):
                    value = float(route[f"frac_{axis}"])
                    extrema[f"frac_{axis}_minimum"] = min(extrema[f"frac_{axis}_minimum"], value)
                    extrema[f"frac_{axis}_maximum"] = max(extrema[f"frac_{axis}_maximum"], value)
                if position >= PANEL_COUNT_PER_FAMILY:
                    continue
                record = _selected_record(row)
                loaded = parent_runtime.load_record_input(record)
                selected.append(
                    {
                        **asdict(record),
                        "role": role_by_position[position],
                        "order_sha256": hashlib.sha256(
                            f"r16-dscp-v1|372|{group}".encode("utf8")
                        ).hexdigest(),
                        "nontruth_input_sha256": loaded.nontruth_input_sha256,
                        "route": route,
                    }
                )

    role_family = Counter((row["role"], row["family"]) for row in selected)
    for role, count in PANEL_COUNTS:
        for family in FAMILIES:
            if role_family[(role, family)] != count:
                raise R16PreparationError("balanced panel role census failed")
    groups = [row["group_id"] for row in selected]
    samples = [row["sample_id"] for row in selected]
    if len(groups) != len(set(groups)) or len(samples) != len(set(samples)):
        raise R16PreparationError("panels are not globally group/sample-disjoint")
    manifest_group_split = defaultdict(set)
    for row in manifest["records"]:
        manifest_group_split[str(row["group_id"])].add(str(row["split"]))
    sealed_hits = [
        row for row in selected if manifest_group_split[row["group_id"]] != {"train"}
    ]
    if sealed_hits:
        raise SealedTruthError("selected panel group overlaps validation or test_id")

    panels = {
        "schema": "r16_dscp_train_panels_v1",
        "candidate": CANDIDATE,
        "seed": SEED,
        "selection": "sha256('r16-dscp-v1|372|group_id'), one lowest-source-index record per group",
        "exclusions": {family: sorted(exclusions[family]) for family in FAMILIES},
        "counts_per_family": dict(PANEL_COUNTS),
        "records": selected,
        "census": {
            "total": len(selected),
            "by_family": dict(Counter(row["family"] for row in selected)),
            "by_role": dict(Counter(row["role"] for row in selected)),
            "unique_groups": len(set(groups)),
            "unique_samples": len(set(samples)),
            "validation_records": 0,
            "test_id_records": 0,
        },
        "manifest_digest": str(manifest["digest"]),
    }
    router = {
        "schema": "r16_dscp_velocity_router_train_census_v1",
        "scope": "one deterministic representative per train group; velocity only",
        "thresholds": {
            "constant": "frac_dx == 0 and frac_dz == 0 -> uniform",
            "layered": "frac_dx <= 0.05 and frac_dz <= 0.10",
            "marmousi": "frac_dx >= 0.15 and frac_dz >= 0.50",
            "gray": "abstain exact parent",
            "absolute_gradient_threshold": 1.0e-6,
        },
        "by_manifest_train_family": {
            family: dict(sorted(router_counts[family].items())) for family in FAMILIES
        },
        "gradient_fraction_extrema": router_extrema,
        "truth_opened": False,
        "validation_opened": False,
        "test_id_opened": False,
    }
    return panels, router


def software_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": str(torch.version.cuda),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "yaml": yaml.__version__,
    }


def _code_bindings() -> dict[str, dict[str, Any]]:
    paths = (MODULE_PATH, SCRIPT_PATH, TEST_PATH, CONFIG_PATH)
    return {str(path.resolve()): file_binding(path) for path in paths}


def _source_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stat = parent_runtime.SOURCE_H5_PATH.stat()
    return {
        "path": str(parent_runtime.SOURCE_H5_PATH.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
        "registered_source_file_sha256": str(manifest["source_file_sha256"]),
        "registered_source_manifest_sha256": str(manifest["source_manifest_sha256"]),
        "registered_source_config_sha256": str(manifest["source_config_sha256"]),
        "bytes_not_rehashed": True,
    }


def create_preflight() -> None:
    if RESULT_DIR.exists() or PREREG_PATH.exists():
        raise FileExistsError("r16 preparation target already exists; refuse overwrite")
    free = require_disk()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf8"))
    if config.get("candidate") != CANDIDATE or int(config["basis"]["rank"]) != RANK:
        raise R16PreparationError("config candidate or rank binding changed")
    if sha256_file(parent_runtime.CHECKPOINT_PATH) != PROTECTED_PARENT_SHA256:
        raise R16PreparationError("protected parent hash mismatch")
    manifest = parent_runtime.load_manifest_payload()
    panels, router = build_panels(manifest)
    exact_basis = basis_records(manifest)
    run_identity = json.loads(parent_runtime.RUN_IDENTITY_PATH.read_text(encoding="utf8"))
    payload = {
        "schema": "r16_dscp_design_freeze_preflight_v1",
        "candidate": CANDIDATE,
        "status": "immutable_design_freeze_pre_basis",
        "created_utc": utc_now(),
        "hypothesis": "A 1,202-parameter velocity-routed rank-16 coefficient predictor can recover deployment-compatible train-confirmed temporal-residual gains from two observed frames without future-truth inference access and without material parent efficiency regression.",
        "basis_algorithm": {
            "records": {
                family: [asdict(record) for record in exact_basis[family]] for family in FAMILIES
            },
            "parent": "frozen full-401 physical output, FP32, time_block=16",
            "truth_scope": "train only; all 401 frames used offline only for raw residual POD",
            "covariance": "C_family=sum_records R R^T over spatial chunks; R=true_train-parent",
            "covariance_dtype": "float32",
            "eigendecomposition": "symmetric FP64 eigh descending",
            "rank": 16,
            "sign_rule": "largest-absolute temporal entry is positive",
            "coefficient_scale": "3*RMS(B^T R) across exactly the two family basis records and all spatial points; clamp minimum 1e-12",
            "serialized": "only [3,401,16] basis, [3,16] scales, eigenvalues and scalar metadata; no fields/residual/coefficient maps",
            "optimizer": False,
            "backward": False,
        },
        "design": config,
        "router_train_census": router,
        "bindings": {
            "parent_checkpoint": file_binding(parent_runtime.CHECKPOINT_PATH),
            "oracle_result": file_binding(ORACLE_RESULT_PATH),
            "oracle_preregistration": file_binding(ORACLE_PREREG_PATH),
            "pod_result": file_binding(POD_RESULT_PATH),
            "pod_preregistration": file_binding(POD_PREREG_PATH),
            "parent_runtime_result": file_binding(PARENT_RUNTIME_RESULT),
            "manifest": file_binding(parent_runtime.MANIFEST_PATH),
            "normalization": file_binding(parent_runtime.NORMALIZATION_PATH),
            "run_identity": file_binding(parent_runtime.RUN_IDENTITY_PATH),
            "active_config": file_binding(parent_runtime.CONFIG_PATH),
            "base_config": file_binding(parent_runtime.BASE_CONFIG_PATH),
            "source_data": _source_binding(manifest),
            "new_code_config": _code_bindings(),
        },
        "run_identity": {
            "manifest_digest": run_identity["manifest_digest"],
            "run_digest": run_identity["run_digest"],
            "time_axis_sha256": run_identity["time_axis_sha256"],
        },
        "software": software_identity(),
        "disk_free_bytes": free,
        "sealed_attestation": {
            "train_future_truth_authorized_for_basis_only": True,
            "validation_truth_opened": False,
            "test_id_truth_opened": False,
            "predictor_training_run": False,
            "gpu_training_run": False,
        },
        "reproduce": {
            "static_tests": "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. pytest -q -p no:cacheprovider tests/saved_time_phase_operator_v4/test_r16_dscp.py",
            "preflight": "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python scripts/train_r16_dscp.py --mode preflight",
            "basis": "env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python scripts/train_r16_dscp.py --mode build-basis --physical-gpu-index 0 --device cuda:0",
            "freeze": "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python scripts/train_r16_dscp.py --mode freeze-prereg",
        },
    }
    RESULT_DIR.mkdir(parents=False, exist_ok=False)
    atomic_json_exclusive(panels, PANELS_PATH)
    payload["bindings"]["panels"] = file_binding(PANELS_PATH)
    atomic_json_exclusive(payload, PREFLIGHT_PATH)
    print(json.dumps({"status": "frozen_preflight", "preflight": file_binding(PREFLIGHT_PATH)}))


def _verify_preflight() -> dict[str, Any]:
    payload = json.loads(PREFLIGHT_PATH.read_text(encoding="utf8"))
    if payload.get("candidate") != CANDIDATE or payload.get("status") != "immutable_design_freeze_pre_basis":
        raise R16PreparationError("preflight identity changed")
    correction: dict[str, Any] | None = None
    script_path = str(SCRIPT_PATH.resolve())
    for path, binding in payload["bindings"]["new_code_config"].items():
        actual = sha256_file(path)
        if actual == binding["sha256"]:
            continue
        if path != script_path or correction is not None or not PREFLIGHT_CORRECTION_PATH.is_file():
            raise R16PreparationError(f"post-freeze code/config drift: {path}")
        correction = json.loads(PREFLIGHT_CORRECTION_PATH.read_text(encoding="utf8"))
        import inspect

        build_source_sha256 = hashlib.sha256(
            inspect.getsource(build_basis).encode("utf8")
        ).hexdigest()
        basis_artifact = _load_basis()
        exact_v2 = {
            "schema": "r16_dscp_preflight_schema_correction_v2",
            "status": "frozen_pre_prereg_schema_correction",
            "script_sha256_before": str(binding["sha256"]),
            "script_sha256_after": correction.get("script_sha256_after"),
            "preflight_sha256": sha256_file(PREFLIGHT_PATH),
            "panels_sha256": sha256_file(PANELS_PATH),
            "basis_file_sha256": sha256_file(BASIS_PATH),
            "basis_tensor_sha256": basis_artifact["basis_tensor_sha256"],
            "build_basis_source_sha256_before": build_source_sha256,
            "build_basis_source_sha256_after": build_source_sha256,
            "module_sha256": sha256_file(MODULE_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "test_sha256": sha256_file(TEST_PATH),
        }
        if any(correction.get(key) != value for key, value in exact_v2.items()):
            raise R16PreparationError("preflight correction does not exactly attest the single script drift")
        if correction.get("allowed_diff_scope") != [
            "_selected_record accepts frozen offline panel key family as alias for medium_type",
            "_verify_preflight fail-closed verifier accepts only this exclusive correction artifact",
        ]:
            raise R16PreparationError("preflight correction diff scope changed")
        if not PREFLIGHT_CORRECTION_V3_PATH.is_file():
            raise R16PreparationError("preflight correction v3 chain artifact is absent")
        correction_v3 = json.loads(PREFLIGHT_CORRECTION_V3_PATH.read_text(encoding="utf8"))
        exact_v3 = {
            "schema": "r16_dscp_preflight_schema_correction_v3",
            "status": "frozen_pre_prereg_schema_correction",
            "script_sha256_before": correction["script_sha256_after"],
            "script_sha256_after": actual,
            "correction_v2_sha256": sha256_file(PREFLIGHT_CORRECTION_PATH),
            "preflight_sha256": sha256_file(PREFLIGHT_PATH),
            "panels_sha256": sha256_file(PANELS_PATH),
            "basis_file_sha256": sha256_file(BASIS_PATH),
            "basis_tensor_sha256": basis_artifact["basis_tensor_sha256"],
            "build_basis_source_sha256_before": build_source_sha256,
            "build_basis_source_sha256_after": build_source_sha256,
            "module_sha256": sha256_file(MODULE_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "test_sha256": sha256_file(TEST_PATH),
        }
        if any(correction_v3.get(key) != value for key, value in exact_v3.items()):
            raise R16PreparationError("preflight correction v3 does not exactly attest the correction chain")
        if correction_v3.get("allowed_diff_scope") != [
            "_selected_record accepts frozen manifest_sample_sha256 as alias for sample_sha256",
            "_verify_preflight requires exact old-to-v2-to-v3 correction chain",
        ]:
            raise R16PreparationError("preflight correction v3 diff scope changed")
        payload["bindings"]["preflight_correction_v2"] = file_binding(
            PREFLIGHT_CORRECTION_PATH
        )
        payload["bindings"]["preflight_correction_v3"] = file_binding(
            PREFLIGHT_CORRECTION_V3_PATH
        )
    if sha256_file(PANELS_PATH) != payload["bindings"]["panels"]["sha256"]:
        raise R16PreparationError("panel binding drift")
    if sha256_file(parent_runtime.CHECKPOINT_PATH) != PROTECTED_PARENT_SHA256:
        raise R16PreparationError("protected parent hash mismatch")
    return payload


def _fix_eigenvector_signs(basis: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(basis, device="cpu", dtype=torch.float64).clone()
    for mode in range(value.shape[1]):
        column = value[:, mode]
        pivot = int(column.abs().argmax())
        if float(column[pivot]) < 0.0:
            value[:, mode].neg_()
    return value


@torch.inference_mode()
def build_basis(physical_gpu_index: int, device: torch.device) -> None:
    if BASIS_PATH.exists() or STATIC_PATH.exists() or PREREG_PATH.exists():
        raise FileExistsError("basis/static/prereg artifact already exists; refuse overwrite")
    preflight = _verify_preflight()
    free_before = require_disk()
    if int(physical_gpu_index) != 0 or device != torch.device("cuda:0"):
        raise R16PreparationError("basis build is frozen to physical GPU0 as cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu = parent_runtime.gpu_identity(physical_gpu_index, device)
    model, normalizer, manifest, run_identity = parent_runtime.load_model_context(device)
    records = basis_records(manifest)
    bases: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    eigenvalues_all: list[torch.Tensor] = []
    metadata: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats(device)
    for family in FAMILIES:
        covariance = torch.zeros(
            (legacy_pod.TIME_COUNT, legacy_pod.TIME_COUNT), device=device, dtype=torch.float32
        )
        residuals_cpu: list[torch.Tensor] = []
        record_metadata: list[dict[str, Any]] = []
        for record in records[family]:
            loaded = parent_runtime.load_record_input(record)
            parent = legacy_pod.generate_parent_full401(
                model,
                normalizer,
                loaded,
                manifest,
                device=device,
                time_block=legacy_pod.TIME_BLOCK,
            )
            truth, truth_hash = legacy_pod.load_train_truth(record, device=device)
            residual = (truth - parent).contiguous()
            if not bool(torch.isfinite(residual).all()):
                raise FloatingPointError("basis residual contains NaN or Inf")
            legacy_pod.stream_accumulate_temporal_covariance(covariance, residual)
            residuals_cpu.append(residual.cpu())
            record_metadata.append(
                {
                    **asdict(record),
                    "nontruth_input_sha256": loaded.nontruth_input_sha256,
                    "train_truth_sha256": truth_hash,
                    "residual_energy": float(residual.double().square().sum().item()),
                }
            )
            del parent, truth, residual
            torch.cuda.empty_cache()
        eigenvalues, eigenvectors = legacy_pod.temporal_pod(covariance)
        family_basis64 = _fix_eigenvector_signs(eigenvectors[:, :RANK])
        family_basis = family_basis64.float().contiguous()
        coefficient_square_sum = torch.zeros(RANK, dtype=torch.float64)
        coefficient_count = 0
        for residual in residuals_cpu:
            coefficients = family_basis.T @ residual.reshape(legacy_pod.TIME_COUNT, -1)
            coefficient_square_sum += coefficients.double().square().sum(dim=1)
            coefficient_count += coefficients.shape[1]
            del coefficients
        family_scale = 3.0 * torch.sqrt(
            coefficient_square_sum / float(coefficient_count)
        ).clamp_min(1.0e-12)
        bases.append(family_basis)
        scales.append(family_scale.float())
        eigenvalues_all.append(eigenvalues[:RANK].float())
        metadata[family] = {
            "records": record_metadata,
            "covariance_trace": float(torch.trace(covariance.double()).item()),
            "rank16_energy_capture": float(
                eigenvalues[:RANK].sum().item() / max(eigenvalues.sum().item(), 1.0e-300)
            ),
            "full_basis_orthogonality_max_abs": float(
                (family_basis64.T @ family_basis64 - torch.eye(RANK, dtype=torch.float64))
                .abs()
                .max()
                .item()
            ),
            "coefficient_scale_minimum": float(family_scale.min()),
            "coefficient_scale_maximum": float(family_scale.max()),
        }
        del covariance, residuals_cpu
    basis_tensor = torch.stack(bases).contiguous()
    scale_tensor = torch.stack(scales).contiguous()
    eigenvalue_tensor = torch.stack(eigenvalues_all).contiguous()
    if not bool(torch.isfinite(basis_tensor).all() and torch.isfinite(scale_tensor).all()):
        raise FloatingPointError("basis artifact contains NaN or Inf")
    artifact = {
        "schema": "r16_dscp_trainonly_basis_v1",
        "candidate": CANDIDATE,
        "created_utc": utc_now(),
        "family_order": FAMILIES,
        "basis": basis_tensor,
        "coefficient_scales": scale_tensor,
        "leading_eigenvalues": eigenvalue_tensor,
        "metadata": metadata,
        "basis_tensor_sha256": tensor_sha256(basis_tensor),
        "coefficient_scales_tensor_sha256": tensor_sha256(scale_tensor),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "panels_sha256": sha256_file(PANELS_PATH),
        "parent_path": str(parent_runtime.CHECKPOINT_PATH),
        "parent_sha256": PROTECTED_PARENT_SHA256,
        "manifest_digest": manifest["digest"],
        "run_digest": run_identity["run_digest"],
        "time_axis_sha256": run_identity["time_axis_sha256"],
        "truth_scope": "six exact train basis records only; no validation/test_id",
        "serialized_arrays": ["basis[3,401,16]", "coefficient_scales[3,16]", "leading_eigenvalues[3,16]"],
        "coefficient_maps_serialized": False,
        "fields_serialized": False,
        "optimizer_used": False,
        "backward_used": False,
        "gpu": gpu,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "disk_free_bytes_before": free_before,
        "disk_free_bytes_after_compute": int(shutil.disk_usage(PROJECT_ROOT).free),
    }
    atomic_torch_exclusive(artifact, BASIS_PATH)
    if sha256_file(parent_runtime.CHECKPOINT_PATH) != PROTECTED_PARENT_SHA256:
        raise R16PreparationError("protected parent changed during basis build")
    print(
        json.dumps(
            {
                "status": "basis_built",
                "basis": file_binding(BASIS_PATH),
                "basis_tensor_sha256": artifact["basis_tensor_sha256"],
                "peak_cuda_reserved_bytes": artifact["peak_cuda_reserved_bytes"],
            }
        )
    )


def _load_basis() -> dict[str, Any]:
    artifact = torch.load(BASIS_PATH, map_location="cpu", weights_only=False)
    if artifact.get("candidate") != CANDIDATE or artifact.get("schema") != "r16_dscp_trainonly_basis_v1":
        raise R16PreparationError("basis artifact identity changed")
    if tensor_sha256(artifact["basis"]) != artifact["basis_tensor_sha256"]:
        raise R16PreparationError("basis tensor digest mismatch")
    if tensor_sha256(artifact["coefficient_scales"]) != artifact["coefficient_scales_tensor_sha256"]:
        raise R16PreparationError("coefficient scale digest mismatch")
    return artifact


def _basis_panel_conditions(bases: torch.Tensor, panels: Mapping[str, Any]) -> dict[str, Any]:
    manifest = parent_runtime.load_manifest_payload()
    time_axis = torch.tensor(manifest["time_s"], dtype=torch.float64)
    by_family: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for row in panels["records"]:
        family = str(row["family"])
        record = _selected_record(row)
        loaded = parent_runtime.load_record_input(record)
        k0, k1 = onset_indices(
            time_axis,
            t0_s=float(loaded.source_parameters[3]),
            f0_hz=float(loaded.source_parameters[2]),
        )
        by_family[family].append(basis_condition(bases[FAMILIES.index(family)], k1))
    return {
        family: {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
            "all_at_most_1e3": all(value <= CONDITION_MAXIMUM for value in values),
        }
        for family, values in by_family.items()
    }


def _inference_signature_audit() -> dict[str, Any]:
    forbidden = ("truth", "target", "family", "medium_type", "split", "sample", "group", "oracle")
    audited = {}
    for name in ("forward", "predict_coefficients"):
        parameters = list(inspect.signature(getattr(R16DSCP, name)).parameters)
        hits = [parameter for parameter in parameters if any(token in parameter.lower() for token in forbidden)]
        if hits:
            raise R16PreparationError(f"forbidden deployment signature fields: {name} {hits}")
        audited[name] = parameters
    return {"methods": audited, "forbidden_hits": [], "passed": True}


def _oracle_summary() -> dict[str, Any]:
    oracle = json.loads(ORACLE_RESULT_PATH.read_text(encoding="utf8"))
    runtime = json.loads(PARENT_RUNTIME_RESULT.read_text(encoding="utf8"))
    families = {}
    for family in FAMILIES:
        rank32 = oracle["family_results"][family]["metrics"]["ranks"]["32"]
        families[family] = {
            "rank32_mean_frame_relative_l2_reduction": rank32["paired_changes_vs_parent"][
                "mean_per_frame_relative_l2_reduction"
            ],
            "corrected_mean_frame_relative_l2": rank32["corrected_metrics"][
                "mean_per_frame_relative_l2"
            ],
            "basis_design_condition_number": rank32["basis_design_condition_number"],
            "weighted_design_condition_number": rank32["weighted_design_condition_number"],
            "finite": rank32["finite"],
        }
    return {
        "families": families,
        "oracle_peak_cuda_allocated_bytes": oracle["peak_cuda_allocated_bytes"],
        "oracle_peak_cuda_reserved_bytes": oracle["peak_cuda_reserved_bytes"],
        "oracle_gpu_seconds": oracle["budget"]["one_gpu_seconds_used"],
        "parent_runtime_mean_s": runtime["aggregate"]["arithmetic_mean_runtime_s"],
        "parent_runtime_p95_s": runtime["aggregate"]["nearest_rank_p95_runtime_s"],
        "parent_peak_cuda_reserved_bytes": runtime["aggregate"]["measured_peak_cuda_reserved_bytes"],
        "parent_speedup_vs_traditional_x": runtime["aggregate"]["reference_over_mean_speedup_x"],
        "claim_boundary": "Offline fresh train-distribution oracle capacity only. It uses future truth to fit per-spatial-point coefficients and is not deployable, not validation, not test_id. The parent is only 1.608x faster than the traditional reference; R16-DSCP cannot claim 10x or faster-than-parent without measured complete E2E evidence.",
    }


def freeze_prereg() -> None:
    if STATIC_PATH.exists() or PREREG_PATH.exists():
        raise FileExistsError("static/prereg artifact already exists; refuse overwrite")
    preflight = _verify_preflight()
    basis = _load_basis()
    panels = json.loads(PANELS_PATH.read_text(encoding="utf8"))
    conditions = _basis_panel_conditions(basis["basis"], panels)
    if not all(row["all_at_most_1e3"] for row in conditions.values()):
        raise R16PreparationError("basis conditioning gate failed on train panels")
    model = R16DSCP(basis["basis"], basis["coefficient_scales"])
    ridge = RidgePointwiseBaseline()
    parameters = predictor_parameter_count(model)
    ridge_parameters = predictor_parameter_count(ridge)
    if parameters != 1202 or ridge_parameters != 480 or analytic_model_macs() != MODEL_MACS_201:
        raise R16PreparationError("parameter or MAC contract failed")
    parent_after = sha256_file(parent_runtime.CHECKPOINT_PATH)
    if parent_after != PROTECTED_PARENT_SHA256:
        raise R16PreparationError("protected parent hash changed")
    disk_free = int(shutil.disk_usage(PROJECT_ROOT).free)
    static = {
        "schema": "r16_dscp_static_evidence_v1",
        "candidate": CANDIDATE,
        "created_utc": utc_now(),
        "parameters": parameters,
        "ridge_parameters": ridge_parameters,
        "model_only_macs_201x201": analytic_model_macs(),
        "all_time_projection_macs": TEMPORAL_PROJECTION_MACS_201,
        "all_time_materialization_macs": TEMPORAL_MATERIALIZATION_MACS_201,
        "basis_conditions_on_all_291_frozen_train_panel_records": conditions,
        "inference_signature": _inference_signature_audit(),
        "basis_binding": file_binding(BASIS_PATH),
        "basis_tensor_sha256": basis["basis_tensor_sha256"],
        "coefficient_scales_tensor_sha256": basis["coefficient_scales_tensor_sha256"],
        "panels_binding": file_binding(PANELS_PATH),
        "preflight_binding": file_binding(PREFLIGHT_PATH),
        "parent_sha256_after": parent_after,
        "parent_unchanged": True,
        "sealed_attestation": {
            "validation_truth_opened": False,
            "test_id_truth_opened": False,
            "coefficient_maps_serialized": False,
            "fields_serialized": False,
            "predictor_training_run": False,
        },
        "disk_free_bytes": disk_free,
    }
    atomic_json_exclusive(static, STATIC_PATH)

    space_formula = 2 * 1024**3 + 3 * (2 * 1024**2) + 64 * 1024**2
    prereg = {
        "schema": "r16_dscp_preregistration_v1",
        "candidate": CANDIDATE,
        "status": "frozen",
        "created_utc": utc_now(),
        "hypothesis": preflight["hypothesis"],
        "mechanism": "A velocity-only family router selects one immutable train-only raw residual POD basis. A 1,202-parameter local depthwise/pointwise network maps 29 permitted channels from velocity/source/travel/coordinates/two observed residual frames/frozen-parent projections/route one-hot to rank-16 spatial coefficient maps. Gray routes or condition>1e3 abstain to exact parent.",
        "claim_boundary": _oracle_summary()["claim_boundary"],
        "oracle_capacity_verification": _oracle_summary(),
        "design": preflight["design"],
        "panels": {
            "binding": file_binding(PANELS_PATH),
            "census": panels["census"],
            "counts_per_family": panels["counts_per_family"],
            "selection": panels["selection"],
            "exclusions": panels["exclusions"],
            "all_roles_group_disjoint": True,
            "validation_records": 0,
            "test_id_records": 0,
        },
        "basis": {
            "binding": file_binding(BASIS_PATH),
            "basis_tensor_sha256": basis["basis_tensor_sha256"],
            "coefficient_scales_tensor_sha256": basis["coefficient_scales_tensor_sha256"],
            "family_order": list(basis["family_order"]),
            "conditions": conditions,
            "truth_scope": basis["truth_scope"],
            "coefficient_maps_serialized": False,
            "fields_serialized": False,
        },
        "gates": {
            "static": [
                "future-truth mutation invariance",
                "HDF5 observed reads exactly {k0,k1}",
                "registered onset contract",
                "no future truth/family/id fields in inference signatures",
                "C1 causal correction zero through k1 and top row zero",
                "zero coefficient/state is bitwise exact parent",
                "finite and basis condition <=1e3 else abstain",
                "exact parameters=1202, model-only MACs=45451125, ridge parameters=480",
            ],
            "unconditional_reject": [
                "NaN/Inf",
                "OOM",
                "leakage or non-train development truth access",
                "binding or protected-parent hash drift",
                "condition>1e3 without exact-parent abstention",
                "atomic-resume contract failure",
                "disk-space gate failure",
                "material efficiency regression",
            ],
            "smoke": {
                "scope": "frozen smoke panel, train only, one group per family",
                "maximum_steps": 500,
                "maximum_wall_time_s": 600,
                "loss_decrease_minimum": 0.80,
                "each_family_fraction_of_rank16_oracle_framewise_gain_minimum": 0.50,
                "aggregate_nonworse": True,
            },
            "pilot_confirmation": {
                "scope": "24 frozen disjoint train confirmation records",
                "joint_aggregate_paired_relative_l2_improvement_minimum": 0.01,
                "each_family_paired_improvement_minimum": 0.005,
                "nonworse_records_minimum": 23,
                "record_count": 24,
                "margin_over_ridge_percentage_points_minimum": 0.25,
                "late_third_nonworse": True,
                "high_band_nonworse": True,
                "correction_energy_ratio_maximum": 0.35,
                "finite": True,
            },
            "efficiency": {
                "adapter_mean_s_maximum": 0.35,
                "adapter_nearest_rank_p95_s_maximum": 0.50,
                "e2e_mean_ratio_to_parent_maximum": 1.05,
                "e2e_p95_ratio_to_parent_maximum": 1.05,
                "inference_peak_bytes_maximum": int(6.5 * 1024**3),
                "training_peak_bytes_maximum": 8 * 1024**3,
                "checkpoint_bytes_maximum": 2 * 1024**2,
                "new_artifacts_bytes_maximum": 64 * 1024**2,
            },
            "one_vs_four_gpu_probe": {
                "records": "first 12 long_fit records in frozen panel order",
                "warmup_runs": 1,
                "timed_runs": 3,
                "four_gpu_speedup_minimum": 3.0,
                "four_gpu_efficiency_minimum": 0.75,
                "four_gpu_gpu_seconds_ratio_maximum": 1.35,
                "metric_and_hash_identity_required": True,
                "peak_bytes_per_gpu_maximum": int(6.5 * 1024**3),
                "fallback": "one GPU",
            },
            "post_long_sequence": "After final_train_confirm passes, evaluate validation exactly once. Lock candidate, thresholds, and checkpoint hash before opening test_id; test_id exactly once only if validation passes. Any change requires a new version.",
        },
        "training_contract": {
            "precision": "BF16 autocast; reductions and metrics FP32/FP64",
            "seed": 372,
            "deterministic": True,
            "tf32": False,
            "workers_per_gpu_maximum": 2,
            "checkpoint": "atomic resumable best/last only: predictor+basis+optimizer+scaler+RNG+sampler/progress; parent path/hash only and never embedded",
            "one_gpu_wall_time_maximum_s": 7200,
            "four_gpu_wall_time_maximum_s": 2700,
            "total_gpu_hours_maximum": 3.0,
            "space_gate": {
                "formula": "free>=2GiB+3*C+64MiB",
                "conservative_C_bytes": 2 * 1024**2,
                "required_bytes": space_formula,
                "free_bytes_at_freeze": disk_free,
                "currently_passed": disk_free >= space_formula,
            },
        },
        "metrics_required": [
            "paired per-family and aggregate relative L2",
            "early/middle/late including late-third stability",
            "low/mid/high spectrum including high-band stability",
            "per-record nonworse and failure cases",
            "correction-energy ratio",
            "adapter and complete E2E mean/nearest-rank P95 latency",
            "peak allocated/reserved VRAM",
            "parameter/MAC/checkpoint/artifact size",
        ],
        "bindings": {
            **preflight["bindings"],
            "basis": file_binding(BASIS_PATH),
            "static_evidence": file_binding(STATIC_PATH),
            "preflight": file_binding(PREFLIGHT_PATH),
            "panels": file_binding(PANELS_PATH),
        },
        "protected_parent": {
            "rollback_path": str(parent_runtime.CHECKPOINT_PATH),
            "sha256_before": PROTECTED_PARENT_SHA256,
            "sha256_after": parent_after,
            "unchanged": True,
            "embedded_in_candidate_checkpoint": False,
        },
        "software": software_identity(),
        "gpu_basis_build": basis["gpu"],
        "reproduce": preflight["reproduce"],
        "sealed_attestation": {
            "basis_truth": "six exact train basis records only",
            "panel_truth_opened": False,
            "validation_truth_opened": False,
            "test_id_truth_opened": False,
            "future_truth_in_inference": False,
            "predictor_training_started": False,
        },
    }
    atomic_json_exclusive(prereg, PREREG_PATH)
    print(
        json.dumps(
            {
                "status": "frozen",
                "static": file_binding(STATIC_PATH),
                "preregistration": file_binding(PREREG_PATH),
            }
        )
    )


def read_two_observed_frames(
    source_h5: str | Path,
    source_index: int,
    k0: int,
    k1: int,
    *,
    split: str,
    access_log: list[int] | None = None,
) -> np.ndarray:
    """Read exactly the two registered observed frames, never a future slice."""
    if str(split) != "train":
        raise SealedTruthError("pre-validation preparation may read train observations only")
    if int(k1) != int(k0) + 1 or int(k0) < 0:
        raise ValueError("observed indices must be adjacent and nonnegative")
    values = []
    with h5py.File(Path(source_h5), "r", swmr=True) as handle:
        for index in (int(k0), int(k1)):
            if access_log is not None:
                access_log.append(index)
            values.append(np.asarray(handle["wavefield"][int(source_index), index], dtype=np.float32))
    output = np.stack(values)
    if not np.isfinite(output).all():
        raise FloatingPointError("observed frames contain NaN or Inf")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("preflight", "build-basis", "freeze-prereg")
    )
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        create_preflight()
    elif args.mode == "build-basis":
        build_basis(args.physical_gpu_index, torch.device(args.device))
    elif args.mode == "freeze-prereg":
        freeze_prereg()
    else:  # pragma: no cover
        raise AssertionError("unreachable mode")


if __name__ == "__main__":
    main()
