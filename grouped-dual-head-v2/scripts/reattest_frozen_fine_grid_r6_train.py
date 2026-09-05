#!/usr/bin/env python3
"""Current-environment reattest of frozen R6 on an exposed train panel."""
from __future__ import annotations

import argparse
import ast
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

from audit_target5 import audit as audit_target5  # noqa: E402
from gate_lwc84_cuda_graph_fine_grid_trainonly import (  # noqa: E402
    _add_terms,
    _error_terms,
    _fine_velocity,
    _nearest_rank,
    _read_manifest_rows,
    _relative_l2,
)
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig  # noqa: E402
from fno_acoustic.data_generation.restriction import restrict_nodal_2x  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84_fused import (  # noqa: E402
    FusedLWC84CPMLSolver,
)


CANDIDATE = "frozen_fine_grid_r6_current_env_reattest_v1_20260905"
NAMESPACE = "R6_current_environment_reattest_v1"
FAMILIES = ("uniform", "layered", "marmousi")
GROUP_SIZES = {"uniform": 1, "layered": 4, "marmousi": 5}
GROUP_TAKES = {"uniform": 20, "layered": 5, "marmousi": 4}
EXPECTED_DIGESTS = {
    "indices": "fd5cac5d541dd65bbe15dbe79e4334992492346ff4e5eba9e9ea0869b2241664",
    "sample_ids": "024fb03cdb2f685c2a52b8cbbdb603fafdf89180723b31e2801144913bff43a8",
    "groups_first_occurrence": "08172ad3cb404858412ee708337eb72eb0b30b9cf1821cbc12774253f7141e3d",
    "sample_hashes": "cced838665a127c477d304e54d453fd230b3d62022baf675e49f54b1bb031b49",
    "records": "2083c95b9742c9eb4b0c42efb2f340377a8a554016fa2153c562d01c1b342b98",
}
RUNTIME_LIMIT_S = 2.034137312322855
TEMPORAL_RANGES = {"early": (0, 134), "middle": (134, 267), "late": (267, 401)}


def decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def serialize_report(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def report_bytes_fixed_point(payload: dict[str, Any]) -> bytes:
    resources = payload.setdefault("resources", {})
    resources["output_bytes"] = int(resources.get("output_bytes", 0))
    for _ in range(16):
        encoded = serialize_report(payload)
        if resources["output_bytes"] == len(encoded):
            return encoded
        resources["output_bytes"] = len(encoded)
    raise RuntimeError("report output_bytes fixed point did not converge")


def atomic_write_bytes(data: bytes, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale JSON staging path: {staging}")
    try:
        with staging.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def first_occurrence(values: Sequence[str]) -> list[str]:
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
            first_occurrence([str(row["group_id"]) for row in records])
        ),
        "sample_hashes": canonical_sha256([str(row["sample_sha256"]) for row in records]),
        "records": canonical_sha256(list(records)),
    }


def select_from_metadata(handle: h5py.File) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, list[int]]] = {
        family: defaultdict(list) for family in FAMILIES
    }
    for index in range(int(handle["split"].shape[0])):
        if decode(handle["split"][index]) != "train":
            continue
        family = decode(handle["medium_type"][index])
        if family in groups:
            groups[family][decode(handle["group_id"][index])].append(index)
    records: list[dict[str, Any]] = []
    for family in FAMILIES:
        eligible = [
            group_id
            for group_id, members in groups[family].items()
            if len(members) == GROUP_SIZES[family]
        ]
        eligible.sort(
            key=lambda group_id: (
                hashlib.sha256(
                    (NAMESPACE + "\0" + family + "\0" + group_id).encode("utf-8")
                ).digest(),
                group_id,
            )
        )
        chosen = eligible[: GROUP_TAKES[family]]
        if len(chosen) != GROUP_TAKES[family]:
            raise RuntimeError(f"insufficient eligible groups for {family}")
        for group_id in chosen:
            members = sorted(
                groups[family][group_id],
                key=lambda index: (int(index), decode(handle["sample_id"][index])),
            )
            for index in members:
                records.append(
                    {
                        "source_index": int(index),
                        "sample_id": decode(handle["sample_id"][index]),
                        "group_id": group_id,
                        "family": family,
                        "sample_sha256": decode(handle["sample_sha256"][index]),
                    }
                )
    if len(records) != 60 or selection_digests(records) != EXPECTED_DIGESTS:
        raise RuntimeError("frozen train selection digest drift")
    return records


def validate_selection_manifest(source_h5: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("namespace") != NAMESPACE or manifest.get("split") != "train":
        raise RuntimeError("selection manifest namespace/split drift")
    with h5py.File(source_h5, "r", swmr=True) as handle:
        records = select_from_metadata(handle)
    if records != manifest.get("records") or selection_digests(records) != manifest.get("digests"):
        raise RuntimeError("selection manifest records/digests drift")
    counts = {family: sum(row["family"] == family for row in records) for family in FAMILIES}
    groups = {family: len({row["group_id"] for row in records if row["family"] == family}) for family in FAMILIES}
    if counts != {family: 20 for family in FAMILIES} or groups != GROUP_TAKES:
        raise RuntimeError("selection family/group count drift")
    return records


def _module_path(module: str) -> Path | None:
    candidates = []
    if "." not in module:
        candidates.append(ROOT / "scripts" / module)
    if module == "scripts" or module.startswith("scripts."):
        suffix = module.split(".")[1:]
        candidates.append(ROOT / "scripts" / Path(*suffix))
    if module == "fno_acoustic" or module.startswith("fno_acoustic."):
        suffix = module.split(".")[1:]
        candidates.append(ROOT / "src/fno_acoustic" / Path(*suffix))
    for base in candidates:
        file_path = base.with_suffix(".py")
        package_path = base / "__init__.py"
        if file_path.is_file():
            return file_path.resolve()
        if package_path.is_file():
            return package_path.resolve()
    return None


def _module_name(path: Path) -> str | None:
    path = path.resolve()
    try:
        rel = path.relative_to(ROOT / "src")
        return ".".join(rel.with_suffix("").parts)
    except ValueError:
        pass
    try:
        rel = path.relative_to(ROOT)
        if rel.parts[0] == "scripts":
            return ".".join(rel.with_suffix("").parts)
    except ValueError:
        pass
    return None


def local_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current = _module_name(path)
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
                prefix = package[: max(keep, 0)]
                base = ".".join([*prefix, base] if base else prefix)
            if base:
                modules.append(base)
                modules.extend(f"{base}.{alias.name}" for alias in node.names)
        for module in modules:
            dependency = _module_path(module)
            if dependency is not None:
                result.add(dependency)
    return result


def dependency_closure(roots: Sequence[Path]) -> list[Path]:
    pending = [path.resolve() for path in roots]
    closure: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(ROOT)
        except ValueError as error:
            raise RuntimeError(f"dependency lies outside workspace: {path}") from error
        if relative.parts[0] not in {"scripts", "src"}:
            raise RuntimeError(f"dependency outside permitted source roots: {relative}")
        closure.add(path)
        pending.extend(sorted(local_imports(path)))
        if relative.parts[0] == "scripts":
            initializer = ROOT / "scripts/__init__.py"
            if initializer.is_file():
                pending.append(initializer.resolve())
        elif relative.parts[:2] == ("src", "fno_acoustic"):
            parent = path.parent
            package_root = (ROOT / "src/fno_acoustic").resolve()
            while parent == package_root or package_root in parent.parents:
                initializer = parent / "__init__.py"
                if initializer.is_file():
                    pending.append(initializer.resolve())
                if parent == package_root:
                    break
                parent = parent.parent
    return sorted(closure, key=lambda path: path.relative_to(ROOT).as_posix())


def verify_dependency_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    roots = [(ROOT / path).resolve() for path in manifest["roots"]]
    closure = dependency_closure(roots)
    observed = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in closure
    }
    expected = {row["path"]: row["sha256"] for row in manifest["files"]}
    if observed != expected:
        raise RuntimeError("recursive local dependency closure drift")
    for row in manifest.get("explicit_bindings", []):
        path = _resolved(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"explicit dependency binding drift: {row['name']}")
    return observed


class TruthStateGuard:
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.expected = {int(row["source_index"]): dict(row) for row in records}
        if len(self.expected) != len(records):
            raise RuntimeError("duplicate selected source index")
        self.states = {index: "metadata_loaded" for index in self.expected}
        self.transitions: list[dict[str, Any]] = []
        self.truth_counts = {index: 0 for index in self.expected}
        self.prediction_hashes: dict[int, str] = {}

    def register_prediction(self, *, index: int, sample_id: str, family: str, digest: str) -> None:
        row = self.expected.get(int(index))
        if row is None or row["sample_id"] != sample_id or row["family"] != family:
            raise PermissionError("prediction identity differs from frozen selection")
        if self.states[index] != "metadata_loaded" or len(digest) != 64:
            raise RuntimeError("prediction hash registration state/digest invalid")
        int(digest, 16)
        self.prediction_hashes[index] = digest
        self._transition(index, "prediction_hashed")

    def read_truth(
        self,
        handle: h5py.File,
        *,
        index: int,
        sample_id: str,
        split: str,
        family: str,
    ) -> np.ndarray:
        row = self.expected.get(int(index))
        if self.states.get(index) != "prediction_hashed":
            raise PermissionError("truth forbidden before prediction hash")
        if row is None or row["sample_id"] != sample_id or row["family"] != family:
            raise PermissionError("truth identity differs from frozen selection")
        if split != "train" or family not in FAMILIES or family == "anomaly":
            raise PermissionError("truth restricted to selected train target families")
        if self.truth_counts[index] != 0:
            raise PermissionError("truth may be read exactly once")
        self._transition(index, "truth_authorized")
        truth = np.asarray(handle["wavefield"][int(index), 0:401], dtype=np.float32)
        self.truth_counts[index] += 1
        self._transition(index, "truth_read")
        return truth

    def mark_scored(self, index: int) -> None:
        if self.states.get(index) != "truth_read":
            raise RuntimeError("score transition requires one truth read")
        self._transition(index, "scored")

    def _transition(self, index: int, state: str) -> None:
        previous = self.states[index]
        self.states[index] = state
        self.transitions.append(
            {"source_index": index, "from": previous, "to": state}
        )

    def summary(self) -> dict[str, Any]:
        completed = [index for index, state in self.states.items() if state == "scored"]
        return {
            "selected_count": len(self.expected),
            "prediction_hashed_count": len(self.prediction_hashes),
            "truth_read_count": sum(self.truth_counts.values()),
            "unique_truth_indices": len([count for count in self.truth_counts.values() if count == 1]),
            "all_truth_counts_at_most_one": all(count <= 1 for count in self.truth_counts.values()),
            "scored_count": len(completed),
            "prediction_hashed_before_truth": all(
                any(t["source_index"] == index and t["to"] == "prediction_hashed" for t in self.transitions)
                for index, count in self.truth_counts.items()
                if count
            ),
            "transitions": list(self.transitions),
            "validation_truth_reopened_this_run": False,
            "test_id_truth_reopened_this_run": False,
        }


def prediction_qc_and_hash(prediction: np.ndarray) -> tuple[np.ndarray, str]:
    value = np.ascontiguousarray(prediction, dtype=np.float32)
    if value.shape != (401, 201, 201):
        raise RuntimeError(f"prediction shape drift: {value.shape}")
    if not np.isfinite(value).all():
        raise FloatingPointError("prediction contains non-finite values")
    if not np.array_equal(value[:, 0, :], np.zeros((401, 201), dtype=np.float32)):
        raise RuntimeError("prediction top row is not exact zero")
    digest = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    if len(digest) != 64:
        raise RuntimeError("prediction SHA-256 length drift")
    return value, digest


def aggregate_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty measurement set")
    total = (0.0, 0.0)
    family_terms = {family: (0.0, 0.0) for family in FAMILIES}
    temporal_terms = {name: (0.0, 0.0) for name in TEMPORAL_RANGES}
    for row in rows:
        terms = tuple(row["error_terms"])
        total = _add_terms(total, terms)
        family = str(row["family"])
        family_terms[family] = _add_terms(family_terms[family], terms)
        for name in TEMPORAL_RANGES:
            temporal_terms[name] = _add_terms(
                temporal_terms[name], tuple(row["temporal_error_terms"][name])
            )
    runtimes = [float(row["outer_runtime_s"]) for row in rows]
    return {
        "aggregate_relative_l2": _relative_l2(total),
        "family_relative_l2": {
            family: _relative_l2(family_terms[family]) for family in FAMILIES
        },
        "temporal_band_relative_l2": {
            name: _relative_l2(temporal_terms[name]) for name in TEMPORAL_RANGES
        },
        "maximum_instance_relative_l2": max(float(row["relative_l2"]) for row in rows),
        "outer_runtime_s": {
            "mean": float(np.mean(runtimes)),
            "p95_nearest_rank": _nearest_rank(runtimes, 0.95),
            "p95_rank_for_60": 57 if len(rows) == 60 else None,
            "minimum": min(runtimes),
            "maximum": max(runtimes),
        },
    }


def classify_complete(metrics: Mapping[str, Any], *, contract_valid: bool = True) -> str:
    if not contract_valid:
        return "invalid"
    accuracy = (
        float(metrics["aggregate_relative_l2"]) <= 0.05
        and max(float(value) for value in metrics["family_relative_l2"].values()) <= 0.05
        and float(metrics["maximum_instance_relative_l2"]) <= 0.05
    )
    runtime = (
        float(metrics["outer_runtime_s"]["mean"]) <= RUNTIME_LIMIT_S
        and float(metrics["outer_runtime_s"]["p95_nearest_rank"]) <= RUNTIME_LIMIT_S
    )
    return "current_environment_reattest_passed" if accuracy and runtime else "rejected"


def environment_whitelist() -> dict[str, Any]:
    result = {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "scipy": scipy.__version__,
    }
    try:
        query = __import__("subprocess").run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        driver, model = [value.strip() for value in query.split(",", 1)]
        result.update({"nvidia_driver": driver, "gpu_model": model})
    except Exception as error:
        result["gpu_query_error"] = repr(error)
    return result


def _resolved(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_invocation(args: argparse.Namespace, prereg: Mapping[str, Any]) -> tuple[str, Path]:
    if prereg.get("schema") != "frozen_fine_grid_r6_current_env_reattest_preregistration_v1":
        raise RuntimeError("preregistration schema drift")
    if prereg.get("candidate") != CANDIDATE:
        raise RuntimeError("candidate drift")
    mode = "smoke" if args.smoke else "run"
    required_status = "draft_pending_audit" if mode == "smoke" else "smoke_passed_run_pending_independent_audit"
    if prereg.get("status") != required_status:
        raise RuntimeError(f"{mode} preregistration status rejected")
    paths = prereg["paths"]
    supplied = {
        "preregistration": args.preregistration.resolve(),
        "selection": args.selection.resolve(),
        "source_h5": args.source_h5.resolve(),
        "source_manifest": args.manifest.resolve(),
        "marmousi_npy": args.marmousi.resolve(),
        "reference_report": args.reference.resolve(),
        f"{mode}_output": args.output.resolve(),
    }
    for key, path in supplied.items():
        if path != _resolved(paths[key]):
            raise RuntimeError(f"fixed path override rejected: {key}")
    if mode == "run":
        prerequisite = prereg["prerequisites"]["smoke"]
        path = _resolved(prerequisite["path"])
        if prerequisite.get("status") != "passed" or sha256(path) != prerequisite.get("sha256"):
            raise RuntimeError("run smoke prerequisite is not frozen/passed")
    output = _resolved(paths[f"{mode}_output"])
    if output.exists():
        raise FileExistsError("refusing to overwrite reattest output")
    return mode, output


def validate_bindings(args: argparse.Namespace, prereg: Mapping[str, Any]) -> dict[str, str]:
    paths = prereg["paths"]
    binding_paths = {
        "script_sha256": Path(__file__),
        "test_sha256": _resolved(paths["test"]),
        "selection_sha256": args.selection,
        "dependency_manifest_sha256": _resolved(paths["dependency_manifest"]),
        "target_terminal_sha256": _resolved(paths["target_terminal"]),
        "frozen_candidate_sha256": _resolved(paths["frozen_candidate"]),
        "validation_report_sha256": _resolved(paths["validation_report"]),
        "validation_terminal_sha256": _resolved(paths["validation_terminal"]),
        "test_id_report_sha256": _resolved(paths["test_id_report"]),
        "test_id_terminal_sha256": _resolved(paths["test_id_terminal"]),
        "reference_report_sha256": args.reference,
        "source_h5_sha256": args.source_h5,
        "source_manifest_sha256": args.manifest,
        "marmousi_npy_sha256": args.marmousi,
        "audit_target5_sha256": ROOT / "scripts/audit_target5.py",
    }
    observed = {key: sha256(path) for key, path in binding_paths.items()}
    drift = [key for key, value in observed.items() if prereg["bindings"].get(key) != value]
    if drift:
        raise RuntimeError(f"explicit binding drift: {drift}")
    dependency = json.loads(_resolved(paths["dependency_manifest"]).read_text(encoding="utf-8"))
    closure = verify_dependency_manifest(dependency)
    observed.update({f"dependency:{key}": value for key, value in closure.items()})
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--marmousi", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    mode_name, output_path = validate_invocation(args, prereg)
    started = time.time()
    guard = None
    hashes_before: dict[str, str] = {}
    prereg_hash_before = sha256(args.preregistration)
    environment = environment_whitelist()
    report: dict[str, Any] = {
        "schema": "frozen_fine_grid_r6_current_environment_reattest_v1",
        "candidate": CANDIDATE,
        "mode": mode_name,
        "status": "running",
        "argv": list(sys.argv),
        "observed_environment": environment,
        "preregistration_sha256_observed_before": prereg_hash_before,
        "fresh": False,
        "independent_confirmation": False,
        "exposure_assumption": "all_selected_records_treated_as_exposed",
        "promotion_authorized": False,
        "historical_validation_previously_consumed": True,
        "historical_test_id_previously_consumed": True,
        "validation_truth_reopened_this_run": False,
        "test_id_truth_reopened_this_run": False,
    }
    try:
        expected_environment = prereg["environment"]
        for key, expected in expected_environment.items():
            if environment.get(key) != expected:
                raise RuntimeError(f"environment drift: {key}")
        if torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.benchmark or torch.backends.cudnn.deterministic:
            raise RuntimeError("algorithm flags differ from frozen default")
        hashes_before = validate_bindings(args, prereg)
        report["input_hashes_before"] = hashes_before
        target5 = audit_target5(ROOT, args.reference.resolve())
        report["historical_target5_audit"] = target5
        if not target5.get("passed"):
            raise RuntimeError("historical Target-5 binding audit failed")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        records = validate_selection_manifest(args.source_h5, selection)
        active = [records[0], records[20], records[40]] if mode_name == "smoke" else records
        guard = TruthStateGuard(active)
        with h5py.File(args.source_h5, "r", swmr=True) as source:
            output_times = np.asarray(source["time_s"][:], dtype=np.float64)
            if output_times.shape != (401,):
                raise RuntimeError("stored-time shape drift")
            metadata = []
            for expected in active:
                index = int(expected["source_index"])
                row = {
                    **expected,
                    "split": decode(source["split"][index]),
                    "source_x_m": float(source["source_x_m"][index]),
                    "source_z_m": float(source["source_z_m"][index]),
                    "source_f0_hz": float(source["source_f0_hz"][index]),
                    "source_t0_s": float(source["source_t0_s"][index]),
                    "source_amplitude": float(source["source_amplitude"][index]),
                }
                if row["split"] != "train" or decode(source["medium_type"][index]) != row["family"]:
                    raise RuntimeError("selected metadata split/family drift")
                metadata.append(row)
            manifest_rows = _read_manifest_rows(args.manifest, {row["sample_id"] for row in metadata})
            grid = AcousticGrid(nx=401, nz=401, dx_m=5.0, dz_m=5.0, lx_m=2000.0, lz_m=2000.0, centering="node")
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
            solver = FusedLWC84CPMLSolver(
                grid=grid,
                boundaries=BoundaryConfig(npml=40),
                dt_s=0.000625,
                output_times_s=output_times,
                c_ref_mps=6750.0,
                device=device,
                dtype=torch.float32,
                kappa_max=3.0,
                minimum_frequency_hz=8.0,
                output_restriction_factor=2,
                cuda_graphs=True,
            )
            warmup_seconds = float(solver.warmup(batch=1))
            measurements = []
            for row in metadata:
                index = int(row["source_index"])
                stored_velocity = np.asarray(source["velocity_mps"][index], dtype=np.float32)
                manifest_row = manifest_rows[row["sample_id"]]
                if manifest_row["split"] != "train" or manifest_row["medium_type"] != row["family"]:
                    raise RuntimeError("source manifest identity drift")
                fine_velocity, velocity_metadata = _fine_velocity(
                    family=row["family"], manifest_row=manifest_row, grid=grid, marmousi_npy=args.marmousi
                )
                restriction_difference = float(np.max(np.abs(restrict_nodal_2x(fine_velocity) - stored_velocity)))
                if restriction_difference != 0.0:
                    raise RuntimeError("fine velocity restriction QC failed")
                torch.cuda.synchronize(device)
                outer_started = time.perf_counter()
                result = solver.simulate(
                    fine_velocity,
                    source_x_m=row["source_x_m"],
                    source_z_m=row["source_z_m"],
                    source_f0_hz=row["source_f0_hz"],
                    source_t0_s=row["source_t0_s"],
                    source_amplitude=row["source_amplitude"],
                )
                torch.cuda.synchronize(device)
                outer_runtime = float(time.perf_counter() - outer_started)
                prediction, prediction_sha256 = prediction_qc_and_hash(result.wavefield[0])
                guard.register_prediction(
                    index=index,
                    sample_id=row["sample_id"],
                    family=row["family"],
                    digest=prediction_sha256,
                )
                truth = guard.read_truth(
                    source,
                    index=index,
                    sample_id=row["sample_id"],
                    split=row["split"],
                    family=row["family"],
                )
                terms = _error_terms(prediction, truth)
                temporal = {}
                for name, (start, stop) in TEMPORAL_RANGES.items():
                    temporal[name] = _error_terms(prediction[start:stop], truth[start:stop])
                guard.mark_scored(index)
                measurements.append(
                    {
                        "source_index": index,
                        "sample_id": row["sample_id"],
                        "group_id": row["group_id"],
                        "family": row["family"],
                        "sample_sha256": row["sample_sha256"],
                        "prediction_sha256": prediction_sha256,
                        "prediction_hashed_before_truth": True,
                        "relative_l2": _relative_l2(terms),
                        "error_terms": list(terms),
                        "temporal_error_terms": {key: list(value) for key, value in temporal.items()},
                        "temporal_band_relative_l2": {key: _relative_l2(value) for key, value in temporal.items()},
                        "outer_runtime_s": outer_runtime,
                        "solver_internal_compute_elapsed_s": float(result.metrics[0]["compute_elapsed_s"]),
                        "restriction_max_absolute_difference": restriction_difference,
                        "fine_velocity_metadata": velocity_metadata,
                        "prediction_shape": list(prediction.shape),
                        "prediction_dtype": str(prediction.dtype),
                        "prediction_contiguous": bool(prediction.flags.c_contiguous),
                        "prediction_finite": True,
                        "prediction_top_row_exact_zero": True,
                    }
                )
        metrics = aggregate_measurements(measurements)
        ledger = guard.summary()
        if ledger["truth_read_count"] != len(active) or ledger["scored_count"] != len(active):
            raise RuntimeError("truth ledger completeness failure")
        status = "smoke_complete" if mode_name == "smoke" else classify_complete(metrics)
        hashes_after = validate_bindings(args, prereg)
        prereg_hash_after = sha256(args.preregistration)
        if hashes_after != hashes_before or prereg_hash_after != prereg_hash_before:
            raise RuntimeError("input/preregistration changed during reattest")
        resources = {
            "elapsed_seconds": time.time() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "output_bytes": 0,
        }
        wall_limit = 120.0 if mode_name == "smoke" else 300.0
        if resources["elapsed_seconds"] > wall_limit or resources["peak_allocated_bytes"] >= 8 * 2**30:
            raise RuntimeError("reattest resource budget exceeded")
        claim = (
            "smoke implementation/path gate only; no reattest claim"
            if mode_name == "smoke"
            else "current environment exposed train regression panel passed and frozen historical Target-5 artifact bindings remain passed; not fresh/new validation"
        )
        report.update(
            {
                "status": status,
                "claim": claim,
                "record_count": len(measurements),
                "selection_digests": selection_digests(active),
                "solver_contract": {
                    "grid_shape": [401, 401],
                    "spacing_m": 5.0,
                    "npml": 40,
                    "dt_s": 0.000625,
                    "c_ref_mps": 6750.0,
                    "kappa_max": 3.0,
                    "minimum_frequency_hz": 8.0,
                    "dtype": "float32",
                    "output_restriction_factor": 2,
                    "cuda_graphs": True,
                    "stored_time_count": 401,
                },
                "warmup_seconds_excluded": warmup_seconds,
                "deployment_runtime_definition": "outer synchronized timer around complete solver.simulate; simulate includes CPU output materialization",
                "runtime_gate_uses": "outer_runtime_s only",
                "metrics": metrics,
                "measurements": measurements,
                "truth_ledger": ledger,
                "input_hashes_after": hashes_after,
                "preregistration_sha256_observed_after": prereg_hash_after,
                "preregistration_hash_unchanged": True,
                "resources": resources,
                "promotion_authorized": False,
                "historical_validation_previously_consumed": True,
                "historical_test_id_previously_consumed": True,
                "validation_truth_reopened_this_run": False,
                "test_id_truth_reopened_this_run": False,
            }
        )
        encoded = report_bytes_fixed_point(report)
        if len(encoded) >= 32 * 2**20:
            raise RuntimeError("report disk budget exceeded")
        atomic_write_bytes(encoded, output_path)
        return 0 if status in {"smoke_complete", "current_environment_reattest_passed"} else 2
    except Exception as error:
        report.update(
            {
                "status": "invalid",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "partial_truth_ledger": guard.summary() if guard is not None else None,
                "resources": {
                    "elapsed_seconds": time.time() - started,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0,
                    "output_bytes": 0,
                },
                "promotion_authorized": False,
                "historical_validation_previously_consumed": True,
                "historical_test_id_previously_consumed": True,
                "validation_truth_reopened_this_run": False,
                "test_id_truth_reopened_this_run": False,
            }
        )
        try:
            report["input_hashes_after"] = validate_bindings(args, prereg)
            report["preregistration_sha256_observed_after"] = sha256(args.preregistration)
            report["preregistration_hash_unchanged"] = (
                report["preregistration_sha256_observed_after"] == prereg_hash_before
            )
        except Exception as after_error:
            report["after_audit_error"] = repr(after_error)
        failure_bytes = report_bytes_fixed_point(report)
        if len(failure_bytes) >= 32 * 2**20:
            raise RuntimeError("failure report disk budget exceeded") from error
        atomic_write_bytes(failure_bytes, output_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
