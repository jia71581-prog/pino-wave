#!/usr/bin/env python3
"""Run the registered AIS normalization-v2 screen one review boundary at a time."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import platform
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fno_acoustic.ais_screen import (  # noqa: E402
    CANDIDATE_IDS,
    CATEGORIES,
    FINAL_CANDIDATE_METRICS,
    GATE_O_METRICS,
    HIGHER_GUARDS,
    HALVING_METRICS,
    LOWER_GUARDS,
    CandidateGateRecord,
    gate_h1,
    gate_h2,
    gate_h3,
    gate_o,
)
from fno_acoustic.ais_dataset_binding import (  # noqa: E402
    load_dataset_content_binding,
    validate_process_verification,
)
from fno_acoustic.query_census import (  # noqa: E402
    REQUIRED_NUMERIC_COLUMNS,
    SAMPLE_COLUMNS,
    SCREEN_NUMERIC_COLUMNS,
    SCREEN_SAMPLE_COLUMNS,
    aggregate_category_metrics,
    aggregate_screen_category_metrics,
)


CONFIG_FILENAMES = (
    "n0_norm.yaml",
    "n1_wide.yaml",
    "n2_spatial.yaml",
    "n3_temporal.yaml",
    "n4_large_local.yaml",
    "n5_multi_local.yaml",
    "n6_dispersion.yaml",
    "n7_multi_dispersion.yaml",
)
GATE_UPDATES = {"O": 400, "H1": 600, "H2": 1500, "H3": 3000}
ADDITIONAL_UPDATES = {"O": 400, "H1": 600, "H2": 900, "H3": 1500}
MANIFEST_SCHEMA_VERSION = 1
B1_CHECKPOINT_SHA256 = "91546ba21c31e0e875c0ed7051068c279b455a94f38206274094e1a9a9233653"
B1_CONFIG_SHA256 = "f9a8b81e116f3ba4c463ee0875b67f90ab46eba413a8e78f10d834ebc8e8dd81"
B1_CONFIG_PATH = REPO_ROOT / "configs/ais_mqfno_64x160_b1_uniform.yaml"
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "source_commit",
        "source_tree_sha256",
        "dirty_entries_sha256",
        "dataset_binding_sha256",
        "execution_binding_sha256",
        "command",
        "candidate_id",
        "config_path",
        "config_sha256",
        "split_path",
        "split_sha256",
        "normalization_stats_path",
        "normalization_stats_sha256",
        "last_checkpoint",
        "last_checkpoint_sha256",
        "best_checkpoint",
        "best_checkpoint_sha256",
        "metrics_path",
        "metrics_sha256",
        "lineage_path",
        "lineage_sha256",
        "runtime_seed",
        "parent_checkpoint_sha256",
        "optimizer_updates",
        "parameter_count",
        "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes",
        "wall_seconds",
        "gpu_name",
        "category_metrics",
        "finite",
    }
)
SCREEN_EVIDENCE_FIELDS = frozenset(
    {
        "evaluation_purpose",
        "validation_site_manifest_path",
        "validation_site_manifest_sha256",
        "samples_path",
        "samples_sha256",
    }
)
NATIVE_EVIDENCE_FIELDS = frozenset(
    {
        "native_metrics_path",
        "native_metrics_sha256",
        "native_samples_path",
        "native_samples_sha256",
        "native_category_metrics",
    }
)
REQUIRED_SMOKE_FIELDS = frozenset(
    {
        "source_commit", "source_tree_sha256", "dirty_entries_sha256", "command",
        "candidate_id", "config_sha256", "split_sha256",
        "normalization_stats_sha256", "last_checkpoint", "last_checkpoint_sha256",
        "best_checkpoint", "best_checkpoint_sha256", "metrics_path", "metrics_sha256",
        "runtime_seed", "optimizer_updates", "parameter_count",
        "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "wall_seconds", "gpu_name",
    }
)


@dataclass(frozen=True)
class ScreenCommand:
    candidate_id: str
    gate: str
    kind: str
    stop_update: int
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "gate": self.gate,
            "kind": self.kind,
            "stop_update": self.stop_update,
            "command": list(self.command),
        }


@dataclass(frozen=True)
class ScreenPlan:
    review_boundary: str
    smoke: tuple[ScreenCommand, ...]
    stage_commands: tuple[ScreenCommand, ...]
    action: str = "run"

    @property
    def gate_o(self) -> tuple[ScreenCommand, ...]:
        if self.review_boundary != "O":
            return ()
        return tuple(item for item in self.stage_commands if item.kind == "train")

    @property
    def all_commands(self) -> tuple[ScreenCommand, ...]:
        return self.smoke + self.stage_commands

    def to_dict(self) -> dict[str, object]:
        return {
            "review_boundary": self.review_boundary,
            "action": self.action,
            "smoke": [item.to_dict() for item in self.smoke],
            "commands": [item.to_dict() for item in self.stage_commands],
        }


@dataclass(frozen=True)
class SourceBinding:
    source_commit: str
    source_tree_sha256: str
    dirty_entries_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_commit": self.source_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "dirty_entries_sha256": self.dirty_entries_sha256,
        }


def compute_execution_binding(device: str) -> dict[str, object]:
    binding: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA execution binding requested without CUDA")
        properties = torch.cuda.get_device_properties(0)
        binding["gpu"] = {
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
            "total_memory": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    else:
        binding["gpu"] = None
    return {"values": binding, "sha256": canonical_sha256(binding)}


def load_dataset_binding(config_dir: Path) -> dict[str, object]:
    path = config_dir / "registration/dataset_content_manifest.json"
    if not path.is_file():
        raise ValueError("formal screen requires dataset_content_manifest.json")
    config = yaml.safe_load((config_dir / CONFIG_FILENAMES[0]).read_bytes())
    binding = load_dataset_content_binding(path, Path(config["data"]["path"]))
    payload = json.loads(path.read_bytes())
    return {
        "path": str(binding.manifest_path),
        "sha256": binding.manifest_sha256,
        **payload,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _immutable_bytes(path: Path) -> tuple[bytes, str]:
    """Read one regular-file snapshot used for both parsing and hashing."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"artifact is unavailable or symlinked: {path}") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"artifact changed during immutable read: {path}")
        raw = b"".join(chunks)
        return raw, hashlib.sha256(raw).hexdigest()
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Publish canonical JSON only after the file and containing directory are durable."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


class ScreenLock:
    """A persistent advisory lock inherited atomically by the active child."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def fd(self) -> int:
        if self._fd is None:
            raise RuntimeError("screen lock is not owned")
        return self._fd

    @staticmethod
    def _process_identity(pid: int) -> dict[str, object]:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="utf-8"
            ).strip()
            start_ticks: int | None = int(stat[21])
        except (OSError, IndexError, ValueError):
            boot_id, start_ticks = "unavailable", None
        return {"pid": pid, "start_ticks": start_ticks, "boot_id": boot_id}

    def _write_metadata(
        self, child: Mapping[str, object] | None, command: Sequence[str] | None
    ) -> None:
        payload = {
            "runner": self._process_identity(os.getpid()),
            "active_child": child,
            "active_child_command": list(command) if command is not None else None,
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.ftruncate(self.fd, 0)
        os.write(self.fd, encoded)
        os.fsync(self.fd)

    def __enter__(self) -> ScreenLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RuntimeError("screen lock is held by a live runner or child") from error
        self._fd = descriptor
        self._write_metadata(None, None)
        return self

    def set_active_child(self, pid: int, command: Sequence[str]) -> None:
        if self._fd is None or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise RuntimeError("active child requires an owned lock and positive PID")
        self._write_metadata(self._process_identity(pid), command)

    def clear_active_child(self) -> None:
        if self._fd is None:
            raise RuntimeError("cannot clear child without owning the screen lock")
        self._write_metadata(None, None)

    def __exit__(self, *_: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _resolve_registered_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a registered path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_configs(config_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    expected_paths = [config_dir / name for name in CONFIG_FILENAMES]
    missing = [path.name for path in expected_paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing registered AIS v2 configs: {missing}")
    unexpected = sorted(
        path.name for path in config_dir.glob("*.yaml") if path.name not in CONFIG_FILENAMES
    )
    if unexpected:
        raise ValueError(f"unexpected AIS v2 configs: {unexpected}")
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for candidate_id, path in zip(CANDIDATE_IDS, expected_paths, strict=True):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("screen", {}).get("candidate_id") != candidate_id:
            raise ValueError(f"{path.name} candidate registration mismatch")
        result[candidate_id] = (path, config)
    return result


def _load_manifest(value: Mapping[str, object] | str | Path) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    loaded = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("screen manifest must contain a mapping")
    if loaded.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported screen manifest schema version")
    return loaded


def _validate_current_registrations(
    manifest: Mapping[str, object],
    configs: Mapping[str, tuple[Path, dict[str, Any]]],
) -> None:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest artifacts must be a mapping")
    smoke_rows = manifest.get("smokes", {})
    if not isinstance(smoke_rows, Mapping):
        raise ValueError("manifest smokes must be a mapping")
    for rows in (*artifacts.values(), smoke_rows):
        if not isinstance(rows, Mapping):
            raise ValueError("manifest gate artifacts must be a mapping")
        for candidate_id, row in rows.items():
            if candidate_id not in configs or not isinstance(row, Mapping):
                raise ValueError("manifest contains an unregistered candidate")
            _, config = configs[candidate_id]
            split_path = _resolve_registered_path(
                config["data"]["split_manifest"], "split manifest"
            )
            stats_path = _resolve_registered_path(
                config["normalization"]["stats_path"], "normalization stats"
            )
            expected = {
                "config_sha256": canonical_sha256(config),
                "split_sha256": sha256_file(split_path),
                "normalization_stats_sha256": sha256_file(stats_path),
                "runtime_seed": int(config.get("seed", 2026)),
            }
            expected.update(
                {
                    name: manifest[name]
                    for name in (
                        "source_commit",
                        "source_tree_sha256",
                        "dirty_entries_sha256",
                    )
                    if name in manifest
                }
            )
            for name, value in expected.items():
                if name in row and row[name] != value:
                    raise ValueError(f"resume {name} differs from current registration")


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"canonical artifact path contains symlink: {path}")


def _validate_canonical_artifact_paths(
    manifest: Mapping[str, object], config_dir: Path, output_dir: Path
) -> None:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest artifacts must be a mapping")
    for gate, rows in artifacts.items():
        if gate not in GATE_UPDATES or not isinstance(rows, Mapping):
            raise ValueError("manifest gate artifacts are invalid")
        for candidate_id, row in rows.items():
            if candidate_id not in CANDIDATE_IDS or not isinstance(row, Mapping):
                raise ValueError("manifest contains an unregistered candidate")
            config_path = config_dir / CONFIG_FILENAMES[int(str(candidate_id)[1:])]
            config = yaml.safe_load(config_path.read_bytes())
            run_dir = output_dir / str(gate).lower() / str(candidate_id)
            expected = {
                "config_path": config_path,
                "split_path": _resolve_registered_path(
                    config["data"]["split_manifest"], "split manifest"
                ),
                "normalization_stats_path": _resolve_registered_path(
                    config["normalization"]["stats_path"], "normalization stats"
                ),
                "last_checkpoint": run_dir / "checkpoints/last.pt",
                "best_checkpoint": run_dir / "checkpoints/best.pt",
                "metrics_path": (
                    run_dir / "screen_metrics.json"
                    if gate == "O"
                    else run_dir / "evaluation/screen64/summary.json"
                ),
                "lineage_path": run_dir / "checkpoints/screen_lineage.json",
            }
            for key, canonical in expected.items():
                registered = row.get(key)
                if not isinstance(registered, str) or Path(registered) != canonical:
                    raise ValueError(f"{key} does not equal canonical artifact path")
                _reject_symlink_components(canonical)
            if gate != "O":
                evidence = {
                    "validation_site_manifest_path": _resolve_registered_path(
                        config["screen"]["validation_site_manifest"],
                        "validation site manifest",
                    ),
                    "samples_path": run_dir / "evaluation/screen64/samples.csv",
                }
                if gate == "H3":
                    evidence.update(
                        {
                            "native_metrics_path": run_dir
                            / "evaluation/native400/summary.json",
                            "native_samples_path": run_dir
                            / "evaluation/native400/samples.csv",
                        }
                    )
                for key, canonical in evidence.items():
                    if row.get(key) != str(canonical):
                        raise ValueError(f"{key} does not equal canonical artifact path")
                    _reject_symlink_components(canonical)


def validate_completed_artifacts(row: Mapping[str, object]) -> Mapping[str, object]:
    """Reject any completed row whose exact immutable artifact changed."""

    update = row.get("optimizer_updates")
    expected_fields = set(REQUIRED_ARTIFACT_FIELDS)
    if update in {600, 1500, 3000}:
        expected_fields.update(SCREEN_EVIDENCE_FIELDS)
    if update == 3000:
        expected_fields.update(NATIVE_EVIDENCE_FIELDS)
    if set(row) != expected_fields:
        raise ValueError("completed artifact row schema must be exact")
    for name in (
        "source_tree_sha256",
        "dirty_entries_sha256",
        "dataset_binding_sha256",
        "execution_binding_sha256",
    ):
        if not isinstance(row[name], str) or len(row[name]) != 64:
            raise ValueError(f"completed artifact row has invalid {name}")

    snapshots: dict[str, bytes] = {}
    bindings = [
        ("config_path", "config_sha256"),
        ("split_path", "split_sha256"),
        ("normalization_stats_path", "normalization_stats_sha256"),
        ("last_checkpoint", "last_checkpoint_sha256"),
        ("best_checkpoint", "best_checkpoint_sha256"),
        ("metrics_path", "metrics_sha256"),
        ("lineage_path", "lineage_sha256"),
    ]
    if update in {600, 1500, 3000}:
        bindings.extend(
            [
                ("validation_site_manifest_path", "validation_site_manifest_sha256"),
                ("samples_path", "samples_sha256"),
            ]
        )
    if update == 3000:
        bindings.extend(
            [
                ("native_metrics_path", "native_metrics_sha256"),
                ("native_samples_path", "native_samples_sha256"),
            ]
        )
    for path_key, hash_key in bindings:
        path_value, expected = row[path_key], row[hash_key]
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise ValueError(f"completed artifact row lacks {path_key} hash binding")
        path = Path(path_value)
        raw, raw_hash = _immutable_bytes(path)
        snapshots[path_key] = raw
        actual = canonical_sha256(yaml.safe_load(raw)) if path_key == "config_path" else raw_hash
        if actual != expected:
            raise ValueError(f"completed artifact hash mismatch: {path_key}")
    payload = torch.load(io.BytesIO(snapshots["last_checkpoint"]), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 5
        or payload.get("global_step") != row["optimizer_updates"]
        or payload.get("config_sha256") != row["config_sha256"]
        or payload.get("split_manifest_sha256") != row["split_sha256"]
        or payload.get("normalization_stats_sha256") != row["normalization_stats_sha256"]
        or payload.get("screen_candidate_id") != row["candidate_id"]
        or payload.get("parent_checkpoint_sha256") != row["parent_checkpoint_sha256"]
        or payload.get("dataset_binding_sha256") != row["dataset_binding_sha256"]
        or payload.get("execution_binding_sha256") != row["execution_binding_sha256"]
    ):
        raise ValueError("completed checkpoint schema, update, or binding hash mismatch")
    lineage = json.loads(snapshots["lineage_path"])
    if not isinstance(lineage, dict) or lineage.get("parent_checkpoint_sha256") != row[
        "parent_checkpoint_sha256"
    ]:
        raise ValueError("lineage sidecar differs from checkpoint authority")
    metrics = json.loads(snapshots["metrics_path"])
    raw_categories = metrics.get("category_metrics") if isinstance(metrics, dict) else None
    manifest_categories = row["category_metrics"]
    if not isinstance(raw_categories, Mapping) or not isinstance(manifest_categories, Mapping):
        raise ValueError("completed metrics artifact lacks category metrics")
    if not set(CATEGORIES) <= set(raw_categories) or set(manifest_categories) != set(CATEGORIES):
        raise ValueError("completed metrics category schema differs")
    for category, values in manifest_categories.items():
        raw = raw_categories[category]
        if (
            not isinstance(values, Mapping)
            or not isinstance(raw, Mapping)
            or any(raw.get(name) != value for name, value in values.items())
        ):
            raise ValueError("completed metrics hash-valid content differs from manifest row")
    if row["optimizer_updates"] == 400:
        if (
            metrics.get("global_step") != 400
            or metrics.get("candidate_id") != row["candidate_id"]
            or metrics.get("last_checkpoint_sha256") != row["last_checkpoint_sha256"]
        ):
            raise ValueError("completed Gate O metrics do not bind the exact last checkpoint")
    elif metrics.get("provenance", {}).get("checkpoint_sha256") != row["last_checkpoint_sha256"]:
        raise ValueError("completed metrics lineage does not bind exact last checkpoint")
    if update in {600, 1500, 3000}:
        split_payload = json.loads(snapshots["split_path"])
        expected_ids = set(split_payload["val"])
        screen_provenance = {
            "config_sha256": row["config_sha256"],
            "checkpoint_sha256": row["last_checkpoint_sha256"],
            "split_manifest_sha256": row["split_sha256"],
            "normalization_sha256": row["normalization_stats_sha256"],
            "validation_site_manifest_sha256": row[
                "validation_site_manifest_sha256"
            ],
        }
        aggregated = _validated_evaluation_evidence(
            snapshots["metrics_path"],
            snapshots["samples_path"],
            purpose="screen64_fixed2048",
            expected_ids=expected_ids,
            expected_provenance=screen_provenance,
        )
        selected = {
            category: {
                name: aggregated[category][name] for name in HALVING_METRICS
            }
            for category in CATEGORIES
        }
        if not _metric_tree_equal(selected, row["category_metrics"]):
            raise ValueError("completed screen CSV aggregation differs from manifest row")
    if update == 3000:
        native_provenance = {
            "config_sha256": row["config_sha256"],
            "checkpoint_sha256": row["last_checkpoint_sha256"],
            "split_manifest_sha256": row["split_sha256"],
            "normalization_sha256": row["normalization_stats_sha256"],
        }
        aggregated = _validated_evaluation_evidence(
            snapshots["native_metrics_path"],
            snapshots["native_samples_path"],
            purpose="native400_final_census",
            expected_ids=expected_ids,
            expected_provenance=native_provenance,
        )
        selected = {
            category: {
                name: aggregated[category][name]
                for name in FINAL_CANDIDATE_METRICS
            }
            for category in CATEGORIES
        }
        if not _metric_tree_equal(selected, row["native_category_metrics"]):
            raise ValueError("completed native CSV aggregation differs from manifest row")
    return row


def _artifact_rows(manifest: Mapping[str, object], gate: str) -> dict[str, Mapping[str, object]]:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest artifacts must be a mapping")
    rows = artifacts.get(gate, {})
    if not isinstance(rows, Mapping):
        raise ValueError(f"manifest artifacts.{gate} must be a mapping")
    result: dict[str, Mapping[str, object]] = {}
    for candidate_id, row in rows.items():
        if candidate_id not in CANDIDATE_IDS or not isinstance(row, Mapping):
            raise ValueError(f"invalid completed artifact row in {gate}")
        validate_completed_artifacts(row)
        result[str(candidate_id)] = row
    return result


def _smoke_rows(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = manifest.get("smokes", {})
    if not isinstance(rows, Mapping):
        raise ValueError("manifest smokes must be a mapping")
    result: dict[str, Mapping[str, object]] = {}
    for candidate_id, row in rows.items():
        if candidate_id not in CANDIDATE_IDS or not isinstance(row, Mapping):
            raise ValueError("invalid completed smoke row")
        validate_completed_smoke_artifacts(row)
        result[str(candidate_id)] = row
    return result


def validate_completed_smoke_artifacts(
    row: Mapping[str, object],
) -> Mapping[str, object]:
    if set(row) != set(REQUIRED_SMOKE_FIELDS):
        raise ValueError("completed smoke row schema must be exact")
    snapshots: dict[str, bytes] = {}
    for path_key, hash_key in (
        ("last_checkpoint", "last_checkpoint_sha256"),
        ("best_checkpoint", "best_checkpoint_sha256"),
        ("metrics_path", "metrics_sha256"),
    ):
        raw, digest = _immutable_bytes(Path(row[path_key]))
        if digest != row[hash_key]:
            raise ValueError(f"completed smoke hash mismatch: {path_key}")
        snapshots[path_key] = raw
    payload = torch.load(io.BytesIO(snapshots["last_checkpoint"]), weights_only=True)
    if not isinstance(payload, dict) or any(
        payload.get(name) != value
        for name, value in {
            "global_step": 1,
            "config_sha256": row["config_sha256"],
            "split_manifest_sha256": row["split_sha256"],
            "normalization_stats_sha256": row["normalization_stats_sha256"],
        }.items()
    ):
        raise ValueError("completed smoke checkpoint binding mismatch")
    rows = [json.loads(line) for line in snapshots["metrics_path"].splitlines() if line]
    if len(rows) != 1 or rows[0].get("global_step") != 1:
        raise ValueError("completed smoke metrics must bind update 1")
    return row


def _record(row: Mapping[str, object]) -> CandidateGateRecord:
    return CandidateGateRecord(
        candidate_id=row["candidate_id"],
        update=row["optimizer_updates"],
        checkpoint_sha256=row["last_checkpoint_sha256"],
        category_metrics=row["category_metrics"],
        finite=row["finite"],
        parent_checkpoint_sha256=row.get("parent_checkpoint_sha256"),
    )


def _native_record(row: Mapping[str, object]) -> CandidateGateRecord:
    return CandidateGateRecord(
        candidate_id=row["candidate_id"],
        update=row["optimizer_updates"],
        checkpoint_sha256=row["last_checkpoint_sha256"],
        category_metrics=row["native_category_metrics"],
        finite=row["finite"],
        parent_checkpoint_sha256=row.get("parent_checkpoint_sha256"),
    )


def _recomputed_decisions(manifest: Mapping[str, object]) -> dict[str, object]:
    stored = manifest.get("decisions", {})
    if not isinstance(stored, Mapping):
        raise ValueError("manifest decisions must be a mapping")
    rows_o = _artifact_rows(manifest, "O")
    decisions: dict[str, object] = {}
    if set(rows_o) == set(CANDIDATE_IDS):
        decisions["O"] = gate_o([_record(rows_o[item]) for item in CANDIDATE_IDS])
    rows_h1 = _artifact_rows(manifest, "H1")
    expected_h1 = set(decisions["O"].advanced) if "O" in decisions else set()
    if set(rows_h1) - expected_h1:
        raise ValueError("H1 artifacts include a candidate not advanced by Gate O")
    if rows_h1 and set(rows_h1) == expected_h1:
        decisions["H1"] = gate_h1(
            [_record(rows_h1[item]) for item in sorted(rows_h1)],
            [_record(rows_o[item]) for item in CANDIDATE_IDS],
        )
    rows_h2 = _artifact_rows(manifest, "H2")
    expected_h2 = set(decisions["H1"].advanced) if "H1" in decisions else set()
    if set(rows_h2) - expected_h2:
        raise ValueError("H2 artifacts include a candidate not advanced by H1")
    if rows_h2 and set(rows_h2) == expected_h2:
        decisions["H2"] = gate_h2(
            [_record(rows_h2[item]) for item in sorted(rows_h2)],
            [_record(rows_h1[item]) for item in sorted(rows_h1)],
            [_record(rows_o[item]) for item in CANDIDATE_IDS],
        )
    rows_h3 = _artifact_rows(manifest, "H3")
    expected_h3 = set(decisions["H2"].advanced) if "H2" in decisions else set()
    if set(rows_h3) - expected_h3:
        raise ValueError("H3 artifacts include a candidate not advanced by H2")
    if rows_h3 and set(rows_h3) == expected_h3:
        baseline = manifest.get("baseline_metrics")
        if isinstance(baseline, Mapping):
            baseline_path = manifest.get("baseline_metrics_path")
            baseline_sha = manifest.get("baseline_metrics_sha256")
            if not isinstance(baseline_path, str) or not isinstance(baseline_sha, str):
                raise ValueError("bound baseline requires path and hash")
            path = Path(baseline_path)
            if not path.is_file() or sha256_file(path) != baseline_sha:
                raise ValueError("baseline metrics hash mismatch")
            reread, reread_sha = _read_baseline_metrics(path, manifest)
            if reread_sha != baseline_sha or reread != baseline:
                raise ValueError("baseline metrics content differs from bound hash")
            decisions["H3"] = gate_h3(
                [_native_record(rows_h3[item]) for item in sorted(rows_h3)],
                [_record(rows_h2[item]) for item in sorted(rows_h2)],
                [_record(rows_h1[item]) for item in sorted(rows_h1)],
                [_record(rows_o[item]) for item in CANDIDATE_IDS],
                baseline,
                ranking_records=[
                    _record(rows_h3[item]) for item in sorted(rows_h3)
                ],
            )
    for gate, decision in decisions.items():
        stored_value = stored.get(gate)
        if stored_value is not None and stored_value != decision.to_dict():
            raise ValueError(f"stored {gate} decision differs from exact recomputation")
    return decisions


def _read_baseline_metrics(
    path: Path, manifest: Mapping[str, object]
) -> tuple[dict[str, dict[str, float]], str]:
    raw, digest = _immutable_bytes(path)
    payload = json.loads(raw)
    config = yaml.safe_load(B1_CONFIG_PATH.read_bytes())
    split_path = _resolve_registered_path(config["data"]["split_manifest"], "B1 split")
    stats_path = _resolve_registered_path(
        config["normalization"]["stats_path"], "B1 normalization stats"
    )
    split = json.loads(split_path.read_bytes())
    required_identity = {
        "schema_version": 1,
        "purpose": "ais_b1_validation_baseline",
        "baseline_id": "B1",
        "checkpoint_sha256": B1_CHECKPOINT_SHA256,
        "config_sha256": B1_CONFIG_SHA256,
        "split": "val",
        "sample_count": len(split["val"]),
        "representative_manifest": None,
        "split_sha256": sha256_file(split_path),
        "normalization_stats_sha256": sha256_file(stats_path),
        "source_commit": manifest.get("source_commit"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "dirty_entries_sha256": manifest.get("dirty_entries_sha256"),
        "samples_path": str(path.parent / "samples.csv"),
        "samples_sha256": sha256_file(path.parent / "samples.csv"),
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in required_identity.items()
    ):
        raise ValueError("baseline B1 schema or immutable provenance mismatch")
    if canonical_sha256(config) != B1_CONFIG_SHA256:
        raise ValueError("registered B1 config differs from pinned hash")
    categories = payload.get("category_metrics") if isinstance(payload, dict) else None
    if not isinstance(categories, Mapping) or not set(CATEGORIES) <= set(categories):
        raise ValueError(f"baseline must contain categories {CATEGORIES}")
    required = frozenset({"relative_l2", "relative_l2_q4", *LOWER_GUARDS, *HIGHER_GUARDS})
    selected: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        values = categories[category]
        if not isinstance(values, Mapping) or not required <= set(values):
            raise ValueError(f"baseline {category} lacks full final baseline metrics")
        selected[category] = {}
        for name in required:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"baseline {category}.{name} must be finite")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"baseline {category}.{name} must be finite")
            selected[category][name] = converted
    return selected, digest


def finalize_h3_with_baseline(
    manifest: Mapping[str, object], baseline_path: str | Path
) -> dict[str, object]:
    stage = manifest.get("stage")
    if not isinstance(stage, Mapping) or stage != {
        "gate": "H3",
        "status": "awaiting_baseline",
    }:
        raise ValueError("baseline metrics are legal only while H3 awaits baseline")
    result = dict(manifest)
    baseline = Path(baseline_path)
    h3_rows = _artifact_rows(manifest, "H3")
    if not h3_rows:
        raise ValueError("H3 candidate records are incomplete")
    output_dir = Path(next(iter(h3_rows.values()))["last_checkpoint"]).parents[3]
    if baseline != output_dir / "baseline_b1/summary.json":
        raise ValueError("baseline must use canonical baseline_b1/summary.json path")
    checkpoint = output_dir / "baseline_b1/checkpoints/best.pt"
    if not checkpoint.is_file() or sha256_file(checkpoint) != B1_CHECKPOINT_SHA256:
        raise ValueError("baseline B1 checkpoint differs from pinned hash")
    metrics, digest = _read_baseline_metrics(baseline, manifest)
    existing_digest = result.get("baseline_metrics_sha256")
    if existing_digest is not None and existing_digest != digest:
        raise ValueError("baseline metrics SHA-256 differs from immutable binding")
    result["baseline_metrics"] = metrics
    result["baseline_metrics_path"] = str(Path(baseline_path))
    result["baseline_metrics_sha256"] = digest
    decisions = _recomputed_decisions(result)
    if "H3" not in decisions:
        raise ValueError("H3 candidate records are incomplete")
    result["decisions"] = {
        gate: decision.to_dict() for gate, decision in decisions.items()
    }
    result["stage"] = {"gate": "H3", "status": "complete"}
    return result


def finalize_current_stage(
    manifest: Mapping[str, object], gate: str
) -> dict[str, object]:
    stage = manifest.get("stage")
    if not isinstance(stage, Mapping) or stage.get("gate") != gate or stage.get(
        "status"
    ) != "in_progress":
        raise ValueError("current stage is not eligible for finalization")
    decisions = _recomputed_decisions(manifest)
    if gate != "H3" and gate not in decisions:
        raise ValueError(f"{gate} records are incomplete")
    if gate == "H3":
        rows = _artifact_rows(manifest, "H3")
        expected = set(decisions["H2"].advanced) if "H2" in decisions else set()
        if set(rows) != expected:
            raise ValueError("H3 records are incomplete")
    result = dict(manifest)
    result["decisions"] = {
        name: decision.to_dict() for name, decision in decisions.items()
    }
    result["stage"] = {
        "gate": gate,
        "status": "awaiting_baseline" if gate == "H3" and "H3" not in decisions else "complete",
    }
    result["last_review_boundary"] = gate
    return result


def _command(
    candidate_id: str,
    gate: str,
    kind: str,
    update: int,
    device: str,
    *,
    resume_from: str | None = None,
    max_train_batches: int | None = None,
    finalize_overfit: bool = False,
    finalize_training: bool = False,
) -> ScreenCommand:
    filename = CONFIG_FILENAMES[int(candidate_id[1:])]
    run_dir = f"<OUTPUT_DIR>/{gate.lower()}/{candidate_id}"
    config = f"<CONFIG_DIR>/{filename}"
    if kind in {"evaluate", "evaluate_native"}:
        native = kind == "evaluate_native"
        output = (
            f"{run_dir}/evaluation/native400"
            if native
            else f"{run_dir}/evaluation/screen64"
        )
        command_list = [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluate_ais_mqfno.py"),
            "--config",
            config,
            "--checkpoint",
            f"{run_dir}/checkpoints/last.pt",
            "--split",
            "val",
            "--output-dir",
            output,
            "--device",
            device,
            "--dataset-binding-sha256",
            "<DATASET_BINDING_SHA256>",
            "--execution-binding-sha256",
            "<EXECUTION_BINDING_SHA256>",
            "--dataset-content-manifest",
            "<CONFIG_DIR>/registration/dataset_content_manifest.json",
            "--evaluation-purpose",
            "native400_final_census" if native else "screen64_fixed2048",
        ]
        if not native:
            command_list.extend(
                [
                    "--screen-validation-site-manifest",
                    "<CONFIG_DIR>/registration/fixed_validation_sites_2048.json",
                ]
            )
        command = tuple(command_list)
    else:
        command_list = [
            sys.executable,
            str(REPO_ROOT / "scripts/train_ais_mqfno.py"),
            "--config",
            config,
            "--output-dir",
            run_dir,
            "--device",
            device,
            "--screen-gate",
            gate,
            "--dataset-binding-sha256",
            "<DATASET_BINDING_SHA256>",
            "--execution-binding-sha256",
            "<EXECUTION_BINDING_SHA256>",
            "--dataset-content-manifest",
            "<CONFIG_DIR>/registration/dataset_content_manifest.json",
        ]
        if not (finalize_overfit or finalize_training):
            command_list.extend(
                [
                    "--max-train-batches",
                    str(
                        ADDITIONAL_UPDATES[gate]
                        if max_train_batches is None
                        else max_train_batches
                    ),
                ]
            )
        if gate == "O":
            command_list.extend(
                [
                    "--overfit-sample-id",
                    "2",
                    "--overfit-site-manifest",
                    "<CONFIG_DIR>/registration/gate_o_sample_0002_sites_2048.json",
                ]
            )
        if resume_from is not None:
            command_list.extend(["--resume", resume_from])
        if finalize_overfit:
            command_list.append("--finalize-overfit-only")
        if finalize_training:
            command_list.append("--finalize-training-only")
        command = tuple(command_list)
    return ScreenCommand(candidate_id, gate, kind, update, tuple(command))


def _inspect_unpublished_last(
    candidate_id: str,
    gate: str,
    config_dir: Path,
    output_dir: Path,
    expected_parent_sha256: str | None,
) -> tuple[str, int] | None:
    run_dir = output_dir / gate.lower() / candidate_id
    if not run_dir.exists():
        return None
    last = run_dir / "checkpoints/last.pt"
    runtime = run_dir / "checkpoints/screen_lineage.json"
    if not last.is_file() or not runtime.is_file():
        raise ValueError("unpublished candidate directory lacks canonical last checkpoint")
    config = yaml.safe_load(
        (config_dir / CONFIG_FILENAMES[int(candidate_id[1:])]).read_text(
            encoding="utf-8"
        )
    )
    split = _resolve_registered_path(config["data"]["split_manifest"], "split")
    stats = _resolve_registered_path(config["normalization"]["stats_path"], "stats")
    last_raw, _ = _immutable_bytes(last)
    payload = torch.load(io.BytesIO(last_raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 5:
        raise ValueError("unpublished last checkpoint must use schema 5")
    expected = {
        "config_sha256": canonical_sha256(config),
        "split_manifest_sha256": sha256_file(split),
        "normalization_stats_sha256": sha256_file(stats),
        "runtime_seed": int(config.get("seed", 2026)),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            label = "config hash" if name == "config_sha256" else name
            raise ValueError(f"unpublished checkpoint {label} mismatch")
    if payload.get("screen_candidate_id") != candidate_id or payload.get(
        "screen_gate"
    ) != gate:
        raise ValueError("unpublished checkpoint screen identity mismatch")
    runtime_raw, _ = _immutable_bytes(runtime)
    runtime_payload = json.loads(runtime_raw)
    if (
        not isinstance(runtime_payload, Mapping)
        or runtime_payload.get("runtime_seed") != expected["runtime_seed"]
        or runtime_payload.get("parent_checkpoint_sha256")
        != expected_parent_sha256
    ):
        raise ValueError("unpublished checkpoint seed or parent lineage mismatch")
    if payload.get("parent_checkpoint_sha256") != expected_parent_sha256:
        raise ValueError("unpublished checkpoint parent authority mismatch")
    global_step = payload.get("global_step")
    target = GATE_UPDATES[gate]
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step < 0
        or global_step > target
    ):
        raise ValueError("unpublished checkpoint global_step exceeds target or is invalid")
    return ("complete" if global_step == target else "partial", global_step)


def _candidate_stage_commands(
    candidate_id: str,
    gate: str,
    device: str,
    config_dir: Path,
    output_dir: Path,
    parent_checkpoint_sha256: str | None,
    normal_resume_from: str | None,
) -> list[ScreenCommand]:
    recovery = _inspect_unpublished_last(
        candidate_id,
        gate,
        config_dir,
        output_dir,
        parent_checkpoint_sha256,
    )
    target = GATE_UPDATES[gate]
    evidence_state = "absent"
    if recovery is not None and gate != "O":
        evidence_state = _existing_evaluation_state(
            candidate_id, gate, config_dir, output_dir
        )
        if recovery[0] != "complete" and evidence_state != "absent":
            raise ValueError("evaluation evidence exists before exact gate checkpoint")
    if recovery is None:
        commands = [
            _command(
                candidate_id,
                gate,
                "train",
                target,
                device,
                resume_from=normal_resume_from,
            )
        ]
    elif recovery[0] == "partial":
        current_last = f"<OUTPUT_DIR>/{gate.lower()}/{candidate_id}/checkpoints/last.pt"
        commands = [
            _command(
                candidate_id,
                gate,
                "train",
                target,
                device,
                resume_from=current_last,
                max_train_batches=target - recovery[1],
            )
        ]
    elif gate == "O":
        current_last = f"<OUTPUT_DIR>/o/{candidate_id}/checkpoints/last.pt"
        return [
            _command(
                candidate_id,
                gate,
                "finalize_overfit",
                target,
                device,
                resume_from=current_last,
                finalize_overfit=True,
            )
        ]
    else:
        commands = []
        if not (output_dir / gate.lower() / candidate_id / "summary.json").is_file():
            current_last = (
                f"<OUTPUT_DIR>/{gate.lower()}/{candidate_id}/checkpoints/last.pt"
            )
            commands.append(
                _command(
                    candidate_id,
                    gate,
                    "finalize_training",
                    target,
                    device,
                    resume_from=current_last,
                    finalize_training=True,
                )
            )
    if gate != "O":
        if evidence_state == "absent":
            commands.append(_command(candidate_id, gate, "evaluate", target, device))
        if gate == "H3" and evidence_state != "complete":
            commands.append(
                _command(candidate_id, gate, "evaluate_native", target, device)
            )
        elif evidence_state in {"screen", "complete"}:
            commands.append(
                ScreenCommand(
                    candidate_id,
                    gate,
                    "publish",
                    target,
                    ("<RECOVERED_CANONICAL_EVIDENCE>",),
                )
            )
    return commands


def _smoke_command(candidate_id: str, device: str) -> ScreenCommand:
    filename = CONFIG_FILENAMES[int(candidate_id[1:])]
    return ScreenCommand(
        candidate_id,
        "smoke",
        "train",
        1,
        (
            sys.executable,
            str(REPO_ROOT / "scripts/train_ais_mqfno.py"),
            "--config",
            f"<CONFIG_DIR>/{filename}",
            "--output-dir",
            f"<OUTPUT_DIR>/smoke/{candidate_id}",
            "--device",
            device,
            "--max-train-batches",
            "1",
            "--max-val-batches",
            "1",
        ),
    )


def _smoke_recovery_command(
    candidate_id: str, device: str, config_dir: Path, output_dir: Path
) -> ScreenCommand:
    run_dir = output_dir / "smoke" / candidate_id
    last = run_dir / "checkpoints/last.pt"
    if not run_dir.exists():
        return _smoke_command(candidate_id, device)
    if not last.is_file():
        raise ValueError("partial smoke directory lacks canonical last checkpoint")
    raw, _ = _immutable_bytes(last)
    payload = torch.load(io.BytesIO(raw), weights_only=True)
    config_path = config_dir / CONFIG_FILENAMES[int(candidate_id[1:])]
    config = yaml.safe_load(config_path.read_bytes())
    split = _resolve_registered_path(config["data"]["split_manifest"], "split")
    stats = _resolve_registered_path(config["normalization"]["stats_path"], "stats")
    expected = {
        "global_step": 1, "config_sha256": canonical_sha256(config),
        "split_manifest_sha256": sha256_file(split),
        "normalization_stats_sha256": sha256_file(stats),
    }
    if not isinstance(payload, dict) or any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("partial smoke last checkpoint binding mismatch")
    command = (
        sys.executable, str(REPO_ROOT / "scripts/train_ais_mqfno.py"),
        "--config", f"<CONFIG_DIR>/{CONFIG_FILENAMES[int(candidate_id[1:])]}",
        "--output-dir", f"<OUTPUT_DIR>/smoke/{candidate_id}",
        "--device", device, "--resume",
        f"<OUTPUT_DIR>/smoke/{candidate_id}/checkpoints/last.pt",
        "--finalize-training-only",
    )
    return ScreenCommand(candidate_id, "smoke", "finalize_training", 1, command)


def build_screen_plan(
    config_dir: str | Path,
    output_dir: str | Path,
    resume_manifest: Mapping[str, object] | str | Path | None = None,
    *,
    device: str = "cuda",
) -> ScreenPlan:
    """Construct a deterministic, side-effect-free plan for one review boundary."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    config_path = Path(config_dir)
    configs = _load_configs(config_path)
    if resume_manifest is None:
        smoke = tuple(_smoke_command(item, device) for item in CANDIDATE_IDS)
        stage = tuple(_command(item, "O", "train", 400, device) for item in CANDIDATE_IDS)
        return ScreenPlan("O", smoke, stage)

    manifest = _load_manifest(resume_manifest)
    _validate_canonical_artifact_paths(
        manifest, config_path.absolute(), Path(output_dir).absolute()
    )
    _validate_current_registrations(manifest, configs)
    decisions = _recomputed_decisions(manifest)
    current_stage = manifest.get("stage")
    if (
        isinstance(current_stage, Mapping)
        and current_stage.get("status") == "in_progress"
        and current_stage.get("gate") in {"O", "H1", "H2", "H3"}
    ):
        current_gate = str(current_stage["gate"])
        complete = current_gate in decisions
        if current_gate == "H3" and "H2" in decisions:
            complete = set(_artifact_rows(manifest, "H3")) == set(
                decisions["H2"].advanced
            )
        if complete:
            return ScreenPlan(
                current_gate, (), (), action="finalize_current_stage"
            )
    if "O" not in decisions:
        stage = manifest.get("stage")
        if (
            isinstance(stage, Mapping)
            and stage.get("gate") == "smoke"
            and stage.get("status") == "in_progress"
        ):
            completed_smokes = _smoke_rows(manifest)
            smoke = tuple(
                _smoke_recovery_command(
                    item, device, config_path, Path(output_dir)
                )
                for item in CANDIDATE_IDS
                if item not in completed_smokes
            )
            gate_o_commands = tuple(
                command
                for item in CANDIDATE_IDS
                for command in _candidate_stage_commands(
                    item, "O", device, config_path, Path(output_dir), None, None
                )
            )
            return ScreenPlan("O", smoke, gate_o_commands)
        rows_o = _artifact_rows(manifest, "O")
        if (
            not isinstance(stage, Mapping)
            or stage.get("gate") != "O"
            or stage.get("status") != "in_progress"
        ):
            raise ValueError("--resume requires a completed or in-progress Gate O")
        commands = tuple(
            command
            for item in CANDIDATE_IDS
            if item not in rows_o
            for command in _candidate_stage_commands(
                item, "O", device, config_path, Path(output_dir), None, None
            )
        )
        return ScreenPlan("O", (), commands)
    last_gate = next(gate for gate in ("H3", "H2", "H1", "O") if gate in decisions)
    prior = decisions[last_gate]
    if prior.status not in {"advance"}:
        return ScreenPlan(last_gate, (), ())
    next_gate = {"O": "H1", "H1": "H2", "H2": "H3"}.get(last_gate)
    if next_gate is None:
        return ScreenPlan("H3", (), ())
    completed = _artifact_rows(manifest, next_gate)
    commands: list[ScreenCommand] = []
    for candidate_id in prior.advanced:
        if candidate_id in completed:
            continue
        resume_from = None
        if next_gate == "H2":
            resume_from = f"<OUTPUT_DIR>/h1/{candidate_id}/checkpoints/last.pt"
        elif next_gate == "H3":
            resume_from = f"<OUTPUT_DIR>/h2/{candidate_id}/checkpoints/last.pt"
        parent_sha = (
            None
            if next_gate == "H1"
            else prior.checkpoint_sha256s[candidate_id]
        )
        commands.extend(
            _candidate_stage_commands(
                candidate_id,
                next_gate,
                device,
                config_path,
                Path(output_dir),
                parent_sha,
                resume_from,
            )
        )
    return ScreenPlan(next_gate, (), tuple(commands))


def _materialize_command(
    command: Sequence[str],
    config_dir: Path,
    output_dir: Path,
    dataset_binding_sha256: str = "<DATASET_BINDING_SHA256>",
    execution_binding_sha256: str = "<EXECUTION_BINDING_SHA256>",
) -> list[str]:
    return [
        token.replace("<CONFIG_DIR>", str(config_dir)).replace(
            "<OUTPUT_DIR>", str(output_dir)
        ).replace("<DATASET_BINDING_SHA256>", dataset_binding_sha256).replace(
            "<EXECUTION_BINDING_SHA256>", execution_binding_sha256
        )
        for token in command
    ]


def compute_source_binding(repo_root: str | Path = REPO_ROOT) -> SourceBinding:
    root = Path(repo_root).resolve()
    relevant = ("src/fno_acoustic", "scripts", "configs/ais_zero_collapse_v2")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "ls-files", "--", *relevant],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *relevant],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    tree_digest = hashlib.sha256()
    for relative in sorted(set(tracked + untracked)):
        path = root / relative
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        if path.is_file():
            tree_digest.update(path.read_bytes())
        else:
            tree_digest.update(b"<deleted>")
        tree_digest.update(b"\0")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *relevant],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.encode("utf-8")
    return SourceBinding(
        source_commit=commit,
        source_tree_sha256=tree_digest.hexdigest(),
        dirty_entries_sha256=hashlib.sha256(dirty).hexdigest(),
    )


def require_source_unchanged(before: SourceBinding, after: SourceBinding) -> None:
    if before != after:
        raise RuntimeError("relevant source tree changed during screen stage")


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _metric_tree_equal(actual: object, expected: object) -> bool:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _metric_tree_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def _validated_evaluation_evidence(
    summary_raw: bytes,
    samples_raw: bytes,
    *,
    purpose: str,
    expected_ids: set[int],
    expected_provenance: Mapping[str, str],
) -> dict[str, dict[str, float | int]]:
    """Strictly parse one immutable CSV snapshot and recompute its summary."""

    if purpose == "screen64_fixed2048":
        columns, numeric = SCREEN_SAMPLE_COLUMNS, SCREEN_NUMERIC_COLUMNS
        aggregate = aggregate_screen_category_metrics
    elif purpose == "native400_final_census":
        columns, numeric = SAMPLE_COLUMNS, REQUIRED_NUMERIC_COLUMNS
        aggregate = aggregate_category_metrics
    else:
        raise ValueError("unknown evaluation evidence purpose")
    summary = json.loads(summary_raw)
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != 1
        or summary.get("purpose") != purpose
        or summary.get("split") != "val"
        or summary.get("sample_count") != len(expected_ids)
    ):
        raise ValueError("evaluation summary purpose, split, or count mismatch")
    provenance = summary.get("provenance")
    if not isinstance(provenance, Mapping) or any(
        provenance.get(name) != value for name, value in expected_provenance.items()
    ):
        raise ValueError("evaluation summary provenance mismatch")
    reader = csv.DictReader(io.StringIO(samples_raw.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != columns:
        raise ValueError("evaluation samples CSV column contract mismatch")
    rows: list[dict[str, object]] = []
    try:
        for raw in reader:
            row: dict[str, object] = dict(raw)
            row["sample_id"] = int(raw["sample_id"])
            for name in numeric:
                row[name] = float(raw[name])
            rows.append(row)
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation samples CSV numeric value is invalid") from error
    if {int(row["sample_id"]) for row in rows} != expected_ids or len(rows) != len(
        expected_ids
    ):
        raise ValueError("evaluation samples CSV IDs differ from validation split")
    for row in rows:
        if any(row.get(name) != value for name, value in expected_provenance.items()):
            raise ValueError("evaluation samples CSV provenance mismatch")
    aggregated = aggregate(rows)
    if not set(CATEGORIES) <= set(aggregated):
        raise ValueError("evaluation CSV lacks one or more registered categories")
    if not _metric_tree_equal(summary.get("category_metrics"), aggregated):
        raise ValueError("evaluation summary metrics differ from strict CSV aggregation")
    return aggregated


def _existing_evaluation_state(
    candidate_id: str,
    gate: str,
    config_dir: Path,
    output_dir: Path,
) -> str:
    """Validate canonical evaluation snapshots before scheduling any child."""

    run_dir = output_dir / gate.lower() / candidate_id
    config_path = config_dir / CONFIG_FILENAMES[int(candidate_id[1:])]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_path = _resolve_registered_path(config["data"]["split_manifest"], "split")
    stats_path = _resolve_registered_path(
        config["normalization"]["stats_path"], "stats"
    )
    split_raw, split_hash = _immutable_bytes(split_path)
    split_payload = json.loads(split_raw)
    expected_ids = set(split_payload["val"])
    _, checkpoint_hash = _immutable_bytes(run_dir / "checkpoints/last.pt")
    common_provenance = {
        "config_sha256": canonical_sha256(config),
        "checkpoint_sha256": checkpoint_hash,
        "split_manifest_sha256": split_hash,
        "normalization_sha256": sha256_file(stats_path),
    }

    def validate_pair(
        name: str,
        purpose: str,
        provenance: Mapping[str, str],
    ) -> bool:
        directory = run_dir / "evaluation" / name
        if not directory.exists():
            return False
        summary_path = directory / "summary.json"
        samples_path = directory / "samples.csv"
        if not directory.is_dir() or not summary_path.is_file() or not samples_path.is_file():
            raise ValueError(f"incomplete canonical evaluation evidence: {name}")
        summary_raw, _ = _immutable_bytes(summary_path)
        samples_raw, _ = _immutable_bytes(samples_path)
        _validated_evaluation_evidence(
            summary_raw,
            samples_raw,
            purpose=purpose,
            expected_ids=expected_ids,
            expected_provenance=provenance,
        )
        return True

    site_path = _resolve_registered_path(
        config["screen"]["validation_site_manifest"], "validation site manifest"
    )
    _, site_hash = _immutable_bytes(site_path)
    if site_hash != config["screen"]["validation_site_manifest_sha256"]:
        raise ValueError("validation site manifest hash differs from config")
    has_screen = validate_pair(
        "screen64",
        "screen64_fixed2048",
        {**common_provenance, "validation_site_manifest_sha256": site_hash},
    )
    native_dir = run_dir / "evaluation/native400"
    if gate != "H3":
        if native_dir.exists():
            raise ValueError("unexpected canonical native400 evidence before H3")
        return "screen" if has_screen else "absent"
    has_native = validate_pair(
        "native400", "native400_final_census", common_provenance
    )
    if has_native and not has_screen:
        raise ValueError("native400 evidence exists without canonical screen64 evidence")
    if has_native:
        return "complete"
    return "screen" if has_screen else "absent"


def census_candidate_artifacts(
    item: ScreenCommand,
    command: Sequence[str],
    config_dir: Path,
    output_dir: Path,
    source_commit: str,
    source_tree_sha256: str,
    parent_checkpoint_sha256: str | None,
    dirty_entries_sha256: str | None = None,
    dataset_binding_sha256: str | None = None,
    execution_binding_sha256: str | None = None,
    dataset_content_root: str | None = None,
    dataset_sample_count: int | None = None,
) -> dict[str, object]:
    """Validate exact-last artifacts and return a complete immutable census row."""

    config_path = config_dir / CONFIG_FILENAMES[int(item.candidate_id[1:])]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = output_dir / item.gate.lower() / item.candidate_id
    last_path, best_path = run_dir / "checkpoints/last.pt", run_dir / "checkpoints/best.pt"
    metrics_path = (
        run_dir / "screen_metrics.json"
        if item.gate == "O"
        else run_dir / "evaluation/screen64/summary.json"
    )
    lineage_path = run_dir / "checkpoints/screen_lineage.json"
    samples_path = run_dir / "evaluation/screen64/samples.csv"
    native_metrics_path = run_dir / "evaluation/native400/summary.json"
    native_samples_path = run_dir / "evaluation/native400/samples.csv"
    required_paths = [last_path, best_path, metrics_path, lineage_path]
    if item.gate != "O":
        required_paths.append(samples_path)
    if item.gate == "H3":
        required_paths.extend((native_metrics_path, native_samples_path))
    for path in required_paths:
        if not path.is_file():
            raise ValueError(f"child completed without required artifact: {path}")
    last_raw, last_hash = _immutable_bytes(last_path)
    payload = torch.load(io.BytesIO(last_raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 5:
        raise ValueError("exact last checkpoint must use schema version 5")
    if payload.get("global_step") != item.stop_update:
        raise ValueError("exact last checkpoint optimizer update mismatch")
    if payload.get("screen_candidate_id") != item.candidate_id or payload.get(
        "screen_gate"
    ) != item.gate:
        raise ValueError("exact last checkpoint screen identity mismatch")
    if payload.get("parent_checkpoint_sha256") != parent_checkpoint_sha256:
        raise ValueError("exact last checkpoint parent authority mismatch")
    if (
        payload.get("dataset_binding_sha256") != dataset_binding_sha256
        or payload.get("execution_binding_sha256") != execution_binding_sha256
    ):
        raise ValueError("exact last checkpoint runtime binding mismatch")
    split_path = _resolve_registered_path(config["data"]["split_manifest"], "split manifest")
    stats_path = _resolve_registered_path(config["normalization"]["stats_path"], "stats")
    config_hash = canonical_sha256(config)
    split_hash, stats_hash = sha256_file(split_path), sha256_file(stats_path)
    bindings = {
        "config_sha256": config_hash,
        "split_manifest_sha256": split_hash,
        "normalization_stats_sha256": stats_hash,
    }
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint {key} hash mismatch")
    state = payload.get("model_state_dict")
    state_finite = isinstance(state, Mapping) and bool(state) and all(
        isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
        for value in state.values()
    )
    metrics_raw, metrics_hash = _immutable_bytes(metrics_path)
    metrics_payload = json.loads(metrics_raw)
    if not isinstance(metrics_payload, dict):
        raise ValueError("metrics artifact must contain a mapping")
    if item.gate == "O" and metrics_payload.get("global_step") != item.stop_update:
        raise ValueError("Gate O metrics are not from exact last update")
    if item.gate == "O" and (
        metrics_payload.get("candidate_id") != item.candidate_id
        or metrics_payload.get("last_checkpoint_sha256") != last_hash
    ):
        raise ValueError("Gate O metrics do not bind the exact last checkpoint")
    if item.gate != "O":
        provenance = metrics_payload.get("provenance", {})
        if not isinstance(provenance, Mapping) or provenance.get("checkpoint_sha256") != sha256_file(last_path):
            raise ValueError("category metrics do not bind the exact last checkpoint hash")
    split_payload = json.loads(split_path.read_bytes())
    process_binding = (
        {
            "dataset_content_root": dataset_content_root,
            "sample_count": dataset_sample_count,
        }
        if dataset_content_root is not None and dataset_sample_count is not None
        else None
    )
    if item.gate != "O":
        samples_raw, samples_hash = _immutable_bytes(samples_path)
        site_path = _resolve_registered_path(
            config["screen"]["validation_site_manifest"],
            "validation site manifest",
        )
        _, site_hash = _immutable_bytes(site_path)
        if site_hash != config["screen"]["validation_site_manifest_sha256"]:
            raise ValueError("validation site manifest hash differs from config")
        screen_provenance = {
            "config_sha256": config_hash,
            "checkpoint_sha256": last_hash,
            "split_manifest_sha256": split_hash,
            "normalization_sha256": stats_hash,
            "validation_site_manifest_sha256": site_hash,
        }
        if process_binding is not None:
            validate_process_verification(
                metrics_payload,
                process_binding,
                expected_ids=split_payload["val"],
            )
        raw_categories = _validated_evaluation_evidence(
            metrics_raw,
            samples_raw,
            purpose="screen64_fixed2048",
            expected_ids=set(split_payload["val"]),
            expected_provenance=screen_provenance,
        )
    else:
        raw_categories = metrics_payload.get("category_metrics")
    if not isinstance(raw_categories, Mapping):
        raise ValueError("metrics artifact lacks category_metrics")
    expected_metrics = (
        GATE_O_METRICS
        if item.gate == "O"
        else HALVING_METRICS
    )
    category_metrics: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        raw = raw_categories.get(category)
        if not isinstance(raw, Mapping):
            raise ValueError(f"metrics lack registered category {category}")
        category_metrics[category] = {
            name: float(raw[name]) for name in expected_metrics if name in raw
        }
        if set(category_metrics[category]) != set(expected_metrics):
            raise ValueError(f"{category} metrics do not satisfy gate schema")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if process_binding is not None:
        validate_process_verification(summary, process_binding)
    if (
        summary.get("dataset_binding_sha256") != dataset_binding_sha256
        or summary.get("execution_binding_sha256") != execution_binding_sha256
    ):
        raise ValueError("training summary runtime binding mismatch")
    if item.gate != "O" and (
        summary.get("validation_purpose") != "screen64_fixed2048"
        or summary.get("validation_site_manifest_sha256") != site_hash
    ):
        raise ValueError("training summary validation site binding mismatch")
    _, best_hash = _immutable_bytes(best_path)
    lineage_raw, lineage_hash = _immutable_bytes(lineage_path)
    lineage = json.loads(lineage_raw)
    if not isinstance(lineage, dict) or lineage.get(
        "parent_checkpoint_sha256"
    ) != parent_checkpoint_sha256:
        raise ValueError("lineage sidecar differs from checkpoint authority")
    row: dict[str, object] = {
        "source_commit": source_commit,
        "source_tree_sha256": source_tree_sha256,
        "dirty_entries_sha256": dirty_entries_sha256 or source_tree_sha256,
        "dataset_binding_sha256": dataset_binding_sha256,
        "execution_binding_sha256": execution_binding_sha256,
        "command": list(command),
        "candidate_id": item.candidate_id,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "split_path": str(split_path),
        "split_sha256": split_hash,
        "normalization_stats_path": str(stats_path),
        "normalization_stats_sha256": stats_hash,
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": last_hash,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": best_hash,
        "metrics_path": str(metrics_path),
        "metrics_sha256": metrics_hash,
        "lineage_path": str(lineage_path),
        "lineage_sha256": lineage_hash,
        "runtime_seed": int(config.get("seed", 2026)),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "optimizer_updates": item.stop_update,
        "parameter_count": int(summary["parameter_count"]),
        "peak_gpu_allocated_bytes": int(summary["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(summary["peak_gpu_reserved_bytes"]),
        "wall_seconds": _finite_nonnegative(summary["wall_seconds"], "wall_seconds"),
        "gpu_name": str(summary["gpu_name"]),
        "category_metrics": category_metrics,
        "finite": bool(metrics_payload.get("finite", True)) and state_finite,
    }
    if item.gate != "O":
        row.update(
            {
                "evaluation_purpose": "screen64_fixed2048",
                "validation_site_manifest_path": str(site_path),
                "validation_site_manifest_sha256": site_hash,
                "samples_path": str(samples_path),
                "samples_sha256": samples_hash,
            }
        )
    if item.gate == "H3":
        native_raw, native_hash = _immutable_bytes(native_metrics_path)
        native_samples_raw, native_samples_hash = _immutable_bytes(native_samples_path)
        native_payload = json.loads(native_raw)
        if process_binding is not None:
            validate_process_verification(
                native_payload,
                process_binding,
                expected_ids=split_payload["val"],
            )
        native_aggregated = _validated_evaluation_evidence(
            native_raw,
            native_samples_raw,
            purpose="native400_final_census",
            expected_ids=set(split_payload["val"]),
            expected_provenance={
                "config_sha256": config_hash,
                "checkpoint_sha256": last_hash,
                "split_manifest_sha256": split_hash,
                "normalization_sha256": stats_hash,
            },
        )
        native_categories = {
            category: {
                name: float(native_aggregated[category][name])
                for name in FINAL_CANDIDATE_METRICS
            }
            for category in CATEGORIES
        }
        row.update(
            {
                "native_metrics_path": str(native_metrics_path),
                "native_metrics_sha256": native_hash,
                "native_samples_path": str(native_samples_path),
                "native_samples_sha256": native_samples_hash,
                "native_category_metrics": native_categories,
            }
        )
    expected_fields = set(REQUIRED_ARTIFACT_FIELDS)
    if item.gate != "O":
        expected_fields.update(SCREEN_EVIDENCE_FIELDS)
    if item.gate == "H3":
        expected_fields.update(NATIVE_EVIDENCE_FIELDS)
    if set(row) != expected_fields:
        raise RuntimeError("internal artifact census schema mismatch")
    return row


def _census_smoke_artifacts(
    item: ScreenCommand,
    command: Sequence[str],
    config_dir: Path,
    output_dir: Path,
    source_commit: str,
    source_tree_sha256: str,
    dirty_entries_sha256: str,
) -> dict[str, object]:
    config_path = config_dir / CONFIG_FILENAMES[int(item.candidate_id[1:])]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_path = _resolve_registered_path(config["data"]["split_manifest"], "split manifest")
    stats_path = _resolve_registered_path(config["normalization"]["stats_path"], "stats")
    run_dir = output_dir / "smoke" / item.candidate_id
    last_path, best_path = run_dir / "checkpoints/last.pt", run_dir / "checkpoints/best.pt"
    metrics_path, summary_path = run_dir / "metrics.jsonl", run_dir / "summary.json"
    for path in (last_path, best_path, metrics_path, summary_path):
        if not path.is_file():
            raise ValueError(f"smoke completed without required artifact: {path}")
    payload = torch.load(last_path, map_location="cpu", weights_only=True)
    expected_bindings = {
        "schema_version": 4,
        "global_step": 1,
        "config_sha256": canonical_sha256(config),
        "split_manifest_sha256": sha256_file(split_path),
        "normalization_stats_sha256": sha256_file(stats_path),
    }
    if not isinstance(payload, dict) or any(
        payload.get(name) != value for name, value in expected_bindings.items()
    ):
        raise ValueError("smoke checkpoint schema, update, or provenance binding mismatch")
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or rows[0].get("global_step") != 1:
        raise ValueError("smoke metrics must contain exact update 1")
    numeric_values = [
        value
        for value in rows[0].values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric_values or any(not math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("smoke metrics must be finite")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    peak_reserved = int(summary["peak_gpu_reserved_bytes"])
    if peak_reserved >= 24 * 1024**3:
        raise ValueError("smoke peak reserved GPU memory must remain below 24 GiB")
    return {
        "source_commit": source_commit,
        "source_tree_sha256": source_tree_sha256,
        "dirty_entries_sha256": dirty_entries_sha256,
        "command": list(command),
        "candidate_id": item.candidate_id,
        "config_sha256": expected_bindings["config_sha256"],
        "split_sha256": expected_bindings["split_manifest_sha256"],
        "normalization_stats_sha256": expected_bindings["normalization_stats_sha256"],
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "runtime_seed": int(config.get("seed", 2026)),
        "optimizer_updates": 1,
        "parameter_count": int(summary["parameter_count"]),
        "peak_gpu_allocated_bytes": int(summary["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": peak_reserved,
        "wall_seconds": _finite_nonnegative(summary["wall_seconds"], "wall_seconds"),
        "gpu_name": str(summary["gpu_name"]),
    }


def _run_child(
    command: Sequence[str], lock: ScreenLock, expected_source: SourceBinding
) -> None:
    require_source_unchanged(expected_source, compute_source_binding())
    process = subprocess.Popen(list(command), cwd=REPO_ROOT, pass_fds=(lock.fd,))
    lock.set_active_child(process.pid, command)
    try:
        return_code = process.wait()
    except BaseException:
        # Keep the child binding durable when the runner is interrupted while the
        # child may still be alive. A subsequent runner will reject the orphan.
        raise
    else:
        lock.clear_active_child()
    require_source_unchanged(expected_source, compute_source_binding())
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _reject_unpublished_run_directory(
    item: ScreenCommand, output_dir: Path
) -> None:
    run_dir = (
        output_dir / "smoke" / item.candidate_id
        if item.gate == "smoke"
        else output_dir / item.gate.lower() / item.candidate_id
    )
    if item.kind == "train" and run_dir.exists():
        if "--resume" not in item.command:
            raise ValueError(
                f"unpublished candidate directory exists; refusing overwrite: {run_dir}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Task 12 finalization: rerun with --resume --baseline-metrics "
            "PATH after H3 reports status awaiting_baseline."
        ),
    )
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.baseline_metrics is not None and not args.resume:
        raise ValueError("--baseline-metrics is legal only together with --resume")
    if args.baseline_metrics is not None and args.dry_run:
        raise ValueError("--baseline-metrics finalization cannot be combined with --dry-run")
    args.config_dir = args.config_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    manifest_path = args.output_dir / "screen_manifest.json"
    resume_value: Path | None = manifest_path if args.resume else None
    if args.resume and not manifest_path.is_file():
        raise ValueError("--resume requires an existing screen_manifest.json")
    if not args.resume and manifest_path.exists():
        raise ValueError("screen manifest already exists; use --resume to avoid overwrite")
    if args.baseline_metrics is not None:
        with ScreenLock(args.output_dir / ".screen.lock"):
            manifest = _load_manifest(manifest_path)
            _validate_canonical_artifact_paths(
                manifest, args.config_dir, args.output_dir
            )
            source = compute_source_binding()
            if any(
                manifest.get(name) != value
                for name, value in source.to_dict().items()
            ):
                raise ValueError(
                    "baseline finalization source binding differs from manifest"
                )
            finalized = finalize_h3_with_baseline(
                manifest, args.baseline_metrics.resolve()
            )
            atomic_write_json(manifest_path, finalized)
        print(json.dumps(finalized["decisions"]["H3"], sort_keys=True))
        return 0
    plan = build_screen_plan(
        args.config_dir,
        args.output_dir,
        resume_manifest=resume_value,
        device=args.device,
    )
    if args.dry_run:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0
    if plan.action == "finalize_current_stage":
        with ScreenLock(args.output_dir / ".screen.lock"):
            manifest = _load_manifest(manifest_path)
            source = compute_source_binding()
            if any(
                manifest.get(name) != value
                for name, value in source.to_dict().items()
            ):
                raise ValueError("stage finalization source binding differs from manifest")
            finalized = finalize_current_stage(manifest, plan.review_boundary)
            atomic_write_json(manifest_path, finalized)
        print(json.dumps(finalized["stage"], sort_keys=True))
        return 0
    if not plan.all_commands:
        print(json.dumps(plan.to_dict(), sort_keys=True))
        return 0
    with ScreenLock(args.output_dir / ".screen.lock") as lock:
        source = compute_source_binding()
        dataset_binding = load_dataset_binding(args.config_dir)
        execution_binding = compute_execution_binding(args.device)
        manifest: dict[str, Any] = (
            _load_manifest(manifest_path)
            if args.resume
            else {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                **source.to_dict(),
                "dataset_binding": dataset_binding,
                "execution_binding": execution_binding,
                "artifacts": {},
                "decisions": {},
                "stage": {"gate": "smoke", "status": "in_progress"},
            }
        )
        if args.resume and any(
            manifest.get(name) != value for name, value in source.to_dict().items()
        ):
            raise ValueError("resume source binding differs from immutable manifest")
        if args.resume and manifest.get("dataset_binding") != dataset_binding:
            raise ValueError("resume dataset binding differs from immutable manifest")
        if args.resume and manifest.get("execution_binding") != execution_binding:
            raise ValueError("resume execution environment differs from immutable manifest")
        if not args.resume:
            atomic_write_json(manifest_path, manifest)
        started = time.monotonic()
        train_items = {
            item.candidate_id: item
            for item in plan.stage_commands
            if item.kind in {"train", "finalize_training"}
        }
        for item in plan.all_commands:
            command = _materialize_command(
                item.command,
                args.config_dir,
                args.output_dir,
                dataset_binding["sha256"],
                execution_binding["sha256"],
            )
            _reject_unpublished_run_directory(item, args.output_dir)
            if item.kind != "publish":
                _run_child(command, lock, source)
            if item.gate == "smoke":
                manifest.setdefault("smokes", {})[item.candidate_id] = (
                    _census_smoke_artifacts(
                        item,
                        command,
                        args.config_dir,
                        args.output_dir,
                        source.source_commit,
                        source.source_tree_sha256,
                        source.dirty_entries_sha256,
                    )
                )
                manifest["stage"] = {
                    "gate": "smoke",
                    "status": "in_progress",
                    "completed_candidates": sorted(manifest["smokes"]),
                }
                if set(manifest["smokes"]) == set(CANDIDATE_IDS):
                    manifest["stage"] = {"gate": "O", "status": "in_progress"}
                atomic_write_json(manifest_path, manifest)
                continue
            if item.gate == "O" and item.kind in {"train", "finalize_overfit"}:
                row = census_candidate_artifacts(
                    item,
                    command,
                    args.config_dir,
                    args.output_dir,
                    source.source_commit,
                    source.source_tree_sha256,
                    None,
                    source.dirty_entries_sha256,
                    dataset_binding["sha256"],
                    execution_binding["sha256"],
                    dataset_binding["dataset_content_root"],
                    dataset_binding["sample_count"],
                )
                gate_rows = manifest.setdefault("artifacts", {}).setdefault("O", {})
                if item.candidate_id in gate_rows:
                    raise ValueError("refusing to overwrite completed Gate O row")
                gate_rows[item.candidate_id] = row
                manifest["stage"] = {
                    "gate": "O",
                    "status": "in_progress",
                    "completed_candidates": sorted(gate_rows),
                }
                atomic_write_json(manifest_path, manifest)
                continue
            if item.kind not in {"evaluate", "evaluate_native", "publish"}:
                continue
            if item.gate == "H3" and item.kind == "evaluate":
                continue
            train_item = train_items.get(item.candidate_id, item)
            prior_gate = (
                "H1" if item.gate == "H2" else "H2" if item.gate == "H3" else None
            )
            parent_hash = (
                manifest["artifacts"][prior_gate][item.candidate_id][
                    "last_checkpoint_sha256"
                ]
                if prior_gate is not None
                else None
            )
            row = census_candidate_artifacts(
                train_item,
                _materialize_command(
                    train_item.command,
                    args.config_dir,
                    args.output_dir,
                    dataset_binding["sha256"],
                    execution_binding["sha256"],
                ),
                args.config_dir,
                args.output_dir,
                source.source_commit,
                source.source_tree_sha256,
                parent_hash,
                source.dirty_entries_sha256,
                dataset_binding["sha256"],
                execution_binding["sha256"],
                dataset_binding["dataset_content_root"],
                dataset_binding["sample_count"],
            )
            gate_rows = manifest.setdefault("artifacts", {}).setdefault(item.gate, {})
            if item.candidate_id in gate_rows:
                raise ValueError(f"refusing to overwrite completed {item.gate} row")
            gate_rows[item.candidate_id] = row
            manifest["stage"] = {
                "gate": item.gate,
                "status": "in_progress",
                "completed_candidates": sorted(gate_rows),
            }
            atomic_write_json(manifest_path, manifest)
        decisions = _recomputed_decisions(manifest)
        manifest["decisions"] = {
            gate: decision.to_dict() for gate, decision in decisions.items()
        }
        manifest["stage"] = {
            "gate": plan.review_boundary,
            "status": (
                "awaiting_baseline"
                if plan.review_boundary == "H3" and "H3" not in decisions
                else "complete"
            ),
        }
        manifest["last_review_boundary"] = plan.review_boundary
        manifest["invocation_wall_seconds"] = time.monotonic() - started
        atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            manifest["decisions"].get(plan.review_boundary, manifest["stage"]),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
