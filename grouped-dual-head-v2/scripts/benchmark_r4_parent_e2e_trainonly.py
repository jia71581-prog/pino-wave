#!/usr/bin/env python
"""Leakage-safe train-only end-to-end runtime benchmark for the frozen R4 parent.

The source VDS is accessed through an explicit non-truth allowlist.  Predictions
are materialized as C-contiguous CPU arrays and are never serialized.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import V3DataManifest, V3RecordIndex
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.evaluation import time_axis_sha256
from scripts.train_saved_time_v4_full_support import probe_variant_for_config
from scripts.train_saved_time_v4_probe import _model


CANDIDATE = "r4e7_parent_e2e_runtime_train9_v1"
CONFIG_PATH = PROJECT_ROOT / "configs/saved_time_v4/generated/a3_alltrain_rel_l2_pde_pml_ic_current_vds_r4.yaml"
BASE_CONFIG_PATH = PROJECT_ROOT / "configs/grouped_v3/continuous_pilot_w128_legacy_norm_marmousi1_4m_v2.yaml"
PARENT_IDENTITY_PATH = PROJECT_ROOT / "configs/saved_time_v4/generated/w2_parent_identity_legacy_norm_marmousi1_4m_v2.json"
RUN_IDENTITY_PATH = Path("/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/a3_alltrain_rel_l2_pde_pml_ic_current_vds_r4/run/run_identity.json")
CHECKPOINT_PATH = Path("/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/a3_alltrain_rel_l2_pde_pml_ic_current_vds_r4/run/checkpoints/epoch_0007.pt")
CHECKPOINT_SHA256 = "448035bd0061205c67799eeef3b023a71b49e2db7028b6155078be1a886de789"
MANIFEST_PATH = Path("/home/jiayh/Data/data/processed/grouped_v3_manifest_marmousi1_4m_v2.json")
NORMALIZATION_PATH = Path("/home/jiayh/Data/data/processed/grouped_v3_normalization_marmousi1_4m_v2_checkpoint_compatible.json")
SOURCE_H5_PATH = Path("/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5")
RUNTIME_REFERENCE_PATH = Path("/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/target10_instance_finetune_r2_muon/traditional_lwc84_runtime.json")
PREREGISTRATION_PATH = PROJECT_ROOT / "results/r4e7_parent_e2e_runtime_train9_v1_preregistration_20260825.json"
RESULT_PATH = PROJECT_ROOT / "results/r4e7_parent_e2e_runtime_train9_v1_20260825.json"
TEST_PATH = PROJECT_ROOT / "tests/saved_time_phase_operator_v4/test_r4_parent_runtime_benchmark.py"
SCRIPT_PATH = Path(__file__).resolve()

FAMILIES = ("uniform", "layered", "marmousi")
RECORDS_PER_FAMILY = 3
STORED_TIME_COUNT = 401
OUTPUT_SHAPE = (1, 401, 201, 201)
TRADITIONAL_REFERENCE_S = 20.34137312322855
RUNTIME_GATE_S = 2.034137312322855
PEAK_GATE_GIB = 23.5
PEAK_GATE_BYTES = int(PEAK_GATE_GIB * 1024**3)
MIN_FREE_BYTES = 2_147_483_648
MAX_RESULT_BYTES = 10 * 1024**2
GPU_BUDGET_SECONDS = 900.0
DEFAULT_TIME_BLOCK = 16

ALLOWED_H5_DATASETS = frozenset(
    {
        "velocity_mps",
        "source_map",
        "source_x_m",
        "source_z_m",
        "source_f0_hz",
        "source_t0_s",
        "source_amplitude",
    }
)
FORBIDDEN_DATASET_TOKENS = (
    "wavefield",
    "truth",
    "target",
    "label",
    "pressure",
    "onset_frame",
    "observation",
)

TIMING_BOUNDARY = {
    "start": "after excluded record disk reads and a pre-timing CUDA synchronization; before all per-record CPU tensor construction and H2D transfer",
    "end": "after parent inference at all 401 ordered stored times, physical decode, blocking D2H, explicit CUDA synchronization, C-contiguous CPU copy, and shape/finite/contiguity checks",
    "included": [
        "per-record CPU tensor construction",
        "per-record H2D transfers for velocity, source, source map, time axis, x axis, z axis, and record mapping",
        "medium encoding",
        "source preparation",
        "dense-grid preparation",
        "parent inference for all 401 stored times in manifest order",
        "physical pressure decode",
        "blocking D2H transfer",
        "explicit CUDA synchronization",
        "C-contiguous CPU output copy and shape/finite/contiguity verification",
    ],
    "excluded": ["record input disk I/O", "one-time model/checkpoint/normalizer loading"],
}

CODE_BINDING_PATHS = (
    SCRIPT_PATH,
    TEST_PATH,
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


class BenchmarkContractError(RuntimeError):
    """A frozen benchmark contract is invalid."""


class TruthAccessError(BenchmarkContractError):
    """A sealed truth dataset access was attempted."""


class BindingDriftError(BenchmarkContractError):
    """A preregistered immutable binding changed."""


class DiskRiskError(BenchmarkContractError):
    """The minimum free-disk requirement is not met."""


class BudgetExceededError(BenchmarkContractError):
    """The one-GPU budget was exhausted."""


@dataclass(frozen=True)
class SelectedRecord:
    source_index: int
    sample_id: str
    group_id: str
    family: str
    split: str
    split_id: int
    manifest_sample_sha256: str


@dataclass(frozen=True)
class RecordInput:
    record: SelectedRecord
    velocity_mps: np.ndarray
    source_parameters: np.ndarray
    source_map: np.ndarray
    nontruth_input_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    if target.suffix.lower() in {".h5", ".hdf5"}:
        raise TruthAccessError("HDF5 files must not be byte-hashed by this benchmark")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: Sequence[float], probability: float) -> float:
    numbers = sorted(float(value) for value in values)
    fraction = float(probability)
    if not numbers or not 0.0 < fraction <= 1.0:
        raise ValueError("nearest-rank requires nonempty values and p in (0,1]")
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("nearest-rank values must be finite")
    return numbers[math.ceil(fraction * len(numbers)) - 1]


def assert_dataset_read_allowed(name: str) -> None:
    normalized = str(name).strip("/")
    lowered = normalized.lower()
    if normalized not in ALLOWED_H5_DATASETS or any(
        token in lowered for token in FORBIDDEN_DATASET_TOKENS
    ):
        raise TruthAccessError(f"sealed or unregistered HDF5 dataset access denied: {name}")


def _read_h5_array(handle: h5py.File, name: str, index: int) -> np.ndarray:
    assert_dataset_read_allowed(name)
    return np.asarray(handle[name][int(index)])


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest_payload(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(payload, dict):
        raise BenchmarkContractError("registered manifest is not a JSON object")
    content = dict(payload)
    registered = str(content.pop("digest", ""))
    if not registered or _canonical_digest(content) != registered:
        raise BindingDriftError("registered manifest content digest mismatch")
    if len(payload.get("time_s", [])) != STORED_TIME_COUNT:
        raise BindingDriftError("registered manifest no longer has 401 stored times")
    time_values = np.asarray(payload["time_s"], dtype=np.float64)
    if not np.isfinite(time_values).all() or not np.all(np.diff(time_values) > 0):
        raise BindingDriftError("stored time axis must be finite and strictly ordered")
    return payload


def manifest_object(payload: Mapping[str, Any]) -> V3DataManifest:
    return V3DataManifest(
        schema=str(payload["schema"]),
        source_path=str(payload["source_path"]),
        source_file_sha256=str(payload["source_file_sha256"]),
        source_manifest_sha256=str(payload["source_manifest_sha256"]),
        source_config_sha256=str(payload["source_config_sha256"]),
        allowed_medium_types=tuple(str(value) for value in payload["allowed_medium_types"]),
        excluded_medium_types=tuple(str(value) for value in payload["excluded_medium_types"]),
        counts_before={
            str(split): {str(family): int(count) for family, count in counts.items()}
            for split, counts in payload["counts_before"].items()
        },
        counts_after={str(split): int(count) for split, count in payload["counts_after"].items()},
        indices_by_split={
            str(split): tuple(int(value) for value in values)
            for split, values in payload["indices_by_split"].items()
        },
        records=tuple(
            V3RecordIndex(
                source_index=int(row["source_index"]),
                sample_id=str(row["sample_id"]),
                group_id=str(row["group_id"]),
                sample_sha256=str(row["sample_sha256"]),
                split=str(row["split"]),
                split_id=int(row["split_id"]),
                medium_type=str(row["medium_type"]),
            )
            for row in payload["records"]
        ),
        time_s=tuple(float(value) for value in payload["time_s"]),
        x_m=tuple(float(value) for value in payload["x_m"]),
        z_m=tuple(float(value) for value in payload["z_m"]),
        digest=str(payload["digest"]),
    )


def select_train_records(payload: Mapping[str, Any]) -> tuple[SelectedRecord, ...]:
    selected: list[SelectedRecord] = []
    used_groups: set[str] = set()
    rows = payload.get("records", [])
    for family in FAMILIES:
        family_count = 0
        for row in rows:
            if row.get("split") != "train" or row.get("medium_type") != family:
                continue
            group = str(row["group_id"])
            if group in used_groups:
                continue
            selected.append(
                SelectedRecord(
                    source_index=int(row["source_index"]),
                    sample_id=str(row["sample_id"]),
                    group_id=group,
                    family=family,
                    split="train",
                    split_id=int(row["split_id"]),
                    manifest_sample_sha256=str(row["sample_sha256"]),
                )
            )
            used_groups.add(group)
            family_count += 1
            if family_count == RECORDS_PER_FAMILY:
                break
        if family_count != RECORDS_PER_FAMILY:
            raise BenchmarkContractError(f"insufficient group-disjoint train records: {family}")
    counts = Counter(row.family for row in selected)
    if len(selected) != 9 or counts != Counter({family: 3 for family in FAMILIES}):
        raise BenchmarkContractError("selected sample census is not exactly 3 per family")
    if len({row.group_id for row in selected}) != len(selected):
        raise BenchmarkContractError("selected train groups are not globally disjoint")
    if any(row.split != "train" for row in selected):
        raise TruthAccessError("non-train record entered the runtime panel")
    return tuple(selected)


def _hash_input_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def nontruth_input_digest(
    record: SelectedRecord,
    velocity_mps: np.ndarray,
    source_parameters: np.ndarray,
    source_map: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(asdict(record), sort_keys=True, separators=(",", ":")).encode("utf8")
    )
    _hash_input_array(digest, "velocity_mps", velocity_mps)
    _hash_input_array(digest, "source_parameters", source_parameters)
    _hash_input_array(digest, "source_map", source_map)
    return digest.hexdigest()


def load_record_input(record: SelectedRecord, source_h5: Path = SOURCE_H5_PATH) -> RecordInput:
    if record.split != "train":
        raise TruthAccessError("only train inputs may be loaded")
    with h5py.File(source_h5, "r", swmr=True) as handle:
        velocity = np.ascontiguousarray(
            _read_h5_array(handle, "velocity_mps", record.source_index), dtype=np.float32
        )
        source_map = np.ascontiguousarray(
            _read_h5_array(handle, "source_map", record.source_index), dtype=np.float32
        )
        source = np.asarray(
            [
                _read_h5_array(handle, "source_x_m", record.source_index).item(),
                _read_h5_array(handle, "source_z_m", record.source_index).item(),
                _read_h5_array(handle, "source_f0_hz", record.source_index).item(),
                _read_h5_array(handle, "source_t0_s", record.source_index).item(),
                _read_h5_array(handle, "source_amplitude", record.source_index).item(),
            ],
            dtype=np.float32,
        )
    if velocity.shape != (201, 201) or source_map.shape != (201, 201) or source.shape != (5,):
        raise BenchmarkContractError("non-truth model input shape changed")
    if not (np.isfinite(velocity).all() and np.isfinite(source_map).all() and np.isfinite(source).all()):
        raise BenchmarkContractError("non-truth model inputs contain non-finite values")
    digest = nontruth_input_digest(record, velocity, source, source_map)
    return RecordInput(record, velocity, source, source_map, digest)


def free_disk_bytes(path: Path = PROJECT_ROOT) -> int:
    return int(shutil.disk_usage(path).free)


def require_free_disk(path: Path = PROJECT_ROOT, minimum: int = MIN_FREE_BYTES) -> int:
    free = free_disk_bytes(path)
    if free < int(minimum):
        raise DiskRiskError(f"free disk {free} bytes is below required {int(minimum)} bytes")
    return free


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def source_data_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    resolved = SOURCE_H5_PATH.expanduser().resolve()
    stat = resolved.stat()
    if resolved != Path(str(manifest["source_path"])).resolve():
        raise BindingDriftError("source VDS path and manifest source path disagree")
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "existing_manifest_source_file_sha256": str(manifest["source_file_sha256"]),
        "existing_source_manifest_sha256": str(manifest["source_manifest_sha256"]),
        "existing_source_config_sha256": str(manifest["source_config_sha256"]),
        "verification": "stat identity plus registered manifest hashes; source VDS bytes and raw truth arrays are not hashed",
    }


def _nvidia_smi_rows() -> tuple[list[dict[str, Any]], str]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise BenchmarkContractError("unexpected nvidia-smi GPU query output")
        rows.append(
            {
                "physical_index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": int(parts[4]),
            }
        )
    summary = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"CUDA Version:\s*([0-9.]+)", summary)
    return rows, match.group(1) if match else "unavailable"


def gpu_identity(physical_index: int, device: torch.device) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(int(physical_index)):
        raise BenchmarkContractError(
            f"CUDA_VISIBLE_DEVICES must expose exactly physical GPU {physical_index}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or device != torch.device("cuda:0"):
        raise BenchmarkContractError("benchmark requires exactly one visible GPU as cuda:0")
    rows, driver_cuda = _nvidia_smi_rows()
    matches = [row for row in rows if row["physical_index"] == int(physical_index)]
    if len(matches) != 1:
        raise BenchmarkContractError("physical GPU identity is unavailable")
    props = torch.cuda.get_device_properties(device)
    row = dict(matches[0])
    if row["name"] != props.name:
        raise BindingDriftError("torch and nvidia-smi GPU names disagree")
    row.update(
        {
            "visible_index": 0,
            "cuda_visible_devices": visible,
            "torch_name": props.name,
            "torch_total_memory_bytes": int(props.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "nvidia_driver_cuda_version": driver_cuda,
        }
    )
    return row


def software_identity() -> dict[str, Any]:
    cudnn = torch.backends.cudnn.version()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": str(torch.version.cuda),
        "cudnn": "unavailable" if cudnn is None else str(cudnn),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }


def assert_no_placeholders_or_nulls(value: Any, path: str = "root") -> None:
    if value is None:
        raise BenchmarkContractError(f"null preregistration value at {path}")
    if isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in ("<...>", "tbd", "placeholder")):
            raise BenchmarkContractError(f"placeholder preregistration value at {path}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_placeholders_or_nulls(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_placeholders_or_nulls(item, f"{path}[{index}]")


def atomic_json_exclusive(payload: Mapping[str, Any], path: Path, *, limit: int = MAX_RESULT_BYTES) -> int:
    destination = path.resolve()
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"atomic destination directory is absent: {destination.parent}")
    if destination.exists():
        raise FileExistsError(f"terminal artifact already exists: {destination}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf8")
    if len(encoded) > int(limit):
        raise BenchmarkContractError(
            f"terminal JSON size {len(encoded)} exceeds limit {int(limit)}"
        )
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


class BudgetGuard:
    def __init__(self, maximum_seconds: float = GPU_BUDGET_SECONDS) -> None:
        self.maximum_seconds = float(maximum_seconds)
        self.started = time.monotonic()

    def elapsed(self) -> float:
        return float(time.monotonic() - self.started)

    def check(self, stage: str) -> float:
        value = self.elapsed()
        if value > self.maximum_seconds:
            raise BudgetExceededError(
                f"one-GPU budget exceeded at {stage}: {value:.6f}s > {self.maximum_seconds:.6f}s"
            )
        return value


def load_model_context(device: torch.device):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf8"))
    parent_identity = json.loads(PARENT_IDENTITY_PATH.read_text(encoding="utf8"))
    run_identity = json.loads(RUN_IDENTITY_PATH.read_text(encoding="utf8"))
    manifest_payload = load_manifest_payload()
    manifest = manifest_object(manifest_payload)
    if manifest.digest != str(run_identity["manifest_digest"]):
        raise BindingDriftError("run identity and active manifest digest disagree")
    if time_axis_sha256(manifest.time_s) != str(run_identity["time_axis_sha256"]):
        raise BindingDriftError("run identity and 401-point time axis disagree")
    base = V3Config.from_yaml(str(BASE_CONFIG_PATH))
    if Path(base.data.source_h5).resolve() != SOURCE_H5_PATH.resolve():
        raise BindingDriftError("base config source path changed")
    variant = probe_variant_for_config(config, parent_identity)
    model = _model(base, manifest, variant).to(device)
    load_checkpoint(
        CHECKPOINT_PATH,
        model=model,
        expected_manifest_digest=str(run_identity["manifest_digest"]),
        expected_config_digest=str(run_identity["run_digest"]),
        restore_rng=False,
        map_location=device,
    )
    model.eval()
    normalizer_payload = json.loads(NORMALIZATION_PATH.read_text(encoding="utf8"))
    normalizer = PhysicalNormalizer.from_dict(
        normalizer_payload, expected_manifest=manifest.digest
    )
    return model, normalizer, manifest_payload, run_identity


def validate_cpu_output(output: np.ndarray) -> dict[str, Any]:
    if tuple(output.shape) != OUTPUT_SHAPE:
        raise BenchmarkContractError(
            f"materialized output shape {tuple(output.shape)} != {OUTPUT_SHAPE}"
        )
    if output.dtype != np.float32:
        raise BenchmarkContractError(f"materialized output dtype is {output.dtype}, not float32")
    if not output.flags.c_contiguous:
        raise BenchmarkContractError("materialized CPU output is not C-contiguous")
    if not np.isfinite(output).all():
        raise FloatingPointError("materialized CPU output contains non-finite values")
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "device": "cpu",
        "c_contiguous": True,
        "finite": True,
        "maximum_absolute_value": float(np.max(np.abs(output))),
        "serialized": False,
    }


@torch.inference_mode()
def execute_record(
    model,
    normalizer: PhysicalNormalizer,
    record_input: RecordInput,
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
    time_block: int,
) -> dict[str, Any]:
    if record_input.record.split != "train":
        raise TruthAccessError("timed execution received a non-train record")
    if len(manifest["time_s"]) != STORED_TIME_COUNT:
        raise BenchmarkContractError("timed execution requires all 401 stored times")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    velocity = torch.from_numpy(record_input.velocity_mps)[None, None].to(device)
    source = torch.from_numpy(record_input.source_parameters)[None].to(device)
    source_map = torch.from_numpy(record_input.source_map)[None, None].to(device)
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
    physical = normalizer.decode_pressure(normalized.float(), source[:, 4])
    cpu_tensor = physical.detach().to("cpu", non_blocking=False).contiguous()
    torch.cuda.synchronize(device)
    output = np.array(cpu_tensor.numpy(), dtype=np.float32, copy=True, order="C")
    output_metadata = validate_cpu_output(output)
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise BenchmarkContractError("timed runtime is not finite and positive")
    return {
        "runtime_s": float(elapsed),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "output": output_metadata,
    }


def _record_census_with_hashes(records: Sequence[SelectedRecord]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        loaded = load_record_input(record)
        result.append({**asdict(record), "nontruth_input_sha256": loaded.nontruth_input_sha256})
    return result


def _binding_map(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {str(path.resolve()): file_binding(path) for path in paths}


def build_preregistration(
    *,
    physical_gpu_index: int,
    device: torch.device,
    time_block: int,
    focused_test_command: str,
    focused_test_result: str,
    smoke_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    free = require_free_disk()
    manifest = load_manifest_payload()
    records = select_train_records(manifest)
    run_identity = json.loads(RUN_IDENTITY_PATH.read_text(encoding="utf8"))
    reference = json.loads(RUNTIME_REFERENCE_PATH.read_text(encoding="utf8"))
    if (
        reference.get("status") != "complete"
        or float(reference.get("conservative_reference_runtime_s", -1.0))
        != TRADITIONAL_REFERENCE_S
        or reference.get("protocol", {}).get("saved_frames") != STORED_TIME_COUNT
        or reference.get("protocol", {}).get("includes_output_materialization") is not True
        or reference.get("protocol", {}).get("cuda_synchronized_timing") is not True
    ):
        raise BindingDriftError("traditional runtime reference contract changed")
    gpu = gpu_identity(physical_gpu_index, device)
    checkpoint_binding = file_binding(CHECKPOINT_PATH)
    if checkpoint_binding["sha256"] != CHECKPOINT_SHA256:
        raise BindingDriftError("rollback parent checkpoint SHA-256 mismatch")
    measured_command = (
        "env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python "
        "scripts/benchmark_r4_parent_e2e_trainonly.py --mode measured "
        "--physical-gpu-index 0 --device cuda:0 --time-block 16 "
        "--preregistration results/r4e7_parent_e2e_runtime_train9_v1_preregistration_20260825.json "
        "--result results/r4e7_parent_e2e_runtime_train9_v1_20260825.json"
    )
    if physical_gpu_index != 0 or str(device) != "cuda:0" or int(time_block) != 16:
        raise BenchmarkContractError("preregistered measured command is fixed to physical GPU 0 and time block 16")
    payload = {
        "schema": "r4_parent_trainonly_runtime_preregistration_v1",
        "candidate": CANDIDATE,
        "status": "frozen",
        "created_utc": utc_now(),
        "hypothesis": "The frozen epoch-7 R4 parent can materialize its complete 401-time 201x201 physical CPU output for a group-disjoint nine-record train panel in at most one tenth of the conservative traditional runtime in arithmetic mean and nearest-rank P95 while peak reserved CUDA memory remains at most 23.5 GiB.",
        "falsifiable_claim": "runtime-only: reject unless all three jointly registered latency and peak-memory gates pass on the exact single-GPU command",
        "claim_scope": "train_only_runtime_only",
        "acceptance": {
            "record_count": 9,
            "families": {family: 3 for family in FAMILIES},
            "mean_runtime_s_maximum": RUNTIME_GATE_S,
            "p95_runtime_s_maximum": RUNTIME_GATE_S,
            "p95_definition": "nearest-rank ceil(p*n), no interpolation; p=0.95 and n=9 therefore maximum",
            "peak_cuda_reserved_bytes_maximum": PEAK_GATE_BYTES,
            "peak_cuda_gib_maximum": PEAK_GATE_GIB,
            "joint_gate": "all latency and conservative peak-reserved conditions must pass",
        },
        "failure_signal": "Any gate miss yields success/fail_gate; any OOM, non-finite output, truth access attempt, shape/time/contiguity violation, binding drift, disk risk, atomic-write risk, multi-GPU exposure, or budget overrun yields failed or blocked without a runtime claim.",
        "budget": {
            "physical_gpu_count": 1,
            "gpu_hours_maximum": 0.25,
            "gpu_seconds_maximum": GPU_BUDGET_SECONDS,
            "minimum_free_disk_bytes": MIN_FREE_BYTES,
            "free_disk_bytes_at_freeze": free,
            "result_size_bytes_maximum": MAX_RESULT_BYTES,
        },
        "atomic_terminal_protocol": {
            "single_result_path": str(RESULT_PATH.resolve()),
            "same_filesystem_partial_then_os_replace": True,
            "exclusive_no_overwrite": True,
            "partial_removed_on_exit": True,
            "terminal_states": ["success/pass", "success/fail_gate", "blocked", "failed"],
        },
        "rollback": {
            **checkpoint_binding,
            "read_only": True,
            "expected_sha256": CHECKPOINT_SHA256,
            "checkpoint_writes_permitted": False,
        },
        "measured_command": measured_command,
        "warmup": {
            "count": 1,
            "record": asdict(records[0]),
            "unscored": True,
            "same_materialization_path_as_measured": True,
        },
        "timing_boundary": TIMING_BOUNDARY,
        "time_block": int(time_block),
        "sample_selection": {
            "algorithm": "for each family in uniform, layered, marmousi order, select the first three train manifest rows whose group has not appeared previously",
            "group_disjoint": True,
            "records": _record_census_with_hashes(records),
        },
        "sealed_data_contract": {
            "allowed_split": "train",
            "validation_opened": False,
            "test_id_opened": False,
            "future_truth_opened": False,
            "train_truth_opened": False,
            "onset_truth_opened": False,
            "allowed_h5_datasets": sorted(ALLOWED_H5_DATASETS),
            "forbidden_h5_dataset_tokens": list(FORBIDDEN_DATASET_TOKENS),
            "prediction_arrays_written_to_disk": False,
        },
        "bindings": {
            "parent_checkpoint": checkpoint_binding,
            "run_identity": file_binding(RUN_IDENTITY_PATH),
            "run_digest": str(run_identity["run_digest"]),
            "manifest_digest": str(run_identity["manifest_digest"]),
            "time_axis_sha256": str(run_identity["time_axis_sha256"]),
            "config": file_binding(CONFIG_PATH),
            "base_config": file_binding(BASE_CONFIG_PATH),
            "parent_identity": file_binding(PARENT_IDENTITY_PATH),
            "manifest_json": file_binding(MANIFEST_PATH),
            "normalization_json": file_binding(NORMALIZATION_PATH),
            "source_data": source_data_binding(manifest),
            "code": _binding_map(CODE_BINDING_PATHS),
            "traditional_runtime_reference": {
                **file_binding(RUNTIME_REFERENCE_PATH),
                "conservative_reference_runtime_s": TRADITIONAL_REFERENCE_S,
                "protocol_schema": str(reference["schema"]),
            },
            "gpu": gpu,
            "software": software_identity(),
        },
        "preflight_evidence": {
            "focused_test_command": focused_test_command,
            "focused_test_result": focused_test_result,
            "focused_tests_passed": True,
            "dry_smoke": dict(smoke_evidence),
        },
    }
    assert_no_placeholders_or_nulls(payload)
    return payload


def verify_binding(binding: Mapping[str, Any]) -> None:
    path = Path(str(binding["path"]))
    current = file_binding(path)
    for key in ("sha256", "size_bytes", "mtime_ns"):
        if current[key] != binding[key]:
            raise BindingDriftError(f"binding drift for {path}: {key}")


def verify_frozen_preregistration(payload: Mapping[str, Any]) -> None:
    assert_no_placeholders_or_nulls(payload)
    if payload.get("candidate") != CANDIDATE or payload.get("status") != "frozen":
        raise BindingDriftError("preregistration candidate or frozen status mismatch")
    bindings = payload["bindings"]
    for name in (
        "parent_checkpoint",
        "run_identity",
        "config",
        "base_config",
        "parent_identity",
        "manifest_json",
        "normalization_json",
        "traditional_runtime_reference",
    ):
        verify_binding(bindings[name])
    for binding in bindings["code"].values():
        verify_binding(binding)
    source = bindings["source_data"]
    stat = Path(str(source["path"])).stat()
    current_source = {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }
    for key, value in current_source.items():
        if value != source[key]:
            raise BindingDriftError(f"source data stat binding drift: {key}")
    manifest = load_manifest_payload()
    if str(manifest["digest"]) != bindings["manifest_digest"]:
        raise BindingDriftError("manifest digest drift")
    if time_axis_sha256(manifest["time_s"]) != bindings["time_axis_sha256"]:
        raise BindingDriftError("time-axis binding drift")
    selected = select_train_records(manifest)
    current_records = _record_census_with_hashes(selected)
    if current_records != payload["sample_selection"]["records"]:
        raise BindingDriftError("selected sample or non-truth input binding drift")


def dry_smoke(
    *, physical_gpu_index: int, device: torch.device, time_block: int
) -> dict[str, Any]:
    require_free_disk()
    gpu = gpu_identity(physical_gpu_index, device)
    if sha256_file(CHECKPOINT_PATH) != CHECKPOINT_SHA256:
        raise BindingDriftError("parent checkpoint SHA-256 mismatch before dry smoke")
    model, normalizer, manifest, _ = load_model_context(device)
    record = select_train_records(manifest)[0]
    loaded = load_record_input(record)
    measurement = execute_record(
        model, normalizer, loaded, manifest, device=device, time_block=time_block
    )
    return {
        "status": "passed",
        "unscored": True,
        "utc": utc_now(),
        "record": asdict(record),
        "nontruth_input_sha256": loaded.nontruth_input_sha256,
        "time_block": int(time_block),
        "gpu": gpu,
        "measurement": measurement,
        "truth_read_attempts": 0,
        "prediction_serialized": False,
        "checkpoint_writes": 0,
    }


def run_measured(
    *,
    preregistration_path: Path,
    result_path: Path,
    physical_gpu_index: int,
    device: torch.device,
    time_block: int,
) -> dict[str, Any]:
    prereg_sha = sha256_file(preregistration_path)
    prereg = json.loads(preregistration_path.read_text(encoding="utf8"))
    started_utc = utc_now()
    disk_before = free_disk_bytes()
    base_terminal: dict[str, Any] = {
        "schema": "r4_parent_trainonly_runtime_result_v1",
        "candidate": CANDIDATE,
        "preregistration_path": str(preregistration_path.resolve()),
        "preregistration_sha256": prereg_sha,
        "started_utc": started_utc,
        "claim_scope": "train_only_runtime_only",
        "disk_free_bytes_before": disk_before,
        "result_size_bytes_maximum": MAX_RESULT_BYTES,
        "sealed_data_attestation": {
            "validation_opened": False,
            "test_id_opened": False,
            "train_truth_opened": False,
            "onset_truth_opened": False,
            "truth_read_attempts": 0,
            "prediction_arrays_written_to_disk": False,
        },
        "checkpoint_attestation": {
            "writes": 0,
            "rollback_path": str(CHECKPOINT_PATH),
            "rollback_sha256_before": CHECKPOINT_SHA256,
        },
    }
    budget: BudgetGuard | None = None
    try:
        if result_path.exists():
            raise FileExistsError(f"result terminal already exists: {result_path}")
        if int(time_block) != int(prereg["time_block"]):
            raise BindingDriftError("measured time block differs from frozen preregistration")
        verify_frozen_preregistration(prereg)
        disk_verified = require_free_disk()
        gpu = gpu_identity(physical_gpu_index, device)
        if gpu != prereg["bindings"]["gpu"]:
            raise BindingDriftError("measured GPU identity differs from frozen preregistration")
        budget = BudgetGuard()
        model, normalizer, manifest, run_identity = load_model_context(device)
        budget.check("model_load")
        records = select_train_records(manifest)
        frozen_rows = prereg["sample_selection"]["records"]
        frozen_by_id = {str(row["sample_id"]): row for row in frozen_rows}

        warm_input = load_record_input(records[0])
        if warm_input.nontruth_input_sha256 != frozen_by_id[records[0].sample_id]["nontruth_input_sha256"]:
            raise BindingDriftError("warm-up non-truth input digest drift")
        warmup = execute_record(
            model,
            normalizer,
            warm_input,
            manifest,
            device=device,
            time_block=time_block,
        )
        warmup.update({"unscored": True, "record": asdict(records[0])})
        budget.check("warmup")

        measurements: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            budget.check(f"before_record_{index + 1}")
            free_now = require_free_disk()
            loaded = load_record_input(record)
            frozen = frozen_by_id.get(record.sample_id)
            if frozen is None or loaded.nontruth_input_sha256 != frozen["nontruth_input_sha256"]:
                raise BindingDriftError(f"non-truth input digest drift: {record.sample_id}")
            measured = execute_record(
                model,
                normalizer,
                loaded,
                manifest,
                device=device,
                time_block=time_block,
            )
            measured.update(
                {
                    "ordinal": index + 1,
                    **asdict(record),
                    "nontruth_input_sha256": loaded.nontruth_input_sha256,
                    "traditional_reference_s": TRADITIONAL_REFERENCE_S,
                    "speedup_x": TRADITIONAL_REFERENCE_S / float(measured["runtime_s"]),
                    "disk_free_bytes_before_record": free_now,
                }
            )
            measurements.append(measured)
            budget.check(f"after_record_{index + 1}")
        runtimes = [float(row["runtime_s"]) for row in measurements]
        mean_runtime = float(statistics.fmean(runtimes))
        p95_runtime = nearest_rank(runtimes, 0.95)
        measured_peak_allocated = max(
            int(row["peak_cuda_allocated_bytes"]) for row in measurements
        )
        measured_peak_reserved = max(
            int(row["peak_cuda_reserved_bytes"]) for row in measurements
        )
        gate_peak_allocated = max(
            measured_peak_allocated, int(warmup["peak_cuda_allocated_bytes"])
        )
        gate_peak_reserved = max(
            measured_peak_reserved, int(warmup["peak_cuda_reserved_bytes"])
        )
        gate = {
            "mean_runtime": {
                "value_s": mean_runtime,
                "maximum_s": RUNTIME_GATE_S,
                "passed": mean_runtime <= RUNTIME_GATE_S,
            },
            "nearest_rank_p95_runtime": {
                "value_s": p95_runtime,
                "maximum_s": RUNTIME_GATE_S,
                "passed": p95_runtime <= RUNTIME_GATE_S,
                "rank_for_nine": 9,
            },
            "conservative_peak_reserved": {
                "value_bytes": gate_peak_reserved,
                "maximum_bytes": PEAK_GATE_BYTES,
                "passed": gate_peak_reserved <= PEAK_GATE_BYTES,
                "includes_unscored_warmup": True,
            },
        }
        joint_pass = all(bool(value["passed"]) for value in gate.values())
        elapsed_gpu_seconds = budget.check("aggregation")
        disk_after = free_disk_bytes()
        checkpoint_after = sha256_file(CHECKPOINT_PATH)
        if checkpoint_after != CHECKPOINT_SHA256:
            raise BindingDriftError("parent checkpoint changed during benchmark")
        payload = {
            **base_terminal,
            "status": "success",
            "decision": "pass" if joint_pass else "fail_gate",
            "completed_utc": utc_now(),
            "exact_blocker": "none" if joint_pass else "one or more preregistered runtime or peak-memory gates failed",
            "disk_free_bytes_at_binding_gate": disk_verified,
            "disk_free_bytes_after": disk_after,
            "gpu": gpu,
            "software": software_identity(),
            "budget": {
                "one_gpu_seconds_used": elapsed_gpu_seconds,
                "one_gpu_seconds_maximum": GPU_BUDGET_SECONDS,
                "within_budget": elapsed_gpu_seconds <= GPU_BUDGET_SECONDS,
            },
            "run_binding": {
                "run_digest": str(run_identity["run_digest"]),
                "manifest_digest": str(run_identity["manifest_digest"]),
                "time_axis_sha256": str(run_identity["time_axis_sha256"]),
                "checkpoint_sha256": CHECKPOINT_SHA256,
            },
            "timing_boundary": TIMING_BOUNDARY,
            "warmup": warmup,
            "sample_census": {
                "count": len(measurements),
                "by_family": dict(sorted(Counter(row["family"] for row in measurements).items())),
                "unique_group_count": len({row["group_id"] for row in measurements}),
                "all_train": all(row["split"] == "train" for row in measurements),
            },
            "measurements": measurements,
            "aggregate": {
                "count": len(runtimes),
                "arithmetic_mean_runtime_s": mean_runtime,
                "nearest_rank_p95_runtime_s": p95_runtime,
                "minimum_runtime_s": min(runtimes),
                "maximum_runtime_s": max(runtimes),
                "traditional_reference_s": TRADITIONAL_REFERENCE_S,
                "reference_over_mean_speedup_x": TRADITIONAL_REFERENCE_S / mean_runtime,
                "reference_over_p95_speedup_x": TRADITIONAL_REFERENCE_S / p95_runtime,
                "arithmetic_mean_per_record_speedup_x": float(
                    statistics.fmean(float(row["speedup_x"]) for row in measurements)
                ),
                "measured_peak_cuda_allocated_bytes": measured_peak_allocated,
                "measured_peak_cuda_reserved_bytes": measured_peak_reserved,
                "gate_peak_cuda_allocated_bytes": gate_peak_allocated,
                "gate_peak_cuda_reserved_bytes": gate_peak_reserved,
            },
            "gate": {**gate, "joint_passed": joint_pass},
        }
    except (BindingDriftError, DiskRiskError, FileNotFoundError) as error:
        payload = {
            **base_terminal,
            "status": "blocked",
            "decision": "not_run_or_incomplete",
            "completed_utc": utc_now(),
            "exact_blocker": f"{type(error).__name__}: {error}",
            "disk_free_bytes_after": free_disk_bytes(),
            "budget": {
                "one_gpu_seconds_used": 0.0 if budget is None else budget.elapsed(),
                "one_gpu_seconds_maximum": GPU_BUDGET_SECONDS,
            },
        }
    except Exception as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        payload = {
            **base_terminal,
            "status": "failed",
            "decision": "incomplete",
            "completed_utc": utc_now(),
            "exact_blocker": f"{type(error).__name__}: {error}",
            "disk_free_bytes_after": free_disk_bytes(),
            "budget": {
                "one_gpu_seconds_used": 0.0 if budget is None else budget.elapsed(),
                "one_gpu_seconds_maximum": GPU_BUDGET_SECONDS,
            },
        }
    payload["checkpoint_attestation"]["rollback_sha256_after"] = sha256_file(CHECKPOINT_PATH)
    payload["checkpoint_attestation"]["unchanged"] = (
        payload["checkpoint_attestation"]["rollback_sha256_after"] == CHECKPOINT_SHA256
    )
    atomic_json_exclusive(payload, result_path)
    return payload


def _load_smoke_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if payload.get("status") != "passed" or payload.get("unscored") is not True:
        raise BenchmarkContractError("dry-smoke evidence is not a passed unscored run")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-smoke", "preregister", "measured"), required=True)
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-block", type=int, default=DEFAULT_TIME_BLOCK)
    parser.add_argument("--preregistration", type=Path, default=PREREGISTRATION_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--smoke-evidence", type=Path)
    parser.add_argument("--focused-test-command", default="")
    parser.add_argument("--focused-test-result", default="")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.time_block <= 0:
        raise BenchmarkContractError("time block must be positive")
    if args.mode == "dry-smoke":
        payload = dry_smoke(
            physical_gpu_index=args.physical_gpu_index,
            device=device,
            time_block=args.time_block,
        )
    elif args.mode == "preregister":
        if args.smoke_evidence is None:
            raise BenchmarkContractError("preregistration requires dry-smoke evidence")
        payload = build_preregistration(
            physical_gpu_index=args.physical_gpu_index,
            device=device,
            time_block=args.time_block,
            focused_test_command=str(args.focused_test_command),
            focused_test_result=str(args.focused_test_result),
            smoke_evidence=_load_smoke_evidence(args.smoke_evidence),
        )
        atomic_json_exclusive(payload, args.preregistration)
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
    "ALLOWED_H5_DATASETS",
    "BenchmarkContractError",
    "BindingDriftError",
    "BudgetExceededError",
    "BudgetGuard",
    "DiskRiskError",
    "FAMILIES",
    "FORBIDDEN_DATASET_TOKENS",
    "OUTPUT_SHAPE",
    "SelectedRecord",
    "TIMING_BOUNDARY",
    "TruthAccessError",
    "assert_dataset_read_allowed",
    "assert_no_placeholders_or_nulls",
    "atomic_json_exclusive",
    "load_manifest_payload",
    "nearest_rank",
    "require_free_disk",
    "select_train_records",
    "sha256_file",
    "validate_cpu_output",
    "verify_binding",
]
