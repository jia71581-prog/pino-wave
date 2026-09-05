"""Atomic, exactly resumable checkpoints for adaptive query training."""

from __future__ import annotations

import math
import numbers
import os
import operator
import random
import string
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .ais_normalization import (
    AISNormalizationBinding,
    build_normalized_static_features,
)
from .ais_sampler import SpatialSamplingFeatures
from .query_data import DenseCPUQueryStore, QueryScene, resize_query_scene
from .query_losses import (
    HHFieldLoss,
    auxiliary_query_losses,
    hansen_hurwitz_field_loss,
    standardized_hh_field_loss,
)
from .temporal_operator import _ValidatedTimeGrid, _make_validated_time_grid


SCHEMA_VERSION = 1
TRAINER_SCHEMA_VERSION = 2
RUNTIME_SEED_SCHEMA_VERSION = 3
NORMALIZATION_SCHEMA_VERSION = 4
SCREEN_SCHEMA_VERSION = 5
AIS_NORMALIZATION_CONTRACT = "ais_normalization_v2"
QUERY_CHECKPOINT_BOUNDARY = "after_optimizer_step_before_query_draw"
_PAYLOAD_FIELDS = {
    "schema_version",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "sampler_state_dict",
    "torch_rng_state",
    "cuda_rng_state_all",
    "numpy_rng_state",
    "python_rng_state",
    "epoch",
    "epoch_batch_cursor",
    "epoch_scene_order",
    "data_generator_state",
    "global_step",
    "config_sha256",
    "split_manifest_sha256",
}
_TRAINER_PAYLOAD_FIELDS = _PAYLOAD_FIELDS | {"phase_index", "phase_update"}
_RUNTIME_SEED_PAYLOAD_FIELDS = _TRAINER_PAYLOAD_FIELDS | {"runtime_seed"}
_NORMALIZATION_PAYLOAD_FIELDS = {
    "normalization_stats_sha256",
    "normalization_contract",
}
_SCREEN_PAYLOAD_FIELDS = {
    "screen_candidate_id",
    "screen_gate",
    "parent_checkpoint_sha256",
    "dataset_binding_sha256",
    "execution_binding_sha256",
}
_SUPPORTED_OPTIMIZER_TYPES = (
    torch.optim.SGD,
    torch.optim.Adam,
    torch.optim.AdamW,
)
_DISPERSION_NORMALIZATION_BUFFERS = {
    "velocity_mean": "dispersion_residual_head.velocity_mean",
    "velocity_std": "dispersion_residual_head.velocity_std",
}


@dataclass(frozen=True)
class QueryResumeMetadata:
    epoch: int
    epoch_batch_cursor: int
    epoch_scene_order: tuple[int, ...]
    global_step: int
    phase_index: int | None = None
    phase_update: int | None = None


@dataclass(frozen=True)
class SceneModelInputs:
    global_inputs: torch.Tensor
    native_static: torch.Tensor
    time_s: torch.Tensor
    dx_m: float
    dz_m: float
    validated_time_grid: _ValidatedTimeGrid


@dataclass(frozen=True)
class TrainEpochResult:
    mean_total_loss: float
    mean_components: dict[str, float]
    global_step: int
    optimizer_updates: int
    next_epoch_batch_cursor: int
    scene_draws: int
    full160_label_sites: int
    sampler_diagnostics: dict[str, float]


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a nonnegative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _validate_screen_dispersion_normalization(
    payload: Mapping[str, object],
    expected_velocity_mean: float | None,
    expected_velocity_std: float | None,
) -> None:
    if payload.get("schema_version") != SCREEN_SCHEMA_VERSION:
        return
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("schema-v5 checkpoint model_state_dict must be a mapping")
    present = {
        name: key in state for name, key in _DISPERSION_NORMALIZATION_BUFFERS.items()
    }
    if not any(present.values()):
        return
    if not all(present.values()):
        raise ValueError("schema-v5 dispersion normalization buffers are incomplete")
    if expected_velocity_mean is None or expected_velocity_std is None:
        raise ValueError(
            "schema-v5 dispersion buffers require bound normalization velocity values"
        )
    expected = {
        "velocity_mean": float(expected_velocity_mean),
        "velocity_std": float(expected_velocity_std),
    }
    if not math.isfinite(expected["velocity_mean"]) or not (
        math.isfinite(expected["velocity_std"]) and expected["velocity_std"] > 0.0
    ):
        raise ValueError("bound normalization velocity mean/std are invalid")
    for name, key in _DISPERSION_NORMALIZATION_BUFFERS.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 0
            or value.dtype != torch.float64
            or float(value.detach().cpu().item()) != expected[name]
        ):
            raise ValueError(
                f"schema-v5 dispersion {name} differs from bound normalization stats"
            )


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _scene_resume_metadata(
    epoch: object,
    epoch_batch_cursor: object,
    epoch_scene_order: object,
    global_step: object,
    phase_index: object | None = None,
    phase_update: object | None = None,
) -> QueryResumeMetadata:
    normalized_epoch = _nonnegative_int(epoch, "epoch")
    normalized_cursor = _nonnegative_int(epoch_batch_cursor, "epoch_batch_cursor")
    normalized_step = _nonnegative_int(global_step, "global_step")
    if isinstance(epoch_scene_order, (str, bytes)) or not isinstance(
        epoch_scene_order, Sequence
    ):
        raise TypeError("epoch_scene_order must be a sequence of sample IDs")
    normalized_order = tuple(
        _nonnegative_int(sample_id, "epoch_scene_order sample ID")
        for sample_id in epoch_scene_order
    )
    if len(set(normalized_order)) != len(normalized_order):
        raise ValueError("epoch_scene_order must not contain duplicate sample IDs")
    if normalized_cursor > len(normalized_order):
        raise ValueError("epoch_batch_cursor exceeds epoch_scene_order")
    if (phase_index is None) != (phase_update is None):
        raise ValueError("phase_index and phase_update must be provided together")
    normalized_phase_index = (
        None if phase_index is None else _nonnegative_int(phase_index, "phase_index")
    )
    normalized_phase_update = (
        None if phase_update is None else _nonnegative_int(phase_update, "phase_update")
    )
    return QueryResumeMetadata(
        normalized_epoch,
        normalized_cursor,
        normalized_order,
        normalized_step,
        normalized_phase_index,
        normalized_phase_update,
    )


def _generator_state(generator: object) -> torch.Tensor:
    if not isinstance(generator, torch.Generator):
        raise TypeError("data_generator must be a torch.Generator")
    return generator.get_state().clone()


def _validate_supported_optimizer(optimizer: object) -> None:
    if not isinstance(optimizer, _SUPPORTED_OPTIMIZER_TYPES):
        raise TypeError(
            "exact query checkpoint resume supports only torch.optim.SGD, "
            "torch.optim.Adam, and torch.optim.AdamW"
        )


def _encode_numpy_rng_state(
    state: tuple[str, np.ndarray, int, int, float],
) -> dict[str, object]:
    bit_generator, keys, position, has_gauss, cached_gaussian = state
    return {
        "bit_generator": bit_generator,
        "keys": torch.from_numpy(keys.astype(np.int64, copy=True)),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _decode_numpy_rng_state(value: object) -> tuple[str, np.ndarray, int, int, float]:
    required = {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("invalid NumPy RNG state structure")
    bit_generator = value["bit_generator"]
    keys = value["keys"]
    position = value["position"]
    has_gauss = value["has_gauss"]
    cached_gaussian = value["cached_gaussian"]
    if not isinstance(bit_generator, str):
        raise ValueError("invalid NumPy RNG bit generator")
    if (
        not isinstance(keys, torch.Tensor)
        or keys.dtype != torch.int64
        or keys.ndim != 1
    ):
        raise ValueError("invalid NumPy RNG keys")
    cpu_keys = keys.detach().to(device="cpu")
    if bool(torch.any(cpu_keys < 0)) or bool(torch.any(cpu_keys > 2**32 - 1)):
        raise ValueError("invalid NumPy RNG keys")
    normalized_position = _nonnegative_int(position, "NumPy RNG position")
    normalized_has_gauss = _nonnegative_int(has_gauss, "NumPy RNG has_gauss")
    if normalized_has_gauss not in (0, 1):
        raise ValueError("NumPy RNG has_gauss must be zero or one")
    if isinstance(cached_gaussian, bool) or not isinstance(
        cached_gaussian, (int, float)
    ):
        raise ValueError("invalid NumPy cached Gaussian")
    normalized_cached = float(cached_gaussian)
    if not np.isfinite(normalized_cached):
        raise ValueError("invalid NumPy cached Gaussian")
    return (
        bit_generator,
        cpu_keys.numpy().astype(np.uint32, copy=True),
        normalized_position,
        normalized_has_gauss,
        normalized_cached,
    )


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        # Rename is the irreversible commit point. Directory-sync failure cannot
        # atomically restore the old target, so this durability sync is best effort.
        temporary_path = None
        try:
            directory_fd = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def save_query_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    sampler: Any,
    epoch: int,
    epoch_batch_cursor: int,
    epoch_scene_order: Sequence[int],
    data_generator: torch.Generator,
    global_step: int,
    config_sha256: str,
    split_manifest_sha256: str,
    checkpoint_boundary: str,
    phase_index: int | None = None,
    phase_update: int | None = None,
    runtime_seed: int | None = None,
    normalization_stats_sha256: str | None = None,
    normalization_contract: str | None = None,
    normalization_velocity_mean: float | None = None,
    normalization_velocity_std: float | None = None,
    screen_candidate_id: str | None = None,
    screen_gate: str | None = None,
    parent_checkpoint_sha256: str | None = None,
    dataset_binding_sha256: str | None = None,
    execution_binding_sha256: str | None = None,
) -> None:
    """Save at the next-draw boundary under a single-writer contract.

    Exact optimizer validation is intentionally scoped to SGD, Adam, and AdamW.
    The rename is the commit point. Task 8 must resume by consuming the returned
    epoch order starting exactly at ``epoch_batch_cursor`` before drawing queries.
    """
    _validate_supported_optimizer(optimizer)
    if checkpoint_boundary != QUERY_CHECKPOINT_BOUNDARY:
        raise ValueError(
            "query checkpoints are only valid after an optimizer step and before "
            "the next spatial query draw"
        )
    metadata = _scene_resume_metadata(
        epoch,
        epoch_batch_cursor,
        epoch_scene_order,
        global_step,
        phase_index,
        phase_update,
    )
    config_digest = _sha256(config_sha256, "config_sha256")
    split_digest = _sha256(split_manifest_sha256, "split_manifest_sha256")
    has_normalization = (
        normalization_stats_sha256 is not None or normalization_contract is not None
    )
    if has_normalization and (
        normalization_stats_sha256 is None or normalization_contract is None
    ):
        raise ValueError("normalization hash and contract must be provided together")
    normalization_digest = (
        _sha256(normalization_stats_sha256, "normalization stats SHA-256")
        if has_normalization
        else None
    )
    if has_normalization and normalization_contract != AIS_NORMALIZATION_CONTRACT:
        raise ValueError(
            f"normalization contract must be {AIS_NORMALIZATION_CONTRACT!r}"
        )
    has_screen = screen_candidate_id is not None or screen_gate is not None
    if has_screen and (
        not isinstance(screen_candidate_id, str)
        or not isinstance(screen_gate, str)
        or screen_gate not in {"O", "H1", "H2", "H3"}
    ):
        raise ValueError("screen checkpoint requires candidate id and registered gate")
    if has_screen and not has_normalization:
        raise ValueError("screen checkpoint requires normalization binding")
    if has_screen:
        _sha256(dataset_binding_sha256, "dataset binding SHA-256")
        _sha256(execution_binding_sha256, "execution binding SHA-256")
    if parent_checkpoint_sha256 is not None:
        _sha256(parent_checkpoint_sha256, "parent checkpoint SHA-256")
    payload = {
        "schema_version": (
            SCREEN_SCHEMA_VERSION
            if has_screen
            else NORMALIZATION_SCHEMA_VERSION
            if has_normalization
            else RUNTIME_SEED_SCHEMA_VERSION
            if runtime_seed is not None
            else TRAINER_SCHEMA_VERSION
            if metadata.phase_index is not None
            else SCHEMA_VERSION
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
        "sampler_state_dict": sampler.state_dict(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state_all": [
            state.clone() for state in torch.cuda.get_rng_state_all()
        ]
        if torch.cuda.is_available()
        else [],
        "numpy_rng_state": _encode_numpy_rng_state(np.random.get_state()),
        "python_rng_state": random.getstate(),
        "epoch": metadata.epoch,
        "epoch_batch_cursor": metadata.epoch_batch_cursor,
        "epoch_scene_order": list(metadata.epoch_scene_order),
        "data_generator_state": _generator_state(data_generator),
        "global_step": metadata.global_step,
        "config_sha256": config_digest,
        "split_manifest_sha256": split_digest,
    }
    if has_screen:
        payload.update(
            {
                "screen_candidate_id": screen_candidate_id,
                "screen_gate": screen_gate,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "dataset_binding_sha256": dataset_binding_sha256,
                "execution_binding_sha256": execution_binding_sha256,
            }
        )
    if metadata.phase_index is not None:
        payload["phase_index"] = metadata.phase_index
        payload["phase_update"] = metadata.phase_update
    if runtime_seed is not None:
        payload["runtime_seed"] = _nonnegative_int(runtime_seed, "runtime_seed")
    if has_normalization:
        payload["normalization_stats_sha256"] = normalization_digest
        payload["normalization_contract"] = normalization_contract
    _validate_screen_dispersion_normalization(
        payload, normalization_velocity_mean, normalization_velocity_std
    )
    _atomic_torch_save(Path(path), payload)


def validate_query_checkpoint_normalization(
    payload: object,
    expected_normalization_stats_sha256: str,
    expected_normalization_contract: str = AIS_NORMALIZATION_CONTRACT,
    expected_normalization_velocity_mean: float | None = None,
    expected_normalization_velocity_std: float | None = None,
) -> None:
    """Validate a formal v2 checkpoint binding without mutating runtime state."""
    if not isinstance(payload, Mapping):
        raise ValueError("query checkpoint must contain a mapping payload")
    if payload.get("schema_version") not in {
        NORMALIZATION_SCHEMA_VERSION,
        SCREEN_SCHEMA_VERSION,
    }:
        raise ValueError("normalization-bound resume requires schema 4 or 5")
    expected_digest = _sha256(
        expected_normalization_stats_sha256,
        "expected normalization stats SHA-256",
    )
    if expected_normalization_contract != AIS_NORMALIZATION_CONTRACT:
        raise ValueError(
            f"expected normalization contract must be {AIS_NORMALIZATION_CONTRACT!r}"
        )
    if "normalization_stats_sha256" not in payload:
        raise ValueError("checkpoint normalization stats SHA-256 is missing")
    if "normalization_contract" not in payload:
        raise ValueError("checkpoint normalization contract is missing")
    checkpoint_digest = _sha256(
        payload["normalization_stats_sha256"],
        "checkpoint normalization stats SHA-256",
    )
    if checkpoint_digest != expected_digest:
        raise ValueError("checkpoint normalization stats SHA-256 mismatch")
    if payload["normalization_contract"] != expected_normalization_contract:
        raise ValueError("checkpoint normalization contract mismatch")
    _validate_screen_dispersion_normalization(
        payload,
        expected_normalization_velocity_mean,
        expected_normalization_velocity_std,
    )


def validate_query_checkpoint_preflight(
    payload: object,
    *,
    expected_split_manifest_sha256: str,
    expected_runtime_seed: int,
    expected_normalization_stats_sha256: str,
    expected_normalization_contract: str = AIS_NORMALIZATION_CONTRACT,
    expected_normalization_velocity_mean: float | None = None,
    expected_normalization_velocity_std: float | None = None,
    expected_config_sha256: str | None = None,
) -> QueryResumeMetadata:
    """Validate formal metadata before constructing mutable runtime state."""
    validate_query_checkpoint_normalization(
        payload,
        expected_normalization_stats_sha256,
        expected_normalization_contract,
        expected_normalization_velocity_mean,
        expected_normalization_velocity_std,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("query checkpoint must contain a mapping payload")
    if expected_config_sha256 is not None and _sha256(
        payload.get("config_sha256"), "checkpoint config SHA-256"
    ) != _sha256(expected_config_sha256, "expected config SHA-256"):
        raise ValueError("checkpoint config SHA-256 mismatch")
    if _sha256(
        payload.get("split_manifest_sha256"), "checkpoint split SHA-256"
    ) != _sha256(expected_split_manifest_sha256, "expected split manifest SHA-256"):
        raise ValueError("checkpoint split manifest SHA-256 mismatch")
    if "runtime_seed" not in payload:
        raise ValueError("checkpoint runtime seed is missing")
    runtime_seed = _nonnegative_int(payload["runtime_seed"], "runtime_seed")
    if runtime_seed != _nonnegative_int(expected_runtime_seed, "expected_runtime_seed"):
        raise ValueError("checkpoint runtime seed mismatch")
    required_metadata = {
        "epoch",
        "epoch_batch_cursor",
        "epoch_scene_order",
        "global_step",
    }
    missing = sorted(required_metadata - set(payload))
    if missing:
        raise ValueError(f"checkpoint resume metadata is missing: {missing}")
    return _scene_resume_metadata(
        payload["epoch"],
        payload["epoch_batch_cursor"],
        payload["epoch_scene_order"],
        payload["global_step"],
        payload.get("phase_index"),
        payload.get("phase_update"),
    )


def _validate_rng_tensor(value: object, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
    ):
        raise ValueError(f"invalid {name}")
    return value.detach().to(device="cpu").clone()


def _tensors_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensors_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tensors_to_cpu(item) for item in value]
    return value


def _validate_state_structure(expected: object, restored: object, *, path: str) -> None:
    """Validate state shape without requiring runtime values to be equal."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(restored, torch.Tensor):
            raise ValueError(f"{path} structure differs: expected a tensor")
        if expected.shape != restored.shape or expected.dtype != restored.dtype:
            raise ValueError(f"{path} structure differs: tensor metadata mismatch")
        return
    if isinstance(expected, Mapping):
        if type(restored) is not type(expected):
            raise ValueError(f"{path} structure differs: mapping type mismatch")
        if expected.keys() != restored.keys():
            raise ValueError(f"{path} structure differs: mapping keys mismatch")
        for key in expected:
            _validate_state_structure(
                expected[key], restored[key], path=f"{path}[{key!r}]"
            )
        return
    if isinstance(expected, (list, tuple)):
        if type(restored) is not type(expected):
            raise ValueError(f"{path} structure differs: sequence type mismatch")
        if len(expected) != len(restored):
            raise ValueError(f"{path} structure differs: sequence length mismatch")
        for index, (expected_item, restored_item) in enumerate(
            zip(expected, restored, strict=True)
        ):
            _validate_state_structure(
                expected_item, restored_item, path=f"{path}[{index}]"
            )
        return
    if type(restored) is not type(expected):
        raise ValueError(f"{path} structure differs: value type mismatch")


def _validate_payload(
    payload: object,
    *,
    expected_config_sha256: str,
    expected_split_manifest_sha256: str,
    scheduler: object | None,
    data_generator: torch.Generator,
    expected_runtime_seed: int | None = None,
    expected_normalization_stats_sha256: str | None = None,
    expected_normalization_contract: str | None = None,
    expected_normalization_velocity_mean: float | None = None,
    expected_normalization_velocity_std: float | None = None,
) -> tuple[Mapping[str, object], QueryResumeMetadata]:
    if not isinstance(payload, Mapping):
        raise ValueError("query checkpoint must contain a mapping payload")
    schema_version = payload.get("schema_version")
    if expected_runtime_seed is not None and "runtime_seed" not in payload:
        raise ValueError("checkpoint runtime seed is missing")
    if (
        expected_normalization_stats_sha256 is None
    ) != (expected_normalization_contract is None):
        raise ValueError(
            "expected normalization hash and contract must be provided together"
        )
    if expected_normalization_stats_sha256 is not None:
        validate_query_checkpoint_normalization(
            payload,
            expected_normalization_stats_sha256,
            expected_normalization_contract,
            expected_normalization_velocity_mean,
            expected_normalization_velocity_std,
        )
    elif schema_version in {NORMALIZATION_SCHEMA_VERSION, SCREEN_SCHEMA_VERSION}:
        raise ValueError(
            "schema-v4/v5 checkpoint requires an expected normalization binding"
        )

    expected_fields = (
        _TRAINER_PAYLOAD_FIELDS
        | _NORMALIZATION_PAYLOAD_FIELDS
        | _SCREEN_PAYLOAD_FIELDS
        if schema_version == SCREEN_SCHEMA_VERSION
        else _PAYLOAD_FIELDS
        if schema_version == SCHEMA_VERSION
        else _TRAINER_PAYLOAD_FIELDS
        if schema_version == TRAINER_SCHEMA_VERSION
        else _RUNTIME_SEED_PAYLOAD_FIELDS
        if schema_version == RUNTIME_SEED_SCHEMA_VERSION
        else (
            _PAYLOAD_FIELDS
            | _NORMALIZATION_PAYLOAD_FIELDS
            | (
                {"phase_index", "phase_update"}
                if "phase_index" in payload or "phase_update" in payload
                else set()
            )
            | ({"runtime_seed"} if "runtime_seed" in payload else set())
        )
        if schema_version == NORMALIZATION_SCHEMA_VERSION
        else _PAYLOAD_FIELDS
    )
    if set(payload) != expected_fields:
        missing = sorted(expected_fields - set(payload))
        extra = sorted(set(payload) - expected_fields)
        raise ValueError(
            f"query checkpoint schema fields differ: missing={missing}, extra={extra}"
        )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] not in (
            SCHEMA_VERSION,
            TRAINER_SCHEMA_VERSION,
            RUNTIME_SEED_SCHEMA_VERSION,
            NORMALIZATION_SCHEMA_VERSION,
            SCREEN_SCHEMA_VERSION,
        )
    ):
        raise ValueError("unsupported query checkpoint schema version")
    payload = dict(payload)
    expected_config = _sha256(expected_config_sha256, "expected config SHA-256")
    expected_split = _sha256(
        expected_split_manifest_sha256, "expected split manifest SHA-256"
    )
    if (
        _sha256(payload["config_sha256"], "checkpoint config SHA-256")
        != expected_config
    ):
        raise ValueError("checkpoint config SHA-256 mismatch")
    if (
        _sha256(payload["split_manifest_sha256"], "checkpoint split SHA-256")
        != expected_split
    ):
        raise ValueError("checkpoint split manifest SHA-256 mismatch")
    if (
        schema_version
        in (
            RUNTIME_SEED_SCHEMA_VERSION,
            NORMALIZATION_SCHEMA_VERSION,
            SCREEN_SCHEMA_VERSION,
        )
        and "runtime_seed" in payload
    ):
        runtime_seed = _nonnegative_int(payload["runtime_seed"], "runtime_seed")
        if expected_runtime_seed is not None and runtime_seed != _nonnegative_int(
            expected_runtime_seed, "expected_runtime_seed"
        ):
            raise ValueError("checkpoint runtime seed mismatch")

    metadata = _scene_resume_metadata(
        payload["epoch"],
        payload["epoch_batch_cursor"],
        payload["epoch_scene_order"],
        payload["global_step"],
        payload.get("phase_index"),
        payload.get("phase_update"),
    )
    state_names = (
        "model_state_dict",
        "optimizer_state_dict",
        "sampler_state_dict",
    )
    for name in state_names:
        if not isinstance(payload[name], Mapping):
            raise ValueError(f"invalid {name}")
    scheduler_state = payload["scheduler_state_dict"]
    if (scheduler is None) != (scheduler_state is None):
        raise ValueError(
            "checkpoint scheduler state does not match configured scheduler"
        )
    if scheduler_state is not None and not isinstance(scheduler_state, Mapping):
        raise ValueError("invalid scheduler_state_dict")
    if scheduler is not None:
        _validate_state_structure(
            scheduler.state_dict(), scheduler_state, path="scheduler state"
        )

    payload["torch_rng_state"] = _validate_rng_tensor(
        payload["torch_rng_state"], "torch RNG state"
    )
    payload["data_generator_state"] = _validate_rng_tensor(
        payload["data_generator_state"], "data generator state"
    )
    payload["sampler_state_dict"] = _tensors_to_cpu(payload["sampler_state_dict"])
    cuda_states = payload["cuda_rng_state_all"]
    if not isinstance(cuda_states, (list, tuple)):
        raise ValueError("invalid CUDA RNG states")
    payload["cuda_rng_state_all"] = [
        _validate_rng_tensor(state, "CUDA RNG state") for state in cuda_states
    ]
    cuda_device_count = torch.cuda.device_count()
    if len(payload["cuda_rng_state_all"]) != cuda_device_count:
        raise ValueError(
            "checkpoint CUDA RNG state count differs from available device count"
        )

    payload["numpy_rng_state"] = _decode_numpy_rng_state(payload["numpy_rng_state"])
    numpy_validator = np.random.RandomState()
    try:
        numpy_validator.set_state(payload["numpy_rng_state"])
    except Exception as error:
        raise ValueError("invalid NumPy RNG state") from error
    python_validator = random.Random()
    try:
        python_validator.setstate(payload["python_rng_state"])
    except Exception as error:
        raise ValueError("invalid Python RNG state") from error
    generator_validator = torch.Generator(device=data_generator.device)
    try:
        generator_validator.set_state(payload["data_generator_state"])
    except Exception as error:
        raise ValueError("invalid data generator state") from error
    return payload, metadata


def _validate_model_state(model: Any, state: Mapping[str, object]) -> None:
    current = model.state_dict()
    if current.keys() != state.keys():
        raise ValueError("checkpoint model state keys differ from configured model")
    for key, target in current.items():
        source = state[key]
        if not isinstance(source, torch.Tensor) or source.shape != target.shape:
            raise ValueError(f"checkpoint model tensor {key!r} has incompatible shape")
        if source.dtype != target.dtype:
            raise ValueError(f"checkpoint model tensor {key!r} has incompatible dtype")


def _validate_optimizer_hyperparameter(
    expected: object, restored: object, *, path: str
) -> None:
    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(restored, torch.Tensor)
            or restored.shape != expected.shape
            or restored.dtype != expected.dtype
        ):
            raise ValueError(f"{path} has incompatible optimizer tensor metadata")
        if (restored.is_floating_point() or restored.is_complex()) and not bool(
            torch.isfinite(restored).all()
        ):
            raise ValueError(f"{path} contains a non-finite optimizer value")
        return
    if isinstance(expected, Mapping):
        if not isinstance(restored, Mapping) or expected.keys() != restored.keys():
            raise ValueError(f"{path} has incompatible optimizer mapping keys")
        for key in expected:
            _validate_optimizer_hyperparameter(
                expected[key], restored[key], path=f"{path}[{key!r}]"
            )
        return
    if isinstance(expected, (list, tuple)):
        if type(restored) is not type(expected) or len(restored) != len(expected):
            raise ValueError(f"{path} has incompatible optimizer sequence structure")
        for index, (expected_item, restored_item) in enumerate(
            zip(expected, restored, strict=True)
        ):
            _validate_optimizer_hyperparameter(
                expected_item, restored_item, path=f"{path}[{index}]"
            )
        return
    if isinstance(expected, bool):
        if not isinstance(restored, bool):
            raise ValueError(f"{path} has incompatible optimizer boolean type")
        return
    if isinstance(expected, numbers.Real):
        if isinstance(restored, bool) or not isinstance(restored, numbers.Real):
            raise ValueError(f"{path} has incompatible optimizer numeric type")
        if not math.isfinite(float(restored)):
            raise ValueError(f"{path} contains a non-finite optimizer value")
        return
    if isinstance(expected, numbers.Complex):
        if isinstance(restored, bool) or not isinstance(restored, numbers.Complex):
            raise ValueError(f"{path} has incompatible optimizer numeric type")
        restored_complex = complex(restored)
        if not (
            math.isfinite(restored_complex.real)
            and math.isfinite(restored_complex.imag)
        ):
            raise ValueError(f"{path} contains a non-finite optimizer value")
        return
    if type(restored) is not type(expected):
        raise ValueError(f"{path} has incompatible optimizer value type")


def _validate_optimizer_parameter_state(
    value: object, parameter: torch.Tensor, *, path: str
) -> None:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return
        if value.shape != parameter.shape:
            raise ValueError(f"{path} has incompatible optimizer state shape")
        if (value.is_floating_point() or value.is_complex()) and (
            value.dtype != parameter.dtype
        ):
            raise ValueError(f"{path} has incompatible optimizer state dtype")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_optimizer_parameter_state(
                item, parameter, path=f"{path}[{key!r}]"
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_optimizer_parameter_state(
                item, parameter, path=f"{path}[{index}]"
            )
        return
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, numbers.Number):
        if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
            raise ValueError(f"{path} contains a non-finite optimizer state value")
        return
    raise ValueError(f"{path} contains an unsupported optimizer state value")


def _validate_optimizer_state(optimizer: Any, restored: Mapping[str, object]) -> None:
    if set(restored) != {"state", "param_groups"}:
        raise ValueError("optimizer state has incompatible top-level keys")
    restored_state = restored["state"]
    restored_groups = restored["param_groups"]
    current_serialized = optimizer.state_dict()
    current_groups = current_serialized.get("param_groups")
    live_groups = optimizer.param_groups
    if not isinstance(restored_state, Mapping) or not isinstance(restored_groups, list):
        raise ValueError("optimizer state containers are invalid")
    if (
        not isinstance(current_groups, list)
        or len(restored_groups) != len(current_groups)
        or len(restored_groups) != len(live_groups)
    ):
        raise ValueError("optimizer param group count differs")

    parameter_by_saved_id: dict[int, torch.Tensor] = {}
    current_parameter_ids: set[int] = set()
    for group_index, (restored_group, current_group, live_group) in enumerate(
        zip(restored_groups, current_groups, live_groups, strict=True)
    ):
        if not isinstance(restored_group, Mapping):
            raise ValueError("optimizer param group must be a mapping")
        if restored_group.keys() != current_group.keys():
            raise ValueError("optimizer param group keys differ")
        restored_ids = restored_group.get("params")
        current_ids = current_group.get("params")
        live_parameters = live_group.get("params")
        if not all(
            isinstance(items, list)
            for items in (restored_ids, current_ids, live_parameters)
        ):
            raise ValueError("optimizer params entries must be lists")
        if len(restored_ids) != len(current_ids) or len(restored_ids) != len(
            live_parameters
        ):
            raise ValueError("optimizer params length differs")
        normalized_restored_ids = [
            _nonnegative_int(value, "optimizer restored parameter ID")
            for value in restored_ids
        ]
        normalized_current_ids = [
            _nonnegative_int(value, "optimizer current parameter ID")
            for value in current_ids
        ]
        if len(set(normalized_restored_ids)) != len(normalized_restored_ids):
            raise ValueError("optimizer restored parameter IDs must be unique")
        if len(set(normalized_current_ids)) != len(normalized_current_ids):
            raise ValueError("optimizer current parameter IDs must be unique")
        if any(value in parameter_by_saved_id for value in normalized_restored_ids):
            raise ValueError("optimizer restored parameter IDs repeat across groups")
        if any(value in current_parameter_ids for value in normalized_current_ids):
            raise ValueError("optimizer current parameter IDs repeat across groups")
        if normalized_restored_ids != normalized_current_ids:
            raise ValueError(
                "optimizer parameter ID values or order differ from current optimizer"
            )
        current_parameter_ids.update(normalized_current_ids)
        for key in restored_group:
            if key != "params":
                _validate_optimizer_hyperparameter(
                    current_group[key],
                    restored_group[key],
                    path=f"optimizer param_groups[{group_index}][{key!r}]",
                )
        for normalized_id, parameter in zip(
            normalized_restored_ids, live_parameters, strict=True
        ):
            if not isinstance(parameter, torch.Tensor):
                raise ValueError("optimizer live parameter is not a tensor")
            parameter_by_saved_id[normalized_id] = parameter

    for raw_parameter_id, parameter_state in restored_state.items():
        parameter_id = _nonnegative_int(
            raw_parameter_id, "optimizer state parameter ID"
        )
        if parameter_id not in parameter_by_saved_id:
            raise ValueError("optimizer state references an unknown parameter ID")
        if not isinstance(parameter_state, Mapping):
            raise ValueError("optimizer per-parameter state must be a mapping")
        _validate_optimizer_parameter_state(
            parameter_state,
            parameter_by_saved_id[parameter_id],
            path=f"optimizer state[{parameter_id}]",
        )


def load_query_checkpoint(
    path: str | Path | None = None,
    *,
    payload: object | None = None,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    sampler: Any,
    data_generator: torch.Generator,
    expected_config_sha256: str,
    expected_split_manifest_sha256: str,
    expected_runtime_seed: int | None = None,
    expected_normalization_stats_sha256: str | None = None,
    expected_normalization_contract: str | None = None,
    expected_normalization_velocity_mean: float | None = None,
    expected_normalization_velocity_std: float | None = None,
    map_location: str | torch.device = "cpu",
) -> QueryResumeMetadata:
    """Restore all state needed to consume the saved cursor before the next draw.

    Exact optimizer validation is intentionally scoped to SGD, Adam, and AdamW.
    Rollback is best effort if application fails. If a rollback action also fails,
    the original exception is preserved and annotated; affected objects must then
    be treated as unusable and reconstructed by the caller.
    """
    _validate_supported_optimizer(optimizer)
    if (path is None) == (payload is None):
        raise ValueError("provide exactly one of checkpoint path or preloaded payload")
    if payload is None:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    payload, metadata = _validate_payload(
        payload,
        expected_config_sha256=expected_config_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        scheduler=scheduler,
        data_generator=data_generator,
        expected_runtime_seed=expected_runtime_seed,
        expected_normalization_stats_sha256=expected_normalization_stats_sha256,
        expected_normalization_contract=expected_normalization_contract,
        expected_normalization_velocity_mean=expected_normalization_velocity_mean,
        expected_normalization_velocity_std=expected_normalization_velocity_std,
    )
    _validate_model_state(model, payload["model_state_dict"])
    _validate_optimizer_state(optimizer, payload["optimizer_state_dict"])

    component_snapshots = {
        "model": _tensors_to_cpu(model.state_dict()),
        "optimizer": _tensors_to_cpu(optimizer.state_dict()),
        "scheduler": _tensors_to_cpu(scheduler.state_dict())
        if scheduler is not None
        else None,
        "sampler": _tensors_to_cpu(sampler.state_dict()),
        "data_generator": data_generator.get_state().detach().cpu().clone(),
        "torch_rng": torch.get_rng_state().detach().cpu().clone(),
        "cuda_rng": [
            state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()
        ]
        if torch.cuda.device_count() > 0
        else [],
        "numpy_rng": _decode_numpy_rng_state(
            _encode_numpy_rng_state(np.random.get_state())
        ),
        "python_rng": random.getstate(),
    }
    try:
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        sampler.load_state_dict(payload["sampler_state_dict"])
        data_generator.set_state(payload["data_generator_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        cuda_states = payload["cuda_rng_state_all"]
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
    except BaseException as original_error:
        rollback_actions = [
            ("model", lambda: model.load_state_dict(component_snapshots["model"])),
            (
                "optimizer",
                lambda: optimizer.load_state_dict(component_snapshots["optimizer"]),
            ),
        ]
        if scheduler is not None:
            rollback_actions.append(
                (
                    "scheduler",
                    lambda: scheduler.load_state_dict(component_snapshots["scheduler"]),
                )
            )
        rollback_actions.extend(
            [
                (
                    "sampler",
                    lambda: sampler.load_state_dict(component_snapshots["sampler"]),
                ),
                (
                    "data generator RNG",
                    lambda: data_generator.set_state(
                        component_snapshots["data_generator"]
                    ),
                ),
                (
                    "torch RNG",
                    lambda: torch.set_rng_state(component_snapshots["torch_rng"]),
                ),
            ]
        )
        if component_snapshots["cuda_rng"]:
            rollback_actions.append(
                (
                    "CUDA RNG",
                    lambda: torch.cuda.set_rng_state_all(
                        component_snapshots["cuda_rng"]
                    ),
                )
            )
        rollback_actions.extend(
            [
                (
                    "NumPy RNG",
                    lambda: np.random.set_state(component_snapshots["numpy_rng"]),
                ),
                (
                    "Python RNG",
                    lambda: random.setstate(component_snapshots["python_rng"]),
                ),
            ]
        )
        rollback_failures: list[str] = []
        for name, action in rollback_actions:
            try:
                action()
            except BaseException as rollback_error:
                rollback_failures.append(
                    f"{name}: {type(rollback_error).__name__}: {rollback_error}"
                )
        if rollback_failures:
            summary = "checkpoint rollback failures: " + "; ".join(rollback_failures)
            add_note = getattr(original_error, "add_note", None)
            if callable(add_note):
                add_note(summary)
            try:
                original_error.rollback_failures = tuple(rollback_failures)
            except BaseException:
                pass
        raise
    return metadata


def _positive_training_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def normalize_physical_xz(
    physical_xz: torch.Tensor, x_m: torch.Tensor, z_m: torch.Tensor
) -> torch.Tensor:
    """Map physical coordinates to the model's normalized x-z query domain."""
    if not isinstance(physical_xz, torch.Tensor) or physical_xz.shape[-1:] != (2,):
        raise ValueError("physical_xz must end with two coordinates")
    if x_m.ndim != 1 or z_m.ndim != 1 or x_m.numel() < 2 or z_m.numel() < 2:
        raise ValueError("x_m and z_m must be one-dimensional coordinate vectors")
    x_span, z_span = x_m[-1] - x_m[0], z_m[-1] - z_m[0]
    if not bool(x_span > 0) or not bool(z_span > 0):
        raise ValueError("coordinate vectors must have positive extent")
    x = (physical_xz[..., 0] - x_m[0]) / x_span
    z = (physical_xz[..., 1] - z_m[0]) / z_span
    return torch.stack((x, z), dim=-1)


def build_scene_model_inputs(
    scene: QueryScene,
    global_size: int,
    device: torch.device,
    normalization: AISNormalizationBinding | None = None,
) -> SceneModelInputs:
    """Build only coarse full-time inputs and native static channels on device."""
    global_size = _positive_training_int(global_size, "global_size")
    if scene.time_s.shape != (160,):
        raise ValueError("query training requires exactly 160 saved times")
    if normalization is None:
        velocity = scene.velocity_cpu[None, None]
        source = scene.source_cpu[None, None]
        if min(velocity.shape[-2:]) < 3:
            raise ValueError("scene must be at least 3 by 3 for physical gradients")
        grad_x, grad_z = torch.gradient(
            velocity,
            spacing=(scene.x_m.float(), scene.z_m.float()),
            dim=(2, 3),
            edge_order=2,
        )
        slow2 = velocity.clamp_min(1.0).reciprocal().square()
        native = torch.cat((velocity, source, grad_x, grad_z, slow2), dim=1)
    else:
        native = build_normalized_static_features(scene, normalization)
    static = F.interpolate(
        native,
        size=(global_size, global_size),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    ).permute(0, 2, 3, 1)
    duration = scene.time_s[-1] - scene.time_s[0]
    time_condition = (
        torch.isfinite(scene.time_s).all()
        & torch.all(torch.diff(scene.time_s) > 0)
        & torch.isfinite(duration)
        & (duration > 0)
    )
    if not bool(time_condition):
        raise ValueError(
            "time coordinates must be finite, strictly increasing, and have "
            "positive duration"
        )
    tau = ((scene.time_s - scene.time_s[0]) / duration).float()
    repeated = static[:, :, :, None].expand(-1, -1, -1, 160, -1)
    global_inputs = torch.cat(
        (
            repeated,
            tau[None, None, None, :, None].expand(
                1, global_size, global_size, -1, -1
            ),
        ),
        dim=-1,
    )
    def uniform_spacing(coordinates: torch.Tensor, name: str) -> float:
        if (
            not isinstance(coordinates, torch.Tensor)
            or coordinates.ndim != 1
            or coordinates.numel() < 2
            or not coordinates.is_floating_point()
            or not bool(torch.isfinite(coordinates).all())
        ):
            raise ValueError(f"{name} must contain finite floating coordinates")
        differences = torch.diff(coordinates)
        if not bool((differences > 0).all()) or not torch.allclose(
            differences,
            differences[0].expand_as(differences),
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError(f"{name} must be uniformly spaced and increasing")
        return float(differences[0])

    time_grid = _make_validated_time_grid(scene.time_s.to(device=device))
    return SceneModelInputs(
        global_inputs.to(device=device),
        native.to(device=device),
        time_grid.time_s,
        uniform_spacing(scene.x_m, "x_m"),
        uniform_spacing(scene.z_m, "z_m"),
        time_grid,
    )


def _encode_model_global(
    model: Any, model_inputs: SceneModelInputs
) -> torch.Tensor:
    if getattr(model, "supports_validated_time_grid", False):
        return model._encode_global_validated(
            model_inputs.global_inputs, model_inputs.validated_time_grid
        )
    return model.encode_global(model_inputs.global_inputs, model_inputs.time_s)


def _decode_model_queries(
    model: Any,
    context: torch.Tensor,
    model_inputs: SceneModelInputs,
    query_xz: torch.Tensor,
) -> torch.Tensor:
    kwargs = (
        {"dx_m": model_inputs.dx_m, "dz_m": model_inputs.dz_m}
        if getattr(model, "supports_physical_spacing", False)
        else {}
    )
    if getattr(model, "supports_validated_time_grid", False):
        return model._decode_queries_validated(
            context,
            model_inputs.native_static,
            query_xz,
            model_inputs.validated_time_grid,
            **kwargs,
            spacing_host_validated=True,
        )
    return model.decode_queries(
        context, model_inputs.native_static, query_xz, model_inputs.time_s, **kwargs
    )


def receiver_site_indices(scene: QueryScene, receiver_config: Mapping[str, object]) -> torch.Tensor:
    required = {"x_start_m", "x_stop_m", "x_stride_m", "z_m"}
    if not required.issubset(receiver_config):
        raise ValueError("receiver config is missing required geometry")
    def coordinate(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"receiver {name} must be a real number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"receiver {name} must be finite")
        return result

    stride = coordinate(receiver_config["x_stride_m"], "x_stride_m")
    if stride <= 0:
        raise ValueError("receiver x_stride_m must be positive")
    start = coordinate(receiver_config["x_start_m"], "x_start_m")
    stop = coordinate(receiver_config["x_stop_m"], "x_stop_m")
    if start > stop:
        raise ValueError("receiver x_start_m must not exceed x_stop_m")
    if start < float(scene.x_m[0]) or stop > float(scene.x_m[-1]):
        raise ValueError("receiver x coordinates must lie inside the physical domain")
    ratio = (stop - start) / stride
    ratio_tolerance = 1e-12 * max(1.0, abs(ratio))
    nearest = round(ratio)
    step_count = nearest if abs(ratio - nearest) <= ratio_tolerance else math.floor(ratio)
    requested_x = start + stride * torch.arange(step_count + 1, dtype=torch.float64)
    coordinate_tolerance = 1e-12 * max(1.0, abs(start), abs(stop))
    requested_x = torch.where(
        (requested_x > stop) & (requested_x - stop <= coordinate_tolerance),
        torch.full_like(requested_x, stop),
        requested_x,
    )
    requested_x = requested_x[(requested_x >= start) & (requested_x <= stop)]
    indices: list[int] = []
    width = scene.target_cpu.shape[1]
    z_values = receiver_config["z_m"]
    if isinstance(z_values, (str, bytes)) or not isinstance(z_values, Sequence) or not z_values:
        raise ValueError("receiver z_m must be a nonempty sequence")
    for z_value in z_values:
        requested_z = coordinate(z_value, "z_m")
        if requested_z < float(scene.z_m[0]) or requested_z > float(scene.z_m[-1]):
            raise ValueError("receiver z coordinates must lie inside the physical domain")
        z_index = int(torch.argmin((scene.z_m - requested_z).abs()))
        for x_value in requested_x:
            x_index = int(torch.argmin((scene.x_m - x_value).abs()))
            indices.append(x_index * width + z_index)
    result = torch.tensor(indices, dtype=torch.long)
    if result.numel() == 0 or torch.unique(result).numel() != result.numel():
        raise ValueError("receiver geometry aliases on this stage grid")
    return result


def build_spatial_sampling_features(
    scene: QueryScene, receiver_indices: torch.Tensor
) -> SpatialSamplingFeatures:
    velocity = scene.velocity_cpu
    grad_x, grad_z = torch.gradient(
        velocity,
        spacing=(scene.x_m.float(), scene.z_m.float()),
        dim=(0, 1),
        edge_order=2,
    )
    interface = torch.sqrt(grad_x.square() + grad_z.square()).reshape(-1)
    maxima = torch.nonzero(
        scene.source_cpu == scene.source_cpu.max(), as_tuple=False
    ).float()
    if maxima.numel() == 0:
        raise ValueError("source map has no maximum")
    source_xy = maxima.mean(0)
    x, z = torch.meshgrid(
        torch.arange(velocity.shape[0]),
        torch.arange(velocity.shape[1]),
        indexing="ij",
    )
    distance = torch.sqrt((x - source_xy[0]).square() + (z - source_xy[1]).square())
    source_wavefront = (1.0 / (1.0 + distance)).reshape(-1)
    edge_distance = torch.minimum(
        torch.minimum(x, z),
        torch.minimum(velocity.shape[0] - 1 - x, velocity.shape[1] - 1 - z),
    )
    edge = (1.0 / (1.0 + edge_distance)).reshape(-1)
    receiver = torch.full((velocity.numel(),), 1e-6)
    receiver[receiver_indices] = 1.0
    return SpatialSamplingFeatures(
        torch.ones(velocity.numel()),
        interface.clamp_min(1e-6),
        source_wavefront.clamp_min(1e-6),
        edge.clamp_min(1e-6),
        receiver,
    )


def gather_receiver_targets(
    scene: QueryScene, receiver_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sites = DenseCPUQueryStore.gather_loaded_scene(scene, receiver_indices)
    query_xz = normalize_physical_xz(sites.physical_xz, scene.x_m, scene.z_m)[None]
    return query_xz, sites.target_cpu[None]


def valid_halo_centers(height: int, width: int, halo_size: int) -> torch.Tensor:
    height = _positive_training_int(height, "height")
    width = _positive_training_int(width, "width")
    halo_size = _positive_training_int(halo_size, "halo_size")
    if halo_size < 3 or halo_size % 2 != 1 or height < halo_size or width < halo_size:
        raise ValueError("halo must be odd, at least 3, and fit the grid")
    radius = halo_size // 2
    x, z = torch.meshgrid(
        torch.arange(radius, height - radius),
        torch.arange(radius, width - radius),
        indexing="ij",
    )
    return (x * width + z).reshape(-1)


def centered_site_offsets(
    patch_centers: torch.Tensor, halo_size: int, grid_shape: tuple[int, int]
) -> torch.Tensor:
    height, width = grid_shape
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if not isinstance(patch_centers, torch.Tensor):
        raise TypeError("patch centers must be a tensor")
    if patch_centers.dtype not in integer_dtypes:
        raise TypeError("patch centers must have an integer dtype and cannot be bool")
    centers = patch_centers.to(device="cpu", dtype=torch.long).reshape(-1)
    if centers.numel() == 0:
        raise ValueError("patch centers must be nonempty")
    if bool(torch.any(centers < 0)) or bool(torch.any(centers >= height * width)):
        raise ValueError("patch centers are outside grid bounds")
    allowed = torch.zeros(height * width, dtype=torch.bool)
    allowed[valid_halo_centers(height, width, halo_size)] = True
    if not bool(allowed[centers].all()):
        raise ValueError("every halo center must admit a complete unclamped patch")
    radius = halo_size // 2
    cx = torch.div(centers, width, rounding_mode="floor")
    cz = centers.remainder(width)
    dx, dz = torch.meshgrid(
        torch.arange(-radius, radius + 1),
        torch.arange(-radius, radius + 1),
        indexing="ij",
    )
    return (cx[:, None, None] + dx) * width + (cz[:, None, None] + dz)


def decode_halo_patches(
    model: Any,
    model_inputs: SceneModelInputs,
    scene: QueryScene,
    patch_centers: torch.Tensor,
    halo_size: int,
    context: torch.Tensor | None = None,
    normalization: AISNormalizationBinding | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = centered_site_offsets(patch_centers, halo_size, scene.target_cpu.shape[:2])
    sites = offsets.reshape(-1)
    width = scene.target_cpu.shape[1]
    x_idx = torch.div(sites, width, rounding_mode="floor")
    z_idx = sites.remainder(width)
    physical = torch.stack((scene.x_m[x_idx], scene.z_m[z_idx]), dim=-1)
    query_xz = normalize_physical_xz(physical, scene.x_m, scene.z_m)[None].float()
    if context is None:
        context = _encode_model_global(model, model_inputs)
    prediction = _decode_model_queries(
        model,
        context,
        model_inputs,
        query_xz.to(model_inputs.native_static.device),
    )
    if normalization is not None:
        prediction = normalization.decode_wavefield(prediction)
    target = scene.target_cpu.reshape(-1, 160).index_select(0, sites).reshape(
        1, len(patch_centers), halo_size, halo_size, 160
    )
    return prediction.reshape_as(target.to(prediction.device)), target.to(prediction.device)


def query_field_loss_chunked(
    model: Any,
    model_inputs: SceneModelInputs,
    query_xz: torch.Tensor,
    target_cpu: torch.Tensor,
    draw_probability: torch.Tensor,
    population_size: int,
    chunk_size: int,
    hh_reweight: bool,
    late_time_weights: torch.Tensor | None = None,
    normalization: AISNormalizationBinding | None = None,
) -> tuple[torch.Tensor, HHFieldLoss]:
    """Decode chunks, then form one globally normalized Q-site objective."""
    chunk_size = _positive_training_int(chunk_size, "query_chunk_size")
    if query_xz.ndim != 3 or query_xz.shape[1] == 0 or query_xz.shape[-1] != 2:
        raise ValueError("query_xz must contain at least one query")
    if target_cpu.shape != (*query_xz.shape[:2], 160):
        raise ValueError("targets must contain one complete 160-step trace per query")
    context = _encode_model_global(model, model_inputs)
    total, field, _ = _query_field_loss_with_context(
        model, context, model_inputs, query_xz, target_cpu, draw_probability,
        population_size, chunk_size, hh_reweight, late_time_weights, normalization,
    )
    return total, field


def _field_loss_for_normalization(
    prediction: torch.Tensor,
    target: torch.Tensor,
    draw_probability: torch.Tensor,
    population_size: int,
    late_time_weights: torch.Tensor | None,
    normalization: AISNormalizationBinding | None,
) -> HHFieldLoss:
    if normalization is None:
        return hansen_hurwitz_field_loss(
            prediction,
            target,
            draw_probability,
            population_size,
            late_time_weights=late_time_weights,
        )
    return standardized_hh_field_loss(
        prediction,
        target,
        draw_probability,
        population_size,
        normalization,
        late_time_weights=late_time_weights,
    )


def _query_field_loss_with_context(
    model: Any,
    context: torch.Tensor,
    model_inputs: SceneModelInputs,
    query_xz: torch.Tensor,
    target_cpu: torch.Tensor,
    draw_probability: torch.Tensor,
    population_size: int,
    chunk_size: int,
    hh_reweight: bool,
    late_time_weights: torch.Tensor | None,
    normalization: AISNormalizationBinding | None = None,
) -> tuple[torch.Tensor, HHFieldLoss, torch.Tensor]:
    chunk_size = _positive_training_int(chunk_size, "query_chunk_size")
    if query_xz.ndim != 3 or query_xz.shape[1] == 0 or query_xz.shape[-1] != 2:
        raise ValueError("query_xz must contain at least one query")
    if target_cpu.shape != (*query_xz.shape[:2], 160):
        raise ValueError("targets must contain one complete 160-step trace per query")
    predictions: list[torch.Tensor] = []
    device = model_inputs.native_static.device
    for start in range(0, query_xz.shape[1], chunk_size):
        end = min(start + chunk_size, query_xz.shape[1])
        predictions.append(
            _decode_model_queries(
                model,
                context,
                model_inputs,
                query_xz[:, start:end].to(device=device),
            )
        )
    prediction = torch.cat(predictions, dim=1)
    target_physical = target_cpu.to(device=device, non_blocking=True)
    target = (
        normalization.encode_wavefield(target_physical)
        if normalization is not None
        else target_physical
    )
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError("nonfinite prediction or target in query field loss")
    if hh_reweight:
        field = _field_loss_for_normalization(
            prediction,
            target,
            draw_probability.to(device=device),
            population_size,
            late_time_weights,
            normalization,
        )
    else:
        uniform_q = torch.full(
            prediction.shape[:2], 1.0 / prediction.shape[1],
            device=device, dtype=prediction.dtype,
        )
        field = _field_loss_for_normalization(
            prediction,
            target,
            uniform_q,
            prediction.shape[1],
            late_time_weights,
            normalization,
        )
    return field.loss, field, prediction


def _memory_bounded_field_backward(
    model: Any,
    context: torch.Tensor,
    model_inputs: SceneModelInputs,
    query_xz: torch.Tensor,
    target_cpu: torch.Tensor,
    draw_probability: torch.Tensor,
    population_size: int,
    chunk_size: int,
    hh_reweight: bool,
    late_time_weights: torch.Tensor | None,
    normalization: AISNormalizationBinding | None = None,
) -> tuple[HHFieldLoss, torch.Tensor, torch.Tensor]:
    """Backpropagate the exact global field loss without retaining every decode graph.

    The first pass stores only the small ``[B,Q,160]`` output.  Autograd then
    differentiates the globally normalized objective with respect to that output.
    A second, chunked pass applies those exact output cotangents immediately, so
    decoder activations are bounded by ``chunk_size``.  The returned context leaf
    accumulates the gradient that must subsequently be sent through the encoder.
    """
    device = model_inputs.native_static.device
    detached_context = context.detach()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, query_xz.shape[1], chunk_size):
            end = min(start + chunk_size, query_xz.shape[1])
            predictions.append(
                _decode_model_queries(
                    model,
                    detached_context,
                    model_inputs,
                    query_xz[:, start:end].to(device=device),
                )
            )
    sampled_prediction = torch.cat(predictions, dim=1)
    target_physical = target_cpu.to(device=device, non_blocking=True)
    target = (
        normalization.encode_wavefield(target_physical)
        if normalization is not None
        else target_physical
    )
    if not bool(torch.isfinite(sampled_prediction).all()) or not bool(
        torch.isfinite(target).all()
    ):
        raise FloatingPointError("nonfinite prediction or target in query field loss")

    proxy_prediction = sampled_prediction.detach().requires_grad_(True)
    if hh_reweight:
        field = _field_loss_for_normalization(
            proxy_prediction,
            target,
            draw_probability.to(device=device),
            population_size,
            late_time_weights,
            normalization,
        )
    else:
        uniform_q = torch.full(
            proxy_prediction.shape[:2],
            1.0 / proxy_prediction.shape[1],
            device=device,
            dtype=proxy_prediction.dtype,
        )
        field = _field_loss_for_normalization(
            proxy_prediction,
            target,
            uniform_q,
            proxy_prediction.shape[1],
            late_time_weights,
            normalization,
        )
    output_gradient = torch.autograd.grad(field.loss, proxy_prediction)[0]
    context_leaf = detached_context.requires_grad_(True)
    for start in range(0, query_xz.shape[1], chunk_size):
        end = min(start + chunk_size, query_xz.shape[1])
        prediction_chunk = _decode_model_queries(
            model,
            context_leaf,
            model_inputs,
            query_xz[:, start:end].to(device=device),
        )
        torch.autograd.backward(
            prediction_chunk,
            grad_tensors=output_gradient[:, start:end],
        )
    return field, sampled_prediction, context_leaf


def _late_time_weights(loss_config: Mapping[str, object]) -> torch.Tensor | None:
    raw = loss_config.get("late_time_weights")
    if raw is None:
        return None
    try:
        weights = torch.as_tensor(raw, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "late_time_weights must contain 4 quarter weights or 160 positive finite weights"
        ) from error
    if weights.shape == (4,):
        weights = weights.repeat_interleave(40)
    if weights.shape != (160,) or not bool(torch.isfinite(weights).all()) or bool(torch.any(weights <= 0)):
        raise ValueError("late_time_weights must contain 4 quarter weights or 160 positive finite weights")
    return weights


def _validate_epoch_config(config: Mapping[str, object]) -> tuple[int, int]:
    try:
        train = config["train"]
        loss = config["loss"]
        sampling = config["sampling"]
    except KeyError as error:
        raise ValueError(f"training config is missing {error.args[0]}") from error
    if not all(isinstance(value, Mapping) for value in (train, loss, sampling)):
        raise TypeError("train, loss, and sampling configs must be mappings")
    query_count = _positive_training_int(train.get("query_sites_per_scene"), "query_sites_per_scene")
    chunk_size = _positive_training_int(train.get("query_chunk_size"), "query_chunk_size")
    if train.get("amp", False) is not False:
        raise ValueError("AIS query training currently requires amp: false")
    grad_clip = train.get("grad_clip")
    if isinstance(grad_clip, bool) or not isinstance(grad_clip, numbers.Real):
        raise TypeError("grad_clip must be a positive finite real number")
    if not math.isfinite(float(grad_clip)) or float(grad_clip) <= 0:
        raise ValueError("grad_clip must be a positive finite real number")
    _late_time_weights(loss)
    for key in ("receiver_weight", "phase_weight", "local_spectrum_weight", "energy_weight"):
        value = float(loss.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"loss {key} must be finite and nonnegative")
    _positive_training_int(sampling.get("target_height"), "target_height")
    _positive_training_int(sampling.get("target_width"), "target_width")
    _positive_training_int(sampling.get("global_size"), "global_size")
    return query_count, chunk_size


def train_query_epoch(
    model: Any,
    store: Any,
    train_sample_ids: Sequence[int],
    sampler: Any,
    optimizer: Any,
    config: Mapping[str, object],
    device: torch.device,
    epoch: int,
    global_step: int,
    epoch_batch_cursor: int = 0,
    max_scene_draws: int | None = None,
    normalization: AISNormalizationBinding | None = None,
) -> TrainEpochResult:
    _nonnegative_int(epoch, "epoch")
    query_count, chunk_size = _validate_epoch_config(config)
    cursor = _nonnegative_int(epoch_batch_cursor, "epoch_batch_cursor")
    if cursor > len(train_sample_ids):
        raise ValueError("epoch_batch_cursor exceeds scene order")
    if max_scene_draws is not None:
        max_scene_draws = _positive_training_int(max_scene_draws, "max_scene_draws")
    totals = {name: 0.0 for name in ("field", "receiver", "phase", "local_spectrum", "energy")}
    diagnostics: dict[str, float] = {}
    draws = label_sites = updates = 0
    stop = len(train_sample_ids)
    if max_scene_draws is not None:
        stop = min(stop, cursor + max_scene_draws)
    model.train()
    for scene_cursor in range(cursor, stop):
        sample_id = int(train_sample_ids[scene_cursor])
        scene = resize_query_scene(
            store.read_scene(sample_id),
            int(config["sampling"]["target_height"]),
            int(config["sampling"]["target_width"]),
        )
        height, width = scene.target_cpu.shape[:2]
        inputs = build_scene_model_inputs(
            scene,
            int(config["sampling"]["global_size"]),
            device,
            normalization=normalization,
        )
        receivers = receiver_site_indices(scene, config["receiver"])
        features = build_spatial_sampling_features(scene, receivers)
        draw = sampler.draw([sample_id], [features], count=query_count)
        sites = DenseCPUQueryStore.gather_loaded_scene(scene, draw.site_indices[0])
        query_xz = normalize_physical_xz(sites.physical_xz, scene.x_m, scene.z_m).float()[None]
        optimizer.zero_grad(set_to_none=True)
        context = _encode_model_global(model, inputs)
        field, field_prediction, context_leaf = _memory_bounded_field_backward(
            model,
            context,
            inputs,
            query_xz,
            sites.target_cpu[None],
            draw.draw_probability,
            height * width,
            chunk_size,
            bool(config["loss"].get("hh_reweight", True)),
            _late_time_weights(config["loss"]),
            normalization,
        )

        receiver_active = any(
            float(config["loss"].get(key, 0.0)) > 0
            for key in ("receiver_weight", "phase_weight", "energy_weight")
        )
        if receiver_active:
            receiver_xz, receiver_target = gather_receiver_targets(scene, receivers)
            receiver_prediction = _decode_model_queries(
                model,
                context_leaf,
                inputs,
                receiver_xz.float().to(device),
            )
            if normalization is not None:
                receiver_prediction = normalization.decode_wavefield(
                    receiver_prediction
                )
            receiver_target = receiver_target.to(device)
        else:
            receiver_prediction = torch.zeros(1, 1, 160, device=device)
            receiver_target = torch.zeros_like(receiver_prediction)

        halo = _positive_training_int(config["train"].get("patch_size", 3), "patch_size")
        spectrum_active = float(config["loss"].get("local_spectrum_weight", 0.0)) > 0
        if spectrum_active:
            allowed = valid_halo_centers(height, width, halo)
            patch_count = _positive_training_int(
                config["train"].get("patch_centers_per_scene"), "patch_centers_per_scene"
            )
            if patch_count > allowed.numel():
                raise ValueError("patch_centers_per_scene exceeds valid halo centers")
            generator = torch.Generator().manual_seed(int(config["seed"]) + sample_id)
            patch_centers = allowed[torch.randperm(len(allowed), generator=generator)[:patch_count]]
            patch_prediction, patch_target = decode_halo_patches(
                model,
                inputs,
                scene,
                patch_centers,
                halo,
                context=context_leaf,
                normalization=normalization,
            )
        else:
            patch_centers = torch.empty(0, dtype=torch.long)
            patch_prediction = torch.zeros(1, 1, halo, halo, 160, device=device)
            patch_target = torch.zeros_like(patch_prediction)
        aux = auxiliary_query_losses(
            receiver_prediction,
            receiver_target,
            patch_prediction,
            patch_target,
            inputs.time_s,
        )
        auxiliary_total = (
            float(config["loss"].get("receiver_weight", 0.0)) * aux.receiver
            + float(config["loss"].get("phase_weight", 0.0)) * aux.phase
            + float(config["loss"].get("local_spectrum_weight", 0.0)) * aux.local_spectrum
            + float(config["loss"].get("energy_weight", 0.0)) * aux.energy
        )
        total = field.loss.detach() + auxiliary_total.detach()
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"nonfinite loss for sample {sample_id}")
        if auxiliary_total.requires_grad:
            auxiliary_total.backward()
        if context_leaf.grad is None:
            raise RuntimeError("query decoders did not produce a global-context gradient")
        context.backward(context_leaf.grad)
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"nonfinite gradient for sample {sample_id}")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["train"]["grad_clip"]),
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"nonfinite gradient norm for sample {sample_id}")
        optimizer.step()
        physical_field_prediction = (
            normalization.decode_wavefield(field_prediction)
            if normalization is not None
            else field_prediction
        )
        residual = (
            physical_field_prediction.detach().cpu() - sites.target_cpu[None]
        ).square().mean(-1)
        sampler.update_residual_tiles(draw.sample_ids, draw.site_indices, residual)
        values = {
            "field": field.loss,
            "receiver": aux.receiver,
            "phase": aux.phase,
            "local_spectrum": aux.local_spectrum,
            "energy": aux.energy,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu())
        for name, value in draw.diagnostics.items():
            diagnostics[name] = diagnostics.get(name, 0.0) + float(value)
        global_step += 1
        updates += 1
        draws += 1
        label_sites += query_count + (receivers.numel() if receiver_active else 0) + patch_centers.numel() * halo * halo
    if draws == 0:
        raise ValueError("epoch contains no remaining scene draws")
    next_cursor = 0 if stop == len(train_sample_ids) else stop
    if next_cursor == 0:
        sampler.commit_epoch()
    means = {name: value / draws for name, value in totals.items()}
    weighted_total = (
        means["field"]
        + float(config["loss"].get("receiver_weight", 0.0)) * means["receiver"]
        + float(config["loss"].get("phase_weight", 0.0)) * means["phase"]
        + float(config["loss"].get("local_spectrum_weight", 0.0)) * means["local_spectrum"]
        + float(config["loss"].get("energy_weight", 0.0)) * means["energy"]
    )
    return TrainEpochResult(
        weighted_total,
        means,
        global_step,
        updates,
        next_cursor,
        draws,
        label_sites,
        {name: value / draws for name, value in diagnostics.items()},
    )


@torch.no_grad()
def validate_query_guard(
    model: Any,
    store: Any,
    guard_sample_ids: Sequence[int],
    fixed_site_manifest: Mapping[int, torch.Tensor],
    config: Mapping[str, object],
    device: torch.device,
    normalization: AISNormalizationBinding | None = None,
) -> dict[str, float]:
    _, chunk = _validate_epoch_config(config)
    model.eval()
    whole: list[float] = []
    q4: list[float] = []
    for sample_id_value in guard_sample_ids:
        sample_id = int(sample_id_value)
        if sample_id not in fixed_site_manifest:
            raise ValueError(f"validation sites missing sample {sample_id}")
        scene = resize_query_scene(
            store.read_scene(sample_id),
            int(config["sampling"]["target_height"]),
            int(config["sampling"]["target_width"]),
        )
        validation_indices = fixed_site_manifest[sample_id]
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        total_sites = scene.target_cpu.shape[0] * scene.target_cpu.shape[1]
        if (
            not isinstance(validation_indices, torch.Tensor)
            or validation_indices.ndim != 1
            or validation_indices.dtype not in integer_dtypes
            or validation_indices.numel() == 0
        ):
            raise TypeError("validation site manifest requires a nonempty integer tensor")
        indices_cpu = validation_indices.to(device="cpu", dtype=torch.long)
        if (
            bool(torch.any(indices_cpu < 0))
            or bool(torch.any(indices_cpu >= total_sites))
        ):
            raise ValueError("validation site indices are outside resized grid bounds")
        if torch.unique(indices_cpu).numel() != indices_cpu.numel():
            raise ValueError("validation site indices must be unique")
        sites = DenseCPUQueryStore.gather_loaded_scene(scene, indices_cpu)
        inputs = build_scene_model_inputs(
            scene,
            int(config["sampling"]["global_size"]),
            device,
            normalization=normalization,
        )
        context = _encode_model_global(model, inputs)
        query_xz = normalize_physical_xz(sites.physical_xz, scene.x_m, scene.z_m).float()[None]
        predictions = []
        for start in range(0, query_xz.shape[1], chunk):
            predictions.append(
                _decode_model_queries(
                    model,
                    context,
                    inputs,
                    query_xz[:, start : start + chunk].to(device),
                ).cpu()
            )
        prediction, target = torch.cat(predictions, 1), sites.target_cpu[None]
        if normalization is not None:
            prediction = normalization.decode_wavefield(prediction)
        whole.append(float(torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target).clamp_min(1e-8)))
        q4.append(float(torch.linalg.vector_norm(prediction[..., 120:] - target[..., 120:]) / torch.linalg.vector_norm(target[..., 120:]).clamp_min(1e-8)))
    if not whole:
        raise ValueError("validation guard is empty")
    return {
        "relative_l2": sum(whole) / len(whole),
        "relative_l2_q4": sum(q4) / len(q4),
        "physical_samples": float(len(whole)),
    }


def curriculum_phase_events(phases: Sequence[Mapping[str, object]]) -> list[str]:
    """Validate curriculum transitions and expose their required lifecycle events."""
    if not phases:
        raise ValueError("training requires at least one phase")
    events: list[str] = []
    for index, phase in enumerate(phases):
        name = phase.get("name")
        init_from = phase.get("init_from")
        if not isinstance(name, str) or not name:
            raise ValueError("every phase requires a nonempty name")
        if index == 0:
            if init_from not in ("random", "external_checkpoint"):
                raise ValueError("the first phase must initialize from random or external_checkpoint")
            if init_from == "external_checkpoint":
                events.extend((f"{name}:load_external_checkpoint", f"{name}:new_optimizer"))
            else:
                events.extend((f"{name}:new_optimizer", f"{name}:best"))
        else:
            if init_from == "random":
                raise ValueError("init_from=random is legal only for the first phase")
            if init_from != "phase_best":
                raise ValueError("later phases must use init_from=phase_best")
            events.extend((f"{name}:load_phase_best", f"{name}:new_optimizer"))
    return events
