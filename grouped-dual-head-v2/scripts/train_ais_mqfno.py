#!/usr/bin/env python3
"""Train AIS-MQFNO from CPU-resident dense scenes using full-trace site queries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fno_acoustic.ais_sampler import AdaptiveSpatialSampler  # noqa: E402
from fno_acoustic.ais_dataset_binding import (  # noqa: E402
    DatasetContentBinding,
    load_dataset_content_binding,
)
from fno_acoustic.ais_normalization import (  # noqa: E402
    AISNormalizationBinding,
    load_ais_normalization,
)
from fno_acoustic.model_ais_mqfno import AISMQFNO  # noqa: E402
from fno_acoustic.query_data import (  # noqa: E402
    DenseCPUQueryStore,
    resize_query_scene,
)
from fno_acoustic.query_training import (  # noqa: E402
    QUERY_CHECKPOINT_BOUNDARY,
    _decode_model_queries,
    _encode_model_global,
    build_scene_model_inputs,
    curriculum_phase_events,
    load_query_checkpoint,
    save_query_checkpoint,
    train_query_epoch,
    validate_query_checkpoint_preflight,
    validate_query_guard,
    normalize_physical_xz,
)


DEFAULT_VALIDATION_INTERVAL_UPDATES = 100


@dataclass(frozen=True)
class TrainingCheckpointSnapshot:
    path: Path
    payload: Mapping[str, object]
    sha256: str


def load_training_checkpoint_snapshot(path: Path) -> TrainingCheckpointSnapshot:
    """Read and safely deserialize one immutable top-level checkpoint snapshot."""
    raw = path.read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("training checkpoint must contain a mapping")
    return TrainingCheckpointSnapshot(
        path, MappingProxyType(payload), hashlib.sha256(raw).hexdigest()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--overfit-sample-id", type=int)
    parser.add_argument("--overfit-site-manifest", type=Path)
    parser.add_argument("--finalize-overfit-only", action="store_true")
    parser.add_argument("--finalize-training-only", action="store_true")
    parser.add_argument("--screen-gate", choices=("O", "H1", "H2", "H3"))
    parser.add_argument("--dataset-binding-sha256")
    parser.add_argument("--execution-binding-sha256")
    parser.add_argument("--dataset-content-manifest", type=Path)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _local_execution_binding_sha256(device: str) -> str:
    values: dict[str, object] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        values["gpu"] = {
            "name": properties.name, "uuid": str(getattr(properties, "uuid", "unavailable")),
            "total_memory": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    else:
        values["gpu"] = None
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_screen_resume_identity(
    payload: Mapping[str, object],
    *,
    candidate_id: str,
    current_gate: str,
    same_run: bool,
    dataset_binding_sha256: str,
    execution_binding_sha256: str,
    lineage_parent_sha256: str | None,
) -> None:
    expected_gate = (
        current_gate
        if same_run
        else {"H2": "H1", "H3": "H2"}.get(current_gate, current_gate)
    )
    if payload.get("screen_candidate_id") != candidate_id or payload.get(
        "screen_gate"
    ) != expected_gate:
        raise ValueError("resume checkpoint candidate or gate identity mismatch")
    if (
        payload.get("dataset_binding_sha256") != dataset_binding_sha256
        or payload.get("execution_binding_sha256") != execution_binding_sha256
    ):
        raise ValueError("resume checkpoint runtime binding mismatch")
    if same_run and payload.get("parent_checkpoint_sha256") != lineage_parent_sha256:
        raise ValueError("resume checkpoint parent identity mismatch")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _publish_root_checkpoint(canonical: Path, compatibility: Path) -> None:
    temporary = compatibility.with_suffix(compatibility.suffix + ".link.tmp")
    temporary.unlink(missing_ok=True)
    os.link(canonical, temporary)
    os.replace(temporary, compatibility)


def _strict_positive(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validation_schedule(
    phase_update: int,
    phase_updates: int,
    train_batches_consumed: int,
    max_train_batches: int | None,
    interval: int = DEFAULT_VALIDATION_INTERVAL_UPDATES,
) -> tuple[str | None, bool]:
    """Schedule full guards without changing any training-state transition."""
    current = _strict_positive(phase_update, "phase_update")
    limit = _strict_positive(phase_updates, "phase_updates")
    cadence = _strict_positive(interval, "validation interval")
    if current > limit:
        raise ValueError("phase_update exceeds phase_updates")
    budget_stop = (
        max_train_batches is not None
        and train_batches_consumed >= _strict_positive(
            max_train_batches, "max_train_batches"
        )
    )
    if current == limit:
        return "phase_end", True
    if current % cadence == 0:
        return "cadence", True
    if budget_stop:
        return "budget_stop", False
    return None, False


def _load_configuration(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config must contain a mapping")
    for section in ("data", "sampling", "model", "receiver", "train", "loss"):
        if not isinstance(loaded.get(section), dict):
            raise ValueError(f"config section {section!r} must be a mapping")
    _validate_configuration(loaded)
    return loaded


def _load_normalization_binding(
    config: dict[str, Any],
) -> AISNormalizationBinding:
    normalization = config.get("normalization")
    if normalization is None:
        raise ValueError("formal AIS training requires normalization v2")
    if not isinstance(normalization, dict):
        raise ValueError("config normalization section must be a mapping")
    if normalization.get("contract", "ais_normalization_v2") != "ais_normalization_v2":
        raise ValueError("normalization contract must be 'ais_normalization_v2'")
    stats_path = normalization.get("stats_path")
    if not isinstance(stats_path, (str, Path)):
        raise ValueError("normalization.stats_path is required for formal v2 training")
    return load_ais_normalization(Path(stats_path))


def _phase_updates(phase: dict[str, Any]) -> int:
    canonical = phase.get("optimizer_updates")
    legacy = phase.get("updates")
    if canonical is None and legacy is None:
        raise ValueError("phase requires optimizer_updates")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise ValueError("phase optimizer_updates and updates conflict")
    return _strict_positive(canonical if canonical is not None else legacy, "phase optimizer_updates")


def _model_kwargs(
    config: dict[str, Any],
    normalization: AISNormalizationBinding | None = None,
) -> dict[str, Any]:
    model = config["model"]
    allowed = {
        "name", "global_max_size", "local_patch_size", "halo_size",
        "global_in_features", "native_in_channels", "spatial_width", "spatial_modes",
        "temporal_modes", "local_dim", "fusion_dim", "spatial_layers",
        "activation_checkpointing", "spatial_chunk_size", "local_encoder_kind",
        "dispersion_head",
    }
    unknown = set(model) - allowed
    if unknown:
        raise ValueError(f"unknown AIS-MQFNO model fields: {sorted(unknown)}")
    if model.get("name", "ais_mqfno") != "ais_mqfno":
        raise ValueError("model.name must be ais_mqfno")
    maximum = _strict_positive(model.get("global_max_size", config["sampling"]["global_size"]), "global_max_size")
    if int(config["sampling"]["global_size"]) > maximum:
        raise ValueError("sampling.global_size exceeds model.global_max_size")
    local = model.get("local_patch_size")
    halo = model.get("halo_size")
    if local is not None and halo is not None and local != halo:
        raise ValueError("local_patch_size and halo_size conflict")
    kwargs = {key: value for key, value in model.items() if key not in {"name", "global_max_size", "local_patch_size"}}
    if local is not None:
        kwargs["halo_size"] = local
    if model.get("dispersion_head", "none") == "phase_residual_24":
        if normalization is not None:
            kwargs["velocity_mean"] = normalization.velocity_mean
            kwargs["velocity_std"] = normalization.velocity_std
    return kwargs


def _sampler_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    sampling = config["sampling"]
    sampler = config.get("sampler")
    if sampler is None:
        return {
            "mixture": sampling["mixture"],
            "tile_size": int(sampling.get("tile_size", 16)),
            "ema_momentum": float(sampling.get("ema_momentum", 0.2)),
        }
    allowed = {"mixture", "tile_grid", "ema_momentum", "lag_epochs", "residual_power", "min_probability"}
    unknown = set(sampler) - allowed
    if unknown:
        raise ValueError(f"unknown sampler fields: {sorted(unknown)}")
    for key in ("mixture", "ema_momentum"):
        if key in sampling and key in sampler and sampling[key] != sampler[key]:
            raise ValueError(f"sampling and sampler {key} conflict")
    grid = sampler.get("tile_grid")
    if not isinstance(grid, list) or len(grid) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in grid
    ):
        raise ValueError("sampler.tile_grid must contain two positive integers")
    height, width = int(sampling["target_height"]), int(sampling["target_width"])
    if height % grid[0] or width % grid[1] or height // grid[0] != width // grid[1]:
        raise ValueError("sampler tile_grid must evenly define equal-size spatial tiles")
    tile_size = height // grid[0]
    if "tile_size" in sampling and int(sampling["tile_size"]) != tile_size:
        raise ValueError("sampling.tile_size conflicts with sampler.tile_grid")
    lag = sampler.get("lag_epochs", 1)
    power = sampler.get("residual_power", 1.0)
    minimum = sampler.get("min_probability", 1e-12)
    if (
        isinstance(lag, bool)
        or isinstance(power, bool)
        or isinstance(minimum, bool)
        or lag != 1
        or power != 1.0
        or minimum != 1e-12
    ):
        raise ValueError("current sampler requires lag_epochs=1, residual_power=1, min_probability=1e-12")
    mixture = sampler.get("mixture", sampling.get("mixture"))
    if mixture is None:
        raise ValueError("sampler mixture is required")
    if isinstance(mixture, dict):
        component_order = (
            "uniform",
            "interface",
            "source_wavefront",
            "residual",
            "edge",
            "receiver",
        )
        if set(mixture) != set(component_order):
            raise ValueError(
                "named sampler mixture requires exactly uniform, interface, "
                "source_wavefront, residual, edge, and receiver"
            )
        mixture = [mixture[name] for name in component_order]
    momentum = sampler.get("ema_momentum", 0.2)
    if (
        isinstance(momentum, bool)
        or not isinstance(momentum, (int, float))
        or not np.isfinite(float(momentum))
        or not 0.0 <= float(momentum) <= 1.0
    ):
        raise ValueError("sampler ema_momentum must be finite and between zero and one")
    return {
        "mixture": mixture,
        "tile_size": tile_size,
        "ema_momentum": float(momentum),
    }


def _validate_configuration(loaded: dict[str, Any]) -> None:
    for key in ("target_height", "target_width", "global_size"):
        _strict_positive(loaded["sampling"].get(key), f"sampling.{key}")
    if loaded["sampling"].get("max_time_steps", 160) != 160:
        raise ValueError("sampling.max_time_steps must equal 160")
    if "query_chunk_size" in loaded["sampling"]:
        _strict_positive(
            loaded["sampling"]["query_chunk_size"],
            "sampling.query_chunk_size",
        )
    phases = loaded["train"].get("phases")
    if not isinstance(phases, list):
        raise ValueError("train.phases must be a list")
    curriculum_phase_events(phases)
    for phase in phases:
        _phase_updates(phase)
    if loaded["train"].get("amp", False) is not False:
        raise ValueError("AMP is disabled for the stable full160 trainer")
    _model_kwargs(loaded)
    _sampler_kwargs(loaded)


def _load_splits(path: Path) -> dict[str, list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("split manifest must be a mapping")
    result: dict[str, list[int]] = {}
    seen: set[int] = set()
    for split in ("train", "val", "test"):
        values = raw.get(split)
        if not isinstance(values, list):
            raise ValueError(f"split manifest requires list {split!r}")
        normalized: list[int] = []
        for value in values:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("split sample IDs must be nonnegative integers")
            if value in seen:
                raise ValueError("split manifest sample IDs must be disjoint")
            seen.add(value)
            normalized.append(value)
        result[split] = normalized
    if not result["train"] or not result["val"]:
        raise ValueError("train and val splits must be nonempty")
    return result


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_overfit_sites(
    config: Mapping[str, object],
    splits: Mapping[str, list[int]],
    sample_id: int,
    manifest_path: Path,
    max_train_batches: int | None,
    max_val_batches: int | None,
    starting_update: int = 0,
    finalize_only: bool = False,
) -> dict[int, torch.Tensor]:
    """Validate and load the config-bound Gate O train-only site registration."""

    if finalize_only:
        if starting_update != 400 or max_train_batches is not None:
            raise ValueError("Gate O finalize-only requires exact update 400 and no training budget")
    elif max_train_batches is None or starting_update + max_train_batches != 400:
        raise ValueError("Gate O overfit requires an exact total of 400 optimizer updates")
    if max_val_batches is not None:
        raise ValueError("Gate O overfit replaces validation; --max-val-batches is illegal")
    screen = config.get("screen")
    if not isinstance(screen, Mapping):
        raise ValueError("Gate O overfit requires a screen registration")
    registered_sample = screen.get("gate_o_sample_id")
    if sample_id != registered_sample:
        raise ValueError("overfit sample must equal the config-registered Gate O sample")
    if sample_id not in splits["train"] or sample_id in splits["val"] or sample_id in splits["test"]:
        raise ValueError("overfit sample must belong exclusively to the train split")
    registered_path = _repository_path(screen.get("gate_o_site_manifest", ""))
    supplied_path = _repository_path(manifest_path)
    if supplied_path.resolve() != registered_path.resolve():
        raise ValueError("overfit site manifest must equal the config-registered path")
    expected_hash = screen.get("gate_o_site_manifest_sha256")
    if not isinstance(expected_hash, str) or _sha256(supplied_path) != expected_hash:
        raise ValueError("overfit site manifest SHA-256 mismatch")
    payload = json.loads(supplied_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("overfit site manifest must contain a mapping")
    expected_grid = {
        "height": int(config["sampling"]["target_height"]),
        "width": int(config["sampling"]["target_width"]),
        "index_order": "row_major_x_then_z",
    }
    if (
        payload.get("schema") != "ais_fixed_spatial_sites"
        or payload.get("schema_version") != 1
        or payload.get("purpose") != "gate_o_train_representability"
        or payload.get("split") != "train"
        or payload.get("sample_id") != sample_id
        or payload.get("grid") != expected_grid
        or payload.get("split_manifest_sha256")
        != config["data"].get("split_manifest_sha256")
    ):
        raise ValueError("overfit site manifest registration mismatch")
    raw_sites = payload.get("site_indices")
    expected_count = screen.get("gate_o_fixed_unique_sites")
    if (
        not isinstance(raw_sites, list)
        or len(raw_sites) != expected_count
        or any(not isinstance(site, int) or isinstance(site, bool) for site in raw_sites)
        or len(set(raw_sites)) != len(raw_sites)
        or payload.get("site_count") != len(raw_sites)
    ):
        raise ValueError("overfit site manifest requires exact unique integer sites")
    total = expected_grid["height"] * expected_grid["width"]
    if any(site < 0 or site >= total for site in raw_sites):
        raise ValueError("overfit site manifest contains out-of-grid sites")
    return {sample_id: torch.tensor(raw_sites, dtype=torch.long)}


def _load_validation_site_manifest(
    config: Mapping[str, object],
    splits: Mapping[str, list[int]],
) -> tuple[dict[int, torch.Tensor], str]:
    """Load the immutable 64-grid/2048-site registration used by H1--H3."""

    screen = config.get("screen")
    sampling = config.get("sampling")
    data = config.get("data")
    if not all(isinstance(value, Mapping) for value in (screen, sampling, data)):
        raise ValueError("formal validation requires screen, sampling, and data mappings")
    path = _repository_path(screen.get("validation_site_manifest", ""))
    expected_digest = screen.get("validation_site_manifest_sha256")
    digest = _sha256(path)
    if not isinstance(expected_digest, str) or digest != expected_digest:
        raise ValueError("validation site manifest SHA-256 mismatch")
    payload = json.loads(path.read_bytes())
    required = {
        "grid", "purpose", "sample_ids", "schema", "schema_version",
        "selection_strategy", "site_count", "site_indices",
        "sites_shared_across_scenes", "split", "split_manifest",
        "split_manifest_sha256",
    }
    expected_grid = {
        "height": int(sampling["target_height"]),
        "width": int(sampling["target_width"]),
        "index_order": "row_major_x_then_z",
    }
    validation_ids = list(splits["val"])
    raw_sites = payload.get("site_indices") if isinstance(payload, dict) else None
    expected_count = screen.get("validation_sites_per_scene")
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("schema") != "ais_fixed_spatial_sites"
        or payload.get("schema_version") != 1
        or payload.get("purpose") != "fixed_validation_sites"
        or payload.get("split") != "val"
        or payload.get("grid") != expected_grid
        or payload.get("sample_ids") != validation_ids
        or payload.get("sites_shared_across_scenes") is not True
        or payload.get("split_manifest") != data.get("split_manifest")
        or payload.get("split_manifest_sha256")
        != data.get("split_manifest_sha256")
        or not isinstance(raw_sites, list)
        or len(raw_sites) != expected_count
        or payload.get("site_count") != expected_count
        or any(not isinstance(site, int) or isinstance(site, bool) for site in raw_sites)
        or len(set(raw_sites)) != len(raw_sites)
    ):
        raise ValueError("validation site manifest registration mismatch")
    total = expected_grid["height"] * expected_grid["width"]
    if any(site < 0 or site >= total for site in raw_sites):
        raise ValueError("validation site manifest contains out-of-grid sites")
    indices = torch.tensor(raw_sites, dtype=torch.long)
    return {sample_id: indices.clone() for sample_id in validation_ids}, digest


@torch.no_grad()
def _overfit_physical_metrics(
    model: torch.nn.Module,
    store: DenseCPUQueryStore,
    sample_id: int,
    site_indices: torch.Tensor,
    config: Mapping[str, object],
    device: torch.device,
    normalization: AISNormalizationBinding,
) -> dict[str, float]:
    """Evaluate the exact-last Gate O model on its registered physical sites."""

    model.eval()
    sampling = config["sampling"]
    scene = resize_query_scene(
        store.read_scene(sample_id),
        int(sampling["target_height"]),
        int(sampling["target_width"]),
    )
    sites = DenseCPUQueryStore.gather_loaded_scene(scene, site_indices)
    inputs = build_scene_model_inputs(
        scene, int(sampling["global_size"]), device, normalization=normalization
    )
    context = _encode_model_global(model, inputs)
    query_xz = normalize_physical_xz(
        sites.physical_xz, scene.x_m, scene.z_m
    ).float()[None]
    chunk = int(config["train"].get("query_chunk_size", query_xz.shape[1]))
    predictions = [
        _decode_model_queries(
            model,
            context,
            inputs,
            query_xz[:, start : start + chunk].to(device),
        ).cpu()
        for start in range(0, query_xz.shape[1], chunk)
    ]
    prediction = normalization.decode_wavefield(torch.cat(predictions, dim=1))
    target = sites.target_cpu[None]
    prediction_flat, target_flat = prediction.float().flatten(), target.float().flatten()
    prediction_centered = prediction_flat - prediction_flat.mean()
    target_centered = target_flat - target_flat.mean()
    pearson_denominator = (
        torch.linalg.vector_norm(prediction_centered)
        * torch.linalg.vector_norm(target_centered)
    )
    pearson = (
        float(torch.equal(prediction_flat, target_flat))
        if float(pearson_denominator) <= 1e-12
        else float(
            torch.dot(prediction_centered, target_centered)
            .div(pearson_denominator)
            .clamp(-1.0, 1.0)
        )
    )
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(prediction - target)
            / torch.linalg.vector_norm(target).clamp_min(1e-8)
        ),
        "relative_l2_q4": float(
            torch.linalg.vector_norm(prediction[..., 120:] - target[..., 120:])
            / torch.linalg.vector_norm(target[..., 120:]).clamp_min(1e-8)
        ),
        "prediction_target_norm_ratio": float(
            torch.linalg.vector_norm(prediction_flat)
            / torch.linalg.vector_norm(target_flat).clamp_min(1e-12)
        ),
        "prediction_target_pearson": pearson,
    }


def _phase_config(
    config: dict[str, Any], phase: dict[str, Any], *, phase_index: int
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if "learning_rate" in resolved["sampling"]:
        resolved["train"]["learning_rate"] = resolved["sampling"]["learning_rate"]
    if "query_chunk_size" in resolved["sampling"]:
        resolved["train"]["query_chunk_size"] = _strict_positive(
            resolved["sampling"]["query_chunk_size"],
            "sampling.query_chunk_size",
        )
    if "loss_profile" in phase:
        profiles = config.get("loss_profiles")
        if not isinstance(profiles, dict) or phase["loss_profile"] not in profiles:
            raise ValueError(f"unknown loss profile {phase['loss_profile']!r}")
        resolved["loss"].update(profiles[phase["loss_profile"]])
    if "loss" in phase:
        if not isinstance(phase["loss"], dict):
            raise ValueError("phase loss override must be a mapping")
        resolved["loss"].update(phase["loss"])
    if phase_index == 0 and phase.get("init_from") == "random" and any(
        float(resolved["loss"].get(key, 0.0)) != 0.0
        for key in ("receiver_weight", "phase_weight", "local_spectrum_weight", "energy_weight")
    ):
        raise ValueError("the first curriculum phase must be field-only")
    return resolved


def _fixed_validation_sites(
    sample_ids: list[int], target_height: int, target_width: int, count: int
) -> dict[int, torch.Tensor]:
    total = int(target_height) * int(target_width)
    count = min(_strict_positive(count, "validation site count"), total)
    indices = torch.linspace(0, total - 1, count, dtype=torch.float64).round().long()
    if torch.unique(indices).numel() != count:
        raise RuntimeError("validation site construction produced duplicates")
    return {sample_id: indices.clone() for sample_id in sample_ids}


def _new_optimizer(model: torch.nn.Module, config: dict[str, Any], updates: int):
    train = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train.get("learning_rate", 2e-4)),
        weight_decay=float(train.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(updates, 1))
    return optimizer, scheduler


def _existing_phase_best(metrics_path: Path, phase_name: str, metric_name: str) -> float:
    if not metrics_path.exists():
        return float("inf")
    key = f"val_{metric_name}"
    values: list[float] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("phase") == phase_name
            and key in row
            and row.get("best_eligible", True) is not False
        ):
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError("metrics JSONL contains a nonfinite validation metric")
            values.append(value)
    return min(values, default=float("inf"))


def _load_model_weights(
    snapshot: TrainingCheckpointSnapshot,
    model: torch.nn.Module,
) -> None:
    payload = snapshot.payload
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("initialization checkpoint has no model state")
    incompatibility = model.load_state_dict(state, strict=False)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError(
            f"initialization checkpoint keys differ: missing={incompatibility.missing_keys}, "
            f"unexpected={incompatibility.unexpected_keys}"
        )
    print(json.dumps({"checkpoint": str(snapshot.path), "loaded_keys": len(state), "missing_keys": [], "unexpected_keys": []}))


def _restore_finalize_checkpoint(
    snapshot: TrainingCheckpointSnapshot,
    model: torch.nn.Module,
) -> None:
    """Restore model weights only for a metadata-finalization process."""
    _load_model_weights(snapshot, model)


def main(argv: list[str] | None = None) -> int:
    wall_started = time.monotonic()
    args = build_parser().parse_args(argv)
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint cannot be combined")
    if args.finalize_overfit_only and args.finalize_training_only:
        raise ValueError("finalize-only modes are mutually exclusive")
    if args.screen_gate is not None and (
        args.dataset_binding_sha256 is None
        or args.execution_binding_sha256 is None
        or args.dataset_content_manifest is None
    ):
        raise ValueError("formal screen training requires dataset and execution bindings")
    if args.max_train_batches is not None:
        _strict_positive(args.max_train_batches, "max_train_batches")
    if args.max_val_batches is not None:
        _strict_positive(args.max_val_batches, "max_val_batches")
    if (args.overfit_sample_id is None) != (args.overfit_site_manifest is None):
        raise ValueError("--overfit-sample-id and --overfit-site-manifest require each other")
    if args.finalize_overfit_only and (
        args.resume is None
        or args.overfit_sample_id is None
        or args.max_train_batches is not None
    ):
        raise ValueError(
            "--finalize-overfit-only requires --resume, both overfit flags, and no training budget"
        )
    if args.finalize_training_only and (
        args.resume is None
        or args.overfit_sample_id is not None
        or args.max_train_batches is not None
        or args.max_val_batches is not None
    ):
        raise ValueError(
            "--finalize-training-only requires --resume, no overfit flags, and no training budget"
        )
    finalize_only = args.finalize_overfit_only or args.finalize_training_only
    config = _load_configuration(args.config)
    data_path = Path(config["data"]["path"])
    dataset_binding: DatasetContentBinding | None = None
    if any(
        value is not None
        for value in (
            args.dataset_binding_sha256,
            args.execution_binding_sha256,
            args.dataset_content_manifest,
        )
    ):
        if (
            args.dataset_binding_sha256 is None
            or args.execution_binding_sha256 is None
            or args.dataset_content_manifest is None
        ):
            raise ValueError("formal training requires all dataset/execution bindings")
        dataset_binding = load_dataset_content_binding(
            args.dataset_content_manifest,
            data_path,
            expected_manifest_sha256=args.dataset_binding_sha256,
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if dataset_binding is not None and (
        _local_execution_binding_sha256(args.device)
        != args.execution_binding_sha256
    ):
        raise ValueError("child execution environment differs from runner binding")
    config_hash = _config_sha256(config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    seed = int(config.get("seed", 2026))
    normalization = _load_normalization_binding(config)
    split_path = Path(config["data"]["split_manifest"])
    splits = _load_splits(split_path)
    split_hash = _sha256(split_path)
    phases = config["train"]["phases"]
    first_init = phases[0]["init_from"]
    smoke_without_init = args.max_train_batches is not None
    if (
        first_init == "external_checkpoint"
        and args.init_checkpoint is None
        and args.resume is None
        and not smoke_without_init
    ):
        raise ValueError("external_checkpoint initialization requires --init-checkpoint")
    if first_init == "random" and args.init_checkpoint is not None:
        raise ValueError("random initialization cannot be combined with --init-checkpoint")

    preflight_path = args.resume or args.init_checkpoint
    checkpoint_snapshot = (
        load_training_checkpoint_snapshot(preflight_path)
        if preflight_path is not None
        else None
    )
    preflight_metadata = None
    if preflight_path is not None:
        if checkpoint_snapshot is None:
            raise RuntimeError("checkpoint snapshot was not loaded")
        preflight_metadata = validate_query_checkpoint_preflight(
            checkpoint_snapshot.payload,
            expected_config_sha256=config_hash if args.resume is not None else None,
            expected_split_manifest_sha256=split_hash,
            expected_runtime_seed=seed,
            expected_normalization_stats_sha256=normalization.stats_sha256,
            expected_normalization_contract=normalization.contract_id,
            expected_normalization_velocity_mean=normalization.velocity_mean,
            expected_normalization_velocity_std=normalization.velocity_std,
        )

    resume_global_step = None
    resume_phase_index = None
    resume_phase_update = None
    if args.resume is not None:
        if checkpoint_snapshot is None or preflight_metadata is None:
            raise RuntimeError("resume checkpoint preflight was not completed")
        resume_global_step = preflight_metadata.global_step
        resume_phase_index = preflight_metadata.phase_index
        resume_phase_update = preflight_metadata.phase_update
        if args.screen_gate is not None:
            same_run = args.resume.resolve() == (
                args.output_dir / "checkpoints/last.pt"
            ).resolve()
            lineage_parent = checkpoint_snapshot.payload.get(
                "parent_checkpoint_sha256"
            )
            lineage_path = args.output_dir / "checkpoints/screen_lineage.json"
            if same_run and lineage_path.is_file():
                lineage_parent = json.loads(lineage_path.read_bytes()).get(
                    "parent_checkpoint_sha256"
                )
            _validate_screen_resume_identity(
                checkpoint_snapshot.payload,
                candidate_id=config["screen"]["candidate_id"],
                current_gate=args.screen_gate,
                same_run=same_run,
                dataset_binding_sha256=args.dataset_binding_sha256,
                execution_binding_sha256=args.execution_binding_sha256,
                lineage_parent_sha256=lineage_parent,
            )
        if resume_phase_index is None or not 0 <= resume_phase_index < len(phases):
            raise ValueError("resume checkpoint has invalid phase index")
        if resume_phase_update is None:
            raise ValueError("resume checkpoint has invalid phase update")
        resume_phase_limit = _phase_updates(phases[resume_phase_index])
        if not 0 <= resume_phase_update <= resume_phase_limit:
            raise ValueError("resume checkpoint has invalid phase update")
        resume_phase_start = sum(
            _phase_updates(item) for item in phases[:resume_phase_index]
        )
        if resume_global_step != resume_phase_start + resume_phase_update:
            raise ValueError("resume phase update is inconsistent with global_step")

    overfit_sites = (
        _load_overfit_sites(
            config,
            splits,
            args.overfit_sample_id,
            args.overfit_site_manifest,
            args.max_train_batches,
            args.max_val_batches,
            starting_update=resume_global_step or 0,
            finalize_only=finalize_only,
        )
        if args.overfit_sample_id is not None and args.overfit_site_manifest is not None
        else None
    )
    registered_validation_sites: dict[int, torch.Tensor] | None = None
    validation_site_manifest_sha256: str | None = None
    if args.screen_gate in {"H1", "H2", "H3"}:
        (
            registered_validation_sites,
            validation_site_manifest_sha256,
        ) = _load_validation_site_manifest(config, splits)

    if not finalize_only:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = checkpoint_dir / "runtime_seed.json"
    lineage_path = checkpoint_dir / "screen_lineage.json"
    parent_checkpoint_sha256 = None
    if args.resume is not None:
        if args.resume.resolve() == (checkpoint_dir / "last.pt").resolve() and lineage_path.is_file():
            prior_runtime = json.loads(lineage_path.read_text(encoding="utf-8"))
            parent_checkpoint_sha256 = prior_runtime.get("parent_checkpoint_sha256")
        else:
            if checkpoint_snapshot is None:
                raise RuntimeError("resume checkpoint snapshot is unavailable")
            parent_checkpoint_sha256 = checkpoint_snapshot.sha256
    _atomic_json(runtime_path, {"runtime_seed": seed})
    _atomic_json(
        lineage_path,
        {
            "runtime_seed": seed,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
        },
    )

    store = (
        DenseCPUQueryStore(data_path)
        if dataset_binding is None
        else DenseCPUQueryStore(
            data_path, dataset_content_manifest=args.dataset_content_manifest
        )
    )
    if finalize_only:
        torch_rng_state = torch.random.get_rng_state()
        try:
            model = AISMQFNO(**_model_kwargs(config, normalization)).to(device)
        finally:
            torch.random.set_rng_state(torch_rng_state)
    else:
        model = AISMQFNO(**_model_kwargs(config, normalization)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.init_checkpoint is not None:
        if checkpoint_snapshot is None:
            raise RuntimeError("initialization checkpoint preflight was not completed")
        _load_model_weights(checkpoint_snapshot, model)
    sampling = config["sampling"]
    sampler_options = _sampler_kwargs(config)
    sampler = (
        None
        if finalize_only
        else AdaptiveSpatialSampler(
            int(sampling["target_height"]),
            int(sampling["target_width"]),
            sampler_options["mixture"],
            seed,
            tile_size=sampler_options["tile_size"],
            ema_momentum=sampler_options["ema_momentum"],
        )
    )
    generator = None if finalize_only else torch.Generator().manual_seed(seed + 101)
    train_order = list(splits["train"])
    validation_ids = list(splits["val"])
    if overfit_sites is not None:
        train_order = [args.overfit_sample_id]
        validation_ids = [args.overfit_sample_id]
    elif args.max_val_batches is not None:
        validation_ids = validation_ids[: args.max_val_batches]
    validation_sites = (
        None
        if finalize_only
        else (
            overfit_sites
            if overfit_sites is not None
            else registered_validation_sites
            if registered_validation_sites is not None
            else _fixed_validation_sites(
                validation_ids,
                int(sampling["target_height"]),
                int(sampling["target_width"]),
                int(config["train"]["query_sites_per_scene"]),
            )
        )
    )
    metrics_path = output_dir / "metrics.jsonl"
    if args.resume is None:
        metrics_path.write_text("", encoding="utf-8")
    events: list[str] = []
    if first_init == "external_checkpoint" and args.init_checkpoint is None and smoke_without_init:
        events.append("smoke_random_init_without_external_checkpoint")
    if args.init_checkpoint is not None and first_init == "external_checkpoint":
        events.append(f"{config['train']['phases'][0]['name']}:load_external_checkpoint")
    global_step = 0
    data_epoch = 0
    cursor = 0
    resume_metadata = None
    cumulative = 0
    train_batches_consumed = 0
    budget_exhausted = False
    active_phase_index = 0
    active_phase_update = 0
    resume_optimizer = None
    resume_scheduler = None
    resume_cursor = 0
    if args.resume is not None:
        if (
            checkpoint_snapshot is None
            or preflight_metadata is None
            or resume_phase_index is None
        ):
            raise RuntimeError("resume checkpoint preflight was not completed")
        if finalize_only:
            _restore_finalize_checkpoint(checkpoint_snapshot, model)
            resume_metadata = preflight_metadata
            events.append(
                f"{phases[resume_phase_index]['name']}:finalize_checkpoint_metadata"
            )
        else:
            resume_phase = phases[resume_phase_index]
            resume_cfg = _phase_config(
                config, resume_phase, phase_index=resume_phase_index
            )
            resume_optimizer, resume_scheduler = _new_optimizer(
                model, resume_cfg, _phase_updates(resume_phase)
            )
            resume_metadata = load_query_checkpoint(
                None,
                payload=checkpoint_snapshot.payload,
                model=model,
                optimizer=resume_optimizer,
                scheduler=resume_scheduler,
                sampler=sampler,
                data_generator=generator,
                expected_config_sha256=config_hash,
                expected_split_manifest_sha256=split_hash,
                expected_runtime_seed=seed,
                expected_normalization_stats_sha256=normalization.stats_sha256,
                expected_normalization_contract=normalization.contract_id,
                expected_normalization_velocity_mean=normalization.velocity_mean,
                expected_normalization_velocity_std=normalization.velocity_std,
                map_location=device,
            )
        global_step = resume_metadata.global_step
        if global_step != resume_global_step:
            raise ValueError("resume checkpoint global step changed during validation")
        train_order = list(resume_metadata.epoch_scene_order)
        resume_cursor = resume_metadata.epoch_batch_cursor
        data_epoch = resume_metadata.epoch
        cursor = resume_cursor

    phase_cfg = _phase_config(
        config,
        phases[resume_phase_index if resume_phase_index is not None else 0],
        phase_index=resume_phase_index if resume_phase_index is not None else 0,
    )
    if finalize_only:
        active_phase_index = resume_phase_index or 0
        active_phase_update = resume_phase_update or 0

    for phase_index, phase in (() if finalize_only else enumerate(phases)):
        active_phase_index = phase_index
        phase_updates = _phase_updates(phase)
        phase_start = cumulative
        cumulative += phase_updates
        if global_step >= cumulative:
            continue
        phase_name = str(phase["name"])
        phase_cfg = _phase_config(config, phase, phase_index=phase_index)
        if phase_index > 0 and global_step == phase_start:
            phase_best_snapshot = load_training_checkpoint_snapshot(
                output_dir / "best.pt"
            )
            validate_query_checkpoint_preflight(
                phase_best_snapshot.payload,
                expected_config_sha256=config_hash,
                expected_split_manifest_sha256=split_hash,
                expected_runtime_seed=seed,
                expected_normalization_stats_sha256=normalization.stats_sha256,
                expected_normalization_contract=normalization.contract_id,
                expected_normalization_velocity_mean=normalization.velocity_mean,
                expected_normalization_velocity_std=normalization.velocity_std,
            )
            _load_model_weights(phase_best_snapshot, model)
            events.append(f"{phase_name}:load_phase_best")
        if resume_phase_index == phase_index and global_step < cumulative:
            if resume_optimizer is None or resume_scheduler is None:
                raise RuntimeError("resume optimizer state was not restored")
            optimizer, scheduler = resume_optimizer, resume_scheduler
            events.append(f"{phase_name}:resume_optimizer")
        else:
            optimizer, scheduler = _new_optimizer(model, phase_cfg, phase_updates)
            events.append(f"{phase_name}:new_optimizer")
        phase_best = _existing_phase_best(
            metrics_path, phase_name, str(phase.get("validation_metric", "relative_l2"))
        )
        while global_step < cumulative:
            # One scene draw is one optimizer update, so the cosine scheduler and
            # checkpoint boundary advance exactly once per call.
            draw_limit = 1
            draw_limit = min(draw_limit, len(train_order) - cursor)
            if draw_limit <= 0:
                cursor = 0
                continue
            result = train_query_epoch(
                model,
                store,
                train_order,
                sampler,
                optimizer,
                phase_cfg,
                device,
                phase_index,
                global_step,
                epoch_batch_cursor=cursor,
                max_scene_draws=draw_limit,
                normalization=normalization,
            )
            global_step = result.global_step
            train_batches_consumed += result.optimizer_updates
            active_phase_update = global_step - phase_start
            cursor = result.next_epoch_batch_cursor
            if cursor == 0:
                data_epoch += 1
            scheduler.step()
            validation_reason, best_eligible = _validation_schedule(
                active_phase_update,
                phase_updates,
                train_batches_consumed,
                args.max_train_batches,
            )
            should_validate = validation_reason is not None
            validation = (
                validate_query_guard(
                    model,
                    store,
                    validation_ids,
                    validation_sites,
                    phase_cfg,
                    device,
                    normalization=normalization,
                )
                if should_validate
                else {}
            )
            row = {
                "phase": phase_name,
                "phase_index": phase_index,
                "global_step": global_step,
                "train_loss": result.mean_total_loss,
                "validation_reason": validation_reason,
                "best_eligible": best_eligible,
                **{f"train_{key}": value for key, value in result.mean_components.items()},
                **{f"sampler_{key}": value for key, value in result.sampler_diagnostics.items()},
                **{f"val_{key}": value for key, value in validation.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            save_kwargs = dict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                epoch=data_epoch,
                epoch_batch_cursor=cursor,
                epoch_scene_order=train_order,
                data_generator=generator,
                global_step=global_step,
                config_sha256=config_hash,
                split_manifest_sha256=split_hash,
                checkpoint_boundary=QUERY_CHECKPOINT_BOUNDARY,
                phase_index=phase_index,
                phase_update=global_step - phase_start,
                runtime_seed=seed,
                normalization_stats_sha256=normalization.stats_sha256,
                normalization_contract=normalization.contract_id,
                normalization_velocity_mean=normalization.velocity_mean,
                normalization_velocity_std=normalization.velocity_std,
                screen_candidate_id=(
                    config["screen"]["candidate_id"]
                    if args.screen_gate is not None
                    else None
                ),
                screen_gate=args.screen_gate,
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                dataset_binding_sha256=args.dataset_binding_sha256,
                execution_binding_sha256=args.execution_binding_sha256,
            )
            canonical_last = checkpoint_dir / "last.pt"
            save_query_checkpoint(canonical_last, **save_kwargs)
            _publish_root_checkpoint(canonical_last, output_dir / "last.pt")
            if should_validate:
                metric_name = str(phase.get("validation_metric", "relative_l2"))
                if metric_name not in validation:
                    raise ValueError(f"unknown validation metric {metric_name!r}")
                if best_eligible and validation[metric_name] < phase_best:
                    phase_best = validation[metric_name]
                    canonical_best = checkpoint_dir / "best.pt"
                    save_query_checkpoint(canonical_best, **save_kwargs)
                    _publish_root_checkpoint(canonical_best, output_dir / "best.pt")
                    if phase_index == 0:
                        events.append(f"{phase_name}:best")
                elif not best_eligible and not (checkpoint_dir / "best.pt").exists():
                    canonical_best = checkpoint_dir / "best.pt"
                    save_query_checkpoint(canonical_best, **save_kwargs)
                    _publish_root_checkpoint(canonical_best, output_dir / "best.pt")
                    events.append(f"{phase_name}:provisional_best")
            if (
                args.max_train_batches is not None
                and train_batches_consumed >= args.max_train_batches
            ):
                budget_exhausted = True
                break
        if budget_exhausted:
            break
        if global_step >= cumulative and phase_index + 1 < len(phases):
            continue

    _atomic_json(
        output_dir / "training_state.json",
        {
            "phase_index": active_phase_index,
            "phase_update": active_phase_update,
            "global_step": global_step,
            "events": events,
            "checkpoint_boundary": QUERY_CHECKPOINT_BOUNDARY,
        },
    )
    if overfit_sites is not None:
        exact_rows = (
            []
            if args.finalize_overfit_only
            else [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("global_step") == global_step
            ]
        )
        if not args.finalize_overfit_only and len(exact_rows) != 1:
            raise ValueError("Gate O requires exactly one metrics row at the exact last update")
        metric_names = (
            "relative_l2",
            "relative_l2_q4",
            "prediction_target_norm_ratio",
            "prediction_target_pearson",
        )
        physical = _overfit_physical_metrics(
            model,
            store,
            args.overfit_sample_id,
            overfit_sites[args.overfit_sample_id],
            phase_cfg,
            device,
            normalization,
        )
        if set(physical) != set(metric_names):
            raise RuntimeError("Gate O physical metric schema mismatch")
        _atomic_json(
            output_dir / "screen_metrics.json",
            {
                "global_step": global_step,
                "candidate_id": config["screen"]["candidate_id"],
                "last_checkpoint_sha256": _sha256(checkpoint_dir / "last.pt"),
                "sample_id": args.overfit_sample_id,
                "site_manifest_sha256": _sha256(args.overfit_site_manifest),
                "finite": all(np.isfinite(value) for value in physical.values()),
                # Gate O is an architecture representability check on one registered
                # train scene. Task8 deliberately uses one uniform record schema, so
                # the same physical observation is bound under each category key.
                "category_metrics": {
                    category: dict(physical)
                    for category in ("uniform", "layered", "marmousi")
                },
            },
        )
    wall_seconds = time.monotonic() - wall_started
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    _atomic_json(
        output_dir / "summary.json",
        {
            "status": "complete" if global_step >= sum(_phase_updates(item) for item in phases) else "partial",
            "time_steps": 160,
            "query_unit": "spatial_site_full_trace",
            "device": str(device),
            "global_step": global_step,
            "config_sha256": config_hash,
            "split_manifest_sha256": split_hash,
            "events": events,
            "amp": False,
            "metrics_jsonl": str(metrics_path),
            "runtime_seed": seed,
            "last_checkpoint": str(checkpoint_dir / "last.pt"),
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
            "parameter_count": parameter_count,
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "wall_seconds": wall_seconds,
            "gpu_name": gpu_name,
            "dataset_binding_sha256": args.dataset_binding_sha256,
            "execution_binding_sha256": args.execution_binding_sha256,
            "validation_purpose": (
                "screen64_fixed2048"
                if args.screen_gate in {"H1", "H2", "H3"}
                else None
            ),
            "validation_site_manifest_sha256": validation_site_manifest_sha256,
            **(store.binding_summary() if dataset_binding is not None else {}),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
