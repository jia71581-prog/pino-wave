#!/usr/bin/env python3
"""Run a validation-only native full160 AIS-MQFNO census."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import MappingProxyType
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from fno_acoustic.long_horizon_metrics import ReceiverSiteManifest  # noqa: E402
from fno_acoustic.ais_dataset_binding import (  # noqa: E402
    DatasetContentBinding,
    load_dataset_content_binding,
)
from fno_acoustic.ais_normalization import AISNormalizationBinding  # noqa: E402
from fno_acoustic.model_ais_mqfno import AISMQFNO  # noqa: E402
from fno_acoustic.model_factorized import FactorizedAcousticFNO  # noqa: E402
from fno_acoustic.query_census import (  # noqa: E402
    LegacyFullFieldPredictorAdapter, NativePrediction, QueryPredictorAdapter,
    SCREEN_SAMPLE_COLUMNS, aggregate_screen_category_metrics, run_query_census,
)
from fno_acoustic.query_data import DenseCPUQueryStore, resize_query_scene  # noqa: E402
from fno_acoustic.query_reconstruction import (  # noqa: E402
    QueryInferenceScene,
    reconstruct_native,
)
from fno_acoustic.query_training import (  # noqa: E402
    build_scene_model_inputs,
    receiver_site_indices,
    validate_query_checkpoint_normalization,
)
from train_ais_mqfno import (  # noqa: E402
    _config_sha256,
    _load_configuration,
    _load_normalization_binding,
    _load_validation_site_manifest,
    _overfit_physical_metrics,
    _model_kwargs,
    _local_execution_binding_sha256,
)


def _model_patch_geometry(
    model: torch.nn.Module, dx_m: float, dz_m: float
) -> dict[str, object]:
    """Describe the local receptive fields owned by the loaded model."""

    halo_size = getattr(model, "halo_size", None)
    encoder_kind = getattr(model, "local_encoder_kind", None)
    if not isinstance(halo_size, int) or isinstance(halo_size, bool):
        raise ValueError("AIS model must expose an integer halo_size")
    branches = [9, 25] if encoder_kind == "multiscale_9_25" else [halo_size]
    return {
        "source": "model",
        "branches": branches,
        "branch_physical_spans_m": [
            {
                "halo_size": branch,
                "x_m": (branch - 1) * dx_m,
                "z_m": (branch - 1) * dz_m,
            }
            for branch in branches
        ],
    }


def _run_screen64_census(
    model: torch.nn.Module,
    store: DenseCPUQueryStore,
    sample_ids: list[int],
    splits: dict[str, list[int]],
    config: dict[str, Any],
    normalization: AISNormalizationBinding,
    device: torch.device,
    output_dir: Path,
    provenance: Mapping[str, str],
    supplied_manifest: Path,
) -> tuple[Path, Path, int]:
    registered = resolve_repository_config_path(
        config["screen"]["validation_site_manifest"]
    )
    if supplied_manifest.resolve() != registered.resolve():
        raise ValueError("screen validation site manifest path mismatch")
    sites_by_sample, site_digest = _load_validation_site_manifest(config, splits)
    rows: list[dict[str, object]] = []
    spacing: tuple[float, float] | None = None
    for sample_id in sample_ids:
        scene = resize_query_scene(store.read_scene(sample_id), 64, 64)
        dx_m = float(scene.x_m[1] - scene.x_m[0])
        dz_m = float(scene.z_m[1] - scene.z_m[0])
        if spacing is None:
            spacing = (dx_m, dz_m)
        elif spacing != (dx_m, dz_m):
            raise ValueError("screen64 scene spacing differs across validation scenes")
        values = _overfit_physical_metrics(
            model,
            store,
            sample_id,
            sites_by_sample[sample_id],
            config,
            device,
            normalization,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "category": str(scene.metadata["model_type"]),
                "split": "val",
                "predictor_family": "ais_mqfno",
                "config_sha256": provenance["config_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "split_manifest_sha256": provenance["split_manifest_sha256"],
                "normalization_sha256": provenance["normalization_sha256"],
                "validation_site_manifest_sha256": site_digest,
                "relative_l2": values["relative_l2"],
                "relative_l2_q4": values["relative_l2_q4"],
                "prediction_target_norm_ratio": values[
                    "prediction_target_norm_ratio"
                ],
                "zero_relative_l2": 1.0,
                "zero_relative_l2_q4": 1.0,
            }
        )
    aggregated = aggregate_screen_category_metrics(rows)
    if spacing is None:
        raise ValueError("screen64 validation selection is empty")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        with (stage / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCREEN_SAMPLE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "schema_version": 1,
            "purpose": "screen64_fixed2048",
            "split": "val",
            "sample_count": len(rows),
            "provenance": {
                **dict(provenance),
                "validation_site_manifest_sha256": site_digest,
            },
            "grid": {
                "height": 64,
                "width": 64,
                "dx_m": spacing[0],
                "dz_m": spacing[1],
                "patch_geometry": _model_patch_geometry(
                    model, spacing[0], spacing[1]
                ),
            },
            "categories": {
                category: {
                    "sample_count": values["sample_count"],
                    "mean": {
                        name: value
                        for name, value in values.items()
                        if name != "sample_count"
                    },
                }
                for category, values in aggregated.items()
                if category != "global"
            },
            "category_metrics": aggregated,
            **(
                store.binding_summary()
                if store.dataset_binding is not None
                else {}
            ),
        }
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, output_dir)
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
    return output_dir / "samples.csv", output_dir / "summary.json", len(rows)

PINNED_LEGACY_RAW_B1_SHA256 = (
    "91546ba21c31e0e875c0ed7051068c279b455a94f38206274094e1a9a9233653"
)
PINNED_LEGACY_RAW_B1_CONFIG = REPO_ROOT / "configs/ais_mqfno_64x160_b1_uniform.yaml"
PINNED_LEGACY_RAW_B1_CONFIG_SHA256 = (
    "f9a8b81e116f3ba4c463ee0875b67f90ab46eba413a8e78f10d834ebc8e8dd81"
)
_LEGACY_RAW_B1_ALLOWED_OPERATION = "validation_baseline"
_LEGACY_RAW_B1_FORBIDDEN_OPERATIONS = frozenset(
    {
        "representative_tuning",
        "test",
        "resume",
        "initialization",
        "recipe_freeze",
        "authorization",
        "sealed_test",
    }
)


def validate_legacy_raw_b1_operation(operation: str) -> str:
    if operation == _LEGACY_RAW_B1_ALLOWED_OPERATION:
        return operation
    if operation in _LEGACY_RAW_B1_FORBIDDEN_OPERATIONS:
        raise ValueError(
            "legacy raw B1 is restricted to the read-only validation baseline"
        )
    raise ValueError("unknown legacy raw B1 operation")


def validate_legacy_raw_b1_config(config: dict[str, Any]) -> str:
    """Require the exact original raw-B1 configuration before model construction."""

    config_sha256 = _config_sha256(config)
    if config_sha256 != PINNED_LEGACY_RAW_B1_CONFIG_SHA256:
        raise ValueError("legacy raw B1 requires the pinned original config")
    return config_sha256


def resolve_repository_config_path(value: str | Path) -> Path:
    """Resolve paths stored inside configs against the repository root."""

    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--chunk-sites", type=int, default=2048)
    parser.add_argument("--verify-monolithic", action="store_true")
    parser.add_argument("--representative-manifest", type=Path)
    parser.add_argument("--legacy-raw-b1-baseline", action="store_true")
    parser.add_argument("--dataset-binding-sha256")
    parser.add_argument("--execution-binding-sha256")
    parser.add_argument("--dataset-content-manifest", type=Path)
    parser.add_argument(
        "--evaluation-purpose",
        choices=(
            "screen64_fixed2048",
            "native400_final_census",
            "native_full_census",
        ),
        default="native_full_census",
    )
    parser.add_argument("--screen-validation-site-manifest", type=Path)
    return parser


def _positive(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_checkpoint_binding(
    payload: object,
    config_sha256: str,
    split_manifest_sha256: str,
    normalization_stats_sha256: str | None = None,
    normalization_contract: str | None = None,
    normalization_velocity_mean: float | None = None,
    normalization_velocity_std: float | None = None,
) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    if (normalization_stats_sha256 is None) != (normalization_contract is None):
        raise ValueError(
            "evaluation normalization hash and contract must be provided together"
        )
    if normalization_stats_sha256 is None:
        if payload.get("schema_version") not in (1, 2, 3):
            raise ValueError(
                "historical evaluation requires a schema-v1, schema-v2, or "
                "schema-v3 query checkpoint"
            )
    else:
        validate_query_checkpoint_normalization(
            payload,
            normalization_stats_sha256,
            normalization_contract,
            normalization_velocity_mean,
            normalization_velocity_std,
        )
    if payload.get("config_sha256") != config_sha256:
        raise ValueError("checkpoint config SHA-256 mismatch")
    if payload.get("split_manifest_sha256") != split_manifest_sha256:
        raise ValueError("checkpoint split manifest SHA-256 mismatch")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("checkpoint model_state_dict must be a nonempty tensor mapping")
    return state


def validate_legacy_checkpoint_binding(
    payload: object, expected_config: dict[str, Any], expected_split: dict[str, list[int]],
    expected_normalization: dict[str, object] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if not isinstance(payload, dict) or payload.get("full_config") != expected_config:
        raise ValueError("legacy checkpoint full config mismatch")
    if payload.get("model_config") != expected_config.get("model"):
        raise ValueError("legacy checkpoint model config mismatch")
    checkpoint_split = payload.get("split_manifest")
    if not isinstance(checkpoint_split, dict) or any(
        checkpoint_split.get(name) != expected_split.get(name) for name in ("train", "val", "test")
    ):
        raise ValueError("legacy checkpoint split manifest mismatch")
    state = payload.get("model_state_dict")
    stats = payload.get("normalization_stats")
    if not isinstance(state, dict) or not state or not isinstance(stats, dict):
        raise ValueError("legacy checkpoint state or normalization metadata is invalid")
    if expected_normalization is not None and stats != expected_normalization:
        raise ValueError("legacy checkpoint normalization metadata mismatch")
    return state, expected_normalization if expected_normalization is not None else stats


def load_checkpoint_snapshot(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("checkpoint is not the pinned legacy B1 checkpoint")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    return payload, actual


def checkpoint_runtime_seed(payload: dict[str, object], fallback: object) -> str:
    if payload.get("schema_version") not in (3, 4):
        return str(fallback)
    value = payload.get("runtime_seed")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("checkpoint runtime_seed is invalid")
    return str(value)


def load_split_snapshot(path: Path) -> tuple[dict[str, list[int]], str]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("split manifest is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("split manifest must be a mapping")
    result: dict[str, list[int]] = {}
    seen: set[int] = set()
    for split in ("train", "val", "test"):
        values = payload.get(split)
        if not isinstance(values, list):
            raise ValueError(f"split manifest requires list {split!r}")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in values):
            raise ValueError("split sample IDs must be nonnegative integers")
        if any(value in seen for value in values) or len(set(values)) != len(values):
            raise ValueError("split manifest sample IDs must be unique and disjoint")
        seen.update(values)
        result[split] = list(values)
    if not result["train"] or not result["val"]:
        raise ValueError("train and val splits must be nonempty")
    return result, hashlib.sha256(raw).hexdigest()


def normalization_hash(config: dict[str, Any]) -> str:
    formal = config.get("normalization")
    data = config["data"]
    formal_path = (
        resolve_repository_config_path(formal["stats_path"])
        if isinstance(formal, dict) and formal.get("stats_path") is not None
        else None
    )
    legacy_paths = [
        resolve_repository_config_path(data[key])
        for key in ("normalization_manifest", "normalization_stats")
        if data.get(key) is not None
    ]
    paths = ([formal_path] if formal_path is not None else []) + legacy_paths
    hashes: list[str] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"normalization stats path is not a readable file: {path}")
        hashes.append(_file_sha256(path))
    if formal_path is not None and any(value != hashes[0] for value in hashes[1:]):
        raise ValueError("formal and legacy normalization stats conflict")
    if formal_path is not None:
        return hashes[0]
    if hashes:
        if len(set(hashes)) != 1:
            raise ValueError("legacy normalization stats paths conflict")
        return hashes[0]
    return _canonical_sha256({"normalization": "identity"})


def _receiver_manifest(config: dict[str, Any], store: DenseCPUQueryStore, sample_id: int) -> ReceiverSiteManifest:
    receiver = config.get("receiver")
    if not isinstance(receiver, dict):
        raise ValueError("config receiver section must be a mapping")
    scene = store.read_scene(sample_id)
    indices = receiver_site_indices(scene, receiver)
    width = scene.target_cpu.shape[1]
    x_index = torch.div(indices, width, rounding_mode="floor")
    z_index = indices.remainder(width)
    coordinates = torch.stack((scene.x_m[x_index], scene.z_m[z_index]), dim=-1)
    payload = {
        "height": scene.target_cpu.shape[0], "width": width,
        "site_indices": indices.tolist(), "physical_xz": coordinates.tolist(),
        "receiver": receiver,
    }
    digest = _canonical_sha256(payload)
    return ReceiverSiteManifest(tuple(scene.target_cpu.shape[:2]), indices, coordinates, digest)


@dataclass(frozen=True)
class RepresentativeSelection:
    sample_ids: tuple[int, ...]
    sha256: str
    selection_rule: str
    metric_source_sha256: str
    seed: int


def _readonly_checkpoint_clone(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _readonly_checkpoint_clone(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_readonly_checkpoint_clone(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True, init=False)
class AISEvaluationCheckpoint:
    _model_state_dict: Mapping[str, object]
    checkpoint_sha256: str
    runtime_seed: str
    normalization: AISNormalizationBinding | None
    legacy_raw_b1: bool

    def __init__(
        self,
        model_state_dict: Mapping[str, object],
        checkpoint_sha256: str,
        runtime_seed: str,
        normalization: AISNormalizationBinding | None,
        legacy_raw_b1: bool,
    ) -> None:
        private_state = _readonly_checkpoint_clone(model_state_dict)
        if not isinstance(private_state, Mapping):
            raise TypeError("model_state_dict must be a mapping")
        object.__setattr__(self, "_model_state_dict", private_state)
        object.__setattr__(self, "checkpoint_sha256", checkpoint_sha256)
        object.__setattr__(self, "runtime_seed", runtime_seed)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "legacy_raw_b1", legacy_raw_b1)

    @property
    def model_state_dict(self) -> Mapping[str, object]:
        public_state = _readonly_checkpoint_clone(self._model_state_dict)
        if not isinstance(public_state, Mapping):
            raise RuntimeError("private model state is not a mapping")
        return public_state


def _load_evaluation_normalization_binding(
    config: dict[str, Any],
) -> AISNormalizationBinding:
    resolved = copy.deepcopy(config)
    normalization = resolved.get("normalization")
    if isinstance(normalization, dict) and normalization.get("stats_path") is not None:
        normalization["stats_path"] = str(
            resolve_repository_config_path(normalization["stats_path"])
        )
    return _load_normalization_binding(resolved)


def preflight_ais_evaluation_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    split_manifest_sha256: str,
    *,
    legacy_raw_b1_baseline: bool,
) -> AISEvaluationCheckpoint:
    """Bind an AIS checkpoint before model/data construction and strip training state."""
    payload, checkpoint_sha256 = load_checkpoint_snapshot(
        checkpoint_path,
        expected_sha256=(
            PINNED_LEGACY_RAW_B1_SHA256 if legacy_raw_b1_baseline else None
        ),
    )
    config_sha256 = _config_sha256(config)
    if legacy_raw_b1_baseline:
        validate_legacy_raw_b1_operation("validation_baseline")
        config_sha256 = validate_legacy_raw_b1_config(config)
        model_state = validate_checkpoint_binding(
            payload,
            config_sha256,
            split_manifest_sha256,
        )
        normalization = None
    else:
        normalization = _load_evaluation_normalization_binding(config)
        model_state = validate_checkpoint_binding(
            payload,
            config_sha256,
            split_manifest_sha256,
            normalization.stats_sha256,
            normalization.contract_id,
            normalization.velocity_mean,
            normalization.velocity_std,
        )
    return AISEvaluationCheckpoint(
        model_state,
        checkpoint_sha256,
        checkpoint_runtime_seed(payload, config.get("seed", 2026)),
        normalization,
        legacy_raw_b1_baseline,
    )


def load_legacy_raw_b1(path: Path) -> AISEvaluationCheckpoint:
    """Load the sole pinned raw B1 model as a read-only validation binding."""

    raw_config = yaml.safe_load(
        PINNED_LEGACY_RAW_B1_CONFIG.read_text(encoding="utf-8")
    )
    if not isinstance(raw_config, dict):
        raise ValueError("pinned legacy B1 config must contain a mapping")
    validate_legacy_raw_b1_config(raw_config)
    split_path = resolve_repository_config_path(raw_config["data"]["split_manifest"])
    _, split_sha256 = load_split_snapshot(split_path)
    return preflight_ais_evaluation_checkpoint(
        Path(path),
        raw_config,
        split_sha256,
        legacy_raw_b1_baseline=True,
    )


@dataclass(frozen=True)
class NormalizedQueryPredictorAdapter:
    model: torch.nn.Module
    device: torch.device
    chunk_sites: int
    normalization: AISNormalizationBinding
    global_size: int = 100
    predictor_family: str = "ais_mqfno"

    def __post_init__(self) -> None:
        _positive(self.chunk_sites, "chunk_sites")
        _positive(self.global_size, "global_size")
        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device")

    def predict_scene(self, scene: Any) -> NativePrediction:
        height, width = scene.target_cpu.shape[:2]
        inputs = build_scene_model_inputs(
            scene,
            min(self.global_size, height, width),
            self.device,
            normalization=self.normalization,
        )
        inference = QueryInferenceScene(
            inputs.global_inputs,
            inputs.native_static,
            inputs.time_s,
            height,
            width,
            1,
            self.device,
            dx_m=inputs.dx_m,
            dz_m=inputs.dz_m,
        )
        self.model.eval()
        result = reconstruct_native(self.model, inference, self.chunk_sites)
        if not isinstance(result.field, torch.Tensor):
            raise TypeError("in-memory query reconstruction must return a tensor")
        physical = self.normalization.decode_wavefield(result.field)
        return NativePrediction(
            physical.float().cpu(),
            int(result.coverage.min()),
            int(result.coverage.max()),
        )


def load_representative_manifest(
    path: Path,
    store: DenseCPUQueryStore,
    validation_ids: list[int],
    split_manifest_sha256: str,
) -> RepresentativeSelection:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("representative manifest is not valid UTF-8 JSON") from error
    required = {
        "split", "split_manifest_sha256", "representatives", "selection_rule",
        "metric_source_sha256", "seed",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("representative manifest has an invalid schema")
    if payload["split"] != "val" or payload["split_manifest_sha256"] != split_manifest_sha256:
        raise ValueError("representative manifest is not bound to this validation split")
    representatives = payload["representatives"]
    categories = ("uniform", "layered", "marmousi")
    if (
        not isinstance(representatives, dict) or set(representatives) != set(categories)
    ):
        raise ValueError("representative manifest requires uniform/layered/marmousi IDs")
    ids = [representatives[category] for category in categories]
    if (
        any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in ids)
        or len(set(ids)) != len(ids)
        or any(value not in validation_ids for value in ids)
    ):
        raise ValueError("representative manifest sample IDs must be unique validation IDs")
    selection_rule = payload["selection_rule"]
    if not isinstance(selection_rule, str) or not selection_rule.strip():
        raise ValueError("representative selection_rule must be nonempty")
    metric_hash = payload["metric_source_sha256"]
    if (
        not isinstance(metric_hash, str) or len(metric_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in metric_hash)
    ):
        raise ValueError("representative metric_source_sha256 must be a 64-character hex digest")
    seed = payload["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("representative seed must be a nonnegative integer")
    for category, sample_id in zip(categories, ids, strict=True):
        scene_category = store.read_scene(sample_id).metadata.get("model_type")
        if scene_category != category:
            raise ValueError(
                f"representative category mismatch for {category}: {scene_category!r}"
            )
    return RepresentativeSelection(
        tuple(ids), actual, selection_rule, metric_hash.lower(), seed
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunk_sites = _positive(args.chunk_sites, "chunk_sites")
    if args.max_samples is not None:
        _positive(args.max_samples, "max_samples")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("config must contain a mapping")
    model_name = raw_config.get("model", {}).get("name")
    is_b0 = model_name == "factorized_fno"
    if is_b0 and args.legacy_raw_b1_baseline:
        raise ValueError("legacy raw B1 mode is only valid for AIS-MQFNO")
    if args.legacy_raw_b1_baseline and args.representative_manifest is not None:
        validate_legacy_raw_b1_operation("representative_tuning")
    config = raw_config if is_b0 else _load_configuration(args.config)
    if args.evaluation_purpose == "screen64_fixed2048":
        if args.screen_validation_site_manifest is None:
            raise ValueError("screen64 evaluation requires validation site manifest")
        if is_b0 or args.legacy_raw_b1_baseline:
            raise ValueError("screen64 evaluation is only valid for formal AIS candidates")
    elif args.screen_validation_site_manifest is not None:
        raise ValueError("native400 census must not consume the screen site manifest")
    data_path = resolve_repository_config_path(config["data"]["path"])
    formal_purpose = args.evaluation_purpose in {
        "screen64_fixed2048",
        "native400_final_census",
    }
    dataset_binding: DatasetContentBinding | None = None
    if formal_purpose or any(
        value is not None
        for value in (
            args.dataset_binding_sha256,
            args.execution_binding_sha256,
            args.dataset_content_manifest,
        )
    ):
        if (
            args.dataset_binding_sha256 is None
            or args.dataset_content_manifest is None
            or args.execution_binding_sha256 is None
        ):
            raise ValueError("formal evaluation requires dataset and execution bindings")
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
        raise ValueError("evaluator execution environment mismatch")
    normalization: AISNormalizationBinding | None = None
    device = torch.device(args.device)
    split_path = resolve_repository_config_path(config["data"]["split_manifest"])
    splits, split_hash = load_split_snapshot(split_path)
    canonical_config_hash = _config_sha256(config)
    ais_checkpoint: AISEvaluationCheckpoint | None = None
    checkpoint_payload: dict[str, object] | None = None
    if is_b0:
        checkpoint_payload, checkpoint_hash = load_checkpoint_snapshot(args.checkpoint)
    else:
        ais_checkpoint = (
            load_legacy_raw_b1(args.checkpoint)
            if args.legacy_raw_b1_baseline
            else preflight_ais_evaluation_checkpoint(
                args.checkpoint,
                config,
                split_hash,
                legacy_raw_b1_baseline=False,
            )
        )
        if args.legacy_raw_b1_baseline:
            validate_legacy_raw_b1_config(config)
        checkpoint_hash = ais_checkpoint.checkpoint_sha256
        normalization = ais_checkpoint.normalization
    sample_ids = list(splits["val"])
    store = DenseCPUQueryStore(
        data_path,
        dataset_content_manifest=(
            args.dataset_content_manifest if dataset_binding is not None else None
        ),
    )
    representative: RepresentativeSelection | None = None
    if args.representative_manifest is not None:
        representative = load_representative_manifest(
            args.representative_manifest, store, sample_ids, split_hash
        )
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    if not sample_ids:
        raise ValueError("validation census selection is empty")

    receiver_manifest = _receiver_manifest(config, store, sample_ids[0])
    uses_normalization = not is_b0 and not args.legacy_raw_b1_baseline
    sampling = config["sampling"]
    if is_b0:
        if checkpoint_payload is None:
            raise RuntimeError("B0 checkpoint preflight was not completed")
        frozen_stats = json.loads(
            resolve_repository_config_path(
                config["normalization"]["stats_path"]
            ).read_text(encoding="utf-8")
        )
        if not isinstance(frozen_stats, dict):
            raise ValueError("frozen normalization stats must be a mapping")
        checkpoint_state, checkpoint_stats = validate_legacy_checkpoint_binding(
            checkpoint_payload, config, splits, frozen_stats
        )
        model = FactorizedAcousticFNO(**{
            key: value for key, value in config["model"].items() if key != "name"
        }).to(device)
        model.load_state_dict(checkpoint_state, strict=True)
        adapter = LegacyFullFieldPredictorAdapter(model, device, checkpoint_stats)
    else:
        if ais_checkpoint is None:
            raise RuntimeError("AIS checkpoint preflight was not completed")
        checkpoint_state = ais_checkpoint.model_state_dict
        model = AISMQFNO(**_model_kwargs(config, normalization)).to(device)
        model.load_state_dict(checkpoint_state, strict=True)
        adapter = (
            NormalizedQueryPredictorAdapter(
                model,
                device,
                chunk_sites,
                normalization,
                global_size=int(sampling.get("global_size", 100)),
            )
            if uses_normalization
            else QueryPredictorAdapter(
                model,
                device,
                chunk_sites,
                global_size=int(sampling.get("global_size", 100)),
            )
        )
    if args.verify_monolithic:
        scene = store.read_scene(sample_ids[0])
        chunked = adapter.predict_scene(scene).field_cpu
        if is_b0:
            monolithic = adapter.predict_scene(scene).field_cpu
        else:
            total = scene.target_cpu.shape[0] * scene.target_cpu.shape[1]
            monolithic_adapter = (
                NormalizedQueryPredictorAdapter(
                    model,
                    device,
                    total,
                    normalization,
                    global_size=adapter.global_size,
                )
                if uses_normalization
                else QueryPredictorAdapter(
                    model, device, total, global_size=adapter.global_size
                )
            )
            monolithic = monolithic_adapter.predict_scene(scene).field_cpu
        if not torch.allclose(chunked, monolithic, rtol=1e-5, atol=1e-6):
            raise ValueError("chunked prediction differs from monolithic prediction")

    provenance = {
        "seed": "b0" if is_b0 else ais_checkpoint.runtime_seed,
        "config_sha256": canonical_config_hash,
        "checkpoint_sha256": checkpoint_hash,
        "split_manifest_sha256": split_hash,
        "normalization_sha256": (
            normalization.stats_sha256
            if uses_normalization
            else normalization_hash(config)
        ),
        "receiver_geometry_sha256": receiver_manifest.sha256,
    }
    if args.evaluation_purpose == "screen64_fixed2048":
        if normalization is None or is_b0:
            raise RuntimeError("screen64 evaluation requires normalized AIS model")
        samples_csv, summary_json, sample_count = _run_screen64_census(
            model,
            store,
            sample_ids,
            splits,
            config,
            normalization,
            device,
            args.output_dir,
            provenance,
            args.screen_validation_site_manifest,
        )
        print(
            json.dumps(
                {
                    "purpose": "screen64_fixed2048",
                    "samples_csv": str(samples_csv),
                    "summary_json": str(summary_json),
                    "sample_count": sample_count,
                },
                sort_keys=True,
            )
        )
        return 0
    representative_ids = (
        sorted(set(representative.sample_ids).intersection(sample_ids))
        if representative is not None else None
    )
    evaluation_provenance = (
        {"representative_manifest_sha256": representative.sha256,
         "representative_metric_source_sha256": representative.metric_source_sha256,
         "representative_selection_rule": representative.selection_rule,
         "representative_seed": str(representative.seed)}
        if representative is not None else None
    )
    native_shape = tuple(store.read_scene(sample_ids[0]).target_cpu.shape)
    if (
        args.evaluation_purpose == "native400_final_census"
        and native_shape != (400, 400, 160)
    ):
        raise ValueError(
            "native400_final_census requires exact [400,400,160] validation scenes"
        )
    report = run_query_census(
        adapter, store, sample_ids, "val", args.output_dir,
        receiver_manifest, provenance, evaluation_provenance,
        representative_sample_ids=representative_ids,
        representative_provenance=(
            {"representative_manifest_sha256": representative.sha256}
            if representative is not None else None
        ),
        purpose=args.evaluation_purpose,
    )
    print(json.dumps({
        "samples_csv": str(report.samples_csv),
        "summary_json": str(report.summary_json),
        "sample_count": report.sample_count,
        "coverage_min": report.coverage_min,
        "coverage_max": report.coverage_max,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
