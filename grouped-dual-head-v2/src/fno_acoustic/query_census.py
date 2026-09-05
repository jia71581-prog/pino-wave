"""Streamed native-grid census evaluation with atomic bound reports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import string
import tempfile
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import nn

from .long_horizon_metrics import ReceiverSiteManifest, long_horizon_metrics
from .normalization import decode_standard, encode_standard
from .query_data import DenseCPUQueryStore, QueryScene
from .query_reconstruction import QueryInferenceScene, reconstruct_native
from .query_training import build_scene_model_inputs


PRIMARY_COLUMNS = (
    "relative_l2", "relative_l2_q4",
    "receiver_relative_l2", "receiver_relative_l2_q4",
)
AIS_CATEGORIES = ("uniform", "layered", "marmousi")
SAMPLE_COLUMNS = (
    "sample_id", "category", "split", "seed", "predictor_family",
    "config_sha256", "checkpoint_sha256", "split_manifest_sha256",
    "normalization_sha256", "receiver_geometry_sha256",
    "relative_l2", "relative_l2_q1", "relative_l2_q2", "relative_l2_q3", "relative_l2_q4",
    "receiver_relative_l2", "receiver_relative_l2_q1", "receiver_relative_l2_q2",
    "receiver_relative_l2_q3", "receiver_relative_l2_q4",
    "field_q4_q1_ratio", "receiver_q4_q1_ratio",
    "active_time_coverage", "active_time_error_slope_per_s", "active_time_error_max",
    "arrival_mae_s", "arrival_miss_rate", "arrival_target_coverage",
    "receiver_lag_abs_s", "receiver_xcorr_peak", "receiver_phase_error",
    "receiver_phase_coherence", "energy_log_ratio", "komega_relative_l2",
    "komega_relative_l2_q4", "komega_high", "komega_high_q4",
    "zero_relative_l2", "zero_relative_l2_q4",
    "zero_receiver_relative_l2", "zero_receiver_relative_l2_q4",
    "prediction_target_norm_ratio", "prediction_target_pearson",
    "sampler_ess", "sampler_duplicate_fraction", "sampler_coverage",
    "sampler_max_median_inverse_weight",
    "prediction_finite", "prediction_nonzero",
    "output_height", "output_width", "output_time_steps",
)
REQUIRED_NUMERIC_COLUMNS = SAMPLE_COLUMNS[10:]
SCREEN_NUMERIC_COLUMNS = (
    "relative_l2",
    "relative_l2_q4",
    "prediction_target_norm_ratio",
    "zero_relative_l2",
    "zero_relative_l2_q4",
)
SCREEN_SAMPLE_COLUMNS = (
    "sample_id",
    "category",
    "split",
    "predictor_family",
    "config_sha256",
    "checkpoint_sha256",
    "split_manifest_sha256",
    "normalization_sha256",
    "validation_site_manifest_sha256",
    *SCREEN_NUMERIC_COLUMNS,
)
_PROVENANCE_HASHES = (
    "config_sha256", "checkpoint_sha256", "split_manifest_sha256",
    "normalization_sha256", "receiver_geometry_sha256",
)


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def validate_provenance(
    provenance: Mapping[str, object], receiver_manifest: ReceiverSiteManifest
) -> dict[str, str]:
    required = {"seed", *_PROVENANCE_HASHES}
    if not isinstance(provenance, Mapping) or set(provenance) != required:
        raise ValueError(f"provenance must contain exactly {sorted(required)}")
    seed = provenance["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (str, int)) or not str(seed):
        raise ValueError("provenance seed must be a nonempty string or integer")
    result = {"seed": str(seed)}
    for name in _PROVENANCE_HASHES:
        result[name] = _sha256(provenance[name], f"provenance {name} SHA-256")
    if result["receiver_geometry_sha256"] != receiver_manifest.sha256:
        raise ValueError("provenance receiver geometry SHA-256 does not match the manifest")
    return result


@dataclass(frozen=True)
class NativePrediction:
    field_cpu: torch.Tensor
    coverage_min: int
    coverage_max: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.field_cpu, torch.Tensor)
            or self.field_cpu.ndim != 4
            or self.field_cpu.shape[0] != 1
            or self.field_cpu.shape[-1] != 160
            or not self.field_cpu.is_floating_point()
            or self.field_cpu.device.type != "cpu"
        ):
            raise ValueError("native prediction must be a CPU floating [1,H,W,160] tensor")
        if not bool(torch.isfinite(self.field_cpu).all()):
            raise ValueError("native prediction must contain only finite values")
        for value, name in ((self.coverage_min, "coverage_min"), (self.coverage_max, "coverage_max")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.coverage_min > self.coverage_max:
            raise ValueError("coverage_min must not exceed coverage_max")


class NativePredictor(Protocol):
    predictor_family: str

    def predict_scene(self, scene: QueryScene) -> NativePrediction:
        ...


@dataclass(frozen=True)
class QueryPredictorAdapter:
    model: nn.Module
    device: torch.device
    chunk_sites: int
    global_size: int = 100
    predictor_family: str = "ais_mqfno"

    def __post_init__(self) -> None:
        _positive_integer(self.chunk_sites, "chunk_sites")
        _positive_integer(self.global_size, "global_size")
        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device")

    def predict_scene(self, scene: QueryScene) -> NativePrediction:
        height, width = scene.target_cpu.shape[:2]
        inputs = build_scene_model_inputs(
            scene, min(self.global_size, height, width), self.device
        )
        inference = QueryInferenceScene(
            inputs.global_inputs, inputs.native_static, inputs.time_s,
            height, width, 1, self.device,
        )
        self.model.eval()
        result = reconstruct_native(self.model, inference, self.chunk_sites)
        if not isinstance(result.field, torch.Tensor):
            raise TypeError("in-memory query reconstruction must return a tensor")
        return NativePrediction(
            result.field.float().cpu(), int(result.coverage.min()), int(result.coverage.max())
        )


def _normalization_stats(value: Mapping[str, object]) -> dict[str, object]:
    required = {"velocity", "wavefield", "eps"}
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError("normalization_stats require velocity, wavefield, and eps")
    eps = value["eps"]
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("normalization eps must be positive and finite")
    result: dict[str, object] = {"eps": float(eps)}
    for name in ("velocity", "wavefield"):
        stats = value[name]
        if not isinstance(stats, Mapping) or not {"mean", "std"}.issubset(stats):
            raise ValueError(f"normalization {name} requires mean and std")
        mean, std = stats["mean"], stats["std"]
        if (
            isinstance(mean, bool) or isinstance(std, bool)
            or not isinstance(mean, (int, float)) or not isinstance(std, (int, float))
            or not math.isfinite(float(mean)) or not math.isfinite(float(std))
            or float(std) <= 0
        ):
            raise ValueError(f"normalization {name} mean/std must be finite with std > 0")
        result[name] = {"mean": float(mean), "std": float(std)}
    return result


def build_legacy_dense_input(
    scene: QueryScene,
    normalization_stats: Mapping[str, object],
    input_features: Sequence[str] | None = None,
) -> torch.Tensor:
    """Build a canonical ``[1,H,W,160,C]`` legacy full-field input."""

    if scene.target_cpu.ndim != 3 or scene.target_cpu.shape[-1] != 160:
        raise ValueError("legacy dense input requires a [H,W,160] scene")
    if min(scene.target_cpu.shape[:2]) < 3:
        raise ValueError("legacy dense input requires at least a 3 by 3 grid")
    stats = _normalization_stats(normalization_stats)
    velocity = scene.velocity_cpu[None, None].float()
    velocity_feature = encode_standard(
        velocity, stats["velocity"], float(stats["eps"])
    )
    source = scene.source_cpu[None, None].float()
    grad_x, grad_z = torch.gradient(
        velocity, spacing=(scene.x_m.float(), scene.z_m.float()), dim=(2, 3), edge_order=2
    )
    slow2 = velocity.clamp_min(1.0).reciprocal().square()
    time = scene.time_s
    if time.shape != (160,) or not bool(torch.all(torch.diff(time) > 0)):
        raise ValueError("legacy dense input requires 160 increasing times")
    tau = ((time - time[0]) / (time[-1] - time[0])).float()
    height, width = scene.target_cpu.shape[:2]
    pool = {
        "time": tau[None, None, None].expand(1, height, width, -1),
        "source_map": source[:, 0, :, :, None].expand(-1, -1, -1, 160),
        "source_mask": source[:, 0, :, :, None].expand(-1, -1, -1, 160),
        "velocity": velocity_feature[:, 0, :, :, None].expand(-1, -1, -1, 160),
        "velocity_gradient_x": grad_x[:, 0, :, :, None].expand(-1, -1, -1, 160),
        "velocity_gradient_z": grad_z[:, 0, :, :, None].expand(-1, -1, -1, 160),
        "slowness_squared": slow2[:, 0, :, :, None].expand(-1, -1, -1, 160),
    }
    names = tuple(input_features or ("time", "source_map", "velocity"))
    if not names or len(set(names)) != len(names) or any(name not in pool for name in names):
        raise ValueError("legacy input_features contain unsupported or duplicate names")
    return torch.stack([pool[name] for name in names], dim=-1).contiguous()


@dataclass(frozen=True)
class LegacyFullFieldPredictorAdapter:
    model: nn.Module
    device: torch.device
    normalization_stats: Mapping[str, object]
    predictor_family: str = "factorized_fno_b0"
    input_features: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device")
        object.__setattr__(self, "normalization_stats", _normalization_stats(self.normalization_stats))

    @torch.no_grad()
    def predict_scene(self, scene: QueryScene) -> NativePrediction:
        self.model.eval()
        features = self.input_features
        if features is None and getattr(self.model, "in_features", 3) == 6:
            features = (
                "velocity", "source_map", "velocity_gradient_x",
                "velocity_gradient_z", "slowness_squared", "time",
            )
        dense_input = build_legacy_dense_input(
            scene, self.normalization_stats, features
        ).to(self.device)
        prediction = self.model(dense_input)
        if prediction.ndim == 5 and prediction.shape[-1] == 1:
            prediction = prediction[..., 0]
        expected = (1, *scene.target_cpu.shape)
        if not isinstance(prediction, torch.Tensor) or prediction.shape != expected:
            raise ValueError(f"legacy predictor must return {expected}")
        prediction_physical = decode_standard(
            prediction,
            self.normalization_stats["wavefield"],
            float(self.normalization_stats["eps"]),
        )
        return NativePrediction(prediction_physical.detach().float().cpu(), 1, 1)


@dataclass(frozen=True)
class CensusReport:
    samples_csv: Path
    summary_json: Path
    coverage_min: int
    coverage_max: int
    sample_count: int


def _validate_row(row: Mapping[str, object]) -> None:
    if not isinstance(row, Mapping) or set(row) != set(SAMPLE_COLUMNS):
        raise ValueError("census row columns must equal SAMPLE_COLUMNS exactly")
    sample_id = row["sample_id"]
    if not isinstance(sample_id, int) or isinstance(sample_id, bool) or sample_id < 0:
        raise ValueError("sample_id must be a nonnegative integer")
    for name in ("category", "split", "seed", "predictor_family"):
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError(f"{name} must be a nonempty string")
    for name in _PROVENANCE_HASHES:
        _sha256(row[name], name)
    for name in REQUIRED_NUMERIC_COLUMNS:
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"numeric census column {name} must be a real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"numeric census column {name} must be finite")


def _category(value: object) -> str:
    if not isinstance(value, str) or value not in AIS_CATEGORIES:
        raise ValueError(
            "scene metadata model_type category must be one of "
            f"{AIS_CATEGORIES}"
        )
    return value


def native_enumeration_sampler_diagnostics(
    height: int, width: int, *, unique_evaluated_sites: int | None = None
) -> dict[str, float]:
    """Describe deterministic native-grid enumeration in sampler-compatible fields."""

    height = _positive_integer(height, "native enumeration height")
    width = _positive_integer(width, "native enumeration width")
    total = height * width
    unique = total if unique_evaluated_sites is None else _positive_integer(
        unique_evaluated_sites, "unique evaluated sites"
    )
    if unique > total:
        raise ValueError("unique evaluated sites cannot exceed the native grid")
    return {
        "sampler_ess": float(unique),
        "sampler_duplicate_fraction": 0.0,
        "sampler_coverage": float(unique / total),
        "sampler_max_median_inverse_weight": 1.0,
    }


def aggregate_category_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Average already-computed physical-scene metrics globally and by category."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("category aggregation requires physical sample rows")
    groups: dict[str, list[Mapping[str, object]]] = {"global": []}
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or "category" not in row:
            raise ValueError("category aggregation requires a metadata category")
        _validate_row(row)
        sample_id = int(row["sample_id"])
        if sample_id in seen:
            raise ValueError("category aggregation requires unique physical sample IDs")
        seen.add(sample_id)
        category = _category(row["category"])
        groups["global"].append(row)
        groups.setdefault(category, []).append(row)

    result: dict[str, dict[str, float | int]] = {}
    for name, group in groups.items():
        result[name] = {
            "sample_count": len(group),
            **{
                metric: sum(float(row[metric]) for row in group) / len(group)
                for metric in REQUIRED_NUMERIC_COLUMNS
            },
        }
    return result


def aggregate_screen_category_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate strict per-scene sparse-64 screen metrics by category."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("screen category aggregation requires sample rows")
    groups: dict[str, list[Mapping[str, object]]] = {"global": []}
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(SCREEN_SAMPLE_COLUMNS):
            raise ValueError("screen sample row schema differs")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, int) or isinstance(sample_id, bool) or sample_id < 0:
            raise ValueError("screen sample_id must be a nonnegative integer")
        if sample_id in seen:
            raise ValueError("screen sample rows require unique sample IDs")
        seen.add(sample_id)
        category = _category(row["category"])
        if row["split"] != "val" or row["predictor_family"] != "ais_mqfno":
            raise ValueError("screen sample metadata differs")
        for name in (
            "config_sha256", "checkpoint_sha256", "split_manifest_sha256",
            "normalization_sha256", "validation_site_manifest_sha256",
        ):
            _sha256(row[name], name)
        for name in SCREEN_NUMERIC_COLUMNS:
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"screen {name} must be finite numeric")
        groups["global"].append(row)
        groups.setdefault(category, []).append(row)
    return {
        name: {
            "sample_count": len(group),
            **{
                metric: sum(float(row[metric]) for row in group) / len(group)
                for metric in SCREEN_NUMERIC_COLUMNS
            },
        }
        for name, group in groups.items()
    }


def write_rows_atomically(path: str | Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path = Path(path)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("census rows must be a nonempty sequence")
    for row in rows:
        _validate_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", prefix=f".{path.name}.",
            suffix=".tmp", dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=SAMPLE_COLUMNS, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row[name] for name in SAMPLE_COLUMNS})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return path


def _write_prediction_atomically(path: Path, field_cpu: torch.Tensor) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.save(stream, field_cpu.numpy(), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return path


def _atomic_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return path


def write_census_summary(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    split_name: str,
    provenance: Mapping[str, str],
    evaluation_provenance: Mapping[str, str] | None = None,
    purpose: str = "native_full_census",
    binding_summary: Mapping[str, object] | None = None,
) -> Path:
    if not rows:
        raise ValueError("summary requires at least one physical sample row")
    aggregated = aggregate_category_metrics(rows)
    categories = {
        category: {
            "sample_count": values["sample_count"],
            "mean": {
                name: values[name] for name in REQUIRED_NUMERIC_COLUMNS
            },
        }
        for category, values in aggregated.items()
        if category != "global"
    }
    payload = {
        "schema_version": 1,
        "purpose": purpose,
        "split": split_name,
        "sample_count": len(rows),
        "predictor_family": rows[0]["predictor_family"],
        "provenance": dict(provenance),
        "mean": {
            name: aggregated["global"][name] for name in REQUIRED_NUMERIC_COLUMNS
        },
        "categories": categories,
        "category_metrics": aggregated,
    }
    if evaluation_provenance is not None:
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in evaluation_provenance.items()):
            raise ValueError("evaluation provenance must map strings to strings")
        payload["evaluation_provenance"] = dict(evaluation_provenance)
    if binding_summary is not None:
        required = {
            "dataset_content_root",
            "verified_sample_count",
            "verified_sample_ids_sha256",
            "verified_sample_scope",
        }
        if set(binding_summary) != required:
            raise ValueError("dataset binding summary key set is invalid")
        payload.update(binding_summary)
    return _atomic_json(Path(path), payload)


def run_query_census(
    predictor: NativePredictor,
    store: DenseCPUQueryStore,
    sample_ids: Sequence[int],
    split_name: str,
    output_dir: Path,
    receiver_manifest: ReceiverSiteManifest,
    provenance: Mapping[str, object],
    evaluation_provenance: Mapping[str, str] | None = None,
    representative_sample_ids: Sequence[int] | None = None,
    representative_provenance: Mapping[str, str] | None = None,
    purpose: str = "native_full_census",
    binding_summary: Mapping[str, object] | None = None,
) -> CensusReport:
    if purpose not in {"native_full_census", "native400_final_census"}:
        raise ValueError("native census purpose is invalid")
    if split_name not in {"val", "test"}:
        raise ValueError("census split must be val or test")
    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence) or not sample_ids:
        raise ValueError("sample_ids must be a nonempty sequence")
    normalized_ids: list[int] = []
    for sample_id in sample_ids:
        if not isinstance(sample_id, int) or isinstance(sample_id, bool) or sample_id < 0:
            raise ValueError("sample IDs must be nonnegative integers")
        normalized_ids.append(sample_id)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("sample IDs must be unique physical scenes")
    if not isinstance(getattr(predictor, "predictor_family", None), str) or not predictor.predictor_family:
        raise ValueError("predictor_family must be a nonempty string")
    bound = validate_provenance(provenance, receiver_manifest)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"census output directory already exists: {output_dir}")
    representative_ids = set(representative_sample_ids or ())
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in representative_ids):
        raise ValueError("representative sample IDs must be nonnegative integers")
    if not representative_ids.issubset(normalized_ids):
        raise ValueError("representative sample IDs must belong to the census selection")
    if representative_ids:
        if not isinstance(representative_provenance, Mapping) or not representative_provenance:
            raise ValueError("representative predictions require provenance")
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in representative_provenance.items()):
            raise ValueError("representative provenance must map strings to strings")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent
    ))
    try:
        rows: list[dict[str, object]] = []
        coverage_min, coverage_max = 1, 1
        for sample_id in normalized_ids:
            scene = store.read_scene(sample_id)
            if scene.sample_id != sample_id:
                raise ValueError("store returned a scene with the wrong sample ID")
            if tuple(scene.target_cpu.shape[:2]) != (
                receiver_manifest.height, receiver_manifest.width
            ):
                raise ValueError("scene grid differs from the bound receiver manifest")
            receiver_x = torch.div(
                receiver_manifest.site_indices,
                receiver_manifest.width,
                rounding_mode="floor",
            )
            receiver_z = receiver_manifest.site_indices.remainder(receiver_manifest.width)
            scene_receiver_xz = torch.stack(
                (scene.x_m[receiver_x], scene.z_m[receiver_z]), dim=-1
            ).to(receiver_manifest.physical_xz)
            if not torch.equal(scene_receiver_xz, receiver_manifest.physical_xz):
                raise ValueError("scene coordinates differ from the bound receiver manifest")
            native = predictor.predict_scene(scene)
            if native.field_cpu.shape != (1, *scene.target_cpu.shape):
                raise ValueError("native predictor output shape differs from the physical scene")
            if (native.coverage_min, native.coverage_max) != (1, 1):
                raise ValueError(f"native census coverage failed for sample {sample_id}")
            coverage_min = min(coverage_min, native.coverage_min)
            coverage_max = max(coverage_max, native.coverage_max)
            target = scene.target_cpu[None].to(dtype=native.field_cpu.dtype)
            metrics = long_horizon_metrics(
                native.field_cpu, target, scene.time_s, receiver_manifest
            )
            metrics.update(
                native_enumeration_sampler_diagnostics(
                    native.field_cpu.shape[1], native.field_cpu.shape[2]
                )
            )
            category = _category(scene.metadata.get("model_type"))
            row: dict[str, object] = {
                "sample_id": sample_id, "category": category, "split": split_name,
                "seed": bound["seed"], "predictor_family": predictor.predictor_family,
                **{name: bound[name] for name in _PROVENANCE_HASHES}, **metrics,
            }
            _validate_row(row)
            rows.append(row)
            if sample_id in representative_ids:
                _write_prediction_atomically(
                    stage / "predictions" / f"sample_{sample_id}.npy",
                    native.field_cpu,
                )
            del native
        if representative_ids:
            _atomic_json(stage / "predictions" / "provenance.json", {
                **dict(representative_provenance or {}),
                "sample_ids": sorted(representative_ids),
            })
        write_rows_atomically(stage / "samples.csv", rows)
        if binding_summary is None and getattr(store, "dataset_binding", None) is not None:
            binding_summary = store.binding_summary()
        write_census_summary(
            stage / "summary.json", rows, split_name, bound,
            evaluation_provenance, purpose, binding_summary,
        )
        os.replace(stage, output_dir)
        stage = None
        return CensusReport(
            output_dir / "samples.csv", output_dir / "summary.json",
            coverage_min, coverage_max, len(rows),
        )
    except BaseException:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "AIS_CATEGORIES", "CensusReport", "LegacyFullFieldPredictorAdapter", "NativePrediction",
    "NativePredictor", "PRIMARY_COLUMNS", "QueryPredictorAdapter",
    "REQUIRED_NUMERIC_COLUMNS", "SAMPLE_COLUMNS", "SCREEN_NUMERIC_COLUMNS",
    "SCREEN_SAMPLE_COLUMNS", "build_legacy_dense_input",
    "aggregate_category_metrics", "aggregate_screen_category_metrics",
    "native_enumeration_sampler_diagnostics",
    "run_query_census", "validate_provenance", "write_census_summary",
    "write_rows_atomically",
]
