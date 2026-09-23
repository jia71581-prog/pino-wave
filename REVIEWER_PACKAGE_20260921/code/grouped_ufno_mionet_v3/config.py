"""Strict immutable configuration schema for V3 experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TypeVar


ALLOWED_MEDIUM_TYPES = ("uniform", "layered", "marmousi")
_T = TypeVar("_T")


def _make(cls: type[_T], values: Mapping[str, Any] | None) -> _T:
    payload = dict(values or {})
    unknown = sorted(set(payload) - {item.name for item in fields(cls)})
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    return cls(**payload)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    spectral_rank: int = 40
    modes: tuple[int, ...] = (20, 16, 12, 8)
    dense_modes: tuple[int, ...] = (20, 16)
    mionet_rank: int = 128
    token_count: int = 64
    heads: int = 4
    position_bands: int = 4
    fourier_bands: int = 6
    gabor_scales_s: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1)
    ray_samples: int = 12
    dense_time_block: int = 4
    dense_equivariant: bool = False
    dense_radial_cutoff: int = 0
    ic_frames: int = 0
    domain_x_m: float = 2000.0
    domain_z_m: float = 2000.0
    domain_t_s: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    source_h5: str = "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5"
    manifest_json: str = "/home/jiayh/Data/data/processed/grouped_v3_manifest.json"
    normalization_json: str = "/home/jiayh/Data/data/processed/grouped_v3_normalization.json"
    train_cache: str = "/home/jiayh/Data/data/processed/grouped_dual_head_v2_train.h5"
    validation_cache: str = "/home/jiayh/Data/data/processed/grouped_dual_head_v2_validation.h5"
    allowed_medium_types: tuple[str, ...] = ALLOWED_MEDIUM_TYPES
    train_families: tuple[str, ...] = ALLOWED_MEDIUM_TYPES
    expected_train_records: int = 2240
    expected_validation_records: int = 480
    continuous_fraction: float = 0.125
    gate_sample_id: str = ""
    gate_sample_ids: tuple[str, ...] = ()
    ic_frames: int = 0
    dense_time_steps: int = 3
    active_horizon_s: float = 0.60
    time_sampling_policy: str = "legacy"
    # Residual-adaptive sampling.  Off by default so every pre-existing config
    # keeps its exact behaviour and digest; the adaptive arm turns these on.
    residual_adaptive_space: bool = False
    residual_adaptive_time: bool = False
    residual_ema_gamma: float = 0.95
    query_uniform_floor: float = 0.2
    # Activation checkpointing on the dense spectral blocks.  DDP-safe through
    # the non-reentrant implementation; off was a workaround for the reentrant
    # "parameters marked ready twice" abort, which no longer applies.
    dense_checkpoint: bool = True
    # Split the dense supervision window into chunks of this many frames and
    # accumulate gradients across them.  The dense loss is a sum over frames, so
    # the accumulated gradient equals the single-pass one up to floating-point
    # summation order, while peak activation memory scales with the chunk rather
    # than the whole window.  0 disables chunking (single pass over all frames).
    dense_chunk_frames: int = 0
    # Also checkpoint the per-time-step dense call, not just the blocks inside
    # it.  Without this, peak memory grows by roughly 0.7 GiB per supervised
    # frame because every time step of the window keeps its activations live;
    # with it, peak is set by a single time step.  Costs recomputation on the
    # backward pass, which is what makes a 51-frame window fit in 24 GiB.
    dense_outer_checkpoint: bool = False


@dataclass(frozen=True)
class LossConfig:
    point: float = 1.0
    frame: float = 1.0
    complex_spectrum: float = 0.15
    spectral_phase: float = 0.1
    spatial_gradient: float = 0.1
    time_difference: float = 0.1
    consistency: float = 0.05
    phase_energy_fraction: float = 1.0e-4
    relative_energy_floor_fraction: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str = "adamw"
    batch_records: int = 1
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    epochs: int = 1
    max_steps: int = 2
    evaluation_every: int = 1
    steps_per_epoch: int = 1
    query_points_per_step: int = 512
    workers: int = 0
    prefetch_factor: int = 2
    seed: int = 17
    smoke: bool = True
    device: str = "cuda"
    micro_batch_records: int = 1
    checkpoint_dir: str = "artifacts/grouped_ufno_mionet_v3/smoke/checkpoints"
    gradient_clip: float = 1.0
    plateau_patience: int = 20
    plateau_min_delta: float = 1.0e-4
    lbfgs_learning_rate: float = 0.5
    lbfgs_max_iter: int = 10
    lbfgs_history_size: int = 20


@dataclass(frozen=True)
class V3Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "V3Config":
        payload = dict(values or {})
        unknown = sorted(set(payload) - {"model", "data", "loss", "train"})
        if unknown:
            raise ValueError(f"unknown V3Config keys: {unknown}")

        model_values = dict(payload.get("model") or {})
        if "modes" in model_values:
            model_values["modes"] = tuple(int(v) for v in model_values["modes"])
        if "dense_modes" in model_values:
            model_values["dense_modes"] = tuple(int(v) for v in model_values["dense_modes"])
        if "gabor_scales_s" in model_values:
            model_values["gabor_scales_s"] = tuple(float(v) for v in model_values["gabor_scales_s"])
        model = _make(ModelConfig, model_values)

        data_values = dict(payload.get("data") or {})
        if "allowed_medium_types" in data_values:
            requested = tuple(str(v) for v in data_values["allowed_medium_types"])
            if requested != ALLOWED_MEDIUM_TYPES:
                raise ValueError(
                    "allowed_medium_types is fixed to uniform, layered, marmousi; "
                    f"received {requested}"
                )
            data_values["allowed_medium_types"] = requested
        if "train_families" in data_values:
            requested = tuple(str(v) for v in data_values["train_families"])
            unknown = set(requested) - set(ALLOWED_MEDIUM_TYPES)
            if not requested or unknown:
                raise ValueError(
                    "train_families must be a non-empty subset of uniform/layered/marmousi; "
                    f"received {requested}"
                )
            data_values["train_families"] = requested
        if "gate_sample_ids" in data_values:
            data_values["gate_sample_ids"] = tuple(
                str(value) for value in data_values["gate_sample_ids"]
            )
        data = _make(DataConfig, data_values)
        loss = _make(LossConfig, payload.get("loss"))
        train = _make(TrainConfig, payload.get("train"))
        cls._validate(model, data, loss, train)
        return cls(model=model, data=data, loss=loss, train=train)

    @staticmethod
    def _validate(
        model: ModelConfig,
        data: DataConfig,
        loss: LossConfig,
        train: TrainConfig,
    ) -> None:
        if model.width <= 0 or model.spectral_rank <= 0 or model.mionet_rank <= 0:
            raise ValueError("model widths and ranks must be positive")
        if (
            not model.modes
            or not model.dense_modes
            or any(mode <= 0 for mode in (*model.modes, *model.dense_modes))
        ):
            raise ValueError("spectral modes must be a non-empty positive sequence")
        if model.ray_samples < 2 or model.dense_time_block <= 0:
            raise ValueError("ray_samples and dense_time_block are invalid")
        if model.dense_radial_cutoff < 0:
            raise ValueError("dense_radial_cutoff must be nonnegative")
        if model.dense_equivariant and model.dense_radial_cutoff <= 0:
            raise ValueError(
                "dense_equivariant requires a positive dense_radial_cutoff"
            )
        if data.allowed_medium_types != ALLOWED_MEDIUM_TYPES:
            raise ValueError("allowed_medium_types violates the V3 data contract")
        if (
            not data.train_families
            or set(data.train_families) - set(ALLOWED_MEDIUM_TYPES)
        ):
            raise ValueError(
                "train_families must be a non-empty subset of uniform/layered/marmousi"
            )
        if not 0.0 <= data.continuous_fraction < 1.0:
            raise ValueError("continuous_fraction must be in [0,1)")
        if model.ic_frames < 0 or data.ic_frames < 0:
            raise ValueError("ic_frames must be nonnegative")
        if model.ic_frames != data.ic_frames:
            raise ValueError("model.ic_frames and data.ic_frames must agree")
        if data.dense_time_steps < 1:
            raise ValueError("dense_time_steps must be at least 1")
        if data.dense_chunk_frames < 0:
            raise ValueError("dense_chunk_frames must be nonnegative")
        if data.active_horizon_s <= 0:
            raise ValueError("active_horizon_s must be positive")
        if data.time_sampling_policy not in ("legacy", "full_support", "residual_adaptive"):
            raise ValueError(
                "time_sampling_policy must be legacy, full_support or residual_adaptive"
            )
        if data.residual_adaptive_time and data.time_sampling_policy != "residual_adaptive":
            raise ValueError(
                "residual_adaptive_time requires time_sampling_policy=residual_adaptive"
            )
        if not 0.0 < data.residual_ema_gamma <= 1.0:
            raise ValueError("residual_ema_gamma must lie in (0,1]")
        if not 0.0 <= data.query_uniform_floor <= 1.0:
            raise ValueError("query_uniform_floor must lie in [0,1]")
        if any(getattr(loss, item.name) < 0 for item in fields(loss)):
            raise ValueError("loss values must be nonnegative")
        if loss.relative_energy_floor_fraction > 1.0:
            raise ValueError("relative_energy_floor_fraction must be in [0,1]")
        if train.optimizer not in {"adamw", "lbfgs"}:
            raise ValueError(f"unsupported optimizer: {train.optimizer}")
        if (
            train.batch_records <= 0
            or train.max_steps <= 0
            or train.gradient_clip <= 0
            or train.evaluation_every <= 0
            or train.steps_per_epoch <= 0
            or train.query_points_per_step <= 0
            or train.micro_batch_records <= 0
        ):
            raise ValueError("training batch, micro-batch, and step counts must be positive")
        if train.micro_batch_records > train.batch_records:
            raise ValueError("micro_batch_records must not exceed batch_records")
        if train.device not in {"cpu", "cuda"}:
            raise ValueError("training device must be cpu or cuda")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "V3Config":
        import yaml

        with Path(path).open(encoding="utf8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {})

    def digest(self) -> str:
        payload = asdict(self)
        # Fields added after the v3 checkpoints were written are omitted while
        # they hold their legacy default, so existing runs keep their digest and
        # stay resumable.  Any non-default value must change the digest.
        if payload["data"].get("time_sampling_policy") == "legacy":
            payload["data"].pop("time_sampling_policy", None)
        # Residual-adaptive sampling was added later still; its defaults are
        # off, so strip them while inert and let an enabled arm change the
        # digest (which is what marks it as a new experiment identity).
        for name, default in (
            ("residual_adaptive_space", False),
            ("residual_adaptive_time", False),
            ("residual_ema_gamma", 0.95),
            ("query_uniform_floor", 0.2),
            ("dense_checkpoint", True),
            ("dense_chunk_frames", 0),
            ("dense_outer_checkpoint", False),
        ):
            if payload["data"].get(name) == default:
                payload["data"].pop(name, None)
        # The equivariant radial dense kernel was added later still; strip its
        # inert defaults so pre-existing checkpoints keep their digest.
        for name, default in (
            ("dense_equivariant", False),
            ("dense_radial_cutoff", 0),
        ):
            if payload["model"].get(name) == default:
                payload["model"].pop(name, None)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf8")
        return hashlib.sha256(encoded).hexdigest()
