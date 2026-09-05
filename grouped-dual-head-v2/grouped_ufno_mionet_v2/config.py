"""Strict configuration schema for V2 experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


def _make(cls, values):
    values = dict(values or {})
    unknown = sorted(set(values) - set(cls.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    return cls(**values)


@dataclass
class ModelConfig:
    width: int = 64
    rank: int = 64
    modes: tuple[int, int, int, int] = (16, 12, 8, 8)
    dense_time_block: int = 8
    heads: int = 4


@dataclass
class DataConfig:
    train_cache: str = ""
    validation_cache: str = ""
    normalization_json: str = ""
    uniform_floor: float = 0.2
    pressure_percentile: float = 99.9


@dataclass
class LossConfig:
    query: float = 1.0
    dense: float = 1.0
    trace: float = 0.05
    consistency: float = 0.1
    gradient: float = 0.1
    spatial_fft: float = 0.05
    trace_fft: float = 0.05


@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    batch_records: int = 8
    learning_rate: float = 2e-4
    epochs: int = 1
    seed: int = 17
    max_steps: int = 1000
    validation_every: int = 25
    workers: int = 4
    weight_decay: float = 1e-4
    smoke: bool = False
    run_kind: str = "pilot"
    validation_batches: int = 8
    prefetch_factor: int = 4
    lbfgs_max_iter: int = 10
    lbfgs_history_size: int = 20
    lbfgs_line_search_fn: str = "strong_wolfe"


@dataclass
class V2Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]):
        values = dict(values or {})
        unknown = sorted(set(values) - {"model", "data", "loss", "train"})
        if unknown:
            raise ValueError(f"unknown V2Config keys: {unknown}")
        model = _make(ModelConfig, values.get("model")); model.modes = tuple(model.modes)
        data = _make(DataConfig, values.get("data")); loss = _make(LossConfig, values.get("loss")); train = _make(TrainConfig, values.get("train"))
        if model.dense_time_block <= 0 or data.uniform_floor < 0.2 or not 0 < data.pressure_percentile < 100:
            raise ValueError("invalid dense_time_block, uniform_floor, or pressure_percentile")
        if any(getattr(loss, name) < 0 for name in loss.__dataclass_fields__):
            raise ValueError("loss weights must be nonnegative")
        if train.optimizer not in {"adamw", "lbfgs"}:
            raise ValueError(f"unsupported optimizer: {train.optimizer}")
        if train.lbfgs_max_iter <= 0 or train.lbfgs_history_size <= 0:
            raise ValueError("LBFGS iteration and history sizes must be positive")
        if train.lbfgs_line_search_fn != "strong_wolfe":
            raise ValueError("LBFGS line search must be strong_wolfe")
        return cls(model, data, loss, train)

    @classmethod
    def from_yaml(cls, path: str | Path):
        import yaml
        with open(path, encoding="utf8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {})
