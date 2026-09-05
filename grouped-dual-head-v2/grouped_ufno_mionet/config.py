"""Configuration objects for the grouped operator.

Dataclasses make the tensor and output-grid contract explicit and keep model
construction reproducible.  ``from_mapping`` rejects unknown keys so a typo
cannot silently change a production run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _strict(cls, value: Mapping[str, Any] | None):
    value = dict(value or {})
    names = set(cls.__dataclass_fields__)
    unknown = sorted(set(value) - names)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    return cls(**value)


@dataclass
class ModelConfig:
    width: int = 96
    rank: int = 64
    fourier_modes: tuple[int, int, int, int] = (24, 16, 12, 8)
    heads: int = 4
    domain_x_m: float = 2000.0
    domain_z_m: float = 2000.0
    output_nx: int = 201
    output_nz: int = 201
    source_feature_dim: int = 64
    token_grid: int = 8

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        obj = _strict(cls, value)
        obj.fourier_modes = tuple(int(v) for v in obj.fourier_modes)
        if obj.width < 8 or obj.rank <= 0 or len(obj.fourier_modes) != 4:
            raise ValueError("invalid model width/rank/fourier_modes")
        return obj


@dataclass
class DataConfig:
    frames_per_record: int = 32
    query_blocks: int = 4
    receivers_per_frame: int = 512
    max_records_per_macro_batch: int = 24
    output_times: int = 401
    spatial_shape: tuple[int, int] = (201, 201)
    dataset: str = "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        obj = _strict(cls, value)
        obj.spatial_shape = tuple(int(v) for v in obj.spatial_shape)
        if obj.spatial_shape != (201, 201):
            raise ValueError("the grouped operator currently requires a 201x201 output grid")
        return obj


@dataclass
class TrainConfig:
    batch_records: int = 24
    query_chunk_size: int = 4096
    learning_rate: float = 2e-4
    epochs: int = 1
    max_steps: int = 1000
    num_workers: int = 8
    prefetch_batches: int = 4
    seed: int = 17

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        obj = _strict(cls, value)
        if obj.batch_records <= 0 or obj.query_chunk_size <= 0:
            raise ValueError("batch_records and query_chunk_size must be positive")
        return obj


@dataclass
class OperatorConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        value = dict(value or {})
        unknown = sorted(set(value) - {"model", "data", "train"})
        if unknown:
            raise ValueError(f"unknown OperatorConfig keys: {unknown}")
        return cls(
            model=ModelConfig.from_mapping(value.get("model")),
            data=DataConfig.from_mapping(value.get("data")),
            train=TrainConfig.from_mapping(value.get("train")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path):
        import yaml
        with open(path, "r", encoding="utf8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {})
