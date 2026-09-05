from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainConfig:
    """Physical extent and smooth hard-constraint scales."""

    lx_m: float
    lz_m: float
    t_end_s: float
    initial_gate_scale_s: float = 0.01
    surface_gate_scale_m: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "lx_m",
            "lz_m",
            "t_end_s",
            "initial_gate_scale_s",
            "surface_gate_scale_m",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters shared by the medium, source, and query encoders."""

    width: int = 96
    decoder_width: int = 256
    decoder_layers: int = 4
    attention_heads: int = 4
    token_grid_size: int = 12
    spectral_modes: tuple[int, int, int, int] = (24, 16, 12, 8)
    query_fourier_bands: int = 6
    reference_velocity_mps: float = 3000.0
    output_pressure_scale: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.decoder_width <= 0 or self.decoder_layers <= 0:
            raise ValueError("decoder dimensions must be positive")
        if self.attention_heads <= 0 or self.width % self.attention_heads != 0:
            raise ValueError("attention_heads must be positive and divide width")
        if self.token_grid_size <= 0:
            raise ValueError("token_grid_size must be positive")
        if len(self.spectral_modes) != 4 or any(mode <= 0 for mode in self.spectral_modes):
            raise ValueError("spectral_modes must contain four positive values")
        if self.reference_velocity_mps <= 0.0:
            raise ValueError("reference_velocity_mps must be positive")
        if self.output_pressure_scale <= 0.0:
            raise ValueError("output_pressure_scale must be positive")
        if self.query_fourier_bands <= 0:
            raise ValueError("query_fourier_bands must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 1000
    batch_size: int = 2
    time_frames: int = 8
    points_per_frame: int = 128
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    time_bins: int = 16
    spatial_ais_shape: tuple[int, int] = (12, 12)
    validate_every: int = 100
    checkpoint_every: int = 50
    validation_samples: int = 4
    stage1_steps: int = 500
    physics_ramp_steps: int = 500
    spectral_weight: float = 0.0
    pde_weight: float = 0.0
    pde_queries: int = 32
    pde_space_step_m: float = 10.0
    pde_time_step_s: float = 0.0025
    data_workers: int = 0
    prefetch_batches: int = 1
    seed: int = 2026

    def __post_init__(self) -> None:
        integer_fields = (
            "max_steps", "batch_size", "time_frames", "points_per_frame", "time_bins",
            "validate_every", "checkpoint_every", "validation_samples", "stage1_steps",
            "physics_ramp_steps", "pde_queries",
            "prefetch_batches",
        )
        if any(int(getattr(self, name)) <= 0 for name in integer_fields):
            raise ValueError("training counts must be positive")
        if min(self.spatial_ais_shape) <= 0:
            raise ValueError("spatial_ais_shape must be positive")
        if min(self.learning_rate, self.gradient_clip) <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings are invalid")
        if min(self.spectral_weight, self.pde_weight) < 0.0:
            raise ValueError("physics loss weights cannot be negative")
        if min(self.pde_space_step_m, self.pde_time_step_s) <= 0.0:
            raise ValueError("PDE stencil steps must be positive")
        if self.data_workers < 0:
            raise ValueError("data_workers cannot be negative")
