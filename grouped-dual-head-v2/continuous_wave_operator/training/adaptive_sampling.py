from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AISConfig:
    uniform_fraction: float = 0.25
    family_balanced_fraction: float = 0.25
    residual_fraction: float = 0.50
    ema_momentum: float = 0.20
    residual_warmup_epochs: int = 10
    minimum_probability: float = 1.0e-8
    max_importance_weight: float = 20.0

    def __post_init__(self) -> None:
        fractions = self.uniform_fraction + self.family_balanced_fraction + self.residual_fraction
        if abs(fractions - 1.0) > 1.0e-8:
            raise ValueError("AIS mixture fractions must sum to one")
        if not 0.0 < self.ema_momentum <= 1.0:
            raise ValueError("ema_momentum must be in (0,1]")
        if self.residual_warmup_epochs <= 0:
            raise ValueError("residual_warmup_epochs must be positive")


@dataclass(frozen=True)
class AISProbabilities:
    sample: torch.Tensor
    time: torch.Tensor
    spatial: torch.Tensor


class HierarchicalAIS:
    """Factorized sample/time/space residual EMA sampler kept on CPU."""

    def __init__(
        self,
        *,
        sample_count: int,
        time_bins: int,
        spatial_shape: tuple[int, int],
        seed: int,
        family_ids: torch.Tensor | None = None,
        config: AISConfig | None = None,
    ) -> None:
        if sample_count <= 0 or time_bins <= 0 or min(spatial_shape) <= 0:
            raise ValueError("AIS dimensions must be positive")
        self.sample_count = int(sample_count)
        self.time_bins = int(time_bins)
        self.spatial_shape = tuple(int(value) for value in spatial_shape)
        self.seed = int(seed)
        self.config = config or AISConfig()
        self.sample_ema = torch.ones(self.sample_count, dtype=torch.float32)
        self.time_ema = torch.ones(self.sample_count, self.time_bins, dtype=torch.float32)
        self.spatial_ema = torch.ones(
            self.sample_count, *self.spatial_shape, dtype=torch.float32
        )
        self.family_ids = (
            torch.zeros(self.sample_count, dtype=torch.long)
            if family_ids is None
            else torch.as_tensor(family_ids, dtype=torch.long, device="cpu").clone()
        )
        if self.family_ids.shape != (self.sample_count,):
            raise ValueError("family_ids must have shape [sample_count]")
        self.step = 0
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def _ema_update(self, tensor: torch.Tensor, index: tuple[Any, ...], value: float) -> None:
        momentum = self.config.ema_momentum
        tensor[index] = (1.0 - momentum) * tensor[index] + momentum * max(float(value), 0.0)

    def update(
        self,
        sample_indices: torch.Tensor,
        time_bins: torch.Tensor,
        spatial_cells: torch.Tensor,
        residuals: torch.Tensor,
    ) -> None:
        samples = torch.as_tensor(sample_indices, dtype=torch.long, device="cpu").reshape(-1)
        times = torch.as_tensor(time_bins, dtype=torch.long, device="cpu")
        cells = torch.as_tensor(spatial_cells, dtype=torch.long, device="cpu")
        values = torch.as_tensor(residuals, dtype=torch.float32, device="cpu")
        if times.ndim == 1:
            times = times[:, None]
        if cells.ndim == 2:
            cells = cells[:, None]
        values = values.reshape(samples.numel(), -1)
        if times.shape[:2] != values.shape or cells.shape[:2] != values.shape:
            raise ValueError("AIS update index and residual shapes differ")
        for row, sample in enumerate(samples.tolist()):
            if not 0 <= sample < self.sample_count:
                raise IndexError(sample)
            row_values = values[row]
            self._ema_update(self.sample_ema, (sample,), float(row_values.mean()))
            for column, residual in enumerate(row_values.tolist()):
                time_bin = int(times[row, column])
                z_cell, x_cell = (int(value) for value in cells[row, column].tolist())
                if not 0 <= time_bin < self.time_bins:
                    raise IndexError(time_bin)
                if not (0 <= z_cell < self.spatial_shape[0] and 0 <= x_cell < self.spatial_shape[1]):
                    raise IndexError((z_cell, x_cell))
                self._ema_update(self.time_ema, (sample, time_bin), residual)
                self._ema_update(self.spatial_ema, (sample, z_cell, x_cell), residual)
        self.step += 1

    def _residual_power(self, epoch: int) -> float:
        return min(max(float(epoch) / self.config.residual_warmup_epochs, 0.0), 1.0)

    def _normalized(self, values: torch.Tensor, dimensions: int | tuple[int, ...]) -> torch.Tensor:
        denominator = values.sum(dim=dimensions, keepdim=True).clamp_min(
            self.config.minimum_probability
        )
        return values / denominator

    def _family_distribution(self) -> torch.Tensor:
        result = torch.zeros(self.sample_count, dtype=torch.float32)
        families = torch.unique(self.family_ids)
        for family in families:
            mask = self.family_ids == family
            result[mask] = 1.0 / (families.numel() * int(mask.sum()))
        return result

    def probabilities(self, *, epoch: int) -> AISProbabilities:
        power = self._residual_power(epoch)
        sample_residual = self._normalized(self.sample_ema.clamp_min(1.0e-12).pow(power), 0)
        sample_uniform = torch.full_like(sample_residual, 1.0 / self.sample_count)
        sample = (
            self.config.uniform_fraction * sample_uniform
            + self.config.family_balanced_fraction * self._family_distribution()
            + self.config.residual_fraction * sample_residual
        )
        time_residual = self._normalized(self.time_ema.clamp_min(1.0e-12).pow(power), 1)
        time_uniform = torch.full_like(time_residual, 1.0 / self.time_bins)
        time = (
            (self.config.uniform_fraction + self.config.family_balanced_fraction) * time_uniform
            + self.config.residual_fraction * time_residual
        )
        spatial_residual = self._normalized(
            self.spatial_ema.clamp_min(1.0e-12).pow(power), (1, 2)
        )
        cell_count = self.spatial_shape[0] * self.spatial_shape[1]
        spatial_uniform = torch.full_like(spatial_residual, 1.0 / cell_count)
        spatial = (
            (self.config.uniform_fraction + self.config.family_balanced_fraction) * spatial_uniform
            + self.config.residual_fraction * spatial_residual
        )
        floor = self.config.minimum_probability
        return AISProbabilities(
            sample=self._normalized(sample.clamp_min(floor), 0),
            time=self._normalized(time.clamp_min(floor), 1),
            spatial=self._normalized(spatial.clamp_min(floor), (1, 2)),
        )

    def inverse_probability_weights(self, probabilities: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(probabilities, dtype=torch.float32)
        inverse = values.clamp_min(self.config.minimum_probability).reciprocal()
        inverse = inverse.clamp(max=self.config.max_importance_weight)
        return inverse / inverse.mean().clamp_min(1.0e-12)

    def sample_indices(self, count: int, *, epoch: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("count must be positive")
        return torch.multinomial(
            self.probabilities(epoch=epoch).sample,
            count,
            replacement=True,
            generator=self.generator,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "time_bins": self.time_bins,
            "spatial_shape": self.spatial_shape,
            "seed": self.seed,
            "config": asdict(self.config),
            "family_ids": self.family_ids.clone(),
            "sample_ema": self.sample_ema.clone(),
            "time_ema": self.time_ema.clone(),
            "spatial_ema": self.spatial_ema.clone(),
            "step": self.step,
            "generator_state": self.generator.get_state(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "HierarchicalAIS":
        result = cls(
            sample_count=int(state["sample_count"]),
            time_bins=int(state["time_bins"]),
            spatial_shape=tuple(state["spatial_shape"]),
            seed=int(state["seed"]),
            family_ids=state["family_ids"],
            config=AISConfig(**state["config"]),
        )
        result.sample_ema.copy_(state["sample_ema"])
        result.time_ema.copy_(state["time_ema"])
        result.spatial_ema.copy_(state["spatial_ema"])
        result.step = int(state["step"])
        result.generator.set_state(state["generator_state"].cpu())
        return result
