from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _strict_int(value: object, name: str) -> int:
    dtype = getattr(value, "dtype", None)
    if isinstance(value, bool) or str(dtype) in {"bool", "torch.bool"}:
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _strict_sample_id(value: object, name: str = "sample_id") -> int:
    sample_id = _strict_int(value, name)
    if sample_id < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return sample_id


@dataclass(frozen=True)
class SpatialDraw:
    sample_ids: torch.Tensor
    site_indices: torch.Tensor
    draw_probability: torch.Tensor
    component: torch.Tensor
    ema_version: int
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class SpatialSamplingFeatures:
    uniform: torch.Tensor
    interface: torch.Tensor
    source_wavefront: torch.Tensor
    physical_edge: torch.Tensor
    receiver_neighborhood: torch.Tensor


class AdaptiveSpatialSampler:
    """Per-scene mixture sampler for complete spatial-site trajectories."""

    def __init__(
        self,
        height: int,
        width: int,
        mixture: Sequence[float],
        seed: int,
        tile_size: int = 16,
        ema_momentum: float = 0.2,
    ) -> None:
        self.mixture = torch.as_tensor(
            mixture, dtype=torch.float64, device="cpu"
        ).clone()
        unit = torch.tensor(1.0, dtype=torch.float64)
        if (
            self.mixture.shape != (6,)
            or not bool(torch.isfinite(self.mixture).all())
            or bool(torch.any(self.mixture < 0))
            or not bool(torch.isclose(self.mixture.sum(), unit))
        ):
            raise ValueError("mixture requires six nonnegative weights summing to one")
        height = _strict_int(height, "height")
        width = _strict_int(width, "width")
        tile_size = _strict_int(tile_size, "tile_size")
        seed = _strict_int(seed, "seed")
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0.0 <= float(ema_momentum) <= 1.0:
            raise ValueError("ema_momentum must be between zero and one")

        self.height, self.width = height, width
        self.tile_size, self.ema_momentum = tile_size, float(ema_momentum)
        self.seed, self.ema_version = seed, 0
        self.residual_tiles: dict[int, torch.Tensor] = {}
        self.generators: dict[int, torch.Generator] = {}
        self.pending_updates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    @property
    def _tile_shape(self) -> tuple[int, int]:
        return (
            math.ceil(self.height / self.tile_size),
            math.ceil(self.width / self.tile_size),
        )

    def generator_for(self, sample_id: int) -> torch.Generator:
        sample_id = _strict_sample_id(sample_id)
        if sample_id not in self.generators:
            seed = self.seed + 1_000_003 * sample_id
            self.generators[sample_id] = torch.Generator().manual_seed(seed)
        return self.generators[sample_id]

    def residual_distribution(self, sample_id: int) -> torch.Tensor:
        sample_id = _strict_sample_id(sample_id)
        tile_h, tile_w = self._tile_shape
        default = torch.ones(tile_h, tile_w, dtype=torch.float64)
        tiles = self.residual_tiles.get(sample_id, default).to(torch.float64)
        x_tiles = torch.div(
            torch.arange(self.height), self.tile_size, rounding_mode="floor"
        )
        z_tiles = torch.div(
            torch.arange(self.width), self.tile_size, rounding_mode="floor"
        )
        dense = tiles[x_tiles[:, None], z_tiles[None, :]]
        return dense.reshape(-1).clamp_min(1e-12)

    def apply_pending_tile_ema(
        self,
        updates: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        tile_h, tile_w = self._tile_shape
        observations: dict[tuple[int, int], list[torch.Tensor]] = {}
        for sample_ids, site_indices, squared_error in updates:
            for row, sample_id in enumerate(sample_ids.tolist()):
                x_idx = torch.div(site_indices[row], self.width, rounding_mode="floor")
                z_idx = site_indices[row].remainder(self.width)
                tile_idx = (
                    torch.div(x_idx, self.tile_size, rounding_mode="floor") * tile_w
                    + torch.div(z_idx, self.tile_size, rounding_mode="floor")
                )
                for index in torch.unique(tile_idx):
                    key = (sample_id, int(index.item()))
                    observations.setdefault(key, []).append(
                        squared_error[row][tile_idx == index].double()
                    )

        by_sample: dict[int, torch.Tensor] = {}
        for (sample_id, tile_index), chunks in observations.items():
            current = by_sample.setdefault(
                sample_id,
                self.residual_tiles.get(
                    sample_id, torch.ones(tile_h, tile_w, dtype=torch.float64)
                ).clone(),
            )
            # Sorting makes pooling independent of pending-update partition and order.
            observed = torch.cat(chunks).sort().values.mean()
            flat = current.reshape(-1)
            flat[tile_index] = (
                (1.0 - self.ema_momentum) * flat[tile_index]
                + self.ema_momentum * observed
            )
        self.residual_tiles.update(by_sample)

    def _component_distributions(
        self, sample_id: int, features: SpatialSamplingFeatures
    ) -> torch.Tensor:
        sample_id = _strict_sample_id(sample_id)
        raw_feature_values = (
            features.uniform,
            features.interface,
            features.source_wavefront,
            features.physical_edge,
            features.receiver_neighborhood,
        )
        expected_shape = (self.height * self.width,)
        feature_values: list[torch.Tensor] = []
        for value in raw_feature_values:
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError("feature vectors must match the flattened spatial grid")
            normalized = value.detach().to(dtype=torch.float64, device="cpu")
            if not bool(torch.isfinite(normalized).all()) or bool(torch.any(normalized < 0)):
                raise ValueError("feature vectors must be finite and nonnegative")
            feature_values.append(normalized)

        components = torch.stack(
            (
                feature_values[0],
                feature_values[1],
                feature_values[2],
                self.residual_distribution(sample_id),
                feature_values[3],
                feature_values[4],
            )
        )
        row_sums = components.sum(dim=1, keepdim=True)
        normalized = components / row_sums.clamp_min(torch.finfo(torch.float64).tiny)
        uniform = torch.full_like(components, 1.0 / (self.height * self.width))
        return torch.where(row_sums > 0, normalized, uniform)

    def proposal(
        self, sample_id: int, features: SpatialSamplingFeatures
    ) -> torch.Tensor:
        sample_id = _strict_sample_id(sample_id)
        components = self._component_distributions(sample_id, features)
        proposal = torch.einsum("k,kn->n", self.mixture, components)
        return proposal / proposal.sum()

    def draw(
        self,
        sample_ids: Sequence[int],
        features: Sequence[SpatialSamplingFeatures],
        count: int,
    ) -> SpatialDraw:
        if len(sample_ids) == 0:
            raise ValueError("draw requires a nonempty batch")
        if len(sample_ids) != len(features):
            raise ValueError("sample_ids and features must have equal lengths")
        count = _strict_int(count, "count")
        normalized_sample_ids = tuple(
            _strict_sample_id(sample_id) for sample_id in sample_ids
        )
        if count <= 0:
            raise ValueError("count must be positive")

        all_indices: list[torch.Tensor] = []
        all_probabilities: list[torch.Tensor] = []
        all_components: list[torch.Tensor] = []
        for sample_id, item in zip(normalized_sample_ids, features, strict=True):
            generator = self.generator_for(sample_id)
            components = self._component_distributions(sample_id, item)
            component_ids = torch.multinomial(
                self.mixture, count, replacement=True, generator=generator
            )
            indices = torch.empty(count, dtype=torch.long)
            for component_id in torch.unique(component_ids).tolist():
                positions = torch.nonzero(
                    component_ids == component_id, as_tuple=False
                ).flatten()
                indices[positions] = torch.multinomial(
                    components[component_id],
                    positions.numel(),
                    replacement=True,
                    generator=generator,
                )
            proposal = torch.einsum("k,kn->n", self.mixture, components)
            proposal = proposal / proposal.sum()
            all_indices.append(indices)
            all_probabilities.append(proposal.index_select(0, indices))
            all_components.append(component_ids)

        sample_ids_tensor = torch.tensor(normalized_sample_ids, dtype=torch.long)
        site_indices = torch.stack(all_indices)
        probabilities = torch.stack(all_probabilities)
        inverse = probabilities.reciprocal()

        # A site is unique within a physical scene, not globally by flat index.
        scene_site_pairs = torch.stack(
            (
                sample_ids_tensor[:, None].expand_as(site_indices).reshape(-1),
                site_indices.reshape(-1),
            ),
            dim=1,
        )
        unique_count = torch.unique(scene_site_pairs, dim=0).shape[0]
        total_draws = site_indices.numel()
        tile_h, tile_w = self._tile_shape
        x_idx = torch.div(site_indices, self.width, rounding_mode="floor")
        z_idx = site_indices.remainder(self.width)
        tile_indices = (
            torch.div(x_idx, self.tile_size, rounding_mode="floor") * tile_w
            + torch.div(z_idx, self.tile_size, rounding_mode="floor")
        )
        scene_tile_pairs = torch.stack(
            (
                sample_ids_tensor[:, None].expand_as(tile_indices).reshape(-1),
                tile_indices.reshape(-1),
            ),
            dim=1,
        )
        covered_tiles = torch.unique(scene_tile_pairs, dim=0).shape[0]
        total_tiles = torch.unique(sample_ids_tensor).numel() * tile_h * tile_w
        diagnostics = {
            "ess": float((inverse.sum().square() / inverse.square().sum()).item()),
            "duplicate_fraction": float(1.0 - unique_count / total_draws),
            # Coverage is scene-tile coverage, matching the residual EMA grid.
            "coverage": float(covered_tiles / total_tiles),
            "max_median_inverse_weight": float(
                (inverse.max() / inverse.median()).item()
            ),
        }
        return SpatialDraw(
            sample_ids_tensor,
            site_indices,
            probabilities,
            torch.stack(all_components),
            self.ema_version,
            diagnostics,
        )

    def update_residual_tiles(
        self,
        sample_ids: torch.Tensor,
        site_indices: torch.Tensor,
        squared_error: torch.Tensor,
    ) -> None:
        update = self._validate_residual_update(
            sample_ids, site_indices, squared_error
        )
        self.pending_updates.append(update)

    def _validate_residual_update(
        self,
        sample_ids: object,
        site_indices: object,
        squared_error: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (sample_ids, site_indices, squared_error)
        ):
            raise TypeError("residual updates must contain tensors")
        if sample_ids.ndim != 1 or site_indices.ndim != 2 or squared_error.ndim != 2:
            raise ValueError("residual updates require [B], [B,Q], and [B,Q] tensors")
        if sample_ids.dtype not in _INTEGER_DTYPES or site_indices.dtype not in _INTEGER_DTYPES:
            raise TypeError("sample_ids and site_indices must have integer tensor dtypes")
        if not squared_error.is_floating_point():
            raise TypeError("squared residuals must have a floating tensor dtype")
        if site_indices.shape != squared_error.shape or site_indices.shape[0] != sample_ids.numel():
            raise ValueError("residual update shapes do not agree")
        if site_indices.numel() == 0:
            raise ValueError("residual updates must be nonempty")
        if bool(torch.any(site_indices < 0)) or bool(
            torch.any(site_indices >= self.height * self.width)
        ):
            raise ValueError("residual site indices are outside the spatial grid")
        if not bool(torch.isfinite(squared_error).all()) or bool(
            torch.any(squared_error < 0)
        ):
            raise ValueError("squared residuals must be finite and nonnegative")
        if bool(torch.any(sample_ids < 0)):
            raise ValueError("sample_ids must be nonnegative integers")
        return (
            sample_ids.detach().to(dtype=torch.long, device="cpu").clone(),
            site_indices.detach().to(dtype=torch.long, device="cpu").clone(),
            squared_error.detach().to(dtype=torch.float64, device="cpu").clone(),
        )

    def commit_epoch(self) -> None:
        self.apply_pending_tile_ema(self.pending_updates)
        self.pending_updates.clear()
        self.ema_version += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "height": self.height,
            "width": self.width,
            "tile_size": self.tile_size,
            "ema_momentum": self.ema_momentum,
            "seed": self.seed,
            "mixture": self.mixture.clone(),
            "ema_version": self.ema_version,
            "residual_tiles": {
                key: value.clone() for key, value in self.residual_tiles.items()
            },
            "generator_states": {
                key: value.get_state().clone() for key, value in self.generators.items()
            },
            "pending_updates": [
                (a.clone(), b.clone(), c.clone())
                for a, b, c in self.pending_updates
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("sampler checkpoint must be a mapping")
        required = {
            "height",
            "width",
            "tile_size",
            "ema_momentum",
            "seed",
            "mixture",
            "ema_version",
            "residual_tiles",
            "generator_states",
            "pending_updates",
        }
        if not required.issubset(state):
            raise ValueError("sampler checkpoint is missing required fields")

        raw_momentum = state["ema_momentum"]
        if isinstance(raw_momentum, bool):
            raise TypeError("state ema_momentum must be a real number")
        try:
            restored_momentum = float(raw_momentum)
        except (TypeError, ValueError) as error:
            raise TypeError("state ema_momentum must be a real number") from error
        if not math.isfinite(restored_momentum) or not 0 <= restored_momentum <= 1:
            raise ValueError("state ema_momentum must be finite and between zero and one")
        restored_config = (
            _strict_int(state["height"], "state height"),
            _strict_int(state["width"], "state width"),
            _strict_int(state["tile_size"], "state tile_size"),
            restored_momentum,
        )
        current_config = (
            self.height,
            self.width,
            self.tile_size,
            self.ema_momentum,
        )
        if restored_config != current_config:
            raise ValueError("sampler configuration differs from checkpoint")
        if not isinstance(state["mixture"], torch.Tensor):
            raise TypeError("state mixture must be a tensor")
        restored_mixture = state["mixture"].detach().to(
            dtype=torch.float64, device="cpu"
        ).clone()
        if (
            restored_mixture.shape != (6,)
            or not bool(torch.isfinite(restored_mixture).all())
            or bool(torch.any(restored_mixture < 0))
            or not bool(
                torch.isclose(
                    restored_mixture.sum(), torch.tensor(1.0, dtype=torch.float64)
                )
            )
        ):
            raise ValueError("invalid sampler mixture in checkpoint")
        if not torch.equal(restored_mixture, self.mixture):
            raise ValueError("sampler mixture differs from checkpoint")

        restored_seed = _strict_int(state["seed"], "state seed")
        restored_version = _strict_int(state["ema_version"], "state ema_version")
        if restored_version < 0:
            raise ValueError("state ema_version must be nonnegative")
        residual_tiles = state["residual_tiles"]
        pending_updates = state["pending_updates"]
        generator_states = state["generator_states"]
        if not isinstance(residual_tiles, Mapping) or not isinstance(generator_states, Mapping):
            raise ValueError("invalid sampler state mappings")
        if not isinstance(pending_updates, (list, tuple)):
            raise ValueError("invalid pending sampler updates")

        tile_shape = self._tile_shape
        restored_tiles: dict[int, torch.Tensor] = {}
        for key, value in residual_tiles.items():
            sample_id = _strict_sample_id(key, "residual sample_id")
            if sample_id in restored_tiles:
                raise ValueError("invalid residual sample_id in checkpoint")
            if not isinstance(value, torch.Tensor):
                raise TypeError("residual tile state must contain tensors")
            tile = value.detach().to(dtype=torch.float64, device="cpu").clone()
            if tuple(tile.shape) != tile_shape:
                raise ValueError("residual tile shape differs from sampler configuration")
            if not bool(torch.isfinite(tile).all()) or bool(torch.any(tile < 0)):
                raise ValueError("residual tiles must be finite and nonnegative")
            restored_tiles[sample_id] = tile

        restored_pending: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for update in pending_updates:
            if not isinstance(update, (list, tuple)) or len(update) != 3:
                raise ValueError("invalid pending sampler update tuple")
            restored_pending.append(self._validate_residual_update(*update))

        restored_generators: dict[int, torch.Generator] = {}
        for key, generator_state in generator_states.items():
            sample_id = _strict_sample_id(key, "generator sample_id")
            if sample_id in restored_generators:
                raise ValueError("invalid generator sample_id in checkpoint")
            if (
                not isinstance(generator_state, torch.Tensor)
                or generator_state.dtype != torch.uint8
                or generator_state.ndim != 1
                or generator_state.device.type != "cpu"
            ):
                raise ValueError("invalid generator state tensor")
            generator = torch.Generator()
            generator.set_state(generator_state.clone())
            restored_generators[sample_id] = generator

        # Transaction boundary: do not mutate the object until every field is valid.
        self.seed = restored_seed
        self.ema_version = restored_version
        self.residual_tiles = restored_tiles
        self.pending_updates = restored_pending
        self.generators = restored_generators
