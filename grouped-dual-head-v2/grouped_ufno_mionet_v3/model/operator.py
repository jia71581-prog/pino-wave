"""Phase-aligned complex-FNO multi-input DeepONet point-query operator."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer

from .dense import DenseGridCache, TimeConditionedComplexDenseDecoder
from .features import PhaseAlignedCoordinateEncoder
from .fusion import PhaseAlignedMIONetFusion, TravelTimeBranch
from .medium import ComplexMediumEncoder, MediumEncoding
from .source import SourceEncoderV3, SourceEncoding
from .travel_time import RayTravelTime, dense_query_coordinates, straight_ray_travel_time


@dataclass(frozen=True)
class EncodedMediumState:
    velocity_mps: torch.Tensor
    encoding: MediumEncoding


@dataclass(frozen=True)
class PreparedV3State:
    medium: EncodedMediumState
    source_parameters: torch.Tensor
    source_normalized: torch.Tensor
    source_map: torch.Tensor
    source_encoding: SourceEncoding
    record_to_medium: torch.Tensor
    normalizer: PhysicalNormalizer


def free_surface_factor(z_m: torch.Tensor, *, transition_m: float = 20.0) -> torch.Tensor:
    if transition_m <= 0:
        raise ValueError("free-surface transition must be positive")
    return torch.tanh(torch.clamp_min(z_m, 0.0) / transition_m).square()


class PhaseAlignedComplexFNOMIONet(nn.Module):
    def __init__(
        self,
        *,
        width: int = 64,
        rank: int = 128,
        spectral_rank: int = 40,
        modes: tuple[int, ...] = (20, 16, 12, 8),
        dense_modes: tuple[int, ...] = (20, 16),
        dense_time_block: int = 4,
        heads: int = 4,
        token_grid: int = 8,
        position_bands: int = 4,
        fourier_bands: int = 6,
        gabor_scales_s: Sequence[float] = (0.01, 0.025, 0.05, 0.1),
        ray_samples: int = 12,
        domain_x_m: float = 2000.0,
        domain_z_m: float = 2000.0,
        domain_t_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.domain_x_m = float(domain_x_m)
        self.domain_z_m = float(domain_z_m)
        self.domain_t_s = float(domain_t_s)
        self.ray_samples = int(ray_samples)
        self.medium_encoder = ComplexMediumEncoder(
            width=width,
            rank=rank,
            spectral_rank=spectral_rank,
            modes=tuple(modes),
            token_grid=token_grid,
            position_bands=position_bands,
        )
        self.source_encoder = SourceEncoderV3(width=width, rank=rank)
        self.coordinate_encoder = PhaseAlignedCoordinateEncoder(
            width=width,
            rank=rank,
            fourier_bands=fourier_bands,
            gabor_scales_s=gabor_scales_s,
        )
        self.travel_branch = TravelTimeBranch(width=width, rank=rank)
        self.fusion = PhaseAlignedMIONetFusion(
            width=width,
            rank=rank,
            pyramid_levels=len(modes),
            heads=heads,
        )
        self.dense_decoder = TimeConditionedComplexDenseDecoder(
            width=width,
            spectral_rank=spectral_rank,
            pyramid_levels=len(modes),
            dense_modes=tuple(dense_modes),
            time_block=dense_time_block,
            domain_t_s=self.domain_t_s,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_medium(
        self,
        velocity_mps: torch.Tensor,
        normalizer: PhysicalNormalizer,
    ) -> EncodedMediumState:
        velocity = torch.as_tensor(velocity_mps, dtype=torch.float32, device=self.device)
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("velocity_mps must be [medium,1,z,x]")
        encoding = self.medium_encoder(normalizer.encode_velocity(velocity))
        return EncodedMediumState(velocity_mps=velocity, encoding=encoding)

    def _mapping(
        self,
        medium_count: int,
        record_count: int,
        record_to_medium: torch.Tensor | None,
    ) -> torch.Tensor:
        if record_to_medium is None:
            if medium_count == 1:
                return torch.zeros(record_count, dtype=torch.long, device=self.device)
            if medium_count == record_count:
                return torch.arange(record_count, dtype=torch.long, device=self.device)
            raise ValueError("record_to_medium is required for grouped source records")
        mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=self.device)
        if mapping.shape != (record_count,) or torch.any(mapping < 0) or torch.any(mapping >= medium_count):
            raise ValueError("record_to_medium contains invalid indices")
        return mapping

    def prepare_sources(
        self,
        medium: EncodedMediumState,
        source_parameters: torch.Tensor,
        source_map: torch.Tensor,
        normalizer: PhysicalNormalizer,
        *,
        record_to_medium: torch.Tensor | None = None,
    ) -> PreparedV3State:
        source = torch.as_tensor(source_parameters, dtype=torch.float32, device=self.device)
        source_map_tensor = torch.as_tensor(source_map, dtype=torch.float32, device=self.device)
        if source.ndim != 2 or source.shape[-1] != 5:
            raise ValueError("source_parameters must be [record,5]")
        mapping = self._mapping(
            medium.velocity_mps.shape[0],
            source.shape[0],
            record_to_medium,
        )
        source_normalized = normalizer.encode_source(source)
        source_encoding = self.source_encoder(
            source_normalized,
            source_map_tensor,
            medium.encoding,
            mapping,
        )
        return PreparedV3State(
            medium=medium,
            source_parameters=source,
            source_normalized=source_normalized,
            source_map=source_map_tensor,
            source_encoding=source_encoding,
            record_to_medium=mapping,
            normalizer=normalizer,
        )

    def _query_chunk(
        self,
        prepared: PreparedV3State,
        coords_xyz_m: torch.Tensor,
    ) -> torch.Tensor:
        coords = torch.as_tensor(coords_xyz_m, dtype=torch.float32, device=self.device)
        travel = straight_ray_travel_time(
            prepared.medium.velocity_mps,
            prepared.source_parameters[:, :2],
            coords[..., :2],
            x_extent_m=(0.0, self.domain_x_m),
            z_extent_m=(0.0, self.domain_z_m),
            record_to_medium=prepared.record_to_medium,
            samples=self.ray_samples,
        )
        trunk_rank, bundle = self.coordinate_encoder(
            coords,
            prepared.source_parameters,
            travel,
            domain_x_m=self.domain_x_m,
            domain_z_m=self.domain_z_m,
            domain_t_s=self.domain_t_s,
        )
        travel_rank = self.travel_branch(
            travel,
            bundle,
            domain_t_s=self.domain_t_s,
            domain_diagonal_m=math.sqrt(self.domain_x_m**2 + self.domain_z_m**2),
        )
        coords_normalized = torch.stack(
            (coords[..., 0] / self.domain_x_m, coords[..., 1] / self.domain_z_m),
            dim=-1,
        )
        fused = self.fusion(
            prepared.medium.encoding,
            prepared.source_encoding,
            coords_normalized,
            prepared.record_to_medium,
            travel_rank,
            trunk_rank,
        )
        return fused.normalized_pressure * free_surface_factor(coords[..., 1])

    def query_normalized(
        self,
        prepared: PreparedV3State,
        coords_xyz_m: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        coords = torch.as_tensor(coords_xyz_m, dtype=torch.float32, device=self.device)
        if coords.ndim != 3 or coords.shape[0] != prepared.source_parameters.shape[0] or coords.shape[-1] != 3:
            raise ValueError("coords_xyz_m must be [record,query,3]")
        step = coords.shape[1] if chunk_size is None else int(chunk_size)
        if step <= 0:
            raise ValueError("chunk_size must be positive")
        return torch.cat(
            [self._query_chunk(prepared, coords[:, start : start + step]) for start in range(0, coords.shape[1], step)],
            dim=1,
        )

    def query_pressure(
        self,
        prepared: PreparedV3State,
        coords_xyz_m: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        normalized = self.query_normalized(prepared, coords_xyz_m, chunk_size=chunk_size)
        return prepared.normalizer.decode_pressure(
            normalized.float(),
            prepared.source_parameters[:, 4],
        )

    def prepare_dense_grid(
        self,
        prepared: PreparedV3State,
        *,
        x_m: torch.Tensor,
        z_m: torch.Tensor,
        travel_time_s: torch.Tensor | None = None,
    ) -> DenseGridCache:
        x = torch.as_tensor(x_m, dtype=torch.float32, device=self.device)
        z = torch.as_tensor(z_m, dtype=torch.float32, device=self.device)
        full_size = prepared.medium.encoding.pyramid[0].shape[-2:]
        if x.ndim != 1 or z.ndim != 1 or (len(z), len(x)) != full_size:
            raise ValueError("dense x/z axes must match the encoded full-resolution grid")
        coords = dense_query_coordinates(x, z, records=prepared.source_parameters.shape[0])
        travel = straight_ray_travel_time(
            prepared.medium.velocity_mps,
            prepared.source_parameters[:, :2],
            coords,
            x_extent_m=(0.0, self.domain_x_m),
            z_extent_m=(0.0, self.domain_z_m),
            record_to_medium=prepared.record_to_medium,
            samples=self.ray_samples,
        )
        if travel_time_s is not None:
            cached = torch.as_tensor(
                travel_time_s, dtype=torch.float32, device=self.device
            )
            expected = (prepared.source_parameters.shape[0], len(z), len(x))
            if cached.shape != expected:
                raise ValueError(
                    f"cached dense travel time must have shape {expected}, got {tuple(cached.shape)}"
                )
            if not torch.isfinite(cached).all() or torch.any(cached < 0.0):
                raise ValueError("cached dense travel time must be finite and nonnegative")
            seconds = cached.flatten(1)
            distance = travel.distance_m
            positive_distance = distance > 1.0e-8
            positive_time = seconds > 1.0e-8
            mean_slowness = torch.where(
                positive_distance,
                seconds / distance.clamp_min(1.0e-8),
                travel.mean_slowness_s_per_m,
            )
            path_velocity = torch.where(
                positive_time,
                distance / seconds.clamp_min(1.0e-8),
                travel.endpoint_velocity_mps,
            )
            travel = RayTravelTime(
                seconds=seconds,
                distance_m=distance,
                path_velocity_mps=path_velocity,
                endpoint_velocity_mps=travel.endpoint_velocity_mps,
                mean_slowness_s_per_m=mean_slowness,
            )
        normalized = torch.stack(
            (coords[..., 0] / self.domain_x_m, coords[..., 1] / self.domain_z_m), dim=-1
        )
        return DenseGridCache(
            x_m=x,
            z_m=z,
            coords_xy_m=coords,
            coords_xy_normalized=normalized,
            travel=travel,
            height=len(z),
            width=len(x),
        )

    @staticmethod
    def _repeat_travel(travel: RayTravelTime, count: int) -> RayTravelTime:
        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value[:, None, :].expand(-1, count, -1).reshape(value.shape[0], -1)

        return RayTravelTime(
            seconds=repeat(travel.seconds),
            distance_m=repeat(travel.distance_m),
            path_velocity_mps=repeat(travel.path_velocity_mps),
            endpoint_velocity_mps=repeat(travel.endpoint_velocity_mps),
            mean_slowness_s_per_m=repeat(travel.mean_slowness_s_per_m),
        )

    def _dense_block(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        dense_grid: DenseGridCache,
        *,
        apply_correction: bool,
    ) -> torch.Tensor:
        records, count = time_s.shape
        points = dense_grid.height * dense_grid.width
        xy = dense_grid.coords_xy_m[:, None].expand(-1, count, -1, -1)
        time_values = time_s[:, :, None].expand(-1, -1, points)
        coords = torch.cat((xy, time_values[..., None]), dim=-1).reshape(records, count * points, 3)
        travel = self._repeat_travel(dense_grid.travel, count)
        trunk_rank, bundle = self.coordinate_encoder(
            coords,
            prepared.source_parameters,
            travel,
            domain_x_m=self.domain_x_m,
            domain_z_m=self.domain_z_m,
            domain_t_s=self.domain_t_s,
        )
        travel_rank = self.travel_branch(
            travel,
            bundle,
            domain_t_s=self.domain_t_s,
            domain_diagonal_m=math.sqrt(self.domain_x_m**2 + self.domain_z_m**2),
        )
        normalized_xy = dense_grid.coords_xy_normalized[:, None].expand(-1, count, -1, -1)
        fused = self.fusion(
            prepared.medium.encoding,
            prepared.source_encoding,
            normalized_xy.reshape(records, count * points, 2),
            prepared.record_to_medium,
            travel_rank,
            trunk_rank,
        )
        coarse = fused.normalized_pressure.reshape(
            records, count, dense_grid.height, dense_grid.width
        )
        if apply_correction:
            raw = self.dense_decoder(
                prepared.medium.encoding,
                prepared.source_encoding,
                prepared.source_parameters,
                prepared.record_to_medium,
                time_s,
                coarse,
            )
        else:
            raw = coarse
        surface = free_surface_factor(dense_grid.z_m)[None, None, :, None]
        return raw * surface

    def _expand_dense_times(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        times = torch.as_tensor(time_s, dtype=torch.float32, device=self.device)
        records = prepared.source_parameters.shape[0]
        if times.ndim == 1:
            times = times[None].expand(records, -1)
        if times.ndim != 2 or times.shape[0] != records or times.shape[1] == 0:
            raise ValueError("dense time_s must be [time] or [record,time]")
        if not torch.isfinite(times).all() or torch.any(times < 0) or torch.any(times > self.domain_t_s):
            raise ValueError("dense times lie outside the physical time domain")
        return times

    def iter_dense_normalized(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        *,
        x_m: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        dense_grid: DenseGridCache | None = None,
        time_block: int | None = None,
        apply_correction: bool = True,
    ):
        times = self._expand_dense_times(prepared, time_s)
        if dense_grid is None:
            if x_m is None or z_m is None:
                raise ValueError("x_m and z_m are required when dense_grid is not supplied")
            dense_grid = self.prepare_dense_grid(prepared, x_m=x_m, z_m=z_m)
        block = self.dense_decoder.time_block if time_block is None else int(time_block)
        if block <= 0:
            raise ValueError("time_block must be positive")
        for start in range(0, times.shape[1], block):
            yield start, self._dense_block(
                prepared,
                times[:, start : start + block],
                dense_grid,
                apply_correction=apply_correction,
            )

    def dense_normalized(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        *,
        x_m: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        dense_grid: DenseGridCache | None = None,
        time_block: int | None = None,
        apply_correction: bool = True,
    ) -> torch.Tensor:
        return torch.cat(
            [
                value
                for _, value in self.iter_dense_normalized(
                    prepared,
                    time_s,
                    x_m=x_m,
                    z_m=z_m,
                    dense_grid=dense_grid,
                    time_block=time_block,
                    apply_correction=apply_correction,
                )
            ],
            dim=1,
        )

    def predict_wavefield(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        **dense_kwargs,
    ) -> torch.Tensor:
        normalized = self.dense_normalized(prepared, time_s, **dense_kwargs)
        return prepared.normalizer.decode_pressure(
            normalized.float(),
            prepared.source_parameters[:, 4],
        )

    def required_gradient_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        groups: dict[str, tuple[nn.Parameter, ...]] = {}
        groups.update(self.medium_encoder.required_gradient_groups())
        groups.update(self.source_encoder.required_gradient_groups())
        groups["travel_branch"] = tuple(self.travel_branch.parameters())
        groups["periodic_trunk"] = tuple(self.coordinate_encoder.parameters())
        groups.update(self.fusion.required_gradient_groups())
        groups.update(self.dense_decoder.required_gradient_groups())
        return groups


__all__ = [
    "EncodedMediumState",
    "DenseGridCache",
    "PhaseAlignedComplexFNOMIONet",
    "PreparedV3State",
    "free_surface_factor",
]
