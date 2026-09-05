"""Deterministic grouped full-data batches with exact/interpolated target prefetch."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES

from .batch import pack_v3_groups
from .index import V3DataManifest, assert_allowed_families
from .records import V3WavefieldDataset


@dataclass(frozen=True)
class PilotStepSpec:
    step: int
    record_indices: tuple[int, ...]


@dataclass(frozen=True)
class PilotBatch:
    step: int
    velocity_mps: torch.Tensor
    record_to_medium: torch.Tensor
    source_parameters: torch.Tensor
    source_map: torch.Tensor
    requested_time_s: torch.Tensor
    dense_target_physical: torch.Tensor
    target_exact: torch.Tensor
    interpolation_alpha: torch.Tensor
    left_index: torch.Tensor
    right_index: torch.Tensor
    query_coords: torch.Tensor
    query_target_physical: torch.Tensor
    query_probability: torch.Tensor
    x_m: torch.Tensor
    z_m: torch.Tensor
    sample_id: tuple[str, ...]
    group_id: tuple[str, ...]
    medium_type: tuple[str, ...]
    dense_travel_time_s: torch.Tensor | None = None

    def pin_memory(self) -> "PilotBatch":
        values: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.pin_memory() if isinstance(value, torch.Tensor) else value
        return PilotBatch(**values)


def _cyclic_window(values: Sequence[object], start: int, count: int) -> list[object]:
    if len(values) < count:
        raise ValueError("pilot schedule does not have enough unique records")
    return [values[(start + offset) % len(values)] for offset in range(count)]


def build_pilot_schedule(
    manifest: V3DataManifest,
    *,
    split: str,
    steps: int,
    seed: int,
) -> tuple[PilotStepSpec, ...]:
    if steps <= 0:
        raise ValueError("pilot schedule steps must be positive")
    records = tuple(record for record in manifest.records if record.split == split)
    if not records:
        raise ValueError(f"pilot split is empty: {split}")
    assert_allowed_families(record.medium_type for record in records)
    groups: dict[str, dict[str, list[int]]] = {
        family: defaultdict(list) for family in ALLOWED_MEDIUM_TYPES
    }
    for local_index, record in enumerate(records):
        groups[record.medium_type][record.group_id].append(local_index)
    uniform = list(groups["uniform"].values())
    layered = list(groups["layered"].values())
    marmousi = list(groups["marmousi"].values())
    if any(len(group) != 1 for group in uniform):
        raise ValueError("pilot expects one source per uniform medium group")
    if any(len(group) != 4 for group in layered):
        raise ValueError("pilot expects four sources per layered medium group")
    if any(len(group) != 5 for group in marmousi):
        raise ValueError("pilot expects five sources per marmousi medium group")
    if len(uniform) < 4 or not layered or not marmousi:
        raise ValueError("pilot schedule cannot balance all three medium families")
    generator = np.random.default_rng(int(seed))
    generator.shuffle(uniform)
    generator.shuffle(layered)
    generator.shuffle(marmousi)
    uniform_indices = [group[0] for group in uniform]
    schedule: list[PilotStepSpec] = []
    for step in range(int(steps)):
        selected_uniform = _cyclic_window(uniform_indices, step * 4, 4)
        selected_layered = layered[step % len(layered)]
        selected_marmousi_group = marmousi[step % len(marmousi)]
        omitted = (step // len(marmousi)) % len(selected_marmousi_group)
        selected_marmousi = [
            index for position, index in enumerate(selected_marmousi_group) if position != omitted
        ]
        selected = tuple(
            int(value)
            for value in (*selected_uniform, *selected_layered, *selected_marmousi)
        )
        if len(selected) != 12 or len(set(selected)) != 12:
            raise RuntimeError("pilot schedule failed its unique balanced record contract")
        schedule.append(PilotStepSpec(step=step, record_indices=selected))
    return tuple(schedule)


def _requested_times(
    time_s: torch.Tensor,
    *,
    source_t0_s: float,
    continuous: torch.Tensor,
    phase_offset: int,
    active_horizon_s: float = 0.60,
) -> torch.Tensor:
    axis = time_s.double()
    onset = min(int(torch.searchsorted(axis, torch.tensor(source_t0_s, dtype=axis.dtype))), len(axis) - 1)
    end_time = min(float(axis[-1]), float(source_t0_s) + float(active_horizon_s))
    end = min(int(torch.searchsorted(axis, torch.tensor(end_time, dtype=axis.dtype))), len(axis) - 1)
    candidates = np.arange(onset, end + 1, dtype=np.int64)
    phases = np.array_split(candidates, 3)
    if any(len(phase) == 0 for phase in phases):
        raise ValueError("pilot active horizon cannot form early/middle/late phases")
    exact_indices_list: list[int] = []
    for phase_index, phase in enumerate(phases):
        trim = len(phase) // 5
        interior = phase[trim : len(phase) - trim] if trim else phase
        position = (int(phase_offset) + phase_index * 11) % len(interior)
        exact_indices_list.append(int(interior[position]))
    exact_indices = np.asarray(exact_indices_list, dtype=np.int64)
    requested = axis[torch.from_numpy(exact_indices)].clone()
    for position in torch.nonzero(continuous, as_tuple=True)[0].tolist():
        index = int(exact_indices[position])
        left = index if index < len(axis) - 1 else index - 1
        requested[position] = 0.5 * (axis[left] + axis[left + 1])
    return requested.float()


def _sample_queries(
    target: torch.Tensor,
    requested_time_s: torch.Tensor,
    x_m: torch.Tensor,
    z_m: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames, nz, nx = target.shape
    flat_target = target.reshape(-1)
    weights = flat_target.abs().float() + torch.finfo(torch.float32).eps
    weights /= weights.sum()
    probability = 0.2 / len(weights) + 0.8 * weights
    draws = min(int(count), len(probability))
    indices = torch.multinomial(probability, draws, replacement=False, generator=generator)
    time_index = torch.div(indices, nz * nx, rounding_mode="floor")
    spatial = indices.remainder(nz * nx)
    z_index = torch.div(spatial, nx, rounding_mode="floor")
    x_index = spatial.remainder(nx)
    coords = torch.stack(
        (x_m[x_index], z_m[z_index], requested_time_s[time_index]), dim=-1
    ).float()
    return coords, flat_target[indices].float(), probability[indices].float()


class PilotBatchDataset(Dataset[PilotBatch]):
    def __init__(
        self,
        source_h5: str | Path,
        manifest: V3DataManifest,
        *,
        split: str,
        schedule: Sequence[PilotStepSpec],
        continuous_fraction: float,
        query_points: int,
        seed: int,
    ) -> None:
        if not schedule or not 0.0 <= continuous_fraction <= 1.0 or query_points <= 0:
            raise ValueError("pilot batch dataset configuration is invalid")
        self.records = V3WavefieldDataset(source_h5, manifest, split=split)
        self.schedule = tuple(schedule)
        self.continuous_fraction = float(continuous_fraction)
        self.query_points = int(query_points)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> PilotBatch:
        return self.materialize_spec(self.schedule[int(index)])

    def materialize_spec(self, spec: PilotStepSpec) -> PilotBatch:
        records = [self.records[record_index] for record_index in spec.record_indices]
        macro = pack_v3_groups(records)
        total_targets = len(records) * 3
        continuous_count = int(round(total_targets * self.continuous_fraction))
        mask_generator = torch.Generator().manual_seed(self.seed + spec.step * 104729)
        continuous_flat = torch.zeros(total_targets, dtype=torch.bool)
        if continuous_count:
            chosen = torch.randperm(total_targets, generator=mask_generator)[:continuous_count]
            continuous_flat[chosen] = True
        continuous_mask = continuous_flat.reshape(len(records), 3)

        requested_times: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        exact: list[torch.Tensor] = []
        alpha: list[torch.Tensor] = []
        left: list[torch.Tensor] = []
        right: list[torch.Tensor] = []
        query_coords: list[torch.Tensor] = []
        query_targets: list[torch.Tensor] = []
        query_probability: list[torch.Tensor] = []
        for record_offset, (record_index, record) in enumerate(
            zip(spec.record_indices, records, strict=True)
        ):
            times = _requested_times(
                record.time_s,
                source_t0_s=float(record.source_parameters[3]),
                continuous=continuous_mask[record_offset],
                phase_offset=spec.step + record_offset,
            )
            target = self.records.read_wavefield(record_index, times)
            generator = torch.Generator().manual_seed(
                self.seed + spec.step * 1009 + record_offset * 9176
            )
            coords, values, probability = _sample_queries(
                target.values,
                target.requested_time_s,
                record.x_m,
                record.z_m,
                count=self.query_points,
                generator=generator,
            )
            requested_times.append(target.requested_time_s)
            targets.append(target.values)
            exact.append(target.exact)
            alpha.append(target.alpha)
            left.append(target.left_index)
            right.append(target.right_index)
            query_coords.append(coords)
            query_targets.append(values)
            query_probability.append(probability)
        return PilotBatch(
            step=spec.step,
            velocity_mps=macro.velocity_mps,
            record_to_medium=macro.record_to_medium,
            source_parameters=macro.source_parameters,
            source_map=macro.source_map,
            requested_time_s=torch.stack(requested_times),
            dense_target_physical=torch.stack(targets),
            target_exact=torch.stack(exact),
            interpolation_alpha=torch.stack(alpha),
            left_index=torch.stack(left),
            right_index=torch.stack(right),
            query_coords=torch.stack(query_coords),
            query_target_physical=torch.stack(query_targets),
            query_probability=torch.stack(query_probability),
            x_m=macro.x_m,
            z_m=macro.z_m,
            sample_id=macro.sample_id,
            group_id=macro.group_id,
            medium_type=macro.medium_type,
        )


def identity_pilot_batch(batch: PilotBatch) -> PilotBatch:
    return batch


def make_pilot_loader(
    dataset: PilotBatchDataset,
    *,
    workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader[PilotBatch]:
    if workers < 0 or prefetch_factor <= 0:
        raise ValueError("pilot loader worker settings are invalid")
    options: dict[str, object] = {}
    if workers:
        options.update(prefetch_factor=int(prefetch_factor), persistent_workers=True)
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        collate_fn=identity_pilot_batch,
        **options,
    )


__all__ = [
    "PilotBatch",
    "PilotBatchDataset",
    "PilotStepSpec",
    "build_pilot_schedule",
    "identity_pilot_batch",
    "make_pilot_loader",
]
