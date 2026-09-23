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
    ic_time_s: torch.Tensor | None = None
    ic_snapshots_physical: torch.Tensor | None = None

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
    families: tuple[str, ...] = ALLOWED_MEDIUM_TYPES,
    batch_records: int = 12,
    excluded_group_ids: Sequence[str] = (),
) -> tuple[PilotStepSpec, ...]:
    if steps <= 0:
        raise ValueError("pilot schedule steps must be positive")
    if batch_records <= 0:
        raise ValueError("pilot schedule batch_records must be positive")
    if not families:
        raise ValueError("pilot schedule families must be non-empty")
    unknown = set(families) - set(ALLOWED_MEDIUM_TYPES)
    if unknown:
        raise ValueError(f"unknown pilot schedule families: {sorted(unknown)}")
    records = tuple(record for record in manifest.records if record.split == split)
    if not records:
        raise ValueError(f"pilot split is empty: {split}")
    assert_allowed_families(record.medium_type for record in records)
    groups: dict[str, dict[str, list[int]]] = {
        family: defaultdict(list) for family in ALLOWED_MEDIUM_TYPES
    }
    excluded = frozenset(str(group_id) for group_id in excluded_group_ids)
    unknown_exclusions = excluded - {record.group_id for record in records}
    if unknown_exclusions:
        raise ValueError(f"excluded groups are absent from split {split}: {sorted(unknown_exclusions)}")
    # Enumerate the ORIGINAL split before filtering: PilotBatchDataset indexes
    # that split, so compacting these indices would silently select other groups.
    for local_index, record in enumerate(records):
        if record.group_id in excluded:
            continue
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
    # One balanced block is 4 uniform records + one full 4-source layered group
    # + one 5-source marmousi group minus one member = 12 records, i.e. a 1:1:1
    # split across the families by record count.  `batch_records` scales the
    # block count, so the family balance is preserved exactly; a batch that is
    # not a multiple of the block is rejected rather than silently rounded
    # (rounding would change which family dominates the gradient).  The
    # uniform-only branch takes `batch_records` directly and needs no block.
    block = 12
    if families != ("uniform",):
        if batch_records % block != 0:
            raise ValueError(
                f"balanced-family batch_records must be a multiple of {block}, "
                f"got {batch_records}"
            )
        blocks = batch_records // block
        uniform_units = 4 * blocks
        layered_groups = 1 * blocks
        marmousi_groups = 1 * blocks
        for name, needed, available in (
            ("uniform records", uniform_units, len(uniform)),
            ("layered groups", layered_groups, len(layered)),
            ("marmousi groups", marmousi_groups, len(marmousi)),
        ):
            if needed > available:
                raise ValueError(
                    f"batch_records={batch_records} needs {needed} {name} but only "
                    f"{available} are available"
                )
    schedule: list[PilotStepSpec] = []
    for step in range(int(steps)):
        if families == ("uniform",):
            # Uniform-only pilot: every batch is drawn from the uniform records.
            selected_uniform = _cyclic_window(uniform_indices, step * batch_records, batch_records)
            selected = tuple(int(value) for value in selected_uniform)
            if len(selected) != batch_records or len(set(selected)) != batch_records:
                raise RuntimeError("pilot schedule failed its unique uniform contract")
            schedule.append(PilotStepSpec(step=step, record_indices=selected))
            continue
        # Each block reproduces the historical step advance for one unit of the
        # mix, so batch_records=12 (blocks=1) is byte-for-byte the frozen
        # schedule and larger batches are exact replications of it side by side.
        selected_uniform = _cyclic_window(
            uniform_indices, step * uniform_units, uniform_units
        )
        selected_layered: list[int] = []
        selected_marmousi: list[int] = []
        for block_index in range(blocks):
            # Offset each block so a larger batch covers distinct groups rather
            # than repeating the same one `blocks` times (which the uniqueness
            # check would reject anyway).  block_index=0 reproduces the frozen
            # behaviour exactly.
            offset = block_index
            selected_layered.extend(layered[(step + offset) % len(layered)])
            group = marmousi[(step + offset) % len(marmousi)]
            # The omission advances once per full pass over the marmousi groups,
            # which is what makes all five sources of a medium reachable.
            omitted = ((step + offset) // len(marmousi)) % len(group)
            selected_marmousi.extend(
                index for position, index in enumerate(group) if position != omitted
            )
        selected = tuple(
            int(value)
            for value in (*selected_uniform, *selected_layered, *selected_marmousi)
        )
        if len(selected) != batch_records or len(set(selected)) != batch_records:
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
    steps: int = 3,
    start_offset: int = 0,
    time_policy: str = "legacy",
    cycle: int = 0,
    stratum_weights: "np.ndarray | torch.Tensor | None" = None,
) -> torch.Tensor:
    if steps < 1 or start_offset < 0:
        raise ValueError("steps must be >=1 and start_offset nonnegative")
    if time_policy not in ("legacy", "full_support", "residual_adaptive"):
        raise ValueError(f"unknown time_policy: {time_policy!r}")
    if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 0:
        raise ValueError("cycle must be a nonnegative integer")
    if len(continuous) != steps:
        raise ValueError("continuous mask length must equal the requested step count")
    axis = time_s.double()
    onset = min(int(torch.searchsorted(axis, torch.tensor(source_t0_s, dtype=axis.dtype))), len(axis) - 1)
    onset = min(onset + int(start_offset), len(axis) - 1)
    end_time = min(float(axis[-1]), float(source_t0_s) + float(active_horizon_s))
    end = min(int(torch.searchsorted(axis, torch.tensor(end_time, dtype=axis.dtype))), len(axis) - 1)
    if end < onset:
        raise ValueError("pilot active horizon ends before the first admissible frame")
    candidates = np.arange(onset, end + 1, dtype=np.int64)
    phases = np.array_split(candidates, steps)
    if any(len(phase) == 0 for phase in phases):
        raise ValueError("pilot active horizon cannot form the requested time phases")
    exact_indices_list: list[int] = []
    if time_policy == "legacy":
        # Historical behaviour: each stratum drops its first/last 20% (those
        # frames have sampling probability exactly zero) and picks a fixed
        # phase inside the interior.  Kept bit-identical for old configs.
        for phase_index, phase in enumerate(phases):
            trim = len(phase) // 5
            interior = phase[trim : len(phase) - trim] if trim else phase
            position = (int(phase_offset) + phase_index * 11) % len(interior)
            exact_indices_list.append(int(interior[position]))
    elif time_policy == "residual_adaptive":
        # Every candidate frame is reachable, including the stratum endpoints
        # that `legacy` excludes with its 20% trim, and the picked position is
        # a per-(record, step, stratum) draw rather than a fixed-phase stride.
        #
        # Why not reuse full_support: its position is
        # (phase_offset + phase_index*11 + cycle*29) % len(phase) with
        # cycle = step % 3, while a record is revisited every 35 or 105 steps in
        # the balanced schedule.  Both strides are multiples of 35, so the
        # reachable (phase, position) pairs collapse onto a handful of
        # combinations: measured on the real 22440-step schedule, a record seen
        # only 16 times keeps returning to the same stratum offsets.  A fresh
        # draw per visit removes that aliasing by construction.
        #
        # `stratum_weights` has one entry per dense time step, i.e. per stratum.
        #
        # Two modes, because they trade off differently:
        #   * stratified (weights None) -- exactly one draw per stratum, so the
        #     supervision stays balanced in time.  This is the coverage-first
        #     default; with a per-visit random position it already fixes both
        #     measured defects (trim unreachability and revisit aliasing).
        #   * residual-weighted (weights given) -- the 16 slots draw their
        #     stratum WITH replacement under `0.5 uniform + 0.5 residual`, so a
        #     large-residual time window receives more of this step's slots at
        #     the price of temporarily skipping quiet strata.  The uniform half
        #     is what keeps every stratum reachable: a window whose residual
        #     decayed to zero is still drawn, so a regression there is seen and
        #     the EMA can raise its weight again.
        #
        # The per-slot seed is a splitmix64 mix of (visit, slot).  A plain
        # multiply-and-modulo seed left neighbouring visits correlated and a
        # phase only ever reached ~16 of its 24 frames; mixing realigns them.
        per_stratum = None
        if stratum_weights is not None:
            per_stratum = np.asarray(stratum_weights, dtype=np.float64).reshape(-1)
            if per_stratum.shape[0] != len(phases):
                raise ValueError(
                    f"stratum_weights must have one entry per dense time step "
                    f"({len(phases)}), got {per_stratum.shape[0]}"
                )
            per_stratum = np.clip(per_stratum, 0.0, None)
            total = per_stratum.sum()
            stratum_probability = (
                0.5 / len(phases) + 0.5 * per_stratum / total
                if total > 0
                else np.full(len(phases), 1.0 / len(phases))
            )

        visit = int(phase_offset) * 16 + int(cycle)
        for slot in range(len(phases)):
            mixed = (visit * 0x9E3779B97F4A7C15 + slot) & 0xFFFFFFFFFFFFFFFF
            mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
            mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
            rng = np.random.default_rng((mixed ^ (mixed >> 31)) & 0x7FFFFFFFFFFFFFFF)
            if per_stratum is None:
                phase = phases[slot]
            else:
                phase = phases[int(rng.choice(len(phases), p=stratum_probability))]
            exact_indices_list.append(int(phase[int(rng.integers(0, len(phase)))]))
        # time order, so downstream per-frame tensors stay monotone in time
        exact_indices_list.sort()
    elif time_policy == "full_support":
        # full_support: every frame of every stratum is reachable, and the
        # window endpoints are forced on a 3-cycle rotation (cycle%3==0 pins
        # the first future frame, ==1 pins the last stored frame, ==2 both),
        # matching the stratified estimator in research/fullfam_followup.
        pin_first = cycle % 3 in (0, 2)
        pin_last = cycle % 3 in (1, 2)
        for phase_index, phase in enumerate(phases):
            if pin_first and phase_index == 0:
                exact_indices_list.append(int(phase[0]))
                continue
            if pin_last and phase_index == len(phases) - 1:
                exact_indices_list.append(int(phase[-1]))
                continue
            position = (int(phase_offset) + phase_index * 11 + cycle * 29) % len(phase)
            exact_indices_list.append(int(phase[position]))
    else:
        raise ValueError("time_policy is not a recognized sampling policy")
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
    sample_weights: torch.Tensor | None = None,
    uniform_floor: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames, nz, nx = target.shape
    flat_target = target.reshape(-1)
    # Default: static energy weighting (probability proportional to |target|).
    # Adaptive mode: caller supplies nonnegative sample_weights (residual map),
    # broadcast over frames, so sampling favours high-residual space-time points.
    #
    # `uniform_floor` mixes in a uniform component.  It is what keeps every
    # point reachable: with a pure residual weighting, a region whose residual
    # has decayed to zero would stop being drawn entirely and would never be
    # corrected again, and reciprocal-probability reweighting in the query loss
    # would blow up on it.
    if not 0.0 <= float(uniform_floor) <= 1.0:
        raise ValueError("uniform_floor must lie in [0,1]")
    if sample_weights is None:
        weight_field = flat_target.abs().float()
    else:
        weight_field = sample_weights.float()
        if weight_field.shape == (nz, nx):
            weight_field = weight_field.reshape(1, nz, nx).expand(frames, nz, nx)
        if weight_field.shape != (frames, nz, nx):
            raise ValueError(
                f"sample_weights must be [frames,{nz},{nx}] or [{nz},{nx}], "
                f"got {tuple(weight_field.shape)}"
            )
        weight_field = weight_field.reshape(-1)
    weights = weight_field + torch.finfo(torch.float32).eps
    weights /= weights.sum()
    uniform = float(uniform_floor)
    probability = uniform / len(weights) + (1.0 - uniform) * weights
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
        dense_time_steps: int = 3,
        active_horizon_s: float = 0.60,
        ic_frames: int = 0,
        residual_map: torch.Tensor | None = None,
        residual_time_profile: torch.Tensor | None = None,
        time_sampling_policy: str = "legacy",
    ) -> None:
        if not schedule or not 0.0 <= continuous_fraction <= 1.0 or query_points <= 0:
            raise ValueError("pilot batch dataset configuration is invalid")
        if dense_time_steps < 1 or ic_frames < 0 or active_horizon_s <= 0:
            raise ValueError("dense_time_steps/ic_frames/active_horizon_s are invalid")
        self.records = V3WavefieldDataset(source_h5, manifest, split=split)
        self.schedule = tuple(schedule)
        self.continuous_fraction = float(continuous_fraction)
        self.query_points = int(query_points)
        self.seed = int(seed)
        self.dense_time_steps = int(dense_time_steps)
        self.active_horizon_s = float(active_horizon_s)
        self.ic_frames = int(ic_frames)
        if time_sampling_policy not in ("legacy", "full_support", "residual_adaptive"):
            raise ValueError(
                "time_sampling_policy must be legacy, full_support or residual_adaptive"
            )
        self.time_sampling_policy = str(time_sampling_policy)
        # Optional residual-adaptive spatial sampling map [nz,nx]; updated by the
        # training loop each step (self.residual_map is mutated in place so the
        # DataLoader worker sees the latest EMA weights).
        if residual_map is not None:
            residual_map = torch.as_tensor(residual_map, dtype=torch.float32)
            if residual_map.ndim != 2 or torch.any(residual_map < 0.0):
                raise ValueError("residual_map must be a nonnegative 2-D [nz,nx] tensor")
        self.residual_map = residual_map
        # Optional per-record residual profile over the time axis, indexed by
        # the record's positional index.  Only the residual_adaptive time policy
        # reads it; None keeps the draw uniform inside each stratum.  Mutated in
        # place by the training loop, like residual_map.
        if residual_time_profile is not None:
            residual_time_profile = torch.as_tensor(
                residual_time_profile, dtype=torch.float32
            )
            if residual_time_profile.ndim != 2 or torch.any(residual_time_profile < 0.0):
                raise ValueError(
                    "residual_time_profile must be a nonnegative [records, time] tensor"
                )
        self.residual_time_profile = residual_time_profile

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> PilotBatch:
        return self.materialize_spec(self.schedule[int(index)])

    def _ic_window(self, record_index: int, record) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact stored frames covering source onset: [onset, onset+ic_frames)."""
        axis = record.time_s.double()
        onset = min(
            int(torch.searchsorted(axis, torch.tensor(float(record.source_parameters[3]), dtype=axis.dtype))),
            len(axis) - 1,
        )
        if onset + self.ic_frames > len(axis):
            raise ValueError("ic window exceeds the stored time axis")
        ic_times = record.time_s[onset : onset + self.ic_frames].float()
        target = self.records.read_wavefield(record_index, ic_times)
        if not bool(target.exact.all()):
            raise RuntimeError("ic frames must be exact stored snapshots")
        return ic_times, target.values

    def materialize_spec(self, spec: PilotStepSpec) -> PilotBatch:
        records = [self.records[record_index] for record_index in spec.record_indices]
        macro = pack_v3_groups(records)
        total_targets = len(records) * self.dense_time_steps
        continuous_count = int(round(total_targets * self.continuous_fraction))
        mask_generator = torch.Generator().manual_seed(self.seed + spec.step * 104729)
        continuous_flat = torch.zeros(total_targets, dtype=torch.bool)
        if continuous_count:
            chosen = torch.randperm(total_targets, generator=mask_generator)[:continuous_count]
            continuous_flat[chosen] = True
        continuous_mask = continuous_flat.reshape(len(records), self.dense_time_steps)

        requested_times: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        exact: list[torch.Tensor] = []
        alpha: list[torch.Tensor] = []
        left: list[torch.Tensor] = []
        right: list[torch.Tensor] = []
        query_coords: list[torch.Tensor] = []
        query_targets: list[torch.Tensor] = []
        query_probability: list[torch.Tensor] = []
        ic_times_list: list[torch.Tensor] = []
        ic_values_list: list[torch.Tensor] = []
        for record_offset, (record_index, record) in enumerate(
            zip(spec.record_indices, records, strict=True)
        ):
            times = _requested_times(
                record.time_s,
                source_t0_s=float(record.source_parameters[3]),
                continuous=continuous_mask[record_offset],
                phase_offset=spec.step + record_offset,
                active_horizon_s=self.active_horizon_s,
                steps=self.dense_time_steps,
                start_offset=self.ic_frames,
                time_policy=self.time_sampling_policy,
                cycle=spec.step % 3,
                stratum_weights=(
                    self.residual_time_profile[int(record_index)]
                    if self.residual_time_profile is not None
                    else None
                ),
            )
            if self.ic_frames:
                ic_times, ic_values = self._ic_window(record_index, record)
                ic_times_list.append(ic_times)
                ic_values_list.append(ic_values)
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
                sample_weights=self.residual_map,
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
            ic_time_s=torch.stack(ic_times_list) if self.ic_frames else None,
            ic_snapshots_physical=torch.stack(ic_values_list) if self.ic_frames else None,
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
