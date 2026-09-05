"""Exact stored-frame macro batches and safe grouped microbatch splitting."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from grouped_ufno_mionet_v3.data.batch import pack_v3_groups
from grouped_ufno_mionet_v3.data.index import V3DataManifest
from grouped_ufno_mionet_v3.data.pilot import PilotBatch, PilotStepSpec, _sample_queries
from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset

from .eikonal import EikonalTravelCache
from .multifidelity import (
    numerical_teacher_pool_indices,
    residual_adaptive_teacher_pool_indices,
)
from .sampling import (
    appearance_time_indices,
    stored_time_indices,
    validation_time_indices,
)


class ExactStoredTimeBatchDataset(Dataset[PilotBatch]):
    """Materialize four exact HDF5 frames spanning the complete time axis."""

    def __init__(
        self,
        source_h5: str | Path,
        manifest: V3DataManifest,
        *,
        split: str | Sequence[str],
        schedule: Sequence[PilotStepSpec],
        query_points: int,
        seed: int,
        time_policy: str = "legacy4",
        frames_per_record: int | None = None,
        travel_time_h5: str | Path | None = None,
        allow_travel_source_path_mismatch: bool = False,
        time_index_pool: Sequence[int] | None = None,
        time_index_probabilities: Sequence[float] | None = None,
        bin_frame_budget: tuple[int, int, int] | None = None,
    ) -> None:
        if not schedule or query_points <= 0:
            raise ValueError("exact stored-time dataset configuration is invalid")
        if time_policy not in {
            "legacy4", "appearance4", "appearance16", "validation16",
            "validation_fixed", "fixed_train_gate", "all_saved",
            "numerical_teacher_pool",
        }:
            raise ValueError(f"unknown exact stored-time policy: {time_policy}")
        self.records = V3WavefieldDataset(source_h5, manifest, split=split)
        self.schedule = tuple(schedule)
        self.query_points = int(query_points)
        self.seed = int(seed)
        self.time_policy = str(time_policy)
        self.bin_frame_budget = (
            None if bin_frame_budget is None
            else tuple(int(v) for v in bin_frame_budget)
        )
        self.time_index_pool = (
            None
            if time_index_pool is None
            else tuple(int(value) for value in time_index_pool)
        )
        self.time_index_probabilities = (
            None
            if time_index_probabilities is None
            else tuple(float(value) for value in time_index_probabilities)
        )
        if self.time_policy == "numerical_teacher_pool":
            if (
                not self.time_index_pool
                or len(set(self.time_index_pool)) != len(self.time_index_pool)
                or any(
                    left >= right
                    for left, right in zip(
                        self.time_index_pool, self.time_index_pool[1:]
                    )
                )
            ):
                raise ValueError(
                    "numerical teacher policy requires a strict time-index pool"
                )
            if self.time_index_probabilities is not None:
                probability = np.asarray(self.time_index_probabilities, dtype=np.float64)
                if (
                    probability.shape != (len(self.time_index_pool),)
                    or not np.isfinite(probability).all()
                    or np.any(probability < 0.0)
                    or float(probability.sum()) <= 0.0
                ):
                    raise ValueError(
                        "numerical teacher time probabilities must be finite, nonnegative, "
                        "and match the time-index pool"
                    )
        elif self.time_index_pool is not None:
            raise ValueError("time-index pool is only valid for numerical teacher policy")
        elif self.time_index_probabilities is not None:
            raise ValueError(
                "time-index probabilities are only valid for numerical teacher policy"
            )
        self.travel_cache = (
            None
            if travel_time_h5 is None
            else EikonalTravelCache(
                travel_time_h5,
                source_h5=source_h5,
                allow_source_path_mismatch=bool(
                    allow_travel_source_path_mismatch
                ),
            )
        )
        default_frames = {
            "legacy4": 4,
            "appearance4": 4,
            "appearance16": 16,
            "validation16": 16,
            "all_saved": len(manifest.time_s),
            "numerical_teacher_pool": 16,
        }.get(self.time_policy)
        self.frames_per_record = int(
            default_frames if frames_per_record is None else frames_per_record
        )
        if self.frames_per_record <= 0:
            raise ValueError("frames_per_record must be positive")

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> PilotBatch:
        return self.materialize_spec(self.schedule[int(index)])

    def materialize_spec(self, spec: PilotStepSpec) -> PilotBatch:
        records = [self.records[record_index] for record_index in spec.record_indices]
        macro = pack_v3_groups(records)
        requested_times: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        exact: list[torch.Tensor] = []
        alpha: list[torch.Tensor] = []
        left: list[torch.Tensor] = []
        right: list[torch.Tensor] = []
        query_coords: list[torch.Tensor] = []
        query_targets: list[torch.Tensor] = []
        query_probability: list[torch.Tensor] = []
        appearances = getattr(spec, "appearance_indices", None)
        if self.time_policy in {
            "appearance4",
            "appearance16",
            "numerical_teacher_pool",
        } and (
            appearances is None or len(appearances) != len(spec.record_indices)
        ):
            raise ValueError("appearance policy requires one appearance index per record")
        for record_offset, (record_index, record) in enumerate(
            zip(spec.record_indices, records, strict=True)
        ):
            axis = record.time_s.detach().cpu().numpy()
            if self.time_policy == "numerical_teacher_pool":
                assert self.time_index_pool is not None
                if self.time_index_pool[-1] >= len(axis):
                    raise ValueError("numerical teacher time pool exceeds stored axis")
                selector = (
                    numerical_teacher_pool_indices
                    if self.time_index_probabilities is None
                    else residual_adaptive_teacher_pool_indices
                )
                selector_kwargs = {}
                if self.time_index_probabilities is not None:
                    selector_kwargs["probabilities"] = self.time_index_probabilities
                indices = np.asarray(
                    selector(
                        self.time_index_pool,
                        sample_id=record.sample_id,
                        appearance=int(appearances[record_offset]),
                        seed=self.seed,
                        count=self.frames_per_record,
                        **selector_kwargs,
                    ),
                    dtype=np.int64,
                )
            elif self.time_policy in {"appearance4", "appearance16"}:
                indices = appearance_time_indices(
                    axis,
                    source_t0_s=float(record.source_parameters[3]),
                    source_f0_hz=(
                        float(record.source_parameters[2])
                        if self.time_policy == "appearance16"
                        else None
                    ),
                    sample_id=record.sample_id,
                    appearance=int(appearances[record_offset]),
                    seed=self.seed,
                    count=self.frames_per_record,
                    bin_frame_budget=(
                        self.bin_frame_budget
                        if self.time_policy == "appearance16"
                        else None
                    ),
                )
            elif self.time_policy in {
                "validation16",
                "validation_fixed",
                "fixed_train_gate",
            }:
                indices = validation_time_indices(
                    axis,
                    source_t0_s=float(record.source_parameters[3]),
                    source_f0_hz=(
                        float(record.source_parameters[2])
                        if self.time_policy in {"validation_fixed", "fixed_train_gate"}
                        else None
                    ),
                    sample_id=record.sample_id,
                    panel_offset=(
                        0
                        if self.time_policy == "fixed_train_gate"
                        else int(getattr(spec, "epoch", spec.step))
                    ),
                    seed=self.seed,
                    count=self.frames_per_record,
                )
            elif self.time_policy == "all_saved":
                indices = torch.arange(len(axis), dtype=torch.long).numpy()
            else:
                indices = stored_time_indices(
                    axis,
                    source_t0_s=float(record.source_parameters[3]),
                    phase_offset=spec.step + record_offset,
                )
            times = record.time_s[torch.from_numpy(indices)]
            target = self.records.read_wavefield(record_index, times)
            if not bool(target.exact.all()) or not torch.equal(
                target.left_index, target.right_index
            ):
                raise RuntimeError("V4 exact dataset materialized an interpolated target")
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
            dense_travel_time_s=(
                None
                if self.travel_cache is None
                else self.travel_cache.read(macro.sample_id)
            ),
        )


def _slice_records(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.index_select(0, indices.to(value.device))


def merge_pilot_batches(batches: Sequence[PilotBatch]) -> PilotBatch:
    """Merge fixed macro batches while offsetting their local medium mappings."""
    values = tuple(batches)
    if not values:
        raise ValueError("cannot merge an empty pilot batch sequence")
    reference = values[0]
    for batch in values[1:]:
        if not torch.equal(batch.x_m, reference.x_m) or not torch.equal(batch.z_m, reference.z_m):
            raise ValueError("pilot batches use different physical grids")
    offsets: list[torch.Tensor] = []
    medium_offset = 0
    for batch in values:
        offsets.append(batch.record_to_medium + medium_offset)
        medium_offset += batch.velocity_mps.shape[0]
    record_tensor_names = (
        "source_parameters", "source_map", "requested_time_s",
        "dense_target_physical", "target_exact", "interpolation_alpha",
        "left_index", "right_index", "query_coords",
        "query_target_physical", "query_probability",
    )
    merged = {
        name: torch.cat(tuple(getattr(batch, name) for batch in values), dim=0)
        for name in record_tensor_names
    }
    travel_fields = tuple(batch.dense_travel_time_s for batch in values)
    if any(value is None for value in travel_fields) and not all(
        value is None for value in travel_fields
    ):
        raise ValueError("cannot merge a mixture of cached and uncached travel fields")
    merged_travel = (
        None
        if travel_fields[0] is None
        else torch.cat(tuple(value for value in travel_fields if value is not None), dim=0)
    )
    return PilotBatch(
        step=reference.step,
        velocity_mps=torch.cat(tuple(batch.velocity_mps for batch in values), dim=0),
        record_to_medium=torch.cat(offsets, dim=0),
        x_m=reference.x_m,
        z_m=reference.z_m,
        sample_id=tuple(value for batch in values for value in batch.sample_id),
        group_id=tuple(value for batch in values for value in batch.group_id),
        medium_type=tuple(value for batch in values for value in batch.medium_type),
        dense_travel_time_s=merged_travel,
        **merged,
    )


def split_pilot_batch(
    batch: PilotBatch,
    *,
    microbatch_records: int,
) -> tuple[PilotBatch, ...]:
    """Split records while rebuilding each local medium-index mapping."""

    record_count = len(batch.sample_id)
    if microbatch_records <= 0:
        raise ValueError("microbatch record count must be positive")
    pieces: list[PilotBatch] = []
    for start in range(0, record_count, microbatch_records):
        stop = min(start + microbatch_records, record_count)
        indices = torch.arange(start, stop, dtype=torch.long)
        old_mapping = batch.record_to_medium[indices].tolist()
        medium_ids: list[int] = []
        for medium_id in old_mapping:
            if int(medium_id) not in medium_ids:
                medium_ids.append(int(medium_id))
        remap = {medium_id: position for position, medium_id in enumerate(medium_ids)}
        mapping = torch.tensor([remap[int(value)] for value in old_mapping], dtype=torch.long)
        medium_indices = torch.tensor(medium_ids, dtype=torch.long)
        text_slice = slice(start, stop)
        pieces.append(
            PilotBatch(
                step=batch.step,
                velocity_mps=_slice_records(batch.velocity_mps, medium_indices),
                record_to_medium=mapping,
                source_parameters=_slice_records(batch.source_parameters, indices),
                source_map=_slice_records(batch.source_map, indices),
                requested_time_s=_slice_records(batch.requested_time_s, indices),
                dense_target_physical=_slice_records(batch.dense_target_physical, indices),
                target_exact=_slice_records(batch.target_exact, indices),
                interpolation_alpha=_slice_records(batch.interpolation_alpha, indices),
                left_index=_slice_records(batch.left_index, indices),
                right_index=_slice_records(batch.right_index, indices),
                query_coords=_slice_records(batch.query_coords, indices),
                query_target_physical=_slice_records(batch.query_target_physical, indices),
                query_probability=_slice_records(batch.query_probability, indices),
                x_m=batch.x_m,
                z_m=batch.z_m,
                sample_id=batch.sample_id[text_slice],
                group_id=batch.group_id[text_slice],
                medium_type=batch.medium_type[text_slice],
                dense_travel_time_s=(
                    None
                    if batch.dense_travel_time_s is None
                    else _slice_records(batch.dense_travel_time_s, indices)
                ),
            )
        )
    return tuple(pieces)


__all__ = ["ExactStoredTimeBatchDataset", "merge_pilot_batches", "split_pilot_batch"]
