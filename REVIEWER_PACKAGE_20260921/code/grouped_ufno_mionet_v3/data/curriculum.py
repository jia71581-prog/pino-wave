"""Deterministic easy-to-hard family and replay schedules for V3."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES
from .index import assert_allowed_families
from .pilot import PilotBatch, PilotBatchDataset, PilotStepSpec


@dataclass(frozen=True)
class CurriculumMicrobatchSpec:
    record_indices: tuple[int, ...]
    family: str
    replay: bool
    loss_scale: float = 1.0


@dataclass(frozen=True)
class CurriculumStepSpec:
    optimizer_step: int
    stage: str
    microbatches: tuple[CurriculumMicrobatchSpec, ...]


@dataclass(frozen=True)
class CurriculumStepBatch:
    optimizer_step: int
    stage: str
    microbatches: tuple[PilotBatch, ...]
    family: tuple[str, ...]
    replay: tuple[bool, ...]
    loss_scale: tuple[float, ...]

    def pin_memory(self) -> "CurriculumStepBatch":
        return CurriculumStepBatch(
            optimizer_step=self.optimizer_step,
            stage=self.stage,
            microbatches=tuple(batch.pin_memory() for batch in self.microbatches),
            family=self.family,
            replay=self.replay,
            loss_scale=self.loss_scale,
        )


def _window(values: Sequence[object], start: int, count: int) -> list[object]:
    if len(values) < count:
        raise ValueError("curriculum schedule does not have enough unique groups or records")
    return [values[(start + offset) % len(values)] for offset in range(count)]


def _uniform_microbatches(
    indices: Sequence[int],
    *,
    start: int,
    replay: bool,
) -> tuple[CurriculumMicrobatchSpec, ...]:
    selected = _window(indices, start, 12)
    return tuple(
        CurriculumMicrobatchSpec(
            record_indices=tuple(int(value) for value in selected[offset : offset + 4]),
            family="uniform",
            replay=replay,
        )
        for offset in range(0, 12, 4)
    )


def build_curriculum_schedule(
    manifest,
    *,
    split: str,
    stage: str,
    optimizer_steps: int,
    seed: int,
) -> tuple[CurriculumStepSpec, ...]:
    if stage not in ALLOWED_MEDIUM_TYPES:
        raise ValueError(f"unknown curriculum stage: {stage}")
    if optimizer_steps <= 0:
        raise ValueError("curriculum optimizer steps must be positive")
    records = tuple(record for record in manifest.records if record.split == split)
    if not records:
        raise ValueError(f"curriculum split is empty: {split}")
    assert_allowed_families(record.medium_type for record in records)
    groups: dict[str, dict[str, list[int]]] = {
        family: defaultdict(list) for family in ALLOWED_MEDIUM_TYPES
    }
    for local_index, record in enumerate(records):
        groups[record.medium_type][record.group_id].append(local_index)
    uniform_groups = list(groups["uniform"].values())
    layered_groups = list(groups["layered"].values())
    marmousi_groups = list(groups["marmousi"].values())
    if any(len(group) != 1 for group in uniform_groups):
        raise ValueError("curriculum expects one source per uniform group")
    if any(len(group) != 4 for group in layered_groups):
        raise ValueError("curriculum expects four sources per layered group")
    if any(len(group) != 5 for group in marmousi_groups):
        raise ValueError("curriculum expects five sources per Marmousi group")
    if len(uniform_groups) < 12 or len(layered_groups) < 3 or len(marmousi_groups) < 2:
        raise ValueError("curriculum census is too small for the requested batching contract")
    generator = np.random.default_rng(int(seed))
    generator.shuffle(uniform_groups)
    generator.shuffle(layered_groups)
    generator.shuffle(marmousi_groups)
    uniform = [int(group[0]) for group in uniform_groups]
    result: list[CurriculumStepSpec] = []
    uniform_cursor = 0
    layered_cursor = 0
    marmousi_cursor = 0
    for optimizer_step in range(int(optimizer_steps)):
        if stage == "uniform":
            microbatches = _uniform_microbatches(
                uniform, start=uniform_cursor, replay=False
            )
            uniform_cursor += 12
        elif stage == "layered" and optimizer_step % 4 == 3:
            microbatches = _uniform_microbatches(uniform, start=uniform_cursor, replay=True)
            uniform_cursor += 12
        elif stage == "layered":
            selected = _window(layered_groups, layered_cursor, 3)
            layered_cursor += 3
            microbatches = (
                CurriculumMicrobatchSpec(
                    record_indices=tuple(int(value) for group in selected for value in group),
                    family="layered",
                    replay=False,
                ),
            )
        elif optimizer_step % 5 == 3:
            microbatches = _uniform_microbatches(uniform, start=uniform_cursor, replay=True)
            uniform_cursor += 12
        elif optimizer_step % 5 == 4:
            selected = _window(layered_groups, layered_cursor, 3)
            layered_cursor += 3
            microbatches = (
                CurriculumMicrobatchSpec(
                    record_indices=tuple(int(value) for group in selected for value in group),
                    family="layered",
                    replay=True,
                ),
            )
        else:
            selected = _window(marmousi_groups, marmousi_cursor, 2)
            marmousi_cursor += 2
            microbatches = (
                CurriculumMicrobatchSpec(
                    record_indices=tuple(int(value) for group in selected for value in group),
                    family="marmousi",
                    replay=False,
                    loss_scale=1.2,
                ),
            )
        result.append(
            CurriculumStepSpec(
                optimizer_step=optimizer_step,
                stage=stage,
                microbatches=microbatches,
            )
        )
    return tuple(result)


class CurriculumStepDataset(Dataset[CurriculumStepBatch]):
    def __init__(
        self,
        source_h5: str | Path,
        manifest,
        *,
        split: str,
        schedule: Sequence[CurriculumStepSpec],
        continuous_fraction: float,
        query_points: int,
        seed: int,
    ) -> None:
        if not schedule:
            raise ValueError("curriculum step dataset requires a nonempty schedule")
        self.schedule = tuple(schedule)
        first = self.schedule[0]
        if not first.microbatches:
            raise ValueError("curriculum optimizer step has no microbatches")
        dummy = PilotStepSpec(
            step=first.optimizer_step * 16,
            record_indices=first.microbatches[0].record_indices,
        )
        self.materializer = PilotBatchDataset(
            source_h5,
            manifest,
            split=split,
            schedule=(dummy,),
            continuous_fraction=continuous_fraction,
            query_points=query_points,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> CurriculumStepBatch:
        spec = self.schedule[int(index)]
        batches: list[PilotBatch] = []
        for microbatch_index, microbatch in enumerate(spec.microbatches):
            if len(set(microbatch.record_indices)) != len(microbatch.record_indices):
                raise RuntimeError("curriculum microbatch contains duplicate records")
            batch = self.materializer.materialize_spec(
                PilotStepSpec(
                    step=spec.optimizer_step * 16 + microbatch_index,
                    record_indices=microbatch.record_indices,
                )
            )
            if set(batch.medium_type) != {microbatch.family}:
                raise RuntimeError("curriculum microbatch family does not match its schedule")
            batches.append(batch)
        return CurriculumStepBatch(
            optimizer_step=spec.optimizer_step,
            stage=spec.stage,
            microbatches=tuple(batches),
            family=tuple(microbatch.family for microbatch in spec.microbatches),
            replay=tuple(microbatch.replay for microbatch in spec.microbatches),
            loss_scale=tuple(float(microbatch.loss_scale) for microbatch in spec.microbatches),
        )


def identity_curriculum_step(batch: CurriculumStepBatch) -> CurriculumStepBatch:
    return batch


def make_curriculum_loader(
    dataset: CurriculumStepDataset,
    *,
    workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader[CurriculumStepBatch]:
    if workers < 0 or prefetch_factor <= 0:
        raise ValueError("curriculum loader worker settings are invalid")
    options: dict[str, object] = {}
    if workers:
        options.update(prefetch_factor=int(prefetch_factor), persistent_workers=True)
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        collate_fn=identity_curriculum_step,
        **options,
    )


__all__ = [
    "CurriculumMicrobatchSpec",
    "CurriculumStepBatch",
    "CurriculumStepDataset",
    "CurriculumStepSpec",
    "build_curriculum_schedule",
    "identity_curriculum_step",
    "make_curriculum_loader",
]
