"""Sealed development gate for a native 201x201 LWC-84 baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES
from saved_time_phase_operator_v4.streaming_metrics import ExactWavefieldMetricAccumulator


EXPECTED_VALIDATION_FAMILY_COUNTS = {
    "uniform": 90,
    "layered": 240,
    "marmousi": 150,
}


@dataclass(frozen=True)
class CoarseRecordIndex:
    source_index: int
    sample_id: str
    group_id: str
    sample_sha256: str
    split: str
    medium_type: str


@dataclass(frozen=True)
class SealedPrediction:
    path: Path
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class CoarseGateDecision:
    action: str
    reason: str
    thresholds: dict[str, float]


def _text(handle: h5py.File, name: str) -> np.ndarray:
    return np.asarray(handle[name].asstr()[:], dtype=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_validation_records(
    source_h5: str | Path,
    *,
    seed: int = 17,
    per_family: int = 1,
) -> tuple[CoarseRecordIndex, ...]:
    """Select a deterministic, group-distinct validation set in family order."""

    if int(per_family) < 1:
        raise ValueError("per_family must be positive")
    path = Path(source_h5).expanduser().resolve()
    with h5py.File(path, "r", swmr=True) as handle:
        family = _text(handle, "medium_type")
        split = _text(handle, "split")
        sample = _text(handle, "sample_id")
        group = _text(handle, "group_id")
        sample_hash = _text(handle, "sample_sha256")
    lengths = {len(values) for values in (family, split, sample, group, sample_hash)}
    if len(lengths) != 1:
        raise ValueError("coarse baseline index datasets have inconsistent lengths")

    rng = np.random.default_rng(int(seed))
    selected: list[CoarseRecordIndex] = []
    for name in ALLOWED_MEDIUM_TYPES:
        candidates = np.flatnonzero((split == "validation") & (family == name))
        rng.shuffle(candidates)
        seen: set[str] = set()
        for raw_index in candidates:
            index = int(raw_index)
            if group[index] in seen:
                continue
            selected.append(
                CoarseRecordIndex(
                    source_index=index,
                    sample_id=sample[index],
                    group_id=group[index],
                    sample_sha256=sample_hash[index],
                    split=split[index],
                    medium_type=family[index],
                )
            )
            seen.add(group[index])
            if len(seen) == int(per_family):
                break
        if len(seen) != int(per_family):
            raise ValueError(
                f"validation family {name} has only {len(seen)} distinct groups"
            )
    if len({row.group_id for row in selected}) != len(selected):
        raise ValueError("selected records reuse a medium group across families")
    return tuple(selected)


def select_all_validation_records(
    source_h5: str | Path,
    *,
    expected_family_counts: Mapping[str, int] = EXPECTED_VALIDATION_FAMILY_COUNTS,
) -> tuple[CoarseRecordIndex, ...]:
    """Return the complete allowed validation census in source-index order."""

    expected = {
        str(key): int(value) for key, value in expected_family_counts.items()
    }
    if set(expected) != set(ALLOWED_MEDIUM_TYPES) or min(expected.values()) < 1:
        raise ValueError("expected family counts must cover all allowed families")
    path = Path(source_h5).expanduser().resolve()
    with h5py.File(path, "r", swmr=True) as handle:
        family = _text(handle, "medium_type")
        split = _text(handle, "split")
        sample = _text(handle, "sample_id")
        group = _text(handle, "group_id")
        sample_hash = _text(handle, "sample_sha256")
    lengths = {len(values) for values in (family, split, sample, group, sample_hash)}
    if len(lengths) != 1:
        raise ValueError("coarse baseline index datasets have inconsistent lengths")
    rows = tuple(
        CoarseRecordIndex(
            source_index=index,
            sample_id=sample[index],
            group_id=group[index],
            sample_sha256=sample_hash[index],
            split=split[index],
            medium_type=family[index],
        )
        for index in range(len(family))
        if split[index] == "validation" and family[index] in ALLOWED_MEDIUM_TYPES
    )
    counts = {
        name: sum(row.medium_type == name for row in rows)
        for name in ALLOWED_MEDIUM_TYPES
    }
    if counts != expected:
        raise ValueError(f"allowed validation census mismatch: {counts} != {expected}")
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("allowed validation sample IDs are not unique")
    return rows


def shard_records(
    records: Sequence[CoarseRecordIndex],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[CoarseRecordIndex, ...]:
    """Assign ordered records to deterministic stride shards."""

    count = int(shard_count)
    index = int(shard_index)
    if count < 1 or index < 0 or index >= count:
        raise ValueError("shard index must lie in [0, shard_count)")
    return tuple(records[index::count])


def seal_prediction(
    path: str | Path,
    wavefield: torch.Tensor,
    *,
    metadata: Mapping[str, object],
) -> SealedPrediction:
    """Atomically seal a CPU prediction before any future truth is opened."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        torch.save(
            {
                "sealed": True,
                "wavefield": torch.as_tensor(wavefield)
                .detach()
                .to(device="cpu")
                .contiguous(),
                "metadata": dict(metadata),
            },
            partial,
        )
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        partial.unlink(missing_ok=True)
    return SealedPrediction(
        path=destination,
        sha256=_sha256_file(destination),
        byte_count=destination.stat().st_size,
    )


def accumulate_exact_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    families: Sequence[str],
    group_ids: Sequence[str],
    sample_ids: Sequence[str],
    source_onset_indices: Sequence[int],
    block_size: int = 20,
) -> dict[str, object]:
    """Accumulate official exact-time metrics in bounded time blocks."""

    predicted = torch.as_tensor(prediction)
    truth = torch.as_tensor(target, device=predicted.device)
    if predicted.shape != truth.shape or predicted.ndim != 4:
        raise ValueError("prediction and target must match [record,time,z,x]")
    if int(block_size) < 1:
        raise ValueError("block_size must be positive")
    accumulator = ExactWavefieldMetricAccumulator(
        require_unique=True,
        stored_time_count=predicted.shape[1],
    )
    for start in range(0, predicted.shape[1], int(block_size)):
        stop = min(start + int(block_size), predicted.shape[1])
        indices = torch.arange(start, stop, dtype=torch.long)[None].expand(
            predicted.shape[0], -1
        )
        accumulator.update(
            predicted[:, start:stop],
            truth[:, start:stop],
            families=families,
            group_ids=group_ids,
            sample_ids=sample_ids,
            time_indices=indices,
            source_onset_indices=source_onset_indices,
        )
    return accumulator.finalize()


def complete_field_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    block_size: int = 20,
) -> float:
    """Compute one record's joint space-time relative L2 in float64 blocks."""

    predicted = torch.as_tensor(prediction, device="cpu")
    truth = torch.as_tensor(target, device="cpu")
    if predicted.shape != truth.shape or predicted.ndim not in (3, 4):
        raise ValueError("complete fields must match [time,z,x] or [1,time,z,x]")
    if predicted.ndim == 4:
        if predicted.shape[0] != 1:
            raise ValueError("complete metric accepts one record at a time")
        predicted, truth = predicted[0], truth[0]
    if int(block_size) < 1:
        raise ValueError("block_size must be positive")
    error_square = 0.0
    target_square = 0.0
    for start in range(0, predicted.shape[0], int(block_size)):
        stop = min(start + int(block_size), predicted.shape[0])
        difference = predicted[start:stop].double() - truth[start:stop].double()
        error_square += float(difference.square().sum())
        target_square += float(truth[start:stop].double().square().sum())
    return math.sqrt(max(error_square, 0.0)) / math.sqrt(
        max(target_square, 1.0e-16)
    )


def merge_record_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_records: Sequence[CoarseRecordIndex],
) -> dict[str, object]:
    """Validate and aggregate exact per-record rows without averaging shards."""

    materialized = [dict(row) for row in rows]
    keys = [
        (int(row["source_index"]), str(row["sample_id"]))
        for row in materialized
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate record row")
    expected_by_key = {
        (record.source_index, record.sample_id): record
        for record in expected_records
    }
    if set(keys) != set(expected_by_key):
        raise ValueError("record row set does not match expected census")
    for row, key in zip(materialized, keys, strict=True):
        if str(row["family"]) != expected_by_key[key].medium_type:
            raise ValueError(f"record family mismatch for {key[1]}")
    if not all(
        bool(row["truth_opened_after_all_shard_predictions_sealed"])
        for row in materialized
    ):
        raise ValueError("seal-before-truth contract failed")
    stored_counts = {int(row["stored_time_count"]) for row in materialized}
    if stored_counts != {401}:
        raise ValueError("sealed panel must contain all 401 stored times")
    relative = np.asarray(
        [float(row["relative_l2"]) for row in materialized], dtype=np.float64
    )
    if not np.isfinite(relative).all() or bool((relative < 0.0).any()):
        raise ValueError("record relative L2 values must be finite and nonnegative")
    by_family = {
        family: [
            float(row["relative_l2"])
            for row in materialized
            if row["family"] == family
        ]
        for family in ALLOWED_MEDIUM_TYPES
    }
    if any(not values for values in by_family.values()):
        raise ValueError("every allowed family must have record metrics")
    family_metrics = {
        family: float(np.mean(values))
        for family, values in by_family.items()
    }
    aggregate = float(relative.mean())
    gate = decide_coarse_gate(aggregate, family_metrics)
    ordered = sorted(
        materialized,
        key=lambda row: float(row["relative_l2"]),
        reverse=True,
    )
    worst_by_family = {
        family: max(
            (row for row in materialized if row["family"] == family),
            key=lambda row: float(row["relative_l2"]),
        )
        for family in ALLOWED_MEDIUM_TYPES
    }
    return {
        "record_count": len(materialized),
        "stored_time_count": 401,
        "aggregate_relative_l2": aggregate,
        "family_relative_l2": family_metrics,
        "minimum_relative_l2": float(relative.min()),
        "median_relative_l2": float(np.median(relative)),
        "maximum_relative_l2": float(relative.max()),
        "worst_record": ordered[0],
        "worst_record_by_family": worst_by_family,
        "record_relative_l2": {
            str(row["sample_id"]): float(row["relative_l2"])
            for row in materialized
        },
        "gate": asdict(gate),
    }


def decide_coarse_gate(
    aggregate_relative_l2: float,
    family_relative_l2: Mapping[str, float],
) -> CoarseGateDecision:
    """Apply the prospective direct/correct/reject accuracy thresholds."""

    thresholds = {
        "direct_aggregate_lt": 0.10,
        "direct_family_lt": 0.12,
        "corrector_aggregate_lte": 0.35,
        "corrector_family_lt": 0.45,
    }
    aggregate = float(aggregate_relative_l2)
    families = {str(key): float(value) for key, value in family_relative_l2.items()}
    if set(families) != set(ALLOWED_MEDIUM_TYPES):
        raise ValueError("gate requires uniform, layered, and marmousi metrics")
    values = np.asarray([aggregate, *families.values()], dtype=np.float64)
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("gate metrics must be finite and nonnegative")
    if aggregate < 0.10 and max(families.values()) < 0.12:
        return CoarseGateDecision(
            action="direct_baseline",
            reason="accuracy target met",
            thresholds=thresholds,
        )
    if aggregate <= 0.35 and max(families.values()) < 0.45:
        return CoarseGateDecision(
            action="neural_corrector",
            reason="coarse field is within correction range",
            thresholds=thresholds,
        )
    return CoarseGateDecision(
        action="reject",
        reason="coarse field is outside correction range",
        thresholds=thresholds,
    )


__all__ = [
    "CoarseGateDecision",
    "CoarseRecordIndex",
    "EXPECTED_VALIDATION_FAMILY_COUNTS",
    "SealedPrediction",
    "accumulate_exact_metrics",
    "complete_field_relative_l2",
    "decide_coarse_gate",
    "merge_record_rows",
    "seal_prediction",
    "select_all_validation_records",
    "select_validation_records",
    "shard_records",
]
