"""Exact stored-frame stratification and planned time-coverage audits."""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Sequence

import numpy as np


def rad_bin_budget(
    time_bin_errors: Sequence[float],
    total_active: int,
    *,
    k: float = 1.0,
    c: float = 1.0,
    floor_per_bin: int = 1,
) -> tuple[int, int, int]:
    """Residual-adaptive frame budget for the (early, middle, late) time bins.

    Splits ``total_active`` frames across the three bins with a RAD density
    ``p_bin ∝ (e_bin / mean(e))^k + c`` (same family as the fine-tune RAD
    sampler), so a high-error bin (e.g. late) receives more of the frame budget.
    ``c`` keeps every bin non-empty; ``floor_per_bin`` guarantees a hard minimum
    coverage.  Deterministic given the inputs; ties break toward later bins
    (the historically weaker regime).
    """

    errors = np.asarray(time_bin_errors, dtype=np.float64)
    if errors.shape != (3,):
        raise ValueError("time_bin_errors must have three entries (early, middle, late)")
    total = int(total_active)
    floor = int(floor_per_bin)
    if total < 3 * floor or floor < 0:
        raise ValueError("total_active must cover the per-bin floor for three bins")
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("time-bin errors must be finite and nonnegative")
    mean = float(errors.mean())
    if mean <= 0.0:
        weights = np.ones(3, dtype=np.float64)
    else:
        weights = np.power(errors / mean, max(0.0, float(k))) + max(0.0, float(c))
    weights = weights / weights.sum()
    # allocate the above-floor remainder by largest-remainder, ties to later bins
    remaining = total - 3 * floor
    raw = weights * remaining
    base = np.floor(raw).astype(np.int64)
    leftover = int(remaining - int(base.sum()))
    order = sorted(range(3), key=lambda i: (raw[i] - base[i], i), reverse=True)
    for j in range(leftover):
        base[order[j % 3]] += 1
    budget = base + floor
    return int(budget[0]), int(budget[1]), int(budget[2])


def _validated_axis(time_s: Sequence[float] | np.ndarray) -> np.ndarray:
    axis = np.asarray(time_s, dtype=np.float64)
    if axis.ndim != 1 or len(axis) < 5 or not np.isfinite(axis).all():
        raise ValueError("stored time axis must be a finite one-dimensional sequence")
    if np.any(np.diff(axis) <= 0):
        raise ValueError("stored time axis must be strictly increasing")
    return axis


def stored_time_indices(
    time_s: Sequence[float] | np.ndarray,
    *,
    source_t0_s: float,
    phase_offset: int,
) -> np.ndarray:
    """Select one exact index from pre-onset, early, middle and late bins."""

    axis = _validated_axis(time_s)
    if not np.isfinite(source_t0_s) or not float(axis[0]) < source_t0_s < float(axis[-1]):
        raise ValueError("source onset must lie inside the stored time axis")
    onset = int(np.searchsorted(axis, float(source_t0_s), side="left"))
    if onset <= 0 or onset >= len(axis):
        raise ValueError("source onset cannot form pre-onset and propagation bins")
    propagation_edges = np.linspace(onset, len(axis), num=4, dtype=np.int64)
    boundaries = (
        (0, onset),
        (int(propagation_edges[0]), int(propagation_edges[1])),
        (int(propagation_edges[1]), int(propagation_edges[2])),
        (int(propagation_edges[2]), len(axis)),
    )
    selected: list[int] = []
    for bin_index, (start, stop) in enumerate(boundaries):
        length = stop - start
        if length <= 0:
            raise ValueError("stored time bins must all be nonempty")
        position = (int(phase_offset) + bin_index * 31) % length
        selected.append(start + position)
    return np.asarray(selected, dtype=np.int64)


def _permuted_indices(
    values: np.ndarray,
    *,
    seed: int,
    sample_id: str,
    bin_name: str,
) -> np.ndarray:
    candidates = np.asarray(values, dtype=np.int64)
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError(f"stored time bin is empty: {bin_name}")
    payload = f"{int(seed)}:{sample_id}:{bin_name}".encode("utf8")
    stable_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(stable_seed).permutation(candidates)


def appearance_time_indices(
    time_s: Sequence[float] | np.ndarray,
    *,
    source_t0_s: float,
    source_f0_hz: float | None = None,
    sample_id: str,
    appearance: int,
    seed: int,
    count: int = 4,
    bin_frame_budget: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Choose exact indices with deterministic per-record coverage.

    The legacy four-frame policy is preserved.  Recovery panels of at least
    sixteen frames always include the first two stored frames at/after source
    onset, three contiguous four-frame propagation windows, one pre-onset
    causality audit frame, and deterministic additional exact stored frames.

    ``bin_frame_budget`` (early, middle, late) optionally overrides the uniform
    "one 4-frame window per bin" allocation with a residual-adaptive frame count
    per bin (see :func:`rad_bin_budget`).  The onset pair and pre-onset causality
    frame remain fixed, and the total returned index count is unchanged, so the
    query contract and coverage guarantees hold; only *which* active frames are
    chosen shifts toward the high-error bins.  ``None`` reproduces the legacy
    allocation bit-for-bit.
    """

    axis = _validated_axis(time_s)
    requested = int(count)
    if requested != 4 and not 16 <= requested <= len(axis):
        raise ValueError(
            "appearance time policy requires four or at least sixteen stored frames"
        )
    if int(appearance) < 0:
        raise ValueError("appearance must be nonnegative")
    if not str(sample_id):
        raise ValueError("sample_id is required for deterministic time coverage")
    if not np.isfinite(source_t0_s) or not float(axis[0]) < source_t0_s < float(axis[-1]):
        raise ValueError("source onset must lie inside the stored time axis")
    start_time = float(source_t0_s)
    if source_f0_hz is not None:
        frequency = float(source_f0_hz)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("source frequency must be positive and finite")
        start_time = max(float(axis[0]), start_time - 1.0 / frequency)
    onset = int(np.searchsorted(axis, start_time, side="left"))
    if onset < 0 or onset >= len(axis):
        raise ValueError("source onset cannot form propagation bins")
    early, middle, late = np.array_split(np.arange(onset, len(axis), dtype=np.int64), 3)
    index = int(appearance)
    if requested >= 16:
        if onset + 1 >= len(axis):
            raise ValueError("source onset cannot form the required onset pair")
        selected = {onset, onset + 1}
        # Per-bin active-frame budget: uniform 4-per-bin by default, or a
        # residual-adaptive allocation when bin_frame_budget is supplied.
        if bin_frame_budget is None:
            bin_counts = (4, 4, 4)
        else:
            bin_counts = tuple(int(value) for value in bin_frame_budget)
            if len(bin_counts) != 3 or any(value < 1 for value in bin_counts):
                raise ValueError("bin_frame_budget must be three positive counts")
        for name, phase, want in zip(
            ("early", "middle", "late"), (early, middle, late), bin_counts, strict=True
        ):
            start_min = int(phase[0]) + (2 if name == "early" else 0)
            starts = np.arange(start_min, int(phase[-1]) - 2, dtype=np.int64)
            order = _permuted_indices(
                starts,
                seed=seed,
                sample_id=str(sample_id),
                bin_name=f"recovery_{name}_window",
            )
            start = int(order[index % len(order)])
            # Take a contiguous run of `want` frames from the window start,
            # clamped inside the bin; extend along the bin order if clamped.
            hi = int(phase[-1])
            run = list(range(start, min(start + want, hi + 1)))
            oi = 0
            while len(run) < want and oi < len(order):
                candidate = int(order[(index + oi) % len(order)])
                if candidate not in run:
                    run.append(candidate)
                oi += 1
            selected.update(run[:want])
        if onset > 0:
            pre_order = _permuted_indices(
                np.arange(onset, dtype=np.int64),
                seed=seed,
                sample_id=str(sample_id),
                bin_name="recovery_pre",
            )
            selected.add(int(pre_order[index % len(pre_order)]))
        active_order = _permuted_indices(
            np.arange(onset, len(axis), dtype=np.int64),
            seed=seed,
            sample_id=str(sample_id),
            bin_name="recovery_active",
        )
        additional = requested - len(selected)
        active_offset = (index * max(1, additional)) % len(active_order)
        for candidate in np.roll(active_order, -active_offset):
            if len(selected) >= requested:
                break
            selected.add(int(candidate))
        if len(selected) < requested and onset > 0:
            pre_offset = (index * max(1, requested - len(selected))) % len(pre_order)
            for candidate in np.roll(pre_order, -pre_offset):
                selected.add(int(candidate))
                if len(selected) == requested:
                    break
        result = np.asarray(sorted(selected), dtype=np.int64)
        if len(result) > requested:
            # A residual-adaptive budget can overshoot via boundary overlap; trim
            # the surplus deterministically, never dropping the causal onset pair.
            protected = {onset, onset + 1}
            droppable = [int(v) for v in result if int(v) not in protected]
            # drop from the most-covered region first (stable, seed-free order)
            surplus = len(result) - requested
            drop = set(droppable[len(droppable) - surplus :])
            result = np.asarray(sorted(v for v in result.tolist() if v not in drop), dtype=np.int64)
        if len(result) != requested:
            raise RuntimeError(
                "recovery time policy did not select the requested unique indices"
            )
        return result

    early_order = _permuted_indices(
        early, seed=seed, sample_id=str(sample_id), bin_name="early"
    )
    primary_early = int(early_order[(2 * index) % len(early_order)])
    if (index + 1) % 5 == 0:
        pre_order = _permuted_indices(
            np.arange(onset, dtype=np.int64),
            seed=seed,
            sample_id=str(sample_id),
            bin_name="pre",
        )
        first = int(pre_order[(index // 5) % len(pre_order)])
    else:
        first = int(early_order[(2 * index + 1) % len(early_order)])
    middle_order = _permuted_indices(
        middle, seed=seed, sample_id=str(sample_id), bin_name="middle"
    )
    late_order = _permuted_indices(
        late, seed=seed, sample_id=str(sample_id), bin_name="late"
    )
    selected = np.asarray(
        (
            first,
            primary_early,
            int(middle_order[index % len(middle_order)]),
            int(late_order[index % len(late_order)]),
        ),
        dtype=np.int64,
    )
    if len(np.unique(selected)) != 4:
        raise RuntimeError("appearance time policy selected duplicate indices")
    return selected


def appearance_coverage_ledger(
    time_s: Sequence[float] | np.ndarray,
    *,
    source_t0_s: Sequence[float],
    sample_ids: Sequence[str],
    appearance_counts: Sequence[int],
    seed: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Return bit-packed exact-index coverage and a compact audit summary."""

    axis = _validated_axis(time_s)
    onsets = tuple(float(value) for value in source_t0_s)
    identifiers = tuple(str(value) for value in sample_ids)
    counts = tuple(int(value) for value in appearance_counts)
    if not onsets or not (len(onsets) == len(identifiers) == len(counts)):
        raise ValueError("coverage ledger fields must have the same nonzero length")
    coverage = np.zeros((len(onsets), len(axis)), dtype=np.uint8)
    for record, (onset, sample_id, appearances) in enumerate(
        zip(onsets, identifiers, counts, strict=True)
    ):
        if appearances <= 0:
            raise ValueError("appearance counts must be positive")
        for appearance in range(appearances):
            indices = appearance_time_indices(
                axis,
                source_t0_s=onset,
                sample_id=sample_id,
                appearance=appearance,
                seed=seed,
            )
            coverage[record, indices] = 1
    unique = coverage.sum(axis=1, dtype=np.int64)
    packed = np.packbits(coverage, axis=1, bitorder="little")
    return packed, {
        "record_count": len(onsets),
        "requested_time_count": len(axis),
        "minimum_unique_indices": int(unique.min()),
        "median_unique_indices": float(np.median(unique)),
        "maximum_unique_indices": int(unique.max()),
        "interpolated_requests": 0,
    }


def validation_time_indices(
    time_s: Sequence[float] | np.ndarray,
    *,
    source_t0_s: float,
    source_f0_hz: float | None = None,
    sample_id: str,
    panel_offset: int,
    seed: int,
    count: int = 16,
) -> np.ndarray:
    """Choose a fixed number of stratified exact validation indices."""

    axis = _validated_axis(time_s)
    requested = int(count)
    if requested < 4 or requested > len(axis):
        raise ValueError("validation time policy count is outside the stored axis")
    if int(panel_offset) < 0:
        raise ValueError("validation panel offset must be nonnegative")
    if not np.isfinite(source_t0_s) or not float(axis[0]) < source_t0_s < float(axis[-1]):
        raise ValueError("source onset must lie inside the stored time axis")
    start_time = float(source_t0_s)
    if source_f0_hz is not None:
        frequency = float(source_f0_hz)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("source frequency must be positive and finite")
        start_time = max(float(axis[0]), start_time - 1.0 / frequency)
    onset = int(np.searchsorted(axis, start_time, side="left"))
    if onset < 0 or onset >= len(axis):
        raise ValueError("source onset cannot form validation time bins")
    phases = np.array_split(np.arange(onset, len(axis), dtype=np.int64), 3)
    selected: list[int] = []
    if onset > 0:
        pre = _permuted_indices(
            np.arange(onset, dtype=np.int64),
            seed=seed,
            sample_id=str(sample_id),
            bin_name="validation_pre",
        )
        selected.append(int(pre[int(panel_offset) % len(pre)]))
    active_count = requested - len(selected)
    allocation = [active_count // 3] * 3
    for offset in range(active_count % 3):
        allocation[offset] += 1
    for name, phase, phase_count in zip(
        ("early", "middle", "late"), phases, allocation, strict=True
    ):
        order = _permuted_indices(
            phase,
            seed=seed,
            sample_id=str(sample_id),
            bin_name=f"validation_{name}",
        )
        if len(order) < phase_count:
            raise ValueError(f"validation {name} bin has too few frames")
        start = (int(panel_offset) * phase_count) % len(order)
        selected.extend(
            int(order[(start + offset) % len(order)]) for offset in range(phase_count)
        )
    result = np.asarray(selected, dtype=np.int64)
    if len(np.unique(result)) != requested:
        raise RuntimeError("validation time policy selected duplicate indices")
    return np.sort(result)


def coverage_summary(
    *,
    record_count: int,
    appearances: int,
    seed: int,
    time_s: Sequence[float] | np.ndarray | None = None,
) -> dict[str, int | float]:
    """Audit exact-index coverage for a deterministic planned schedule."""

    if record_count <= 0 or appearances <= 0:
        raise ValueError("record_count and appearances must be positive")
    axis = np.linspace(0.0, 1.0, 401) if time_s is None else _validated_axis(time_s)
    coverage = [set() for _ in range(int(record_count))]
    for record in range(int(record_count)):
        onset_fraction = ((record * 104729 + int(seed)) % 1000) / 999.0
        onset = float(axis[0]) + (float(axis[-1]) - float(axis[0])) * (
            0.05 + 0.14 * onset_fraction
        )
        for appearance in range(int(appearances)):
            indices = stored_time_indices(
                axis,
                source_t0_s=onset,
                phase_offset=int(seed) + record * 9176 + appearance,
            )
            coverage[record].update(int(index) for index in indices)
    counts = np.asarray([len(values) for values in coverage], dtype=np.int64)
    selected = set().union(*coverage)
    return {
        "record_count": int(record_count),
        "appearances_per_record": int(appearances),
        "requested_time_count": int(len(axis)),
        "minimum_unique_indices": int(counts.min()),
        "median_unique_indices": float(np.median(counts)),
        "maximum_unique_indices": int(counts.max()),
        "minimum_index": int(min(selected)),
        "maximum_index": int(max(selected)),
        "interpolated_requests": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--appearances", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            coverage_summary(
                record_count=args.records,
                appearances=args.appearances,
                seed=args.seed,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "appearance_coverage_ledger",
    "appearance_time_indices",
    "coverage_summary",
    "stored_time_indices",
    "validation_time_indices",
]
