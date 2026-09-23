"""Release-side evaluation closure helpers (plan section 2.5, DEFECTS 3-5).

This module is intentionally free of torch/model dependencies so that every
rule it encodes can be unit tested without a GPU.  It provides:

* ``resolve_onset_frame`` -- the IC start frame of a record, derived from the
  record's own time axis and ``source_t0_s``.  ``make_devsets_v2.py`` never
  emits an ``onset_model`` field, so nothing may read one back.
* ``checked_onset`` -- same, but also audits a pre-existing cached onset field
  when the caller happens to have one.
* ``enumerate_split_records`` -- every record of a split, exactly once, in a
  deterministic order.  The published 40-step schedule presents 480 slots but
  covers only 380 distinct validation records (plan section 0.2), so a
  schedule-shaped enumeration can never stand in for a full enumeration.
* ``future_window`` / ``assert_truth_shape`` -- the evaluation window from the
  record's own IC endpoint, and the ``[T, Z, X]`` shape contract that the old
  ``read_wavefield(...).values[0]`` silently broke.
* ``family_equal_aggregate`` / ``record_equal_aggregate`` -- the two aggregate
  weightings the plan requires to be reported separately.
* ``schedule_coverage`` -- what a schedule-shaped pass actually covered, so a
  sampler can never be quoted as an enumeration.
* ``pixel_extent`` / ``node_position`` -- node/pixel boundary convention for
  plotting, and the matching coordinate for markers on the same panel.
* ``all_rank_mean`` / ``rank_mean_available`` -- the correct cross-rank
  reduction for a per-step series, and whether it was really all-rank.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "assert_truth_shape",
    "all_rank_mean",
    "checked_onset",
    "enumerate_split_records",
    "family_equal_aggregate",
    "future_window",
    "ic_window",
    "node_position",
    "pixel_extent",
    "rank_mean_available",
    "record_equal_aggregate",
    "resolve_onset_frame",
    "schedule_coverage",
]


# ---------------------------------------------------------------------------
# DEFECT 3: onset from the record's own time axis
# ---------------------------------------------------------------------------
def resolve_onset_frame(time_s, source_t0_s: float) -> int:
    """First stored frame at or after the source's t0, clamped to the axis.

    This mirrors ``PilotBatchDataset._ic_window`` and the inference entry point
    (``min(searchsorted(axis, source_t0_s), len(axis) - 1)``) so every consumer
    computes the same IC start from the same inputs.
    """
    axis = np.asarray(time_s, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError("time axis must be a nonempty 1-D vector")
    if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError("time axis must be finite and strictly increasing")
    if not np.isfinite(float(source_t0_s)):
        raise ValueError("source_t0_s must be finite")
    return int(min(np.searchsorted(axis, float(source_t0_s), side="left"), axis.size - 1))


def checked_onset(time_s, source_t0_s: float, cached: object | None = None) -> int:
    """Resolve the onset frame and audit an optional cached onset field.

    The release's dev-set JSON has no ``onset_model`` field; if a caller has
    one anyway (for example an older monitor file) the derived value is
    authoritative, and a disagreement is a hard error rather than a silent
    override.
    """
    onset = resolve_onset_frame(time_s, source_t0_s)
    if cached is not None:
        if isinstance(cached, bool) or not isinstance(cached, (int, np.integer)):
            raise TypeError("cached onset must be an integer frame index")
        if int(cached) != onset:
            raise ValueError(
                f"cached onset {int(cached)} disagrees with the derived onset {onset}"
            )
    return onset


def ic_window(time_s, onset: int, ic_frames: int) -> np.ndarray:
    """Index array of the ``ic_frames`` stored frames starting at ``onset``."""
    axis = np.asarray(time_s, dtype=np.float64)
    if ic_frames <= 0:
        raise ValueError("ic_frames must be positive")
    if onset < 0 or onset + ic_frames > axis.size:
        raise ValueError("IC window exceeds the stored time axis")
    return np.arange(onset, onset + ic_frames, dtype=np.int64)


def future_window(time_s, onset: int, ic_frames: int) -> np.ndarray:
    """Index array of every frame strictly after the IC endpoint.

    ``ic_frames`` must match the IC length the model was given, so the first
    future frame is ``onset + ic_frames`` for every record.
    """
    axis = np.asarray(time_s, dtype=np.float64)
    ic = ic_window(axis, onset, ic_frames)
    if onset + ic_frames >= axis.size:
        raise ValueError("a record must keep at least one future frame")
    return np.arange(onset + ic_frames, axis.size, dtype=np.int64)


# ---------------------------------------------------------------------------
# DEFECT 2: the [T, Z, X] truth contract
# ---------------------------------------------------------------------------
def assert_truth_shape(values, time_s, z_m, x_m, *, label: str = "truth") -> None:
    """Raise unless ``values`` is exactly ``[T, Z, X]`` on the record's axes.

    The old scoring code did ``read_wavefield(...).values[0]``, which turned a
    ``[T, Z, X]`` array into ``[Z, X]`` and then indexed it by time.  This
    assertion makes that class of bug impossible to reintroduce silently.
    """
    shape = tuple(np.shape(values))
    expected = (len(time_s), len(z_m), len(x_m))
    if shape != expected:
        raise ValueError(
            f"{label} must have shape [T,Z,X]={expected}, got {shape}; a leading "
            "index such as .values[0] drops the time axis"
        )


# ---------------------------------------------------------------------------
# DEFECT 4: complete, non-duplicated enumeration
# ---------------------------------------------------------------------------
def enumerate_split_records(
    records: Sequence[object],
    *,
    split: str,
    families: Iterable[str] | None = None,
    key: Callable[[object], str] = lambda record: record.sample_id,
) -> list[int]:
    """Filtered positions of every record of ``split`` exactly once, in stable order.

    ``records`` is the manifest's record sequence and the returned positions
    index the *filtered split view* -- the same index space as
    ``V3WavefieldDataset(...).records`` and ``build_pilot_schedule``, which both
    count only the records that survive the split and family filters.  Getting
    this wrong is easy and silent: the manifest's own raw positions for the
    validation split start at 2240, so an unfiltered index would address train
    records.

    Duplicates are impossible by construction and the function refuses to
    return a partial enumeration: every record the filter accepts is included.

    The returned positions are provably ``list(range(len(accepted)))``: they are
    appended in acceptance order with the running length, and the filters only
    depend on ``record.split`` and ``record.medium_type``.  That equality is the
    whole reason the index space is shared with the dataset, so it is asserted
    here rather than left as a comment -- if a future filter ever makes the
    acceptance order differ from the dataset's own build order, this raises
    instead of silently scoring the wrong wavefield.
    """
    wanted = None if families is None else {str(value) for value in families}
    excluded = {"anomaly"}
    selected: list[int] = []
    seen: dict[str, int] = {}
    for position, record in enumerate(records):
        if record.split != split:
            continue
        if wanted is not None and record.medium_type not in wanted:
            continue
        # Records outside the V3 family contract are dropped by the dataset too,
        # so they must not occupy a position here either.
        if record.medium_type in excluded:
            continue
        identity = str(key(record))
        if identity in seen:
            raise ValueError(
                f"record {identity!r} appears twice in the {split} manifest "
                f"(filtered positions {seen[identity]} and {len(selected)})"
            )
        seen[identity] = len(selected)
        selected.append(len(selected))
    if not selected:
        raise ValueError(f"the {split} split has no records for families {families}")
    if selected != list(range(len(selected))):
        raise RuntimeError(
            f"the {split} enumeration is not the dataset's positional index space"
        )
    return selected


def schedule_coverage(schedule: Sequence[Sequence[int]], total: int) -> dict[str, object]:
    """Audit what a schedule-shaped enumeration actually covers.

    Returns the slot count, the distinct-record count, the multiplicity
    histogram and the positions that were never presented.  A schedule is
    *not* an enumeration: for the published 40-step validation schedule this
    reports 480 slots over 380 distinct records out of 480.
    """
    slots = [int(index) for step in schedule for index in step]
    counts = Counter(slots)
    missing = sorted(set(range(int(total))) - set(counts))
    return {
        "slots": len(slots),
        "distinct_records": len(counts),
        "duplicated_slots": len(slots) - len(counts),
        "max_multiplicity": max(counts.values()) if counts else 0,
        "missing_records": missing,
        "complete": not missing and len(slots) == len(counts),
    }


# ---------------------------------------------------------------------------
# DEFECT 4: both aggregate weightings
# ---------------------------------------------------------------------------
def _record_weighted_mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def record_equal_aggregate(per_record: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Mean over records inside each family, then over families (equal weight).

    ``per_record`` items carry ``family`` and ``value`` (a per-record
    relative-L2 style scalar).  Families with no records are reported as
    ``None`` and are excluded from the family mean, never silently treated as
    zero.
    """
    if not per_record:
        raise ValueError("per-record aggregate needs at least one record")
    families: dict[str, list[float]] = {}
    for item in per_record:
        if "family" not in item or "value" not in item:
            raise KeyError("per-record items need 'family' and 'value'")
        families.setdefault(str(item["family"]), []).append(item["value"])
    per_family = {name: _record_weighted_mean(values) for name, values in sorted(families.items())}
    return {
        "records": len(per_record),
        "per_family_record_mean": per_family,
        "family_counts": {name: len(values) for name, values in sorted(families.items())},
    }


def family_equal_aggregate(per_record: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Equal weight per family, and the record-equal mean, side by side.

    Both numbers are returned from one call so a report can never quote one
    while describing the other.  ``family_equal_weight`` averages the per-family
    record means; ``record_equal_weight`` pools every record.
    """
    summary = record_equal_aggregate(per_record)
    means = [value for value in summary["per_family_record_mean"].values() if value is not None]
    return {
        "records": summary["records"],
        "family_counts": summary["family_counts"],
        "per_family_record_mean": summary["per_family_record_mean"],
        "family_equal_weight": float(np.mean(means)) if means else None,
        "record_equal_weight": _record_weighted_mean([item["value"] for item in per_record]),
        "definition": {
            "family_equal_weight": "mean over families of the per-family record mean",
            "record_equal_weight": "mean over all records, ignoring family sizes",
        },
    }


# ---------------------------------------------------------------------------
# DEFECT 6: node/pixel boundary convention for plots
# ---------------------------------------------------------------------------
def pixel_extent(axis) -> tuple[float, float]:
    """Half-cell padded ``(low, high)`` extent of an imshow node axis.

    ``imshow`` places pixel *centres* at ``0, dx, ..., (n-1)*dx``; passing
    ``[0, n*dx]`` as the extent therefore draws the centres correctly but
    mislabels the outer half-cell as data.  The honest boundary is half a cell
    outside the first and last node: for 201 nodes at 10 m this is
    ``(-5.0, 2005.0)``, not ``(0.0, 2010.0)``.
    """
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("a pixel extent needs a 1-D axis with at least two nodes")
    steps = np.diff(values)
    if not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-12) or steps[0] <= 0:
        raise ValueError("pixel extent requires a uniform, increasing axis")
    return float(values[0] - steps[0] / 2.0), float(values[-1] + steps[0] / 2.0)


def node_position(axis, value: float) -> float:
    """Map a physical coordinate to continuous node coordinates for imshow.

    An ``imshow`` panel drawn with ``extent=pixel_extent(axis)`` (or any extent
    built from uniform node centres) places the *centre* of column ``j`` at that
    column's physical coordinate, so a physical coordinate ``v`` sits at
    continuous node position ``(v - axis[0]) / step - 0.5``.  Markers must use
    this convention, not ``v / step``, or they land half a cell away from the
    data they are supposed to annotate.
    """
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("node_position needs a 1-D axis with at least two nodes")
    step = float(values[1] - values[0])
    if step <= 0:
        raise ValueError("node_position requires an increasing axis")
    return (float(value) - float(values[0])) / step - 0.5


# ---------------------------------------------------------------------------
# DEFECT 5: cross-rank reduction of a per-step series
# ---------------------------------------------------------------------------
def all_rank_mean(per_rank_values: Sequence[float | None]) -> float | None:
    """Mean over the ranks that actually contributed a value.

    A DDP evaluation writes one series per step; the published series was the
    value of the *writing* rank's own record shard, not the all-rank mean of
    the step.  Reducing here is explicit about which ranks contributed and
    refuses a non-finite entry instead of averaging it into the result.
    """
    if not per_rank_values:
        raise ValueError("all_rank_mean needs at least one rank entry")
    finite: list[float] = []
    for value in per_rank_values:
        if value is None:
            continue
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("all_rank_mean refuses non-finite rank values")
        finite.append(number)
    return float(np.mean(finite)) if finite else None


def rank_mean_available(per_rank_values: Sequence[float | None]) -> bool:
    """True when every rank contributed, i.e. the mean is a true all-rank mean."""
    return all(value is not None for value in per_rank_values)
