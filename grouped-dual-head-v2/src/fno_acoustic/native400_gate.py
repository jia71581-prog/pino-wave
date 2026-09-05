"""Paired native-400 accuracy confirmation gates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import statistics
import string
from typing import Mapping, Sequence

import numpy as np

from .query_census import REQUIRED_NUMERIC_COLUMNS


METRICS = (
    "relative_l2",
    "relative_l2_q4",
    "receiver_relative_l2",
    "receiver_relative_l2_q4",
)
CATEGORIES = ("uniform", "layered", "marmousi")
SIGNED_NUMERIC_COLUMNS = frozenset(
    {
        "active_time_error_slope_per_s",
        "receiver_xcorr_peak",
        "receiver_phase_coherence",
    }
)
_REQUIRED_STABILITY = {
    "field_q4_q1_ratio",
    "receiver_q4_q1_ratio",
    "zero_relative_l2_q4",
    "zero_receiver_relative_l2_q4",
    "arrival_miss_rate",
    "receiver_lag_abs_s",
    "receiver_phase_error",
    "komega_high_q4",
    "prediction_finite",
    "prediction_nonzero",
    "output_height",
    "output_width",
    "output_time_steps",
}
_RUN_METADATA = (
    "seed",
    "predictor_family",
    "config_sha256",
    "checkpoint_sha256",
    "split",
    "split_manifest_sha256",
    "normalization_sha256",
    "receiver_geometry_sha256",
)
_SHARED_METADATA = (
    "split",
    "split_manifest_sha256",
    "normalization_sha256",
    "receiver_geometry_sha256",
)
_HASH_METADATA = tuple(name for name in _RUN_METADATA if name.endswith("_sha256"))


class _AggregatedCandidateRow(dict[str, object]):
    """Internal marker for the controlled median-of-three provenance label."""


@dataclass(frozen=True)
class NativeGateCell:
    category: str
    metric: str
    baseline_mean: float
    candidate_mean: float
    improvement: float
    ci_lower: float
    ci_upper: float
    passed: bool


@dataclass(frozen=True)
class NativeGateResult:
    cells: tuple[NativeGateCell, ...]
    stability: dict[str, bool]
    passed: bool
    classification: str


@dataclass(frozen=True)
class SeedGateResult:
    passed: bool
    passed_seed_count: int
    aggregate: NativeGateResult


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("threshold must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("threshold must be a finite number in [0, 1]")
    return result


def _sample_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("sample_id must be an integer")
    return value


def _run_metadata(rows: Sequence[Mapping[str, object]], name: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for field in _RUN_METADATA:
        values = []
        for row in rows:
            if field not in row:
                raise ValueError(f"{name} rows require metadata column {field}")
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) == "":
                raise ValueError(f"{name} metadata {field} must be a nonempty string or integer")
            values.append(value)
        if len(set(values)) != 1:
            raise ValueError(f"{name} metadata {field} must be internally consistent")
        metadata[field] = values[0]
    for field in _HASH_METADATA:
        value = metadata[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in string.hexdigits for character in value)
        ):
            raise ValueError(f"{name} metadata {field} must be a 64-character SHA-256 hex digest")
        metadata[field] = value.lower()
    if not isinstance(metadata["predictor_family"], str) or not isinstance(metadata["split"], str):
        raise ValueError(f"{name} predictor_family and split must be strings")
    return metadata


def _canonical_hash(values: Sequence[object]) -> str:
    payload = "\n".join(sorted(str(value).lower() for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_candidate_seed(
    rows: Sequence[Mapping[str, object]], metadata: Mapping[str, object], name: str
) -> int | None:
    value = metadata["seed"]
    if value == "median_of_three" and rows and all(
        isinstance(row, _AggregatedCandidateRow) for row in rows
    ):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} seed must be a canonical nonnegative decimal integer")
    if isinstance(value, int):
        if value >= 0:
            return value
        raise ValueError(f"{name} seed must be a canonical nonnegative decimal integer")
    if isinstance(value, str) and (value == "0" or (value[:1] in "123456789" and value.isdigit())):
        return int(value)
    raise ValueError(f"{name} seed must be a canonical nonnegative decimal integer")


def _index_rows(rows: Sequence[Mapping[str, object]], name: str) -> dict[tuple[str, int], Mapping[str, object]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError(f"{name} rows must be a sequence")
    indexed: dict[tuple[str, int], Mapping[str, object]] = {}
    seen_categories: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or "category" not in row or "sample_id" not in row:
            raise ValueError(f"{name} rows require category and sample_id")
        category = row["category"]
        if not isinstance(category, str):
            raise ValueError("category must be a string")
        seen_categories.add(category)
        key = (category, _sample_id(row["sample_id"]))
        if key in indexed:
            raise ValueError(f"duplicate {name} physical sample key: {key}")
        indexed[key] = row
    if seen_categories != set(CATEGORIES):
        raise ValueError(f"{name} categories must be exactly {CATEGORIES}")
    _run_metadata(rows, name)
    return indexed


def _finite_nonnegative(row: Mapping[str, object], column: str, context: str) -> float:
    if column not in row:
        raise ValueError(f"missing metric {column} for {context}")
    value = row[column]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {column} must be numeric for {context}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"metric {column} must be finite and nonnegative for {context}")
    return number


def validate_task9_numeric(column: str, value: object, context: str) -> float:
    """Validate one Task9 numeric value using its signed-column schema."""

    if column not in REQUIRED_NUMERIC_COLUMNS:
        raise ValueError(f"unknown Task9 numeric column: {column}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {column} must be a real non-boolean number for {context}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric {column} must be finite for {context}")
    if column not in SIGNED_NUMERIC_COLUMNS and number < 0.0:
        raise ValueError(f"metric {column} must be nonnegative for {context}")
    return number


def _paired_indexes(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[tuple[str, int], Mapping[str, object]], dict[tuple[str, int], Mapping[str, object]]]:
    baseline = _index_rows(baseline_rows, "baseline")
    candidate = _index_rows(candidate_rows, "candidate")
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate require identical unique category/sample_id pairs")
    baseline_metadata = _run_metadata(baseline_rows, "baseline")
    candidate_metadata = _run_metadata(candidate_rows, "candidate")
    _canonical_candidate_seed(candidate_rows, candidate_metadata, "candidate")
    for field in _SHARED_METADATA:
        if baseline_metadata[field] != candidate_metadata[field]:
            raise ValueError(f"baseline and candidate metadata {field} must match")
    return baseline, candidate


def _paired_bootstrap_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
    generator: np.random.Generator,
) -> tuple[float, float]:
    sample_count = baseline.size
    draws = np.empty(replicates, dtype=np.float64)
    max_index_entries = 1_000_000
    chunk_size = max(1, min(replicates, max_index_entries // sample_count))
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(0, sample_count, size=(stop - start, sample_count))
        baseline_means = baseline[indices].mean(axis=1, dtype=np.float64)
        if not np.all(np.isfinite(baseline_means)) or not np.all(baseline_means > 0.0):
            raise ValueError("bootstrap baseline means must be positive and finite")
        candidate_means = candidate[indices].mean(axis=1, dtype=np.float64)
        draws[start:stop] = 1.0 - candidate_means / baseline_means
    if not np.all(np.isfinite(draws)):
        raise ValueError("bootstrap improvements must be finite")
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(lower), float(upper)


def evaluate_native400_gate(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    threshold: float,
    bootstrap_replicates: int,
    seed: int,
) -> NativeGateResult:
    """Evaluate all category/metric cells using paired physical samples."""

    threshold = _threshold(threshold)
    replicates = _positive_int(bootstrap_replicates, "bootstrap_replicates")
    seed = _positive_int(seed, "seed", allow_zero=True)
    baseline, candidate = _paired_indexes(baseline_rows, candidate_rows)
    generator = np.random.default_rng(seed)
    cells: list[NativeGateCell] = []
    for category in CATEGORIES:
        ids = sorted(sample_id for row_category, sample_id in baseline if row_category == category)
        if len(ids) < 2:
            raise ValueError(f"category {category} requires at least two unique physical samples")
        for metric in METRICS:
            base = np.asarray(
                [_finite_nonnegative(baseline[(category, sample_id)], metric, f"{category}/{sample_id}") for sample_id in ids],
                dtype=np.float64,
            )
            cand = np.asarray(
                [_finite_nonnegative(candidate[(category, sample_id)], metric, f"{category}/{sample_id}") for sample_id in ids],
                dtype=np.float64,
            )
            if not np.all(base > 0.0):
                raise ValueError(f"every baseline primary metric must be strictly positive: {category}/{metric}")
            with np.errstate(over="ignore", invalid="ignore"):
                baseline_mean = float(base.mean(dtype=np.float64))
                candidate_mean = float(cand.mean(dtype=np.float64))
            if not math.isfinite(baseline_mean) or not math.isfinite(candidate_mean):
                raise ValueError(f"primary metric means must be finite: {category}/{metric}")
            improvement = 1.0 - candidate_mean / baseline_mean
            lower, upper = _paired_bootstrap_ci(base, cand, replicates, generator)
            passed = improvement >= threshold - 1e-12 and lower >= threshold - 1e-12
            cells.append(
                NativeGateCell(
                    category,
                    metric,
                    baseline_mean,
                    candidate_mean,
                    improvement,
                    lower,
                    upper,
                    bool(passed),
                )
            )
    stability = native_stability_guards(baseline, candidate)
    passed = all(cell.passed for cell in cells) and all(stability.values())
    point_passed = all(cell.improvement >= threshold - 1e-12 for cell in cells)
    classification = "confirmed" if passed else ("exploratory" if point_passed else "failed")
    return NativeGateResult(tuple(cells), stability, passed, classification)


def aggregate_seed_gates(
    seed_results: Sequence[NativeGateResult], aggregate: NativeGateResult
) -> SeedGateResult:
    if len(seed_results) != 3:
        raise ValueError("confirmation requires exactly three seed gates")
    if not all(isinstance(result, NativeGateResult) for result in seed_results):
        raise ValueError("seed_results must contain NativeGateResult values")
    if not isinstance(aggregate, NativeGateResult):
        raise ValueError("aggregate must be a NativeGateResult")
    passed_seed_count = sum(result.passed for result in seed_results)
    return SeedGateResult(passed_seed_count >= 2 and aggregate.passed, passed_seed_count, aggregate)


def aggregate_candidate_rows_by_sample(
    seed_rows: Sequence[Sequence[Mapping[str, object]]], reducer: str = "median"
) -> list[dict[str, object]]:
    if reducer != "median" or len(seed_rows) != 3:
        raise ValueError("confirmation aggregation requires the preregistered median of exactly three seeds")
    indexed = [_index_rows(rows, f"seed {index + 1}") for index, rows in enumerate(seed_rows)]
    if not (indexed[0].keys() == indexed[1].keys() == indexed[2].keys()):
        raise ValueError("all seeds require identical physical sample IDs")
    metadata = [_run_metadata(rows, f"seed {index + 1}") for index, rows in enumerate(seed_rows)]
    normalized_seeds = [
        _canonical_candidate_seed(rows, item, f"seed {index + 1}")
        for index, (rows, item) in enumerate(zip(seed_rows, metadata, strict=True))
    ]
    if len(set(normalized_seeds)) != 3:
        raise ValueError("three candidate CSVs require unique seeds")
    if len({item["checkpoint_sha256"] for item in metadata}) != 3:
        raise ValueError("three candidate CSVs require unique checkpoints")
    if len({item["predictor_family"] for item in metadata}) != 1:
        raise ValueError("three candidate CSVs require the same predictor_family")
    for field in _SHARED_METADATA:
        if len({item[field] for item in metadata}) != 1:
            raise ValueError(f"three candidate CSVs metadata {field} must match")
    aggregate_config_hash = _canonical_hash([item["config_sha256"] for item in metadata])
    aggregate_checkpoint_hash = _canonical_hash([item["checkpoint_sha256"] for item in metadata])
    output: list[dict[str, object]] = []
    for key in sorted(indexed[0]):
        row = _AggregatedCandidateRow(indexed[0][key])
        row["seed"] = "median_of_three"
        row["config_sha256"] = aggregate_config_hash
        row["checkpoint_sha256"] = aggregate_checkpoint_hash
        for metric in REQUIRED_NUMERIC_COLUMNS:
            row[metric] = float(
                statistics.median(
                    validate_task9_numeric(metric, mapping[key].get(metric), f"{key[0]}/{key[1]}")
                    for mapping in indexed
                )
            )
        output.append(row)
    return output


def _numeric_boolean(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number in (0.0, 1.0):
            return bool(number)
    raise ValueError(f"{name} must be a boolean or numeric 0/1")


def native_stability_guards(
    baseline_rows: Sequence[Mapping[str, object]] | Mapping[tuple[str, int], Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]] | Mapping[tuple[str, int], Mapping[str, object]],
) -> dict[str, bool]:
    if isinstance(baseline_rows, Mapping) and isinstance(candidate_rows, Mapping):
        baseline, candidate = dict(baseline_rows), dict(candidate_rows)
        if baseline.keys() != candidate.keys():
            raise ValueError("baseline and candidate require identical physical sample keys")
    elif not isinstance(baseline_rows, Mapping) and not isinstance(candidate_rows, Mapping):
        baseline, candidate = _paired_indexes(baseline_rows, candidate_rows)
    else:
        raise ValueError("baseline and candidate row containers must have the same kind")
    pairs = [(key[0], baseline[key], candidate[key]) for key in sorted(baseline)]
    if any(not _REQUIRED_STABILITY.issubset(base) or not _REQUIRED_STABILITY.issubset(cand)
           for _, base, cand in pairs):
        raise ValueError("native stability columns are incomplete")

    def mean(rows: Sequence[Mapping[str, object]], metric: str, category: str) -> float:
        return sum(_finite_nonnegative(row, metric, category) for row in rows) / len(rows)

    guards: dict[str, bool] = {}
    absolute_tolerances = {
        "receiver_lag_abs_s": 0.0032,
        "receiver_phase_error": 0.01,
        "komega_high_q4": 1e-8,
    }
    for category in CATEGORIES:
        category_pairs = [(base, cand) for row_category, base, cand in pairs if row_category == category]
        if not category_pairs:
            raise ValueError(f"missing stability rows for category {category}")
        base_rows, cand_rows = zip(*category_pairs, strict=True)
        prefix = f"{category}:"
        candidate_field_ratio = mean(cand_rows, "field_q4_q1_ratio", category)
        candidate_receiver_ratio = mean(cand_rows, "receiver_q4_q1_ratio", category)
        guards[prefix + "field_q4_q1"] = (
            candidate_field_ratio <= 1.5
            and candidate_field_ratio <= mean(base_rows, "field_q4_q1_ratio", category)
        )
        guards[prefix + "receiver_q4_q1"] = (
            candidate_receiver_ratio <= 1.5
            and candidate_receiver_ratio <= mean(base_rows, "receiver_q4_q1_ratio", category)
        )
        guards[prefix + "better_than_zero_q4"] = all(
            _finite_nonnegative(row, "relative_l2_q4", category)
            < _finite_nonnegative(row, "zero_relative_l2_q4", category)
            and _finite_nonnegative(row, "receiver_relative_l2_q4", category)
            < _finite_nonnegative(row, "zero_receiver_relative_l2_q4", category)
            for row in cand_rows
        )
        guards[prefix + "arrival_miss"] = mean(cand_rows, "arrival_miss_rate", category) <= mean(
            base_rows, "arrival_miss_rate", category
        )
        for guard_name, metric in (
            ("lag", "receiver_lag_abs_s"),
            ("phase", "receiver_phase_error"),
            ("high_k", "komega_high_q4"),
        ):
            baseline_mean = mean(base_rows, metric, category)
            limit = max(baseline_mean * 1.05, baseline_mean + absolute_tolerances[metric])
            guards[prefix + guard_name] = mean(cand_rows, metric, category) <= limit
        guards[prefix + "finite_nonzero"] = all(
            _numeric_boolean(row["prediction_finite"], "prediction_finite")
            and _numeric_boolean(row["prediction_nonzero"], "prediction_nonzero")
            for row in cand_rows
        )
        guards[prefix + "native_shape"] = all(
            (
                _finite_nonnegative(row, "output_height", category),
                _finite_nonnegative(row, "output_width", category),
                _finite_nonnegative(row, "output_time_steps", category),
            )
            == (400.0, 400.0, 160.0)
            for row in cand_rows
        )
    return guards
