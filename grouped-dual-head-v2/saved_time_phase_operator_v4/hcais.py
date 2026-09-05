"""Coverage-constrained adaptive importance sampling for Transfer DG."""
from __future__ import annotations

import numpy as np


def within_stratum_percentile(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    group = np.asarray(strata).reshape(-1)
    if value.shape != group.shape or not np.isfinite(value).all():
        raise ValueError("values and strata must be finite aligned vectors")
    output = np.empty_like(value)
    for key in np.unique(group):
        index = np.flatnonzero(group == key)
        ordered = np.sort(value[index])
        left = np.searchsorted(ordered, value[index], side="left")
        right = np.searchsorted(ordered, value[index], side="right")
        output[index] = (0.5 * (left + right) + 0.5) / float(len(index))
    return output


def _equal_stratum_distribution(scores: np.ndarray, strata: np.ndarray) -> np.ndarray:
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    group = np.asarray(strata).reshape(-1)
    if score.shape != group.shape or not np.isfinite(score).all() or np.any(score < 0.0):
        raise ValueError("scores and strata must be nonnegative finite vectors")
    keys = np.unique(group)
    output = np.zeros_like(score)
    for key in keys:
        index = np.flatnonzero(group == key)
        local = score[index]
        total = float(local.sum())
        if total <= 0.0:
            local = np.ones_like(local)
            total = float(len(local))
        output[index] = local / total / float(len(keys))
    return output


def coverage_distribution(strata: np.ndarray) -> np.ndarray:
    """Half uniform-risk coverage and half equal-stratum coverage."""
    group = np.asarray(strata).reshape(-1)
    if group.size == 0:
        raise ValueError("strata cannot be empty")
    uniform = np.full(group.size, 1.0 / float(group.size), dtype=np.float64)
    stratified = _equal_stratum_distribution(np.ones(group.size), group)
    return 0.5 * uniform + 0.5 * stratified


def regularized_leverage_scores(features: np.ndarray, *, ridge: float = 1.0e-3) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or ridge <= 0.0:
        raise ValueError("features must be [N,D] and ridge positive")
    if not np.isfinite(value).all():
        raise ValueError("leverage features must be finite")
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    normalized = value / np.maximum(norm, 1.0e-12)
    gram = normalized.T @ normalized / float(len(normalized))
    gram.flat[:: gram.shape[0] + 1] += float(ridge)
    solved = np.linalg.solve(gram, normalized.T).T
    leverage = np.sum(normalized * solved, axis=1)
    return np.maximum(leverage, 0.0)


def hcais_distribution(
    difficulty: np.ndarray,
    leverage: np.ndarray,
    strata: np.ndarray,
    *,
    coverage_mix: float = 0.40,
    temperature_power: float = 1.0,
) -> np.ndarray:
    if not 0.0 <= coverage_mix <= 1.0 or temperature_power < 0.0:
        raise ValueError("invalid HCAIS mixture controls")
    difficulty = np.asarray(difficulty, dtype=np.float64).reshape(-1)
    leverage = np.asarray(leverage, dtype=np.float64).reshape(-1)
    strata = np.asarray(strata).reshape(-1)
    if not (difficulty.shape == leverage.shape == strata.shape):
        raise ValueError("HCAIS vectors must align")
    q_coverage = coverage_distribution(strata)
    q_difficulty = _equal_stratum_distribution(
        np.power(np.maximum(difficulty, 0.0) + 1.0e-6, temperature_power),
        strata,
    )
    q_leverage = _equal_stratum_distribution(leverage + 1.0e-6, strata)
    adaptive = (2.0 / 3.0) * q_difficulty + (1.0 / 3.0) * q_leverage
    output = coverage_mix * q_coverage + (1.0 - coverage_mix) * adaptive
    output /= output.sum()
    return output


def inverse_probability_weights(indices: np.ndarray, distribution: np.ndarray) -> np.ndarray:
    probability = np.asarray(distribution, dtype=np.float64).reshape(-1)
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if probability.size == 0 or np.any(probability <= 0.0):
        raise ValueError("sampling distribution must be strictly positive")
    if np.any(selected < 0) or np.any(selected >= probability.size):
        raise ValueError("sample index outside distribution")
    return 1.0 / (float(probability.size) * probability[selected])


def effective_sample_size(weights: np.ndarray) -> float:
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError("importance weights must be finite and nonnegative")
    return float(value.sum() ** 2 / max(float(np.square(value).sum()), 1.0e-16))


__all__ = [
    "coverage_distribution",
    "effective_sample_size",
    "hcais_distribution",
    "inverse_probability_weights",
    "regularized_leverage_scores",
    "within_stratum_percentile",
]
