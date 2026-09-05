"""Leakage-safe feedback control primitives for adaptive operator pretraining.

The controller deliberately uses a fixed train-only probe and changes only a
bounded family-by-frequency-block loss multiplier.  It is a full-information
online controller, not a claim that validation or test labels are rewards.
"""
from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


def stable_probe_positions(
    records: Sequence[tuple],
    families: Sequence[str],
    *,
    per_family: int,
    namespace: str,
) -> dict[str, list[int]]:
    """Select a deterministic, family-balanced train-only control probe."""

    if per_family <= 0:
        raise ValueError("per_family must be positive")
    selected: dict[str, list[int]] = {}
    for family in families:
        candidates = [
            (
                hashlib.sha256(f"{namespace}:{row[2]}".encode()).digest(),
                position,
            )
            for position, row in enumerate(records)
            if row[3] == family
        ]
        candidates.sort()
        if len(candidates) < per_family:
            raise ValueError(f"family {family!r} has fewer than {per_family} records")
        selected[family] = [position for _, position in candidates[:per_family]]
    return selected


def probe_selection_sha256(
    records: Sequence[tuple], selected: Mapping[str, Sequence[int]]
) -> str:
    """Bind the selected sample IDs without exposing any target values."""

    payload = "\n".join(
        f"{family}:{records[position][2]}"
        for family in sorted(selected)
        for position in selected[family]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def initial_controller_state(family_count: int, block_count: int) -> dict:
    if family_count <= 0 or block_count <= 0:
        raise ValueError("controller dimensions must be positive")
    return {
        "schema": "adaptive_pretraining_controller_state_v1",
        "weights": np.ones((family_count, block_count), dtype=np.float64).tolist(),
        "ema_energy": None,
        "previous_objective": None,
        "evaluations": 0,
        "last_epoch": None,
    }


def _as_energy_matrix(metrics: Mapping[str, object]) -> np.ndarray:
    values = np.asarray(metrics["energy"], dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("energy metrics must be a non-empty family-by-block matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("energy metrics must be finite and nonnegative")
    return np.maximum(values, 1.0e-16)


def controller_objective(energy, *, worst_weight: float) -> dict[str, float]:
    """Return an auditable minimax-oriented objective in relative-error units."""

    if not 0.0 <= float(worst_weight) <= 1.0:
        raise ValueError("worst_weight must lie in [0, 1]")
    values = np.asarray(energy, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("energy must be a finite nonnegative matrix")
    relative = np.sqrt(np.maximum(values, 0.0))
    aggregate = float(np.mean(relative))
    worst = float(np.max(relative))
    objective = (1.0 - float(worst_weight)) * aggregate + float(worst_weight) * worst
    return {"aggregate": aggregate, "worst": worst, "objective": objective}


def _bounded_geometric_normalize(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for _ in range(8):
        result = np.clip(result, lower, upper)
        geometric_mean = math.exp(float(np.mean(np.log(result))))
        result /= geometric_mean
    return np.clip(result, lower, upper)


def update_controller(
    metrics: Mapping[str, object],
    previous: Mapping[str, object] | None,
    config: Mapping[str, object],
    *,
    epoch: int,
) -> tuple[dict, dict]:
    """Apply one bounded exponentiated-feedback controller update.

    Larger train-probe errors receive larger next-epoch loss multipliers.  EMA,
    per-epoch action clipping, global clipping, and geometric normalization keep
    the feedback from collapsing coverage or changing the global loss scale.
    """

    energy = _as_energy_matrix(metrics)
    family_count, block_count = energy.shape
    state = (
        dict(previous)
        if previous is not None
        else initial_controller_state(family_count, block_count)
    )
    if state.get("schema") != "adaptive_pretraining_controller_state_v1":
        raise ValueError("unsupported controller state schema")
    weights = np.asarray(state["weights"], dtype=np.float64)
    if weights.shape != energy.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("controller weights do not match probe metrics")

    ema_beta = float(config["ema_beta"])
    gain = float(config["gain"])
    max_step_ratio = float(config["max_step_ratio"])
    lower = float(config["min_multiplier"])
    upper = float(config["max_multiplier"])
    worst_weight = float(config["worst_weight"])
    if not 0.0 <= ema_beta < 1.0:
        raise ValueError("ema_beta must lie in [0, 1)")
    if gain < 0.0 or max_step_ratio < 1.0 or not 0.0 < lower <= 1.0 <= upper:
        raise ValueError("invalid controller bounds")

    old_ema = state.get("ema_energy")
    if old_ema is None:
        ema = energy
    else:
        old_ema_array = np.asarray(old_ema, dtype=np.float64)
        if old_ema_array.shape != energy.shape:
            raise ValueError("ema_energy does not match probe metrics")
        ema = ema_beta * old_ema_array + (1.0 - ema_beta) * energy

    log_error = 0.5 * np.log(np.maximum(ema, 1.0e-16))
    centered = log_error - float(np.mean(log_error))
    raw_action = np.exp(gain * centered)
    action = np.clip(raw_action, 1.0 / max_step_ratio, max_step_ratio)
    new_weights = _bounded_geometric_normalize(weights * action, lower, upper)

    objective = controller_objective(energy, worst_weight=worst_weight)
    previous_objective = state.get("previous_objective")
    reward = (
        None
        if previous_objective is None
        else float(previous_objective) - float(objective["objective"])
    )
    new_state = {
        "schema": "adaptive_pretraining_controller_state_v1",
        "weights": new_weights.tolist(),
        "ema_energy": ema.tolist(),
        "previous_objective": float(objective["objective"]),
        "evaluations": int(state.get("evaluations", 0)) + 1,
        "last_epoch": int(epoch),
    }
    event = {
        "epoch": int(epoch),
        "reward": reward,
        "objective": objective,
        "weights_before": weights.tolist(),
        "action": action.tolist(),
        "weights_after": new_weights.tolist(),
        "weight_min": float(np.min(new_weights)),
        "weight_max": float(np.max(new_weights)),
    }
    return new_state, event


__all__ = [
    "controller_objective",
    "initial_controller_state",
    "probe_selection_sha256",
    "stable_probe_positions",
    "update_controller",
]
