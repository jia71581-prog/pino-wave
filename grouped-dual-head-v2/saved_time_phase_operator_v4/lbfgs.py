"""Policies for deterministic, dense-head-only V4 L-BFGS refinement."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Generic, TypeVar

from torch import nn


BatchT = TypeVar("BatchT")


def freeze_for_dense_lbfgs(model: nn.Module) -> tuple[nn.Parameter, ...]:
    decoder = getattr(model, "dense_decoder", None)
    if not isinstance(decoder, nn.Module):
        raise ValueError("L-BFGS model has no dense_decoder module")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = tuple(decoder.parameters())
    if not selected:
        raise ValueError("dense_decoder has no parameters")
    for parameter in selected:
        parameter.requires_grad_(True)
    return selected


def freeze_for_band_adapter_readout_lbfgs(
    model: nn.Module,
) -> tuple[nn.Parameter, ...]:
    """Freeze an operator except for the three scalar band-adapter readouts."""

    decoder = getattr(model, "dense_decoder", None)
    adapter = getattr(decoder, "band_limited_adapter", None)
    experts = getattr(adapter, "experts", None)
    if not isinstance(experts, nn.ModuleList) or len(experts) != 3:
        raise ValueError("band-adapter L-BFGS requires exactly three experts")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[nn.Parameter] = []
    for expert in experts:
        output = getattr(expert, "output", None)
        if not isinstance(output, nn.Module):
            raise ValueError("band-adapter expert has no output readout")
        parameters = tuple(output.parameters())
        if not parameters:
            raise ValueError("band-adapter output readout has no parameters")
        for parameter in parameters:
            parameter.requires_grad_(True)
        selected.extend(parameters)
    return tuple(selected)


def effective_batch_records(*, macro_records: int, accumulated_macros: int) -> int:
    if macro_records <= 0 or accumulated_macros <= 0:
        raise ValueError("batch counts must be positive")
    return int(macro_records) * int(accumulated_macros)


def _signature(batches: Sequence[object]) -> tuple[tuple[str, ...], ...]:
    if not batches:
        raise ValueError("fixed closure batch cannot be empty")
    result = tuple(tuple(str(value) for value in getattr(batch, "sample_id")) for batch in batches)
    if any(not values for values in result):
        raise ValueError("fixed closure batch contains no sample identifiers")
    return result


@dataclass(frozen=True)
class FixedClosureBatch(Generic[BatchT]):
    batches: tuple[BatchT, ...]
    sample_signature: tuple[tuple[str, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_signature", _signature(self.batches))

    def verify(self, batches: Sequence[BatchT]) -> None:
        if _signature(batches) != self.sample_signature:
            raise ValueError("L-BFGS closure batch changed during line search")


def refinement_gate(
    *,
    parent_score: float,
    candidate_score: float,
    parent_family: Mapping[str, float],
    candidate_family: Mapping[str, float],
    minimum_relative_improvement: float = 0.02,
    family_regression_tolerance: float = 0.03,
    maximum_candidate_score: float | None = None,
    maximum_candidate_family_score: float | None = None,
) -> dict[str, object]:
    scalar_values = (float(parent_score), float(candidate_score))
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("refinement scores must be finite")
    if parent_score <= 0 or candidate_score < 0:
        raise ValueError("refinement scores must be nonnegative with a positive parent")
    if not 0 <= minimum_relative_improvement < 1 or family_regression_tolerance < 0:
        raise ValueError("refinement gate thresholds are invalid")
    families = ("uniform", "layered", "marmousi")
    if any(name not in parent_family or name not in candidate_family for name in families):
        raise ValueError("refinement gate is missing a medium family")
    family_values = tuple(
        float(mapping[name])
        for mapping in (parent_family, candidate_family)
        for name in families
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in family_values):
        raise ValueError("refinement family scores must be finite and nonnegative")
    aggregate_target = (
        None if maximum_candidate_score is None else float(maximum_candidate_score)
    )
    family_target = (
        None
        if maximum_candidate_family_score is None
        else float(maximum_candidate_family_score)
    )
    if aggregate_target is not None and (
        not math.isfinite(aggregate_target) or aggregate_target <= 0.0
    ):
        raise ValueError("maximum_candidate_score must be finite and positive")
    if family_target is not None and (
        not math.isfinite(family_target) or family_target <= 0.0
    ):
        raise ValueError(
            "maximum_candidate_family_score must be finite and positive"
        )
    improvement = (float(parent_score) - float(candidate_score)) / float(parent_score)
    family_safe = all(
        float(candidate_family[name])
        <= (1.0 + family_regression_tolerance) * float(parent_family[name])
        for name in families
    )
    aggregate_target_met = (
        aggregate_target is None or float(candidate_score) <= aggregate_target
    )
    family_target_met = family_target is None or all(
        float(candidate_family[name]) <= family_target for name in families
    )
    return {
        "passed": (
            improvement >= minimum_relative_improvement
            and family_safe
            and aggregate_target_met
            and family_target_met
        ),
        "relative_improvement": improvement,
        "family_safe": family_safe,
        "absolute_aggregate_accuracy": aggregate_target_met,
        "absolute_family_accuracy": family_target_met,
        "maximum_candidate_score": aggregate_target,
        "maximum_candidate_family_score": family_target,
    }


__all__ = [
    "FixedClosureBatch",
    "effective_batch_records",
    "freeze_for_band_adapter_readout_lbfgs",
    "freeze_for_dense_lbfgs",
    "refinement_gate",
]
