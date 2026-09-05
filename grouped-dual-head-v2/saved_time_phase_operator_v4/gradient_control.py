"""Validated module-specific gradient trust limits and durable telemetry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch


@dataclass(frozen=True)
class PrefixClipMetrics:
    before: float
    after: float
    limit: float
    scale: float


@dataclass(frozen=True)
class GradientClipReport:
    total_before: float
    total_after: float
    prefixes: dict[str, PrefixClipMetrics]

    def as_dict(self) -> dict[str, object]:
        return {
            "total_before": self.total_before,
            "total_after": self.total_after,
            "prefixes": {
                name: asdict(value) for name, value in self.prefixes.items()
            },
        }


def _norm(parameters: list[torch.nn.Parameter]) -> float:
    values = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    return 0.0 if not values else float(torch.stack(values).norm(2))


def clip_gradients_by_prefix(
    model: torch.nn.Module,
    limits: Mapping[str, float],
) -> GradientClipReport:
    """Clip active gradients by top-level parameter prefix."""

    checked = {str(name): float(value) for name, value in limits.items()}
    if (
        "default" not in checked
        or not math.isfinite(checked["default"])
        or checked["default"] <= 0.0
    ):
        raise ValueError(
            "prefix gradient limits require a positive finite default"
        )
    if any(
        not math.isfinite(value) or value <= 0.0 for value in checked.values()
    ):
        raise ValueError("every prefix gradient limit must be positive and finite")

    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            groups.setdefault(name.split(".", 1)[0], []).append(parameter)

    before = {name: _norm(parameters) for name, parameters in groups.items()}
    for name, value in before.items():
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite gradient norm for prefix {name}")

    rows: dict[str, PrefixClipMetrics] = {}
    for name, parameters in groups.items():
        limit = checked.get(name, checked["default"])
        torch.nn.utils.clip_grad_norm_(parameters, limit)
        after = _norm(parameters)
        if not math.isfinite(after):
            raise FloatingPointError(
                f"non-finite post-clip gradient norm for prefix {name}"
            )
        rows[name] = PrefixClipMetrics(
            before=before[name],
            after=after,
            limit=limit,
            scale=1.0 if before[name] == 0.0 else after / before[name],
        )

    return GradientClipReport(
        total_before=math.sqrt(sum(value * value for value in before.values())),
        total_after=math.sqrt(
            sum(value.after * value.after for value in rows.values())
        ),
        prefixes=rows,
    )


__all__ = [
    "GradientClipReport",
    "PrefixClipMetrics",
    "clip_gradients_by_prefix",
]
