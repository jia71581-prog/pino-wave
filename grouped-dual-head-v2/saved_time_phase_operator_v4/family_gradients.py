"""Evidence-calibrated controls for imbalanced medium-family gradients."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


FAMILIES = ("uniform", "layered", "marmousi")


def inverse_norm_family_weights(
    gradient_norms: Mapping[str, float],
) -> dict[str, float]:
    """Return inverse-gradient-norm weights normalized to arithmetic mean one."""

    if set(gradient_norms) != set(FAMILIES):
        raise ValueError("gradient norms must contain exactly three families")
    values = {name: float(gradient_norms[name]) for name in FAMILIES}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("family gradient norms must be finite and positive")
    inverse = {name: 1.0 / value for name, value in values.items()}
    mean = sum(inverse.values()) / len(inverse)
    return {name: inverse[name] / mean for name in FAMILIES}


def homogeneous_family_scale(
    labels: Sequence[str], weights: Mapping[str, float]
) -> float:
    """Resolve one family scale, allowing mixed labels only for equal weights."""

    family_labels = tuple(str(value) for value in labels)
    if set(weights) != set(FAMILIES):
        raise ValueError("family weights must contain exactly three families")
    values = {name: float(weights[name]) for name in FAMILIES}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("family weights must be finite and positive")
    if not family_labels or any(label not in values for label in family_labels):
        raise ValueError("family-weighted microbatch labels are invalid")
    if len(set(family_labels)) != 1:
        if len(set(values.values())) == 1:
            return values[FAMILIES[0]]
        raise ValueError("family-weighted microbatch must be homogeneous")
    family = family_labels[0]
    return values[family]


__all__ = [
    "FAMILIES",
    "homogeneous_family_scale",
    "inverse_norm_family_weights",
]
