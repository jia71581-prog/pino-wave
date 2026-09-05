"""Sealed accuracy gates that prevent invalid V3 pilot launches."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class OneRecordGateMetrics:
    sample_id: str
    medium_type: str
    early_relative_l2: float
    middle_relative_l2: float
    late_relative_l2: float
    query_relative_l2: float
    radial_centroid_displacement_m: float
    grid_spacing_m: float
    zero_prediction_relative_l2: float
    missing_gradient_groups: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "failures": self.failures}


@dataclass(frozen=True)
class NineRecordGateMetrics:
    record_count_by_family: dict[str, int]
    aggregate_query_relative_l2: float
    aggregate_dense_relative_l2: float
    family_query_relative_l2: dict[str, float]
    family_dense_relative_l2: dict[str, float]
    family_late_relative_l2: dict[str, float]
    zero_prediction_relative_l2: float
    missing_gradient_groups: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_one_record_gate(metrics: OneRecordGateMetrics) -> GateDecision:
    if metrics.medium_type == "anomaly":
        raise ValueError("anomaly medium is forbidden by the V3 one-record gate")
    if metrics.medium_type != "uniform":
        raise ValueError("the V3 one-record gate requires a uniform medium")
    failures: list[str] = []
    relative_fields = (
        "early_relative_l2",
        "middle_relative_l2",
        "late_relative_l2",
        "query_relative_l2",
    )
    for field in relative_fields:
        value = float(getattr(metrics, field))
        if not math.isfinite(value) or value >= 0.05:
            failures.append(f"{field}={value:.8g} must be finite and <0.05")
        if math.isfinite(value) and value >= metrics.zero_prediction_relative_l2:
            failures.append(
                f"{field}={value:.8g} does not beat zero baseline "
                f"{metrics.zero_prediction_relative_l2:.8g}"
            )
    centroid = float(metrics.radial_centroid_displacement_m)
    if (
        not math.isfinite(centroid)
        or not math.isfinite(metrics.grid_spacing_m)
        or metrics.grid_spacing_m <= 0
        or centroid >= metrics.grid_spacing_m
    ):
        failures.append(
            "radial centroid displacement must be finite and below one grid cell: "
            f"{centroid:.8g} vs {metrics.grid_spacing_m:.8g} m"
        )
    if metrics.missing_gradient_groups:
        failures.append(
            "required gradient groups are missing: "
            + ",".join(metrics.missing_gradient_groups)
        )
    return GateDecision(passed=not failures, failures=tuple(failures))


def require_one_record_gate(metrics: OneRecordGateMetrics) -> None:
    decision = evaluate_one_record_gate(metrics)
    if not decision.passed:
        raise RuntimeError("one-record gate failed: " + "; ".join(decision.failures))


def evaluate_nine_record_gate(metrics: NineRecordGateMetrics) -> GateDecision:
    required = {"uniform", "layered", "marmousi"}
    count_keys = set(metrics.record_count_by_family)
    if "anomaly" in count_keys:
        raise ValueError("anomaly medium is forbidden by the V3 nine-record gate")
    if count_keys != required:
        raise ValueError("nine-record gate family keys must be uniform, layered, marmousi")
    mappings = (
        ("family_query_relative_l2", metrics.family_query_relative_l2),
        ("family_dense_relative_l2", metrics.family_dense_relative_l2),
        ("family_late_relative_l2", metrics.family_late_relative_l2),
    )
    for name, values in mappings:
        if set(values) != required:
            raise ValueError(f"{name} family keys do not match the V3 contract")
    failures: list[str] = []
    for family, count in metrics.record_count_by_family.items():
        if int(count) != 3:
            failures.append(f"{family} record count={count} must equal 3")
    aggregate = (
        ("aggregate_query_relative_l2", metrics.aggregate_query_relative_l2),
        ("aggregate_dense_relative_l2", metrics.aggregate_dense_relative_l2),
    )
    for name, value in aggregate:
        if not math.isfinite(value) or value >= 0.10:
            failures.append(f"{name}={value:.8g} must be finite and <0.10")
        if math.isfinite(value) and value >= metrics.zero_prediction_relative_l2:
            failures.append(f"{name}={value:.8g} does not beat the zero baseline")
    for name, values in mappings:
        for family, value in values.items():
            if not math.isfinite(value) or value >= 0.10:
                failures.append(f"{name}[{family}]={value:.8g} must be finite and <0.10")
            if math.isfinite(value) and value >= metrics.zero_prediction_relative_l2:
                failures.append(f"{name}[{family}]={value:.8g} does not beat zero")
    if metrics.missing_gradient_groups:
        failures.append(
            "required gradient groups are missing: "
            + ",".join(metrics.missing_gradient_groups)
        )
    return GateDecision(passed=not failures, failures=tuple(failures))


__all__ = [
    "GateDecision",
    "NineRecordGateMetrics",
    "OneRecordGateMetrics",
    "evaluate_one_record_gate",
    "evaluate_nine_record_gate",
    "require_one_record_gate",
]
