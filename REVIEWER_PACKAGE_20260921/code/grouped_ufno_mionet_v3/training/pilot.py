"""Sealed prerequisite validation for V3 full-data pilot runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch

from .checkpoint import CHECKPOINT_FORMAT


_FAMILIES = ("uniform", "layered", "marmousi")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PilotIdentity:
    manifest_digest: str
    gate_config_digest: str
    gate_step: int
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    prerequisite_report_path: str
    prerequisite_report_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_pilot_benchmark(
    report: Mapping[str, object], *, device_total_bytes: int
) -> dict[str, object]:
    if int(report.get("steps", 0)) < 3:
        raise ValueError("pilot benchmark requires at least three steps")
    for name in ("loss_min", "loss_max", "data_wait_fraction", "mean_step_seconds"):
        value = float(report.get(name, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"pilot benchmark {name} must be finite")
    if report.get("missing_gradient_groups") not in ([], ()):
        raise RuntimeError("pilot benchmark has missing gradient groups")
    interpolated = float(report.get("interpolated_target_fraction", 0.0))
    if not math.isfinite(interpolated) or interpolated <= 0:
        raise ValueError("pilot benchmark requires interpolated targets")
    peak = int(report.get("peak_cuda_memory_bytes", 0))
    if device_total_bytes <= 0 or peak <= 0 or peak >= int(device_total_bytes):
        raise RuntimeError("pilot benchmark CUDA memory is invalid or exceeds device memory")
    if float(report["mean_step_seconds"]) <= 0:
        raise ValueError("pilot benchmark timing must be positive")
    throughput = float(report.get("records_per_second", 0.0))
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("pilot benchmark throughput must be finite and positive")
    return dict(report)


def _require_metric_below(name: str, value: object, threshold: float = 0.10) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"pilot gate metric {name} is not numeric") from error
    if not math.isfinite(numeric) or numeric >= threshold:
        raise ValueError(f"pilot gate metric {name} violates the {threshold} threshold")


def validate_pilot_prerequisite(
    report_path: str | Path,
    checkpoint_path: str | Path,
    *,
    expected_manifest_digest: str,
) -> PilotIdentity:
    report_file = Path(report_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not report_file.is_file():
        raise ValueError(f"pilot prerequisite report does not exist: {report_file}")
    if not checkpoint_file.is_file():
        raise ValueError(f"pilot prerequisite checkpoint does not exist: {checkpoint_file}")
    with report_file.open(encoding="utf8") as handle:
        report = json.load(handle)
    if not isinstance(report, Mapping):
        raise ValueError("pilot prerequisite report is not a mapping")
    if report.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("pilot prerequisite report has the wrong checkpoint format")
    if report.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("pilot prerequisite manifest digest mismatch")
    decision = report.get("decision")
    if not isinstance(decision, Mapping) or decision.get("passed") is not True:
        raise RuntimeError("pilot requires a passed nine-record gate")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("pilot prerequisite metrics are missing")
    counts = metrics.get("record_count_by_family")
    if counts != {family: 3 for family in _FAMILIES}:
        raise ValueError("pilot requires exactly three gate records from each family")
    for name in ("aggregate_query_relative_l2", "aggregate_dense_relative_l2"):
        _require_metric_below(name, metrics.get(name))
    for name in (
        "family_query_relative_l2",
        "family_dense_relative_l2",
        "family_late_relative_l2",
    ):
        values = metrics.get(name)
        if not isinstance(values, Mapping) or set(values) != set(_FAMILIES):
            raise ValueError(f"pilot gate metric {name} has invalid family keys")
        for family in _FAMILIES:
            _require_metric_below(f"{name}[{family}]", values[family])
    missing = metrics.get("missing_gradient_groups")
    if missing not in ([], ()):
        raise ValueError("pilot prerequisite contains missing gradient groups")
    diagnostics = report.get("diagnostics")
    records = diagnostics.get("records") if isinstance(diagnostics, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("pilot prerequisite must contain diagnostic records")
    if any(
        isinstance(record, Mapping) and record.get("medium_type") == "anomaly"
        for record in records
    ):
        raise ValueError("anomaly records are forbidden by the V3 pilot")
    if len(records) != 9:
        raise ValueError("pilot prerequisite must contain nine diagnostic records")
    diagnostic_counts = {family: 0 for family in _FAMILIES}
    for record in records:
        family = record.get("medium_type") if isinstance(record, Mapping) else None
        if family not in diagnostic_counts:
            raise ValueError("pilot prerequisite contains an unknown medium family")
        diagnostic_counts[str(family)] += 1
    if diagnostic_counts != {family: 3 for family in _FAMILIES}:
        raise ValueError("pilot prerequisite diagnostic records are not family balanced")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("pilot parent checkpoint has the wrong format")
    if checkpoint.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("pilot parent checkpoint manifest mismatch")
    gate_config_digest = str(report.get("config_digest", ""))
    if not gate_config_digest or checkpoint.get("config_digest") != gate_config_digest:
        raise ValueError("pilot parent checkpoint config identity mismatch")
    gate_step = int(report.get("step", -1))
    if gate_step < 0 or int(checkpoint.get("global_step", -2)) != gate_step:
        raise ValueError("pilot parent checkpoint step does not match the passed report")
    return PilotIdentity(
        manifest_digest=expected_manifest_digest,
        gate_config_digest=gate_config_digest,
        gate_step=gate_step,
        parent_checkpoint_path=str(checkpoint_file),
        parent_checkpoint_sha256=_sha256(checkpoint_file),
        prerequisite_report_path=str(report_file),
        prerequisite_report_sha256=_sha256(report_file),
    )


__all__ = ["PilotIdentity", "validate_pilot_benchmark", "validate_pilot_prerequisite"]
