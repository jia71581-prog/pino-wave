#!/usr/bin/env python
"""Gate V63 full-dataset pretraining with held-out and GPU evidence."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence


FAMILIES = ("uniform", "layered", "marmousi")


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_values(row: Mapping[str, object]) -> tuple[float, dict[str, float]]:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("epoch row is missing metrics")
    aggregate = float(metrics["aggregate_relative_l2"])
    raw_family = metrics.get("family_relative_l2")
    if not isinstance(raw_family, Mapping) or set(raw_family) != set(FAMILIES):
        raise ValueError("epoch row has an invalid three-family metric")
    family = {name: float(raw_family[name]) for name in FAMILIES}
    return aggregate, family


def v63_candidate_gate(
    *,
    parent_row: Mapping[str, object],
    candidate_row: Mapping[str, object],
    smoke_row: Mapping[str, object],
    smoke_terminal: Mapping[str, object],
    family_tolerance: float = 0.03,
    maximum_peak_cuda_gib: float = 23.0,
    minimum_relative_improvement: float = 0.0,
) -> dict[str, object]:
    """Return a strict, serializable V63 promotion decision."""

    tolerance = float(family_tolerance)
    maximum_gib = float(maximum_peak_cuda_gib)
    minimum_improvement = float(minimum_relative_improvement)
    if not _finite(tolerance) or tolerance < 0.0:
        raise ValueError("family tolerance must be finite and nonnegative")
    if not _finite(maximum_gib) or maximum_gib <= 0.0:
        raise ValueError("maximum CUDA memory must be positive and finite")
    if (
        not _finite(minimum_improvement)
        or minimum_improvement < 0.0
        or minimum_improvement >= 1.0
    ):
        raise ValueError("minimum relative improvement must lie in [0,1)")

    parent_aggregate, parent_family = _metric_values(parent_row)
    candidate_aggregate, candidate_family = _metric_values(candidate_row)
    metric_values = (
        parent_aggregate,
        candidate_aggregate,
        *parent_family.values(),
        *candidate_family.values(),
    )
    metrics_finite = all(math.isfinite(value) and value >= 0.0 for value in metric_values)
    relative_improvement = (
        (parent_aggregate - candidate_aggregate) / parent_aggregate
        if metrics_finite and parent_aggregate > 0.0
        else float("nan")
    )
    family_checks = {
        name: candidate_family[name] <= parent_family[name] + tolerance
        for name in FAMILIES
    }

    candidate_metrics = candidate_row.get("metrics", {})
    gradients = smoke_row.get("gradient_norms", {})
    adapter_gradient = (
        gradients.get("dense_decoder.band_limited_adapter")
        if isinstance(gradients, Mapping)
        else None
    )
    ddp = smoke_row.get("ddp", {})
    if not isinstance(ddp, Mapping):
        ddp = {}
    peak_bytes = smoke_row.get("peak_cuda_bytes")
    maximum_bytes = maximum_gib * 1024**3
    checks = {
        "metrics_finite": metrics_finite,
        "aggregate_improved": metrics_finite
        and candidate_aggregate < parent_aggregate,
        "material_relative_improvement": metrics_finite
        and relative_improvement >= minimum_improvement,
        "families_safe": metrics_finite and all(family_checks.values()),
        "sealed_validation_panel": (
            str(candidate_row.get("validation_scope")) == "pilot_fixed_panel"
            and int(candidate_metrics.get("record_count", -1)) == 48
            and int(candidate_metrics.get("frame_count", -1)) == 48 * 32
        ),
        "smoke_complete": str(smoke_terminal.get("status")) == "complete",
        "two_smoke_updates": int(smoke_row.get("global_step", -1)) >= 2
        and int(smoke_terminal.get("global_step", -1)) >= 2,
        "physical_microbatch_24": int(
            smoke_row.get("physical_microbatch_records", -1)
        )
        == 24,
        "four_gpu_world": int(ddp.get("world_size", -1)) == 4,
        "global_macros_4": int(ddp.get("global_macros_per_update", -1)) == 4,
        "finite_train_loss": _finite(smoke_row.get("train_loss")),
        "adapter_gradient_active": _finite(adapter_gradient)
        and float(adapter_gradient) > 0.0,
        "cuda_peak_safe": _finite(peak_bytes)
        and 0.0 < float(peak_bytes) < maximum_bytes,
    }
    return {
        "schema": "saved_time_v63_full_dataset_gate_v1",
        "passes": bool(all(checks.values())),
        "checks": checks,
        "family_checks": family_checks,
        "family_tolerance": tolerance,
        "maximum_peak_cuda_gib": maximum_gib,
        "minimum_relative_improvement": minimum_improvement,
        "parent": {
            "epoch": int(parent_row["epoch"]),
            "aggregate_relative_l2": parent_aggregate,
            "family_relative_l2": parent_family,
        },
        "candidate": {
            "epoch": int(candidate_row["epoch"]),
            "aggregate_relative_l2": candidate_aggregate,
            "relative_improvement": relative_improvement,
            "family_relative_l2": candidate_family,
        },
        "smoke": {
            "global_step": int(smoke_row.get("global_step", -1)),
            "physical_microbatch_records": int(
                smoke_row.get("physical_microbatch_records", -1)
            ),
            "peak_cuda_bytes": (
                None if not _finite(peak_bytes) else int(float(peak_bytes))
            ),
        },
    }


def _epoch_rows(path: str | Path, *, scope: str) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "epoch" and row.get("validation_scope") == scope:
            rows.append(row)
    if not rows:
        raise ValueError(f"no {scope} epoch metrics in {path}")
    return tuple(rows)


def _best_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return min(
        rows,
        key=lambda row: (
            float(row["metrics"]["aggregate_relative_l2"]),
            int(row["epoch"]),
        ),
    )


def _atomic_json(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-metrics", required=True)
    parser.add_argument("--smoke-metrics", required=True)
    parser.add_argument("--smoke-terminal", required=True)
    parser.add_argument("--pilot-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family-tolerance", type=float, default=0.03)
    parser.add_argument("--maximum-peak-cuda-gib", type=float, default=23.0)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.0)
    args = parser.parse_args(argv)

    parent = _best_row(_epoch_rows(args.parent_metrics, scope="pilot_fixed_panel"))
    candidate = _best_row(
        _epoch_rows(args.pilot_metrics, scope="pilot_fixed_panel")
    )
    smoke = _epoch_rows(args.smoke_metrics, scope="smoke")[-1]
    terminal = json.loads(Path(args.smoke_terminal).read_text())
    report = v63_candidate_gate(
        parent_row=parent,
        candidate_row=candidate,
        smoke_row=smoke,
        smoke_terminal=terminal,
        family_tolerance=args.family_tolerance,
        maximum_peak_cuda_gib=args.maximum_peak_cuda_gib,
        minimum_relative_improvement=args.minimum_relative_improvement,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "v63_candidate_gate"]
