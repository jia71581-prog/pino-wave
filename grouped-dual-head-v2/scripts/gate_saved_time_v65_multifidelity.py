#!/usr/bin/env python3
"""Gate V65 solver-guided pretraining before solver-free long continuation."""
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
    return aggregate, {name: float(raw_family[name]) for name in FAMILIES}


def _panel_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("epoch row is missing metrics")
    raw = metrics.get("source_relative_l2")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("epoch row is missing source-level panel identity")
    return tuple(sorted(str(value) for value in raw))


def _parent_row_from_same_panel_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Convert a checkpoint-bound same-panel evaluation into a gate row."""

    if (
        report.get("schema") != "saved_time_family_expert_same_panel_parent_v1"
        or report.get("status") != "complete"
    ):
        raise ValueError("same-panel parent report schema or status is invalid")
    binding = report.get("binding")
    metrics = report.get("metrics")
    if not isinstance(binding, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("same-panel parent report binding or metrics are missing")
    if (
        int(binding.get("validation_records", -1)) != 48
        or int(binding.get("validation_frames_per_record", -1)) != 32
        or len(tuple(binding.get("validation_indices", ()))) != 48
    ):
        raise ValueError("same-panel parent report must bind the registered 48 by 32 panel")
    if (
        not str(binding.get("parent_checkpoint_sha256", ""))
        or not str(binding.get("active_manifest_digest", ""))
    ):
        raise ValueError("same-panel parent report identity binding is incomplete")
    row = {
        "event": "epoch",
        "epoch": 0,
        "validation_scope": "pilot_fixed_panel",
        "metrics": dict(metrics),
        "same_panel_binding": dict(binding),
    }
    _metric_values(row)
    if len(_panel_ids(row)) != 48:
        raise ValueError("same-panel parent report must contain 48 source metrics")
    return row


def v65_candidate_gate(
    *,
    parent_row: Mapping[str, object],
    candidate_row: Mapping[str, object],
    smoke_row: Mapping[str, object],
    smoke_terminal: Mapping[str, object],
    cache_summary: Mapping[str, object],
    family_tolerance: float = 0.03,
    minimum_relative_improvement: float = 0.02,
    maximum_peak_cuda_gib: float = 23.0,
    target_aggregate_relative_l2: float = 0.10,
    target_family_relative_l2: float = 0.12,
) -> dict[str, object]:
    """Separate the pilot safety gate from the final scientific target."""

    tolerance = float(family_tolerance)
    minimum_improvement = float(minimum_relative_improvement)
    maximum_gib = float(maximum_peak_cuda_gib)
    target_aggregate = float(target_aggregate_relative_l2)
    target_family = float(target_family_relative_l2)
    if (
        not _finite(tolerance)
        or tolerance < 0.0
        or not _finite(minimum_improvement)
        or not 0.0 < minimum_improvement < 1.0
        or not _finite(maximum_gib)
        or maximum_gib <= 0.0
        or not 0.0 < target_aggregate < 1.0
        or not 0.0 < target_family < 1.0
    ):
        raise ValueError("V65 gate thresholds are invalid")
    parent_aggregate, parent_family = _metric_values(parent_row)
    candidate_aggregate, candidate_family = _metric_values(candidate_row)
    relative_improvement = (parent_aggregate - candidate_aggregate) / max(
        parent_aggregate, 1.0e-20
    )
    same_validation_panel = _panel_ids(parent_row) == _panel_ids(candidate_row)
    metric_values = (
        parent_aggregate,
        candidate_aggregate,
        *parent_family.values(),
        *candidate_family.values(),
    )
    metrics_finite = all(
        math.isfinite(value) and value >= 0.0 for value in metric_values
    )
    candidate_metrics = candidate_row.get("metrics", {})
    if not isinstance(candidate_metrics, Mapping):
        candidate_metrics = {}
    gradients = smoke_row.get("gradient_norms", {})
    if not isinstance(gradients, Mapping):
        gradients = {}
    ddp = smoke_row.get("ddp", {})
    if not isinstance(ddp, Mapping):
        ddp = {}
    peak_bytes = smoke_row.get("peak_cuda_bytes")
    family_checks = {
        name: candidate_family[name] <= parent_family[name] + tolerance
        for name in FAMILIES
    }
    checks = {
        "metrics_finite": metrics_finite,
        "same_validation_panel": same_validation_panel,
        "aggregate_improved": metrics_finite
        and same_validation_panel
        and candidate_aggregate < parent_aggregate,
        "minimum_relative_improvement": metrics_finite
        and same_validation_panel
        and relative_improvement >= minimum_improvement,
        "families_safe": metrics_finite
        and same_validation_panel
        and all(family_checks.values()),
        "sealed_validation_panel": (
            str(candidate_row.get("validation_scope")) == "pilot_fixed_panel"
            and int(candidate_metrics.get("record_count", -1)) == 48
            and int(candidate_metrics.get("frame_count", -1)) == 48 * 32
        ),
        "cache_complete": (
            str(cache_summary.get("status")) == "complete"
            and int(cache_summary.get("record_count", -1)) == 2240
            and int(cache_summary.get("time_count", -1)) == 64
        ),
        "smoke_complete": str(smoke_terminal.get("status")) == "complete",
        "two_smoke_updates": int(smoke_row.get("global_step", -1)) >= 2
        and int(smoke_terminal.get("global_step", -1)) >= 2,
        "measured_physical_microbatch": int(
            smoke_row.get("physical_microbatch_records", -1)
        )
        in (2, 3, 4),
        "four_gpu_world": int(ddp.get("world_size", -1)) == 4,
        "global_macros_8": int(ddp.get("global_macros_per_update", -1)) == 8,
        "finite_train_loss": _finite(smoke_row.get("train_loss")),
        "dual_head_gradients_active": all(
            _finite(gradients.get(name)) and float(gradients[name]) > 0.0
            for name in ("fusion", "dense_decoder")
        ),
        "cuda_peak_safe": _finite(peak_bytes)
        and 0.0 < float(peak_bytes) < maximum_gib * 1024**3,
    }
    accuracy_checks = {
        "aggregate_below_target": candidate_aggregate < target_aggregate,
        "families_below_target": max(candidate_family.values()) < target_family,
    }
    return {
        "schema": "saved_time_v65_multifidelity_gate_v1",
        "passes": bool(all(checks.values())),
        "checks": checks,
        "family_checks": family_checks,
        "parent": {
            "epoch": int(parent_row["epoch"]),
            "aggregate_relative_l2": parent_aggregate,
            "family_relative_l2": parent_family,
        },
        "candidate": {
            "epoch": int(candidate_row["epoch"]),
            "aggregate_relative_l2": candidate_aggregate,
            "family_relative_l2": candidate_family,
            "relative_improvement": relative_improvement,
        },
        "minimum_relative_improvement": minimum_improvement,
        "accuracy_target": {
            "passes": bool(all(accuracy_checks.values())),
            "checks": accuracy_checks,
            "aggregate_threshold": target_aggregate,
            "family_threshold": target_family,
        },
        "smoke": {
            "physical_microbatch_records": int(
                smoke_row.get("physical_microbatch_records", -1)
            ),
            "peak_cuda_bytes": (
                None if not _finite(peak_bytes) else int(float(peak_bytes))
            ),
        },
    }


def _epoch_rows(path: str | Path, *, scope: str) -> tuple[dict[str, object], ...]:
    rows = []
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
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", required=True)
    parser.add_argument("--smoke-metrics", required=True)
    parser.add_argument("--smoke-terminal", required=True)
    parser.add_argument("--pilot-metrics", required=True)
    parser.add_argument("--cache-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family-tolerance", type=float, default=0.03)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.02)
    parser.add_argument("--maximum-peak-cuda-gib", type=float, default=23.0)
    args = parser.parse_args(argv)
    parent_report = json.loads(Path(args.parent_report).read_text())
    report = v65_candidate_gate(
        parent_row=_parent_row_from_same_panel_report(parent_report),
        candidate_row=_best_row(
            _epoch_rows(args.pilot_metrics, scope="pilot_fixed_panel")
        ),
        smoke_row=_epoch_rows(args.smoke_metrics, scope="smoke")[-1],
        smoke_terminal=json.loads(Path(args.smoke_terminal).read_text()),
        cache_summary=json.loads(Path(args.cache_summary).read_text()),
        family_tolerance=args.family_tolerance,
        minimum_relative_improvement=args.minimum_relative_improvement,
        maximum_peak_cuda_gib=args.maximum_peak_cuda_gib,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_parent_row_from_same_panel_report", "main", "v65_candidate_gate"]
