#!/usr/bin/env python3
"""Gate a family-expert pilot against direct and global parent evidence."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.expert_gate import FAMILIES, family_expert_gate


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _best_complete_epoch(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    epochs = [
        row
        for row in rows
        if row.get("event") == "epoch"
        and row.get("validation_scope") == "pilot_fixed_panel"
        and isinstance(row.get("metrics"), Mapping)
        and row.get("checkpoint")
    ]
    if not epochs:
        raise ValueError("family expert metrics contain no complete fixed-panel epoch")
    return min(
        epochs,
        key=lambda row: (
            float(row["metrics"]["aggregate_relative_l2"]),
            int(row["epoch"]),
        ),
    )


def _metrics_from_report(raw: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("metrics", "parent_metrics", "candidate", "global_parent"):
        value = raw.get(key)
        if isinstance(value, Mapping) and "aggregate_relative_l2" in value:
            return value
    if "aggregate_relative_l2" in raw:
        return raw
    raise ValueError("global parent report contains no metrics")


def _bound_same_panel_metrics(
    raw: Mapping[str, object],
    *,
    expected_checkpoint: str | Path,
    expected_manifest_digest: str,
) -> Mapping[str, object]:
    """Return metrics only from an evaluation bound to this parent and dataset."""

    if raw.get("schema") != "saved_time_family_expert_same_panel_parent_v1":
        raise ValueError("same-panel parent report schema is invalid")
    if raw.get("status") != "complete":
        raise ValueError("same-panel parent evaluation is incomplete")
    binding = raw.get("binding")
    metrics = raw.get("metrics")
    if not isinstance(binding, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("same-panel parent report is malformed")
    try:
        observed_checkpoint = Path(str(binding["parent_checkpoint"])).resolve()
        observed_manifest = str(binding["active_manifest_digest"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("same-panel parent binding is malformed") from error
    if observed_checkpoint != Path(expected_checkpoint).resolve():
        raise ValueError("same-panel parent checkpoint binding mismatch")
    if observed_manifest != str(expected_manifest_digest):
        raise ValueError("same-panel parent manifest binding mismatch")
    return metrics


def _overfit_reduction(raw: Mapping[str, object]) -> float:
    for key in (
        "anchor_relative_reduction",
        "relative_reduction",
        "overfit_reduction",
    ):
        if key in raw:
            return float(raw[key])
    try:
        initial = float(raw["initial_relative_l2"])
        final = float(raw["final_relative_l2"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("overfit report contains no reduction evidence") from error
    return (initial - final) / max(initial, 1.0e-16)


def _trainable_prefixes(candidate_row: Mapping[str, object]) -> tuple[str, ...]:
    stage = candidate_row.get("trainable_stage")
    if not isinstance(stage, Mapping):
        raise ValueError("candidate metrics lack trainable-stage evidence")
    raw = stage.get("trainable_prefixes")
    if not isinstance(raw, list) or not raw:
        raise ValueError("candidate trainable prefixes are malformed")
    prefixes = tuple(str(value) for value in raw)
    if any(not value for value in prefixes):
        raise ValueError("candidate trainable prefixes are malformed")
    return prefixes


def _coarse_anchor_required(candidate_row: Mapping[str, object]) -> bool:
    """Require the coarse comparison only while the coarse path is frozen."""

    prefixes = _trainable_prefixes(candidate_row)
    expert = "dense_decoder.family_experts"
    return all(value == expert or value.startswith(f"{expert}.") for value in prefixes)


def _minimum_physical_microbatch(candidate_row: Mapping[str, object]) -> int:
    """Return the registered memory-safe floor for the observed trainable stage."""

    prefixes = _trainable_prefixes(candidate_row)
    memory_heavy = ("coordinate_encoder", "travel_branch", "medium_encoder")
    return 2 if any(value.startswith(memory_heavy) for value in prefixes) else 3


def _selection_parent_checkpoint(selection: Mapping[str, object]) -> str:
    """Read either legacy flat or V60 nested parent provenance."""

    value = selection.get("checkpoint")
    if value:
        return str(value)
    parent = selection.get("parent")
    if isinstance(parent, Mapping) and parent.get("checkpoint"):
        return str(parent["checkpoint"])
    raise ValueError("family expert parent checkpoint provenance is missing")


def _selection_effective_batch(selection: Mapping[str, object]) -> int:
    """Read either legacy flat or V60 nested candidate batch evidence."""

    value = selection.get("effective_batch")
    if value is None:
        candidate = selection.get("candidate")
        value = candidate.get("effective_batch") if isinstance(candidate, Mapping) else None
    try:
        batch = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("family expert selection effective batch is missing") from error
    if batch <= 0:
        raise ValueError("family expert selection effective batch is invalid")
    return batch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--global-parent-report", required=True)
    parser.add_argument("--same-panel-parent-report", required=True)
    parser.add_argument("--same-panel-global-report")
    parser.add_argument("--overfit-report", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    selection_path = Path(args.selection_report).resolve()
    global_path = Path(args.global_parent_report).resolve()
    same_panel_parent_path = Path(args.same_panel_parent_report).resolve()
    same_panel_global_path = Path(
        args.same_panel_global_report or args.same_panel_parent_report
    ).resolve()
    overfit_path = Path(args.overfit_report).resolve()
    candidate_path = Path(args.candidate_metrics).resolve()
    selection = json.loads(selection_path.read_text())
    global_report = json.loads(global_path.read_text())
    same_panel_parent_report = json.loads(same_panel_parent_path.read_text())
    same_panel_global_report = json.loads(same_panel_global_path.read_text())
    overfit_report = json.loads(overfit_path.read_text())
    candidate_row = _best_complete_epoch(candidate_path)
    candidate_metrics = candidate_row["metrics"]
    if not all(
        isinstance(value, Mapping)
        for value in (selection, global_report, overfit_report, candidate_metrics)
    ):
        raise ValueError("family expert evidence reports are malformed")
    candidate_identity_path = candidate_path.parent / "run_identity.json"
    candidate_identity = json.loads(candidate_identity_path.read_text())
    active_manifest_digest = str(candidate_identity["manifest_digest"])
    direct_checkpoint = _selection_parent_checkpoint(selection)
    try:
        global_checkpoint = str(global_report["checkpoint"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("family expert global checkpoint provenance is missing") from error
    direct_parent = _bound_same_panel_metrics(
        same_panel_parent_report,
        expected_checkpoint=direct_checkpoint,
        expected_manifest_digest=active_manifest_digest,
    )
    global_parent = _bound_same_panel_metrics(
        same_panel_global_report,
        expected_checkpoint=global_checkpoint,
        expected_manifest_digest=active_manifest_digest,
    )
    candidate_spectrum = candidate_metrics.get("spectrum_relative_l2")
    parent_spectrum = global_parent.get("spectrum_relative_l2")
    route_probabilities = candidate_metrics.get("router_route_probabilities")
    if not all(
        isinstance(value, Mapping)
        for value in (candidate_spectrum, parent_spectrum, route_probabilities)
    ):
        raise ValueError("family expert nested evidence is malformed")
    ddp = candidate_row.get("ddp")
    if isinstance(ddp, Mapping):
        effective_batch = int(ddp["global_macros_per_update"]) * int(
            candidate_row.get("macro_records", 12)
        )
    else:
        effective_batch = _selection_effective_batch(selection)
    report = family_expert_gate(
        direct_parent=direct_parent,
        global_parent=global_parent,
        candidate=candidate_metrics,
        overfit_reduction=_overfit_reduction(overfit_report),
        router_accuracy=float(candidate_metrics["router_accuracy"]),
        route_probabilities=tuple(float(route_probabilities[key]) for key in FAMILIES),
        high_band_parent=float(parent_spectrum["high"]),
        high_band_candidate=float(candidate_spectrum["high"]),
        peak_cuda_bytes=int(candidate_row["peak_cuda_bytes"]),
        physical_microbatch=int(candidate_row["physical_microbatch_records"]),
        minimum_physical_microbatch=_minimum_physical_microbatch(candidate_row),
        effective_batch=effective_batch,
        coarse_anchor_required=_coarse_anchor_required(candidate_row),
    )
    report.update(
        {
            "schema": "saved_time_family_expert_evidence_gate_v1",
            "selection_report": str(selection_path),
            "global_parent_report": str(global_path),
            "same_panel_parent_report": str(same_panel_parent_path),
            "same_panel_global_report": str(same_panel_global_path),
            "candidate_run_identity": str(candidate_identity_path),
            "overfit_report": str(overfit_path),
            "candidate_metrics": str(candidate_path),
            "candidate_epoch": int(candidate_row["epoch"]),
            "candidate_checkpoint": str(candidate_row["checkpoint"]),
        }
    )
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passes"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
