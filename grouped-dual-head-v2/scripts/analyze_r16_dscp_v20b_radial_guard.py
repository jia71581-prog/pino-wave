#!/usr/bin/env python3
"""v20b: paper-metric radial spectrum audit and structural guard test.

The frozen v18 B coefficient maps are evaluated unchanged and after the
existing deployment-computable low/mid radial projection.  No training or
checkpoint write occurs.  All truth is restricted to the 24 train records
already consumed by v19.
"""
from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results/r16_dscp_v20b_radial_guard_spec_20260827.json"
OUT = ROOT / "results/r16_dscp_v20b_radial_guard"
CHECKPOINT = ROOT / "results/r16_dscp_v18/B_data_hinge/best.pt"
V19_TERMINAL = ROOT / "results/r16_dscp_v19_offpanel/terminal.json"
V20_TERMINAL = ROOT / "results/r16_dscp_v20_diagnostic/terminal.json"
FAMILIES = ("layered", "marmousi", "uniform")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v16 = load_module("train_r16_dscp_v16", ROOT / "scripts/train_r16_dscp_v16.py")
v18 = load_module("run_v18_alldata_ddp4", ROOT / "scripts/run_v18_alldata_ddp4.py")
from saved_time_phase_operator_v4.band_adapter import (  # noqa: E402
    project_low_mid_increment,
    registered_high_band_mask,
)
from saved_time_phase_operator_v4.confirmatory_metrics import (  # noqa: E402
    spatial_spectrum_relative_l2,
)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def field_metrics(candidate: torch.Tensor, parent: torch.Tensor, truth: torch.Tensor):
    candidate = torch.as_tensor(candidate).float()
    parent = torch.as_tensor(parent, device=candidate.device).float()
    truth = torch.as_tensor(truth, device=candidate.device).float()
    if candidate.shape != parent.shape or candidate.shape != truth.shape or candidate.ndim != 3:
        raise ValueError("candidate, parent and truth must match [time,z,x]")
    aggregate = float(v16.relative_l2(candidate, truth))
    parent_aggregate = float(v16.relative_l2(parent, truth))
    candidate_spectrum = spatial_spectrum_relative_l2(candidate, truth)
    parent_spectrum = spatial_spectrum_relative_l2(parent, truth)
    candidate_frame = torch.sqrt(
        (candidate - truth).square().flatten(1).sum(1)
        / truth.square().flatten(1).sum(1).clamp_min(1.0e-30)
    )
    parent_frame = torch.sqrt(
        (parent - truth).square().flatten(1).sum(1)
        / truth.square().flatten(1).sum(1).clamp_min(1.0e-30)
    )
    count = int(truth.shape[0])
    thirds = ((0, count // 3), (count // 3, 2 * count // 3), (2 * count // 3, count))
    candidate_time = {
        name: float(candidate_frame[start:end].mean())
        for name, (start, end) in zip(("early", "middle", "late"), thirds)
    }
    parent_time = {
        name: float(parent_frame[start:end].mean())
        for name, (start, end) in zip(("early", "middle", "late"), thirds)
    }
    return {
        "aggregate_rel_l2": aggregate,
        "parent_aggregate_rel_l2": parent_aggregate,
        "aggregate_gain": (parent_aggregate - aggregate) / max(abs(parent_aggregate), 1.0e-30),
        "radial_spectrum_relative_l2": candidate_spectrum,
        "parent_radial_spectrum_relative_l2": parent_spectrum,
        "radial_high_gain": (
            float(parent_spectrum["high"]) - float(candidate_spectrum["high"])
        ) / max(abs(float(parent_spectrum["high"])), 1.0e-30),
        "time_bands": candidate_time,
        "parent_time_bands": parent_time,
        "late_gain": (parent_time["late"] - candidate_time["late"])
        / max(abs(parent_time["late"]), 1.0e-30),
        "correction_energy_ratio": float(
            (candidate - parent).double().square().sum()
            / parent.double().square().sum().clamp_min(1.0e-30)
        ),
    }


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    def block(group: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = [row[key] for row in group]
        tolerance = 1.0e-6
        return {
            "count": len(metrics),
            "aggregate_gain_mean": mean([row["aggregate_gain"] for row in metrics]),
            "aggregate_nonworse": sum(row["aggregate_gain"] >= 0.0 for row in metrics),
            "aggregate_nonworse_within_1pct": sum(
                row["aggregate_gain"] >= -0.01 for row in metrics
            ),
            "worst_aggregate_harm": min(row["aggregate_gain"] for row in metrics),
            "radial_high_gain_mean": mean([row["radial_high_gain"] for row in metrics]),
            "radial_high_nonworse": sum(
                row["radial_high_gain"] >= -tolerance for row in metrics
            ),
            "radial_high_worst_harm": min(row["radial_high_gain"] for row in metrics),
            "late_gain_mean": mean([row["late_gain"] for row in metrics]),
            "late_nonworse": sum(row["late_gain"] >= 0.0 for row in metrics),
        }

    return {
        "joint": block(rows),
        "by_family": {
            family: block([row for row in rows if row["family"] == family])
            for family in FAMILIES
        },
    }


def main() -> int:
    if not SPEC.exists():
        raise v16.V16Refusal(f"missing frozen v20b spec: {SPEC}")
    spec = json.loads(SPEC.read_text())
    bindings = spec["bindings"]
    observed = {
        "script_sha256": v16.sha256_file(Path(__file__)),
        "checkpoint_sha256": v16.sha256_file(CHECKPOINT),
        "v19_terminal_sha256": v16.sha256_file(V19_TERMINAL),
        "v20_terminal_sha256": v16.sha256_file(V20_TERMINAL),
        "confirmatory_metrics_sha256": v16.sha256_file(
            ROOT / "saved_time_phase_operator_v4/confirmatory_metrics.py"
        ),
        "band_adapter_sha256": v16.sha256_file(
            ROOT / "saved_time_phase_operator_v4/band_adapter.py"
        ),
    }
    for name, value in observed.items():
        if value != bindings[name]:
            raise v16.V16Refusal(f"binding mismatch: {name}")
    if OUT.exists():
        raise v16.V16Refusal(f"v20b output already exists: {OUT}")

    started = time.monotonic()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    v16.configure_determinism(v16.SEED)
    log = lambda message: print(f"{v16.utc_now()} {message}", flush=True)

    ids = list(json.loads(V19_TERMINAL.read_text())["eval_records"])
    bundles, skipped, bases, scales, build_s = v18.build_partition_fp16(
        device, ids, log=log
    )
    if skipped or len(bundles) != 24:
        raise v16.V16Refusal(f"v20b bundle coverage failure: {skipped}")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    head = v16.Wide128Head().to(device).float().eval()
    head.load_state_dict(payload["model_state"])
    bases_device = bases.to(device).float()

    records = []
    projection_latencies = []
    equivalence_errors = []
    high_energy_fractions = []
    for position, bundle in enumerate(bundles, start=1):
        with torch.no_grad():
            features = bundle.features.to(device, non_blocking=True).float()
            parent_full = (
                bundle.parent_full.to(device, non_blocking=True).float()
                / v18.FP16_FIELD_SCALE
            )
            truth = (
                bundle.truth_future.to(device, non_blocking=True).float()
                / v18.FP16_FIELD_SCALE
            )
            coefficient = v16.head_coefficient(head, scales, bundle, features)
            raw_full = v16.materialize_confined(
                parent_full, bases_device, bundle, coefficient
            )
            torch.cuda.synchronize(device)
            projection_started = time.perf_counter()
            guarded_coefficient = project_low_mid_increment(coefficient)
            torch.cuda.synchronize(device)
            projection_latencies.append(time.perf_counter() - projection_started)
            guarded_full = v16.materialize_confined(
                parent_full, bases_device, bundle, guarded_coefficient
            )
            raw = raw_full[0, bundle.k1 + 1 :].float()
            guarded = guarded_full[0, bundle.k1 + 1 :].float()
            parent = parent_full[0, bundle.k1 + 1 :].float()
            raw_correction = raw - parent
            full_projected_correction = project_low_mid_increment(raw_correction[None])[0]
            equivalence_errors.append(
                float(
                    torch.linalg.vector_norm((guarded - parent) - full_projected_correction)
                    / torch.linalg.vector_norm(full_projected_correction).clamp_min(1.0e-30)
                )
            )
            spectrum = torch.fft.rfft2((guarded - parent).float(), norm="ortho")
            high_mask = registered_high_band_mask(
                int(guarded.shape[-2]), int(guarded.shape[-1]), spectrum.device
            )
            high_energy_fractions.append(
                float(
                    spectrum[:, high_mask].abs().square().sum()
                    / spectrum.abs().square().sum().clamp_min(1.0e-30)
                )
            )
            raw_metrics = field_metrics(raw, parent, truth)
            guarded_metrics = field_metrics(guarded, parent, truth)
        records.append(
            {
                "sample_id": bundle.sample_id,
                "family": bundle.family,
                "raw": raw_metrics,
                "coefficient_radial_guard": guarded_metrics,
            }
        )
        log(f"v20b {position}/24 {bundle.sample_id}")

    raw_summary = summarize(records, "raw")
    guarded_summary = summarize(records, "coefficient_radial_guard")
    raw_joint = raw_summary["joint"]
    guarded_joint = guarded_summary["joint"]
    readout = {
        "radial_guard_viable": (
            guarded_joint["radial_high_nonworse"] == 24
            and guarded_joint["radial_high_gain_mean"] >= -1.0e-6
            and guarded_joint["aggregate_gain_mean"]
            >= raw_joint["aggregate_gain_mean"]
            - float(spec["readout"]["maximum_aggregate_gain_drop"])
            and guarded_joint["aggregate_nonworse_within_1pct"] >= 23
            and guarded_joint["worst_aggregate_harm"]
            >= float(spec["readout"]["worst_aggregate_harm_floor"])
        ),
        "aggregate_gain_delta_guard_minus_raw": (
            guarded_joint["aggregate_gain_mean"] - raw_joint["aggregate_gain_mean"]
        ),
        "radial_high_gain_delta_guard_minus_raw": (
            guarded_joint["radial_high_gain_mean"] - raw_joint["radial_high_gain_mean"]
        ),
    }
    sorted_latency = sorted(projection_latencies)
    latency_p95 = sorted_latency[max(0, math.ceil(0.95 * len(sorted_latency)) - 1)]
    terminal = {
        "schema": "r16_dscp_v20b_radial_guard_diagnostic_v1",
        "status": "success",
        "claim_scope": "posthoc_train_only_paper_metric_diagnostic_not_promotion_evidence",
        "truth_scope": "v19-consumed train records only; validation/test_id untouched",
        "bindings": {**observed, "spec_sha256": v16.sha256_file(SPEC)},
        "metric_correction": (
            "v20 used the legacy DSCP last-axis thirds; v20b uses the paper/confirmatory "
            "normalized 2-D radial bands from spatial_spectrum_relative_l2"
        ),
        "raw_summary": raw_summary,
        "coefficient_radial_guard_summary": guarded_summary,
        "readout": readout,
        "guard_engineering": {
            "projection_space": "16 coefficient maps before temporal materialization",
            "projection_latency_mean_s": statistics.fmean(projection_latencies),
            "projection_latency_p95_s": latency_p95,
            "coefficient_vs_full_field_projection_max_relative_difference": max(
                equivalence_errors
            ),
            "guarded_correction_max_radial_high_energy_fraction": max(
                high_energy_fractions
            ),
        },
        "records": records,
        "checkpoint_writes": 0,
        "parent_untouched": v16.sha256_file(v16.PARENT_PATH) == v16.PARENT_SHA256,
        "resources": {
            "build_s": build_s,
            "wall_s": time.monotonic() - started,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "completed_utc": v16.utc_now(),
    }
    OUT.mkdir(parents=True, exist_ok=False)
    v16.atomic_json(terminal, OUT / "terminal.json")
    log("v20b radial guard terminal written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
