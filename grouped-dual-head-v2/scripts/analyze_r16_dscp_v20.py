#!/usr/bin/env python3
"""v20 post-hoc train-only diagnostic for the frozen v18 B checkpoint.

This script does not train or write a checkpoint.  It reuses the 24 records
already consumed by v19, measures family-gradient compatibility at the frozen
v18 B checkpoint, and audits low/mid/high spatial bands plus late-time error.
The output is exploratory evidence only and cannot promote a candidate.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results/r16_dscp_v20_diagnostic_spec_20260827.json"
OUT = ROOT / "results/r16_dscp_v20_diagnostic"
CHECKPOINT = ROOT / "results/r16_dscp_v18/B_data_hinge/best.pt"
V19_TERMINAL = ROOT / "results/r16_dscp_v19_offpanel/terminal.json"
FAMILIES = ("layered", "marmousi", "uniform")
FIT2_COUNTS = {"layered": 1095, "marmousi": 675, "uniform": 395}


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
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import (  # noqa: E402
    metric_record,
)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if not bool(torch.isfinite(denom)) or float(denom) <= 0.0:
        return math.nan
    return float(torch.dot(left, right) / denom)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "aggregate_gain_mean": mean([r["aggregate_gain"] for r in group]),
            "aggregate_nonworse": sum(r["aggregate_gain"] >= 0.0 for r in group),
            "high_band_gain_mean": mean([r["high_band_gain"] for r in group]),
            "high_band_nonworse": sum(r["high_band_gain"] >= 0.0 for r in group),
            "late_gain_mean": mean([r["late_gain"] for r in group]),
            "late_nonworse": sum(r["late_gain"] >= 0.0 for r in group),
            "correction_energy_ratio_mean": mean(
                [r["correction_energy_ratio"] for r in group]
            ),
        }

    return {
        "joint": group_summary(rows),
        "by_family": {
            family: group_summary([r for r in rows if r["family"] == family])
            for family in FAMILIES
        },
    }


def main() -> int:
    if not SPEC.exists():
        raise v16.V16Refusal(f"missing frozen diagnostic spec: {SPEC}")
    spec = json.loads(SPEC.read_text())
    if v16.sha256_file(CHECKPOINT) != spec["bindings"]["checkpoint_sha256"]:
        raise v16.V16Refusal("v18 B checkpoint hash mismatch")
    if v16.sha256_file(V19_TERMINAL) != spec["bindings"]["v19_terminal_sha256"]:
        raise v16.V16Refusal("v19 terminal hash mismatch")
    if OUT.exists():
        raise v16.V16Refusal(f"diagnostic output already exists: {OUT}")

    started = time.monotonic()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    v16.configure_determinism(v16.SEED)
    log = lambda message: print(f"{v16.utc_now()} {message}", flush=True)

    v19 = json.loads(V19_TERMINAL.read_text())
    ids = list(v19["eval_records"])
    if len(ids) != 24 or set(v19["arms"]) != {"A_data", "B_data_hinge"}:
        raise v16.V16Refusal("unexpected v19 terminal shape")
    bundles, skipped, bases, scales, build_s = v18.build_partition_fp16(
        device, ids, log=log
    )
    if skipped or len(bundles) != 24:
        raise v16.V16Refusal(f"diagnostic bundle coverage failure: {skipped}")

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    head = v16.Wide128Head().to(device).float()
    head.load_state_dict(payload["model_state"])
    head.train()
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != v16.EXPECTED_PARAMETERS:
        raise v16.V16Refusal("unexpected head parameter count")
    bases_device = bases.to(device).float()

    gradient_sums = {
        family: torch.zeros(parameter_count, dtype=torch.float64) for family in FAMILIES
    }
    gradient_counts = {family: 0 for family in FAMILIES}
    rows: list[dict[str, Any]] = []

    for position, bundle in enumerate(bundles, start=1):
        head.zero_grad(set_to_none=True)
        losses, adapted, parent_full, truth_future = v18.v18_bundle_loss(
            head, scales, bases_device, bundle, device
        )
        candidate_future = adapted[0, bundle.k1 + 1 :].float()
        parent_future = parent_full[0, bundle.k1 + 1 :].float()
        truth = truth_future.float()
        truth_norm = torch.linalg.vector_norm(truth).clamp_min(1.0e-30)
        candidate_rel = torch.linalg.vector_norm(candidate_future - truth) / truth_norm
        parent_rel = torch.linalg.vector_norm(parent_future - truth) / truth_norm
        hinge = torch.relu(candidate_rel / parent_rel.detach().clamp_min(1.0e-30) - 0.98)
        total = losses["total"] + hinge
        gradients = torch.autograd.grad(total, parameters, retain_graph=False)
        vector = torch.cat([gradient.detach().flatten() for gradient in gradients])
        if not bool(torch.isfinite(vector).all()):
            raise v16.V16Refusal(f"non-finite gradient for {bundle.sample_id}")
        gradient_sums[bundle.family].add_(vector.cpu().double())
        gradient_counts[bundle.family] += 1

        with torch.no_grad():
            metrics = metric_record(
                candidate_future.detach(),
                parent_future.detach(),
                truth.detach(),
                family=bundle.family,
                condition=0.0,
                abstain=False,
            )
        high_parent = float(metrics["parent_spectrum_bands"]["high"])
        high_candidate = float(metrics["spectrum_bands"]["high"])
        late_parent = float(metrics["parent_time_bands"]["late"])
        late_candidate = float(metrics["time_bands"]["late"])
        rows.append(
            {
                "sample_id": bundle.sample_id,
                "family": bundle.family,
                "aggregate_rel_l2": float(metrics["aggregate_rel_l2"]),
                "parent_rel_l2": float(metrics["parent_rel_l2"]),
                "aggregate_gain": (float(metrics["parent_rel_l2"]) - float(metrics["aggregate_rel_l2"]))
                / max(abs(float(metrics["parent_rel_l2"])), 1.0e-30),
                "spectrum_bands": metrics["spectrum_bands"],
                "parent_spectrum_bands": metrics["parent_spectrum_bands"],
                "high_band_gain": (high_parent - high_candidate)
                / max(abs(high_parent), 1.0e-30),
                "time_bands": metrics["time_bands"],
                "parent_time_bands": metrics["parent_time_bands"],
                "late_gain": (late_parent - late_candidate)
                / max(abs(late_parent), 1.0e-30),
                "correction_energy_ratio": float(metrics["correction_energy_ratio"]),
                "base_loss": float(losses["total"].detach()),
                "hinge": float(hinge.detach()),
            }
        )
        log(f"diagnostic {position}/24 {bundle.sample_id}")
        del losses, adapted, parent_full, truth_future, total, gradients, vector

    family_gradients = {
        family: gradient_sums[family] / max(gradient_counts[family], 1)
        for family in FAMILIES
    }
    fit_total = float(sum(FIT2_COUNTS.values()))
    proportional = sum(
        family_gradients[family] * (FIT2_COUNTS[family] / fit_total)
        for family in FAMILIES
    )
    balanced = sum(family_gradients.values()) / float(len(FAMILIES))

    pairwise = {}
    for left_index, left in enumerate(FAMILIES):
        for right in FAMILIES[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = cosine(
                family_gradients[left], family_gradients[right]
            )
    gradients_summary = {
        "per_family_record_counts": gradient_counts,
        "per_family_gradient_norm": {
            family: float(torch.linalg.vector_norm(family_gradients[family]))
            for family in FAMILIES
        },
        "pairwise_cosine": pairwise,
        "alignment_with_fit2_proportional_gradient": {
            family: cosine(family_gradients[family], proportional) for family in FAMILIES
        },
        "alignment_with_family_balanced_gradient": {
            family: cosine(family_gradients[family], balanced) for family in FAMILIES
        },
        "proportional_vs_balanced_cosine": cosine(proportional, balanced),
        "readout": {
            "conflict_pairs_below_minus_0p1": sorted(
                name for name, value in pairwise.items() if value < -0.1
            ),
            "weak_pairs_minus_0p1_to_0p1": sorted(
                name for name, value in pairwise.items() if -0.1 <= value <= 0.1
            ),
        },
    }
    frequency_summary = summarize_records(rows)
    frequency_summary["readout"] = {
        "high_band_safe_22_of_24": (
            frequency_summary["joint"]["high_band_nonworse"] >= 22
            and frequency_summary["joint"]["high_band_gain_mean"] > 0.0
        ),
        "late_safe_22_of_24": (
            frequency_summary["joint"]["late_nonworse"] >= 22
            and frequency_summary["joint"]["late_gain_mean"] > 0.0
        ),
    }

    terminal = {
        "schema": "r16_dscp_v20_gradient_spectrum_diagnostic_v1",
        "status": "success",
        "claim_scope": "posthoc_train_only_exploratory_not_promotion_evidence",
        "truth_scope": "train/final_train_confirm already consumed by v19; validation/test_id untouched",
        "bindings": {
            "spec_sha256": v16.sha256_file(SPEC),
            "checkpoint_sha256": v16.sha256_file(CHECKPOINT),
            "v19_terminal_sha256": v16.sha256_file(V19_TERMINAL),
        },
        "gradients": gradients_summary,
        "frequency_and_time": frequency_summary,
        "records": rows,
        "resources": {
            "build_s": build_s,
            "wall_s": time.monotonic() - started,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "parent_untouched": v16.sha256_file(v16.PARENT_PATH) == v16.PARENT_SHA256,
        "checkpoint_writes": 0,
        "completed_utc": v16.utc_now(),
    }
    OUT.mkdir(parents=True, exist_ok=False)
    v16.atomic_json(terminal, OUT / "terminal.json")
    log("v20 diagnostic terminal written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
