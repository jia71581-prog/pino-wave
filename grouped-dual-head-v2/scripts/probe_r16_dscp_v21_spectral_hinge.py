#!/usr/bin/env python3
"""v21 exploratory spectral non-harm hinge probe on consumed train records.

Three arms start from the frozen v18 B checkpoint and receive identical
fine-tuning except for the weight of a sampled high-spatial-band non-harm
hinge.  The 24 records were already consumed by v19; this is a direction and
weight-selection probe, never promotion/generalization evidence.
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
SPEC = ROOT / "results/r16_dscp_v21_spectral_hinge_spec_20260827.json"
OUT = ROOT / "results/r16_dscp_v21_spectral_hinge"
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
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import (  # noqa: E402
    metric_record,
)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def selected_retained_indices(keep: torch.Tensor, maximum: int = 16) -> torch.Tensor:
    retained = torch.nonzero(torch.as_tensor(keep).bool(), as_tuple=False).flatten()
    if retained.numel() == 0:
        raise v16.V16Refusal("spectral hinge has no retained frame")
    if retained.numel() <= maximum:
        return retained
    positions = torch.linspace(
        0, retained.numel() - 1, steps=maximum, device=retained.device
    ).round().long()
    return retained[positions]


def sampled_high_band_error_ratio(
    candidate: torch.Tensor,
    parent: torch.Tensor,
    truth: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    """Candidate/parent high-band error ratio on 16 retained future frames.

    The band cut exactly matches metric_record's rfft2 last-axis thirds.  The
    truth spectrum denominator cancels in the candidate-to-parent ratio.
    """
    indices = selected_retained_indices(keep.to(candidate.device), maximum=16)
    candidate_error = candidate[indices].float() - truth[indices].float()
    parent_error = parent[indices].float() - truth[indices].float()
    candidate_fft = torch.fft.rfft2(candidate_error, dim=(-2, -1), norm="ortho")
    parent_fft = torch.fft.rfft2(parent_error, dim=(-2, -1), norm="ortho")
    high_start = (2 * int(candidate_fft.shape[-1])) // 3
    candidate_energy = candidate_fft[..., high_start:].abs().square().sum()
    parent_energy = parent_fft[..., high_start:].abs().square().sum()
    return torch.sqrt(
        candidate_energy.clamp_min(1.0e-30) / parent_energy.detach().clamp_min(1.0e-30)
    )


def score_one(head, scales, bases_device, bundle, device) -> dict[str, Any]:
    with torch.no_grad():
        losses, adapted, parent_full, truth_future = v18.v18_bundle_loss(
            head, scales, bases_device, bundle, device
        )
        candidate = adapted[0, bundle.k1 + 1 :].float()
        parent = parent_full[0, bundle.k1 + 1 :].float()
        truth = truth_future.float()
        metrics = metric_record(
            candidate,
            parent,
            truth,
            family=bundle.family,
            condition=0.0,
            abstain=False,
        )
    parent_rel = float(metrics["parent_rel_l2"])
    candidate_rel = float(metrics["aggregate_rel_l2"])
    parent_high = float(metrics["parent_spectrum_bands"]["high"])
    candidate_high = float(metrics["spectrum_bands"]["high"])
    parent_late = float(metrics["parent_time_bands"]["late"])
    candidate_late = float(metrics["time_bands"]["late"])
    return {
        "sample_id": bundle.sample_id,
        "family": bundle.family,
        "loss": float(losses["total"]),
        "aggregate_rel_l2": candidate_rel,
        "parent_rel_l2": parent_rel,
        "aggregate_gain": (parent_rel - candidate_rel) / max(abs(parent_rel), 1.0e-30),
        "high_band_relative_l2": candidate_high,
        "parent_high_band_relative_l2": parent_high,
        "high_band_gain": (parent_high - candidate_high) / max(abs(parent_high), 1.0e-30),
        "late_mean_frame_rel_l2": candidate_late,
        "parent_late_mean_frame_rel_l2": parent_late,
        "late_gain": (parent_late - candidate_late) / max(abs(parent_late), 1.0e-30),
        "correction_energy_ratio": float(metrics["correction_energy_ratio"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def block(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "aggregate_gain_mean": mean([row["aggregate_gain"] for row in group]),
            "aggregate_nonworse": sum(row["aggregate_gain"] >= 0.0 for row in group),
            "high_band_gain_mean": mean([row["high_band_gain"] for row in group]),
            "high_band_nonworse": sum(row["high_band_gain"] >= 0.0 for row in group),
            "late_gain_mean": mean([row["late_gain"] for row in group]),
            "late_nonworse": sum(row["late_gain"] >= 0.0 for row in group),
            "worst_aggregate_harm": min(row["aggregate_gain"] for row in group),
        }

    return {
        "joint": block(rows),
        "by_family": {
            family: block([row for row in rows if row["family"] == family])
            for family in FAMILIES
        },
    }


def evaluate(head, scales, bases_device, bundles, device):
    head.eval()
    rows = [score_one(head, scales, bases_device, bundle, device) for bundle in bundles]
    return rows, summarize(rows)


def new_head(payload, device):
    head = v16.Wide128Head().to(device).float()
    head.load_state_dict(payload["model_state"])
    return head


def main() -> int:
    if not SPEC.exists():
        raise v16.V16Refusal(f"missing frozen probe spec: {SPEC}")
    spec = json.loads(SPEC.read_text())
    bindings = spec["bindings"]
    observed = {
        "script_sha256": v16.sha256_file(Path(__file__)),
        "checkpoint_sha256": v16.sha256_file(CHECKPOINT),
        "v19_terminal_sha256": v16.sha256_file(V19_TERMINAL),
        "v20_terminal_sha256": v16.sha256_file(V20_TERMINAL),
    }
    for name, value in observed.items():
        if value != bindings[name]:
            raise v16.V16Refusal(f"binding mismatch: {name}")
    if OUT.exists():
        raise v16.V16Refusal(f"probe output already exists: {OUT}")

    started = time.monotonic()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    v16.configure_determinism(v16.SEED)
    log = lambda message: print(f"{v16.utc_now()} {message}", flush=True)

    v19 = json.loads(V19_TERMINAL.read_text())
    ids = list(v19["eval_records"])
    bundles, skipped, bases, scales, build_s = v18.build_partition_fp16(
        device, ids, log=log
    )
    if skipped or len(bundles) != 24:
        raise v16.V16Refusal(f"probe bundle coverage failure: {skipped}")
    by_family = {
        family: sorted(
            [bundle for bundle in bundles if bundle.family == family],
            key=lambda bundle: bundle.sample_id,
        )
        for family in FAMILIES
    }
    if any(len(rows) != 8 for rows in by_family.values()):
        raise v16.V16Refusal("expected eight v19 records per family")
    fit = [bundle for family in FAMILIES for bundle in by_family[family][:6]]
    holdout = [bundle for family in FAMILIES for bundle in by_family[family][6:]]
    fit_ids = [bundle.sample_id for bundle in fit]
    holdout_ids = [bundle.sample_id for bundle in holdout]
    if len(fit) != 18 or len(holdout) != 6:
        raise v16.V16Refusal("unexpected probe split")

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    bases_device = bases.to(device).float()
    baseline_head = new_head(payload, device)
    baseline_rows, baseline_summary = evaluate(
        baseline_head, scales, bases_device, holdout, device
    )
    del baseline_head
    torch.cuda.empty_cache()

    arm_results = {}
    updates = int(spec["plan"]["updates"])
    eval_every = int(spec["plan"]["eval_every"])
    learning_rate = float(spec["plan"]["learning_rate"])
    for arm in spec["arms_order_fixed"]:
        arm_cfg = spec["arms"][arm]
        spectral_lambda = float(arm_cfg["spectral_lambda"])
        v16.configure_determinism(v16.SEED)
        generator = torch.Generator().manual_seed(v16.SEED)
        head = new_head(payload, device)
        head.train()
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.99),
            eps=1.0e-8,
            weight_decay=1.0e-4,
        )
        order = torch.randperm(len(fit), generator=generator).tolist()
        cursor = 0
        trajectory = []
        sampled_ratios = []
        for update in range(1, updates + 1):
            if cursor == len(order):
                order = torch.randperm(len(fit), generator=generator).tolist()
                cursor = 0
            bundle = fit[order[cursor]]
            cursor += 1
            losses, adapted, parent_full, truth_future = v18.v18_bundle_loss(
                head, scales, bases_device, bundle, device
            )
            candidate = adapted[0, bundle.k1 + 1 :].float()
            parent = parent_full[0, bundle.k1 + 1 :].float()
            truth = truth_future.float()
            keep = bundle.keep_full[bundle.k1 + 1 :].to(device)
            truth_norm = torch.linalg.vector_norm(truth).clamp_min(1.0e-30)
            candidate_rel = torch.linalg.vector_norm(candidate - truth) / truth_norm
            parent_rel = torch.linalg.vector_norm(parent - truth) / truth_norm
            aggregate_hinge = torch.relu(
                candidate_rel / parent_rel.detach().clamp_min(1.0e-30) - 0.98
            )
            high_ratio = sampled_high_band_error_ratio(candidate, parent, truth, keep)
            spectral_hinge = torch.relu(high_ratio - 0.98)
            total = losses["total"] + aggregate_hinge + spectral_lambda * spectral_hinge
            if not torch.isfinite(total):
                raise v16.V16Refusal(f"non-finite probe loss at {arm} update {update}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise v16.V16Refusal(f"non-finite probe gradient at {arm} update {update}")
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            sampled_ratios.append(float(high_ratio.detach()))
            if update % eval_every == 0 or update == updates:
                rows, summary = evaluate(head, scales, bases_device, holdout, device)
                trajectory.append(
                    {
                        "update": update,
                        "sampled_high_ratio_mean": mean(sampled_ratios[-eval_every:]),
                        "summary": summary,
                    }
                )
                log(
                    f"[{arm}] update {update} aggregate "
                    f"{summary['joint']['aggregate_gain_mean']:+.4f} high "
                    f"{summary['joint']['high_band_gain_mean']:+.4f}"
                )
                head.train()
        final_rows, final_summary = evaluate(head, scales, bases_device, holdout, device)
        base_joint = baseline_summary["joint"]
        final_joint = final_summary["joint"]
        viable = (
            final_joint["high_band_gain_mean"]
            >= base_joint["high_band_gain_mean"]
            + float(spec["readout"]["minimum_high_band_gain_delta"])
            and final_joint["high_band_nonworse"] >= base_joint["high_band_nonworse"]
            and final_joint["aggregate_gain_mean"]
            >= base_joint["aggregate_gain_mean"]
            - float(spec["readout"]["maximum_aggregate_gain_drop"])
            and final_joint["worst_aggregate_harm"]
            >= float(spec["readout"]["worst_aggregate_harm_floor"])
        )
        arm_results[arm] = {
            "spectral_lambda": spectral_lambda,
            "status": "viable" if viable else "not_viable",
            "final_summary": final_summary,
            "delta_vs_baseline": {
                "aggregate_gain_mean": final_joint["aggregate_gain_mean"]
                - base_joint["aggregate_gain_mean"],
                "high_band_gain_mean": final_joint["high_band_gain_mean"]
                - base_joint["high_band_gain_mean"],
                "late_gain_mean": final_joint["late_gain_mean"]
                - base_joint["late_gain_mean"],
                "high_band_nonworse": final_joint["high_band_nonworse"]
                - base_joint["high_band_nonworse"],
            },
            "trajectory": trajectory,
            "records": final_rows,
        }
        del head, optimizer
        torch.cuda.empty_cache()

    viable_arms = [
        name for name in spec["arms_order_fixed"] if arm_results[name]["status"] == "viable"
    ]
    selected = (
        max(
            viable_arms,
            key=lambda name: arm_results[name]["final_summary"]["joint"][
                "high_band_gain_mean"
            ],
        )
        if viable_arms
        else None
    )
    terminal = {
        "schema": "r16_dscp_v21_spectral_hinge_probe_v1",
        "status": "success",
        "claim_scope": "posthoc_train_only_direction_probe_not_promotion_evidence",
        "truth_scope": "v19-consumed train records only; validation/test_id untouched",
        "bindings": {
            **observed,
            "spec_sha256": v16.sha256_file(SPEC),
        },
        "split": {"fit_ids": fit_ids, "holdout_ids": holdout_ids},
        "baseline": {"summary": baseline_summary, "records": baseline_rows},
        "arms": arm_results,
        "selected_for_future_preregistered_full_data_probe": selected,
        "selection_is_exploratory": True,
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
    log("v21 spectral hinge probe terminal written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
