#!/usr/bin/env python3
"""V16 stage runner: wide-128 feature head, smoke -> pilot -> long chain.

Frozen preregistration: results/r16_dscp_v16_preregistration_20260826.json.
Authorization: the lead conditional directive of 2026-08-26 ("继续，符合条件启动
长训") recorded in that document; each stage runs only if the previous stage's
frozen gates all passed, and any gate failure stops the chain with an honest
fail_gate terminal.

Engineering basis: the r4e10 probe pipeline components, whose fidelity against
the v15 smoke terminal was verified bit-exact on the uniform record (rel diff
0.0) and to <=2.2e-5 elsewhere.  This runner generalizes that pipeline to
multi-record stages with CPU-resident bundles (the long fit set does not fit
in GPU memory).

Inherited vetoes enforced here:

(a) every threshold is transcribed from the frozen v16 preregistration and
    never moved toward a measured value of its own gated run;
(b) the correction mask derives from parent frame energy only;
(c) oracle bounds are recomputed on the scored record; no transplanted
    constant is a baseline;
(d) the smoke loss gate is a wiring check, never learning evidence;
(e) the correction is an output-field data fit plus a feature-driven ansatz,
    not a physics or PDE residual;
(f) promotion flows from the recorded lead conditional authorization, not
    from smoke evidence.

Truth scope: train split only, allowlisted per panels role for the running
stage.  validation and test_id stay sealed; no code path here can open them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (
    CONDITION_MAXIMUM,
    R16DSCP,
    c1_causal_mask,
    deployment_features,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import (
    canonical_sha,
    relative_l2,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import (
    V4GuardedOnsetLoader,
    canonical_public,
    canonical_travel,
    deployment_args,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import (
    configure_determinism,
    sha256_file,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v3 import (
    TAU,
    apply_confined_correction,
    confined_oracle_upper_bound,
    full_time_keep_mask,
    masked_confined_loss,
)
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4_family_temporal_pod_capacity as legacy_pod
import scripts.probe_r4e8_masked_oracle as r4e8
from scripts.probe_r4e12_convergence_horizon import Wide128Head

CANDIDATE = "r16_dscp_v16_wide128"
PREREG = PROJECT_ROOT / "results/r16_dscp_v16_preregistration_20260826.json"
PREREG_SHA256 = "d91689b4e6e42c66343597f9dc3d97c51d659387c43f054b3bfbb70741e955f4"
OUT = PROJECT_ROOT / "results/r16_dscp_v16"
BASIS = PROJECT_ROOT / "results/r16_dscp_v1/basis_rank16.pt"
PANELS = PROJECT_ROOT / "results/r16_dscp_v1/panels.json"

#: hard bindings transcribed from the frozen preregistration; drift is a stop
FROZEN = {
    "results/r16_dscp_v1/basis_rank16.pt": "8cab01344fc0a88f43fa232458127e87d2f8bbb63523b9389b927b39389c4786",
    "results/r16_dscp_v1/panels.json": "6d2f2facd86b449c058f9f824895f77977a23fbecf954c0de69144ec50162464",
    "saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py": "f906a422c26c86a736e21f77842ee83892ce7bd48364e2ca665a261451070201",
    "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py": "959619b032a77af8f6095ba927e46fa30502df338f8067179ea4a94cfd4945a9",
    "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v3.py": "8e0ae9294ff615633c07a1c686577325ed6785bf46ffb07d98e3e393963ba517",
    "results/r4e12_convergence_horizon_20260826/terminal.json": "4db6844c96b7da5f6c5312ca0eaa3f450ebdd2f31bbf393e784a6ed85aec4fbd",
}
RUNTIME_PINNED = (
    "scripts/train_r16_dscp_v16.py",
    "scripts/probe_r4e12_convergence_horizon.py",
    "results/r16_dscp_v16_preregistration_20260826.json",
)
PARENT_PATH = parent_runtime.CHECKPOINT_PATH
PARENT_SHA256 = "448035bd0061205c67799eeef3b023a71b49e2db7028b6155078be1a886de789"

SEED = 372
LR = 3e-3
EXPECTED_PARAMETERS = 25266
TRAIN_VRAM_LIMIT_BYTES = int(8.0 * 1024**3)
CHECKPOINT_LIMIT_BYTES = int(2 * 1024**2)
MIN_FREE_BYTES = 2 * 1024**3

STAGE_PLAN: dict[str, dict[str, Any]] = {
    "smoke": {
        "fit_role": "smoke", "eval_role": "smoke",
        "updates": 3072, "eval_every": 384, "wall_s": 1200.0,
        "sampling": "round_robin",
        "loss_reduction_min": 0.10, "oracle_fraction": 0.35,
        "oracle_fraction_records_min": 2, "nonworse_min_count": 3,
    },
    "pilot": {
        "fit_role": "pilot_fit", "eval_role": "pilot_confirm",
        "updates": 3072, "eval_every": 384, "wall_s": 1800.0,
        "sampling": "shuffled_epochs",
        "joint_improvement": 0.01, "per_family_improvement": 0.005,
        "nonworse_min_count": 23,
    },
    "long": {
        "fit_role": "long_fit", "eval_role": "long_calibration",
        "updates_per_epoch": 1536, "max_epochs": 20, "patience": 4,
        "wall_s": 7200.0, "sampling": "shuffled_epochs",
        "joint_improvement": 0.01, "per_family_improvement": 0.005,
        "nonworse_min_count": 23,
    },
}

DISCLAIMERS = {
    "smoke": "wiring and 3-record capacity re-confirmation; not learning or generalization evidence",
    "pilot": "first held-out-train generalization evidence; not validation evidence",
    "long": "train-split generalization at scale; the 0.05 acceptance claim requires a future sealed validation-once stage",
    "correction_nature": "output-field data fit plus feature-driven ansatz; no PDE residual anywhere",
}


class V16Refusal(RuntimeError):
    pass


class TruthScopeRefusal(V16Refusal):
    pass


# ---------------------------------------------------------------------------
# bindings and environment
# ---------------------------------------------------------------------------


def verify_bindings() -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise V16Refusal(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 (determinism; r4e12 launch incident)"
        )
    if sha256_file(PREREG) != PREREG_SHA256:
        raise V16Refusal("v16 preregistration drifted after freeze")
    out: dict[str, Any] = {
        "preregistration": {"expected": PREREG_SHA256, "observed": PREREG_SHA256}
    }
    for rel, expected in FROZEN.items():
        observed = sha256_file(PROJECT_ROOT / rel)
        out[rel] = {"expected": expected, "observed": observed}
        if observed != expected:
            raise V16Refusal(f"binding drift: {rel}")
    parent_observed = sha256_file(PARENT_PATH)
    out["parent_checkpoint"] = {"expected": PARENT_SHA256, "observed": parent_observed}
    if parent_observed != PARENT_SHA256:
        raise V16Refusal("parent checkpoint drifted")
    for rel in RUNTIME_PINNED:
        out.setdefault(rel, {"runtime_pinned": sha256_file(PROJECT_ROOT / rel)})
    parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    return out


def role_sample_ids(role: str) -> list[str]:
    rows = json.loads(PANELS.read_text())["records"]
    ids = [r["sample_id"] for r in rows if r.get("role") == role]
    if not ids:
        raise V16Refusal(f"panels carry no records for role {role}")
    return ids


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    os.replace(temp, path)


# ---------------------------------------------------------------------------
# CPU-resident record bundles
# ---------------------------------------------------------------------------


@dataclass
class CpuBundle:
    family: str
    sample_id: str
    route_index: int
    k1: int
    features: torch.Tensor      # [1,29,H,W] cpu
    parent_full: torch.Tensor   # [1,401,H,W] cpu
    truth_future: torch.Tensor  # [T-k1-1,H,W] cpu
    keep_full: torch.Tensor     # [401] bool cpu


def load_truth_allowlisted(record: Any, allowed: frozenset[str]) -> torch.Tensor:
    """Every truth read in this runner passes this role allowlist."""
    if str(record.split) != "train":
        raise TruthScopeRefusal("only train truth may be opened")
    if str(record.sample_id) not in allowed:
        raise TruthScopeRefusal(f"sample id not on the stage allowlist: {record.sample_id}")
    truth, _digest = legacy_pod.load_train_truth(record, device=torch.device("cpu"))
    return truth


def build_bundles(
    device: torch.device, sample_ids: Sequence[str], *, log: Callable[[str], None]
) -> tuple[list[CpuBundle], dict[str, Any]]:
    """Generalized r4e10 pipeline: parent + travel + features per record, CPU-resident."""
    configure_determinism(SEED)
    artifact = torch.load(BASIS, map_location="cpu", weights_only=False)
    parent_model, normalizer, manifest_payload, _ = parent_runtime.load_model_context(device)
    manifest = parent_runtime.manifest_object(manifest_payload)
    allowed = frozenset(sample_ids)
    loader = V4GuardedOnsetLoader(
        GuardedOnsetDataset(
            parent_runtime.SOURCE_H5_PATH, manifest, split="train",
            sample_ids=list(sample_ids),
        )
    )

    @torch.inference_mode()
    def parent_predictor(public):
        v = public.velocity_mps[None, None].to(device)
        s = public.source_parameters[None].to(device)
        sm = public.source_map[None, None].to(device)
        medium = parent_model.encode_medium(v, normalizer)
        prepared = parent_model.prepare_sources(
            medium, s, sm, normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
        )
        normalized = parent_model.dense_normalized(
            prepared, public.time_s.to(device),
            x_m=public.x_m.to(device), z_m=public.z_m.to(device), time_block=16,
        )
        return normalizer.decode_pressure(normalized.float(), s[:, 4])[0].cpu()

    travel_builder = lambda v, s: grid_eikonal_travel_time(
        v, source_indices=[(round(float(s[1]) / 10), round(float(s[0]) / 10))],
        dx_m=10, dz_m=10,
    )[0]

    reference = R16DSCP(artifact["basis"], artifact["coefficient_scales"])
    bases = reference.bases.clone()
    scales = reference.coefficient_scales.clone()
    bases_device = bases.to(device).float()

    loaded: dict[str, Any] = {}
    for index in range(len(loader)):
        public = canonical_public(loader[index])
        loaded[public.sample_id] = public
    missing = allowed - set(loaded)
    if missing:
        raise V16Refusal(f"records missing from the guarded loader: {sorted(missing)}")

    bundles: list[CpuBundle] = []
    started = time.monotonic()
    for position, sample_id in enumerate(sample_ids):
        public = loaded[sample_id]
        parent = torch.as_tensor(parent_predictor(public)).detach().cpu().float()
        travel = canonical_travel(public, travel_builder).float()
        args = deployment_args(public, parent, travel, device)
        features, decisions, conditions = deployment_features(
            *args[:-2], bases_device, args[-2], args[-1]
        )
        decision = decisions[0]
        if decision.abstain or float(conditions[0]) > CONDITION_MAXIMUM:
            raise V16Refusal(f"record abstained, stage undefined: {sample_id}")
        record = r4e8.record_from_manifest(manifest_payload, sample_id)
        truth = load_truth_allowlisted(record, allowed)
        k1 = int(public.observed_indices[1])
        parent_field = args[7][0].detach().float()
        keep_full, _floor = full_time_keep_mask(parent_field, k1=k1, tau=TAU)
        bundles.append(CpuBundle(
            family=str(record.family), sample_id=sample_id,
            route_index=int(decision.index), k1=k1,
            features=features.detach().float().cpu(),
            parent_full=args[7].detach().float().cpu(),
            truth_future=truth[k1 + 1:].float().cpu(),
            keep_full=keep_full.detach().cpu(),
        ))
        del args, features, parent, truth
        if (position + 1) % 24 == 0:
            log(f"bundles {position + 1}/{len(sample_ids)} ({time.monotonic() - started:.0f}s)")
    del parent_model
    torch.cuda.empty_cache()
    context = {
        "bases": bases, "scales": scales, "device": device,
        "cache_build_s": time.monotonic() - started,
    }
    return bundles, context


# ---------------------------------------------------------------------------
# forward, loss, scoring (transcribed from the fidelity-verified r4e10 path)
# ---------------------------------------------------------------------------


def head_coefficient(head, scales, bundle: CpuBundle, features_device) -> torch.Tensor:
    unit = head(features_device)
    coefficient = torch.zeros_like(unit, dtype=torch.float32)
    scale = scales.to(unit.device)[bundle.route_index, :, None, None].float()
    coefficient[0] = unit[0].float() * scale
    return coefficient


def materialize_confined(parent_full, bases_device, bundle: CpuBundle, coefficient):
    basis = bases_device[bundle.route_index]
    correction = torch.einsum("tr,rhw->thw", basis, coefficient[0])
    ramp = c1_causal_mask(
        parent_full.shape[1], bundle.k1, device=parent_full.device, dtype=torch.float32
    )
    correction = correction * ramp[:, None, None]
    correction = torch.cat(
        (torch.zeros_like(correction[:, :1]), correction[:, 1:]), dim=1
    )
    keep = bundle.keep_full.to(parent_full.device)
    confined = apply_confined_correction(parent_full[0].float(), correction, keep)
    return confined[None]


def bundle_loss(head, scales, bases_device, bundle: CpuBundle, device):
    features = bundle.features.to(device, non_blocking=True)
    parent_full = bundle.parent_full.to(device, non_blocking=True)
    truth_future = bundle.truth_future.to(device, non_blocking=True)
    coefficient = head_coefficient(head, scales, bundle, features)
    adapted = materialize_confined(parent_full, bases_device, bundle, coefficient)
    keep_future = bundle.keep_full[bundle.k1 + 1:].to(device)
    losses = masked_confined_loss(
        adapted[:, bundle.k1 + 1:].float(),
        truth_future[None],
        coefficient,
        parent_full[:, bundle.k1 + 1:].float(),
        keep_future,
        tau=TAU,
    )
    return losses, adapted, parent_full, truth_future


@torch.no_grad()
def score_bundle(head, scales, bases_device, bundle: CpuBundle, device) -> dict[str, Any]:
    losses, adapted, parent_full, truth_future = bundle_loss(
        head, scales, bases_device, bundle, device
    )
    cand = float(relative_l2(adapted[0, bundle.k1 + 1:].float(), truth_future))
    par = float(relative_l2(parent_full[0, bundle.k1 + 1:].float(), truth_future))
    return {
        "sample_id": bundle.sample_id, "family": bundle.family,
        "loss": float(losses["total"]),
        "aggregate_rel_l2": cand, "parent_rel_l2": par,
        "gain_iii": float((par - cand) / max(abs(par), 1e-30)),
        "nonworse": bool(cand <= par),
    }


def oracle_bound(bundle: CpuBundle, bases: torch.Tensor) -> dict[str, Any]:
    parent_future = bundle.parent_full[0, bundle.k1 + 1:]
    basis_future = bases[bundle.route_index][bundle.k1 + 1:]
    return confined_oracle_upper_bound(
        parent_future, bundle.truth_future, basis_future, tau=TAU
    )


# ---------------------------------------------------------------------------
# gates (thresholds transcribed from the frozen preregistration)
# ---------------------------------------------------------------------------


def smoke_gates(plan, initial_loss, scores, oracle_by_id, peak_vram, checkpoint_bytes):
    reductions = {
        row["sample_id"]: 1.0 - row["loss"] / max(abs(initial_loss[row["sample_id"]]), 1e-30)
        for row in scores
    }
    loss_pass = all(v >= plan["loss_reduction_min"] for v in reductions.values())
    fractions = {
        row["sample_id"]: row["gain_iii"] / max(
            abs(float(oracle_by_id[row["sample_id"]]["acceptance_convention_gain"])), 1e-30
        )
        for row in scores
    }
    oracle_hits = sum(v >= plan["oracle_fraction"] for v in fractions.values())
    nonworse = sum(bool(r["nonworse"]) for r in scores)
    finite = all(
        math.isfinite(r[k]) for r in scores for k in ("loss", "aggregate_rel_l2", "parent_rel_l2")
    )
    return {
        "loss_per_record": {
            "value": reductions, "threshold": plan["loss_reduction_min"], "passed": loss_pass,
        },
        "oracle_gain": {
            "value": fractions, "fraction": plan["oracle_fraction"],
            "records_min": plan["oracle_fraction_records_min"],
            "passed": oracle_hits >= plan["oracle_fraction_records_min"],
        },
        "nonworse": {
            "value": nonworse, "threshold": plan["nonworse_min_count"],
            "passed": nonworse >= plan["nonworse_min_count"],
        },
        "finite": {"value": finite, "passed": finite},
        "vram": {
            "value_bytes": peak_vram, "limit_bytes": TRAIN_VRAM_LIMIT_BYTES,
            "passed": 0 < peak_vram <= TRAIN_VRAM_LIMIT_BYTES,
        },
        "checkpoint": {
            "value_bytes": checkpoint_bytes, "limit_bytes": CHECKPOINT_LIMIT_BYTES,
            "passed": 0 < checkpoint_bytes <= CHECKPOINT_LIMIT_BYTES,
        },
    }


def eval_gates(plan, scores, peak_vram, checkpoint_bytes):
    joint = sum(r["gain_iii"] for r in scores) / max(len(scores), 1)
    by_family: dict[str, list[float]] = {}
    for row in scores:
        by_family.setdefault(row["family"], []).append(row["gain_iii"])
    family_means = {k: sum(v) / len(v) for k, v in by_family.items()}
    nonworse = sum(bool(r["nonworse"]) for r in scores)
    finite = all(
        math.isfinite(r[k]) for r in scores for k in ("loss", "aggregate_rel_l2", "parent_rel_l2")
    )
    return {
        "joint_improvement": {
            "value": joint, "threshold": plan["joint_improvement"],
            "passed": joint >= plan["joint_improvement"],
        },
        "per_family_improvement": {
            "value": family_means, "threshold": plan["per_family_improvement"],
            "passed": len(family_means) == 3
            and all(v >= plan["per_family_improvement"] for v in family_means.values()),
        },
        "nonworse_records": {
            "value": nonworse, "threshold": plan["nonworse_min_count"],
            "passed": nonworse >= plan["nonworse_min_count"],
        },
        "finite": {"value": finite, "passed": finite},
        "vram": {
            "value_bytes": peak_vram, "limit_bytes": TRAIN_VRAM_LIMIT_BYTES,
            "passed": 0 < peak_vram <= TRAIN_VRAM_LIMIT_BYTES,
        },
        "checkpoint": {
            "value_bytes": checkpoint_bytes, "limit_bytes": CHECKPOINT_LIMIT_BYTES,
            "passed": 0 < checkpoint_bytes <= CHECKPOINT_LIMIT_BYTES,
        },
    }


def save_checkpoint(head, optimizer, path: Path, extra: Mapping[str, Any]) -> int:
    payload = {
        "schema": "r16_dscp_v16_checkpoint_v1", "candidate": CANDIDATE,
        "model_state": head.state_dict(), "optimizer_state": optimizer.state_dict(),
        **dict(extra),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
    return path.stat().st_size


# ---------------------------------------------------------------------------
# stage execution
# ---------------------------------------------------------------------------


def run_stage(stage: str, device: torch.device, *, input_checkpoint: Path | None,
              log: Callable[[str], None]) -> dict[str, Any]:
    plan = STAGE_PLAN[stage]
    run = OUT / stage
    if (run / "terminal.json").exists():
        raise V16Refusal(f"stage terminal already exists: {run / 'terminal.json'}")
    started = time.monotonic()
    deadline = started + plan["wall_s"]
    bindings = verify_bindings()

    fit_ids = role_sample_ids(plan["fit_role"])
    eval_ids = role_sample_ids(plan["eval_role"])
    all_ids = list(dict.fromkeys([*fit_ids, *eval_ids]))
    log(f"[{stage}] building {len(all_ids)} bundles")
    bundles, context = build_bundles(device, all_ids, log=log)
    by_id = {b.sample_id: b for b in bundles}
    fit = [by_id[s] for s in fit_ids]
    confirm = [by_id[s] for s in eval_ids]
    scales = context["scales"]
    bases_device = context["bases"].to(device).float()

    # the vram gate measures the training phase; the build phase (parent model
    # resident) is excluded by this reset, as disclosed in the preregistration
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    configure_determinism(SEED)
    head = Wide128Head().to(device).float()
    if sum(p.numel() for p in head.parameters()) != EXPECTED_PARAMETERS:
        raise V16Refusal("head parameter count is not the preregistered 25266")
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, betas=(0.9, 0.99),
                                  eps=1e-8, weight_decay=1e-4)
    if input_checkpoint is not None:
        payload = torch.load(input_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("candidate") != CANDIDATE:
            raise V16Refusal("input checkpoint belongs to another candidate")
        head.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        log(f"[{stage}] warm start from {input_checkpoint.name} "
            f"(stage {payload.get('stage')})")

    initial_loss: dict[str, float] = {}
    with torch.no_grad():
        for b in fit:
            losses, *_ = bundle_loss(head, scales, bases_device, b, device)
            initial_loss[b.sample_id] = float(losses["total"])

    generator = torch.Generator().manual_seed(SEED)
    trajectory: list[dict[str, Any]] = []
    status = "completed"

    def order_for_epoch() -> list[int]:
        if plan["sampling"] == "round_robin":
            return list(range(len(fit)))
        return torch.randperm(len(fit), generator=generator).tolist()

    def one_update(update: int, order: list[int]) -> float:
        bundle = fit[order[update % len(order)]]
        losses, *_ = bundle_loss(head, scales, bases_device, bundle, device)
        total = losses["total"]
        if not torch.isfinite(total):
            raise V16Refusal(f"non-finite loss at update {update} ({bundle.sample_id})")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise V16Refusal("nonfinite or absent gradient")
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        return float(total)

    checkpoint_bytes = 0
    if stage in {"smoke", "pilot"}:
        updates_total = plan["updates"]
        order = order_for_epoch()
        for update in range(updates_total):
            if time.monotonic() > deadline:
                status = "budget_limit"
                break
            if plan["sampling"] == "shuffled_epochs" and update % len(fit) == 0 and update:
                order = order_for_epoch()
            one_update(update, order)
            if (update + 1) % plan["eval_every"] == 0:
                snap = [score_bundle(head, scales, bases_device, b, device) for b in confirm]
                trajectory.append({"update": update + 1, "records": snap})
        final_scores = [score_bundle(head, scales, bases_device, b, device) for b in confirm]
        checkpoint_bytes = save_checkpoint(head, optimizer, run / "last.pt", {
            "stage": stage, "update": updates_total, "seed": SEED,
        })
        best_path = run / "last.pt"
    else:
        best = float("inf")
        bad = 0
        best_epoch = -1
        updates_per_epoch = plan["updates_per_epoch"]
        for epoch in range(plan["max_epochs"]):
            if time.monotonic() > deadline:
                status = "budget_limit"
                break
            order = order_for_epoch()
            for update in range(updates_per_epoch):
                if time.monotonic() > deadline:
                    status = "budget_limit"
                    break
                if update % len(fit) == 0 and update:
                    order = order_for_epoch()
                one_update(update, order)
            if status == "budget_limit":
                break
            snap = [score_bundle(head, scales, bases_device, b, device) for b in confirm]
            monitor = sum(r["loss"] for r in snap) / len(snap)
            trajectory.append({
                "epoch": epoch + 1, "monitor_mean_calibration_loss": monitor,
                "records": snap,
            })
            checkpoint_bytes = save_checkpoint(head, optimizer, run / "last.pt", {
                "stage": stage, "epoch": epoch + 1, "seed": SEED,
            })
            if monitor < best:
                best = monitor
                best_epoch = epoch + 1
                bad = 0
                save_checkpoint(head, optimizer, run / "best.pt", {
                    "stage": stage, "epoch": epoch + 1, "seed": SEED, "monitor": monitor,
                })
            else:
                bad += 1
                if bad >= plan["patience"]:
                    status = "early_stop_patience"
                    break
            log(f"[long] epoch {epoch + 1} monitor {monitor:.6f} "
                f"best {best:.6f}@{best_epoch} bad {bad}")
        if (run / "best.pt").exists():
            payload = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            head.load_state_dict(payload["model_state"])
            best_path = run / "best.pt"
        else:
            best_path = run / "last.pt"
        final_scores = [score_bundle(head, scales, bases_device, b, device) for b in confirm]

    peak_vram = int(torch.cuda.max_memory_reserved(device))
    if stage == "smoke":
        oracle_by_id = {b.sample_id: oracle_bound(b, context["bases"]) for b in confirm}
        gates = smoke_gates(plan, initial_loss, final_scores, oracle_by_id,
                            peak_vram, checkpoint_bytes)
    else:
        oracle_by_id = None
        gates = eval_gates(plan, final_scores, peak_vram, checkpoint_bytes)

    all_passed = all(bool(v.get("passed")) for v in gates.values())
    if status not in {"completed", "early_stop_patience"}:
        final_status = status  # budget_limit is honest and terminal
    else:
        final_status = "passed" if all_passed else "fail_gate"

    parent_after = sha256_file(PARENT_PATH)
    payload = {
        "schema": "r16_dscp_v16_stage_terminal_v1",
        "candidate": CANDIDATE,
        "stage": stage,
        "status": final_status,
        "run_status": status,
        "preregistration_sha256": PREREG_SHA256,
        "authorization": "lead conditional directive 2026-08-26 recorded in the preregistration",
        "gates": gates,
        "metrics": {
            "initial_loss": initial_loss if stage == "smoke" else None,
            "final_records": final_scores,
            "trajectory": trajectory,
            "oracle_by_record": oracle_by_id,
            "fit_records": len(fit), "eval_records": len(confirm),
            "cache_build_s": context["cache_build_s"],
        },
        "resources": {
            "wall_s": time.monotonic() - started,
            "peak_cuda_reserved_bytes": peak_vram,
            "checkpoint_bytes": checkpoint_bytes,
            "best_checkpoint": str(best_path),
        },
        "parent": {
            "sha256_before": PARENT_SHA256, "sha256_after": parent_after,
            "match": parent_after == PARENT_SHA256, "writes": 0,
        },
        "sealed_splits_opened": False,
        "thresholds_moved_toward_measured_values": False,
        "disclaimers": {"stage": DISCLAIMERS[stage],
                        "correction_nature": DISCLAIMERS["correction_nature"]},
        "bindings": bindings,
        "completed_utc": utc_now(),
    }
    atomic_json(payload, run / "terminal.json")
    log(f"[{stage}] {final_status} wall {payload['resources']['wall_s']:.0f}s "
        f"peak {peak_vram / 2**30:.2f}GiB")
    if parent_after != PARENT_SHA256:
        raise V16Refusal("parent checkpoint changed during the stage")
    return payload


# ---------------------------------------------------------------------------
# entry: single stage or the authorized auto chain
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["verify-bindings", "smoke", "pilot", "long", "auto"])
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    args = parser.parse_args(argv)

    if args.mode == "verify-bindings":
        print(json.dumps(verify_bindings(), indent=1, default=str))
        return 0

    device = torch.device("cuda:0")
    log = lambda msg: print(f"{utc_now()} {msg}", flush=True)

    chain = ["smoke", "pilot", "long"] if args.mode == "auto" else [args.mode]
    previous_checkpoint = None
    for stage in chain:
        terminal = OUT / stage / "terminal.json"
        if terminal.exists():
            existing = json.loads(terminal.read_text())
            if existing.get("status") == "passed":
                log(f"[{stage}] already passed, reusing")
                previous_checkpoint = Path(existing["resources"]["best_checkpoint"])
                continue
            raise V16Refusal(f"{stage} terminal exists with status {existing.get('status')}")
        payload = run_stage(stage, device, input_checkpoint=previous_checkpoint, log=log)
        if payload["status"] != "passed":
            log(f"chain stops at {stage}: {payload['status']}")
            return 1
        previous_checkpoint = Path(payload["resources"]["best_checkpoint"])
    log("chain complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
