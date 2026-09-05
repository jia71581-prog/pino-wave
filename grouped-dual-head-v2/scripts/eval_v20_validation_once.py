#!/usr/bin/env python3
"""v20: ONE-SHOT sealed-validation evaluation of the v18 B_data_hinge checkpoint.

Prereg: results/r16_dscp_v20_validation_once_preregistration_20260827.json.
Unseal: lead explicit approval 2026-08-27 ("批准解封"); this stage opens the
validation split exactly once for the 480 routed-family records.  test_id stays
sealed and no code path here can open it.

Worker mode (WORKER=k of 4): evaluates ids[k::4] on cuda:0 (set via
CUDA_VISIBLE_DEVICES), writes worker_k.json.  Merge mode: verifies 4 shards,
applies the frozen readouts, writes terminal.json.  Scoring is the frozen
v16.score_bundle fp32 path verbatim; records are scored one at a time and
discarded (no bundle retention, no fp16 detour)."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

v16 = load("train_r16_dscp_v16", ROOT / "scripts/train_r16_dscp_v16.py")

PREREG = ROOT / "results/r16_dscp_v20_validation_once_preregistration_20260827.json"
OUT = ROOT / "results/r16_dscp_v20_validation_once"
CKPT = ROOT / "results/r16_dscp_v18/B_data_hinge/best.pt"
CKPT_SHA = "51c0f0f0e0959e82f117ba9dd58c4fcd5bab50dc5d8fb0730473f4a17c887dfc"
FAMILIES = ("uniform", "layered", "marmousi")


def validation_ids():
    from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
    with h5py.File(parent_runtime.SOURCE_H5_PATH, "r", swmr=True) as h:
        dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
        splits, med, sids = dec(h["split"][:]), dec(h["medium_type"][:]), dec(h["sample_id"][:])
    return sorted(s for s, sp, m in zip(sids, splits, med)
                  if sp == "validation" and m in FAMILIES)


def load_validation_truth(source_index: int, sample_id: str, allowed: frozenset):
    """Audited validation truth read: split + sample_id binding per read.
    Mirrors legacy load_train_truth's checks with the split changed to
    validation under the v20 unseal.  Refuses anything not on the allowlist."""
    if sample_id not in allowed:
        raise v16.TruthScopeRefusal(f"not on the v20 validation allowlist: {sample_id}")
    from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
    with h5py.File(parent_runtime.SOURCE_H5_PATH, "r", swmr=True) as h:
        raw_sid = h["sample_id"][source_index]
        raw_split = h["split"][source_index]
        sid = raw_sid.decode() if isinstance(raw_sid, bytes) else str(raw_sid)
        spl = raw_split.decode() if isinstance(raw_split, bytes) else str(raw_split)
        if sid != sample_id or spl != "validation":
            raise v16.TruthScopeRefusal(
                f"binding mismatch at index {source_index}: {sid}/{spl}")
        values = np.asarray(h["wavefield"][source_index], dtype=np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("validation truth contains non-finite values")
    return torch.from_numpy(values)


def run_worker(worker: int) -> int:
    from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
    from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (
        CONDITION_MAXIMUM, R16DSCP, deployment_features,
    )
    from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import (
        V4GuardedOnsetLoader, canonical_public, canonical_travel, deployment_args,
    )
    from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
    from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime

    device = torch.device("cuda:0")
    log = lambda m: print(f"{v16.utc_now()} [w{worker}] {m}", flush=True)
    ids_all = validation_ids()
    my_ids = ids_all[worker::4]
    allowed = frozenset(my_ids)
    log(f"{len(my_ids)}/{len(ids_all)} validation records")

    v16.configure_determinism(v16.SEED)
    artifact = torch.load(v16.BASIS, map_location="cpu", weights_only=False)
    parent_model, normalizer, manifest_payload, _ = parent_runtime.load_model_context(device)
    manifest = parent_runtime.manifest_object(manifest_payload)
    loader = V4GuardedOnsetLoader(
        GuardedOnsetDataset(parent_runtime.SOURCE_H5_PATH, manifest,
                            split="validation", sample_ids=list(my_ids))
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

    travel_builder = lambda vm, s: grid_eikonal_travel_time(
        vm, source_indices=[(round(float(s[1]) / 10), round(float(s[0]) / 10))],
        dx_m=10, dz_m=10,
    )[0]
    reference = R16DSCP(artifact["basis"], artifact["coefficient_scales"])
    bases, scales = reference.bases.clone(), reference.coefficient_scales.clone()
    bases_device = bases.to(device).float()

    if v16.sha256_file(CKPT) != CKPT_SHA:
        raise v16.V16Refusal("checkpoint sha mismatch")
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    head = v16.Wide128Head().to(device).float()
    head.load_state_dict(payload["model_state"])

    loaded = {}
    for index in range(len(loader)):
        public = canonical_public(loader[index])
        loaded[public.sample_id] = public
    missing = allowed - set(loaded)
    if missing:
        raise v16.V16Refusal(f"loader missing: {sorted(missing)[:5]}")

    rows, excluded = [], []
    started = time.monotonic()
    for pos, sid in enumerate(my_ids):
        public = loaded.pop(sid)
        parent = torch.as_tensor(parent_predictor(public)).detach().cpu().float()
        travel = canonical_travel(public, travel_builder).float()
        args = deployment_args(public, parent, travel, device)
        features, decisions, conditions = deployment_features(
            *args[:-2], bases_device, args[-2], args[-1])
        decision = decisions[0]
        if decision.abstain or float(conditions[0]) > CONDITION_MAXIMUM:
            excluded.append({"sample_id": sid, "abstain": bool(decision.abstain),
                             "condition": float(conditions[0])})
            continue
        rec = None
        for cand in manifest_payload.get("records", []):
            if cand.get("sample_id") == sid:
                rec = cand; break
        source_index = int(public.source_index)
        truth = load_validation_truth(source_index, sid, allowed)
        k1 = int(public.observed_indices[1])
        parent_field = args[7][0].detach().float()
        keep_full, _floor = v16.full_time_keep_mask(parent_field, k1=k1, tau=v16.TAU)
        family = sid.split("_")[1]
        bundle = v16.CpuBundle(
            family=family, sample_id=sid, route_index=int(decision.index), k1=k1,
            features=features.detach().float().cpu(),
            parent_full=args[7].detach().float().cpu(),
            truth_future=truth[k1 + 1:].float().cpu(),
            keep_full=keep_full.detach().cpu(),
        )
        rows.append(v16.score_bundle(head, scales, bases_device, bundle, device))
        del args, features, parent, truth, bundle
        if (pos + 1) % 20 == 0:
            log(f"{pos + 1}/{len(my_ids)} ({time.monotonic() - started:.0f}s)")
    OUT.mkdir(parents=True, exist_ok=True)
    v16.atomic_json({"worker": worker, "rows": rows, "excluded": excluded,
                     "wall_s": time.monotonic() - started},
                    OUT / f"worker_{worker}.json")
    log(f"done: {len(rows)} scored, {len(excluded)} excluded")
    return 0


def run_merge() -> int:
    log = lambda m: print(f"{v16.utc_now()} {m}", flush=True)
    if (OUT / "terminal.json").exists():
        raise v16.V16Refusal("v20 terminal already exists (one-shot stage)")
    rows, excluded, walls = [], [], []
    for k in range(4):
        shard = json.loads((OUT / f"worker_{k}.json").read_text())
        rows += shard["rows"]; excluded += shard["excluded"]; walls.append(shard["wall_s"])
    ids_all = validation_ids()
    covered = {r["sample_id"] for r in rows} | {e["sample_id"] for e in excluded}
    if covered != set(ids_all):
        raise v16.V16Refusal(f"coverage mismatch: {len(covered)} vs {len(ids_all)}")
    joint = sum(r["gain_iii"] for r in rows) / len(rows)
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r)
    fam_gain = {k: sum(x["gain_iii"] for x in v) / len(v) for k, v in sorted(fam.items())}
    corrected = [r["aggregate_rel_l2"] for r in rows]
    parent_abs = [r["parent_rel_l2"] for r in rows]
    tol_ok = sum(1 for r in rows if r["gain_iii"] >= -0.01)
    worst = min(r["gain_iii"] for r in rows)
    r1_max = max(corrected); r1_mean = sum(corrected) / len(corrected)
    r2 = {
        "joint_improvement": {"value": joint, "threshold": 0.01, "passed": joint >= 0.01},
        "per_family_improvement": {"value": fam_gain, "threshold": 0.005,
                                   "passed": all(v >= 0.005 for v in fam_gain.values())},
        "nonworse_within_tolerance": {"fraction": tol_ok / len(rows), "min_fraction": 0.90,
                                      "count": tol_ok, "total": len(rows),
                                      "passed": tol_ok / len(rows) >= 0.90},
        "worst_harm": {"value": worst, "floor": -0.02, "passed": worst >= -0.02},
    }
    terminal = {
        "schema": "r16_dscp_v20_validation_once_terminal_v1",
        "status": "success",
        "preregistration_sha256": v16.sha256_file(PREREG),
        "checkpoint_sha256": CKPT_SHA,
        "counts": {"scored": len(rows), "excluded_abstain": len(excluded),
                   "per_family": {k: len(v) for k, v in sorted(fam.items())}},
        "excluded": excluded,
        "R1_inherited_absolute": {
            "corrected_rel_l2_max": r1_max, "corrected_rel_l2_mean": r1_mean,
            "parent_rel_l2_mean": sum(parent_abs) / len(parent_abs),
            "gate_max_le_0p05": r1_max <= 0.05, "gate_mean_le_0p05": r1_mean <= 0.05,
            "per_family_corrected_mean": {k: sum(x["aggregate_rel_l2"] for x in v) / len(v)
                                          for k, v in sorted(fam.items())},
            "per_family_parent_mean": {k: sum(x["parent_rel_l2"] for x in v) / len(v)
                                       for k, v in sorted(fam.items())},
        },
        "R2_correction_transport": r2,
        "records": rows,
        "truth_scope": "validation split opened once for the 480 routed-family records under the recorded lead unseal; test_id sealed",
        "resources": {"worker_wall_s": walls},
        "parent_untouched": v16.sha256_file(v16.PARENT_PATH) == v16.PARENT_SHA256,
        "completed_utc": v16.utc_now(),
    }
    v16.atomic_json(terminal, OUT / "terminal.json")
    r2_pass = all(v["passed"] for v in r2.values())
    log(f"R1 max {r1_max:.3f} mean {r1_mean:.3f} (gate 0.05: max {'PASS' if r1_max<=0.05 else 'FAIL'}/mean {'PASS' if r1_mean<=0.05 else 'FAIL'})")
    log(f"R2 transport: {'ALL PASS' if r2_pass else 'FAIL'} joint {joint:+.4f} fam {fam_gain} tol {tol_ok}/{len(rows)} worst {worst:+.4f}")
    return 0


if __name__ == "__main__":
    w = os.environ.get("WORKER")
    raise SystemExit(run_worker(int(w)) if w is not None else run_merge())
