#!/usr/bin/env python3
"""r4e14: no-harm hinge & long-horizon probe on the v16 long_fit split.

Spec: results/r4e14_noharm_hinge_spec_20260827.md (sha in terminal).
Advisory probe, not a gated stage.  Uses ONLY the 192 long_fit records from
the prebuilt /dev/shm shards; long_calibration and pilot_confirm stay
untouched, sealed splits stay sealed.  The frozen v16 runner is imported for
every training/scoring component; the only new math is the differentiable
hinge added on top of losses["total"]."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "train_r16_dscp_v16", ROOT / "scripts/train_r16_dscp_v16.py"
)
v16 = importlib.util.module_from_spec(spec)
sys.modules["train_r16_dscp_v16"] = v16
spec.loader.exec_module(v16)

SHARD_DIR = Path("/dev/shm/v16_long_bundles")
OUT = ROOT / "results/r4e14_noharm_hinge_20260827"
SPEC_PATH = ROOT / "results/r4e14_noharm_hinge_spec_20260827.md"

UPDATES = 12288
EVAL_EVERY = 2048
ARMS = [
    {"name": "A_control", "lam": 0.0, "margin": 0.0},
    {"name": "B_hinge", "lam": 1.0, "margin": 0.0},
    {"name": "C_hinge_margin", "lam": 1.0, "margin": 0.02},
]


def log(msg: str) -> None:
    print(f"{v16.utc_now()} {msg}", flush=True)


def load_all_shards():
    by_id, bases, scales = {}, None, None
    shards = sorted(SHARD_DIR.glob("shard_*.pt"))
    if len(shards) != 3:
        raise v16.V16Refusal(f"expected 3 shards, found {len(shards)}")
    for path in shards:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for bundle in payload["bundles"]:
            by_id[bundle.sample_id] = bundle
        bases, scales = payload["bases"], payload["scales"]
        log(f"loaded {path.name} ({len(payload['bundles'])} bundles)")
    return by_id, bases, scales


def split_fit_holdout(fit_ids):
    """Per-family lexicographic order, every 4th record (index%4==3) held out."""
    by_family: dict[str, list[str]] = {}
    for sid in sorted(fit_ids):
        family = sid.split("_")[1]
        by_family.setdefault(family, []).append(sid)
    fit, holdout = [], []
    for family in sorted(by_family):
        for idx, sid in enumerate(by_family[family]):
            (holdout if idx % 4 == 3 else fit).append(sid)
    return fit, holdout


def differentiable_rel_l2(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - truth) / \
        torch.linalg.vector_norm(truth).clamp_min(1e-30)


def evaluate(head, scales, bases_device, bundles, device):
    rows = [v16.score_bundle(head, scales, bases_device, b, device) for b in bundles]
    gains = [r["gain_iii"] for r in rows]
    harms = [g for g in gains if g < 0]
    per_family: dict[str, list[float]] = {}
    for r in rows:
        per_family.setdefault(r["family"], []).append(r["gain_iii"])
    return {
        "joint_mean_gain": sum(gains) / len(gains),
        "harm_count": len(harms),
        "worst_harm": min(gains),
        "harm_mean": sum(harms) / len(harms) if harms else 0.0,
        "nonworse_count": sum(1 for r in rows if r["nonworse"]),
        "per_family_mean_gain": {f: sum(v) / len(v) for f, v in sorted(per_family.items())},
        "records": rows,
    }


def run_arm(arm, fit, holdout, bases_device, scales, parent_rel, device):
    v16.configure_determinism(v16.SEED)
    head = v16.Wide128Head().to(device).float()
    if sum(p.numel() for p in head.parameters()) != v16.EXPECTED_PARAMETERS:
        raise v16.V16Refusal("head parameter count is not the preregistered 25266")
    optimizer = torch.optim.AdamW(head.parameters(), lr=v16.LR, betas=(0.9, 0.99),
                                  eps=1e-8, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(v16.SEED)
    order = torch.randperm(len(fit), generator=generator).tolist()
    lam, margin = float(arm["lam"]), float(arm["margin"])
    snapshots, started = [], time.monotonic()
    for update in range(UPDATES):
        if update % len(fit) == 0 and update:
            order = torch.randperm(len(fit), generator=generator).tolist()
        bundle = fit[order[update % len(order)]]
        losses, adapted, _parent_full, truth_future = v16.bundle_loss(
            head, scales, bases_device, bundle, device
        )
        total = losses["total"]
        if lam > 0.0:
            cand_rel = differentiable_rel_l2(
                adapted[0, bundle.k1 + 1:].float(), truth_future.float()
            )
            ratio = cand_rel / parent_rel[bundle.sample_id]
            total = total + lam * torch.relu(ratio - (1.0 - margin))
        if not torch.isfinite(total):
            raise v16.V16Refusal(f"non-finite loss at update {update} ({bundle.sample_id})")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise v16.V16Refusal("nonfinite or absent gradient")
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if (update + 1) % EVAL_EVERY == 0:
            snap = evaluate(head, scales, bases_device, holdout, device)
            snap["update"] = update + 1
            snapshots.append(snap)
            log(f"[{arm['name']}] upd {update + 1} joint {snap['joint_mean_gain']:+.4f} "
                f"harms {snap['harm_count']}/48 worst {snap['worst_harm']:+.4f} "
                f"nonworse {snap['nonworse_count']}/48")
    return {
        "arm": arm, "wall_s": time.monotonic() - started,
        "snapshots": [{k: v for k, v in s.items() if k != "records"} for s in snapshots],
        "final": snapshots[-1],
    }


def main() -> int:
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "terminal.json").exists():
        raise v16.V16Refusal("r4e14 terminal already exists")
    by_id, bases, scales = load_all_shards()
    fit_ids = v16.role_sample_ids("long_fit")
    missing = [s for s in fit_ids if s not in by_id]
    if missing:
        raise v16.V16Refusal(f"shards missing long_fit records: {missing[:5]}")
    fit_ids_split, holdout_ids = split_fit_holdout(fit_ids)
    log(f"split fit {len(fit_ids_split)} holdout {len(holdout_ids)} "
        f"(families {sorted(set(s.split('_')[1] for s in holdout_ids))})")
    fit = [by_id[s] for s in fit_ids_split]
    holdout = [by_id[s] for s in holdout_ids]
    bases_device = bases.to(device).float()
    parent_rel = {}
    for bundle in fit:
        parent_rel[bundle.sample_id] = float(v16.relative_l2(
            bundle.parent_full[0, bundle.k1 + 1:].float(), bundle.truth_future
        ))
    torch.cuda.reset_peak_memory_stats(device)
    results = [run_arm(arm, fit, holdout, bases_device, scales, parent_rel, device)
               for arm in ARMS]
    terminal = {
        "schema": "r4e14_noharm_hinge_terminal_v1",
        "status": "success",
        "spec_sha256": v16.sha256_file(SPEC_PATH),
        "updates": UPDATES, "eval_every": EVAL_EVERY, "seed": v16.SEED,
        "split": {"fit": fit_ids_split, "holdout": holdout_ids},
        "arms": results,
        "truth_scope": "train/long_fit only via prebuilt shards; no new truth opened",
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "completed_utc": v16.utc_now(),
    }
    v16.atomic_json(terminal, OUT / "terminal.json")
    log(f"done; terminal {OUT / 'terminal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
