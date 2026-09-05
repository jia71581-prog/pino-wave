#!/usr/bin/env python3
"""CPU-only full-401 audit of a pyramid-MoE checkpoint on a fixed train panel."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.train_transfer_dg_coupled_mhc_muon_pilot import (  # noqa: E402
    BLOCK,
    PilotData,
    absolute_target,
    model_prediction,
)
from scripts.train_transfer_dg_phase_scatter64_full_ddp import (  # noqa: E402
    FullCollection,
    atomic_json,
    sha256,
)
from scripts.train_transfer_dg_wfp_e1 import CacheCollection  # noqa: E402
from scripts.train_transfer_dg_wfp_e1d import TravelCollection  # noqa: E402
from saved_time_phase_operator_v4.coupled_pyramid_moe_wave import (  # noqa: E402
    PyramidMoECoupledWaveOperator,
    parameter_count,
)
from saved_time_phase_operator_v4.wfp import BackgroundFrequencyOperator  # noqa: E402


def nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize(rows: list[dict]) -> dict:
    candidates = [float(row["candidate"]) for row in rows]
    parents = [float(row["parent"]) for row in rows]
    improvements = [float(row["improvement"]) for row in rows]
    return {
        "record_count": len(rows),
        "candidate_mean": float(np.mean(candidates)),
        "candidate_median": float(np.median(candidates)),
        "candidate_p95_nearest_rank": nearest_rank_p95(candidates),
        "candidate_min": min(candidates),
        "candidate_max": max(candidates),
        "parent_mean": float(np.mean(parents)),
        "parent_median": float(np.median(parents)),
        "parent_p95_nearest_rank": nearest_rank_p95(parents),
        "mean_relative_improvement": float(np.mean(improvements)),
        "aggregate_mean_improvement": 1.0 - float(np.mean(candidates)) / max(float(np.mean(parents)), 1e-300),
        "nonworse_count": sum(candidate <= parent for candidate, parent in zip(candidates, parents)),
        "strict_target_0p05_count": sum(candidate <= 0.05 for candidate in candidates),
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    parent: torch.nn.Module,
    data: PilotData,
    positions: list[int],
    rows_path: Path,
    started: float,
) -> list[dict]:
    device = torch.device("cpu")
    model.eval()
    parent.eval()
    rows: list[dict] = []
    for index, position in enumerate(positions, start=1):
        record_started = time.time()
        candidate_low = 0.0
        parent_low = 0.0
        full_total = None
        unmodeled = None
        family = None
        sample_id = None
        for block_start in range(0, 64, BLOCK):
            batch = data.block(position, block_start, device)
            target, _ = absolute_target(parent, batch)
            prediction, _ = model_prediction(model, batch)
            weights = torch.full((BLOCK, 1, 1, 1), 2.0, dtype=torch.float64)
            if block_start == 0:
                weights[0] = 1.0
            candidate_low += float(
                ((prediction.double() - target.double()).square() * weights).sum()
            )
            parent_prediction = target - batch["residual"]
            parent_low += float(
                ((parent_prediction.double() - target.double()).square() * weights).sum()
            )
            full_total = float(batch["full_total"])
            unmodeled = float(batch["unmodeled"])
            family = str(batch["family"])
            sample_id = str(batch["sample_id"])
            del batch, target, prediction, parent_prediction, weights
        candidate = math.sqrt((candidate_low + unmodeled) / max(full_total, 1e-300))
        baseline = math.sqrt((parent_low + unmodeled) / max(full_total, 1e-300))
        row = {
            "sample_id": sample_id,
            "family": family,
            "candidate": candidate,
            "parent": baseline,
            "improvement": 1.0 - candidate / max(baseline, 1e-300),
            "record_elapsed_s": time.time() - record_started,
        }
        rows.append(row)
        with rows_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "event": "record_complete",
                    "record": index,
                    "record_count": len(positions),
                    "elapsed_s": time.time() - started,
                    **row,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        gc.collect()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-cache", type=Path, action="append", required=True)
    parser.add_argument("--base-cache", type=Path, action="append", required=True)
    parser.add_argument("--travel", type=Path, action="append", required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", default="confirmation")
    parser.add_argument("--family", default="marmousi")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("CPU-only evaluator requires CUDA_VISIBLE_DEVICES to be empty")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(4, args.threads)))

    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    observed_bindings = {
        "evaluator_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(ROOT / "saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py"),
        "full_manifest_sha256": sha256(args.full_manifest),
        "eval_manifest_sha256": sha256(args.eval_manifest),
        "candidate_checkpoint_sha256": sha256(args.candidate_checkpoint),
        "parent_checkpoint_sha256": sha256(args.parent_checkpoint),
    }
    for key, observed in observed_bindings.items():
        if observed != bindings[key]:
            raise RuntimeError(f"binding drift for {key}: {observed}")
    if args.family != prereg["evaluation"]["family"]:
        raise RuntimeError("family differs from preregistration")
    if args.role != prereg["evaluation"]["role"]:
        raise RuntimeError("role differs from preregistration")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    started = time.time()
    run_identity = {
        "schema": "transfer_dg_coupled_pyramid_moe_cpu_eval_identity_v1",
        "pid": os.getpid(),
        "device": "cpu",
        "threads": args.threads,
        "role": args.role,
        "family": args.family,
        "max_records": args.max_records,
        "bindings": observed_bindings,
        "validation_opened": False,
        "test_id_opened": False,
        "started_unix_s": started,
    }
    atomic_json(run_identity, args.output_dir / "run_identity.json")
    print(json.dumps({"event": "identity", **run_identity}, sort_keys=True), flush=True)

    residual = None
    base = None
    travel = None
    try:
        full_manifest = json.loads(args.full_manifest.read_text())
        eval_manifest = json.loads(args.eval_manifest.read_text())
        if full_manifest.get("split") != "train" or eval_manifest.get("split") != "train":
            raise RuntimeError("evaluation must remain train-only")
        if eval_manifest.get("validation_opened") or eval_manifest.get("test_id_opened"):
            raise RuntimeError("sealed split marker is open")

        residual = FullCollection(args.residual_cache, full_manifest)
        base = CacheCollection(args.base_cache, full_manifest, expected_count=2800)
        travel = TravelCollection(args.travel, expected_count=2800)
        data = PilotData(residual, base, travel, eval_manifest)
        positions = [
            position
            for position in data.roles[args.role]
            if residual.records[position][3] == args.family
        ]
        expected_count = int(prereg["evaluation"]["record_count"])
        if len(positions) != expected_count:
            raise RuntimeError(
                f"registered family count drift: expected {expected_count}, got {len(positions)}"
            )
        if args.max_records:
            positions = positions[: args.max_records]

        parent_payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
        parent = BackgroundFrequencyOperator(
            medium_channels=12,
            source_channels=5,
            width=64,
            rank=32,
            depth=6,
            arm="wfp",
            radii=(1, 2, 3, 4, 5, 6),
        )
        parent.load_state_dict(parent_payload["model_state"])
        del parent_payload
        parent.eval().requires_grad_(False)

        candidate_payload = torch.load(
            args.candidate_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_meta = {
            key: candidate_payload.get(key)
            for key in (
                "schema",
                "epoch",
                "next_step",
                "update",
                "validation_opened",
                "test_id_opened",
            )
        }
        if checkpoint_meta["validation_opened"] or checkpoint_meta["test_id_opened"]:
            raise RuntimeError("candidate checkpoint has opened a sealed split")
        model = PyramidMoECoupledWaveOperator(use_mhc=True)
        model.load_state_dict(candidate_payload["model_state"])
        del candidate_payload
        model.eval().requires_grad_(False)
        gc.collect()
        if parameter_count(model) != int(prereg["model"]["parameter_count"]):
            raise RuntimeError("candidate parameter count drift")

        rows = evaluate(
            model,
            parent,
            data,
            positions,
            args.output_dir / "rows.jsonl",
            started,
        )
        terminal = {
            "schema": "transfer_dg_coupled_pyramid_moe_cpu_eval_terminal_v1",
            "status": "complete",
            "scope": "train-only seen-data diagnostic; not a generalization or target claim",
            "device": "cpu",
            "threads": args.threads,
            "family": args.family,
            "role": args.role,
            "checkpoint": checkpoint_meta,
            "metrics": summarize(rows),
            "rows": rows,
            "elapsed_s": time.time() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(terminal, args.output_dir / "terminal.json")
        print(json.dumps({"event": "complete", **terminal}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        terminal = {
            "schema": "transfer_dg_coupled_pyramid_moe_cpu_eval_terminal_v1",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": time.time() - started,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(terminal, args.output_dir / "terminal.json")
        raise
    finally:
        if residual is not None:
            residual.close()
        if base is not None:
            base.close()
        if travel is not None:
            travel.close()


if __name__ == "__main__":
    raise SystemExit(main())
