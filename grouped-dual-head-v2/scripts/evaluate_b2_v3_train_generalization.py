#!/usr/bin/env python3
"""Evaluate frozen B2-v3 checkpoints on a preregistered train-only panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_b2_snapshot_ic import FrameConditionedPropagator, build_cache

FAMILIES = ("uniform", "layered", "marmousi")
TEMPORAL_BANDS = {"early": (0, 19), "middle": (19, 38), "late": (38, 56)}
FREQUENCY_BANDS = {"low": (0, 5), "middle": (5, 12), "high": (12, 29)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _relative_l2(prediction, target, dims):
    numerator = (prediction - target).abs().square().sum(dims).sqrt()
    denominator = target.abs().square().sum(dims).clamp_min(1.0e-16).sqrt()
    return numerator / denominator


def judge_lanes(lanes: list[dict], gate: dict) -> dict:
    required_nonworse = int(gate["minimum_nonworse_records"])
    minimum_gain = float(gate["minimum_absolute_gain_vs_anchor"])
    lane_results = []
    for lane in lanes:
        aggregate_pass = lane["aggregate"] <= lane["baseline_aggregate"] - minimum_gain
        family_pass = all(
            lane["per_family"][name]["candidate"]
            < lane["per_family"][name]["baseline"]
            for name in FAMILIES
        )
        nonworse_pass = lane["nonworse_count"] >= required_nonworse
        lane_results.append({
            "lane": lane["lane"],
            "aggregate_pass": aggregate_pass,
            "family_pass": family_pass,
            "nonworse_pass": nonworse_pass,
            "passed": aggregate_pass and family_pass and nonworse_pass,
        })
    return {
        "passed": len(lane_results) == 2 and all(row["passed"] for row in lane_results),
        "lanes": lane_results,
    }


def _evaluate_checkpoint(checkpoint_path: Path, cache, device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = checkpoint["identity"]
    model = FrameConditionedPropagator(
        state_channels=int(identity["ic_frames"]),
        cond_channels=int(cache["cond"].shape[1]),
        width=int(identity["width"]),
        spectral_rank=int(identity["spectral_rank"]),
        modes=24,
        depth=4,
        gate_init=1.0,
        activation_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    sample_ids = cache["sample_id"][:].astype(str)
    families = cache["family"][:].astype(str)
    records = []
    started = time.time()
    for lo in range(0, len(sample_ids), 2):
        hi = min(lo + 2, len(sample_ids))
        base = torch.from_numpy(cache["base_seq"][lo:hi].astype(np.float32))[:, :, None].to(device)
        target = torch.from_numpy(cache["target"][lo:hi].astype(np.float32))[:, :, None].to(device)
        cond = torch.from_numpy(cache["cond"][lo:hi].astype(np.float32)).to(device)
        initial = target[:, 8 - int(identity["ic_frames"]):8, 0]
        with torch.no_grad():
            prediction = model.forward_anchored(base, cond, initial_state=initial)
        pred = prediction[:, 8:, 0].double()
        truth = target[:, 8:, 0].double()
        anchor = base[:, 8:, 0].double()
        candidate_rel = _relative_l2(pred, truth, (1, 2, 3))
        baseline_rel = _relative_l2(anchor, truth, (1, 2, 3))
        correction = _relative_l2(pred, anchor, (1, 2, 3))
        temporal = {
            name: _relative_l2(pred[:, start:stop], truth[:, start:stop], (1, 2, 3))
            for name, (start, stop) in TEMPORAL_BANDS.items()
        }
        pred_fft = torch.fft.rfft(pred, dim=1)
        truth_fft = torch.fft.rfft(truth, dim=1)
        frequency = {
            name: _relative_l2(
                pred_fft[:, start:stop], truth_fft[:, start:stop], (1, 2, 3)
            )
            for name, (start, stop) in FREQUENCY_BANDS.items()
        }
        for offset in range(hi - lo):
            records.append({
                "sample_id": sample_ids[lo + offset],
                "family": families[lo + offset],
                "candidate": float(candidate_rel[offset]),
                "baseline": float(baseline_rel[offset]),
                "delta": float(candidate_rel[offset] - baseline_rel[offset]),
                "correction_energy": float(correction[offset]),
                "temporal": {name: float(value[offset]) for name, value in temporal.items()},
                "frequency": {name: float(value[offset]) for name, value in frequency.items()},
            })
    result = {
        "lane": checkpoint_path.parent.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "record_count": len(records),
        "aggregate": float(np.mean([row["candidate"] for row in records])),
        "baseline_aggregate": float(np.mean([row["baseline"] for row in records])),
        "maximum": float(max(row["candidate"] for row in records)),
        "nonworse_count": sum(row["delta"] <= 0.0 for row in records),
        "mean_correction_energy": float(np.mean([row["correction_energy"] for row in records])),
        "per_family": {},
        "temporal": {},
        "frequency": {},
        "worst_records": sorted(records, key=lambda row: row["candidate"], reverse=True)[:10],
        "elapsed_s": round(time.time() - started, 3),
    }
    for name in FAMILIES:
        rows = [row for row in records if row["family"] == name]
        result["per_family"][name] = {
            "candidate": float(np.mean([row["candidate"] for row in rows])),
            "baseline": float(np.mean([row["baseline"] for row in rows])),
            "nonworse": sum(row["delta"] <= 0.0 for row in rows),
            "n": len(rows),
        }
    for name in TEMPORAL_BANDS:
        result["temporal"][name] = float(np.mean([row["temporal"][name] for row in records]))
    for name in FREQUENCY_BANDS:
        result["frequency"][name] = float(np.mean([row["frequency"][name] for row in records]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--build-cache", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    terminal_path = args.output_dir / "terminal.json"
    try:
        manifest = json.loads(args.manifest.read_text())
        prereg = json.loads(args.preregistration.read_text())
        if manifest.get("split") != "train":
            raise RuntimeError("diagnostic manifest must be train-only")
        if manifest.get("validation_opened") or manifest.get("test_id_opened"):
            raise RuntimeError("sealed split flag is open")
        if _sha256(args.manifest) != prereg["bindings"]["manifest_sha256"]:
            raise RuntimeError("manifest binding drift")
        expected = prereg["bindings"]["checkpoint_sha256"]
        for checkpoint in args.checkpoint:
            if _sha256(checkpoint) != expected[str(checkpoint)]:
                raise RuntimeError(f"checkpoint binding drift: {checkpoint}")
        if args.cache.exists():
            raise FileExistsError(f"refusing to reuse cache: {args.cache}")
        if not args.build_cache:
            raise RuntimeError("cache is absent; pass --build-cache")
        device = torch.device("cuda")
        build_cache(manifest, device, cache_path=args.cache, data_split="train")
        with h5py.File(args.cache, "r", swmr=True) as cache:
            if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError("cache manifest binding drift")
            if cache.attrs.get("data_split", "") != "train":
                raise RuntimeError("cache is not train-only")
            lanes = [_evaluate_checkpoint(path, cache, device) for path in args.checkpoint]
        judgement = judge_lanes(lanes, prereg["gate"])
        metrics = {
            "schema": "b2_v3_train_generalization_metrics_v1",
            "lanes": lanes,
            "judgement": judgement,
        }
        _atomic_json(metrics, args.output_dir / "metrics.json")
        identity = {
            "schema": "b2_v3_train_generalization_identity_v1",
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "cache": str(args.cache),
            "cache_sha256": _sha256(args.cache),
            "evaluator_sha256": _sha256(Path(__file__)),
            "checkpoint_sha256": expected,
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        _atomic_json({
            "status": "passed" if judgement["passed"] else "rejected",
            "judgement": judgement,
            "metrics": str(args.output_dir / "metrics.json"),
            "run_identity": str(args.output_dir / "run_identity.json"),
        }, terminal_path)
        print(json.dumps({"terminal": str(terminal_path), **json.loads(terminal_path.read_text())}, indent=2))
        return 0
    except Exception as error:
        import traceback
        _atomic_json({
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }, terminal_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
