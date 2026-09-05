#!/usr/bin/env python3
"""Paired observed-only versus FE-weak B2 adaptation on a frozen train panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.instance_adaptation.b2_fe_weak_adapter import (
    FEWeakAdapterConfig,
    adapt_decoder_channel_scales,
)
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator

FAMILIES = ("uniform", "layered", "marmousi")


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


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (prediction.double() - target.double()).square().sum().sqrt()
        / target.double().square().sum().clamp_min(1.0e-16).sqrt()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    candidates_dir = args.output_dir / "candidates"
    candidates_dir.mkdir()
    terminal = args.output_dir / "terminal.json"
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        for path, key in (
            (Path(__file__), "panel_script_sha256"),
            (ROOT / "saved_time_phase_operator_v4/instance_adaptation/fe_weak_residual.py", "fe_residual_sha256"),
            (ROOT / "saved_time_phase_operator_v4/instance_adaptation/b2_fe_weak_adapter.py", "adapter_sha256"),
            (args.checkpoint, "checkpoint_sha256"),
            (args.cache, "cache_sha256"),
            (args.manifest, "manifest_sha256"),
        ):
            if _sha256(path) != bindings[key]:
                raise RuntimeError(f"binding drift: {path}")
        manifest = json.loads(args.manifest.read_text())
        indices = [int(value) for value in prereg["panel"]["record_indices"]]
        rows = [manifest["records"][index] for index in indices]
        if Counter(row["family"] for row in rows) != Counter(
            {"uniform": 4, "layered": 4, "marmousi": 4}
        ):
            raise RuntimeError("panel is not 4/4/4 family balanced")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        identity = checkpoint["identity"]
        device = torch.device("cuda")
        model = FrameConditionedPropagator(
            state_channels=int(identity["ic_frames"]),
            cond_channels=7,
            width=int(identity["width"]),
            spectral_rank=int(identity["spectral_rank"]),
            modes=24,
            depth=4,
            gate_init=1.0,
            activation_checkpointing=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        source_path = Path(manifest["source_h5"])
        reports = []
        for panel_position, (record_index, row) in enumerate(zip(indices, rows)):
            with h5py.File(args.cache, "r", swmr=True) as cache:
                if cache["sample_id"][record_index].decode() != row["sample_id"]:
                    raise RuntimeError("cache record order drift")
                base_seq = torch.from_numpy(
                    cache["base_seq"][record_index:record_index + 1].astype(np.float32)
                )[:, :, None].to(device)
                cond = torch.from_numpy(
                    cache["cond"][record_index:record_index + 1].astype(np.float32)
                ).to(device)
                observed = torch.from_numpy(
                    cache["target"][record_index:record_index + 1, :8].astype(np.float32)
                )[:, :, None].to(device)
            with h5py.File(source_path, "r", swmr=True) as source:
                source_index = int(row["source_index"])
                velocity = torch.from_numpy(
                    np.asarray(source["velocity_mps"][source_index], dtype=np.float32)
                )[None].to(device)
                time_s = np.asarray(source["time_s"][:], dtype=np.float64)
                x_m = np.asarray(source["x_m"][:], dtype=np.float64)
                z_m = np.asarray(source["z_m"][:], dtype=np.float64)
                f0 = float(source["source_f0_hz"][source_index])
                t0 = float(source["source_t0_s"][source_index])
            window_times = time_s[row["window_start"]:row["window_start"] + 64]
            source_off = max(
                8, min(int(np.searchsorted(window_times, t0 + 1.5 / f0)), 61)
            )
            with torch.no_grad():
                parent = model.forward_anchored(
                    base_seq, cond, initial_state=observed[:, :, 0]
                ).cpu()
            sealed = {}
            adaptations = {}
            for arm, weak_weight in (("observed_only", 0.0), ("fe_weak", 0.1)):
                candidate, adaptation = adapt_decoder_channel_scales(
                    model,
                    base_seq,
                    cond,
                    observed[:, :, 0],
                    observed,
                    velocity,
                    source_off_frame=source_off,
                    dt_s=float(np.median(np.diff(time_s))),
                    dx_m=float(np.median(np.diff(x_m))),
                    dz_m=float(np.median(np.diff(z_m))),
                    config=FEWeakAdapterConfig(weak_weight=weak_weight),
                )
                path = candidates_dir / f"{panel_position:02d}_{row['sample_id']}_{arm}.pt"
                torch.save({
                    "sample_id": row["sample_id"],
                    "arm": arm,
                    "candidate": candidate.cpu().half(),
                    "adaptation": adaptation,
                }, path)
                sealed[arm] = {"path": str(path), "sha256": _sha256(path)}
                adaptations[arm] = adaptation
                del candidate
            # Both paired candidates are immutable before future truth is read.
            with h5py.File(args.cache, "r", swmr=True) as cache:
                future = torch.from_numpy(
                    cache["target"][record_index:record_index + 1, 8:].astype(np.float32)
                )[:, :, None]
            parent_rel = _relative_l2(parent[:, 8:], future)
            record_report = {
                "record_index": record_index,
                "sample_id": row["sample_id"],
                "family": row["family"],
                "source_off_frame": source_off,
                "parent_relative_l2": parent_rel,
                "arms": {},
                "paired_candidates_sealed_before_future_truth": True,
            }
            for arm in ("observed_only", "fe_weak"):
                candidate = torch.load(
                    sealed[arm]["path"], map_location="cpu", weights_only=False
                )["candidate"].float()
                relative = _relative_l2(candidate[:, 8:], future)
                record_report["arms"][arm] = {
                    "relative_l2": relative,
                    "relative_gain": (parent_rel - relative) / max(parent_rel, 1.0e-16),
                    "candidate_sha256": sealed[arm]["sha256"],
                    "adaptation": adaptations[arm],
                }
            reports.append(record_report)
            del base_seq, cond, observed, velocity, parent
            torch.cuda.empty_cache()
        summary = {
            "schema": "b2_fe_weak_balanced_panel_v1",
            "record_count": len(reports),
            "family_counts": dict(Counter(row["family"] for row in reports)),
            "arms": {},
            "records": reports,
            "validation_opened": False,
            "test_id_opened": False,
        }
        for arm in ("observed_only", "fe_weak"):
            values = np.asarray([row["arms"][arm]["relative_l2"] for row in reports])
            parents = np.asarray([row["parent_relative_l2"] for row in reports])
            summary["arms"][arm] = {
                "aggregate": float(values.mean()),
                "parent_aggregate": float(parents.mean()),
                "relative_gain": float((parents.mean() - values.mean()) / parents.mean()),
                "nonworse": int((values <= parents).sum()),
                "accepted": sum(row["arms"][arm]["adaptation"]["accepted"] for row in reports),
                "mean_runtime_s": float(np.mean([
                    row["arms"][arm]["adaptation"]["elapsed_s"] for row in reports
                ])),
                "per_family": {
                    family: float(np.mean([
                        row["arms"][arm]["relative_l2"]
                        for row in reports if row["family"] == family
                    ]))
                    for family in FAMILIES
                },
            }
        observed = summary["arms"]["observed_only"]
        fe_weak = summary["arms"]["fe_weak"]
        summary["fe_increment_vs_observed"] = (
            observed["aggregate"] - fe_weak["aggregate"]
        ) / max(observed["aggregate"], 1.0e-16)
        gates = prereg["gates"]
        summary["judgement"] = {
            "fe_gain_pass": fe_weak["relative_gain"] >= gates["minimum_fe_relative_gain_vs_parent"],
            "fe_nonworse_pass": fe_weak["nonworse"] >= gates["minimum_fe_nonworse"],
            "fe_increment_pass": summary["fe_increment_vs_observed"] >= gates["minimum_fe_increment_vs_observed"],
            "fe_family_pass": all(
                fe_weak["per_family"][family] <= observed["per_family"][family]
                for family in FAMILIES
            ),
            "fe_acceptance_pass": fe_weak["accepted"] >= gates["minimum_fe_accepted"],
            "fe_runtime_pass": fe_weak["mean_runtime_s"] <= gates["maximum_mean_runtime_s"],
        }
        summary["judgement"]["passed"] = all(summary["judgement"].values())
        _atomic_json(summary, args.output_dir / "summary.json")
        _atomic_json({
            "status": "passed" if summary["judgement"]["passed"] else "rejected",
            "summary": str(args.output_dir / "summary.json"),
        }, terminal)
        print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
        return 0
    except Exception as error:
        import traceback
        _atomic_json({
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
