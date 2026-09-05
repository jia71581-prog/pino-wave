#!/usr/bin/env python3
"""One-record train-only smoke for B2 Q1-FE weak instance adaptation."""
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
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.instance_adaptation.b2_fe_weak_adapter import (
    FEWeakAdapterConfig,
    adapt_decoder_channel_scales,
)
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


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
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        for path, key in (
            (Path(__file__), "probe_sha256"),
            (ROOT / "saved_time_phase_operator_v4/instance_adaptation/fe_weak_residual.py", "fe_residual_sha256"),
            (ROOT / "saved_time_phase_operator_v4/instance_adaptation/b2_fe_weak_adapter.py", "adapter_sha256"),
            (args.checkpoint, "checkpoint_sha256"),
            (args.cache, "cache_sha256"),
            (args.manifest, "manifest_sha256"),
        ):
            if _sha256(path) != bindings[key]:
                raise RuntimeError(f"binding drift: {path}")
        manifest = json.loads(args.manifest.read_text())
        if not 0 <= args.record_index < len(manifest["records"]):
            raise ValueError("record index outside manifest")
        row = manifest["records"][args.record_index]
        device = torch.device("cuda")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        identity = checkpoint["identity"]
        with h5py.File(args.cache, "r", swmr=True) as cache:
            if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError("cache manifest drift")
            if cache["sample_id"][args.record_index].decode() != row["sample_id"]:
                raise RuntimeError("cache record order drift")
            base_seq = torch.from_numpy(
                cache["base_seq"][args.record_index:args.record_index + 1].astype(np.float32)
            )[:, :, None].to(device)
            cond = torch.from_numpy(
                cache["cond"][args.record_index:args.record_index + 1].astype(np.float32)
            ).to(device)
            observed_true = torch.from_numpy(
                cache["target"][args.record_index:args.record_index + 1, :8].astype(np.float32)
            )[:, :, None].to(device)
        model = FrameConditionedPropagator(
            state_channels=int(identity["ic_frames"]),
            cond_channels=cond.shape[1],
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
        with h5py.File(source_path, "r", swmr=True) as source:
            source_index = int(row["source_index"])
            velocity = torch.from_numpy(
                np.asarray(source["velocity_mps"][source_index], dtype=np.float32)
            )[None].to(device)
            time_s = np.asarray(source["time_s"][:], dtype=np.float64)
            f0 = float(source["source_f0_hz"][source_index])
            t0 = float(source["source_t0_s"][source_index])
            x_m = np.asarray(source["x_m"][:], dtype=np.float64)
            z_m = np.asarray(source["z_m"][:], dtype=np.float64)
        window_times = time_s[row["window_start"]:row["window_start"] + 64]
        source_off_frame = int(np.searchsorted(window_times, t0 + 1.5 / f0))
        source_off_frame = max(8, min(source_off_frame, 61))
        adaptation_started = time.perf_counter()
        candidate, adaptation = adapt_decoder_channel_scales(
            model,
            base_seq,
            cond,
            observed_true[:, :, 0],
            observed_true,
            velocity,
            source_off_frame=source_off_frame,
            dt_s=float(np.median(np.diff(time_s))),
            dx_m=float(np.median(np.diff(x_m))),
            dz_m=float(np.median(np.diff(z_m))),
            config=FEWeakAdapterConfig(steps=args.steps),
        )
        adaptation_wall_s = time.perf_counter() - adaptation_started
        candidate_path = args.output_dir / "sealed_candidate.pt"
        torch.save(
            {
                "sample_id": row["sample_id"],
                "candidate": candidate.cpu().half(),
                "adaptation": adaptation,
            },
            candidate_path,
        )
        candidate_sha256 = _sha256(candidate_path)
        seal = {
            "schema": "b2_fe_weak_candidate_seal_v1",
            "sample_id": row["sample_id"],
            "candidate": str(candidate_path),
            "candidate_sha256": candidate_sha256,
            "future_truth_read_before_seal": False,
            "accessed_true_frames": list(range(8)),
            "source_off_frame": source_off_frame,
            "adaptation_wall_s": adaptation_wall_s,
        }
        _atomic_json(seal, args.output_dir / "candidate_seal.json")

        # Future train truth is opened only after candidate materialization/hash.
        with h5py.File(args.cache, "r", swmr=True) as cache:
            future_truth = torch.from_numpy(
                cache["target"][args.record_index:args.record_index + 1, 8:].astype(np.float32)
            )[:, :, None]
        parent = base_seq.detach().cpu()
        # Reconstruct the frozen B2 parent once for same-protocol scoring.
        with torch.no_grad():
            parent = model.forward_anchored(
                base_seq,
                cond,
                initial_state=observed_true[:, :, 0],
            ).cpu()
        candidate_cpu = torch.load(
            candidate_path, map_location="cpu", weights_only=False
        )["candidate"].float()
        parent_rel = _relative_l2(parent[:, 8:], future_truth)
        adapted_rel = _relative_l2(candidate_cpu[:, 8:], future_truth)
        report = {
            "schema": "b2_fe_weak_train_smoke_report_v1",
            "sample_id": row["sample_id"],
            "record_index": args.record_index,
            "parent_relative_l2": parent_rel,
            "adapted_relative_l2": adapted_rel,
            "relative_gain": (parent_rel - adapted_rel) / max(parent_rel, 1.0e-16),
            "adaptation": adaptation,
            "candidate_sha256": candidate_sha256,
            "future_truth_opened_after_seal": True,
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(report, args.output_dir / "report.json")
        _atomic_json({
            "status": "complete",
            "report": str(args.output_dir / "report.json"),
            "candidate_seal": str(args.output_dir / "candidate_seal.json"),
        }, terminal)
        print(json.dumps(report, indent=2))
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
