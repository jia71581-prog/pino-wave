#!/usr/bin/env python3
"""Evaluate one frozen B2-v10 causal-prefix assimilation arm on train data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from saved_time_phase_operator_v4.instance_adaptation.b2_v10_prefix_assimilation import (
    CausalPrefixAccessAudit,
    PrefixAssimilationConfig,
    assimilate_prefix_pod,
    source_cycle_prefix_count,
)
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


FAMILIES = ("uniform", "layered", "marmousi")
ARMS = ("peak", "cycle025", "cycle050", "fixed24")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (prediction.double() - target.double()).norm()
        / target.double().norm().clamp_min(1.0e-16)
    )


def _thirds_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    indices = np.array_split(np.arange(prediction.shape[1]), 3)
    values = []
    for index in indices:
        if len(index) == 0:
            values.append(float("nan"))
        else:
            start, stop = int(index[0]), int(index[-1]) + 1
            values.append(_relative_l2(prediction[:, start:stop], target[:, start:stop]))
    return values


def _frequency_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_floor_fraction: float = 0.005,
) -> list[float]:
    prediction_fft = torch.fft.rfft(prediction.float(), dim=1)
    target_fft = torch.fft.rfft(target.float(), dim=1)
    bins = np.array_split(np.arange(prediction_fft.shape[1]), 3)
    total_energy = target_fft.abs().square().sum().clamp_min(1.0e-16)
    values = []
    for index in bins:
        if len(index) == 0:
            values.append(float("nan"))
            continue
        start, stop = int(index[0]), int(index[-1]) + 1
        target_band = target_fft[:, start:stop]
        error_band = prediction_fft[:, start:stop] - target_band
        numerator = error_band.abs().square().sum().sqrt()
        denominator = torch.maximum(
            target_band.abs().square().sum(),
            energy_floor_fraction * total_energy,
        ).sqrt()
        values.append(float(numerator / denominator))
    return values


def _nearest_rank_percentile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _load_parent(checkpoint: dict, device: torch.device) -> tuple[torch.nn.Module, str]:
    identity = checkpoint["identity"]
    conditioning_key = str(identity["conditioning_key"])
    model = FrameConditionedPropagator(
        state_channels=8,
        cond_channels=int(identity["cond_channels"]),
        width=int(identity.get("width", 64)),
        spectral_rank=int(identity.get("spectral_rank", 32)),
        modes=24,
        depth=4,
        gate_init=1.0,
        activation_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, conditioning_key


def _observed_count(
    arm: str,
    window_time_s: torch.Tensor,
    row: dict,
    policy: dict,
) -> int:
    arm_policy = policy["arms"][arm]
    if "fixed_frames" in arm_policy:
        count = int(arm_policy["fixed_frames"])
        if not int(policy["anchor_frames"]) <= count < len(window_time_s):
            raise ValueError("fixed observation count is outside the window")
        return count
    return source_cycle_prefix_count(
        window_time_s,
        source_t0_s=float(row["source_t0_s"]),
        source_f0_hz=float(row["source_f0_hz"]),
        after_peak_cycles=float(arm_policy["after_peak_cycles"]),
        minimum_frames=int(policy["anchor_frames"]),
        maximum_frames=int(policy["maximum_observed_frames"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
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
        core_path = ROOT / "saved_time_phase_operator_v4" / "instance_adaptation" / "b2_v10_prefix_assimilation.py"
        for path, key in (
            (Path(__file__), "evaluator_sha256"),
            (core_path, "core_sha256"),
            (args.checkpoint, "checkpoint_sha256"),
            (args.cache, "evaluation_cache_sha256"),
            (args.manifest, "evaluation_manifest_sha256"),
            (args.bundle, "bundle_sha256"),
        ):
            if _sha256(path) != bindings[key]:
                raise RuntimeError(f"binding drift: {path}")
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("split") != "train":
            raise RuntimeError("development evaluation must use train records")
        if manifest.get("validation_opened") or manifest.get("test_id_opened"):
            raise RuntimeError("sealed split flag is open")
        if manifest.get("future_truth_opened_for_window_selection"):
            raise RuntimeError("future-derived window selection is forbidden")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
        if bundle.get("schema") != "b2_v10_residual_pod_bundle_v1":
            raise RuntimeError("unexpected POD bundle schema")
        if bundle["parent_checkpoint_sha256"] != _sha256(args.checkpoint):
            raise RuntimeError("POD bundle parent drift")
        policy = prereg["online_policy"]
        if int(bundle["rank"]) != int(policy["rank"]):
            raise RuntimeError("POD rank/policy drift")
        device = torch.device("cuda")
        model, conditioning_key = _load_parent(checkpoint, device)
        with h5py.File(manifest["source_h5"], "r", swmr=True) as source:
            global_time_s = torch.from_numpy(np.asarray(source["time_s"][:], np.float64))
        with h5py.File(args.cache, "r", swmr=True) as cache:
            if cache.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
                raise RuntimeError("evaluation cache is not B2-v5 causal")
            if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError("evaluation cache/manifest drift")
            if conditioning_key not in cache:
                raise RuntimeError(f"conditioning dataset absent: {conditioning_key}")
            cached_ids = cache["sample_id"][:].astype(str).tolist()
        manifest_ids = [row["sample_id"] for row in manifest["records"]]
        if cached_ids != manifest_ids:
            raise RuntimeError("cache sample order differs from manifest")
        identity = {
            "schema": "b2_v10_online_arm_identity_v1",
            "arm": args.arm,
            "checkpoint_sha256": _sha256(args.checkpoint),
            "cache_sha256": _sha256(args.cache),
            "manifest_sha256": _sha256(args.manifest),
            "bundle_sha256": _sha256(args.bundle),
            "preregistration_sha256": _sha256(args.preregistration),
            "evaluator_sha256": _sha256(Path(__file__)),
            "core_sha256": _sha256(core_path),
            "conditioning_key": conditioning_key,
            "online_policy": policy,
            "candidate_before_future_score": True,
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        config = PrefixAssimilationConfig(
            anchor_frames=int(policy["anchor_frames"]),
            ridge_fraction=float(policy["ridge_fraction"]),
            trust_ratio=float(policy["trust_ratio"]),
            minimum_observed_gain=float(policy["minimum_observed_gain"]),
            minimum_information_ratio=float(policy["minimum_information_ratio"]),
            minimum_observed_frames=(
                None
                if policy.get("minimum_observed_frames") is None
                else int(policy["minimum_observed_frames"])
            ),
        )
        common_tail_start = int(policy["common_tail_start"])
        rows = []
        for index, row in enumerate(manifest["records"]):
            start = int(row["window_start"])
            window_time_s = global_time_s[start : start + int(manifest["k_frames"])]
            observed_count = _observed_count(args.arm, window_time_s, row, policy)
            # This read is deliberately prefix-only.  The HDF5 handle is closed
            # before parent inference, assimilation, and candidate sealing.
            with h5py.File(args.cache, "r", swmr=True) as cache:
                base = torch.from_numpy(
                    cache["base_seq"][index : index + 1].astype(np.float32)
                )[:, :, None].to(device)
                conditioning = torch.from_numpy(
                    cache[conditioning_key][index : index + 1].astype(np.float32)
                ).to(device)
                observed_true = torch.from_numpy(
                    cache["target"][index : index + 1, :observed_count].astype(np.float32)
                )[:, :, None].to(device)
            torch.cuda.synchronize(device)
            parent_started = time.perf_counter()
            with torch.inference_mode():
                parent = model.forward_anchored(
                    base,
                    conditioning,
                    initial_state=observed_true[:, :8, 0],
                )
            torch.cuda.synchronize(device)
            parent_elapsed = time.perf_counter() - parent_started
            modes = bundle["families"][row["family"]]["modes"].float().to(device)
            audit = CausalPrefixAccessAudit(
                total_frames=int(parent.shape[1]), observed_count=observed_count
            )
            torch.cuda.synchronize(device)
            adaptation_started = time.perf_counter()
            candidate, adaptation = assimilate_prefix_pod(
                parent,
                modes,
                observed_true,
                config=config,
                access_audit=audit,
            )
            torch.cuda.synchronize(device)
            adaptation_elapsed = time.perf_counter() - adaptation_started
            materialize_started = time.perf_counter()
            materialized = candidate.detach().cpu().contiguous()
            materialization_elapsed = time.perf_counter() - materialize_started
            candidate_path = candidates_dir / f"{index:03d}_{row['sample_id']}.pt"
            torch.save(
                {
                    "candidate": materialized,
                    "sample_id": row["sample_id"],
                    "family": row["family"],
                    "observed_count": observed_count,
                    "observed_prefix_sha256": _tensor_sha256(observed_true),
                    "adaptation": adaptation,
                },
                candidate_path,
            )
            candidate_sha256 = _sha256(candidate_path)
            # Future truth is opened only after the candidate file exists and
            # has a stable digest.  It is used solely for post-seal scoring.
            with h5py.File(args.cache, "r", swmr=True) as cache:
                future_true = torch.from_numpy(
                    cache["target"][index : index + 1, observed_count:].astype(np.float32)
                )[:, :, None]
                common_tail_true = torch.from_numpy(
                    cache["target"][index : index + 1, common_tail_start:].astype(np.float32)
                )[:, :, None]
            sealed = torch.load(
                candidate_path, map_location="cpu", weights_only=False
            )["candidate"].float()
            parent_cpu = parent.detach().cpu().float()
            parent_future = parent_cpu[:, observed_count:]
            adapted_future = sealed[:, observed_count:]
            parent_relative = _relative_l2(parent_future, future_true)
            adapted_relative = _relative_l2(adapted_future, future_true)
            parent_common = _relative_l2(
                parent_cpu[:, common_tail_start:], common_tail_true
            )
            adapted_common = _relative_l2(
                sealed[:, common_tail_start:], common_tail_true
            )
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "family": row["family"],
                    "observed_count": observed_count,
                    "source_f0_hz": float(row["source_f0_hz"]),
                    "parent_relative_l2": parent_relative,
                    "adapted_relative_l2": adapted_relative,
                    "relative_gain": (parent_relative - adapted_relative)
                    / max(parent_relative, 1.0e-16),
                    "parent_common_tail_relative_l2": parent_common,
                    "adapted_common_tail_relative_l2": adapted_common,
                    "parent_temporal": _thirds_relative_l2(parent_future, future_true),
                    "adapted_temporal": _thirds_relative_l2(adapted_future, future_true),
                    "parent_frequency": _frequency_relative_l2(parent_future, future_true),
                    "adapted_frequency": _frequency_relative_l2(adapted_future, future_true),
                    "candidate_sha256": candidate_sha256,
                    "adaptation": adaptation,
                    "runtime_s": {
                        "parent": parent_elapsed,
                        "adaptation": adaptation_elapsed,
                        "materialization": materialization_elapsed,
                        "end_to_end": parent_elapsed
                        + adaptation_elapsed
                        + materialization_elapsed,
                    },
                }
            )
            del base, conditioning, observed_true, parent, modes, candidate
        parent_values = np.asarray(
            [row["parent_relative_l2"] for row in rows], dtype=np.float64
        )
        adapted_values = np.asarray(
            [row["adapted_relative_l2"] for row in rows], dtype=np.float64
        )
        parent_common_values = np.asarray(
            [row["parent_common_tail_relative_l2"] for row in rows], dtype=np.float64
        )
        adapted_common_values = np.asarray(
            [row["adapted_common_tail_relative_l2"] for row in rows], dtype=np.float64
        )
        runtime_values = [row["runtime_s"]["end_to_end"] for row in rows]
        summary = {
            "schema": "b2_v10_online_prefix_assimilation_report_v1",
            "arm": args.arm,
            "record_count": len(rows),
            "parent_aggregate": float(parent_values.mean()),
            "adapted_aggregate": float(adapted_values.mean()),
            "relative_gain": float(
                (parent_values.mean() - adapted_values.mean())
                / max(parent_values.mean(), 1.0e-16)
            ),
            "nonworse": int((adapted_values <= parent_values).sum()),
            "accepted_online": int(sum(row["adaptation"]["accepted"] for row in rows)),
            "common_tail_start": common_tail_start,
            "parent_common_tail_aggregate": float(parent_common_values.mean()),
            "adapted_common_tail_aggregate": float(adapted_common_values.mean()),
            "observed_count": {
                "minimum": int(min(row["observed_count"] for row in rows)),
                "mean": float(np.mean([row["observed_count"] for row in rows])),
                "maximum": int(max(row["observed_count"] for row in rows)),
            },
            "per_family": {
                family: {
                    "parent": float(
                        np.mean(
                            [
                                row["parent_relative_l2"]
                                for row in rows
                                if row["family"] == family
                            ]
                        )
                    ),
                    "adapted": float(
                        np.mean(
                            [
                                row["adapted_relative_l2"]
                                for row in rows
                                if row["family"] == family
                            ]
                        )
                    ),
                }
                for family in FAMILIES
            },
            "temporal": {
                name: {
                    "parent": float(np.mean([row["parent_temporal"][band] for row in rows])),
                    "adapted": float(np.mean([row["adapted_temporal"][band] for row in rows])),
                }
                for band, name in enumerate(("early", "middle", "late"))
            },
            "frequency": {
                name: {
                    "parent": float(np.mean([row["parent_frequency"][band] for row in rows])),
                    "adapted": float(np.mean([row["adapted_frequency"][band] for row in rows])),
                }
                for band, name in enumerate(("low", "middle", "high"))
            },
            "runtime_s": {
                "mean_end_to_end": float(np.mean(runtime_values)),
                "p95_end_to_end": _nearest_rank_percentile(runtime_values, 0.95),
            },
            "records": rows,
            "candidate_before_future_score": True,
            "future_truth_used_online": False,
            "validation_opened": False,
            "test_id_opened": False,
        }
        summary["passed"] = summary["adapted_aggregate"] < summary["parent_aggregate"]
        _atomic_json(summary, args.output_dir / "summary.json")
        _atomic_json(
            {
                "status": "passed" if summary["passed"] else "rejected",
                "arm": args.arm,
                "summary": str(args.output_dir / "summary.json"),
            },
            terminal,
        )
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "records"},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        import traceback

        _atomic_json(
            {
                "status": "failed",
                "arm": args.arm,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
            terminal,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
