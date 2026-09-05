#!/usr/bin/env python3
"""Analyze final V9 residual structure on train-only multi-phase holdouts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


FAMILIES = ("uniform", "layered", "marmousi")
TEMPORAL_THIRDS = ((0, 19), (19, 38), (38, 56))
FFT_BANDS = ((0, 5), (5, 12), (12, 29))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _relative(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (prediction.double() - target.double()).norm()
        / target.double().norm().clamp_min(1.0e-16)
    )


def _f0_bin(value: float) -> str:
    if value < 15.0:
        return "low_lt15"
    if value < 22.0:
        return "middle_15to22"
    return "high_gte22"


def _load_parent(checkpoint: dict, device: torch.device) -> tuple[torch.nn.Module, str]:
    identity = checkpoint["identity"]
    key = str(identity["conditioning_key"])
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
    return model, key


def _masked_relative(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    selected_prediction = prediction[..., mask]
    selected_target = target[..., mask]
    return _relative(selected_prediction, selected_target)


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "aggregate": float(np.mean([row["aggregate"] for row in rows])),
        "maximum": float(np.max([row["aggregate"] for row in rows])),
        "temporal": {
            name: float(np.mean([row["temporal"][index] for row in rows]))
            for index, name in enumerate(("early", "middle", "late"))
        },
        "frequency": {
            name: float(np.mean([row["frequency"][index] for row in rows]))
            for index, name in enumerate(("low", "middle", "high"))
        },
        "regions": {
            name: float(np.mean([row["regions"][name] for row in rows]))
            for name in ("boundary20", "interior", "top20", "interface_top10", "noninterface")
        },
        "spatial_gradient": float(np.mean([row["spatial_gradient"] for row in rows])),
        "best_shift_histogram": {
            str(shift): int(sum(row["best_temporal_shift_frames"] == shift for row in rows))
            for shift in (-2, -1, 0, 1, 2)
        },
        "mean_shift_oracle_gain": float(
            np.mean([row["temporal_shift_oracle_gain"] for row in rows])
        ),
    }


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.manifest) != 4 or len(args.cache) != 4:
        raise ValueError("four aligned holdout phases are required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    manifests = [json.loads(path.read_text()) for path in args.manifest]
    for manifest in manifests:
        if manifest.get("split") != "train" or manifest.get("validation_opened") or manifest.get("test_id_opened"):
            raise RuntimeError("diagnostic is restricted to sealed train manifests")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model, conditioning_key = _load_parent(checkpoint, device)
    rows = []
    for slot, (cache_path, manifest) in enumerate(zip(args.cache, manifests)):
        with h5py.File(cache_path, "r", swmr=True) as cache:
            if cache.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
                raise RuntimeError(f"unexpected cache schema: {cache_path}")
            if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError(f"cache/manifest drift: {cache_path}")
            for index, metadata in enumerate(manifest["records"]):
                anchor = torch.from_numpy(
                    cache["base_seq"][index : index + 1].astype(np.float32)
                )[:, :, None].to(device)
                target = torch.from_numpy(
                    cache["target"][index : index + 1].astype(np.float32)
                )[:, :, None].to(device)
                conditioning = torch.from_numpy(
                    cache[conditioning_key][index : index + 1].astype(np.float32)
                ).to(device)
                prediction = model.forward_anchored(
                    anchor,
                    conditioning,
                    initial_state=target[:, :8, 0],
                )
                future = prediction[:, 8:]
                truth = target[:, 8:]
                height, width = truth.shape[-2:]
                boundary = torch.zeros(height, width, dtype=torch.bool, device=device)
                boundary[:20] = True
                boundary[-20:] = True
                boundary[:, :20] = True
                boundary[:, -20:] = True
                top = torch.zeros_like(boundary)
                top[:20] = True
                gradient_map = torch.sqrt(
                    conditioning[0, 1].square() + conditioning[0, 2].square()
                )
                threshold = torch.quantile(gradient_map.flatten(), 0.90)
                interface = gradient_map >= threshold
                temporal = [
                    _relative(future[:, start:stop], truth[:, start:stop])
                    for start, stop in TEMPORAL_THIRDS
                ]
                future_fft = torch.fft.rfft(future.float(), dim=1)
                truth_fft = torch.fft.rfft(truth.float(), dim=1)
                total_frequency_energy = truth_fft.abs().square().sum().clamp_min(1.0e-16)
                frequency = []
                for start, stop in FFT_BANDS:
                    band_truth = truth_fft[:, start:stop]
                    band_error = future_fft[:, start:stop] - band_truth
                    numerator = band_error.abs().square().sum().sqrt()
                    denominator = torch.maximum(
                        band_truth.abs().square().sum(),
                        0.005 * total_frequency_energy,
                    ).sqrt()
                    frequency.append(float(numerator / denominator))
                future_dx, truth_dx = torch.diff(future, dim=-1), torch.diff(truth, dim=-1)
                future_dz, truth_dz = torch.diff(future, dim=-2), torch.diff(truth, dim=-2)
                spatial_gradient = float(
                    torch.sqrt(
                        ((future_dx - truth_dx).double().square().sum() + (future_dz - truth_dz).double().square().sum())
                        / (truth_dx.double().square().sum() + truth_dz.double().square().sum()).clamp_min(1.0e-16)
                    )
                )
                base_relative = _relative(future, truth)
                shift_values = {}
                for shift in (-2, -1, 0, 1, 2):
                    if shift < 0:
                        shifted_prediction, shifted_truth = future[:, :shift], truth[:, -shift:]
                    elif shift > 0:
                        shifted_prediction, shifted_truth = future[:, shift:], truth[:, :-shift]
                    else:
                        shifted_prediction, shifted_truth = future, truth
                    shift_values[shift] = _relative(shifted_prediction, shifted_truth)
                best_shift = min(shift_values, key=shift_values.get)
                rows.append(
                    {
                        "slot": slot,
                        "sample_id": str(metadata.get("source_sample_id", metadata["sample_id"])),
                        "family": str(metadata["family"]),
                        "source_f0_hz": float(metadata["source_f0_hz"]),
                        "f0_bin": _f0_bin(float(metadata["source_f0_hz"])),
                        "aggregate": base_relative,
                        "temporal": temporal,
                        "frequency": frequency,
                        "regions": {
                            "boundary20": _masked_relative(future, truth, boundary),
                            "interior": _masked_relative(future, truth, ~boundary),
                            "top20": _masked_relative(future, truth, top),
                            "interface_top10": _masked_relative(future, truth, interface),
                            "noninterface": _masked_relative(future, truth, ~interface),
                        },
                        "spatial_gradient": spatial_gradient,
                        "best_temporal_shift_frames": int(best_shift),
                        "temporal_shift_oracle_gain": float(
                            (base_relative - shift_values[best_shift])
                            / max(base_relative, 1.0e-16)
                        ),
                    }
                )
                print(json.dumps({"event": "residual_progress", "slot": slot, "record": index, "of": len(manifest["records"])}), flush=True)
    payload = {
        "schema": "v9_multiphase_residual_structure_v1",
        "scope": "train_only_group_disjoint_diagnostic",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "script_sha256": _sha256(Path(__file__)),
        "bindings": {
            "manifests": {str(path): _sha256(path) for path in args.manifest},
            "caches": {str(path): _sha256(path) for path in args.cache},
        },
        "by_slot": {
            str(slot): _summarize([row for row in rows if row["slot"] == slot])
            for slot in range(4)
        },
        "by_slot_family": {
            str(slot): {
                family: _summarize(
                    [row for row in rows if row["slot"] == slot and row["family"] == family]
                )
                for family in FAMILIES
            }
            for slot in range(4)
        },
        "by_slot_f0": {
            str(slot): {
                name: _summarize(
                    [row for row in rows if row["slot"] == slot and row["f0_bin"] == name]
                )
                for name in ("low_lt15", "middle_15to22", "high_gte22")
            }
            for slot in range(4)
        },
        "records": rows,
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(payload, args.output)
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
