#!/usr/bin/env python3
"""Leakage-safe train-only audit of the pressure free-surface projection.

The candidate predictions and their hashes are created before future targets are
read.  The projection is analytic and has no fitted or selected hyperparameter.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.boundary_consistency import (
    free_surface_violation,
    project_pressure_free_surface,
)
from scripts.b2_v5_components import per_record_relative_l2, temporal_band_relative_l2
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


FAMILIES = ("uniform", "layered", "marmousi")


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


def _masked_relative_per_record(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return per_record_relative_l2(
        prediction[..., mask].double(), target[..., mask].double()
    )


def _spatial_gradient_relative_per_record(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction_dx, target_dx = torch.diff(prediction, dim=-1), torch.diff(target, dim=-1)
    prediction_dz, target_dz = torch.diff(prediction, dim=-2), torch.diff(target, dim=-2)
    numerator = (prediction_dx.double() - target_dx.double()).flatten(1).square().sum(1)
    numerator += (prediction_dz.double() - target_dz.double()).flatten(1).square().sum(1)
    denominator = target_dx.double().flatten(1).square().sum(1)
    denominator += target_dz.double().flatten(1).square().sum(1)
    return torch.sqrt(numerator / denominator.clamp_min(1.0e-16))


def _summary(rows: list[dict], candidate: str) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty boundary audit")
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row[candidate]["aggregate"])
    return {
        "aggregate": float(np.mean([row[candidate]["aggregate"] for row in rows])),
        "maximum": float(np.max([row[candidate]["aggregate"] for row in rows])),
        "per_family": {
            family: float(np.mean(by_family[family])) for family in FAMILIES
        },
        "regions": {
            region: float(np.mean([row[candidate]["regions"][region] for row in rows]))
            for region in ("boundary20", "top20", "interior")
        },
        "frequency": {
            band: float(np.mean([row[candidate]["frequency"][band] for row in rows]))
            for band in ("low", "middle", "high")
        },
        "spatial_gradient": float(
            np.mean([row[candidate]["spatial_gradient"] for row in rows])
        ),
        "free_surface": {
            "maximum_absolute": float(
                np.max([row[candidate]["free_surface"]["maximum_absolute"] for row in rows])
            ),
            "mean_rms": float(
                np.mean([row[candidate]["free_surface"]["rms"] for row in rows])
            ),
        },
    }


def _load_model(checkpoint: dict, device: torch.device) -> FrameConditionedPropagator:
    state = checkpoint["model_state"]
    cond_channels = int(state["cond_projection.weight"].shape[1])
    width = int(state["cond_projection.weight"].shape[0])
    state_channels = int(state["encoder.0.weight"].shape[1]) - cond_channels
    if (state_channels, cond_channels, width) != (8, 7, 64):
        raise RuntimeError(
            "boundary audit is frozen for the current P1b architecture; "
            f"received {(state_channels, cond_channels, width)}"
        )
    model = FrameConditionedPropagator(
        state_channels=state_channels,
        cond_channels=cond_channels,
        width=width,
        spectral_rank=32,
        modes=24,
        depth=4,
        gate_init=1.0,
        activation_checkpointing=False,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.manifest) != 3 or len(args.cache) != 3:
        raise ValueError("the P1b boundary audit requires three aligned train-only phases")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    manifests = [json.loads(path.read_text()) for path in args.manifest]
    for path, manifest in zip(args.manifest, manifests):
        if (
            manifest.get("split") != "train"
            or manifest.get("validation_opened")
            or manifest.get("test_id_opened")
        ):
            raise RuntimeError(f"non-train or unsealed manifest refused: {path}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = checkpoint.get("identity", {})
    if identity.get("validation_opened") or identity.get("test_id_opened"):
        raise RuntimeError("checkpoint identity reports opened validation/test data")
    device = torch.device(args.device)
    model = _load_model(checkpoint, device)

    raw_digest = hashlib.sha256()
    projected_digest = hashlib.sha256()
    rows = []
    target_top_max = 0.0
    interior_prediction_delta_max = 0.0
    for slot, (cache_path, manifest) in enumerate(zip(args.cache, manifests)):
        with h5py.File(cache_path, "r", swmr=True) as cache:
            if cache.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
                raise RuntimeError(f"unexpected cache schema: {cache_path}")
            if cache.attrs.get("data_split", "") != "train":
                raise RuntimeError(f"non-train cache refused: {cache_path}")
            if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError(f"cache/manifest drift: {cache_path}")
            for index, metadata in enumerate(manifest["records"]):
                # Only deployment inputs and the explicitly visible target prefix
                # are read before the candidate predictions are frozen and hashed.
                anchor = torch.from_numpy(
                    cache["base_seq"][index : index + 1].astype(np.float32)
                )[:, :, None].to(device)
                conditioning = torch.from_numpy(
                    cache["cond"][index : index + 1].astype(np.float32)
                ).to(device)
                visible = torch.from_numpy(
                    cache["target"][index : index + 1, :8].astype(np.float32)
                ).to(device)
                raw = model.forward_anchored(
                    anchor, conditioning, initial_state=visible
                )[:, 8:]
                projected = project_pressure_free_surface(raw)
                raw_cpu = raw.float().cpu().contiguous()
                projected_cpu = projected.float().cpu().contiguous()
                raw_digest.update(raw_cpu.numpy().tobytes())
                projected_digest.update(projected_cpu.numpy().tobytes())

                # Future truth is opened only after both candidates are immutable.
                target = torch.from_numpy(
                    cache["target"][index : index + 1, 8:].astype(np.float32)
                )[:, :, None]
                target_top_max = max(target_top_max, float(target[..., 0, :].abs().max()))
                height, width_px = target.shape[-2:]
                boundary20 = torch.zeros(height, width_px, dtype=torch.bool)
                boundary20[:20] = True
                boundary20[-20:] = True
                boundary20[:, :20] = True
                boundary20[:, -20:] = True
                top20 = torch.zeros_like(boundary20)
                top20[:20] = True
                interior = ~boundary20
                interior_prediction_delta_max = max(
                    interior_prediction_delta_max,
                    float((raw_cpu[..., interior] - projected_cpu[..., interior]).abs().max()),
                )

                row = {
                    "slot": slot,
                    "sample_id": str(
                        metadata.get("source_sample_id", metadata["sample_id"])
                    ),
                    "family": str(metadata["family"]),
                }
                for name, candidate in (("raw", raw_cpu), ("projected", projected_cpu)):
                    frequency = temporal_band_relative_l2(
                        candidate, target, energy_floor_fraction=0.005
                    )[0]
                    violation = free_surface_violation(candidate)
                    row[name] = {
                        "aggregate": float(per_record_relative_l2(candidate.double(), target.double())[0]),
                        "regions": {
                            "boundary20": float(
                                _masked_relative_per_record(candidate, target, boundary20)[0]
                            ),
                            "top20": float(
                                _masked_relative_per_record(candidate, target, top20)[0]
                            ),
                            "interior": float(
                                _masked_relative_per_record(candidate, target, interior)[0]
                            ),
                        },
                        "frequency": {
                            band: float(frequency[band_index])
                            for band_index, band in enumerate(("low", "middle", "high"))
                        },
                        "spatial_gradient": float(
                            _spatial_gradient_relative_per_record(candidate, target)[0]
                        ),
                        "free_surface": {
                            key: float(value) for key, value in violation.items()
                        },
                    }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "boundary_audit_progress",
                            "slot": slot,
                            "record": index + 1,
                            "of": len(manifest["records"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    raw_summary = _summary(rows, "raw")
    projected_summary = _summary(rows, "projected")
    raw_values = np.asarray([row["raw"]["aggregate"] for row in rows])
    projected_values = np.asarray([row["projected"]["aggregate"] for row in rows])
    tolerance = 1.0e-12
    payload = {
        "schema": "pa_cora_free_surface_projection_audit_v1",
        "scope": "train_only_group_disjoint_three_phase_holdout",
        "method": "analytic_hard_projection_p_z0_equals_zero",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "script_sha256": _sha256(Path(__file__)),
        "bindings": {
            "manifests": {str(path): _sha256(path) for path in args.manifest},
            "caches": {str(path): _sha256(path) for path in args.cache},
        },
        "candidate_sha256": {
            "raw": raw_digest.hexdigest(),
            "projected": projected_digest.hexdigest(),
        },
        "candidate_frozen_before_future_truth_read": True,
        "validation_opened": False,
        "test_id_opened": False,
        "target_free_surface_maximum_absolute": target_top_max,
        "interior_prediction_delta_maximum_absolute": interior_prediction_delta_max,
        "raw": raw_summary,
        "projected": projected_summary,
        "comparison": {
            "absolute_aggregate_change": projected_summary["aggregate"]
            - raw_summary["aggregate"],
            "relative_aggregate_gain": (
                raw_summary["aggregate"] - projected_summary["aggregate"]
            )
            / max(raw_summary["aggregate"], 1.0e-16),
            "records_improved": int(np.sum(projected_values < raw_values - tolerance)),
            "records_equal_within_tolerance": int(
                np.sum(np.abs(projected_values - raw_values) <= tolerance)
            ),
            "records_worse": int(np.sum(projected_values > raw_values + tolerance)),
            "strict_aggregate_improvement": bool(
                projected_summary["aggregate"] < raw_summary["aggregate"]
            ),
            "nonworsening_expected_from_exact_zero_target_boundary": True,
        },
        "rows": rows,
    }
    _atomic_json(payload, args.output)
    print(json.dumps({"event": "boundary_audit_complete", **payload["comparison"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
