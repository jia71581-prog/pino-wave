#!/usr/bin/env python3
"""Train-only truth-leaking capacity test for a fixed DG skeleton projection."""
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

from saved_time_phase_operator_v4.boundary_consistency import project_pressure_free_surface
from saved_time_phase_operator_v4.dg_skeleton_oracle import FixedSkeletonFluxProjector
from scripts.b2_v5_components import per_record_relative_l2, temporal_band_relative_l2
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


FAMILIES = ("uniform", "layered", "marmousi")
SLOT_WEIGHTS = (0.60, 0.25, 0.15)


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


def _summarize(rows: list[dict], key: str) -> dict:
    per_slot = {}
    for slot in range(3):
        selected = [row[key]["relative_l2"] for row in rows if row["slot"] == slot]
        per_slot[str(slot)] = float(np.mean(selected))
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row[key]["relative_l2"])
    return {
        "aggregate": float(np.mean([row[key]["relative_l2"] for row in rows])),
        "weighted_multiphase": float(
            sum(SLOT_WEIGHTS[slot] * per_slot[str(slot)] for slot in range(3))
        ),
        "per_slot": per_slot,
        "per_family": {
            family: float(np.mean(by_family[family])) for family in FAMILIES
        },
        "frequency": {
            band: float(np.mean([row[key]["frequency"][band] for row in rows]))
            for band in ("low", "middle", "high")
        },
    }


def _load_model(checkpoint: dict, device: torch.device) -> FrameConditionedPropagator:
    identity = checkpoint["identity"]
    if identity.get("method") != "Transfer DG" or identity.get("stage") != "pretrain_DG_flux_v1":
        raise RuntimeError("oracle parent is not the latest Transfer DG global checkpoint")
    model = FrameConditionedPropagator(
        state_channels=8,
        cond_channels=7,
        width=64,
        spectral_rank=32,
        modes=24,
        depth=4,
        gate_init=1.0,
        activation_checkpointing=False,
        hard_free_surface=True,
        dg_interface_rank=16,
        dg_cpml_margin=20,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-records", type=int, default=4)
    args = parser.parse_args()
    if len(args.manifest) != 3 or len(args.cache) != 3:
        raise ValueError("oracle requires three aligned train-only phases")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    if str(args.output) != prereg["output"] or args.micro_records != 4:
        raise RuntimeError("oracle runtime arguments drift from preregistration")
    checks = {Path(__file__): bindings["evaluator_sha256"], args.checkpoint: bindings["checkpoint_sha256"]}
    checks[ROOT / "saved_time_phase_operator_v4/dg_skeleton_oracle.py"] = bindings[
        "projector_sha256"
    ]
    checks.update({path: bindings["manifest_sha256"][str(path)] for path in args.manifest})
    checks.update({path: bindings["cache_sha256"][str(path)] for path in args.cache})
    for path, expected in checks.items():
        if _sha256(path) != expected:
            raise RuntimeError(f"binding drift: {path}")
    manifests = [json.loads(path.read_text()) for path in args.manifest]
    for manifest in manifests:
        if (
            manifest.get("split") != "train"
            or manifest.get("validation_opened")
            or manifest.get("test_id_opened")
        ):
            raise RuntimeError("oracle is restricted to sealed train manifests")

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _load_model(checkpoint, device)
    projector = FixedSkeletonFluxProjector(device=device, dtype=torch.float32)
    rows = []
    for slot, (cache_path, manifest) in enumerate(zip(args.cache, manifests)):
        with h5py.File(cache_path, "r", swmr=True) as cache:
            if cache.attrs.get("data_split", "") != "train":
                raise RuntimeError("oracle cache is not train")
            if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError("cache/manifest drift")
            for lo in range(0, len(manifest["records"]), args.micro_records):
                hi = min(lo + args.micro_records, len(manifest["records"]))
                anchor = torch.from_numpy(cache["base_seq"][lo:hi].astype(np.float32))[
                    :, :, None
                ].to(device)
                truth = torch.from_numpy(cache["target"][lo:hi].astype(np.float32))[
                    :, :, None
                ].to(device)
                conditioning = torch.from_numpy(cache["cond"][lo:hi].astype(np.float32)).to(device)
                prediction = model.forward_anchored(
                    anchor, conditioning, initial_state=truth[:, :8, 0]
                )[:, 8:, 0]
                target = truth[:, 8:, 0]
                flux_residual = projector.flux_vector(target - prediction)
                correction = projector.project_flux(flux_residual)
                candidate = project_pressure_free_surface(prediction + correction)
                residual_after = projector.flux_vector(target - candidate)
                parent_frequency = temporal_band_relative_l2(
                    prediction[:, :, None], target[:, :, None], energy_floor_fraction=0.005
                ).cpu()
                oracle_frequency = temporal_band_relative_l2(
                    candidate[:, :, None], target[:, :, None], energy_floor_fraction=0.005
                ).cpu()
                parent_relative = per_record_relative_l2(
                    prediction.double(), target.double()
                ).cpu()
                oracle_relative = per_record_relative_l2(candidate.double(), target.double()).cpu()
                correction_ratio = correction.double().flatten(1).norm(dim=1) / prediction.double().flatten(1).norm(dim=1).clamp_min(1.0e-16)
                flux_gain = 1.0 - residual_after.double().flatten(1).norm(dim=1) / flux_residual.double().flatten(1).norm(dim=1).clamp_min(1.0e-16)
                for offset, metadata in enumerate(manifest["records"][lo:hi]):
                    def metric(relative, frequency):
                        return {
                            "relative_l2": float(relative[offset]),
                            "frequency": {
                                name: float(frequency[offset, index])
                                for index, name in enumerate(("low", "middle", "high"))
                            },
                        }

                    rows.append(
                        {
                            "slot": slot,
                            "sample_id": str(metadata.get("source_sample_id", metadata["sample_id"])),
                            "family": str(metadata["family"]),
                            "parent": metric(parent_relative, parent_frequency),
                            "oracle": metric(oracle_relative, oracle_frequency),
                            "correction_ratio": float(correction_ratio[offset]),
                            "skeleton_flux_residual_gain": float(flux_gain[offset]),
                        }
                    )
                print(json.dumps({"event": "oracle_progress", "slot": slot, "records": hi}), flush=True)

    parent_summary = _summarize(rows, "parent")
    oracle_summary = _summarize(rows, "oracle")
    parent_values = np.asarray([row["parent"]["relative_l2"] for row in rows])
    oracle_values = np.asarray([row["oracle"]["relative_l2"] for row in rows])
    comparison = {
        "strict_aggregate_improvement": bool(oracle_summary["aggregate"] < parent_summary["aggregate"]),
        "relative_aggregate_gain": (parent_summary["aggregate"] - oracle_summary["aggregate"])
        / max(parent_summary["aggregate"], 1.0e-16),
        "strict_weighted_multiphase_improvement": bool(
            oracle_summary["weighted_multiphase"] < parent_summary["weighted_multiphase"]
        ),
        "relative_weighted_multiphase_gain": (
            parent_summary["weighted_multiphase"] - oracle_summary["weighted_multiphase"]
        )
        / max(parent_summary["weighted_multiphase"], 1.0e-16),
        "records_improved": int(np.sum(oracle_values < parent_values)),
        "records_worse": int(np.sum(oracle_values > parent_values)),
        "mean_correction_ratio": float(np.mean([row["correction_ratio"] for row in rows])),
        "mean_skeleton_flux_residual_gain": float(
            np.mean([row["skeleton_flux_residual_gain"] for row in rows])
        ),
    }
    payload = {
        "schema": "transfer_dg_fixed_skeleton_truth_flux_oracle_v1",
        "status": "capacity_present" if comparison["strict_aggregate_improvement"] else "capacity_absent",
        "claim_scope": "truth-leaking train-only oracle; rejection/capacity evidence only",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "projector": projector.report,
        "parent": parent_summary,
        "oracle": oracle_summary,
        "comparison": comparison,
        "rows": rows,
        "future_truth_used_to_construct_candidate": True,
        "deployable": False,
        "validation_opened": False,
        "test_id_opened": False,
        "bindings": {
            "evaluator_sha256": _sha256(Path(__file__)),
            "projector_sha256": _sha256(
                ROOT / "saved_time_phase_operator_v4/dg_skeleton_oracle.py"
            ),
            "manifests": {str(path): _sha256(path) for path in args.manifest},
            "caches": {str(path): _sha256(path) for path in args.cache},
        },
    }
    _atomic_json(payload, args.output)
    print(json.dumps({"event": "oracle_complete", **comparison}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
