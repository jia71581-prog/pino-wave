#!/usr/bin/env python3
"""Freeze stage-specific B2-v10 preregistrations after required inputs exist."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "saved_time_phase_operator_v4/instance_adaptation/b2_v10_prefix_assimilation.py"
PRETRAIN = ROOT / "scripts/pretrain_b2_v10_residual_pod.py"
EVALUATOR = ROOT / "scripts/evaluate_b2_v10_prefix_assimilation.py"
RUNNER = ROOT / "scripts/run_b2_v10_four_arm.sh"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("offline", "calibration", "confirmation"), required=True)
    parser.add_argument("--parent-selection", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("results/b2_v10_prefix_assimilation_design_20260901.json"),
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--arm-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {args.output}")
    selection = json.loads(args.parent_selection.read_text())
    if selection.get("status") != "passed" or not selection.get("selected"):
        raise RuntimeError("V9 parent selection did not pass")
    selected = selection["selected"]
    checkpoint = Path(selected["checkpoint"])
    if not checkpoint.is_file() or _sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("selected V9 checkpoint drift")
    design = json.loads(args.design.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("split") != "train":
        raise RuntimeError("B2-v10 preregistration accepts train manifests only")
    if manifest.get("validation_opened") or manifest.get("test_id_opened"):
        raise RuntimeError("sealed split flag is open")
    common = {
        "status": "frozen_before_launch",
        "candidate": design["candidate"],
        "stage": args.stage,
        "hypothesis": design["hypothesis"],
        "parent": {
            "selection": str(args.parent_selection),
            "selection_sha256": _sha256(args.parent_selection),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "variant": selected["variant"],
            "seed": selected["seed"],
        },
        "design": str(args.design),
        "design_sha256": _sha256(args.design),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "cache": str(args.cache),
        "cache_sha256": _sha256(args.cache),
        "binder_sha256": _sha256(Path(__file__)),
        "rollback": design["rollback"],
        "validation_opened": False,
        "test_id_opened": False,
    }
    if args.stage == "offline":
        payload = {
            "schema": "b2_v10_offline_pod_preregistration_v1",
            **common,
            "bindings": {
                "pretrain_sha256": _sha256(PRETRAIN),
                "core_sha256": _sha256(CORE),
                "checkpoint_sha256": _sha256(checkpoint),
                "fit_cache_sha256": _sha256(args.cache),
                "fit_manifest_sha256": _sha256(args.manifest),
            },
            "gates": design["offline_policy"]
            | {"minimum_mean_oracle_gain_per_family": 0.0},
            "truth_scope": "offline_train_only",
        }
    else:
        if args.bundle is None or not args.bundle.is_file():
            raise ValueError("calibration/confirmation preregistration requires --bundle")
        bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
        if bundle.get("parent_checkpoint_sha256") != _sha256(checkpoint):
            raise RuntimeError("POD bundle was not fit for the selected parent")
        selected_arm = None
        arm_selection_sha256 = None
        if args.stage == "confirmation":
            if args.arm_selection is None or not args.arm_selection.is_file():
                raise ValueError("confirmation preregistration requires --arm-selection")
            arm_selection = json.loads(args.arm_selection.read_text())
            if arm_selection.get("status") != "passed":
                raise RuntimeError("calibration produced no passing arm")
            selected_arm = arm_selection["selected"]["arm"]
            arm_selection_sha256 = _sha256(args.arm_selection)
        payload = {
            "schema": f"b2_v10_online_{args.stage}_preregistration_v1",
            **common,
            "bundle": str(args.bundle),
            "bundle_sha256": _sha256(args.bundle),
            "selected_arm": selected_arm,
            "arm_selection": None if args.arm_selection is None else str(args.arm_selection),
            "arm_selection_sha256": arm_selection_sha256,
            "bindings": {
                "evaluator_sha256": _sha256(EVALUATOR),
                "core_sha256": _sha256(CORE),
                "runner_sha256": _sha256(RUNNER),
                "checkpoint_sha256": _sha256(checkpoint),
                "evaluation_cache_sha256": _sha256(args.cache),
                "evaluation_manifest_sha256": _sha256(args.manifest),
                "bundle_sha256": _sha256(args.bundle),
            },
            "online_policy": design["online_policy"],
            "accuracy_and_selection": design["accuracy_and_selection"],
            "future_truth_access": "only after serialized candidate SHA256 exists",
        }
    _atomic_json(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "stage": args.stage,
                "checkpoint": str(checkpoint),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
