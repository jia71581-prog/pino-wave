#!/usr/bin/env python3
"""Freeze PA-CORA P1 training bindings after caches and V9 terminal exist."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FIT_MANIFESTS = (
    Path("results/b2_v9_parent_fit_manifest_240rec_20260901.json"),
    Path("results/pa_cora_p1_fit_slot1.json"),
    Path("results/pa_cora_p1_fit_slot2.json"),
    Path("results/pa_cora_p1_fit_slot3.json"),
)
FIT_CACHES = (
    Path("results/b2_v9_parent_fit_cache_240rec_20260901.h5"),
    Path("results/pa_cora_p1_fit_slot1.h5"),
    Path("results/pa_cora_p1_fit_slot2.h5"),
    Path("results/pa_cora_p1_fit_slot3.h5"),
)
HOLDOUT_MANIFEST = Path("results/b2_v9_parent_holdout_manifest_24rec_20260901.json")
HOLDOUT_CACHE = Path("results/b2_v9_parent_holdout_cache_24rec_20260901.h5")
TRAINER = Path("scripts/train_pa_cora_p1_multiwindow.py")
RUNNER = Path("scripts/run_pa_cora_p1_four_lane.sh")


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
    parser.add_argument("--v9-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {args.output}")
    selection = json.loads(args.v9_selection.read_text())
    if selection.get("status") != "passed":
        raise RuntimeError("V9 parent selection did not pass")
    if selection["selected"]["variant"] != "spectral":
        raise RuntimeError("P1 was designed against the V9 spectral winner")
    for path in (*FIT_MANIFESTS, *FIT_CACHES, HOLDOUT_MANIFEST, HOLDOUT_CACHE, TRAINER, RUNNER):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifests = [json.loads(path.read_text()) for path in FIT_MANIFESTS]
    holdout = json.loads(HOLDOUT_MANIFEST.read_text())
    fit_groups = {row["group_id"] for manifest in manifests for row in manifest["records"]}
    holdout_groups = {row["group_id"] for row in holdout["records"]}
    if fit_groups & holdout_groups:
        raise RuntimeError("fit/holdout group leakage")
    payload = {
        "schema": "pa_cora_p1_multiwindow_training_preregistration_v1",
        "status": "frozen_before_launch",
        "method": "PA-CORA",
        "stage": "P1_multiwindow_only",
        "hypothesis": "Four deterministic train-only temporal windows per source yield strict same-protocol aggregate improvement over V9 spectral training at matched record exposure.",
        "conceptual_variable": "temporal window coverage only",
        "v9_selection": str(args.v9_selection),
        "v9_selection_sha256": _sha256(args.v9_selection),
        "v9_selected_variant": "spectral",
        "hyperparameters": {
            "architecture": "V9 FrameConditionedPropagator",
            "conditioning_channels": 7,
            "width": 64,
            "spectral_rank": 32,
            "modes": 24,
            "depth": 4,
            "ic_frames": 8,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 1e-6,
            "gradient_clip": 1.0,
            "schedule": "cosine to 1e-5",
            "loss": "relative_l2 + 0.1 * temporal_spectral",
            "epochs": 23,
            "micro_records": 4,
            "optimizer_steps_per_epoch": 240,
            "total_optimizer_steps": 5520,
            "record_exposures": 22080,
            "v9_record_exposures": 21600,
            "seeds": [372, 733, 1049, 1403]
        },
        "accuracy_gate": "mean best holdout aggregate of seeds 372 and 733 must be strictly below the corresponding V9 spectral two-seed mean; no minimum percentage",
        "report_only": [
            "auxiliary seeds 1049 and 1403",
            "per-family",
            "early/middle/late",
            "low/middle/high temporal frequency",
            "nonworse",
            "maximum",
            "runtime"
        ],
        "bindings": {
            "trainer_sha256": _sha256(TRAINER),
            "runner_sha256": _sha256(RUNNER),
            "fit_manifest_sha256": {str(path): _sha256(path) for path in FIT_MANIFESTS},
            "fit_cache_sha256": {str(path): _sha256(path) for path in FIT_CACHES},
            "holdout_manifest_sha256": _sha256(HOLDOUT_MANIFEST),
            "holdout_cache_sha256": _sha256(HOLDOUT_CACHE)
        },
        "data_access": {
            "split": "train",
            "fit_source_groups": 240,
            "windows_per_source": 4,
            "fit_window_records": 960,
            "holdout_records": 24,
            "fit_holdout_group_overlap": 0,
            "future_truth_used_for_window_selection": False,
            "validation_opened": False,
            "test_id_opened": False
        },
        "failure_signal": "two-seed mean is not strictly improved, nonfinite loss, OOM, or epoch-1 wall time above 900 seconds",
        "rollback": "additive PA-CORA P1 artifacts only; V9 and all protected checkpoints remain immutable",
        "validation_opened": False,
        "test_id_opened": False
    }
    _atomic_json(payload, args.output)
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
