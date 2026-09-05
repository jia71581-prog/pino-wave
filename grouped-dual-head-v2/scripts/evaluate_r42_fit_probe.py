#!/usr/bin/env python3
"""Evaluate an R42 checkpoint on the hardest train-only fit records."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--r42-script", type=Path, required=True)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    r40 = load_module("r40_training_probe", args.r40_script.resolve())
    r42 = load_module("r42_training_probe", args.r42_script.resolve())
    fit = r40.FrequencyCacheCollection(args.fit_cache, expected_subset="fit")
    try:
        curriculum_path = args.curriculum_manifest.resolve()
        _, _, base_errors = r42.validate_curriculum(curriculum_path, fit)
        checkpoint_path = args.checkpoint.resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "r42_recordwise_hard_curriculum_checkpoint_v1":
            raise RuntimeError("unexpected R42 checkpoint schema")
        if str(checkpoint.get("selection_sha256")) != str(fit.selection_sha256):
            raise RuntimeError("checkpoint/cache selection mismatch")
        config = checkpoint["model_config"]
        model = r40.FrequencyResidualFNO(
            width=int(config["width"]),
            modes=int(config["modes"]),
            blocks=int(config["blocks"]),
            correction_cap=float(config["correction_cap"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        positions = np.argsort(-base_errors)[: int(args.probe_count)].tolist()
        scales = sorted({0.0, 1.0, float(checkpoint["selected_correction_scale"])})
        evaluations = {}
        for scale in scales:
            evaluations[str(scale)] = r42.evaluate_fit_probe(
                r40,
                model,
                fit,
                positions,
                base_errors,
                correction_scale=scale,
                device=device,
                batch_size=int(args.batch_size),
                amp=bool(args.amp),
            )
        payload = {
            "schema": "r42_hard_fit_probe_audit_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_global_step": int(checkpoint["global_step"]),
            "checkpoint_selected_correction_scale": float(
                checkpoint["selected_correction_scale"]
            ),
            "probe_count": len(positions),
            "probe_positions": positions,
            "evaluations": evaluations,
            "data_boundary": {
                "fit_truth_used": True,
                "opened_development_holdout_used": False,
                "r29b_opened": False,
                "final_validation_opened": False,
                "test_id_opened": False,
                "paper_modified": False,
            },
        }
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output)
        for scale, evaluation in evaluations.items():
            aggregate = evaluation["aggregate"]
            print(
                json.dumps(
                    {
                        "event": "fit_probe",
                        "checkpoint_epoch": int(checkpoint["epoch"]),
                        "correction_scale": float(scale),
                        "candidate_mean": aggregate["candidate_mean"],
                        "candidate_max": aggregate["candidate_max"],
                        "parent_mean": aggregate["parent_mean"],
                        "parent_max": aggregate["parent_max"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        fit.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
