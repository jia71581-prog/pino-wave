#!/usr/bin/env python3
"""Evaluate a current CPADC checkpoint on the frozen PI-DeepONet panel.

Unlike the legacy replay evaluator, this script runs the bound r5b parent and
the current online CPADC solve exactly once.  The normal 401-frame sealed
evaluation is retained in ``output_dir`` while the exact 32 PI-selected frames
are accumulated in memory for a same-record, same-frame comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts import run_causal_defect_adaptation as cpadc
from scripts.evaluate_cpadc_on_pi_panel import (
    FAMILIES,
    _add,
    _atomic_json,
    _comparison_evaluator,
    _relative,
    _sha256,
    load_pi_panel,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_sha256 = _sha256(args.basis_checkpoint)
    panel, pi_metadata = load_pi_panel(args.pi_workers, split=args.split)
    selected_sample_ids = tuple(str(value) for value in (args.sample_id or ()))
    if selected_sample_ids:
        if args.split != "validation":
            raise ValueError("sample-limited comparison smoke is validation-only")
        missing = sorted(set(selected_sample_ids) - set(panel))
        if missing:
            raise ValueError(f"sample-limited panel IDs are missing: {missing[:3]}")

    original_evaluator = cpadc.evaluate_after_adaptation
    cpadc.evaluate_after_adaptation = _comparison_evaluator(
        panel, original_evaluator
    )
    try:
        reports = cpadc.run(
            args.config,
            basis_checkpoint=args.basis_checkpoint,
            output_dir=args.output_dir,
            device_name=args.device,
            adaptation_device_name="cpu",
            sample_ids=selected_sample_ids or None,
            all_validation=args.split == "validation" and not selected_sample_ids,
            all_test_id=args.split == "test_id" and not selected_sample_ids,
            shard_index=args.shard_index,
            shard_count=args.num_shards,
            save_fields=False,
        )
    finally:
        cpadc.evaluate_after_adaptation = original_evaluator

    expected_count = (
        len(selected_sample_ids)
        if selected_sample_ids
        else len(panel) // int(args.num_shards)
    )
    if len(reports) != expected_count:
        raise RuntimeError(
            f"current CPADC panel coverage is {len(reports)}, expected {expected_count}"
        )

    adapted_total = (0.0, 0.0)
    parent_total = (0.0, 0.0)
    adapted_family = {family: (0.0, 0.0) for family in FAMILIES}
    parent_family = {family: (0.0, 0.0) for family in FAMILIES}
    measurements: list[dict[str, object]] = []
    parent_paths: set[str] = set()
    parent_hashes: set[str] = set()
    for report in reports:
        sample_id = str(report["sample_id"])
        family = str(report["medium_type"])
        matched = dict(report["pi_matched_panel"])
        adapted_terms = (
            float(matched["adapted_error_numerator_in_pi_normalization"]),
            float(matched["truth_denominator_in_pi_normalization"]),
        )
        parent_terms = (
            float(matched["parent_error_numerator_in_pi_normalization"]),
            float(matched["truth_denominator_in_pi_normalization"]),
        )
        adapted_total = _add(adapted_total, adapted_terms)
        parent_total = _add(parent_total, parent_terms)
        adapted_family[family] = _add(adapted_family[family], adapted_terms)
        parent_family[family] = _add(parent_family[family], parent_terms)
        adaptation = dict(report["adaptation"])
        basis = dict(adaptation["basis"])
        if basis.get("checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError(f"CPADC checkpoint binding changed for {sample_id}")
        parent_paths.add(str(basis["parent_checkpoint"]))
        parent_hashes.add(str(basis["parent_checkpoint_sha256"]))
        measurements.append(
            {
                "sample_id": sample_id,
                "source_index": int(panel[sample_id]["source_index"]),
                "family": family,
                **matched,
                "adaptation_accepted": bool(adaptation["accepted"]),
                "adaptation_elapsed_s": float(adaptation["adaptation_elapsed_s"]),
                "total_inference_elapsed_s_for_401_frames": float(
                    adaptation["total_inference_elapsed_s"]
                ),
                "full401_adapted_relative_l2": float(
                    report["future_fullfield_relative_l2"]
                ),
                "full401_parent_relative_l2": float(
                    report["parent_future_fullfield_relative_l2"]
                ),
            }
        )
    if len(parent_paths) != 1 or len(parent_hashes) != 1:
        raise RuntimeError("current CPADC workers observed inconsistent parent bindings")

    output = {
        "schema": "frozen_cpadc_on_pi_panel_worker_v1",
        "status": "complete",
        "role": "comparison_only",
        "split": args.split,
        "selection_scope": (
            "limited_validation_smoke_on_frozen_pi_panel"
            if selected_sample_ids
            else "same_records_and_32_exact_frames_as_frozen_pi_deeponet"
        ),
        "frames_per_record": 32,
        "cpadc_full_output_frames_per_record": 401,
        "global_record_count": len(panel),
        "record_count": len(measurements),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "aggregate_relative_l2": _relative(adapted_total),
        "parent_aggregate_relative_l2": _relative(parent_total),
        "error_terms": {
            "aggregate": list(adapted_total),
            "parent_aggregate": list(parent_total),
            "family": {
                family: list(adapted_family[family]) for family in FAMILIES
            },
            "parent_family": {
                family: list(parent_family[family]) for family in FAMILIES
            },
            "normalization": "pi_deeponet_pressure_normalization_per_record",
        },
        "measurements": measurements,
        "cpadc_checkpoint": {
            "path": str(args.basis_checkpoint.resolve()),
            "sha256": checkpoint_sha256,
        },
        "pi_panel": pi_metadata,
        "bindings": {
            "actual_parent_checkpoint": next(iter(parent_paths)),
            "actual_parent_checkpoint_sha256": next(iter(parent_hashes)),
            "config_path": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "comparison_script_sha256": _sha256(Path(__file__)),
            "prediction_reconstruction": (
                "live_bound_r5b_parent_plus_live_cpu_cpadc_instance_coefficients"
            ),
            "sealed_401_frame_output_dir": str(args.output_dir.resolve()),
        },
    }
    _atomic_json(output, args.output)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--basis-checkpoint", type=Path, required=True)
    parser.add_argument("--pi-workers", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=("validation", "test_id"), required=True)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = run(args)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
