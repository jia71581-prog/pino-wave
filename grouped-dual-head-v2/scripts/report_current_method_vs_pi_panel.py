#!/usr/bin/env python3
"""Relabel a live r5b+CPADC exact-panel result without legacy R7 field names."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_report(
    comparison_path: Path,
    validation_terminal_path: Path,
) -> dict[str, object]:
    comparison = json.loads(comparison_path.read_text())
    validation = json.loads(validation_terminal_path.read_text())
    if comparison.get("schema") != "frozen_cpadc_vs_pi_deeponet_exact_panel_v1":
        raise ValueError("unexpected source comparison schema")
    if comparison.get("selection_scope") != (
        "same_480_records_same_32_exact_frames_same_energy_relative_l2"
    ):
        raise ValueError("source comparison is not the exact frozen PI panel")
    worker_paths = [Path(value) for value in comparison["bindings"]["worker_outputs"]]
    workers = [json.loads(path.read_text()) for path in worker_paths]
    if not all(
        worker.get("bindings", {}).get("prediction_reconstruction")
        == "live_bound_r5b_parent_plus_live_cpu_cpadc_instance_coefficients"
        for worker in workers
    ):
        raise ValueError("source workers are not live current r5b+CPADC evaluations")

    global_raw = dict(comparison["global_energy_relative_l2"])
    family_raw = dict(comparison["family_energy_relative_l2"])
    record_raw = dict(comparison["record_mean_relative_l2"])
    maximum_raw = dict(comparison["maximum_instance_relative_l2"])
    validation_gate = dict(validation["promotion_gate"])
    return {
        "schema": "frozen_r5b_cpadc_vs_pi_deeponet_exact_panel_v1",
        "status": "complete_comparison_only",
        "split": comparison["split"],
        "method_identity": {
            "our_method": "r5b_neural_operator_plus_cpadc_instance_finetuning",
            "parent": "r5b_neural_operator",
            "baseline": "pi_deeponet_full_training_set",
            "online_neural_weight_update": False,
            "online_instance_coefficients": "rank16_cpu_ridge_from_two_guarded_onset_frames",
        },
        "protocol": {
            "record_count": int(comparison["record_count"]),
            "frames_per_record": int(comparison["frames_per_record_for_accuracy"]),
            "family_counts": comparison["family_counts"],
            "selection_scope": comparison["selection_scope"],
            "metric": "global_energy_relative_l2",
        },
        "global_energy_relative_l2": {
            "our_r5b_plus_cpadc": float(global_raw["cpadc_r7_adapted"]),
            "r5b_parent": float(global_raw["cpadc_parent"]),
            "pi_deeponet": float(global_raw["pi_deeponet"]),
        },
        "family_energy_relative_l2": {
            family: {
                "our_r5b_plus_cpadc": float(values["cpadc_r7_adapted"]),
                "r5b_parent": float(values["cpadc_parent"]),
                "pi_deeponet": float(values["pi_deeponet"]),
            }
            for family, values in family_raw.items()
        },
        "record_mean_relative_l2": {
            "our_r5b_plus_cpadc": float(record_raw["cpadc_r7_adapted"]),
            "r5b_parent": float(record_raw["cpadc_parent"]),
            "pi_deeponet": float(record_raw["pi_deeponet"]),
        },
        "maximum_instance_relative_l2": {
            "our_r5b_plus_cpadc": float(maximum_raw["cpadc_r7_adapted"]),
            "r5b_parent": float(maximum_raw["cpadc_parent"]),
            "pi_deeponet": float(maximum_raw["pi_deeponet"]),
        },
        "comparison": {
            "pi_error_over_our_error": comparison["cpadc_vs_pi"][
                "global_pi_error_over_cpadc_error"
            ],
            "our_error_increase_relative_to_pi": (
                float(global_raw["cpadc_r7_adapted"])
                - float(global_raw["pi_deeponet"])
            )
            / float(global_raw["pi_deeponet"]),
            "our_32frame_error_reduction_from_r5b_parent": comparison[
                "cpadc_instance_adaptation_effect"
            ]["global_relative_error_reduction_from_parent"],
        },
        "full401_validation_instance_adaptation": {
            "mean_relative_improvement": validation_gate["mean_relative_improvement"],
            "nonworse_fraction": validation_gate["nonworse_fraction"],
            "acceptance_fraction": validation_gate["acceptance_fraction"],
            "aggregate_parent_relative_l2": validation_gate[
                "aggregate_parent_future_relative_l2"
            ],
            "aggregate_adapted_relative_l2": validation_gate[
                "aggregate_adapted_future_relative_l2"
            ],
            "checks": validation_gate["checks"],
            "passed": bool(validation_gate["passed"]),
        },
        "runtime": {
            "our_full401_including_adaptation_s": comparison[
                "cpadc_runtime_s_for_full_401_frame_output_including_adaptation"
            ],
            "our_adaptation_s": comparison["cpadc_adaptation_runtime_s"],
            "pi_32frame_s": comparison["pi_runtime_s_for_32_frame_output"],
            "caveat": comparison["runtime_comparison_caveat"],
        },
        "promotion": {
            "same_protocol_validation_passed": bool(
                validation["same_protocol_validation_passed"]
            ),
            "test_id_authorized": False,
            "claim": "validation failed; no accuracy advantage over PI-DeepONet",
        },
        "bindings": {
            **comparison["bindings"],
            "source_comparison": str(comparison_path.resolve()),
            "source_comparison_sha256": _sha256(comparison_path),
            "source_validation_terminal": str(validation_terminal_path.resolve()),
            "source_validation_terminal_sha256": _sha256(validation_terminal_path),
            "report_script_sha256": _sha256(Path(__file__)),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--validation-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.comparison, args.validation_terminal)
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
