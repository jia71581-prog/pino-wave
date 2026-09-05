#!/usr/bin/env python3
"""Read-only reproducibility audit for the frozen Target-5 terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_REFERENCE_SHA256 = (
    "116a1cb7cdd13869e86e14b08ef0a80c31284d5d248cb9c0952a880f2aba5bd8"
)
TARGET_FAMILIES = ("uniform", "layered", "marmousi")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def audit(workspace: Path, reference_report: Path) -> dict[str, object]:
    results = workspace / "results"
    target_path = results / "frozen_fine_grid_target_met_r6_20260815.json"
    candidate_path = results / "frozen_fine_grid_dt625us_candidate_r6_20260815.json"
    validation_report_path = (
        results
        / "frozen_fine_grid_validation_r6_resume1_20260815"
        / "complete_validation_gate.json"
    )
    validation_terminal_path = validation_report_path.with_name("terminal.json")
    test_report_path = (
        results
        / "frozen_fine_grid_test_id_r6_20260815"
        / "complete_test_id_gate.json"
    )
    test_terminal_path = test_report_path.with_name("terminal.json")

    target = load_json(target_path)
    candidate = load_json(candidate_path)
    reference = load_json(reference_report)
    reports = {
        "validation": load_json(validation_report_path),
        "test_id": load_json(test_report_path),
    }
    terminals = {
        "validation": load_json(validation_terminal_path),
        "test_id": load_json(test_terminal_path),
    }
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    check("target_status", target.get("status") == "target_met", target.get("status"), "target_met")
    protocol = target["protocol"]
    assert isinstance(protocol, dict)
    check(
        "future_truth_sealed",
        protocol.get("future_truth_used_by_method") is False,
        protocol.get("future_truth_used_by_method"),
        False,
    )
    check(
        "no_post_validation_tuning",
        protocol.get("post_validation_tuning") is False,
        protocol.get("post_validation_tuning"),
        False,
    )
    check(
        "candidate_output_materialization",
        candidate["frozen_hyperparameters"].get("prediction_materialization_included") is True,
        candidate["frozen_hyperparameters"].get("prediction_materialization_included"),
        True,
    )
    reference_protocol = reference["protocol"]
    assert isinstance(reference_protocol, dict)
    for key in ("same_gpu_as_deployment", "cuda_synchronized_timing", "includes_output_materialization"):
        check(f"reference_{key}", reference_protocol.get(key) is True, reference_protocol.get(key), True)
    check(
        "reference_report_sha256",
        sha256_file(reference_report) == EXPECTED_REFERENCE_SHA256,
        sha256_file(reference_report),
        EXPECTED_REFERENCE_SHA256,
    )
    reference_s = float(reference["conservative_reference_runtime_s"])
    check(
        "reference_runtime_binding",
        math.isclose(reference_s, float(protocol["traditional_reference_s"]), rel_tol=0.0, abs_tol=1.0e-12),
        reference_s,
        protocol["traditional_reference_s"],
    )

    reproducibility = target["reproducibility"]
    assert isinstance(reproducibility, dict)
    bound_files = {
        "frozen_candidate": candidate_path,
        "evaluation_script": workspace / "scripts/evaluate_frozen_fine_grid_split.py",
        "aggregation_script": workspace / "scripts/aggregate_frozen_fine_grid_split.py",
        "fused_solver": workspace / "src/fno_acoustic/data_generation/solver_lwc84_fused.py",
        "functional_kernel": workspace / "src/fno_acoustic/data_generation/fused_lwc84.py",
    }
    source_h5 = Path(str(candidate["bindings"]["source_h5_sha256"] and reference["source_h5"]))
    bound_files["source_h5"] = source_h5
    manifest_path = source_h5.with_name("manifest.jsonl")
    bound_files["manifest"] = manifest_path
    marmousi_path = Path(
        "/root/autodl-tmp/home/jiayh/Data/data/marmousi_zenodo_16114161/"
        "marmousi1_vp_zx_751x2301_4m.npy"
    )
    bound_files["marmousi_npy"] = marmousi_path
    for name, path in bound_files.items():
        expected = str(reproducibility[f"{name}_sha256"])
        observed = sha256_file(path)
        check(f"binding_{name}", observed == expected, observed, expected)

    target_by_split = {name: target[name] for name in reports}
    split_paths = {
        "validation": (validation_report_path, validation_terminal_path),
        "test_id": (test_report_path, test_terminal_path),
    }
    for split, report in reports.items():
        target_split = target_by_split[split]
        assert isinstance(target_split, dict)
        terminal = terminals[split]
        report_path, terminal_path = split_paths[split]
        check(f"{split}_status", report.get("status") == "target_gate_passed", report.get("status"), "target_gate_passed")
        check(f"{split}_terminal", terminal.get("status") == "complete", terminal.get("status"), "complete")
        check(f"{split}_record_count", int(report["record_count"]) == 480, report["record_count"], 480)
        check(f"{split}_aggregate", float(report["aggregate_relative_l2"]) <= 0.05, report["aggregate_relative_l2"], "<=0.05")
        families = report["family_relative_l2"]
        assert isinstance(families, dict)
        for family in TARGET_FAMILIES:
            check(f"{split}_{family}", float(families[family]) <= 0.05, families[family], "<=0.05")
        runtime = report["runtime_s"]
        assert isinstance(runtime, dict)
        mean_speedup = reference_s / float(runtime["mean"])
        p95_speedup = reference_s / float(runtime["p95_nearest_rank"])
        check(f"{split}_mean_speedup", mean_speedup >= 10.0, mean_speedup, ">=10.0")
        check(f"{split}_p95_speedup", p95_speedup >= 10.0, p95_speedup, ">=10.0")
        check(
            f"{split}_report_sha256",
            sha256_file(report_path) == str(target_split["report_sha256"]),
            sha256_file(report_path),
            target_split["report_sha256"],
        )
        check(
            f"{split}_terminal_sha256",
            sha256_file(terminal_path) == str(target_split["terminal_sha256"]),
            sha256_file(terminal_path),
            target_split["terminal_sha256"],
        )

    passed = all(bool(row["passed"]) for row in checks)
    return {
        "schema": "target5_read_only_audit_v1",
        "status": "target_met" if passed else "audit_failed",
        "passed": passed,
        "target_terminal": str(target_path),
        "traditional_reference_report": str(reference_report),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=Path(
            "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
            "pretraining/target10_instance_finetune_r2_muon/traditional_lwc84_runtime.json"
        ),
    )
    args = parser.parse_args()
    report = audit(args.workspace.resolve(), args.reference_report.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
