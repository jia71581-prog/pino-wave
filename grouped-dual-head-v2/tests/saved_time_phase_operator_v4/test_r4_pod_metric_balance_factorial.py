from __future__ import annotations

import ast
from collections import Counter

import pytest
import torch

from scripts import probe_r4_pod_metric_balance_factorial as probe


def test_exact_historical_train9_is_reused_and_thirds_are_exposed_calibration() -> None:
    manifest = probe.parent_runtime.load_manifest_payload()
    records = probe.sample_census(manifest)
    assert tuple(row["sample_id"] for row in records) == probe.EXPECTED_SAMPLE_IDS
    assert tuple(row["group_id"] for row in records) == probe.EXPECTED_GROUP_IDS
    assert {row["split"] for row in records} == {"train"}
    assert len(set(probe.EXPECTED_GROUP_IDS)) == 9
    assert Counter(row["role"] for row in records) == Counter(
        {"basis": 6, "exposed_calibration": 3}
    )
    thirds = [row for row in records if row["family_position"] == 2]
    assert len(thirds) == 3
    assert all(row["historical_role"] == "disjoint_confirmation" for row in thirds)
    assert all(row["independent_confirmation"] is False for row in thirds)


def test_raw_and_equaltrace_covariances_are_distinct_under_record_scale_imbalance() -> None:
    first = torch.tensor([[10.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    second = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    raw = torch.zeros(2, 2, dtype=torch.float64)
    equaltrace = torch.zeros_like(raw)
    trace_first = probe.covariance_pair_update(raw, equaltrace, first, spatial_chunk=1)
    trace_second = probe.covariance_pair_update(raw, equaltrace, second, spatial_chunk=1)
    assert trace_first == pytest.approx(100.0)
    assert trace_second == pytest.approx(1.0)
    torch.testing.assert_close(raw, torch.diag(torch.tensor([100.0, 1.0], dtype=torch.float64)))
    torch.testing.assert_close(equaltrace, torch.eye(2, dtype=torch.float64))
    assert float(raw[0, 0] / raw[1, 1]) == pytest.approx(100.0)
    assert float(equaltrace[0, 0] / equaltrace[1, 1]) == pytest.approx(1.0)


def test_weighted_ls_cannot_worsen_weighted_squared_frame_objective() -> None:
    generator = torch.Generator().manual_seed(250825)
    design = torch.randn(11, 3, generator=generator, dtype=torch.float64)
    residual = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    weights = torch.logspace(-3, 3, 11, dtype=torch.float64)
    ordinary = probe.fit_coefficients(design, residual, weights=None)
    weighted = probe.fit_coefficients(design, residual, weights=weights)

    def objective(coefficients: torch.Tensor) -> torch.Tensor:
        errors = residual - design @ coefficients
        return (weights[:, None] * errors.square()).sum()

    assert float(objective(weighted)) <= float(objective(ordinary)) + 1.0e-10


def test_weighted_squared_objective_is_mean_squared_per_frame_relative_l2() -> None:
    truth_squared = torch.tensor([1.0, 4.0, 16.0], dtype=torch.float64)
    error_squared = torch.tensor([0.25, 1.0, 4.0], dtype=torch.float64)
    weights = truth_squared.clamp_min(probe.WEIGHT_EPS_SQUARED).reciprocal()
    weighted_mean = float((weights * error_squared).mean())
    metrics = probe.mean_frame_metrics_from_squared_errors(error_squared, truth_squared)
    assert metrics["mean_squared_per_frame_relative_l2"] == pytest.approx(weighted_mean)
    assert metrics["mean_per_frame_relative_l2"] == pytest.approx(0.5)


def test_actual_projection_reuses_one_residual_and_weighted_squared_metric_is_nonworse() -> None:
    generator = torch.Generator().manual_seed(825)
    parent = torch.randn(8, 2, 3, generator=generator, dtype=torch.float64)
    truth = torch.randn(8, 2, 3, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(
        torch.randn(8, 3, generator=generator, dtype=torch.float64), mode="reduced"
    )
    shared = probe.prepare_future_projection(parent, truth, future_start=2)
    residual_pointer = shared["future_residual"].data_ptr()
    ordinary = probe.fit_projection_metrics(
        shared, basis, fit_mode="ordinary", ranks=(3,), spatial_chunk=2
    )
    weighted = probe.fit_projection_metrics(
        shared, basis, fit_mode="weighted", ranks=(3,), spatial_chunk=2
    )
    assert shared["future_residual"].data_ptr() == residual_pointer
    ordinary_squared = ordinary["ranks"]["3"]["corrected_metrics"][
        "mean_squared_per_frame_relative_l2"
    ]
    weighted_squared = weighted["ranks"]["3"]["corrected_metrics"][
        "mean_squared_per_frame_relative_l2"
    ]
    assert weighted_squared <= ordinary_squared + 1.0e-12


def test_selection_uses_rank32_unsquared_gate_and_frozen_tie_order() -> None:
    def arm(reduction: float) -> dict:
        parent = 1.0
        corrected = parent * (1.0 - reduction)
        return {
            "parent_metrics": {"mean_per_frame_relative_l2": parent},
            "ranks": {
                "32": {
                    "corrected_metrics": {"mean_per_frame_relative_l2": corrected},
                    "paired_changes_vs_parent": {
                        "mean_per_frame_relative_l2_reduction": reduction
                    },
                    "finite": True,
                }
            },
        }

    families = {
        family: {"arms": {key: arm(0.25) for key in probe.ARM_KEYS}}
        for family in probe.FAMILIES
    }
    selection = probe.select_winner(families)
    assert selection["decision"] == "winner"
    assert selection["winner"] == "raw+weighted"
    families["uniform"]["arms"]["raw+weighted"] = arm(0.19)
    selection = probe.select_winner(families)
    assert selection["arm_summaries"]["raw+weighted"]["eligible"] is False


def test_non_train_truth_is_rejected_before_h5_open() -> None:
    record = probe.parent_runtime.SelectedRecord(
        source_index=0,
        sample_id="validation_uniform_00000",
        group_id="validation:uniform:00000",
        family="uniform",
        split="validation",
        split_id=1,
        manifest_sample_sha256="f" * 64,
    )
    with pytest.raises(probe.TruthScopeError, match="only train truth"):
        probe.legacy_probe.stream_train_truth_sha256(record)


def test_script_has_no_field_array_or_checkpoint_serialization_api() -> None:
    source = probe.SCRIPT_PATH.read_text(encoding="utf8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "save" not in calls
    assert "savez" not in calls
    assert "savez_compressed" not in calls
    assert "memmap" not in calls
    assert "tofile" not in calls
    assert "torch.save(" not in source
    assert "save_checkpoint_atomic" not in source
    assert ".write_text(" not in source
    assert ".write_bytes(" not in source


def test_only_allowed_terminal_paths_and_frozen_resource_contract() -> None:
    assert probe.PREREGISTRATION_PATH.name == (
        "r4e7_pod_metric_balance_factorial_calibration9_v1_"
        "preregistration_20260825.json"
    )
    assert probe.RESULT_PATH.name == (
        "r4e7_pod_metric_balance_factorial_calibration9_v1_20260825.json"
    )
    assert probe.PREREGISTRATION_PATH.parent == probe.PROJECT_ROOT / "results"
    assert probe.RESULT_PATH.parent == probe.PROJECT_ROOT / "results"
    assert probe.RANKS == (8, 16, 32)
    assert probe.TIME_BLOCK == 16
    assert probe.GPU_SECONDS_MAXIMUM == 360.0
    assert probe.MIN_FREE_BYTES == 2 * 1024**3
    assert probe.MAX_OUTPUT_BYTES == 10 * 1024**2
    assert probe.PEAK_CUDA_BYTES_MAXIMUM == int(23.5 * 1024**3)
    assert probe.RANK32_REDUCTION_MINIMUM == 0.20
    assert probe.TIE_ORDER == (
        "raw+weighted",
        "equaltrace+weighted",
        "equaltrace+ordinary",
        "raw+ordinary",
    )


def test_measured_command_is_not_executed_by_smoke_or_preregister_modes() -> None:
    source = probe.SCRIPT_PATH.read_text(encoding="utf8")
    assert 'choices=("smoke", "preregister", "measured")' in source
    assert "if args.mode == \"smoke\"" in source
    assert "elif args.mode == \"preregister\"" in source
    assert "else:\n        payload = run_measured(" in source
