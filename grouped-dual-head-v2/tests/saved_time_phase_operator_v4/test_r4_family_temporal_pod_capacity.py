from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import probe_r4_family_temporal_pod_capacity as probe


def test_registered_sample_selection_is_exact_train9_and_group_disjoint() -> None:
    manifest = probe.parent_runtime.load_manifest_payload()
    records = probe.parent_runtime.select_train_records(manifest)
    assert [record.sample_id for record in records] == [
        "train_uniform_00000",
        "train_uniform_00001",
        "train_uniform_00002",
        "train_layered_00000",
        "train_layered_00004",
        "train_layered_00008",
        "train_marmousi_00000",
        "train_marmousi_00005",
        "train_marmousi_00010",
    ]
    assert len(records) == 9
    assert Counter(record.family for record in records) == Counter(
        {family: 3 for family in probe.FAMILIES}
    )
    assert {record.split for record in records} == {"train"}
    assert len({record.group_id for record in records}) == 9


def test_streamed_covariance_matches_dense_residual_product() -> None:
    generator = torch.Generator().manual_seed(3407)
    residual = torch.randn(7, 3, 5, generator=generator, dtype=torch.float64)
    covariance = torch.zeros(7, 7, dtype=torch.float64)
    probe.stream_accumulate_temporal_covariance(
        covariance, residual, spatial_chunk=4
    )
    matrix = residual.reshape(7, -1)
    torch.testing.assert_close(covariance, matrix @ matrix.T, rtol=1.0e-12, atol=1.0e-12)


def test_synthetic_low_rank_oracle_recovers_reserved_future_residual() -> None:
    generator = torch.Generator().manual_seed(9016)
    time_count, height, width, true_rank = 20, 4, 5, 3
    raw = torch.randn(time_count, true_rank, generator=generator, dtype=torch.float64)
    temporal, _ = torch.linalg.qr(raw, mode="reduced")
    basis_coeff = torch.randn(true_rank, 2 * height * width, generator=generator, dtype=torch.float64)
    basis_residual = (temporal @ basis_coeff).reshape(time_count, 2, height, width)
    covariance = torch.zeros(time_count, time_count, dtype=torch.float64)
    for record in range(2):
        probe.stream_accumulate_temporal_covariance(
            covariance, basis_residual[:, record], spatial_chunk=7
        )
    eigenvalues, eigenvectors = probe.temporal_pod(covariance)
    assert int((eigenvalues > 1.0e-10).sum()) == true_rank

    parent = torch.randn(time_count, height, width, generator=generator, dtype=torch.float64)
    confirmation_coeff = torch.randn(
        true_rank, height * width, generator=generator, dtype=torch.float64
    )
    confirmation_residual = (temporal @ confirmation_coeff).reshape(
        time_count, height, width
    )
    truth = parent + confirmation_residual
    metrics = probe.oracle_projection_metrics(
        parent,
        truth,
        eigenvectors,
        future_start=4,
        ranks=(3,),
        spatial_chunk=6,
    )
    rank3 = metrics["ranks"]["3"]
    assert rank3["residual_energy_capture"] >= 1.0 - 1.0e-12
    assert rank3["corrected_mean_per_frame_relative_l2"] < 1.0e-12
    assert rank3["relative_error_reduction"] > 1.0 - 1.0e-10


def test_mean_per_frame_metric_is_not_joint_space_time_metric() -> None:
    truth = torch.tensor([[[1.0]], [[10.0]]], dtype=torch.float64)
    prediction = torch.tensor([[[0.0]], [[10.0]]], dtype=torch.float64)
    assert probe.mean_per_frame_relative_l2(prediction, truth) == pytest.approx(0.5)
    joint = float(torch.linalg.vector_norm(prediction - truth) / torch.linalg.vector_norm(truth))
    assert joint == pytest.approx(1.0 / np.sqrt(101.0))
    assert probe.mean_per_frame_relative_l2(prediction, truth) != pytest.approx(joint)


def test_future_window_starts_strictly_after_second_registered_observation() -> None:
    times = torch.linspace(0.0, 1.0, 401, dtype=torch.float64)
    observed = probe.onset_indices(times, t0_s=0.2, f0_hz=20.0)
    assert observed[1] == observed[0] + 1
    assert observed[1] + 1 > observed[1]
    assert 0 < observed[1] + 1 < 401


def test_gate_budget_and_output_constants_are_frozen() -> None:
    assert probe.RANKS == (8, 16, 32)
    assert probe.TIME_BLOCK == 16
    assert probe.GPU_SECONDS_MAXIMUM == 360.0
    assert probe.MIN_FREE_BYTES == 2 * 1024**3
    assert probe.MAX_OUTPUT_BYTES == 10 * 1024**2
    assert probe.PEAK_CUDA_BYTES_MAXIMUM == int(23.5 * 1024**3)
    assert probe.RANK32_ERROR_REDUCTION_MINIMUM == 0.20
    assert probe.RANK32_RESIDUAL_CAPTURE_MINIMUM == 0.30


def test_non_train_truth_hash_is_rejected_without_opening_h5() -> None:
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
        probe.stream_train_truth_sha256(record)


def test_script_has_no_array_or_checkpoint_serialization_api() -> None:
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
    assert "torch.save(" not in source
    assert "save_checkpoint_atomic" not in source
    assert "np.memmap" not in source


def test_only_declared_terminal_paths_are_constants() -> None:
    assert probe.PREREGISTRATION_PATH.name == (
        "r4e7_family_temporal_pod_capacity_train9_v1_preregistration_20260825.json"
    )
    assert probe.RESULT_PATH.name == (
        "r4e7_family_temporal_pod_capacity_train9_v1_20260825.json"
    )
    assert probe.PREREGISTRATION_PATH.parent == probe.PROJECT_ROOT / "results"
    assert probe.RESULT_PATH.parent == probe.PROJECT_ROOT / "results"
