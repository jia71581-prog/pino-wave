from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pytest

from scripts import benchmark_r4_parent_e2e_trainonly as bench


def _tiny_manifest() -> dict[str, object]:
    records = []
    source_index = 0
    for family in bench.FAMILIES:
        for group in range(4):
            records.append(
                {
                    "source_index": source_index,
                    "sample_id": f"train_{family}_{source_index:05d}",
                    "group_id": f"train:{family}:{group:05d}",
                    "sample_sha256": f"{source_index + 1:064x}",
                    "split": "train",
                    "split_id": 0,
                    "medium_type": family,
                }
            )
            source_index += 1
    records.append(
        {
            "source_index": source_index,
            "sample_id": "validation_uniform_00000",
            "group_id": "validation:uniform:00000",
            "sample_sha256": "f" * 64,
            "split": "validation",
            "split_id": 0,
            "medium_type": "uniform",
        }
    )
    return {"records": records, "time_s": [index / 400 for index in range(401)]}


def test_selection_is_train_only_three_per_family_and_globally_group_disjoint() -> None:
    records = bench.select_train_records(_tiny_manifest())
    assert len(records) == 9
    assert Counter(row.family for row in records) == Counter(
        {"uniform": 3, "layered": 3, "marmousi": 3}
    )
    assert {row.split for row in records} == {"train"}
    assert len({row.group_id for row in records}) == 9
    assert not any(row.sample_id.startswith("validation") for row in records)


def test_registered_selection_and_time_axis_contract() -> None:
    manifest = bench.load_manifest_payload()
    selected = bench.select_train_records(manifest)
    assert [row.sample_id for row in selected] == [
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
    times = np.asarray(manifest["time_s"], dtype=np.float64)
    assert times.shape == (401,)
    assert np.all(np.diff(times) > 0)
    assert times[0] == 0.0 and times[-1] == 1.0


def test_nearest_rank_p95_for_nine_is_maximum_without_interpolation() -> None:
    values = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5]
    assert bench.nearest_rank(values, 0.95) == 0.9
    with pytest.raises(ValueError):
        bench.nearest_rank([], 0.95)


def test_timing_boundary_excludes_only_disk_io_and_one_time_model_load() -> None:
    assert bench.TIMING_BOUNDARY["excluded"] == [
        "record input disk I/O",
        "one-time model/checkpoint/normalizer loading",
    ]
    included = " ".join(bench.TIMING_BOUNDARY["included"])
    for phrase in (
        "H2D",
        "medium encoding",
        "source preparation",
        "401 stored times",
        "physical pressure decode",
        "D2H",
        "CUDA synchronization",
        "C-contiguous CPU output",
    ):
        assert phrase in included


def test_output_contract_requires_shape_float32_finite_and_c_contiguous() -> None:
    good = np.zeros(bench.OUTPUT_SHAPE, dtype=np.float32, order="C")
    metadata = bench.validate_cpu_output(good)
    assert metadata["shape"] == list(bench.OUTPUT_SHAPE)
    assert metadata["c_contiguous"] is True
    assert metadata["serialized"] is False
    with pytest.raises(bench.BenchmarkContractError, match="shape"):
        bench.validate_cpu_output(np.zeros((1, 400, 201, 201), dtype=np.float32))
    noncontiguous = good[:, :, :, ::-1]
    with pytest.raises(bench.BenchmarkContractError, match="C-contiguous"):
        bench.validate_cpu_output(noncontiguous)
    bad = good.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        bench.validate_cpu_output(bad)


@pytest.mark.parametrize(
    "name",
    [
        "wavefield",
        "/wavefield",
        "target",
        "pressure_truth",
        "onset_frame_0",
        "validation_wavefield",
        "test_id/label",
        "medium_type",
    ],
)
def test_truth_read_guard_rejects_every_unregistered_or_truth_dataset(name: str) -> None:
    with pytest.raises(bench.TruthAccessError):
        bench.assert_dataset_read_allowed(name)


def test_truth_read_guard_allows_only_registered_nontruth_inputs() -> None:
    for name in bench.ALLOWED_H5_DATASETS:
        bench.assert_dataset_read_allowed(name)
    assert "wavefield" not in bench.ALLOWED_H5_DATASETS


def test_static_h5_reads_are_all_guarded_and_allowlisted() -> None:
    tree = ast.parse(bench.SCRIPT_PATH.read_text(encoding="utf8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_read_h5_array" and len(node.args) >= 2:
                assert isinstance(node.args[1], ast.Constant)
                names.append(str(node.args[1].value))
    assert set(names) == set(bench.ALLOWED_H5_DATASETS)
    source = bench.SCRIPT_PATH.read_text(encoding="utf8")
    assert 'handle["wavefield"]' not in source
    assert "read_wavefield(" not in source


def test_hdf5_binding_hash_is_forbidden(tmp_path: Path) -> None:
    target = tmp_path / "data.h5"
    target.write_bytes(b"not truth")
    with pytest.raises(bench.TruthAccessError, match="must not be byte-hashed"):
        bench.sha256_file(target)


def test_binding_drift_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "binding.json"
    target.write_text('{"value":1}\n', encoding="utf8")
    frozen = bench.file_binding(target)
    bench.verify_binding(frozen)
    target.write_text('{"value":2}\n', encoding="utf8")
    with pytest.raises(bench.BindingDriftError):
        bench.verify_binding(frozen)


def test_disk_gate_rejects_below_two_gib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "free_disk_bytes", lambda path=bench.PROJECT_ROOT: 2_147_483_647)
    with pytest.raises(bench.DiskRiskError, match="below required"):
        bench.require_free_disk()


def test_atomic_terminal_is_exclusive_bounded_and_leaves_no_partial(tmp_path: Path) -> None:
    destination = tmp_path / "terminal.json"
    written = bench.atomic_json_exclusive({"status": "success"}, destination, limit=1024)
    assert 0 < written <= 1024
    assert json.loads(destination.read_text())["status"] == "success"
    assert not list(tmp_path.glob("*.partial-*"))
    assert not list(tmp_path.glob(".*.partial-*"))
    with pytest.raises(FileExistsError):
        bench.atomic_json_exclusive({"status": "failed"}, destination, limit=1024)
    with pytest.raises(bench.BenchmarkContractError, match="exceeds limit"):
        bench.atomic_json_exclusive({"payload": "x" * 2000}, tmp_path / "large.json", limit=32)
    assert not (tmp_path / "large.json").exists()


def test_budget_guard_fails_closed() -> None:
    guard = bench.BudgetGuard(maximum_seconds=0.0)
    time.sleep(0.001)
    with pytest.raises(bench.BudgetExceededError, match="budget exceeded"):
        guard.check("unit_test")


def test_preregistration_rejects_nulls_and_markers() -> None:
    with pytest.raises(bench.BenchmarkContractError, match="null"):
        bench.assert_no_placeholders_or_nulls({"x": None})
    with pytest.raises(bench.BenchmarkContractError, match="placeholder"):
        bench.assert_no_placeholders_or_nulls({"x": "TBD"})
    bench.assert_no_placeholders_or_nulls({"x": [1, "bound"]})


def test_no_checkpoint_write_api_or_prediction_serialization_in_benchmark() -> None:
    tree = ast.parse(bench.SCRIPT_PATH.read_text(encoding="utf8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "save_checkpoint_atomic" not in imported
    source = bench.SCRIPT_PATH.read_text(encoding="utf8")
    assert "torch.save(" not in source
    assert "np.save(" not in source
    assert "np.savez(" not in source


def test_failure_taxonomy_and_gate_constants_are_frozen() -> None:
    assert bench.TRADITIONAL_REFERENCE_S == 20.34137312322855
    assert bench.RUNTIME_GATE_S == 2.034137312322855
    assert bench.PEAK_GATE_BYTES == int(23.5 * 1024**3)
    assert bench.MIN_FREE_BYTES == 2_147_483_648
    assert bench.MAX_RESULT_BYTES == 10 * 1024**2
    assert bench.GPU_BUDGET_SECONDS == 900.0
