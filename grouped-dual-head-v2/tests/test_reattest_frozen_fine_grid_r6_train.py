from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from scripts.audit_target5 import audit as audit_target5
from scripts.reattest_frozen_fine_grid_r6_train import (
    CANDIDATE,
    EXPECTED_DIGESTS,
    FAMILIES,
    RUNTIME_LIMIT_S,
    TruthStateGuard,
    aggregate_measurements,
    atomic_write_bytes,
    classify_complete,
    dependency_closure,
    first_occurrence,
    prediction_qc_and_hash,
    report_bytes_fixed_point,
    select_from_metadata,
    selection_digests,
    serialize_report,
    validate_invocation,
    verify_dependency_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/root/autodl-tmp/data/jiayh/data/"
    "acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
)
SELECTION = ROOT / "results/frozen_fine_grid_r6_current_env_reattest_v1_20260905/selection_manifest.json"
DEPENDENCIES = ROOT / "results/frozen_fine_grid_r6_current_env_reattest_v1_20260905/dependency_manifest.json"
REFERENCE = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "pretraining/target10_instance_finetune_r2_muon/traditional_lwc84_runtime.json"
)


def test_selection_exact_rule_lists_digests_and_first_occurrence_groups() -> None:
    manifest = json.loads(SELECTION.read_text(encoding="utf-8"))
    with h5py.File(SOURCE, "r", swmr=True) as handle:
        records = select_from_metadata(handle)
    assert records == manifest["records"]
    assert all(set(row) == {"source_index", "sample_id", "group_id", "family", "sample_sha256"} for row in records)
    assert selection_digests(records) == EXPECTED_DIGESTS == manifest["digests"]
    assert [row["source_index"] for row in records[:1]] == [311]
    assert [row["source_index"] for row in records[20:21]] == [780]
    assert [row["source_index"] for row in records[40:41]] == [2165]
    assert {family: sum(row["family"] == family for row in records) for family in FAMILIES} == {
        "uniform": 20,
        "layered": 20,
        "marmousi": 20,
    }
    groups = first_occurrence([row["group_id"] for row in records])
    assert len(groups) == 29
    assert hashlib.sha256(
        json.dumps(groups, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest() == EXPECTED_DIGESTS["groups_first_occurrence"]


def test_selection_builder_source_contains_no_wavefield_access() -> None:
    source = inspect.getsource(select_from_metadata)
    assert "wavefield" not in source


class _FakeWavefield:
    def __init__(self) -> None:
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return np.zeros((401, 2, 2), dtype=np.float32)


def _record() -> dict:
    return {
        "source_index": 7,
        "sample_id": "train_uniform_00007",
        "group_id": "train:uniform:00007",
        "family": "uniform",
        "sample_sha256": "a" * 64,
    }


def test_truth_guard_hash_before_truth_identity_and_exactly_once() -> None:
    wavefield = _FakeWavefield()
    handle = {"wavefield": wavefield}
    guard = TruthStateGuard([_record()])
    with pytest.raises(PermissionError, match="before prediction hash"):
        guard.read_truth(handle, index=7, sample_id=_record()["sample_id"], split="train", family="uniform")
    assert not wavefield.reads
    guard.register_prediction(index=7, sample_id=_record()["sample_id"], family="uniform", digest="b" * 64)
    with pytest.raises(PermissionError):
        guard.read_truth(handle, index=8, sample_id=_record()["sample_id"], split="train", family="uniform")
    with pytest.raises(PermissionError):
        guard.read_truth(handle, index=7, sample_id="wrong", split="train", family="uniform")
    with pytest.raises(PermissionError):
        guard.read_truth(handle, index=7, sample_id=_record()["sample_id"], split="validation", family="uniform")
    with pytest.raises(PermissionError):
        guard.read_truth(handle, index=7, sample_id=_record()["sample_id"], split="train", family="anomaly")
    truth = guard.read_truth(handle, index=7, sample_id=_record()["sample_id"], split="train", family="uniform")
    assert truth.shape == (401, 2, 2)
    with pytest.raises(PermissionError):
        guard.read_truth(handle, index=7, sample_id=_record()["sample_id"], split="train", family="uniform")
    guard.mark_scored(7)
    ledger = guard.summary()
    assert ledger["truth_read_count"] == ledger["unique_truth_indices"] == ledger["scored_count"] == 1
    assert ledger["prediction_hashed_before_truth"]
    assert ledger["validation_truth_reopened_this_run"] is False
    assert ledger["test_id_truth_reopened_this_run"] is False
    assert [row["to"] for row in ledger["transitions"]] == [
        "prediction_hashed",
        "truth_authorized",
        "truth_read",
        "scored",
    ]


def test_prediction_qc_shape_finite_top_row_and_hash() -> None:
    value = np.zeros((401, 201, 201), dtype=np.float32)
    checked, digest = prediction_qc_and_hash(value)
    assert checked.flags.c_contiguous and checked.dtype == np.float32 and len(digest) == 64
    with pytest.raises(RuntimeError, match="shape"):
        prediction_qc_and_hash(value[:400])
    bad = value.copy()
    bad[0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="top row"):
        prediction_qc_and_hash(bad)
    bad = value.copy()
    bad[0, 1, 0] = np.nan
    with pytest.raises(FloatingPointError):
        prediction_qc_and_hash(bad)


def _synthetic_rows(runtime_values: list[float]) -> list[dict]:
    rows = []
    for index, runtime in enumerate(runtime_values):
        family = FAMILIES[index % 3]
        rows.append(
            {
                "family": family,
                "relative_l2": 0.01,
                "error_terms": [1.0, 10000.0],
                "temporal_error_terms": {
                    "early": [0.2, 2000.0],
                    "middle": [0.3, 3000.0],
                    "late": [0.5, 5000.0],
                },
                "outer_runtime_s": runtime,
                "solver_internal_compute_elapsed_s": 0.001,
            }
        )
    return rows


def test_float64_aggregate_family_time_and_nearest_rank_57() -> None:
    rows = _synthetic_rows([float(index) for index in range(1, 61)])
    metrics = aggregate_measurements(rows)
    assert metrics["aggregate_relative_l2"] == pytest.approx(0.01)
    assert metrics["family_relative_l2"] == {family: pytest.approx(0.01) for family in FAMILIES}
    assert set(metrics["temporal_band_relative_l2"]) == {"early", "middle", "late"}
    assert metrics["outer_runtime_s"]["p95_rank_for_60"] == 57
    assert metrics["outer_runtime_s"]["p95_nearest_rank"] == 57.0


def test_runtime_gate_uses_outer_not_internal_and_status_boundaries() -> None:
    metrics = {
        "aggregate_relative_l2": 0.05,
        "family_relative_l2": {family: 0.05 for family in FAMILIES},
        "maximum_instance_relative_l2": 0.05,
        "outer_runtime_s": {"mean": RUNTIME_LIMIT_S, "p95_nearest_rank": RUNTIME_LIMIT_S},
    }
    assert classify_complete(metrics) == "current_environment_reattest_passed"
    metrics["outer_runtime_s"]["mean"] += 1.0e-12
    assert classify_complete(metrics) == "rejected"
    assert classify_complete(metrics, contract_valid=False) == "invalid"
    rows = _synthetic_rows([RUNTIME_LIMIT_S + 1.0] * 60)
    assert all(row["solver_internal_compute_elapsed_s"] < RUNTIME_LIMIT_S for row in rows)
    assert classify_complete(aggregate_measurements(rows)) == "rejected"


def _invocation(tmp_path: Path, *, mode: str, status: str) -> tuple[argparse.Namespace, dict]:
    paths = {
        "preregistration": str(tmp_path / "prereg.json"),
        "selection": str(tmp_path / "selection.json"),
        "source_h5": str(tmp_path / "source.h5"),
        "source_manifest": str(tmp_path / "manifest.jsonl"),
        "marmousi_npy": str(tmp_path / "marmousi.npy"),
        "reference_report": str(tmp_path / "reference.json"),
        "smoke_output": str(tmp_path / "smoke.json"),
        "run_output": str(tmp_path / "run.json"),
    }
    for key in ("selection", "source_h5", "source_manifest", "marmousi_npy", "reference_report"):
        Path(paths[key]).write_text("bound\n", encoding="utf-8")
    prereg = {
        "schema": "frozen_fine_grid_r6_current_env_reattest_preregistration_v1",
        "candidate": CANDIDATE,
        "status": status,
        "paths": paths,
        "prerequisites": {"smoke": {"path": str(tmp_path / "smoke_terminal.json"), "sha256": None, "status": "pending"}},
    }
    Path(paths["preregistration"]).write_text(json.dumps(prereg), encoding="utf-8")
    args = argparse.Namespace(
        smoke=mode == "smoke",
        run=mode == "run",
        preregistration=Path(paths["preregistration"]),
        selection=Path(paths["selection"]),
        source_h5=Path(paths["source_h5"]),
        manifest=Path(paths["source_manifest"]),
        marmousi=Path(paths["marmousi_npy"]),
        reference=Path(paths["reference_report"]),
        output=Path(paths[f"{mode}_output"]),
    )
    return args, prereg


def test_fixed_path_override_and_run_pending_prerequisite(tmp_path: Path) -> None:
    args, prereg = _invocation(tmp_path, mode="smoke", status="draft_pending_audit")
    assert validate_invocation(args, prereg)[0] == "smoke"
    args.output = tmp_path / "override.json"
    with pytest.raises(RuntimeError, match="override"):
        validate_invocation(args, prereg)

    run_path = tmp_path / "run_case"
    run_path.mkdir()
    args, prereg = _invocation(
        run_path, mode="run", status="smoke_passed_run_pending_independent_audit"
    )
    with pytest.raises(RuntimeError, match="prerequisite"):
        validate_invocation(args, prereg)
    smoke = Path(prereg["prerequisites"]["smoke"]["path"])
    smoke.write_text("{}\n", encoding="utf-8")
    prereg["prerequisites"]["smoke"].update({"status": "passed", "sha256": hashlib.sha256(smoke.read_bytes()).hexdigest()})
    assert validate_invocation(args, prereg)[0] == "run"


def test_dependency_closure_manifest_recomputes_exactly() -> None:
    manifest = json.loads(DEPENDENCIES.read_text(encoding="utf-8"))
    observed = verify_dependency_manifest(manifest)
    assert len(observed) == len(manifest["files"])
    assert sorted(observed) == [row["path"] for row in manifest["files"]]


def test_dependency_closure_expected_paths_placeholder() -> None:
    roots = [
        ROOT / "scripts/reattest_frozen_fine_grid_r6_train.py",
        ROOT / "scripts/gate_lwc84_cuda_graph_fine_grid_trainonly.py",
        ROOT / "scripts/audit_target5.py",
        ROOT / "src/fno_acoustic/data_generation/solver_lwc84_fused.py",
        ROOT / "src/fno_acoustic/data_generation/grid.py",
        ROOT / "src/fno_acoustic/data_generation/restriction.py",
        ROOT / "src/fno_acoustic/data_generation/velocity_models_lwc84.py",
    ]
    actual = [path.relative_to(ROOT).as_posix() for path in dependency_closure(roots)]
    assert actual == [
        "scripts/__init__.py",
        "scripts/audit_target5.py",
        "scripts/gate_lwc84_cuda_graph_fine_grid_trainonly.py",
        "scripts/reattest_frozen_fine_grid_r6_train.py",
        "src/fno_acoustic/__init__.py",
        "src/fno_acoustic/ais_model_components.py",
        "src/fno_acoustic/data.py",
        "src/fno_acoustic/data_generation/__init__.py",
        "src/fno_acoustic/data_generation/cpml.py",
        "src/fno_acoustic/data_generation/free_surface.py",
        "src/fno_acoustic/data_generation/fused_lwc84.py",
        "src/fno_acoustic/data_generation/grid.py",
        "src/fno_acoustic/data_generation/lwc84.py",
        "src/fno_acoustic/data_generation/model_marmousi.py",
        "src/fno_acoustic/data_generation/restriction.py",
        "src/fno_acoustic/data_generation/ricker.py",
        "src/fno_acoustic/data_generation/solver_lwc84.py",
        "src/fno_acoustic/data_generation/solver_lwc84_fused.py",
        "src/fno_acoustic/data_generation/source.py",
        "src/fno_acoustic/data_generation/stencils.py",
        "src/fno_acoustic/data_generation/velocity_models_lwc84.py",
        "src/fno_acoustic/model.py",
        "src/fno_acoustic/model_ais_mqfno.py",
        "src/fno_acoustic/model_factorized.py",
        "src/fno_acoustic/normalization.py",
        "src/fno_acoustic/numerics/__init__.py",
        "src/fno_acoustic/numerics/drp_coefficients.py",
        "src/fno_acoustic/schema.py",
        "src/fno_acoustic/temporal_operator.py",
    ]


def test_fresh_subprocess_imported_local_modules_are_covered() -> None:
    code = """
import json, pathlib, sys
import scripts.reattest_frozen_fine_grid_r6_train
root = pathlib.Path.cwd().resolve()
paths = []
for module in sys.modules.values():
    value = getattr(module, '__file__', None)
    if not value:
        continue
    path = pathlib.Path(value).resolve()
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        continue
    if rel.startswith('scripts/') or rel.startswith('src/fno_acoustic/'):
        paths.append(rel)
print(json.dumps(sorted(set(paths))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True
    )
    imported = set(json.loads(result.stdout))
    manifest = json.loads(DEPENDENCIES.read_text(encoding="utf-8"))
    covered = {row["path"] for row in manifest["files"]}
    assert imported <= covered


def test_report_serializer_fixed_point_matches_actual_file_size(tmp_path: Path) -> None:
    payload = {
        "status": "synthetic",
        "resources": {"output_bytes": 0},
        "validation_truth_reopened_this_run": False,
        "test_id_truth_reopened_this_run": False,
    }
    encoded = report_bytes_fixed_point(payload)
    assert encoded == serialize_report(payload)
    assert payload["resources"]["output_bytes"] == len(encoded)
    path = tmp_path / "report.json"
    atomic_write_bytes(encoded, path)
    assert path.stat().st_size == payload["resources"]["output_bytes"]
    assert path.read_bytes() == encoded


def _wavefield_subscript_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            value = node.value
            if isinstance(value, ast.Name) and value.id == "handle":
                slice_value = node.slice
                if isinstance(slice_value, ast.Constant) and slice_value.value == "wavefield":
                    count += 1
    return count


def test_wavefield_index_surface_is_unique_and_historical_audit_has_none() -> None:
    runner = ROOT / "scripts/reattest_frozen_fine_grid_r6_train.py"
    audit = ROOT / "scripts/audit_target5.py"
    assert _wavefield_subscript_count(runner) == 1
    assert _wavefield_subscript_count(audit) == 0
    assert 'handle["wavefield"]' in inspect.getsource(TruthStateGuard.read_truth)


def test_historical_consumption_fields_are_semantically_explicit() -> None:
    runner = (ROOT / "scripts/reattest_frozen_fine_grid_r6_train.py").read_text(encoding="utf-8")
    prereg = (ROOT / "results/frozen_fine_grid_r6_current_env_reattest_v1_20260905/preregistration.json").read_text(encoding="utf-8")
    combined = runner + prereg
    assert "historical_validation_previously_consumed" in combined
    assert "historical_test_id_previously_consumed" in combined
    assert "validation_truth_reopened_this_run" in combined
    assert "test_id_truth_reopened_this_run" in combined
    assert "historical_validation_opened" not in combined
    assert "historical_test_id_opened" not in combined


def test_audit_target5_current_artifacts_pass() -> None:
    result = audit_target5(ROOT, REFERENCE)
    assert result["passed"] is True
    assert result["status"] == "target_met"


def test_no_weights_checkpoints_or_prediction_output_surface() -> None:
    source = (ROOT / "scripts/reattest_frozen_fine_grid_r6_train.py").read_text(encoding="utf-8")
    assert "torch.save" not in source
    assert "atomic_checkpoint" not in source
    assert '"prediction": prediction' not in source
    assert ".pt" not in source and ".pth" not in source and ".ckpt" not in source
