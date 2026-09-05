from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from saved_time_phase_operator_v4.evaluation import sha256_file
from scripts.evaluate_marmousi_source_position_control import (
    PREDICTION_SCHEMA,
    REFERENCE_SCHEMA,
    _atomic_json,
    _atomic_npz,
    cluster_bootstrap_by_slice,
    load_protocol,
    load_verified_prediction_manifest,
    load_verified_reference_manifest,
    locked_cases,
)


PROTOCOL_PATH = Path(
    "paper/tgrs_helmholtz_operator/marmousi_fixed_frequency_source_ood_protocol_20260813.json"
)


def test_locked_cases_vary_position_only() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    scope = protocol["generalization_scope"]
    assert scope["varied_variable"] == "source position (x_m, z_m) only"
    assert scope["source_frequency_generalization_in_scope"] is False
    assert scope["frequency_sweep_permitted"] is False
    rows = locked_cases(protocol)
    assert len(rows) == 240
    assert {row["slice_rank"] for row in rows} == set(range(1, 31))
    assert {row["source_parameters"][2] for row in rows} == {19.0}
    assert {row["source_parameters"][3] for row in rows} == {1.5 / 19.0}
    assert {row["source_parameters"][4] for row in rows} == {1.0}
    assert sum(row["role"] == "interpolation" for row in rows) == 150
    assert sum(row["role"] == "outside_train_position_range" for row in rows) == 90


def test_predict_call_tree_has_no_target_reader() -> None:
    source_path = Path("scripts/evaluate_marmousi_source_position_control.py")
    tree = ast.parse(source_path.read_text(encoding="utf8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = ["predict"]
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        node = functions[name]
        attributes = {
            call.func.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        }
        assert "read_wavefield" not in attributes
        assert "simulate" not in attributes
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id in functions:
                    pending.append(call.func.id)


def _fake_protocol(tmp_path: Path) -> Path:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf8"))
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf8")
    return path


def test_prediction_seal_rejects_frequency_change_and_tampering(tmp_path: Path) -> None:
    protocol_path = _fake_protocol(tmp_path)
    protocol = load_protocol(protocol_path)
    rows = locked_cases(protocol)
    records = []
    for row in rows:
        path = tmp_path / f"{row['record_id']}.npz"
        _atomic_npz(path, prediction_tzx=np.zeros((4, 2, 2), dtype=np.float32))
        records.append({**row, "prediction_path": str(path), "prediction_sha256": sha256_file(path)})
    manifest = tmp_path / "prediction_manifest.json"
    payload = {
        "schema": PREDICTION_SCHEMA,
        "status": "complete",
        "truth_wavefield_access": False,
        "source_frequency_varied": False,
        "protocol_sha256": sha256_file(protocol_path),
        "fixed_source_parameters": protocol["fixed_source_parameters"],
        "inputs": [],
        "records": records,
    }
    _atomic_json(payload, manifest)
    _, paths = load_verified_prediction_manifest(manifest, protocol_path)
    assert len(paths) == 240

    payload["records"][0]["source_parameters"][2] = 20.0
    changed_manifest = tmp_path / "changed_frequency.json"
    _atomic_json(payload, changed_manifest)
    with pytest.raises(ValueError, match="nonfixed source frequency"):
        load_verified_prediction_manifest(changed_manifest, protocol_path)

    with Path(records[-1]["prediction_path"]).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="seal mismatch"):
        load_verified_prediction_manifest(manifest, protocol_path)


def test_cluster_bootstrap_resamples_velocity_slices_not_positions() -> None:
    rows = []
    for rank in range(5):
        for position in range(8):
            rows.append({"slice_rank": rank, "metrics": {"error": float(rank)}})
    result = cluster_bootstrap_by_slice(rows, repetitions=500, seed=11)
    assert result["independent_unit"] == "velocity_slice"
    assert result["independent_velocity_slice_count"] == 5
    assert result["source_positions_per_slice"] == [8]
    assert result["metrics"]["error"]["mean_of_slice_means"] == 2.0
    assert result["metrics"]["error"]["slice_means"] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_protocol_rejects_frequency_scope_or_incomplete_slice_census(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf8"))
    payload["generalization_scope"]["frequency_sweep_permitted"] = True
    changed_scope = tmp_path / "frequency.json"
    changed_scope.write_text(json.dumps(payload), encoding="utf8")
    with pytest.raises(ValueError, match="isolates source-position"):
        load_protocol(changed_scope)

    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf8"))
    payload["velocity_panel"]["slices"].pop()
    payload["evaluation"]["record_count"] = 232
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps(payload), encoding="utf8")
    with pytest.raises(ValueError, match="30 slices"):
        load_protocol(incomplete)


def test_shared_reference_preserves_origin_seal_but_accepts_same_protocol_model(
    tmp_path: Path,
) -> None:
    protocol_path = _fake_protocol(tmp_path)
    protocol = load_protocol(protocol_path)
    records = []
    references = []
    for row in locked_cases(protocol):
        prediction_path = tmp_path / "predictions" / f"{row['record_id']}.npz"
        reference_path = tmp_path / "references" / f"{row['record_id']}.npz"
        _atomic_npz(prediction_path, prediction_tzx=np.zeros((4, 2, 2), dtype=np.float32))
        _atomic_npz(reference_path, target_tzx=np.zeros((4, 2, 2), dtype=np.float32))
        records.append(
            {**row, "prediction_path": str(prediction_path), "prediction_sha256": sha256_file(prediction_path)}
        )
        references.append(
            {**row, "reference_path": str(reference_path), "reference_sha256": sha256_file(reference_path)}
        )
    origin = tmp_path / "origin_prediction.json"
    origin_payload = {
        "schema": PREDICTION_SCHEMA,
        "status": "complete",
        "truth_wavefield_access": False,
        "source_frequency_varied": False,
        "protocol_sha256": sha256_file(protocol_path),
        "fixed_source_parameters": protocol["fixed_source_parameters"],
        "inputs": [],
        "records": records,
    }
    _atomic_json(origin_payload, origin)
    candidate = tmp_path / "candidate_prediction.json"
    _atomic_json({**origin_payload, "checkpoint_sha256": "c" * 64}, candidate)
    reference_manifest = tmp_path / "reference_manifest.json"
    _atomic_json(
        {
            "schema": REFERENCE_SCHEMA,
            "status": "complete",
            "prediction_seal_verified_before_truth_generation": True,
            "prediction_manifest": str(origin),
            "prediction_manifest_sha256": sha256_file(origin),
            "protocol_sha256": sha256_file(protocol_path),
            "source_frequency_varied": False,
            "solver_reproduction": {"passed": True},
            "records": references,
        },
        reference_manifest,
    )
    _, paths = load_verified_reference_manifest(
        reference_manifest, protocol_path, candidate
    )
    assert len(paths) == 240

    changed = json.loads(candidate.read_text(encoding="utf8"))
    changed["protocol_sha256"] = "d" * 64
    _atomic_json(changed, tmp_path / "different_protocol_prediction.json")
    with pytest.raises(ValueError, match="different protocols"):
        load_verified_reference_manifest(
            reference_manifest, protocol_path, tmp_path / "different_protocol_prediction.json"
        )
