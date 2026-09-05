from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.gate_r6_anchored_r54_device_runtime import (
    CANDIDATE,
    PARENT_BYTES,
    STATIC_BYTES,
    TIME_BYTES,
    TransferLedger,
    bootstrap,
    device_result_qc,
    nearest_rank,
    runtime_gate,
    validate_owned_device_tensor,
    validate_remaining,
    validate_transfer_summary,
    verify_dependency_manifest,
    write_failure,
)
from scripts.gate_r6_anchored_r54_runtime import (
    bit_preserving_residual_add,
    fresh_head_cpu,
    runtime_dependency_closure,
    validate_selection_manifest,
)
from scripts.reattest_frozen_fine_grid_r6_train import atomic_write_bytes, report_bytes_fixed_point
from fno_acoustic.data_generation.restriction import restrict_nodal_2x
from fno_acoustic.data_generation.solver_lwc84_fused import FusedLWC84CPMLSolver
from fno_acoustic.data_generation.solver_lwc84_fused_device import DeviceResidentFusedLWC84CPMLSolver


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/r6_anchored_r54_device_resident_r1_20260905"
OLD_BASE = ROOT / "results/r6_anchored_r54_residual_r1_20260905"
SELECTION = ROOT / "results/frozen_fine_grid_r6_current_env_reattest_v1_20260905/selection_manifest.json"


class MockDeviceTensor:
    def __init__(self, *, shape=(1, 401, 201, 201), dtype=torch.float32, device="cuda",
                 contiguous=True, base=None):
        self.shape = shape; self.dtype = dtype; self.device = SimpleNamespace(type=device)
        self._contiguous = contiguous; self._base = base

    def is_contiguous(self):
        return self._contiguous


def test_device_solver_is_opt_in_and_legacy_simulate_is_inherited_unchanged() -> None:
    assert DeviceResidentFusedLWC84CPMLSolver.simulate is FusedLWC84CPMLSolver.simulate
    assert "simulate_device" in DeviceResidentFusedLWC84CPMLSolver.__dict__
    old = ROOT / "src/fno_acoustic/data_generation/solver_lwc84_fused.py"
    assert __import__("hashlib").sha256(old.read_bytes()).hexdigest() == "6c95433b534fce7e01a2e846bfada0c965b7876225679738c755ade8dad20310"


def test_device_source_recurrence_and_restriction_structure_matches_legacy() -> None:
    old = inspect.getsource(FusedLWC84CPMLSolver.simulate)
    new = inspect.getsource(DeviceResidentFusedLWC84CPMLSolver.simulate_device)
    for token in ("lwc84_startup", "source_terms", "source_sequences", "_time_step",
                  "_saved_interval_block", "graph_runner.replay", "_restrict_output",
                  "bilinear_point_source", "p1[:, 0, :] = 0.0"):
        assert token in old and token in new
    assert ".cpu()" not in new and "wavefield_device=owned_output" in new
    assert "contiguous().clone()" in new


def test_owned_cuda_float32_tensor_contract_with_mocks() -> None:
    assert validate_owned_device_tensor(MockDeviceTensor(), finite_asserted=True)["owns_storage"]
    for kwargs in ({"shape": (1, 1)}, {"dtype": torch.float64}, {"device": "cpu"},
                   {"contiguous": False}, {"base": object()}):
        with pytest.raises(RuntimeError):
            validate_owned_device_tensor(MockDeviceTensor(**kwargs), finite_asserted=True)
    with pytest.raises(RuntimeError):
        validate_owned_device_tensor(MockDeviceTensor(), finite_asserted=False)


def test_device_result_qc_without_materializing_parent() -> None:
    fine = np.full((401, 401), 2000, np.float32); stored = restrict_nodal_2x(fine)
    point_saved = np.zeros((1, 201, 201), np.float32); point_saved[0, 10, 10] = 1
    point_solver = np.zeros((1, 401, 401), np.float32); point_solver[0, 20, 20] = 1
    cfl = 1.01
    result = SimpleNamespace(wavefield_device=MockDeviceTensor(), source_map_saved=point_saved,
                             source_map_solver=point_solver,
                             metrics=[{"finite_asserted_on_device": True, "cfl_2d": cfl,
                                       "lwc_qmax": (2048 / 315) * cfl**2,
                                       "dt_used_s": .000625, "output_frame_count": 401}])
    qc = device_result_qc(result, fine=fine, stored=stored)
    assert qc["passed"] and qc["restriction_max_absolute_difference"] == 0


def test_transfer_ledger_exact_one_final_d2h_and_no_intermediates() -> None:
    ledger = TransferLedger(); ledger.h2d("solver_internal_public_inputs", 100, 5)
    ledger.h2d("static7", STATIC_BYTES)
    for start in range(26):
        count = 1 if start == 25 else 16
        ledger.h2d("time_chunk", count * 4)
    ledger.d2h("candidate", PARENT_BYTES)
    summary = ledger.summary(); validate_transfer_summary(summary)
    assert summary["h2d_call_count"] == 32 and summary["d2h_call_count"] == 1
    assert summary["parent_d2h_call_count"] == summary["correction_d2h_call_count"] == 0
    assert sum(row["bytes"] for row in summary["h2d"] if row["label"] == "time_chunk") == TIME_BYTES
    ledger.d2h("parent", PARENT_BYTES)
    with pytest.raises(RuntimeError):
        validate_transfer_summary(ledger.summary())
    import scripts.gate_r6_anchored_r54_device_runtime as runner
    head_source = inspect.getsource(runner.device_head)
    assert ".item()" not in head_source
    assert "_assert_async" in head_source
    assert head_source.count(".cpu().numpy()") == 1


def test_bit_preserving_zero_head_identity_and_training_gradient_contract() -> None:
    parent = torch.tensor([0.0, -0.0, 2.0], dtype=torch.float32)
    correction = torch.tensor([-0.0, 0.0, 0.0], dtype=torch.float32)
    candidate = bit_preserving_residual_add(parent, correction)
    assert torch.equal(candidate.view(torch.int32), parent.view(torch.int32))
    training_correction = torch.tensor([0.0, 0.5], requires_grad=True)
    (torch.tensor([1.0, 2.0]) + training_correction).sum().backward()
    assert torch.equal(training_correction.grad, torch.ones_like(training_correction))


def test_r6_selection_and_smoke_indices_are_reused_not_copied() -> None:
    records = validate_selection_manifest(json.loads(SELECTION.read_text()))
    assert len(records) == 60 and [records[i]["source_index"] for i in (0, 20, 40)] == [311, 780, 2165]
    assert not (BASE / "selection_manifest.json").exists()


def test_runtime_nearest_rank171_and_boundaries() -> None:
    assert nearest_rank(range(1, 181), .95) == 171
    result = runtime_gate([1.9] * 180, [1.1] * 180, smoke=False)
    assert result["passed"] and result["metrics"]["outer_p95_rank"] == 171
    assert not runtime_gate([1.900001] * 180, [1.0] * 180, smoke=False)["passed"]
    assert runtime_gate([1] * 3, [.5] * 3, smoke=True)["applied"] is False


def test_smoke_and_full_identity_transfer_count_arithmetic() -> None:
    assert 3 * 1 == 3 and 60 * 3 == 180
    assert 32 * 180 == 5760 and PARENT_BYTES * 180 == 11664576720


def test_fresh_head_architecture_digest_unchanged() -> None:
    model, _, state, manifest = fresh_head_cpu()
    assert sum(parameter.numel() for parameter in model.parameters()) == 2791762
    assert state == "e0403fdd967ceb5eb572589f1b5adbdfa69f160b1d322f5e0446d4424fa2d1b8"
    assert manifest == "43e89e1a3cc97ef9a514837ac6a339cca62dc208fcdb9da6424cf242dd640cd9"


def _args(paths, smoke=True):
    return argparse.Namespace(smoke=smoke, full=not smoke, preregistration=Path(paths["preregistration"]),
                              source_h5=Path(paths["source_h5"]), manifest=Path(paths["manifest"]),
                              marmousi=Path(paths["marmousi"]),
                              output=Path(paths["smoke_output" if smoke else "full_output"]))


def test_fixed_output_status_and_full_prerequisite(tmp_path: Path) -> None:
    paths = {key: str(tmp_path / key) for key in
             ("preregistration", "source_h5", "manifest", "marmousi", "smoke_output", "full_output")}
    prereg = {"schema": "r6_anchored_r54_device_runtime_preregistration_v1", "candidate": CANDIDATE,
              "status": "draft_pending_static_audit", "paths": paths,
              "prerequisites": {"smoke": {"status": "pending", "path": paths["smoke_output"], "sha256": None}}}
    Path(paths["preregistration"]).write_text(json.dumps(prereg))
    args = _args(paths); assert bootstrap(args, prereg)[0] == "smoke"
    with pytest.raises(RuntimeError): validate_remaining(args, prereg, "smoke")
    args.output = tmp_path / "override"
    with pytest.raises(RuntimeError): bootstrap(args, prereg)
    assert not args.output.exists()


def test_dependency_manifest_and_fresh_actual_imports() -> None:
    dependency = json.loads((BASE / "runtime_dependency_manifest.json").read_text())
    closure = verify_dependency_manifest(dependency)
    actual = {path.relative_to(ROOT).as_posix() for path in runtime_dependency_closure([ROOT / p for p in dependency["roots"]])}
    assert actual == set(closure)
    code = """
import json,pathlib,sys,numpy as np
import scripts.gate_r6_anchored_r54_device_runtime as runner
runner.fresh_head_cpu()
from scripts import build_r25_coarse_residual_cache as builder
x=np.linspace(0,2000,201,dtype=np.float32)
builder.static_features(np.full((201,201),2000,np.float32),x_m=x,z_m=x,source_x_m=100,source_z_m=100)
root=pathlib.Path.cwd().resolve(); print(json.dumps(sorted({pathlib.Path(m.__file__).resolve().relative_to(root).as_posix() for m in sys.modules.values() if getattr(m,'__file__',None) and str(pathlib.Path(m.__file__).resolve()).startswith(str(root))})))
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
    imported = set(json.loads(completed.stdout.splitlines()[-1]))
    assert imported <= set(closure)


def test_serializer_fixedpoint_and_no_truth_weight_cache_surface(tmp_path: Path) -> None:
    payload = {"status": "invalid", "resources": {"output_bytes": 0}}
    data = report_bytes_fixed_point(payload); output = tmp_path / "report.json"; atomic_write_bytes(data, output)
    assert output.stat().st_size == payload["resources"]["output_bytes"]
    for path in (ROOT / "scripts/gate_r6_anchored_r54_device_runtime.py",
                 ROOT / "src/fno_acoustic/data_generation/solver_lwc84_fused_device.py"):
        source = path.read_text(); tree = ast.parse(source)
        assert not any(isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                       and node.slice.value == "wavefield" for node in ast.walk(tree))
        assert "torch.save" not in source and "torch.load" not in source and "load_state_dict" not in source


def test_failure_serializer_sanitizes_and_literal_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.gate_r6_anchored_r54_device_runtime as runner
    report = {"mode": "smoke", "environment": {"bad": np.nan},
              "preregistration_sha256_before": "a" * 64, "partial_transfer_ledgers": []}
    output = tmp_path / "failure.json"
    write_failure(report, RuntimeError("x"), output, time_started := 0.0, None)
    payload = json.loads(output.read_text())
    assert payload["status"] == "invalid" and payload["environment"]["bad"] is None
    assert output.stat().st_size == payload["resources"]["output_bytes"]
    fallback = tmp_path / "fallback.json"
    monkeypatch.setattr(runner, "report_bytes_fixed_point", lambda *_: (_ for _ in ()).throw(TypeError("forced")))
    write_failure(report, RuntimeError("x"), fallback, time_started, None)
    fallback_payload = json.loads(fallback.read_text())
    assert fallback_payload["error"] == "failure_report_serializer_error"
    assert fallback.stat().st_size == fallback_payload["resources"]["output_bytes"]


def test_master_dag_claim_baseline_and_legacy_files_immutable() -> None:
    master = json.loads((BASE / "master_protocol.json").read_text())
    assert master["immutable_DAG"][0] == "runtime_identity_gate"
    assert master["history"]["prior_host_roundtrip_runtime_report_sha256"] == "2834d64e7dd79eeecf726ea7f4b5aa6d310ed48ef3c4b4667cff9a27680227e7"
    assert "no component causality" in master["bundle_claim"]
    assert __import__("hashlib").sha256((OLD_BASE / "runtime_report.json").read_bytes()).hexdigest() == "2834d64e7dd79eeecf726ea7f4b5aa6d310ed48ef3c4b4667cff9a27680227e7"
