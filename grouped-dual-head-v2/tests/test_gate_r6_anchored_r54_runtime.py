from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.gate_r6_anchored_r54_runtime import (
    ALLOWED_DATASETS,
    AllowlistedHDF,
    CHUNKS,
    CANDIDATE,
    EXPECTED_SELECTION_DIGESTS,
    SMOKE_INDICES,
    bootstrap_canonical_output,
    bit_preserving_residual_add,
    collect_algorithm_flags,
    configure_algorithm_flags,
    fresh_head_cpu,
    minimal_failure_payload,
    nearest_rank,
    output_identity,
    parent_stability_qc,
    preload_public_record,
    runtime_gates,
    residual_add_identity_metrics,
    runtime_dependency_closure,
    selection_digests,
    validate_chunks,
    validate_parent_result,
    validate_remaining_invocation,
    validate_runtime_completeness,
    validate_selection_manifest,
    validate_time_axis,
    validate_zero_head,
    verify_dependency_manifest,
    target5_preflight,
    write_failure_atomic,
)
from scripts.reattest_frozen_fine_grid_r6_train import (
    atomic_write_bytes,
    report_bytes_fixed_point,
)
from fno_acoustic.data_generation.restriction import restrict_nodal_2x
from fno_acoustic.data_generation.source import bilinear_point_source


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/r6_anchored_r54_residual_r1_20260905"
SELECTION = ROOT / "results/frozen_fine_grid_r6_current_env_reattest_v1_20260905/selection_manifest.json"


class FakeDataset:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]


class FakeHDF:
    def __init__(self, data=None):
        self.data = data or {name: FakeDataset([0]) for name in ALLOWED_DATASETS}

    def __getitem__(self, key):
        return self.data[key]


def test_hdf_allowlist_denies_future_truth_and_ledgers() -> None:
    proxy = AllowlistedHDF(FakeHDF())
    assert proxy["velocity_mps"] is not None
    assert proxy["source_f0_hz"] is not None
    with pytest.raises(PermissionError):
        proxy["wavefield"]
    ledger = proxy.ledger()
    assert ledger["access_counts"] == {"source_f0_hz": 1, "velocity_mps": 1}
    assert ledger["denied_attempts"] == {"wavefield": 1}
    assert ledger["denied_attempt_count"] == 1
    assert ledger["wavefield_access_count"] == 0 and ledger["future_truth_accessed"] is False


def test_r6_selection_schema_digests_census_and_smoke_indices() -> None:
    manifest = json.loads(SELECTION.read_text())
    records = validate_selection_manifest(manifest)
    assert selection_digests(records) == EXPECTED_SELECTION_DIGESTS
    assert [records[position]["source_index"] for position in (0, 20, 40)] == list(SMOKE_INDICES)
    assert len(records) == 60 and len({row["group_id"] for row in records}) == 29
    assert {family: sum(row["family"] == family for row in records) for family in ("uniform", "layered", "marmousi")} == {"uniform": 20, "layered": 20, "marmousi": 20}


def test_public_metadata_selection_and_manifest_identity() -> None:
    row = {"source_index": 0, "sample_id": "s", "group_id": "g", "family": "uniform", "sample_sha256": "a" * 64}
    manifest = {"split": "train", "medium_type": "uniform", "sample_id": "s", "group_id": "g",
                "source_x_m": 10.0, "source_z_m": 20.0, "source_f0_hz": 8.0,
                "source_t0_s": 0.1, "source_amplitude": 1.0}
    data = {"split": FakeDataset([b"train"]), "medium_type": FakeDataset([b"uniform"]),
            "sample_id": FakeDataset([b"s"]), "group_id": FakeDataset([b"g"]),
            "sample_sha256": FakeDataset([b"a" * 64]),
            "velocity_mps": FakeDataset([np.full((201, 201), 2000, np.float32)]),
            "source_x_m": FakeDataset([10.0]), "source_z_m": FakeDataset([20.0]),
            "source_f0_hz": FakeDataset([8.0]), "source_t0_s": FakeDataset([0.1]),
            "source_amplitude": FakeDataset([1.0]), "time_s": FakeDataset([0.0])}
    item = preload_public_record(AllowlistedHDF(FakeHDF(data)), row, manifest)
    assert item["public_identity"]["split"] == "train" and item["stored_velocity"].dtype == np.float32
    bad = dict(row); bad["sample_id"] = "wrong"
    with pytest.raises(RuntimeError):
        preload_public_record(AllowlistedHDF(FakeHDF(data)), bad, manifest)


def test_time_axis_exact_contract() -> None:
    times = np.arange(401, dtype=np.float64) * 0.0025
    qc = validate_time_axis(times)
    assert qc["shape"] == [401] and qc["first_s"] == 0 and qc["last_s"] == 1
    bad = times.copy(); bad[200] += 2e-15
    with pytest.raises(RuntimeError):
        validate_time_axis(bad)


def test_parent_native_restriction_source_stability_and_time_qc() -> None:
    fine = np.full((401, 401), 2000.0, np.float32)
    stored = np.asarray(restrict_nodal_2x(fine), dtype=np.float32)
    point_fine = bilinear_point_source(500.0, 500.0, nx=401, nz=401, dx_m=5.0, dz_m=5.0, centering="node")
    point_saved = bilinear_point_source(500.0, 500.0, nx=201, nz=201, dx_m=10.0, dz_m=10.0, centering="node")
    result = SimpleNamespace(
        wavefield=np.zeros((1, 401, 201, 201), np.float32),
        velocity_saved_mps=stored[None], source_map_saved=point_saved.source_map[None],
        source_map_solver=point_fine.source_map[None],
        metrics=[{"cfl_2d": 0.5, "lwc_qmax": (2048 / 315) * 0.5**2, "dt_used_s": 0.000625}],
    )
    parent, qc = validate_parent_result(result, fine_velocity=fine, stored_velocity=stored,
                                        metadata={"source_x_m": 500.0, "source_z_m": 500.0})
    assert parent.dtype == np.float32 and qc["restriction_max_absolute_difference"] == 0.0
    assert qc["output_frame_count"] == 401 and qc["internal_step_count_to_1s"] == 1600
    assert qc["source_map_saved_sum"] == pytest.approx(1.0, abs=2e-7)
    assert qc["delta_solver_integral"] == pytest.approx(1.0, abs=2e-7)


def test_lwc_qmax_exact_boundary_above_cfl_reporting_and_invalid_values() -> None:
    boundary = 9.6 * (1.0 + 1.0e-12)
    cfl = np.sqrt(boundary / (2048.0 / 315.0))
    assert parent_stability_qc(cfl, boundary)["passed"]
    above = boundary + 1.0e-8
    assert not parent_stability_qc(np.sqrt(above / (2048.0 / 315.0)), above)["passed"]
    cfl_above_one = 1.1
    qmax = (2048.0 / 315.0) * cfl_above_one**2
    result = parent_stability_qc(cfl_above_one, qmax)
    assert result["passed"] and result["cfl_is_gate"] is False
    for bad_cfl, bad_qmax in ((-1.0, 1.0), (1.0, -1.0), (np.nan, 1.0), (1.0, np.inf)):
        assert not parent_stability_qc(bad_cfl, bad_qmax)["passed"]


def test_three_postfailure_observed_stability_values_pass() -> None:
    observed = [
        (1.0161783042420762, 6.713658325819664),
        (0.5699612544250984, 2.1120785492249627),
        (0.7866562940700341, 4.023365079365079),
    ]
    results = [parent_stability_qc(cfl, qmax) for cfl, qmax in observed]
    assert all(row["passed"] and row["relation_passed"] for row in results)
    assert results[0]["cfl_2d"] > 1.0


def test_fresh_architecture_zero_heads_digests_and_parameters() -> None:
    first, _, state, manifest = fresh_head_cpu()
    second, _, state2, manifest2 = fresh_head_cpu()
    assert state == state2 == "e0403fdd967ceb5eb572589f1b5adbdfa69f160b1d322f5e0446d4424fa2d1b8"
    assert manifest == manifest2 == "43e89e1a3cc97ef9a514837ac6a339cca62dc208fcdb9da6424cf242dd640cd9"
    assert sum(parameter.numel() for parameter in first.parameters()) == 2791762
    assert validate_zero_head(first) == {"passed": True, "zero_tensor_count": 4, "tensor_count": 4}


def test_chunks_exact_26_no_gap_repeat() -> None:
    validate_chunks()
    assert len(CHUNKS) == 26 and CHUNKS[0] == (0, 16) and CHUNKS[-1] == (400, 401)
    assert [index for start, stop in CHUNKS for index in range(start, stop)] == list(range(401))


def test_identity_scale_raw_add_and_signed_zero() -> None:
    parent = np.zeros((401, 201, 201), np.float32); parent[:, 1, 1] = 1.5
    correction = np.zeros_like(parent); candidate = (parent + correction).astype(np.float32)
    times = np.linspace(0, 1, 401, dtype=np.float64)
    identity = output_identity(parent, correction, candidate, t0=0.2, times=times)
    assert identity["candidate_equals_parent_bytes"] and identity["correction_all_zero"]
    negative = np.copysign(np.zeros_like(correction), -1.0)
    identity2 = output_identity(parent, negative, (parent + negative).astype(np.float32), t0=0.2, times=times)
    assert identity2["correction_signed_zero_count"] == negative.size


def test_bit_preserving_residual_add_preserves_plus_minus_zero_and_mixed_addition() -> None:
    parent = torch.tensor([0.0, -0.0, 1.5, -2.0], dtype=torch.float32)
    correction = torch.tensor([-0.0, 0.0, 0.25, -0.5], dtype=torch.float32)
    candidate = bit_preserving_residual_add(parent, correction)
    assert torch.equal(candidate[:2].view(torch.int32), parent[:2].view(torch.int32))
    assert torch.equal(candidate[2:], (parent + correction)[2:])
    metrics = residual_add_identity_metrics(parent.numpy(), correction.numpy(), candidate.numpy())
    assert metrics["zero_position_bits_preserved"]
    assert metrics["nonzero_additive_numeric_equal"] and metrics["nonzero_additive_bit_equal"]
    assert metrics["full_identity"] is False


def test_bit_preserving_residual_add_rejects_invalid_contracts() -> None:
    value = torch.zeros(2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        bit_preserving_residual_add(value, torch.zeros(3, dtype=torch.float32))
    with pytest.raises(RuntimeError):
        bit_preserving_residual_add(value, torch.zeros(2, dtype=torch.float64))
    with pytest.raises(RuntimeError):
        bit_preserving_residual_add(value, torch.zeros(2, device="meta", dtype=torch.float32))
    with pytest.raises(RuntimeError):
        bit_preserving_residual_add(value, torch.tensor([0.0, np.inf], dtype=torch.float32))


def test_full_identity_requires_all_zero_correction_and_training_add_keeps_gradient() -> None:
    parent = np.zeros((401, 201, 201), np.float32); parent[:, 1, 1] = 1.0
    correction = np.zeros_like(parent)
    candidate = bit_preserving_residual_add(torch.from_numpy(parent), torch.from_numpy(correction)).numpy()
    assert output_identity(parent, correction, candidate, t0=0.1,
                           times=np.linspace(0, 1, 401, dtype=np.float64))["full_identity"]
    correction[:, 1, 1] = 0.25
    mixed = bit_preserving_residual_add(torch.from_numpy(parent), torch.from_numpy(correction)).numpy()
    assert residual_add_identity_metrics(parent, correction, mixed)["full_identity"] is False
    with pytest.raises(RuntimeError):
        output_identity(parent, correction, mixed, t0=0.1, times=np.linspace(0, 1, 401, dtype=np.float64))
    parent_training = torch.tensor([1.0, -2.0])
    correction_training = torch.tensor([0.0, 0.5], requires_grad=True)
    ordinary_training_candidate = parent_training + correction_training
    ordinary_training_candidate.sum().backward()
    assert torch.equal(correction_training.grad, torch.ones_like(correction_training))


def _repeat(digest: str = "a" * 64):
    return {"identity": {"parent_sha256": digest, "correction_sha256": "b" * 64,
                         "candidate_sha256": digest}, "parent_qc": {"passed": True},
            "outer_runtime_s": 1.0, "head_runtime_s": 0.5}


def test_hard_smoke_and_full_counts_hash_consistency() -> None:
    smoke = [{"repeats": [_repeat()]} for _ in range(3)]
    assert validate_runtime_completeness(smoke, mode="smoke")["identity_count"] == 3
    full = [{"repeats": [_repeat(), _repeat(), _repeat()]} for _ in range(60)]
    result = validate_runtime_completeness(full, mode="full")
    assert result["identity_count"] == result["raw_outer_runtime_count"] == 180
    full[0]["repeats"][2] = _repeat("c" * 64)
    with pytest.raises(RuntimeError):
        validate_runtime_completeness(full, mode="full")


def test_runtime_rank171_gate_boundaries_and_raw_counts() -> None:
    assert nearest_rank(list(range(1, 181)), 0.95) == 171
    result = runtime_gates([1.9] * 180, [1.1] * 180, smoke=False)
    assert result["metrics"]["outer_p95_rank"] == 171 and result["passed"]
    assert runtime_gates([1.0] * 3, [0.5] * 3, smoke=True)["raw_outer_count"] == 3
    with pytest.raises(RuntimeError):
        runtime_gates([1.0], [0.5], smoke=True)


def test_timing_scope_has_nested_sync_and_repeat_rebuild() -> None:
    import scripts.gate_r6_anchored_r54_runtime as runner
    outer_source = inspect.getsource(runner.execute_repeat)
    head_source = inspect.getsource(runner.inference_head)
    assert outer_source.count("torch.cuda.synchronize(device)") == 2
    assert head_source.count("torch.cuda.synchronize(device)") == 2
    for token in ("_fine_velocity", "solver.simulate", "validate_parent_result", "inference_head"):
        assert token in outer_source
    for token in ("static_features", "scale =", "candidate_gpu", ".cpu().numpy()"):
        assert token in head_source
    assert "bit_preserving_residual_add" in head_source


def test_target5_audit_precedes_hdf_and_cuda_and_requires_pass() -> None:
    calls = []
    prereg = {"paths": {"reference": "/tmp/reference.json"}}
    result = target5_preflight(
        prereg,
        audit_function=lambda workspace, reference: calls.append((workspace, reference))
        or {"passed": True, "status": "target_met", "checks": []},
    )
    assert result["status"] == "target_met" and len(calls) == 1
    with pytest.raises(RuntimeError):
        target5_preflight(prereg, audit_function=lambda *_: {"passed": False, "status": "audit_failed"})
    import scripts.gate_r6_anchored_r54_runtime as runner
    source = inspect.getsource(runner.main)
    assert source.index('report["target5_audit"] = target5_preflight') < source.index("torch.cuda.set_device")
    assert source.index('report["target5_audit"] = target5_preflight') < source.index("h5py.File")


def test_environment_tf32_contract() -> None:
    prereg = json.loads((BASE / "runtime_preregistration.json").read_text())
    assert prereg["environment"]["tf32_matmul"] is False and prereg["environment"]["tf32_cudnn"] is False
    assert prereg["environment"]["CUBLAS_WORKSPACE_CONFIG"] is None
    assert prereg["environment"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_configure_algorithm_flags_records_before_after_and_preserves_defaults() -> None:
    prereg = json.loads((BASE / "runtime_preregistration.json").read_text())
    original = collect_algorithm_flags()
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        audit = configure_algorithm_flags(prereg["environment"])
        assert audit["observed_before"]["tf32_matmul"] is True
        assert audit["observed_before"]["tf32_cudnn"] is True
        assert audit["configured_after"] == audit["expected"]
        assert all(value is False for value in audit["configured_after"].values())
    finally:
        torch.use_deterministic_algorithms(original["deterministic"])
        torch.backends.cudnn.benchmark = original["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = original["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = original["tf32_matmul"]
        torch.backends.cudnn.allow_tf32 = original["tf32_cudnn"]


def test_fresh_subprocess_exact_env_configures_default_tf32_before_hdf_cuda() -> None:
    code = """
import json, torch, h5py
from scripts.gate_r6_anchored_r54_runtime import configure_algorithm_flags
calls={"hdf":0,"set_device":0,"init":0,"reset":0}
h5py.File=lambda *a,**k: calls.__setitem__("hdf",calls["hdf"]+1)
torch.cuda.set_device=lambda *a,**k: calls.__setitem__("set_device",calls["set_device"]+1)
torch.cuda.init=lambda *a,**k: calls.__setitem__("init",calls["init"]+1)
torch.cuda.reset_peak_memory_stats=lambda *a,**k: calls.__setitem__("reset",calls["reset"]+1)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=False
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
expected={"deterministic":False,"cudnn_benchmark":False,"cudnn_deterministic":False,"tf32_matmul":False,"tf32_cudnn":False}
audit=configure_algorithm_flags(expected)
print(json.dumps({"audit":audit,"calls":calls}))
"""
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                               check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload["audit"]["observed_before"]["tf32_cudnn"] is True
    assert payload["audit"]["configured_after"] == payload["audit"]["expected"]
    assert payload["calls"] == {"hdf": 0, "set_device": 0, "init": 0, "reset": 0}


def _args(paths, *, smoke=True):
    return argparse.Namespace(smoke=smoke, full=not smoke, preregistration=Path(paths["preregistration"]),
                              source_h5=Path(paths["source_h5"]), manifest=Path(paths["manifest"]),
                              marmousi=Path(paths["marmousi"]),
                              output=Path(paths["smoke_output" if smoke else "full_output"]))


def test_canonical_output_first_then_status_paths_and_prerequisite(tmp_path: Path) -> None:
    paths = {key: str(tmp_path / key) for key in
             ("preregistration", "source_h5", "manifest", "marmousi", "smoke_output", "full_output")}
    prereg = {"schema": "r6_anchored_r54_runtime_preregistration_v1", "candidate": CANDIDATE,
              "status": "draft_pending_static_audit", "paths": paths,
              "prerequisites": {"smoke": {"status": "pending", "path": paths["smoke_output"], "sha256": None}}}
    Path(paths["preregistration"]).write_text(json.dumps(prereg))
    args = _args(paths)
    assert bootstrap_canonical_output(args, prereg) == ("smoke", Path(paths["smoke_output"]).resolve())
    with pytest.raises(RuntimeError):
        validate_remaining_invocation(args, prereg, "smoke")
    args.output = tmp_path / "bad-output"
    with pytest.raises(RuntimeError):
        bootstrap_canonical_output(args, prereg)
    assert not args.output.exists() and not Path(paths["smoke_output"]).exists()


def test_failure_payload_sanitized_atomic_fixedpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "failure.json"
    report = {"schema": "x", "candidate": CANDIDATE, "mode": "smoke",
              "observed_environment": {"value": np.float64(np.nan)}, "input_hashes_before": {"a": "b"}}
    payload = minimal_failure_payload(report=report, error=RuntimeError("x"), started=0.0, proxy=None)
    assert payload["observed_environment"]["value"] is None
    write_failure_atomic(report=report, error=RuntimeError("x"), started=0.0, proxy=None, output=output)
    decoded = json.loads(output.read_text())
    assert output.stat().st_size == decoded["resources"]["output_bytes"] and decoded["status"] == "invalid"
    assert output.stat().st_size < 64 * 2**20
    import scripts.gate_r6_anchored_r54_runtime as runner
    fallback = tmp_path / "fallback.json"
    monkeypatch.setattr(runner, "report_bytes_fixed_point", lambda payload: (_ for _ in ()).throw(TypeError("forced")))
    write_failure_atomic(report={**report, "preregistration_sha256_before": "a" * 64},
                         error=RuntimeError("x"), started=0.0, proxy=None, output=fallback)
    fallback_payload = json.loads(fallback.read_text())
    assert fallback_payload["preregistration_sha256_before"] == "a" * 64
    assert {"input_hashes_before", "observed_environment", "hdf_ledger"} <= set(fallback_payload)
    assert fallback_payload["resources"]["peak_allocated_bytes"] == 0


def test_dependency_manifest_fresh_dynamic_import_coverage() -> None:
    dependency = json.loads((BASE / "runtime_dependency_manifest.json").read_text())
    closure = verify_dependency_manifest(dependency)
    code = """
import json, pathlib, sys, numpy as np
import scripts.gate_r6_anchored_r54_runtime as runner
runner.fresh_head_cpu()
from scripts import build_r25_coarse_residual_cache as builder
x=np.linspace(0,2000,201,dtype=np.float32)
builder.static_features(np.full((201,201),2000,np.float32),x_m=x,z_m=x,source_x_m=100,source_z_m=100)
root=pathlib.Path.cwd().resolve()
files=sorted({pathlib.Path(m.__file__).resolve().relative_to(root).as_posix() for m in sys.modules.values() if getattr(m,'__file__',None) and str(pathlib.Path(m.__file__).resolve()).startswith(str(root))})
print(json.dumps(files))
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
    imported = set(json.loads(completed.stdout.splitlines()[-1]))
    covered = set(closure)
    assert imported <= covered
    for path in ("scripts/gate_r6_anchored_r54_runtime.py", "scripts/train_r25_coarse_residual_operator.py",
                 "scripts/train_r26_tail_spectral_pilot.py", "scripts/build_r25_coarse_residual_cache.py",
                 "saved_time_phase_operator_v4/coarse_lwc84.py"):
        assert path in covered


def test_dependency_closure_matches_manifest() -> None:
    dependency = json.loads((BASE / "runtime_dependency_manifest.json").read_text())
    actual = [path.relative_to(ROOT).as_posix() for path in runtime_dependency_closure([ROOT / path for path in dependency["roots"]])]
    assert actual == [row["path"] for row in dependency["files"]]


def test_serializer_and_ast_forbid_truth_weights_cache_and_prediction_outputs(tmp_path: Path) -> None:
    payload = {"status": "invalid", "resources": {"output_bytes": 0}}
    data = report_bytes_fixed_point(payload); path = tmp_path / "x.json"; atomic_write_bytes(data, path)
    assert path.stat().st_size == payload["resources"]["output_bytes"]
    source = (ROOT / "scripts/gate_r6_anchored_r54_runtime.py").read_text(); tree = ast.parse(source)
    assert not any(isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                   and node.slice.value == "wavefield" for node in ast.walk(tree))
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in {"load", "load_state_dict", "save"}:
                forbidden_calls.append(name)
    assert forbidden_calls == []
    assert "--weights" not in source and "--checkpoint" not in source and "predictions.npy" not in source


def test_prereg_r54_best_is_historical_protected_forbidden_input() -> None:
    prereg = json.loads((BASE / "runtime_preregistration.json").read_text())
    item = prereg["historical_artifacts"]["R54_best"]
    assert item["role"] == "historical_protected_forbidden_input"
    assert item["runtime_input"] is False


def test_cfl_binding_and_attempt_paths_are_frozen() -> None:
    prereg = json.loads((BASE / "runtime_preregistration.json").read_text())
    assert prereg["binding_paths"]["cfl"] == "src/fno_acoustic/data_generation/cfl.py"
    assert prereg["bindings"]["cfl"] == "86f57dadf59dfe3e79a0ff4b7dda93d48ada0ef97bc69f56fbdcc0a16d641aec"
    assert prereg["paths"]["smoke_output"].endswith("runtime_smoke_attempt_003.json")
    assert prereg["smoke"]["output"] == prereg["paths"]["smoke_output"]
    assert prereg["prerequisites"]["smoke"]["path"] == prereg["paths"]["smoke_output"]
    assert prereg["attempt_history"][0]["report_sha256"] == "4d8a6dd99e0836f2e8240c39786c4e60718fa053d7518ac380f68ba53fb9edf6"
    assert prereg["attempt_history"][1]["report_sha256"] == "9d79c477dc72b711141d86a273c1309585059acdc1aeb1be816d6481e7c3160d"
    assert prereg["attempt_history"][1]["signed_zero_cause"].startswith("inferred")


def test_master_claim_and_immutable_dag() -> None:
    master = json.loads((BASE / "master_protocol.json").read_text())
    assert "no claim" in master["bundle_claim"].lower()
    assert master["immutable_DAG"] == ["runtime_identity_gate", "R6_cache_build", "R6_cache_audit", "DDP4_smoke",
                                        "same-lineage_3ep_pilot", "resume_epoch40", "new_synthetic_confirmation"]
