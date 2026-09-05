from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts.audit_r6_device_residual_cache import replay_records, validate_cache_handle
from scripts.build_r6_device_residual_cache import (
    CACHE_SCHEMA, DEVELOPMENT_TIME_INDICES, EXPECTED_COUNTS, EXPECTED_FAMILY,
    EXPECTED_GROUPS, FIT_TIME_INDICES, ROLE_DIGESTS, SUMMARY_SCHEMA, TruthGuard,
    cache_output, canonical_sha256, cleanup_role_temporary, dataset_contract, load_roles, normalized_cache_values,
    predicted_cache_bytes, raw_schema_bytes, role_digests, role_time_indices, shard_records,
    aggregate_worker_terminals, validate_invocation, validate_source_identity,
    quantization_diagnostic, verify_stage_inputs,
)
from scripts.launch_r6_device_residual_cache import child_argv, disk_gate, validate_launch
from scripts.reattest_frozen_fine_grid_r6_train import dependency_closure


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/r6_anchored_r54_device_resident_r1_20260905/cache_stage"
R38 = ROOT / "results/r38_full_coverage_manifest_20260828.json"


class FakeWavefield:
    def __init__(self): self.calls = []
    def __getitem__(self, key):
        self.calls.append(key); index, times = key
        return np.zeros((len(times), 201, 201), np.float32)


class FakeSource:
    def __init__(self): self.wavefield = FakeWavefield()
    def __getitem__(self, key):
        assert key == "wavefield"; return self.wavefield


class FakeArray:
    def __init__(self, values): self.values = values
    def __getitem__(self, index): return self.values[index]
    def asstr(self): return self


class FakeMetadataSource:
    def __init__(self, values): self.values = {key: FakeArray(value) for key, value in values.items()}
    def __getitem__(self, key): return self.values[key]


def test_r38_roles_exact_counts_groups_families_and_digests() -> None:
    roles = load_roles(R38)
    for role in ("fit", "development"):
        assert len(roles[role]) == EXPECTED_COUNTS[role]
        assert len({row["group_id"] for row in roles[role]}) == EXPECTED_GROUPS[role]
        assert {family: sum(row["family"] == family for row in roles[role]) for family in EXPECTED_FAMILY[role]} == EXPECTED_FAMILY[role]
        assert role_digests(roles[role]) == ROLE_DIGESTS[role]
    assert not ({row["group_id"] for row in roles["fit"]} & {row["group_id"] for row in roles["development"]})


def test_time_indices_exact_lists_and_digests() -> None:
    assert role_time_indices("fit") == FIT_TIME_INDICES and len(FIT_TIME_INDICES) == 64
    assert role_time_indices("development") == DEVELOPMENT_TIME_INDICES == tuple(range(401))
    assert FIT_TIME_INDICES[0] == 0 and FIT_TIME_INDICES[-1] == 400


def test_truth_guard_requires_parent_hash_and_one_same_train_read() -> None:
    row = {"source_index": 7, "sample_id": "s", "group_id": "g", "family": "uniform",
           "sample_sha256": "a" * 64, "split": "train"}
    guard = TruthGuard([row], "fit"); source = FakeSource()
    with pytest.raises(PermissionError): guard.read_truth(source, index=7, sample_id="s", split="train", time_indices=[0])
    guard.register_parent(index=7, sample_id="s", digest="b" * 64)
    truth = guard.read_truth(source, index=7, sample_id="s", split="train", time_indices=[0, 2])
    assert truth.shape == (2, 201, 201) and len(source.wavefield.calls) == 1
    with pytest.raises(PermissionError): guard.read_truth(source, index=7, sample_id="s", split="train", time_indices=[0])
    summary = guard.summary(); assert summary["parent_hashed_count"] == summary["truth_read_count"] == 1
    assert summary["transitions"][0]["to"] == "parent_hashed" and summary["transitions"][1]["to"] == "truth_read_once"


def test_truth_guard_rejects_wrong_identity_split_and_unselected() -> None:
    row = {"source_index": 1, "sample_id": "s", "group_id": "g", "family": "uniform",
           "sample_sha256": "a" * 64, "split": "train"}; guard = TruthGuard([row], "fit")
    with pytest.raises(PermissionError): guard.register_parent(index=2, sample_id="s", digest="b" * 64)
    guard.register_parent(index=1, sample_id="s", digest="b" * 64)
    for kwargs in ({"index": 1, "sample_id": "bad", "split": "train"},
                   {"index": 1, "sample_id": "s", "split": "validation"}):
        with pytest.raises(PermissionError): guard.read_truth(FakeSource(), time_indices=[0], **kwargs)


def test_full_source_and_manifest_identity_contract() -> None:
    row = {"source_index": 0, "sample_id": "s", "group_id": "g", "family": "uniform",
           "sample_sha256": "a" * 64, "split": "train"}
    numeric = {"source_x_m": 10.0, "source_z_m": 20.0, "source_f0_hz": 8.0,
               "source_t0_s": .1, "source_amplitude": 1.0}
    source = FakeMetadataSource({"split": ["train"], "medium_type": ["uniform"], "sample_id": ["s"],
                                 "group_id": ["g"], "sample_sha256": ["a" * 64],
                                 "velocity_mps": [np.full((201, 201), 2000, np.float32)],
                                 **{key: [value] for key, value in numeric.items()}})
    manifest = {"split": "train", "medium_type": "uniform", "sample_id": "s", "group_id": "g", **numeric}
    result = validate_source_identity(source, row, manifest)
    assert result["identity"]["sample_sha256"] == "a" * 64
    bad = dict(manifest); bad["group_id"] = "bad"
    with pytest.raises(RuntimeError): validate_source_identity(source, row, bad)


def test_normalization_float16_energy_and_scalar_definitions() -> None:
    parent = np.zeros((401, 201, 201), np.float32); parent[:, 1, 1] = 2
    truth = np.ones((2, 201, 201), np.float32); values = normalized_cache_values(parent, truth, [0, 400])
    assert values["scale"] == 2 and values["parent_norm"].dtype == np.float32
    assert values["truth_frame_energy_max_norm"] == values["truth_frame_energy_mean_norm"] == .25
    expected_error = np.square(values["parent_norm"].astype(np.float64) - values["truth_norm"].astype(np.float64)).sum()
    assert values["baseline_error_square_norm"] == expected_error
    assert values["parent_norm"].astype(np.float16).dtype == np.float16


def test_quantization_uses_actual_hdf_readback_and_float64_norms(tmp_path: Path) -> None:
    parent = np.asarray([0., .3111, .5444, .7777], np.float32)[:, None, None]
    truth = np.asarray([0., .9111, .9444, .9777], np.float32)[:, None, None]
    static = np.full((7, 1, 1), .3333, np.float32)
    with h5py.File(tmp_path / "quantization.h5", "x") as cache:
        for name, value in (("P", parent), ("T", truth), ("static", static)):
            cache.create_dataset(name, data=value.astype(np.float16), compression="lzf", shuffle=True)
        cache.flush()
        qp, qt = cache["P"][:], cache["T"][:]
        result = quantization_diagnostic(parent, truth, qp, qt, static, cache["static"][:], [0, 100, 200, 400])
    residual = truth.astype(np.float64) - parent.astype(np.float64)
    cached = qt.astype(np.float32).astype(np.float64) - qp.astype(np.float32).astype(np.float64)
    truth_norm = np.linalg.norm(truth.astype(np.float64))
    assert result["record"]["R_native"] == pytest.approx(np.linalg.norm(residual) / truth_norm, abs=1e-15)
    assert result["record"]["E_q"] == pytest.approx(np.linalg.norm(cached - residual) / truth_norm, abs=1e-15)
    assert result["gate"]["passed"] and result["static_features_stored_float16_bytes_exact"]
    assert [result["bands"][name]["frame_count"] for name in ("early", "mid", "late")] == [2, 1, 1]


def test_quantization_rejects_residual_lost_despite_small_absolute_error() -> None:
    parent = np.ones((3, 1, 1), np.float32); truth = parent + np.float32(.0001)
    static = np.zeros((7, 1, 1), np.float32)
    result = quantization_diagnostic(parent, truth, parent.astype(np.float16), truth.astype(np.float16),
                                     static, static.astype(np.float16), [0, 200, 400])
    assert result["record"]["E_q"] < 1e-3
    assert result["record"]["E_q_over_R_native"] == 1 and result["record"]["residual_cosine"] is None
    assert not result["gate"]["passed"]
    assert result["gate"]["failure_classification"] == "cache_representation_rejected"


def test_quantization_near_zero_and_zero_truth_are_explicit() -> None:
    parent = np.ones((3, 1, 1), np.float32); truth = parent + np.float32(1e-6)
    static = np.zeros((7, 1, 1), np.float32)
    result = quantization_diagnostic(parent, truth, parent.astype(np.float16), truth.astype(np.float16),
                                     static, static.astype(np.float16), [0, 200, 400])
    assert result["gate"]["passed"] and not result["gate"]["residual_ratio_gate_applies"]
    zero = np.zeros_like(parent)
    result = quantization_diagnostic(zero, zero, zero.astype(np.float16), zero.astype(np.float16),
                                     static, static.astype(np.float16), [0, 200, 400])
    assert result["record"]["truth_norm_zero"] and result["record"]["R_native"] is None
    assert result["record"]["zero_truth_values_preserved"] and result["gate"]["passed"]
    corrupted = zero.astype(np.float16); corrupted[0, 0, 0] = 1e-3
    result = quantization_diagnostic(zero, zero, zero.astype(np.float16), corrupted,
                                     static, static.astype(np.float16), [0, 200, 400])
    assert not result["record"]["zero_truth_values_preserved"] and not result["gate"]["passed"]


def test_quantization_rejects_static_readback_drift() -> None:
    parent = np.zeros((3, 1, 1), np.float32); truth = np.ones_like(parent)
    static = np.zeros((7, 1, 1), np.float32); corrupted = np.ones_like(static, dtype=np.float16)
    result = quantization_diagnostic(parent, truth, parent.astype(np.float16), truth.astype(np.float16),
                                     static, corrupted, [0, 200, 400])
    assert not result["static_features_stored_float16_bytes_exact"] and not result["gate"]["passed"]


def test_dataset_contract_has_sample_hash_float16_lzf_shuffle() -> None:
    contract = dataset_contract("fit", 532)
    assert contract["sample_sha256"]["shape"] == [532]
    for name in ("coarse_norm", "truth_norm", "static_features"):
        assert contract[name]["dtype"] == "float16" and contract[name]["compression"] == "lzf" and contract[name]["shuffle"]
    assert contract["coarse_norm"]["chunks"] == [1, 1, 201, 201]
    assert contract["static_features"]["chunks"] == [1, 7, 201, 201]


def _synthetic_cache(path: Path, role: str = "fit", count: int = 1) -> None:
    contract = dataset_contract(role, count); text = h5py.string_dtype("utf-8")
    with h5py.File(path, "x") as cache:
        cache.attrs.update({"schema": CACHE_SCHEMA, "status": "complete", "role": role, "subset": role,
                            "parent_kind": "R6_device_resident_dt625us_restricted2x",
                            "shard_index": 0, "shard_count": 4,
                            "truth_ledger_json": json.dumps({"truth_read_count": count,
                                "parent_hashed_count": count, "truth_frame_count": count * 64,
                                "truth_raw_bytes": count * 64 * 201 * 201 * 4,
                                "each_truth_once": True, "all_parent_hashes_present_and_64hex": True,
                                "parent_full_sha256_digest": canonical_sha256(["b" * 64] * count),
                                "transitions": [{"to": "parent_hashed", "source_index": 0},
                                                {"to": "truth_read_once", "source_index": 0}] * count})})
        for name, spec in contract.items():
            if spec["dtype"] == "utf8":
                data = [("train:uniform:0" if name == "group_id" else "uniform" if name == "family"
                         else "a" * 64 if name == "sample_sha256" else "b" * 64 if name == "parent_full_sha256"
                         else "train_uniform_00000")] * count
                cache.create_dataset(name, data=np.asarray(data, object), dtype=text)
            else:
                kwargs = {key: (tuple(spec[key]) if key == "chunks" else spec[key])
                          for key in ("chunks", "compression", "shuffle") if key in spec}
                if name == "time_indices": cache.create_dataset(name, data=np.asarray(FIT_TIME_INDICES, np.int64))
                elif name == "time_s": cache.create_dataset(name, data=np.arange(64, dtype=np.float64) * .0025)
                elif name == "field_scale": cache.create_dataset(name, data=np.ones(count, dtype=np.float32))
                else: cache.create_dataset(name, shape=tuple(spec["shape"]), dtype=spec["dtype"], **kwargs)


def test_auditor_schema_validation_on_synthetic_cache(tmp_path: Path) -> None:
    path = tmp_path / "cache.h5"; _synthetic_cache(path)
    with h5py.File(path, "r") as cache:
        result = validate_cache_handle(cache, role="fit", expected_count=1, shard=0)
    assert result["records"][0]["sample_sha256"] == "a" * 64


def test_four_way_shard_ownership_is_exact() -> None:
    roles = load_roles(R38)
    fit = [shard_records(roles["fit"], index) for index in range(4)]
    development = [shard_records(roles["development"], index) for index in range(4)]
    assert [len(rows) for rows in fit] == [532] * 4 and [len(rows) for rows in development] == [14] * 4
    assert len({row["source_index"] for rows in fit for row in rows}) == 2128
    assert cache_output(BASE, "fit", 2, smoke=False).name == "fit_shard_2.h5"


def test_truth_ledger_global_exact_arithmetic() -> None:
    frames = 2128 * 64 + 56 * 401; raw = frames * 201 * 201 * 4
    assert (2128 + 56, frames, raw) == (2184, 158648, 25638151392)


def test_disk_prediction_and_hard_gates() -> None:
    assert raw_schema_bytes() == 26873547168
    predicted = predicted_cache_bytes(); assert predicted == 27477526944 and predicted < 28 * 2**30
    assert disk_gate(predicted, predicted + 40 * 2**30)
    assert not disk_gate(predicted, predicted + 40 * 2**30 - 1)
    assert disk_gate(predicted, 40 * 2**30, after=True)


def test_launcher_worker_argv_and_prerequisites(tmp_path: Path) -> None:
    prereg_path = tmp_path / "pre.json"; runtime = tmp_path / "runtime.json"; smoke = tmp_path / "smoke.json"
    runtime.write_text("runtime"); smoke.write_text("smoke")
    import hashlib
    prereg = {"schema": "r6_device_residual_cache_preregistration_v1",
              "candidate": "r6_anchored_r54_device_resident_r1_20260905", "status": "cache_build_authorized",
              "paths": {"preregistration": str(prereg_path)},
              "prerequisites": {"runtime": {"path": str(runtime), "sha256": hashlib.sha256(b"runtime").hexdigest(), "status": "passed"},
                                "smoke": {"path": str(smoke), "sha256": hashlib.sha256(b"smoke").hexdigest(), "status": "passed"}}}
    prereg_path.write_text(json.dumps(prereg)); validate_launch(prereg, prereg_path)
    argv = child_argv(prereg_path, 3); assert argv[-3:] == ["3", "--device", "cuda:0"] and "--worker" in argv


def test_builder_canonical_prereg_device_status_and_path(tmp_path: Path) -> None:
    prereg_path = tmp_path / "pre.json"; runtime = tmp_path / "runtime.json"; runtime.write_text("runtime")
    import hashlib
    prereg = {"schema": "r6_device_residual_cache_preregistration_v1",
              "candidate": "r6_anchored_r54_device_resident_r1_20260905",
              "status": "runtime_passed_cache_smoke_pending_audit",
              "paths": {"preregistration": str(prereg_path), "stage_dir": str(tmp_path)},
              "prerequisites": {"runtime": {"path": str(runtime), "sha256": hashlib.sha256(b"runtime").hexdigest(), "status": "passed"}}}
    prereg_path.write_text(json.dumps(prereg))
    args = argparse.Namespace(smoke=True, worker=False, preregistration=prereg_path, shard_index=None, device="cuda:0")
    validate_invocation(args, prereg, 0)
    args.device = "cuda:1"
    with pytest.raises(RuntimeError): validate_invocation(args, prereg, 0)


def test_single_strict_four_worker_terminal_aggregator(tmp_path: Path) -> None:
    import hashlib
    stage = tmp_path; (stage / "terminals").mkdir()
    for shard in range(4):
        summaries = []
        for role in ("fit", "development"):
            output = stage / f"{role}_{shard}.h5"; output.write_bytes(f"{role}-{shard}".encode())
            summaries.append({"status": "complete", "output": str(output),
                              "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        terminal = {"status": "complete", "shard_index": shard, "summaries": summaries,
                    "input_hashes_unchanged": True}
        (stage / "terminals" / f"worker_{shard}.json").write_text(json.dumps(terminal))
    assert aggregate_worker_terminals(stage)
    final = json.loads((stage / "build_terminal.json").read_text())
    assert final["status"] == "complete" and final["worker_count"] == 4 and len(final["worker_terminals"]) == 4


def test_role_temporary_is_removed_in_finally(tmp_path: Path) -> None:
    import os
    output = tmp_path / "x.h5"; temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    @cleanup_role_temporary
    def fail(*, output):
        temporary.write_bytes(b"partial")
        raise RuntimeError("forced")
    with pytest.raises(RuntimeError): fail(output=output)
    assert not temporary.exists() and not output.exists()


def test_audit_replay_panel_is_two_per_role_family() -> None:
    panel = replay_records(load_roles(R38)); assert len(panel) == 12
    assert {(role, family): sum(row["family"] == family and row in load_roles(R38)[role] for row in panel)
            for role in ("fit", "development") for family in EXPECTED_FAMILY[role]} == {
                (role, family): 2 for role in ("fit", "development") for family in EXPECTED_FAMILY[role]}


def test_dependency_manifest_and_local_closure() -> None:
    manifest = json.loads((BASE / "dependency_manifest.json").read_text())
    observed = [path.relative_to(ROOT).as_posix() for path in dependency_closure([ROOT / path for path in manifest["roots"]])]
    assert observed == [row["path"] for row in manifest["files"]]
    import hashlib
    for row in manifest["explicit_bindings"]:
        path = Path(row["path"]); path = path if path.is_absolute() else ROOT / path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_real_preregistration_verify_stage_inputs_without_truth_or_cuda(monkeypatch) -> None:
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError("static binding verification must not open HDF5 or CUDA")
    monkeypatch.setattr(h5py, "File", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    prereg = json.loads((BASE / "preregistration.json").read_text())
    assert set(prereg["binding_paths"]) == set(prereg["bindings"])
    verified = verify_stage_inputs(prereg)
    assert verified["explicit_bindings"] == prereg["bindings"]
    assert verified["dependency_closure"]
    broken = dict(prereg); broken["binding_paths"] = dict(prereg["binding_paths"])
    del broken["binding_paths"]["R6_selection"]
    with pytest.raises(RuntimeError, match="cache stage input binding drift"):
        verify_stage_inputs(broken)


def test_single_source_wavefield_index_is_guarded_and_no_sealed_roles() -> None:
    builder = (ROOT / "scripts/build_r6_device_residual_cache.py").read_text(); auditor = (ROOT / "scripts/audit_r6_device_residual_cache.py").read_text()
    count = 0
    for source in (builder, auditor):
        tree = ast.parse(source)
        count += sum(isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "wavefield" for node in ast.walk(tree))
    assert count == 1 and 'source["wavefield"]' in builder
    assert 'source["wavefield"]' not in auditor
    assert "confirmation_opened" in builder and "validation_opened" in builder and "test_id_opened" in builder


def test_master_reference_prior_runtime_and_old_artifacts_unchanged() -> None:
    import hashlib
    master = json.loads((BASE / "master_reference.json").read_text())
    assert master["runtime_prerequisite"]["sha256"] == "6c221ae7b882f53aa0e333c9c191cf76d05bd57dd50b8950535b0a2fce4927b5"
    assert hashlib.sha256((ROOT / "src/fno_acoustic/data_generation/solver_lwc84_fused.py").read_bytes()).hexdigest() == "6c95433b534fce7e01a2e846bfada0c965b7876225679738c755ade8dad20310"
    assert hashlib.sha256((ROOT / "results/r6_anchored_r54_device_resident_r1_20260905/runtime_report.json").read_bytes()).hexdigest() == "6c221ae7b882f53aa0e333c9c191cf76d05bd57dd50b8950535b0a2fce4927b5"
