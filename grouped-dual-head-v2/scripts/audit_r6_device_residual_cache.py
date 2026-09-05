#!/usr/bin/env python3
"""Audit all R6 device-residual cache shards and replay twelve train records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if value not in sys.path: sys.path.insert(0, value)

from build_r6_device_residual_cache import (  # noqa: E402
    CACHE_SCHEMA, CANDIDATE, EXPECTED_COUNTS, EXPECTED_FAMILY, EXPECTED_GROUPS,
    ROLE_DIGESTS, SUMMARY_SCHEMA, TruthGuard, cache_output, canonical_sha256, dataset_contract,
    load_roles, normalized_cache_values, predicted_cache_bytes, role_digests,
    role_time_indices, shard_records, validate_environment, validate_source_identity, verify_stage_inputs,
)
from build_r25_coarse_residual_cache import static_features  # noqa: E402
from gate_lwc84_cuda_graph_fine_grid_trainonly import _fine_velocity, _read_manifest_rows  # noqa: E402
from gate_r6_anchored_r54_runtime import target5_preflight  # noqa: E402
from reattest_frozen_fine_grid_r6_train import atomic_write_bytes, report_bytes_fixed_point, sha256  # noqa: E402
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84_fused_device import DeviceResidentFusedLWC84CPMLSolver  # noqa: E402


def validate_cache_handle(cache: h5py.File, *, role: str, expected_count: int, shard: int,
                          expected_time_s: np.ndarray | None = None) -> dict[str, Any]:
    if str(cache.attrs.get("schema")) != CACHE_SCHEMA or str(cache.attrs.get("status")) != "complete":
        raise RuntimeError("cache schema/status drift")
    if str(cache.attrs.get("role")) != role or str(cache.attrs.get("subset")) != role:
        raise RuntimeError("cache role drift")
    if int(cache.attrs.get("shard_index", -1)) != shard or int(cache.attrs.get("shard_count", -1)) != 4:
        raise RuntimeError("cache shard attrs drift")
    if str(cache.attrs.get("parent_kind")) != "R6_device_resident_dt625us_restricted2x":
        raise RuntimeError("cache parent provenance drift")
    contract = dataset_contract(role, expected_count)
    if set(cache.keys()) != set(contract): raise RuntimeError("cache dataset surface drift")
    for name, expected in contract.items():
        dataset = cache[name]
        if list(dataset.shape) != expected["shape"]: raise RuntimeError(f"{name} shape drift")
        if expected["dtype"] != "utf8" and str(dataset.dtype) != expected["dtype"]: raise RuntimeError(f"{name} dtype drift")
        if "chunks" in expected and list(dataset.chunks or ()) != expected["chunks"]: raise RuntimeError(f"{name} chunks drift")
        if expected.get("compression") is not None and dataset.compression != expected["compression"]: raise RuntimeError(f"{name} compression drift")
        if "shuffle" in expected and bool(dataset.shuffle) != expected["shuffle"]: raise RuntimeError(f"{name} shuffle drift")
    ledger = json.loads(str(cache.attrs["truth_ledger_json"]))
    expected_frames = expected_count * len(role_time_indices(role)); expected_raw = expected_frames * 201 * 201 * 4
    if (ledger["truth_read_count"] != expected_count or ledger["parent_hashed_count"] != expected_count
            or ledger["truth_frame_count"] != expected_frames or ledger["truth_raw_bytes"] != expected_raw
            or not ledger["each_truth_once"] or not ledger["all_parent_hashes_present_and_64hex"]):
        raise RuntimeError("cache truth ledger drift")
    records = [{"source_index": int(cache["source_index"][i]), "sample_id": str(cache["sample_id"].asstr()[i]),
                "group_id": str(cache["group_id"].asstr()[i]), "family": str(cache["family"].asstr()[i]),
                "sample_sha256": str(cache["sample_sha256"].asstr()[i]), "split": "train"}
               for i in range(expected_count)]
    expected_indices = np.asarray(role_time_indices(role), np.int64)
    if not np.array_equal(cache["time_indices"][:], expected_indices): raise RuntimeError("cache time index drift")
    times = np.asarray(cache["time_s"][:], np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0): raise RuntimeError("cache time axis finite/order drift")
    if expected_time_s is not None and not np.array_equal(times, expected_time_s[expected_indices]):
        raise RuntimeError("cache time values differ from source metadata")
    hashes = [str(value) for value in cache["parent_full_sha256"].asstr()[:]]
    if any(len(value) != 64 for value in hashes) or canonical_sha256(hashes) != ledger["parent_full_sha256_digest"]:
        raise RuntimeError("cache parent hash dataset/ledger drift")
    if len({row["source_index"] for row in records}) != expected_count or any(len(row["sample_sha256"]) != 64 for row in records):
        raise RuntimeError("cache metadata uniqueness/hash drift")
    for name in ("field_scale", "source_f0_hz", "source_t0_s", "truth_frame_energy_max_norm",
                 "truth_frame_energy_mean_norm", "baseline_error_square_norm", "target_square_norm"):
        if not np.isfinite(cache[name][:]).all(): raise RuntimeError(f"cache scalar nonfinite: {name}")
    if np.any(cache["field_scale"][:] <= 0): raise RuntimeError("cache field scale nonpositive")
    for index in range(expected_count):
        for name in ("coarse_norm", "truth_norm", "static_features"):
            if not np.isfinite(cache[name][index]).all(): raise RuntimeError(f"cache tensor nonfinite: {name}")
    transitions = ledger.get("transitions", [])
    if len(transitions) != 2 * expected_count:
        raise RuntimeError("cache truth transition count drift")
    for position, row in enumerate(records):
        pair = transitions[2 * position:2 * position + 2]
        if ([item.get("to") for item in pair] != ["parent_hashed", "truth_read_once"]
                or any(item.get("source_index") != row["source_index"] for item in pair)):
            raise RuntimeError("cache parent-before-truth transition order drift")
    return {"role": role, "shard": shard, "records": records, "ledger": ledger,
            "parent_full_sha256": hashes, "time_indices_sha256": canonical_sha256(expected_indices.tolist())}


def replay_records(roles: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for role in ("fit", "development") for family in EXPECTED_FAMILY[role]
            for row in [item for item in roles[role] if item["family"] == family][:2]]


def audit_all(prereg: Mapping[str, Any]) -> dict[str, Any]:
    stage = ROOT / prereg["paths"]["stage_dir"]; roles = load_roles(ROOT / prereg["paths"]["R38_manifest"])
    terminal = stage / "build_terminal.json"
    terminal_payload = json.loads(terminal.read_text()) if terminal.is_file() else {}
    if terminal_payload.get("status") != "complete" or terminal_payload.get("worker_count") != 4:
        raise RuntimeError("complete build terminal missing")
    expected_terminals = [stage / "terminals" / f"worker_{index}.json" for index in range(4)]
    if len(terminal_payload.get("worker_terminals", [])) != 4:
        raise RuntimeError("build terminal worker census drift")
    for index, item in enumerate(terminal_payload["worker_terminals"]):
        if Path(item["path"]) != expected_terminals[index] or sha256(expected_terminals[index]) != item["sha256"]:
            raise RuntimeError("build terminal worker hash drift")
    with h5py.File(prereg["paths"]["source_h5"], "r", swmr=True) as source:
        expected_time_s = np.asarray(source["time_s"][:], np.float64)
    if expected_time_s.shape != (401,) or not np.isfinite(expected_time_s).all():
        raise RuntimeError("source time metadata drift")
    combined = {"fit": [], "development": []}; ledgers = []; files = []
    for shard in range(4):
        for role in ("fit", "development"):
            path = cache_output(stage, role, shard, smoke=False); summary_path = path.with_suffix(".summary.json")
            if not path.is_file() or not summary_path.is_file(): raise RuntimeError("cache shard/summary missing")
            with h5py.File(path, "r", swmr=True) as cache:
                result = validate_cache_handle(cache, role=role, expected_count=532 if role == "fit" else 14,
                                               shard=shard, expected_time_s=expected_time_s)
            summary = json.loads(summary_path.read_text())
            if (summary.get("schema") != SUMMARY_SCHEMA or summary.get("status") != "complete"
                    or summary.get("role") != role or summary.get("record_count") != (532 if role == "fit" else 14)
                    or summary.get("frame_count_per_record") != len(role_time_indices(role))
                    or Path(summary.get("output")) != path or summary.get("output_sha256") != sha256(path)
                    or summary.get("output_bytes") != path.stat().st_size
                    or summary.get("truth_ledger") != result["ledger"]):
                raise RuntimeError("cache summary/hash drift")
            expected_shard = shard_records(roles[role], shard)
            if result["records"] != expected_shard: raise RuntimeError("cache shard ownership/order drift")
            combined[role].extend(result["records"]); ledgers.append(result["ledger"])
            files.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                          "summary_path": str(summary_path), "summary_sha256": sha256(summary_path)})
    for role in combined:
        combined[role].sort(key=lambda row: roles[role].index(row))
        if role_digests(combined[role]) != ROLE_DIGESTS[role]: raise RuntimeError("combined role digest drift")
    reads = sum(row["truth_read_count"] for row in ledgers); frames = sum(row["truth_frame_count"] for row in ledgers); raw = sum(row["truth_raw_bytes"] for row in ledgers)
    if (reads, frames, raw) != (2184, 158648, 25638151392): raise RuntimeError("aggregate truth ledger drift")
    total_bytes = sum(row["bytes"] for row in files)
    if sorted((stage / "shards").glob("*.h5")) != sorted(cache_output(stage, role, shard, smoke=False) for shard in range(4) for role in ("fit", "development")):
        raise RuntimeError("unexpected or missing HDF shard output")
    if total_bytes > 28 * 2**30 or shutil.disk_usage(stage).free < 40 * 2**30: raise RuntimeError("cache disk gate")
    return {"files": files, "role_digests": {role: role_digests(rows) for role, rows in combined.items()},
            "truth_ledger": {"reads": reads, "frames": frames, "raw_bytes": raw,
                             "confirmation_opened": False, "validation_opened": False, "test_id_opened": False},
            "cache_bytes": total_bytes, "predicted_cache_bytes": predicted_cache_bytes(),
            "replay_selection": replay_records(roles), "build_terminal_sha256": sha256(terminal)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--preregistration", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); prereg = json.loads(args.preregistration.read_text()); output = (ROOT / prereg["paths"]["audit_output"]).resolve(); started = time.time()
    if args.output.resolve() != output or output.exists(): raise RuntimeError("fixed audit output override/existence")
    if (prereg.get("schema") != "r6_device_residual_cache_preregistration_v1"
            or prereg.get("candidate") != CANDIDATE or prereg.get("status") != "cache_built_audit_pending"
            or args.preregistration.resolve() != (ROOT / prereg["paths"]["preregistration"]).resolve()):
        raise RuntimeError("cache audit preregistration/status/path rejected")
    build = prereg["prerequisites"]["build"]; build_path = ROOT / build["path"]
    if build.get("status") != "passed" or sha256(build_path) != build.get("sha256"):
        raise RuntimeError("cache build prerequisite is not frozen/passed")
    environment = validate_environment(prereg, "0"); before = verify_stage_inputs(prereg)
    target = target5_preflight(prereg)
    result = audit_all(prereg)
    # Replay is deliberately GPU/train-truth gated and runs only in the separately authorized audit stage.
    device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
    grid = AcousticGrid(nx=401, nz=401, dx_m=5, dz_m=5, lx_m=2000, lz_m=2000, centering="node")
    records = result["replay_selection"]; guard = TruthGuard(records, "audit_replay")
    with h5py.File(prereg["paths"]["source_h5"], "r", swmr=True) as source:
        times = np.asarray(source["time_s"][:], np.float64); manifest = _read_manifest_rows(Path(prereg["paths"]["source_manifest"]), {r["sample_id"] for r in records})
        solver = DeviceResidentFusedLWC84CPMLSolver(grid=grid, boundaries=BoundaryConfig(npml=40), dt_s=.000625,
                                                     output_times_s=times, c_ref_mps=6750, device=device, dtype=torch.float32,
                                                     kappa_max=3, minimum_frequency_hz=8, output_restriction_factor=2, cuda_graphs=True)
        solver.warmup(batch=1); replay = []
        for row in records:
            index = row["source_index"]; role = "fit" if row in load_roles(ROOT / prereg["paths"]["R38_manifest"])["fit"] else "development"; indices = role_time_indices(role)
            identity = validate_source_identity(source, row, manifest[row["sample_id"]])
            stored = identity["stored_velocity"]; metadata = identity["metadata"]
            fine, _ = _fine_velocity(family=row["family"], manifest_row=manifest[row["sample_id"]], grid=grid, marmousi_npy=Path(prereg["paths"]["marmousi"]))
            parent = solver.simulate_device(fine, **metadata).wavefield_device.detach().cpu().numpy()[0]; digest = hashlib.sha256(parent.tobytes()).hexdigest(); guard.register_parent(index=index, sample_id=row["sample_id"], digest=digest)
            truth = guard.read_truth(source, index=index, sample_id=row["sample_id"], split="train", time_indices=indices); values = normalized_cache_values(parent, truth, indices)
            shard = next(i for i in range(4) if row in shard_records(load_roles(ROOT / prereg["paths"]["R38_manifest"])[role], i)); local = shard_records(load_roles(ROOT / prereg["paths"]["R38_manifest"])[role], shard).index(row)
            path = cache_output(ROOT / prereg["paths"]["stage_dir"], role, shard, smoke=False)
            with h5py.File(path, "r", swmr=True) as cache:
                axis = np.linspace(0, 2000, 201, dtype=np.float32)
                expected_static = static_features(stored, x_m=axis, z_m=axis,
                                                  source_x_m=metadata["source_x_m"],
                                                  source_z_m=metadata["source_z_m"]).astype(np.float16)
                exact = (cache["coarse_norm"][local].tobytes() == values["parent_norm"].astype(np.float16).tobytes()
                         and cache["truth_norm"][local].tobytes() == values["truth_norm"].astype(np.float16).tobytes()
                         and cache["static_features"][local].tobytes() == expected_static.tobytes())
                scalars = (all(abs(float(cache[name][local]) - float(values[name])) <= 1e-12
                               for name in ("baseline_error_square_norm", "target_square_norm"))
                           and float(cache["field_scale"][local]) == float(np.float32(values["scale"]))
                           and float(cache["source_f0_hz"][local]) == float(np.float32(metadata["source_f0_hz"]))
                           and float(cache["source_t0_s"][local]) == float(np.float32(metadata["source_t0_s"]))
                           and float(cache["truth_frame_energy_max_norm"][local]) == float(np.float32(values["truth_frame_energy_max_norm"]))
                           and float(cache["truth_frame_energy_mean_norm"][local]) == float(np.float32(values["truth_frame_energy_mean_norm"])))
            if not exact or not scalars: raise RuntimeError("cache replay mismatch")
            replay.append({"role": role, "source_index": index, "parent_sha256": digest,
                           "cached_parent_truth_static_bytes_exact": exact,
                           "scalars_tolerance_or_exact_cast_passed": scalars})
    after = verify_stage_inputs(prereg)
    if before != after: raise RuntimeError("cache audit input hash drift")
    replay_ledger = guard.summary()
    if (len(replay) != 12 or replay_ledger["truth_read_count"] != 12
            or replay_ledger["truth_frame_count"] != 2790 or not replay_ledger["each_truth_once"]):
        raise RuntimeError("twelve-record replay truth ledger drift")
    elapsed = time.time() - started; peak = int(torch.cuda.max_memory_allocated(device))
    if elapsed > 600 or peak >= 8 * 2**30: raise RuntimeError("cache audit resource budget")
    payload = {"schema": "r6_device_residual_cache_audit_v1", "status": "passed", "candidate": CANDIDATE,
               **result, "replay": replay, "replay_truth_ledger": replay_ledger, "promotion_authorized": False,
               "environment": environment, "target5_audit": target,
               "input_hashes_before": before, "input_hashes_after": after, "input_hashes_unchanged": True,
               "validation_opened": False, "test_id_opened": False,
               "resources": {"elapsed_seconds": elapsed, "peak_allocated_bytes": peak, "output_bytes": 0}}
    data = report_bytes_fixed_point(payload)
    if len(data) >= 64 * 2**20: raise RuntimeError("cache audit report budget")
    atomic_write_bytes(data, output); return 0


def entrypoint() -> int:
    try:
        return main()
    except Exception as error:
        try:
            index = sys.argv.index("--preregistration") + 1
            prereg = json.loads(Path(sys.argv[index]).read_text())
            output = ROOT / prereg["paths"]["audit_output"]
            if not output.exists():
                failure = {"schema": "r6_device_residual_cache_audit_v1", "status": "invalid",
                           "candidate": CANDIDATE, "error": repr(error), "promotion_authorized": False,
                           "confirmation_opened": False, "validation_opened": False, "test_id_opened": False,
                           "resources": {"output_bytes": 0}}
                atomic_write_bytes(report_bytes_fixed_point(failure), output)
        except Exception:
            pass
        raise


if __name__ == "__main__": raise SystemExit(entrypoint())
