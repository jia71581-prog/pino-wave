#!/usr/bin/env python3
"""Six-record train-only R6 float32 cache comparison; no build or training entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if value not in sys.path:
        sys.path.insert(0, value)

from build_r6_device_residual_cache import (  # noqa: E402
    CACHE_SCHEMA, FAMILIES, TruthGuard, cleanup_role_temporary, dataset_contract,
    load_roles, normalized_cache_values, role_time_indices, validate_source_identity,
    validate_environment, verify_stage_inputs, parent_stability_qc, device_result_qc,
    static_features, _fine_velocity, _read_manifest_rows, AcousticGrid, BoundaryConfig,
    DeviceResidentFusedLWC84CPMLSolver, FusedLWC84CPMLSolver, restrict_nodal_2x,
    target5_preflight, atomic_write_bytes, report_bytes_fixed_point, sha256,
)

CANDIDATE = "r6_device_residual_cache_fp32_v1_20260905"
STAGE = ROOT / "results/r6_anchored_r54_device_resident_r1_20260905/cache_stage_fp32_v1"
SOURCE_INDICES = {"fit": [0, 420, 2100], "development": [357, 968, 2375]}
PRIOR_TERMINAL_SHA256 = "4aaaa5c677654e4e4fc296de0337787be4deaf631618ea531c9f8b14a94b4094"


def selected_records(prereg: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    roles = load_roles(ROOT / prereg["paths"]["R38_manifest"])
    selected = {role: [dict(next(row for row in roles[role] if row["family"] == family))
                       for family in FAMILIES] for role in SOURCE_INDICES}
    if any([row["source_index"] for row in selected[role]] != indices for role, indices in SOURCE_INDICES.items()):
        raise RuntimeError("FP32 six-record panel drift")
    if selected != prereg["selection"]["records"]:
        raise RuntimeError("FP32 frozen selected records drift")
    return selected


def fp32_dataset_contract(role: str, count: int) -> dict[str, dict[str, Any]]:
    contract = dataset_contract(role, count)
    for name in ("coarse_norm", "truth_norm"):
        contract[name]["dtype"] = "float32"
    return contract


def create_datasets(cache: h5py.File, records: Sequence[Mapping[str, Any]], role: str,
                    time_s: np.ndarray) -> dict[str, h5py.Dataset]:
    """Use the frozen R25 schema, changing only the two wavefield dtypes."""
    contract = fp32_dataset_contract(role, len(records)); datasets = {}
    text = h5py.string_dtype("utf-8")
    for name, spec in contract.items():
        kwargs = {key: tuple(spec[key]) if key == "chunks" else spec[key]
                  for key in ("chunks", "compression", "shuffle") if key in spec}
        datasets[name] = cache.create_dataset(name, shape=tuple(spec["shape"]),
                                              dtype=text if spec["dtype"] == "utf8" else spec["dtype"], **kwargs)
    for name in ("sample_id", "group_id", "family", "sample_sha256", "source_index"):
        datasets[name][:] = [row[name] for row in records]
    indices = role_time_indices(role)
    datasets["time_indices"][:] = indices; datasets["time_s"][:] = time_s[list(indices)]
    return datasets


def validate_cache_schema(cache: h5py.File, role: str, count: int) -> None:
    if cache.attrs.get("schema") != CACHE_SCHEMA or cache.attrs.get("subset") != role:
        raise RuntimeError("FP32 cache schema/role drift")
    expected = fp32_dataset_contract(role, count)
    if set(cache.keys()) != set(expected): raise RuntimeError("FP32 cache dataset surface drift")
    for name, spec in expected.items():
        data = cache[name]
        if tuple(data.shape) != tuple(spec["shape"]): raise RuntimeError(f"FP32 shape drift: {name}")
        if spec["dtype"] == "utf8":
            if h5py.check_string_dtype(data.dtype).encoding != "utf-8": raise RuntimeError(f"FP32 text drift: {name}")
        elif str(data.dtype) != spec["dtype"]: raise RuntimeError(f"FP32 dtype drift: {name}")
        for key in ("chunks", "compression", "shuffle"):
            if key in spec and getattr(data, key) != (tuple(spec[key]) if key == "chunks" else spec[key]):
                raise RuntimeError(f"FP32 dataset storage drift: {name}/{key}")


def quantization_diagnostic_fp32(parent: np.ndarray, truth: np.ndarray, cached_parent: np.ndarray,
                            cached_truth: np.ndarray, static: np.ndarray, cached_static: np.ndarray,
                            time_indices: Sequence[int]) -> dict[str, Any]:
    """Measure smoke-only f32 storage error from actual HDF5 readback, without another truth read.

    P/T are the pre-storage normalized f32 arrays. qP/qT are stored f32 arrays
    converted to f32; differences, sums, dot products and norms then use f64.
    The record gate is deliberately separate from diagnostic temporal bands.
    """
    if parent.dtype != np.float32 or truth.dtype != np.float32 or parent.shape != truth.shape:
        raise RuntimeError("quantization diagnostic native field contract")
    if (cached_parent.dtype != np.float32 or cached_truth.dtype != np.float32
            or cached_parent.shape != parent.shape or cached_truth.shape != truth.shape
            or len(time_indices) != parent.shape[0]):
        raise RuntimeError("quantization diagnostic stored field contract")
    if any(not np.isfinite(value).all() for value in (parent, truth, cached_parent, cached_truth, static, cached_static)):
        raise RuntimeError("nonfinite quantization diagnostic input")
    qparent = cached_parent.astype(np.float32); qtruth = cached_truth.astype(np.float32)
    indices = np.asarray(time_indices, np.int64)
    frame_sums = {name: [] for name in ("truth", "parent", "native", "cached", "quantization",
                                      "parent_error", "truth_error", "residual_dot")}
    zero_truth_preserved = []
    for frame in range(len(indices)):
        p = parent[frame].astype(np.float64); t = truth[frame].astype(np.float64)
        qp = qparent[frame].astype(np.float64); qt = qtruth[frame].astype(np.float64)
        native = t - p; cached = qt - qp
        for name, value in (("truth", t), ("parent", p), ("native", native), ("cached", cached),
                            ("quantization", cached - native), ("parent_error", qp - p), ("truth_error", qt - t)):
            frame_sums[name].append(float(np.square(value).sum(dtype=np.float64)))
        frame_sums["residual_dot"].append(float(np.multiply(native, cached).sum(dtype=np.float64)))
        zero_truth_preserved.append(bool(np.all(qt[t == 0] == 0)))
    sums = {name: np.asarray(values, np.float64) for name, values in frame_sums.items()}

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        totals = {name: float(values[mask].sum(dtype=np.float64)) for name, values in sums.items()}
        def relative(numerator: str, denominator: str) -> float | None:
            return math.sqrt(totals[numerator] / totals[denominator]) if totals[denominator] > 0 else None
        native = relative("native", "truth"); error = relative("quantization", "truth")
        ratio = relative("quantization", "native")
        return {"frame_count": int(mask.sum()), "time_indices": indices[mask].tolist(),
                "R_native": native, "R_cached": relative("cached", "truth"), "E_q": error,
                "E_q_over_R_native": ratio,
                "residual_cosine": (totals["residual_dot"] / math.sqrt(totals["native"] * totals["cached"])
                                    if totals["native"] > 0 and totals["cached"] > 0 else None),
                "qP_field_relative_error": relative("parent_error", "parent"),
                "qT_field_relative_error": relative("truth_error", "truth"),
                "qP_error_over_truth_norm": relative("parent_error", "truth"),
                "square_norms_float64": totals,
                "truth_norm_zero": totals["truth"] == 0,
                "zero_truth_values_preserved": bool(all(np.asarray(zero_truth_preserved)[mask])),
                "near_zero_native_residual": native is None or native < 1e-5}

    record = metrics(np.ones(len(indices), dtype=bool))
    static_exact = (cached_static.dtype == np.float16 and cached_static.shape == static.shape
                    and cached_static.tobytes() == static.astype(np.float16).tobytes())
    ratio_applies = record["R_native"] is not None and record["R_native"] >= 1e-5
    absolute_passed = record["E_q"] is None or record["E_q"] < 1e-3
    ratio_passed = not ratio_applies or record["E_q_over_R_native"] <= .25
    passed = bool(absolute_passed and ratio_passed and record["zero_truth_values_preserved"] and static_exact)
    return {"schema": "r6_cache_quantization_diagnostic_fp32_v1", "field_storage_dtype": "float32", "static_storage_dtype": "float16", "record": record,
            "bands": {"early": metrics(indices < 134), "mid": metrics((indices >= 134) & (indices < 267)),
                      "late": metrics(indices >= 267)},
            "static_features_stored_float16_bytes_exact": bool(static_exact),
            "gate": {"passed": passed, "scope": "record_only_bands_diagnostic",
                     "E_q_lt_1e_minus_3": bool(absolute_passed), "residual_ratio_gate_applies": bool(ratio_applies),
                     "E_q_over_R_native_lte_0p25": bool(ratio_passed),
                     "zero_truth_checked_explicitly": True,
                     "failure_classification": None if passed else "cache_representation_rejected"}}

def prior_record_parity(prior: h5py.File, position: int, row: Mapping[str, Any],
                        parent_digest: str, values: Mapping[str, Any], static: np.ndarray) -> dict[str, bool]:
    for name in ("sample_id", "group_id", "family", "sample_sha256"):
        if str(prior[name].asstr()[position]) != str(row[name]): raise RuntimeError("prior f16 record identity drift")
    if int(prior["source_index"][position]) != int(row["source_index"]): raise RuntimeError("prior source index drift")
    if str(prior["parent_full_sha256"].asstr()[position]) != parent_digest: raise RuntimeError("prior parent full hash drift")
    for field, key in (("coarse_norm", "parent_norm"), ("truth_norm", "truth_norm")):
        if prior[field][position].tobytes() != values[key].astype(np.float16).tobytes():
            raise RuntimeError("FP32 native fields do not reproduce prior f16 bytes")
    if prior["static_features"][position].tobytes() != static.astype(np.float16).tobytes():
        raise RuntimeError("prior static bytes drift")
    for name in ("baseline_error_square_norm", "target_square_norm"):
        if float(prior[name][position]) != values[name]: raise RuntimeError("prior native scalar drift")
    if float(prior["field_scale"][position]) != float(np.float32(values["scale"])):
        raise RuntimeError("prior field scale drift")
    return {"parent_full_hash_equal": True, "prior_f16_wavefield_bytes_reproduced": True,
            "static_f16_bytes_equal": True, "native_error_scalars_equal": True, "field_scale_equal": True}


@cleanup_role_temporary
def smoke_role(*, role: str, records: Sequence[Mapping[str, Any]], output: Path,
               prereg: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    indices = role_time_indices(role); guard = TruthGuard(records, role); diagnostics = []; started = time.time()
    grid = AcousticGrid(nx=401, nz=401, dx_m=5, dz_m=5, lx_m=2000, lz_m=2000, centering="node")
    with h5py.File(prereg["paths"]["source_h5"], "r", swmr=True) as source, h5py.File(ROOT / prereg["paths"][f"prior_{role}_h5"], "r") as prior:
        times = np.asarray(source["time_s"][:], np.float64)
        if times.shape != (401,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise RuntimeError("source time contract drift")
        if not np.array_equal(prior["time_indices"][:], indices) or not np.array_equal(prior["time_s"][:], times[list(indices)]):
            raise RuntimeError("prior time axis drift")
        manifest_rows = _read_manifest_rows(Path(prereg["paths"]["source_manifest"]), {row["sample_id"] for row in records})
        solver_args = dict(grid=grid, boundaries=BoundaryConfig(npml=40), dt_s=.000625, output_times_s=times,
                           c_ref_mps=6750, device=device, dtype=torch.float32, kappa_max=3,
                           minimum_frequency_hz=8, output_restriction_factor=2, cuda_graphs=True)
        solver = DeviceResidentFusedLWC84CPMLSolver(**solver_args); solver.warmup(batch=1)
        legacy = FusedLWC84CPMLSolver(**solver_args); legacy.warmup(batch=1)
        with h5py.File(temporary, "x") as cache:
            cache.attrs.update({"schema": CACHE_SCHEMA, "status": "building", "subset": role, "role": role,
                                "parent_kind": "R6_device_resident_dt625us_restricted2x", "shard_index": 0, "shard_count": 4,
                                "parent_internal_dt_s": .000625, "parent_output_restriction_factor": 2,
                                "truth_policy": "train_only_supervision_not_deployment_input",
                                "selection_sha256": "5005f614d1ec7cbd77ddad440ac6132d0b79b5953aea70c566a8713d1433b937"})
            datasets = create_datasets(cache, records, role, times)
            validate_cache_schema(cache, role, len(records))
            for position, row in enumerate(records):
                index = int(row["source_index"])
                identity = validate_source_identity(source, row, manifest_rows[row["sample_id"]])
                stored = identity["stored_velocity"]; metadata = identity["metadata"]
                fine, _ = _fine_velocity(family=row["family"], manifest_row=manifest_rows[row["sample_id"]],
                                         grid=grid, marmousi_npy=Path(prereg["paths"]["marmousi"]))
                if not np.array_equal(restrict_nodal_2x(fine), stored): raise RuntimeError("velocity restriction drift")
                result = solver.simulate_device(fine, **metadata)
                if not device_result_qc(result, fine=fine, stored=stored)["passed"]: raise RuntimeError("device parent QC failure")
                parent = result.wavefield_device.detach().cpu().numpy()[0]
                digest = hashlib.sha256(parent.tobytes()).hexdigest()
                guard.register_parent(index=index, sample_id=row["sample_id"], digest=digest)
                if not parent_stability_qc(result.metrics[0]["cfl_2d"], result.metrics[0]["lwc_qmax"])["passed"]:
                    raise RuntimeError("parent stability drift")
                if legacy.simulate(fine, **metadata).wavefield[0].tobytes() != parent.tobytes():
                    raise RuntimeError("legacy/device parent parity failure")
                truth = guard.read_truth(source, index=index, sample_id=row["sample_id"], split="train", time_indices=indices)
                values = normalized_cache_values(parent, truth, indices)
                axis = np.linspace(0, 2000, 201, dtype=np.float32)
                static = static_features(stored, x_m=axis, z_m=axis, source_x_m=metadata["source_x_m"], source_z_m=metadata["source_z_m"])
                datasets["parent_full_sha256"][position] = digest
                datasets["field_scale"][position] = values["scale"]
                for name in ("source_f0_hz", "source_t0_s"): datasets[name][position] = metadata[name]
                for name in ("truth_frame_energy_max_norm", "truth_frame_energy_mean_norm", "baseline_error_square_norm", "target_square_norm"):
                    datasets[name][position] = values[name]
                datasets["coarse_norm"][position] = values["parent_norm"]
                datasets["truth_norm"][position] = values["truth_norm"]
                datasets["static_features"][position] = static.astype(np.float16)
                cache.flush()
                qp, qt, qs = (datasets[name][position] for name in ("coarse_norm", "truth_norm", "static_features"))
                exact = qp.tobytes() == values["parent_norm"].tobytes() and qt.tobytes() == values["truth_norm"].tobytes()
                if not exact: raise RuntimeError("FP32 actual stored byte reconstruction failure")
                diagnostic = quantization_diagnostic_fp32(values["parent_norm"], values["truth_norm"], qp, qt, static, qs, indices)
                diagnostic.update({"role": role, "family": row["family"], "sample_id": row["sample_id"], "source_index": index,
                                   "stored_float32_wavefield_bytes_exact": exact, "additional_source_truth_reads": 0,
                                   "prior_f16_parity": prior_record_parity(prior, position, row, digest, values, static)})
                diagnostics.append(diagnostic)
                progress = {"schema": "r6_cache_fp32_smoke_progress_v1", "role": role, "completed_records": position + 1,
                            "record_count": len(records), "last_sample_id": row["sample_id"]}
                atomic_write_bytes(report_bytes_fixed_point(progress), output.with_suffix(".progress.json"))
            cache.attrs["status"] = "complete"; cache.attrs["truth_ledger_json"] = json.dumps(guard.summary(), sort_keys=True); cache.flush()
    os.replace(temporary, output)
    summary = {"schema": "r6_cache_fp32_smoke_summary_v1", "status": "complete", "role": role,
               "record_count": len(records), "frame_count_per_record": len(indices), "output": str(output),
               "output_sha256": sha256(output), "output_bytes": output.stat().st_size, "truth_ledger": guard.summary(),
               "elapsed_seconds": time.time() - started, "quantization_diagnostics": diagnostics,
               "cache_representation_gate_passed": all(row["gate"]["passed"] for row in diagnostics),
               "legacy_device_parity": True, "field_storage_dtype": "float32", "static_storage_dtype": "float16",
               "confirmation_opened": False, "validation_opened": False, "test_id_opened": False}
    atomic_write_bytes(report_bytes_fixed_point(summary), output.with_suffix(".summary.json"))
    return summary


def validate_invocation(args: argparse.Namespace, prereg: Mapping[str, Any]) -> None:
    if not args.smoke or args.device != "cuda:0": raise RuntimeError("only the fixed GPU0 smoke is permitted")
    if prereg.get("schema") != "r6_cache_fp32_smoke_preregistration_v1" or prereg.get("candidate") != CANDIDATE:
        raise RuntimeError("FP32 smoke candidate/schema drift")
    if prereg.get("status") != "fp32_smoke_prepared_pending_audit": raise RuntimeError("FP32 smoke status drift")
    if args.preregistration.resolve() != STAGE / "preregistration.json": raise RuntimeError("FP32 preregistration override")
    if (ROOT / prereg["paths"]["stage_dir"]).resolve() != STAGE: raise RuntimeError("FP32 stage override")
    if prereg.get("full_build_authorized") or prereg.get("training_authorized"): raise RuntimeError("FP32 smoke scope drift")
    prior = ROOT / prereg["paths"]["prior_terminal"]
    if sha256(prior) != PRIOR_TERMINAL_SHA256 or json.loads(prior.read_text())["status"] != "cache_representation_rejected":
        raise RuntimeError("prior f16 rejection evidence drift")
    runtime = prereg["prerequisites"]["runtime"]
    if runtime["status"] != "passed" or sha256(ROOT / runtime["path"]) != runtime["sha256"]:
        raise RuntimeError("passed runtime prerequisite missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); prereg = json.loads(args.preregistration.read_text())
    terminal = STAGE / "smoke_terminal.json"
    if terminal.exists() or (STAGE / "smoke").exists(): raise FileExistsError("FP32 smoke output already exists")
    started = time.time(); before = None; summaries = []
    try:
        validate_invocation(args, prereg)
        environment = validate_environment(prereg, "0"); before = verify_stage_inputs(prereg)
        target_audit = target5_preflight(prereg); panel = selected_records(prereg)
        device = torch.device(args.device); torch.cuda.set_device(device); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps({"event": "R6_FP32_CACHE_SMOKE_START", "candidate": CANDIDATE, "records": 6}), flush=True)
        for role in SOURCE_INDICES:
            summaries.append(smoke_role(role=role, records=panel[role], output=STAGE / "smoke" / f"smoke_{role}_shard_0.h5",
                                        prereg=prereg, device=device))
        after = verify_stage_inputs(prereg); elapsed = time.time() - started; peak = int(torch.cuda.max_memory_allocated(device))
        if elapsed > 180 or peak >= 8 * 2**30: raise RuntimeError("FP32 smoke resource budget")
        if before != after: raise RuntimeError("FP32 inputs changed during smoke")
        passed = all(summary["cache_representation_gate_passed"] for summary in summaries)
        payload = {"schema": "r6_cache_fp32_smoke_terminal_v1", "candidate": CANDIDATE,
                   "status": "complete" if passed else "cache_representation_rejected", "cache_representation_gate_passed": passed,
                   "record_count": 6, "summaries": summaries, "input_hashes_before": before, "input_hashes_after": after,
                   "input_hashes_unchanged": before == after, "environment": environment, "target5_audit": target_audit,
                   "resources": {"elapsed_seconds": elapsed, "peak_allocated_bytes": peak},
                   "operator_failure_claimed": False, "full_build_authorized": False, "training_authorized": False,
                   "promotion_authorized": False, "confirmation_opened": False, "validation_opened": False, "test_id_opened": False}
        atomic_write_bytes(report_bytes_fixed_point(payload), terminal)
        return 0 if passed else 2
    except Exception as error:
        if not terminal.exists():
            failure = {"schema": "r6_cache_fp32_smoke_terminal_v1", "candidate": CANDIDATE, "status": "failed",
                       "error": repr(error), "input_hashes_before": before, "summaries": summaries,
                       "resources": {"elapsed_seconds": time.time() - started}, "retry_attempted": False,
                       "full_build_authorized": False, "training_authorized": False, "promotion_authorized": False,
                       "confirmation_opened": False, "validation_opened": False, "test_id_opened": False}
            atomic_write_bytes(report_bytes_fixed_point(failure), terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
