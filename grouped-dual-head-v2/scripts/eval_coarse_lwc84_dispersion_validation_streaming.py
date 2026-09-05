#!/usr/bin/env python3
"""One-shot streamed validation audit of coarse LWC-84 dispersion sensitivity.

The frozen high-fidelity LWC-84 wavefields are used only as validation targets.
For every solver batch, all coarse predictions are materialized and hashed before
the first target field in that batch is opened.  Metrics are accumulated in time
blocks and predictions are then discarded, avoiding roughly 31 GB of unnecessary
dense prediction files while preserving an auditable predict-before-truth order.

Worker mode is selected by ``WORKER=0..3``.  Running without ``WORKER`` verifies
and merges the four immutable worker JSON files.  The test_id split is never read.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from saved_time_phase_operator_v4.coarse_lwc84 import (
    EXPECTED_VALIDATION_FAMILY_COUNTS,
    CoarseRecordIndex,
    select_all_validation_records,
    shard_records,
)
from saved_time_phase_operator_v4.streaming_metrics import (
    ExactWavefieldMetricAccumulator,
)
from scripts.evaluate_coarse_lwc84_201_shard import (
    _numerical_contract,
    _read_axes,
    _read_generation_inputs,
    _source_identity,
    _validate_axes,
    _validate_solver_batch,
    _warm_up_cuda,
)


SOURCE_H5 = Path(
    "/data/jiayh/data/"
    "acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
)
FROZEN_CONFIG = SOURCE_H5.parent / "frozen_config.yaml"
PREREG = ROOT / "results/r16_dscp_v23_coarse_lwc84_dispersion_preregistration_20260827.json"
OUT = ROOT / "results/r16_dscp_v23_coarse_lwc84_dispersion_validation"
WORKER_COUNT = 4
INTERNAL_DT_S = 5.0e-4
NPML = 20
C_REF_MPS = 6750.0
SOLVER_BATCH_SIZE = 8
METRIC_BLOCK_SIZE = 20
RAW_FIELD_BYTES = 401 * 201 * 201 * 4
FAMILIES = ("uniform", "layered", "marmousi")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_and_verify_preregistration() -> dict[str, Any]:
    if not PREREG.is_file():
        raise FileNotFoundError(f"frozen preregistration is missing: {PREREG}")
    payload = json.loads(PREREG.read_text(encoding="utf8"))
    if payload.get("status") != "frozen_before_execution":
        raise RuntimeError("dispersion preregistration is not frozen")
    if payload.get("test_id_policy") != "sealed_not_read":
        raise RuntimeError("test_id policy drift")
    bindings = payload.get("sha256_bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("preregistration lacks sha256_bindings")
    paths = {
        "script": Path(__file__).resolve(),
        "solver_lwc84": ROOT / "src/fno_acoustic/data_generation/solver_lwc84.py",
        "streaming_metrics": ROOT / "saved_time_phase_operator_v4/streaming_metrics.py",
        "coarse_lwc84": ROOT / "saved_time_phase_operator_v4/coarse_lwc84.py",
        "frozen_teacher_config": FROZEN_CONFIG,
        "source_h5": SOURCE_H5,
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = _sha256_file(path)
        if observed[name] != str(bindings.get(name, "")):
            raise RuntimeError(f"sha256 binding drift: {name}")
    payload["observed_sha256_bindings"] = observed
    return payload


def _prediction_sha256(prediction: np.ndarray) -> str:
    value = np.ascontiguousarray(prediction, dtype=np.float32)
    if value.shape != (401, 201, 201) or not np.isfinite(value).all():
        raise ValueError("coarse prediction is not a finite [401,201,201] field")
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def _receiver_points(height: int, width: int) -> tuple[tuple[int, int], ...]:
    z_index = max(1, height // 20)
    x_indices = np.linspace(5, width - 6, 9, dtype=int)
    return tuple((z_index, int(x_index)) for x_index in x_indices)


def _receiver_statistics(
    prediction: np.ndarray,
    truth: np.ndarray,
    points: Sequence[tuple[int, int]],
    *,
    stored_dt_s: float,
) -> dict[str, Any]:
    predicted = np.stack(
        [prediction[:, z_index, x_index] for z_index, x_index in points], axis=1
    ).astype(np.float64, copy=False)
    reference = np.stack(
        [truth[:, z_index, x_index] for z_index, x_index in points], axis=1
    ).astype(np.float64, copy=False)
    difference = predicted - reference
    error_square = float(np.square(difference).sum())
    target_square = float(np.square(reference).sum())
    trace_energy = np.square(reference).sum(axis=0)
    informative = trace_energy >= 0.01 * max(float(trace_energy.max()), 1.0e-30)
    signed_lag_samples: list[float] = []
    for receiver in np.flatnonzero(informative):
        correlation = np.correlate(
            predicted[:, receiver], reference[:, receiver], mode="full"
        )
        peak = int(np.argmax(correlation))
        lag = float(peak - (prediction.shape[0] - 1))
        if 0 < peak < correlation.size - 1:
            left = float(correlation[peak - 1])
            center = float(correlation[peak])
            right = float(correlation[peak + 1])
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1.0e-30:
                offset = 0.5 * (left - right) / denominator
                if abs(offset) <= 1.0:
                    lag += float(offset)
        signed_lag_samples.append(lag)
    return {
        "error_square": error_square,
        "target_square": target_square,
        "relative_l2": math.sqrt(error_square)
        / math.sqrt(max(target_square, 1.0e-30)),
        "informative_receiver_count": len(signed_lag_samples),
        "signed_lag_samples": signed_lag_samples,
        "signed_lag_ms": [1000.0 * value * stored_dt_s for value in signed_lag_samples],
    }


def _update_metrics(
    accumulator: ExactWavefieldMetricAccumulator,
    prediction: np.ndarray,
    truth: np.ndarray,
    record: CoarseRecordIndex,
    *,
    source_onset_index: int,
    device: torch.device,
) -> None:
    for start in range(0, 401, METRIC_BLOCK_SIZE):
        stop = min(start + METRIC_BLOCK_SIZE, 401)
        predicted_block = torch.from_numpy(prediction[start:stop]).to(device)
        truth_block = torch.from_numpy(truth[start:stop]).to(device)
        accumulator.update(
            predicted_block[None],
            truth_block[None],
            families=(record.medium_type,),
            group_ids=(record.group_id,),
            sample_ids=(record.sample_id,),
            time_indices=torch.arange(start, stop, dtype=torch.long)[None],
            source_onset_indices=(source_onset_index,),
        )
        del predicted_block, truth_block


def _serialize_statistics(
    accumulator: ExactWavefieldMetricAccumulator,
) -> dict[str, Any]:
    records = {
        sample_id: {
            "family": state.family,
            "group_id": state.group_id,
            "error_square": state.error_square,
            "target_square": state.target_square,
            "frame_count": state.frame_count,
        }
        for sample_id, state in accumulator._records.items()
    }
    return {
        "records": records,
        "frame_count": accumulator._frame_count,
        "near_zero_frame_count": accumulator._near_zero_frames,
        "unique_time_indices": sorted(accumulator._unique_times),
        "error_square": accumulator._error_square,
        "element_count": accumulator._element_count,
        "phase_sum": accumulator._phase_sum,
        "phase_count": accumulator._phase_count,
        "centroid_shift_sum": accumulator._centroid_shift_sum,
        "xcorr_shift_sum": accumulator._xcorr_shift_sum,
        "displacement_count": accumulator._displacement_count,
        "time_error": dict(accumulator._time_error),
        "time_target": dict(accumulator._time_target),
        "family_time_error": {
            f"{family}|{name}": value
            for (family, name), value in accumulator._family_time_error.items()
        },
        "family_time_target": {
            f"{family}|{name}": value
            for (family, name), value in accumulator._family_time_target.items()
        },
        "spectrum_error": dict(accumulator._spectrum_error),
        "spectrum_target": dict(accumulator._spectrum_target),
    }


def _relative(error_square: float, target_square: float) -> float:
    return math.sqrt(max(float(error_square), 0.0)) / math.sqrt(
        max(float(target_square), 1.0e-30)
    )


def run_worker(worker: int) -> dict[str, Any]:
    prereg = _load_and_verify_preregistration()
    if worker < 0 or worker >= WORKER_COUNT:
        raise ValueError("WORKER must lie in [0,4)")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    torch.cuda.set_device(device)
    all_records = select_all_validation_records(SOURCE_H5)
    records = shard_records(
        all_records, shard_index=worker, shard_count=WORKER_COUNT
    )
    if len(all_records) != 480 or len(records) != 120:
        raise RuntimeError("frozen validation census changed")
    if any(record.split != "validation" for record in records):
        raise RuntimeError("non-validation record entered the coarse audit")

    time_s, x_m, z_m = _read_axes(SOURCE_H5)
    stored_dt_s, dx_m, dz_m = _validate_axes(
        time_s, x_m, z_m, internal_dt_s=INTERNAL_DT_S
    )
    if (len(time_s), len(z_m), len(x_m)) != (401, 201, 201):
        raise RuntimeError("stored teacher axes changed")
    identity = _source_identity(SOURCE_H5)
    numerical_contract = _numerical_contract(
        time_s=time_s,
        x_m=x_m,
        z_m=z_m,
        stored_dt_s=stored_dt_s,
        dx_m=dx_m,
        dz_m=dz_m,
        internal_dt_s=INTERNAL_DT_S,
        npml=NPML,
        c_ref_mps=C_REF_MPS,
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    _warm_up_cuda(
        dx_m=dx_m,
        dz_m=dz_m,
        internal_dt_s=INTERNAL_DT_S,
        npml=NPML,
        c_ref_mps=C_REF_MPS,
        device=device,
    )
    grid = AcousticGrid(
        nx=len(x_m),
        nz=len(z_m),
        dx_m=dx_m,
        dz_m=dz_m,
        lx_m=float(x_m[-1] - x_m[0]),
        lz_m=float(z_m[-1] - z_m[0]),
        centering="node",
    )
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=BoundaryConfig(npml=NPML),
        dt_s=INTERNAL_DT_S,
        output_times_s=time_s,
        c_ref_mps=C_REF_MPS,
        device=device,
        dtype=torch.float32,
        output_restriction_factor=1,
    )
    accumulator = ExactWavefieldMetricAccumulator(
        require_unique=True, stored_time_count=401
    )
    receiver_points = _receiver_points(201, 201)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    solver_seconds = 0.0
    metric_seconds = 0.0
    for start in range(0, len(records), SOLVER_BATCH_SIZE):
        batch_records = records[start : start + SOLVER_BATCH_SIZE]
        velocity, source = _read_generation_inputs(SOURCE_H5, batch_records)
        torch.cuda.synchronize(device)
        solve_started = time.perf_counter()
        result = solver.simulate(
            velocity,
            source_x_m=source["source_x_m"],
            source_z_m=source["source_z_m"],
            source_f0_hz=source["source_f0_hz"],
            source_t0_s=source["source_t0_s"],
            source_amplitude=source["source_amplitude"],
        )
        torch.cuda.synchronize(device)
        solver_seconds += time.perf_counter() - solve_started
        _validate_solver_batch(
            result,
            velocity,
            records=batch_records,
            stored_time_count=401,
        )
        prediction_hashes = [
            _prediction_sha256(result.wavefield[index])
            for index in range(len(batch_records))
        ]
        all_predictions_hashed_ns = time.monotonic_ns()

        for local_index, record in enumerate(batch_records):
            truth_opened_ns = time.monotonic_ns()
            if truth_opened_ns <= all_predictions_hashed_ns:
                raise RuntimeError("monotonic predict-before-truth order failed")
            with h5py.File(SOURCE_H5, "r", swmr=True) as handle:
                raw_sample = handle["sample_id"][record.source_index]
                raw_split = handle["split"][record.source_index]
                sample_id = (
                    raw_sample.decode() if isinstance(raw_sample, bytes) else str(raw_sample)
                )
                split = raw_split.decode() if isinstance(raw_split, bytes) else str(raw_split)
                if sample_id != record.sample_id or split != "validation":
                    raise RuntimeError("target binding or validation split changed")
                truth = np.asarray(
                    handle["wavefield"][record.source_index], dtype=np.float32
                )
            if truth.shape != (401, 201, 201) or not np.isfinite(truth).all():
                raise ValueError("teacher target is invalid")
            prediction = np.asarray(result.wavefield[local_index], dtype=np.float32)
            onset_index = int(
                np.searchsorted(time_s, float(source["source_t0_s"][local_index]), side="left")
            )
            if onset_index < 0 or onset_index >= 401:
                raise RuntimeError("source onset is outside the stored axis")
            metric_started = time.perf_counter()
            _update_metrics(
                accumulator,
                prediction,
                truth,
                record,
                source_onset_index=onset_index,
                device=device,
            )
            receiver = _receiver_statistics(
                prediction,
                truth,
                receiver_points,
                stored_dt_s=stored_dt_s,
            )
            torch.cuda.synchronize(device)
            metric_seconds += time.perf_counter() - metric_started
            rows.append(
                {
                    **asdict(record),
                    "source_onset_index": onset_index,
                    "prediction_uncompressed_sha256": prediction_hashes[local_index],
                    "prediction_raw_byte_count": int(prediction.nbytes),
                    "all_batch_predictions_hashed_monotonic_ns": all_predictions_hashed_ns,
                    "truth_opened_monotonic_ns": truth_opened_ns,
                    "truth_opened_after_all_batch_prediction_hashes": True,
                    "receiver": receiver,
                    "solver_qc": dict(result.metrics[local_index]),
                }
            )
            del prediction, truth
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "worker": worker,
                    "completed": min(start + len(batch_records), len(records)),
                    "total": len(records),
                    "elapsed_s": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del result, velocity, source
        gc.collect()
        torch.cuda.empty_cache()

    statistics_payload = _serialize_statistics(accumulator)
    result_payload = {
        "schema": "coarse_lwc84_dispersion_stream_worker_v1",
        "status": "complete",
        "worker": worker,
        "worker_count": WORKER_COUNT,
        "record_count": len(records),
        "sample_ids": [record.sample_id for record in records],
        "family_counts": {
            family: sum(record.medium_type == family for record in records)
            for family in FAMILIES
        },
        "source_identity": identity,
        "numerical_contract": numerical_contract,
        "preregistration_sha256": _sha256_file(PREREG),
        "observed_sha256_bindings": prereg["observed_sha256_bindings"],
        "test_id_opened": False,
        "prediction_retention": {
            "dense_predictions_written_to_disk": False,
            "retained_prediction_bytes": 0,
            "raw_dense_bytes_that_would_have_been_written": len(records)
            * RAW_FIELD_BYTES,
            "audit_mechanism": "all batch predictions hashed before first batch truth read",
        },
        "metric_sufficient_statistics": statistics_payload,
        "records": rows,
        "runtime": {
            "solver_seconds": solver_seconds,
            "metric_seconds": metric_seconds,
            "wall_seconds": time.perf_counter() - started,
        },
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
    }
    destination = OUT / f"worker_{worker}.json"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite one-shot worker result: {destination}")
    _atomic_json(result_payload, destination)
    print(json.dumps({"event": "worker_complete", "worker": worker}), flush=True)
    return result_payload


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def merge_workers() -> dict[str, Any]:
    prereg = _load_and_verify_preregistration()
    all_records = select_all_validation_records(SOURCE_H5)
    expected = {record.sample_id: record for record in all_records}
    workers: list[dict[str, Any]] = []
    for worker in range(WORKER_COUNT):
        path = OUT / f"worker_{worker}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf8"))
        if payload.get("status") != "complete" or payload.get("test_id_opened") is not False:
            raise RuntimeError(f"worker {worker} is incomplete or opened test_id")
        expected_ids = [record.sample_id for record in all_records[worker::WORKER_COUNT]]
        if payload.get("sample_ids") != expected_ids:
            raise RuntimeError(f"worker {worker} record set drift")
        if payload.get("observed_sha256_bindings") != prereg["observed_sha256_bindings"]:
            raise RuntimeError(f"worker {worker} binding drift")
        workers.append(payload)

    record_stats: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    spectrum_error: defaultdict[str, float] = defaultdict(float)
    spectrum_target: defaultdict[str, float] = defaultdict(float)
    time_error: defaultdict[str, float] = defaultdict(float)
    time_target: defaultdict[str, float] = defaultdict(float)
    family_time_error: defaultdict[str, float] = defaultdict(float)
    family_time_target: defaultdict[str, float] = defaultdict(float)
    scalar_sums: defaultdict[str, float] = defaultdict(float)
    unique_time_indices: set[int] = set()
    for worker in workers:
        stats = worker["metric_sufficient_statistics"]
        for sample_id, values in stats["records"].items():
            if sample_id in record_stats:
                raise RuntimeError(f"duplicate record statistics: {sample_id}")
            record_stats[sample_id] = values
        rows.extend(worker["records"])
        for name, value in stats["spectrum_error"].items():
            spectrum_error[name] += float(value)
        for name, value in stats["spectrum_target"].items():
            spectrum_target[name] += float(value)
        for name, value in stats["time_error"].items():
            time_error[name] += float(value)
        for name, value in stats["time_target"].items():
            time_target[name] += float(value)
        for name, value in stats["family_time_error"].items():
            family_time_error[name] += float(value)
        for name, value in stats["family_time_target"].items():
            family_time_target[name] += float(value)
        for name in (
            "frame_count",
            "near_zero_frame_count",
            "error_square",
            "element_count",
            "phase_sum",
            "phase_count",
            "centroid_shift_sum",
            "xcorr_shift_sum",
            "displacement_count",
        ):
            scalar_sums[name] += float(stats[name])
        unique_time_indices.update(int(value) for value in stats["unique_time_indices"])

    if set(record_stats) != set(expected) or len(rows) != 480:
        raise RuntimeError("merged statistics do not cover exactly 480 validation records")
    row_by_id = {str(row["sample_id"]): row for row in rows}
    if set(row_by_id) != set(expected):
        raise RuntimeError("merged record rows do not match the validation census")
    if not all(
        bool(row["truth_opened_after_all_batch_prediction_hashes"]) for row in rows
    ):
        raise RuntimeError("predict-before-truth order failed")

    relative_by_record = {
        sample_id: _relative(values["error_square"], values["target_square"])
        for sample_id, values in record_stats.items()
    }
    family_relative = {
        family: statistics.fmean(
            relative_by_record[sample_id]
            for sample_id, record in expected.items()
            if record.medium_type == family
        )
        for family in FAMILIES
    }
    aggregate_relative = statistics.fmean(relative_by_record.values())
    spectrum_relative = {
        name: _relative(spectrum_error[name], spectrum_target[name])
        for name in ("low", "middle", "high")
    }
    receiver_error_square = sum(float(row["receiver"]["error_square"]) for row in rows)
    receiver_target_square = sum(float(row["receiver"]["target_square"]) for row in rows)
    receiver_record_relative = [float(row["receiver"]["relative_l2"]) for row in rows]
    signed_lag_ms = [
        float(value)
        for row in rows
        for value in row["receiver"]["signed_lag_ms"]
    ]
    absolute_lag_ms = [abs(value) for value in signed_lag_ms]
    gates = prereg["frozen_hypothesis_gates"]
    gate_results = {
        "coverage_480": len(relative_by_record) == 480,
        "coarse_aggregate_relative_l2_lte": aggregate_relative
        <= float(gates["coarse_aggregate_relative_l2_lte"]),
        "radial_high_relative_l2_gte": spectrum_relative["high"]
        >= float(gates["radial_high_relative_l2_gte"]),
        "high_to_low_ratio_gte": spectrum_relative["high"]
        / max(spectrum_relative["low"], 1.0e-30)
        >= float(gates["high_to_low_ratio_gte"]),
        "high_to_aggregate_ratio_gte": spectrum_relative["high"]
        / max(aggregate_relative, 1.0e-30)
        >= float(gates["high_to_aggregate_ratio_gte"]),
        "test_id_sealed": True,
    }
    sensitivity_supported = all(gate_results.values())
    terminal = {
        "schema": "coarse_lwc84_dispersion_stream_terminal_v1",
        "status": "complete",
        "claim_scope": (
            "Complete validation-panel sensitivity contrast for a coarsened LWC-84 "
            "solver against the frozen high-fidelity LWC-84 teacher; this does not "
            "establish that the teacher or the learned operator is dispersion-free."
        ),
        "preregistration_sha256": _sha256_file(PREREG),
        "observed_sha256_bindings": prereg["observed_sha256_bindings"],
        "coverage": {
            "split": "validation",
            "record_count": 480,
            "family_counts": EXPECTED_VALIDATION_FAMILY_COUNTS,
            "stored_time_count": 401,
            "test_id_opened": False,
        },
        "teacher_contract": prereg["teacher_contract"],
        "coarse_solver_contract": prereg["coarse_solver_contract"],
        "metrics": {
            "aggregate_relative_l2_record_mean": aggregate_relative,
            "family_relative_l2_record_mean": family_relative,
            "minimum_record_relative_l2": min(relative_by_record.values()),
            "median_record_relative_l2": statistics.median(relative_by_record.values()),
            "maximum_record_relative_l2": max(relative_by_record.values()),
            "spatial_spectrum_relative_l2_radial": spectrum_relative,
            "phase_correlation": scalar_sums["phase_sum"]
            / max(scalar_sums["phase_count"], 1.0),
            "centroid_shift_cells": scalar_sums["centroid_shift_sum"]
            / max(scalar_sums["displacement_count"], 1.0),
            "xcorr_peak_shift_cells": scalar_sums["xcorr_shift_sum"]
            / max(scalar_sums["displacement_count"], 1.0),
            "rmse": math.sqrt(
                scalar_sums["error_square"] / max(scalar_sums["element_count"], 1.0)
            ),
            "receiver_joint_relative_l2": _relative(
                receiver_error_square, receiver_target_square
            ),
            "receiver_record_relative_l2_mean": statistics.fmean(
                receiver_record_relative
            ),
            "receiver_signed_lag_ms_mean": statistics.fmean(signed_lag_ms)
            if signed_lag_ms
            else 0.0,
            "receiver_absolute_lag_ms_mean": statistics.fmean(absolute_lag_ms)
            if absolute_lag_ms
            else 0.0,
            "receiver_absolute_lag_ms_p95": _nearest_rank(absolute_lag_ms, 0.95),
            "receiver_absolute_lag_ms_max": max(absolute_lag_ms, default=0.0),
            "receiver_informative_trace_count": len(signed_lag_ms),
            "unique_time_index_count": len(unique_time_indices),
        },
        "dispersion_sensitivity": {
            "high_to_low_error_ratio": spectrum_relative["high"]
            / max(spectrum_relative["low"], 1.0e-30),
            "high_to_aggregate_error_ratio": spectrum_relative["high"]
            / max(aggregate_relative, 1.0e-30),
            "frozen_gate_results": gate_results,
            "supported": sensitivity_supported,
        },
        "storage_accounting": {
            "raw_dense_bytes_per_record": RAW_FIELD_BYTES,
            "raw_dense_bytes_for_480_predictions": 480 * RAW_FIELD_BYTES,
            "prediction_bytes_retained": 0,
            "streamed_prediction_disk_bytes_avoided": 480 * RAW_FIELD_BYTES,
            "note": "metric-pipeline storage only; not an inference working-memory claim",
        },
        "worker_files": [
            {
                "path": str(OUT / f"worker_{worker}.json"),
                "sha256": _sha256_file(OUT / f"worker_{worker}.json"),
            }
            for worker in range(WORKER_COUNT)
        ],
        "peak_cuda_allocated_bytes_max_worker": max(
            int(worker["peak_cuda_allocated_bytes"]) for worker in workers
        ),
        "runtime": {
            "worker_wall_seconds": [worker["runtime"]["wall_seconds"] for worker in workers],
            "worker_solver_seconds": [
                worker["runtime"]["solver_seconds"] for worker in workers
            ],
            "worker_metric_seconds": [
                worker["runtime"]["metric_seconds"] for worker in workers
            ],
        },
        "record_relative_l2": dict(sorted(relative_by_record.items())),
        "receiver_record_metrics": {
            sample_id: row_by_id[sample_id]["receiver"] for sample_id in sorted(row_by_id)
        },
    }
    destination = OUT / "terminal.json"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite one-shot terminal: {destination}")
    _atomic_json(terminal, destination)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return terminal


def main() -> None:
    worker_raw = os.environ.get("WORKER")
    if worker_raw is None:
        merge_workers()
    else:
        run_worker(int(worker_raw))


if __name__ == "__main__":
    main()
