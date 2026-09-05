#!/usr/bin/env python3
"""Score serialized WFP predictions against 401-frame validation truth."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import h5py
import numpy as np
from scipy.fft import rfft


FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
REQUIRED_FAMILIES = ("uniform", "layered", "marmousi")
TIME_BANDS = {"early": (0, 134), "middle": (134, 267), "late": (267, 401)}
FREQUENCY_BANDS = {
    "low_0_15": (0, 16),
    "mid_16_31": (16, 32),
    "trained_high_32_63": (32, 64),
    "unmodeled_64_200": (64, 201),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def ratio(numerator: float, denominator: float) -> float:
    return math.sqrt(float(numerator) / max(float(denominator), 1.0e-300))


def sample_metrics(prediction: np.ndarray, truth: np.ndarray) -> tuple[dict, dict]:
    prediction = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    if prediction.shape != (401, 201, 201) or truth.shape != prediction.shape:
        raise RuntimeError("401-frame prediction/truth shape mismatch")
    difference = prediction.astype(np.float64) - truth.astype(np.float64)
    truth64 = truth.astype(np.float64)
    prediction64 = prediction.astype(np.float64)
    numerator = float(np.square(difference).sum())
    denominator = float(np.square(truth64).sum())
    prediction_square = float(np.square(prediction64).sum())
    dot = float((prediction64 * truth64).sum())
    time_metrics, time_sums = {}, {}
    for name, (start, stop) in TIME_BANDS.items():
        band_num = float(np.square(difference[start:stop]).sum())
        band_den = float(np.square(truth64[start:stop]).sum())
        time_metrics[name] = ratio(band_num, band_den)
        time_sums[name] = (band_num, band_den)

    prediction_spectrum = rfft(prediction, axis=0, norm="ortho", workers=1)
    truth_spectrum = rfft(truth, axis=0, norm="ortho", workers=1)
    spectral_difference = prediction_spectrum - truth_spectrum
    weights = np.full((201, 1, 1), 2.0, dtype=np.float64)
    weights[0] = 1.0
    weights[-1] = 1.0
    frequency_metrics, frequency_sums = {}, {}
    for name, (start, stop) in FREQUENCY_BANDS.items():
        band_weights = weights[start:stop]
        band_num = float(
            (np.square(np.abs(spectral_difference[start:stop])).astype(np.float64) * band_weights).sum()
        )
        band_den = float(
            (np.square(np.abs(truth_spectrum[start:stop])).astype(np.float64) * band_weights).sum()
        )
        frequency_metrics[name] = ratio(band_num, band_den)
        frequency_sums[name] = (band_num, band_den)
    metrics = {
        "relative_l2": ratio(numerator, denominator),
        "prediction_target_norm_ratio": ratio(prediction_square, denominator),
        "cosine": dot / max(math.sqrt(prediction_square * denominator), 1.0e-300),
        "time_band_relative_l2": time_metrics,
        "frequency_band_relative_l2": frequency_metrics,
        "top_pressure_max_abs": float(np.max(np.abs(prediction[:, 0, :]))),
    }
    sums = {
        "full": (numerator, denominator),
        "time": time_sums,
        "frequency": frequency_sums,
    }
    return metrics, sums


def aggregate_values(values: list[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    return {
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p95_nearest_rank": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "maximum": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    terminal_path = args.output_dir / "terminal.json"
    manifest = json.loads(args.manifest.read_text())
    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    if sha256(Path(__file__)) != bindings["scorer_sha256"]:
        raise RuntimeError("scorer binding drift")
    if sha256(args.manifest) != bindings["validation_manifest_sha256"]:
        raise RuntimeError("validation manifest binding drift")
    if len(args.prediction) != 4:
        raise RuntimeError("complete validation requires four prediction shards")

    prediction_handles = []
    locations = {}
    prediction_summaries = []
    try:
        for path in args.prediction:
            summary_path = path.with_suffix(path.suffix + ".summary.json")
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "complete" or summary.get("smoke"):
                raise RuntimeError(f"incomplete or smoke prediction shard: {path}")
            if summary.get("validation_future_truth_read") or summary.get("test_id_opened"):
                raise RuntimeError("prediction stage violated truth-access boundary")
            if sha256(path) != summary["output_sha256"]:
                raise RuntimeError(f"prediction shard hash drift: {path}")
            handle = h5py.File(path, "r", swmr=True)
            prediction_handles.append(handle)
            if handle.attrs.get("status") != "complete":
                raise RuntimeError(f"prediction shard not terminal: {path}")
            if bool(handle.attrs.get("validation_future_truth_read")):
                raise RuntimeError("prediction shard reports future-truth access")
            for local, raw_id in enumerate(handle["sample_id"].asstr()[:]):
                sample_id = str(raw_id)
                if sample_id in locations:
                    raise RuntimeError(f"duplicate serialized prediction: {sample_id}")
                locations[sample_id] = (handle, local)
            prediction_summaries.append(summary)

        expected_ids = [str(row["sample_id"]) for row in manifest["records"]]
        expected_count = int(manifest["record_count"])
        if set(locations) != set(expected_ids) or len(locations) != expected_count:
            missing = sorted(set(expected_ids) - set(locations))
            extra = sorted(set(locations) - set(expected_ids))
            raise RuntimeError(f"prediction coverage mismatch, missing={missing[:3]}, extra={extra[:3]}")

        # Future truth is first opened only after all 600 serialized predictions
        # have passed shape, status, census, split-audit, and hash checks above.
        source_path = Path(manifest["source_h5"])
        rows = []
        pooled_num = pooled_den = 0.0
        pooled_time = defaultdict(lambda: [0.0, 0.0])
        pooled_frequency = defaultdict(lambda: [0.0, 0.0])
        family_values = defaultdict(list)
        started = time.time()
        with h5py.File(source_path, "r", swmr=True) as source:
            for position, manifest_row in enumerate(manifest["records"]):
                sample_id = str(manifest_row["sample_id"])
                family = str(manifest_row["family"])
                index = int(manifest_row["source_index"])
                handle, local = locations[sample_id]
                prediction = np.asarray(handle["prediction"][local], dtype=np.float32)
                truth = np.asarray(source["wavefield"][index], dtype=np.float32)
                metrics, sums = sample_metrics(prediction, truth)
                pooled_num += sums["full"][0]
                pooled_den += sums["full"][1]
                for name, values in sums["time"].items():
                    pooled_time[name][0] += values[0]
                    pooled_time[name][1] += values[1]
                for name, values in sums["frequency"].items():
                    pooled_frequency[name][0] += values[0]
                    pooled_frequency[name][1] += values[1]
                family_values[family].append(metrics["relative_l2"])
                rows.append({"sample_id": sample_id, "family": family, **metrics})
                if (position + 1) % 20 == 0:
                    print(json.dumps({
                        "event": "score_progress",
                        "record": position + 1,
                        "of": len(expected_ids),
                        "elapsed_s": time.time() - started,
                    }, sort_keys=True), flush=True)

        all_values = [row["relative_l2"] for row in rows]
        required_values = [
            row["relative_l2"] for row in rows if row["family"] in REQUIRED_FAMILIES
        ]
        evaluated_families = tuple(str(value) for value in manifest["family_counts"])
        family_summary = {
            family: aggregate_values(family_values[family])
            for family in evaluated_families
        }
        required_family_means = {
            family: family_summary[family]["mean"] for family in REQUIRED_FAMILIES
        }
        target = float(prereg["acceptance"]["relative_l2_max"])
        required_aggregate = float(statistics.fmean(required_values))
        target_passed = required_aggregate <= target and all(
            value <= target for value in required_family_means.values()
        )
        complete_validation = expected_count == 600
        report = {
            "schema": "transfer_dg_wfp_validation401_report_v1",
            "status": (
                "target_gate_passed" if target_passed else "target_gate_failed"
            ) if complete_validation else (
                "panel_target_gate_passed" if target_passed else "panel_target_gate_failed"
            ),
            "evaluation_scope": "complete_validation" if complete_validation else "sampled_panel",
            "split": "validation",
            "record_count": len(rows),
            "frame_count": 401,
            "all_family_relative_l2": aggregate_values(all_values),
            "all_family_pooled_relative_l2": ratio(pooled_num, pooled_den),
            "required_panel_relative_l2_mean": required_aggregate,
            "per_family": family_summary,
            "pooled_time_band_relative_l2": {
                name: ratio(values[0], values[1]) for name, values in pooled_time.items()
            },
            "pooled_frequency_band_relative_l2": {
                name: ratio(values[0], values[1]) for name, values in pooled_frequency.items()
            },
            "acceptance": {
                "threshold": target,
                "required_families": list(REQUIRED_FAMILIES),
                "required_aggregate": required_aggregate,
                "required_family_means": required_family_means,
                "passed": target_passed,
            },
            "prediction_runtime": {
                "mean_of_worker_means_s": float(statistics.fmean(
                    summary["runtime_mean_s"] for summary in prediction_summaries
                )),
                "maximum_worker_p95_s": max(
                    float(summary["runtime_p95_s"]) for summary in prediction_summaries
                ),
            },
            "prediction_shards": [
                {
                    "path": summary["output"],
                    "sha256": summary["output_sha256"],
                    "record_count": summary["record_count"],
                }
                for summary in prediction_summaries
            ],
            "checkpoint": prereg["checkpoint"],
            "checkpoint_sha256": bindings["checkpoint_sha256"],
            "rows": rows,
            "access_audit": {
                "predictions_complete_before_truth_open": True,
                "serialized_prediction_count_before_truth_open": expected_count,
                "model_input_wavefield_frames": 0,
                "validation_public_inputs_opened": True,
                "validation_future_truth_opened_only_for_scoring": True,
                "test_id_opened": False,
                "no_post_validation_tuning": True,
            },
            "elapsed_scoring_s": time.time() - started,
        }
        atomic_json(report, report_path)
        terminal = {
            "schema": "transfer_dg_wfp_validation401_terminal_v1",
            "status": "complete",
            "decision": (
                "eligible_for_test_id" if target_passed else "stop_before_test_id"
            ) if complete_validation else "panel_only_no_test_promotion",
            "report": str(report_path.resolve()),
            "report_sha256": sha256(report_path),
            "record_count": len(rows),
            "frame_count": 401,
            "required_aggregate_relative_l2": required_aggregate,
            "required_family_relative_l2": required_family_means,
            "target_passed": target_passed,
            "validation_future_truth_opened": True,
            "test_id_opened": False,
        }
        atomic_json(terminal, terminal_path)
        print(json.dumps(terminal, sort_keys=True), flush=True)
    finally:
        for handle in prediction_handles:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
