#!/usr/bin/env python3
"""Train-holdout oracle for spatially varying temporal phase correction.

For selected already-opened R28 holdout records, this script materializes the
R28 prediction and measures two truth-derived diagnostics: the best constant
integer time shift independently at each spatial point, and the best clipped
coefficient multiplying a centered temporal difference.  These are oracle
ceilings only and are never deployable predictions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
R28_PATH = SCRIPT_PATH.with_name("train_r28_expanded_tail_spectral.py")
SPEC = importlib.util.spec_from_file_location("r31_r28_components", R28_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R28 components: {R28_PATH}")
r28 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r28
SPEC.loader.exec_module(r28)
r25 = r28.r25
r26 = r28.r26


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    model = r26.TailSpectralResidualUNet(
        base_width=int(config.get("base_width", 32)),
        correction_cap=float(config.get("correction_cap", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict_record(
    model: torch.nn.Module,
    handle: Any,
    local_index: int,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    static = torch.from_numpy(
        np.asarray(handle["static_features"][local_index], dtype=np.float32)
    ).to(device)
    f0 = float(handle["source_f0_hz"][local_index])
    t0 = float(handle["source_t0_s"][local_index])
    times = np.asarray(handle["time_s"][:], dtype=np.float32)
    coarse_all = np.asarray(handle["coarse_norm"][local_index], dtype=np.float32)
    truth_all = np.asarray(handle["truth_norm"][local_index], dtype=np.float32)
    predictions: list[np.ndarray] = []
    for start in range(0, len(times), int(batch_size)):
        stop = min(start + int(batch_size), len(times))
        coarse = torch.from_numpy(coarse_all[start:stop]).to(device)
        block = stop - start
        time_tensor = torch.from_numpy(times[start:stop]).to(device)
        features = r25.make_dynamic_features(
            coarse,
            static[None].expand(block, -1, -1, -1),
            time_s=time_tensor,
            source_f0_hz=torch.full((block,), f0, device=device),
            source_t0_s=torch.full((block,), t0, device=device),
        )
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp
            else nullcontext()
        )
        with context:
            correction = model(features, active=(time_tensor >= t0).float()).float()
        predictions.append((coarse + correction).cpu().numpy())
    return np.concatenate(predictions, axis=0), truth_all, times


def relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return math.sqrt(
        float(np.square(error).sum())
        / max(float(np.square(np.asarray(truth, dtype=np.float64)).sum()), 1.0e-30)
    )


def integer_shift_oracle(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    maximum_shift: int,
) -> dict[str, Any]:
    time_count = prediction.shape[0]
    indices = np.arange(time_count)
    best_square = np.full(prediction.shape[1:], np.inf, dtype=np.float64)
    best_shift = np.zeros(prediction.shape[1:], dtype=np.int16)
    global_rows: list[tuple[int, float]] = []
    truth64 = truth.astype(np.float64)
    target_square = max(float(np.square(truth64).sum()), 1.0e-30)
    for shift in range(-int(maximum_shift), int(maximum_shift) + 1):
        shifted = prediction[np.clip(indices + shift, 0, time_count - 1)].astype(np.float64)
        pixel_square = np.square(shifted - truth64).sum(axis=0)
        improved = pixel_square < best_square
        best_square[improved] = pixel_square[improved]
        best_shift[improved] = int(shift)
        global_rows.append((shift, math.sqrt(float(pixel_square.sum()) / target_square)))
    unique, counts = np.unique(best_shift, return_counts=True)
    return {
        "maximum_shift_frames": int(maximum_shift),
        "oracle_rel_l2": math.sqrt(float(best_square.sum()) / target_square),
        "best_global_shift_frames": int(min(global_rows, key=lambda row: row[1])[0]),
        "best_global_shift_rel_l2": float(min(row[1] for row in global_rows)),
        "spatial_shift_histogram": {
            str(int(value)): int(count) for value, count in zip(unique, counts)
        },
    }


def derivative_oracle(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    lag: int,
    coefficient_cap: float,
) -> dict[str, Any]:
    indices = np.arange(prediction.shape[0])
    previous = prediction[np.clip(indices - int(lag), 0, prediction.shape[0] - 1)]
    following = prediction[np.clip(indices + int(lag), 0, prediction.shape[0] - 1)]
    derivative = 0.5 * (following.astype(np.float64) - previous.astype(np.float64))
    residual = truth.astype(np.float64) - prediction.astype(np.float64)
    numerator = (derivative * residual).sum(axis=0)
    denominator = np.square(derivative).sum(axis=0)
    coefficient = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-30,
    )
    coefficient = np.clip(coefficient, -float(coefficient_cap), float(coefficient_cap))
    corrected = prediction.astype(np.float64) + coefficient[None] * derivative
    return {
        "lag_frames": int(lag),
        "coefficient_cap": float(coefficient_cap),
        "oracle_rel_l2": relative_l2(corrected, truth),
        "coefficient_quantiles": {
            str(q): float(np.quantile(coefficient, q))
            for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-id", nargs="+", required=True)
    parser.add_argument("--maximum-shift", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--derivative-lag", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--coefficient-cap", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    collection = r25.CacheCollection(args.holdout_cache, expected_subset="holdout")
    lookup: dict[str, tuple[int, int]] = {}
    for file_index, local_index in collection.records:
        sample_id = str(collection.handles[file_index]["sample_id"].asstr()[local_index])
        lookup[sample_id] = (file_index, local_index)
    missing = [sample_id for sample_id in args.sample_id if sample_id not in lookup]
    if missing:
        raise RuntimeError(f"sample IDs absent from R28 holdout: {missing}")
    model = load_model(args.checkpoint.expanduser().resolve(), device)
    rows: list[dict[str, Any]] = []
    for sample_id in args.sample_id:
        file_index, local_index = lookup[sample_id]
        handle = collection.handles[file_index]
        prediction, truth, times = predict_record(
            model,
            handle,
            local_index,
            device=device,
            batch_size=int(args.batch_size),
            amp=bool(args.amp),
        )
        row = {
            "sample_id": sample_id,
            "family": str(handle["family"].asstr()[local_index]),
            "source_f0_hz": float(handle["source_f0_hz"][local_index]),
            "stored_dt_s": float(np.median(np.diff(times))),
            "r28_rel_l2": relative_l2(prediction, truth),
            "integer_shift_oracles": [
                integer_shift_oracle(
                    prediction, truth, maximum_shift=int(maximum_shift)
                )
                for maximum_shift in args.maximum_shift
            ],
            "derivative_oracles": [
                derivative_oracle(
                    prediction,
                    truth,
                    lag=int(lag),
                    coefficient_cap=float(args.coefficient_cap),
                )
                for lag in args.derivative_lag
            ],
        }
        rows.append(row)
        print(json.dumps({"event": "R31_RECORD_COMPLETE", **row}, sort_keys=True), flush=True)

    payload = {
        "schema": "r31_local_time_shift_oracle_v1",
        "role": "already_opened_R28_train_holdout_truth_derived_oracle_only",
        "deployable": False,
        "records": rows,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": r25.sha256_file(args.checkpoint.expanduser().resolve()),
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": r25.sha256_file(SCRIPT_PATH),
    }
    r25.atomic_json(payload, args.output.expanduser().resolve())
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
