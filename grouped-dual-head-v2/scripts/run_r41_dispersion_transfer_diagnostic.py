#!/usr/bin/env python3
"""Fit and audit a train-only frequency-wavenumber dispersion transfer model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.fft import idctn

import train_r40_frequency_residual_operator as r40


MODEL_TYPES = ("universal", "velocity_linear")
RIDGE_FRACTIONS = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
CALIBRATION_FRACTION = 0.20
CALIBRATION_SEED = 410828


def canonical_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def complex_channels(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 2:
        raise ValueError(f"unexpected complex-channel shape: {array.shape}")
    return array[:, 0] + 1j * array[:, 1]


def record_rows(collection: r40.FrequencyCacheCollection) -> list[dict[str, Any]]:
    rows = []
    for position, (file_index, local_index) in enumerate(collection.records):
        handle = collection.handles[file_index]
        rows.append(
            {
                "position": position,
                "file_index": file_index,
                "local_index": local_index,
                "sample_id": str(handle["sample_id"].asstr()[local_index]),
                "group_id": str(handle["group_id"].asstr()[local_index]),
                "family": str(handle["family"].asstr()[local_index]),
            }
        )
    return rows


def split_fit_groups(rows: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int], dict]:
    group_family: dict[str, str] = {}
    for row in rows:
        group = str(row["group_id"])
        family = str(row["family"])
        if group in group_family and group_family[group] != family:
            raise RuntimeError("a fit group spans multiple families")
        group_family[group] = family
    calibration_groups: set[str] = set()
    family_counts = {}
    for family in r40.FAMILIES:
        groups = [group for group, value in group_family.items() if value == family]
        groups.sort(
            key=lambda group: hashlib.sha256(
                f"{CALIBRATION_SEED}:{group}".encode("utf-8")
            ).hexdigest()
        )
        count = max(1, int(round(CALIBRATION_FRACTION * len(groups))))
        chosen = set(groups[:count])
        calibration_groups.update(chosen)
        family_counts[family] = {
            "all_groups": len(groups),
            "calibration_groups": len(chosen),
            "estimation_groups": len(groups) - len(chosen),
        }
    estimation = [
        int(row["position"])
        for row in rows
        if str(row["group_id"]) not in calibration_groups
    ]
    calibration = [
        int(row["position"])
        for row in rows
        if str(row["group_id"]) in calibration_groups
    ]
    if not estimation or not calibration:
        raise RuntimeError("empty R41 internal split")
    estimation_groups = {str(rows[index]["group_id"]) for index in estimation}
    calibration_groups_check = {
        str(rows[index]["group_id"]) for index in calibration
    }
    if estimation_groups & calibration_groups_check:
        raise RuntimeError("R41 internal group leakage")
    return estimation, calibration, {
        "seed": CALIBRATION_SEED,
        "fraction": CALIBRATION_FRACTION,
        "estimation_records": len(estimation),
        "calibration_records": len(calibration),
        "estimation_groups": len(estimation_groups),
        "calibration_groups": len(calibration_groups_check),
        "per_family": family_counts,
    }


def read_record(
    collection: r40.FrequencyCacheCollection, position: int
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]:
    file_index, local_index = collection.records[int(position)]
    handle = collection.handles[file_index]
    base = complex_channels(handle["base_dct_norm"][local_index])
    residual = complex_channels(handle["residual_dct_norm"][local_index])
    static_dc = float(handle["static_dct_norm"][local_index, 0, 0, 0])
    static_scale = float(handle["static_dct_scale"][local_index, 0])
    mean_velocity_feature = static_dc * static_scale / float(r40.SOURCE_GRID_SIZE)
    frequency_scale = np.asarray(
        handle["frequency_scale"][local_index], dtype=np.float64
    )
    target_total = float(handle["target_square_total"][local_index])
    unselected = float(handle["base_error_square_unselected"][local_index])
    return base, residual, mean_velocity_feature, frequency_scale, target_total, unselected


def accumulate_statistics(
    collection: r40.FrequencyCacheCollection, positions: Iterable[int]
) -> dict[str, np.ndarray | int]:
    shape = (
        collection.frequency_count,
        int(collection.retained),
        int(collection.retained),
    )
    s00 = np.zeros(shape, dtype=np.float64)
    s01 = np.zeros(shape, dtype=np.float64)
    s11 = np.zeros(shape, dtype=np.float64)
    t0 = np.zeros(shape, dtype=np.complex128)
    t1 = np.zeros(shape, dtype=np.complex128)
    count = 0
    for position in positions:
        base, residual, velocity, _, _, _ = read_record(collection, int(position))
        energy = np.abs(base).astype(np.float64) ** 2
        cross = np.conjugate(base).astype(np.complex128) * residual.astype(
            np.complex128
        )
        s00 += energy
        s01 += velocity * energy
        s11 += velocity * velocity * energy
        t0 += cross
        t1 += velocity * cross
        count += 1
    return {"s00": s00, "s01": s01, "s11": s11, "t0": t0, "t1": t1, "count": count}


def solve_filter(
    stats: Mapping[str, np.ndarray | int], *, model_type: str, ridge_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    s00 = np.asarray(stats["s00"], dtype=np.float64)
    t0 = np.asarray(stats["t0"], dtype=np.complex128)
    positive_mean = np.mean(s00, axis=(1, 2), keepdims=True)
    ridge = float(ridge_fraction) * positive_mean + 1.0e-14
    if model_type == "universal":
        a0 = t0 / (s00 + ridge)
        a1 = np.zeros_like(a0)
    elif model_type == "velocity_linear":
        s01 = np.asarray(stats["s01"], dtype=np.float64)
        s11 = np.asarray(stats["s11"], dtype=np.float64)
        t1 = np.asarray(stats["t1"], dtype=np.complex128)
        d00 = s00 + ridge
        d11 = s11 + ridge
        determinant = d00 * d11 - s01 * s01
        safe = np.where(np.abs(determinant) > 1.0e-20, determinant, np.inf)
        a0 = (t0 * d11 - t1 * s01) / safe
        a1 = (t1 * d00 - t0 * s01) / safe
    else:
        raise ValueError(model_type)
    a0 = np.nan_to_num(a0, copy=False).astype(np.complex64)
    a1 = np.nan_to_num(a1, copy=False).astype(np.complex64)
    return a0, a1


def proxy_rows(
    collection: r40.FrequencyCacheCollection,
    positions: Iterable[int],
    *,
    a0: np.ndarray,
    a1: np.ndarray,
) -> list[dict[str, Any]]:
    weights = r40.rfft_weights(r40.TIME_COUNT)[collection.frequency_indices]
    rows = []
    for position in positions:
        base, residual, velocity, scale, target, unselected = read_record(
            collection, int(position)
        )
        prediction = (a0 + velocity * a1) * base
        candidate_selected = float(
            np.sum(
                weights
                * scale**2
                * np.sum(np.abs(prediction - residual) ** 2, axis=(1, 2))
            )
        )
        parent_selected = float(
            np.sum(
                weights * scale**2 * np.sum(np.abs(residual) ** 2, axis=(1, 2))
            )
        )
        file_index, local_index = collection.records[int(position)]
        handle = collection.handles[file_index]
        rows.append(
            {
                "sample_id": str(handle["sample_id"].asstr()[local_index]),
                "group_id": str(handle["group_id"].asstr()[local_index]),
                "family": str(handle["family"].asstr()[local_index]),
                "candidate_proxy_rel_l2": math.sqrt(
                    (unselected + candidate_selected) / max(target, 1.0e-30)
                ),
                "parent_proxy_rel_l2": math.sqrt(
                    (unselected + parent_selected) / max(target, 1.0e-30)
                ),
            }
        )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], candidate_key: str, parent_key: str) -> dict:
    candidate = np.asarray([float(row[candidate_key]) for row in rows])
    parent = np.asarray([float(row[parent_key]) for row in rows])
    return {
        "count": len(rows),
        "candidate_mean": float(candidate.mean()),
        "candidate_max": float(candidate.max()),
        "candidate_median": float(np.median(candidate)),
        "parent_mean": float(parent.mean()),
        "parent_max": float(parent.max()),
        "mean_relative_improvement": float(1.0 - candidate.mean() / parent.mean()),
        "max_relative_improvement": float(1.0 - candidate.max() / parent.max()),
    }


def exact_holdout_rows(
    collection: r40.FrequencyCacheCollection,
    *,
    a0: np.ndarray,
    a1: np.ndarray,
) -> list[dict[str, Any]]:
    weights = r40.rfft_weights(r40.TIME_COUNT)[collection.frequency_indices]
    retained = int(collection.retained)
    rows = []
    for position, (file_index, local_index) in enumerate(collection.records):
        handle = collection.handles[file_index]
        base_coeff, _, velocity, scale, target, unselected = read_record(
            collection, position
        )
        prediction_coeff = (a0 + velocity * a1) * base_coeff
        prediction_coeff *= scale[:, None, None]
        base_full = complex_channels(handle["base_spectrum_selected"][local_index])
        truth_full = complex_channels(handle["truth_spectrum_selected"][local_index])
        grid = base_full.shape[-1]
        padded = np.zeros((collection.frequency_count, grid, grid), dtype=np.complex64)
        padded[:, :retained, :retained] = prediction_coeff
        correction = idctn(
            padded.real, type=2, norm="ortho", axes=(-2, -1)
        ) + 1j * idctn(padded.imag, type=2, norm="ortho", axes=(-2, -1))
        candidate_selected = float(
            np.sum(
                weights
                * np.sum(
                    np.abs(base_full + correction - truth_full) ** 2,
                    axis=(1, 2),
                    dtype=np.float64,
                )
            )
        )
        parent_selected = float(
            np.sum(
                weights
                * np.sum(
                    np.abs(base_full - truth_full) ** 2,
                    axis=(1, 2),
                    dtype=np.float64,
                )
            )
        )
        candidate = math.sqrt((unselected + candidate_selected) / target)
        parent = math.sqrt((unselected + parent_selected) / target)
        rows.append(
            {
                "sample_id": str(handle["sample_id"].asstr()[local_index]),
                "group_id": str(handle["group_id"].asstr()[local_index]),
                "family": str(handle["family"].asstr()[local_index]),
                "candidate_rel_l2": candidate,
                "parent_rel_l2": parent,
                "relative_improvement": 1.0 - candidate / parent,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    fit = r40.FrequencyCacheCollection(args.fit_cache, expected_subset="fit")
    holdout = r40.FrequencyCacheCollection(
        args.holdout_cache, expected_subset="holdout"
    )
    try:
        if fit.selection_sha256 != holdout.selection_sha256:
            raise RuntimeError("fit/holdout selection mismatch")
        if set(fit.group_ids) & set(holdout.group_ids):
            raise RuntimeError("fit/holdout group leakage")
        fit_rows = record_rows(fit)
        estimation, calibration, split = split_fit_groups(fit_rows)
        preregistration = {
            "schema": "r41_dispersion_transfer_preregistration_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": r40.sha256_file(Path(__file__).resolve()),
            "selection_sha256": fit.selection_sha256,
            "internal_split": split,
            "candidate_model_types": list(MODEL_TYPES),
            "candidate_ridge_fractions": list(RIDGE_FRACTIONS),
            "selection_score": "candidate_proxy_max_plus_0p25_mean",
            "final_gate": {"mean_lte_0p05": True, "max_lte_0p05": True},
            "r29b_opened": False,
            "final_validation_opened": False,
            "test_id_opened": False,
        }
        preregistration["preregistration_sha256"] = canonical_sha(preregistration)
        atomic_json(preregistration, output_dir / "preregistration.json")

        estimation_stats = accumulate_statistics(fit, estimation)
        candidates = []
        solved: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
        for model_type in MODEL_TYPES:
            for ridge_fraction in RIDGE_FRACTIONS:
                a0, a1 = solve_filter(
                    estimation_stats,
                    model_type=model_type,
                    ridge_fraction=ridge_fraction,
                )
                solved[(model_type, ridge_fraction)] = (a0, a1)
                rows = proxy_rows(fit, calibration, a0=a0, a1=a1)
                aggregate = summarize(
                    rows, "candidate_proxy_rel_l2", "parent_proxy_rel_l2"
                )
                score = float(aggregate["candidate_max"]) + 0.25 * float(
                    aggregate["candidate_mean"]
                )
                candidates.append(
                    {
                        "model_type": model_type,
                        "ridge_fraction": ridge_fraction,
                        "score": score,
                        "aggregate": aggregate,
                    }
                )
                print(json.dumps({"event": "calibration", **candidates[-1]}, sort_keys=True), flush=True)
        selected = min(candidates, key=lambda row: float(row["score"]))
        decision = {
            "schema": "r41_dispersion_transfer_calibration_decision_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": fit.selection_sha256,
            "selected_model_type": selected["model_type"],
            "selected_ridge_fraction": selected["ridge_fraction"],
            "selected_score": selected["score"],
            "candidates": candidates,
            "holdout_truth_read_before_decision": False,
        }
        decision["decision_sha256"] = canonical_sha(decision)
        atomic_json(decision, output_dir / "calibration_decision.json")

        full_stats = accumulate_statistics(fit, range(len(fit.records)))
        a0, a1 = solve_filter(
            full_stats,
            model_type=str(selected["model_type"]),
            ridge_fraction=float(selected["ridge_fraction"]),
        )
        filter_path = output_dir / "dispersion_transfer.npz"
        temporary = output_dir / f".dispersion_transfer.tmp-{os.getpid()}.npz"
        try:
            np.savez_compressed(
                temporary,
                a0=a0,
                a1=a1,
                frequency_indices=fit.frequency_indices,
                frequency_hz=fit.frequency_hz,
                model_type=np.asarray(str(selected["model_type"])),
                ridge_fraction=np.asarray(float(selected["ridge_fraction"])),
                selection_sha256=np.asarray(str(fit.selection_sha256)),
            )
            os.replace(temporary, filter_path)
        finally:
            temporary.unlink(missing_ok=True)

        rows = exact_holdout_rows(holdout, a0=a0, a1=a1)
        aggregate = summarize(rows, "candidate_rel_l2", "parent_rel_l2")
        passed = bool(
            float(aggregate["candidate_mean"]) <= 0.05
            and float(aggregate["candidate_max"]) <= 0.05
        )
        family = {
            name: summarize(
                [row for row in rows if row["family"] == name],
                "candidate_rel_l2",
                "parent_rel_l2",
            )
            for name in r40.FAMILIES
        }
        summary = {
            "schema": "r41_dispersion_transfer_result_v1",
            "status": "pass" if passed else "failed_accuracy_gate",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": fit.selection_sha256,
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "decision_sha256": decision["decision_sha256"],
            "selected_model_type": selected["model_type"],
            "selected_ridge_fraction": selected["ridge_fraction"],
            "filter": str(filter_path),
            "filter_sha256": r40.sha256_file(filter_path),
            "aggregate": aggregate,
            "per_family": family,
            "absolute_goal": {
                "mean_lte_0p05": float(aggregate["candidate_mean"]) <= 0.05,
                "max_lte_0p05": float(aggregate["candidate_max"]) <= 0.05,
                "passed": passed,
            },
            "records": rows,
            "r29b_opened": False,
            "final_validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(summary, output_dir / "result.json")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        fit.close()
        holdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
