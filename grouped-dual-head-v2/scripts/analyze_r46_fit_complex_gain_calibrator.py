#!/usr/bin/env python3
"""Fit-only complex-gain dispersion diagnostic for the R46 decision.

Frequency-dependent complex gains are estimated only from the fit caches and
then evaluated on the already-opened, group-disjoint development caches.  The
frozen R29B/final/test partitions and manuscript are never accessed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complex_channels(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 4 or array.shape[1] != 2:
        raise ValueError(f"expected [frequency,2,y,x], got {array.shape}")
    return array[:, 0] + 1j * array[:, 1]


def smooth_delta(gain: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return gain.copy()
    if width % 2 != 1:
        raise ValueError("smoothing width must be odd")
    delta = np.asarray(gain, dtype=np.complex128) - 1.0
    radius = width // 2
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    real = np.convolve(np.pad(delta.real, radius, mode="edge"), kernel, mode="valid")
    imag = np.convolve(np.pad(delta.imag, radius, mode="edge"), kernel, mode="valid")
    return 1.0 + real + 1j * imag


def gain_variant(gain: np.ndarray, mode: str, smooth: int) -> np.ndarray:
    value = smooth_delta(gain, smooth)
    if mode == "complex":
        return value
    if mode == "phase":
        return np.exp(1j * np.angle(value))
    if mode == "amplitude":
        return np.abs(value).astype(np.complex128)
    raise ValueError(mode)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([row["candidate_rel_l2"] for row in records], dtype=np.float64)
    parents = np.asarray([row["parent_rel_l2"] for row in records], dtype=np.float64)
    return {
        "count": int(len(records)),
        "candidate_mean": float(errors.mean()),
        "candidate_median": float(np.median(errors)),
        "candidate_max": float(errors.max()),
        "parent_mean": float(parents.mean()),
        "parent_max": float(parents.max()),
        "mean_relative_improvement": float(1.0 - errors.mean() / parents.mean()),
        "max_relative_improvement": float(1.0 - errors.max() / parents.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    r40_path = args.r40_script.expanduser().resolve()
    r40 = load_module(r40_path, "r46_gain_r40")
    fit = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.fit_cache],
        expected_subset="fit",
    )
    holdout = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.holdout_cache],
        expected_subset="holdout",
    )
    try:
        if fit.selection_sha256 != holdout.selection_sha256:
            raise RuntimeError("fit/holdout selection mismatch")
        overlap = set(fit.group_ids) & set(holdout.group_ids)
        if overlap:
            raise RuntimeError(f"fit/holdout group leakage: {sorted(overlap)[:3]}")
        if not np.array_equal(fit.frequency_indices, holdout.frequency_indices):
            raise RuntimeError("fit/holdout frequency mismatch")

        frequency_count = int(fit.frequency_count)
        global_num = np.zeros(frequency_count, dtype=np.complex128)
        global_den = np.zeros(frequency_count, dtype=np.float64)
        families = sorted(set(fit.families) | set(holdout.families))
        family_num = {
            family: np.zeros(frequency_count, dtype=np.complex128)
            for family in families
        }
        family_den = {
            family: np.zeros(frequency_count, dtype=np.float64)
            for family in families
        }

        for position, (file_index, local_index) in enumerate(fit.records):
            handle = fit.handles[file_index]
            base = complex_channels(handle["base_dct_norm"][local_index])
            residual = complex_channels(handle["residual_dct_norm"][local_index])
            scale = np.asarray(
                handle["frequency_scale"][local_index], dtype=np.float64
            )[:, None, None]
            base = base * scale
            truth = (base / scale + residual) * scale
            numerator = np.sum(np.conj(base) * truth, axis=(1, 2), dtype=np.complex128)
            denominator = np.sum(np.abs(base) ** 2, axis=(1, 2), dtype=np.float64)
            global_num += numerator
            global_den += denominator
            family = str(fit.families[position])
            family_num[family] += numerator
            family_den[family] += denominator

        global_gain = global_num / np.maximum(global_den, 1.0e-30)
        family_gain = {
            family: family_num[family] / np.maximum(family_den[family], 1.0e-30)
            for family in families
        }
        frequency_weights = r40.rfft_weights(r40.TIME_COUNT)[
            holdout.frequency_indices
        ].astype(np.float64)

        specifications = []
        for scope in ("global", "family"):
            for mode in ("complex", "phase", "amplitude"):
                for smooth in (1, 3, 5, 9):
                    for scale in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
                        specifications.append(
                            {
                                "scope": scope,
                                "mode": mode,
                                "smoothing_width": smooth,
                                "correction_scale": scale,
                                "records": [],
                            }
                        )

        # Each large holdout spectrum is read exactly once.  All lightweight
        # gain candidates are evaluated while that record remains in memory.
        for position, (file_index, local_index) in enumerate(holdout.records):
            handle = holdout.handles[file_index]
            base = complex_channels(handle["base_spectrum_selected"][local_index])
            truth = complex_channels(handle["truth_spectrum_selected"][local_index])
            family = str(holdout.families[position])
            unselected = float(handle["base_error_square_unselected"][local_index])
            target_total = float(handle["target_square_total"][local_index])
            base_energy = np.sum(
                np.abs(base) ** 2, axis=(1, 2), dtype=np.float64
            )
            truth_energy = np.sum(
                np.abs(truth) ** 2, axis=(1, 2), dtype=np.float64
            )
            cross = np.sum(
                np.conj(base) * truth, axis=(1, 2), dtype=np.complex128
            )
            base_frequency_error = np.maximum(
                base_energy + truth_energy - 2.0 * np.real(cross), 0.0
            )
            base_selected_error = float(
                np.sum(frequency_weights * base_frequency_error, dtype=np.float64)
            )
            base_error = math.sqrt(
                (unselected + base_selected_error) / max(target_total, 1.0e-30)
            )
            fitted_cache = {}
            for scope in ("global", "family"):
                source_gain = global_gain if scope == "global" else family_gain[family]
                for mode in ("complex", "phase", "amplitude"):
                    for smooth in (1, 3, 5, 9):
                        fitted_cache[(scope, mode, smooth)] = gain_variant(
                            source_gain, mode, smooth
                        )
            for specification in specifications:
                fitted = fitted_cache[
                    (
                        specification["scope"],
                        specification["mode"],
                        specification["smoothing_width"],
                    )
                ]
                applied = 1.0 + float(specification["correction_scale"]) * (
                    fitted - 1.0
                )
                candidate_frequency_error = np.maximum(
                    np.abs(applied) ** 2 * base_energy
                    + truth_energy
                    - 2.0 * np.real(np.conj(applied) * cross),
                    0.0,
                )
                selected_error = float(
                    np.sum(
                        frequency_weights * candidate_frequency_error,
                        dtype=np.float64,
                    )
                )
                candidate_error = math.sqrt(
                    (unselected + selected_error) / max(target_total, 1.0e-30)
                )
                specification["records"].append(
                    {
                        "sample_id": str(holdout.sample_ids[position]),
                        "group_id": str(holdout.group_ids[position]),
                        "family": family,
                        "parent_rel_l2": base_error,
                        "candidate_rel_l2": candidate_error,
                        "relative_improvement": float(
                            1.0 - candidate_error / base_error
                        ),
                    }
                )

        configurations = []
        for specification in specifications:
            records = specification.pop("records")
            metrics = aggregate(records)
            configurations.append(
                {
                    **specification,
                    "aggregate": metrics,
                    "absolute_goal": {
                        "mean_lte_0p05": metrics["candidate_mean"] <= 0.05,
                        "max_lte_0p05": metrics["candidate_max"] <= 0.05,
                        "passed": metrics["candidate_mean"] <= 0.05
                        and metrics["candidate_max"] <= 0.05,
                    },
                    "records": records,
                }
            )

        best = min(
            configurations,
            key=lambda item: item["aggregate"]["candidate_max"]
            + 0.1 * item["aggregate"]["candidate_mean"],
        )
        output = {
            "schema": "r46_fit_complex_gain_calibrator_diagnostic_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": fit.selection_sha256,
            "fit_record_count": len(fit.records),
            "fit_group_count": len(set(fit.group_ids)),
            "development_record_count": len(holdout.records),
            "development_group_count": len(set(holdout.group_ids)),
            "fit_development_group_overlap": 0,
            "best": best,
            "configurations": configurations,
            "fit_gain_summary": {
                "global_max_abs_delta": float(np.max(np.abs(global_gain - 1.0))),
                "global_mean_abs_delta": float(np.mean(np.abs(global_gain - 1.0))),
                "family_max_abs_delta": {
                    family: float(np.max(np.abs(family_gain[family] - 1.0)))
                    for family in families
                },
            },
            "data_boundary": {
                "fit_truth_used_for_gain_estimation": True,
                "opened_group_disjoint_development_used_for_diagnosis": True,
                "r29b_opened": False,
                "final_validation_opened": False,
                "test_id_opened": False,
                "paper_modified": False,
            },
            "evidence": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "r40_script": str(r40_path),
                "r40_script_sha256": sha256_file(r40_path),
            },
        }
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(output_path)
        print(
            json.dumps(
                {
                    "event": "r46_fit_complex_gain_complete",
                    "best_scope": best["scope"],
                    "best_mode": best["mode"],
                    "best_smoothing_width": best["smoothing_width"],
                    "best_correction_scale": best["correction_scale"],
                    "candidate_mean": best["aggregate"]["candidate_mean"],
                    "candidate_max": best["aggregate"]["candidate_max"],
                    "absolute_goal_passed": best["absolute_goal"]["passed"],
                    "worst_record": max(
                        best["records"], key=lambda row: row["candidate_rel_l2"]
                    ),
                    "output": str(output_path),
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        fit.close()
        holdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
