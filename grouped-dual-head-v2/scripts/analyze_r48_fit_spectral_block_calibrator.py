#!/usr/bin/env python3
"""Fit-only spatial-wavenumber block calibration for a possible R48.

Complex gains are estimated solely from the audited R46 fit union in retained
spatial-DCT space.  The opened group-disjoint development set is used only to
diagnose transfer and select a correction scale.  R29B, final validation, test
data, and manuscript files remain frozen.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

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


def complex_channels(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 4 or array.shape[1] != 2:
        raise ValueError(f"expected [frequency,2,y,x], got {array.shape}")
    return array[:, 0] + 1j * array[:, 1]


def block_sufficient_statistics(
    base: np.ndarray, truth: np.ndarray, blocks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency_count, height, width = base.shape
    if height % blocks != 0 or width % blocks != 0:
        raise ValueError("block count must divide retained DCT geometry")
    block_height, block_width = height // blocks, width // blocks
    base_blocks = base.reshape(
        frequency_count, blocks, block_height, blocks, block_width
    ).transpose(0, 1, 3, 2, 4)
    truth_blocks = truth.reshape(
        frequency_count, blocks, block_height, blocks, block_width
    ).transpose(0, 1, 3, 2, 4)
    base_energy = np.sum(
        np.abs(base_blocks) ** 2, axis=(-2, -1), dtype=np.float64
    )
    truth_energy = np.sum(
        np.abs(truth_blocks) ** 2, axis=(-2, -1), dtype=np.float64
    )
    cross = np.sum(
        np.conj(base_blocks) * truth_blocks,
        axis=(-2, -1),
        dtype=np.complex128,
    )
    return base_energy, truth_energy, cross


def smooth_delta(gain: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.asarray(gain, dtype=np.complex128).copy()
    if width % 2 != 1:
        raise ValueError("smoothing width must be odd")
    delta = np.asarray(gain, dtype=np.complex128) - 1.0
    radius = width // 2
    padded = np.pad(delta, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    smoothed = np.zeros_like(delta)
    for offset in range(width):
        smoothed += padded[offset : offset + len(delta)]
    return 1.0 + smoothed / float(width)


def bounded_gain(gain: np.ndarray, *, smooth: int, cap: float) -> np.ndarray:
    value = smooth_delta(gain, smooth)
    delta = value - 1.0
    clipped = np.clip(delta.real, -cap, cap) + 1j * np.clip(
        delta.imag, -cap, cap
    )
    return 1.0 + clipped


def aggregate(records: list[dict]) -> dict:
    candidates = np.asarray(
        [row["candidate_rel_l2"] for row in records], dtype=np.float64
    )
    parents = np.asarray(
        [row["parent_rel_l2"] for row in records], dtype=np.float64
    )
    return {
        "count": len(records),
        "candidate_mean": float(candidates.mean()),
        "candidate_median": float(np.median(candidates)),
        "candidate_max": float(candidates.max()),
        "parent_mean": float(parents.mean()),
        "parent_max": float(parents.max()),
        "mean_relative_improvement": float(1.0 - candidates.mean() / parents.mean()),
        "max_relative_improvement": float(1.0 - candidates.max() / parents.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--union-script", type=Path, required=True)
    parser.add_argument("--base-fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--supplement-fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    r40_path = args.r40_script.expanduser().resolve()
    union_path = args.union_script.expanduser().resolve()
    r40 = load_module(r40_path, "r48_fit_spectral_r40")
    union_module = load_module(union_path, "r48_fit_spectral_union")
    fit = union_module.UnionFrequencyCacheCollection(
        r40,
        [
            [path.expanduser().resolve() for path in args.base_fit_cache],
            [path.expanduser().resolve() for path in args.supplement_fit_cache],
        ],
        expected_subset="fit",
    )
    holdout = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.holdout_cache],
        expected_subset="holdout",
    )
    block_counts = (4, 8)
    smoothing_widths = (1, 3, 5)
    correction_scales = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    gain_cap = 0.5
    try:
        if str(fit.component_selection_sha256[0]) != str(
            holdout.selection_sha256
        ):
            raise RuntimeError("base fit/holdout selection mismatch")
        if set(fit.group_ids) & set(holdout.group_ids):
            raise RuntimeError("fit/holdout group leakage")
        if int(fit.retained) != int(holdout.retained):
            raise RuntimeError("fit/holdout retained DCT mismatch")
        if not np.array_equal(fit.frequency_indices, holdout.frequency_indices):
            raise RuntimeError("fit/holdout frequency mismatch")

        families = sorted(set(fit.families) | set(holdout.families))
        global_num = {
            blocks: np.zeros(
                (fit.frequency_count, blocks, blocks), dtype=np.complex128
            )
            for blocks in block_counts
        }
        global_den = {
            blocks: np.zeros(
                (fit.frequency_count, blocks, blocks), dtype=np.float64
            )
            for blocks in block_counts
        }
        family_num = {
            family: {
                blocks: np.zeros(
                    (fit.frequency_count, blocks, blocks), dtype=np.complex128
                )
                for blocks in block_counts
            }
            for family in families
        }
        family_den = {
            family: {
                blocks: np.zeros(
                    (fit.frequency_count, blocks, blocks), dtype=np.float64
                )
                for blocks in block_counts
            }
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
            truth = base + residual * scale
            family = str(fit.families[position])
            for blocks in block_counts:
                base_energy, _truth_energy, cross = block_sufficient_statistics(
                    base, truth, blocks
                )
                global_num[blocks] += cross
                global_den[blocks] += base_energy
                family_num[family][blocks] += cross
                family_den[family][blocks] += base_energy

        global_gain = {
            blocks: global_num[blocks]
            / np.maximum(global_den[blocks], 1.0e-30)
            for blocks in block_counts
        }
        family_gain = {
            family: {
                blocks: family_num[family][blocks]
                / np.maximum(family_den[family][blocks], 1.0e-30)
                for blocks in block_counts
            }
            for family in families
        }
        fitted_gain = {}
        for scope in ("global", "family"):
            for blocks in block_counts:
                for smooth in smoothing_widths:
                    if scope == "global":
                        fitted_gain[(scope, None, blocks, smooth)] = bounded_gain(
                            global_gain[blocks], smooth=smooth, cap=gain_cap
                        )
                    else:
                        for family in families:
                            fitted_gain[(scope, family, blocks, smooth)] = bounded_gain(
                                family_gain[family][blocks],
                                smooth=smooth,
                                cap=gain_cap,
                            )

        specifications = []
        for scope in ("global", "family"):
            for blocks in block_counts:
                for smooth in smoothing_widths:
                    for correction_scale in correction_scales:
                        specifications.append(
                            {
                                "scope": scope,
                                "blocks": blocks,
                                "smoothing_width": smooth,
                                "gain_cap": gain_cap,
                                "correction_scale": correction_scale,
                                "records": [],
                            }
                        )

        weights = r40.rfft_weights(r40.TIME_COUNT)[
            holdout.frequency_indices
        ].astype(np.float64)
        parent_contract_differences = []
        for position, (file_index, local_index) in enumerate(holdout.records):
            handle = holdout.handles[file_index]
            base = complex_channels(handle["base_dct_norm"][local_index])
            residual = complex_channels(handle["residual_dct_norm"][local_index])
            frequency_scale = np.asarray(
                handle["frequency_scale"][local_index], dtype=np.float64
            )[:, None, None]
            base = base * frequency_scale
            residual = residual * frequency_scale
            truth = base + residual
            base_spatial = complex_channels(
                handle["base_spectrum_selected"][local_index]
            )
            truth_spatial = complex_channels(
                handle["truth_spectrum_selected"][local_index]
            )
            unselected = float(
                handle["base_error_square_unselected"][local_index]
            )
            target_total = float(handle["target_square_total"][local_index])
            retained_parent = float(
                np.sum(
                    weights
                    * np.sum(
                        np.abs(residual) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            )
            full_parent = float(
                np.sum(
                    weights
                    * np.sum(
                        np.abs(base_spatial - truth_spatial) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            )
            omitted_spatial = max(full_parent - retained_parent, 0.0)
            parent = math.sqrt(
                (unselected + full_parent) / max(target_total, 1.0e-30)
            )
            reconstructed_parent = math.sqrt(
                (unselected + omitted_spatial + retained_parent)
                / max(target_total, 1.0e-30)
            )
            parent_contract_differences.append(abs(parent - reconstructed_parent))
            family = str(holdout.families[position])
            statistics = {
                blocks: block_sufficient_statistics(base, truth, blocks)
                for blocks in block_counts
            }
            for specification in specifications:
                blocks = int(specification["blocks"])
                base_energy, truth_energy, cross = statistics[blocks]
                family_key = family if specification["scope"] == "family" else None
                gain = fitted_gain[
                    (
                        specification["scope"],
                        family_key,
                        blocks,
                        specification["smoothing_width"],
                    )
                ]
                applied = 1.0 + float(specification["correction_scale"]) * (
                    gain - 1.0
                )
                block_error = np.maximum(
                    np.abs(applied) ** 2 * base_energy
                    + truth_energy
                    - 2.0 * np.real(np.conj(applied) * cross),
                    0.0,
                )
                retained_error = float(
                    np.sum(
                        weights * block_error.sum(axis=(1, 2)),
                        dtype=np.float64,
                    )
                )
                candidate = math.sqrt(
                    (unselected + omitted_spatial + retained_error)
                    / max(target_total, 1.0e-30)
                )
                specification["records"].append(
                    {
                        "sample_id": str(holdout.sample_ids[position]),
                        "group_id": str(holdout.group_ids[position]),
                        "family": family,
                        "parent_rel_l2": parent,
                        "candidate_rel_l2": candidate,
                        "relative_improvement": float(1.0 - candidate / parent),
                    }
                )

        configurations = []
        for specification in specifications:
            records = specification.pop("records")
            stats = aggregate(records)
            configurations.append(
                {
                    **specification,
                    "aggregate": stats,
                    "absolute_goal": {
                        "mean_lte_0p05": stats["candidate_mean"] <= 0.05,
                        "max_lte_0p05": stats["candidate_max"] <= 0.05,
                        "passed": stats["candidate_mean"] <= 0.05
                        and stats["candidate_max"] <= 0.05,
                    },
                    "worst_record": max(
                        records, key=lambda row: row["candidate_rel_l2"]
                    ),
                    "records": records,
                }
            )
        score = lambda item: item["aggregate"]["candidate_max"] + 0.1 * item[
            "aggregate"
        ]["candidate_mean"]
        global_configurations = [
            item for item in configurations if item["scope"] == "global"
        ]
        family_configurations = [
            item for item in configurations if item["scope"] == "family"
        ]
        best_global = min(global_configurations, key=score)
        best_family = min(family_configurations, key=score)
        payload = {
            "schema": "r48_fit_spectral_block_calibrator_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "deployable_global_scope": True,
            "family_scope_role": "diagnostic upper bound because family metadata may not be available for arbitrary deployment inputs",
            "selection_sha256": fit.selection_sha256,
            "component_selection_sha256": list(fit.component_selection_sha256),
            "fit_record_count": len(fit.records),
            "fit_group_count": len(set(fit.group_ids)),
            "development_record_count": len(holdout.records),
            "fit_development_group_overlap": 0,
            "retained_dct": int(fit.retained),
            "gain_cap": gain_cap,
            "maximum_parent_energy_contract_difference": float(
                max(parent_contract_differences, default=0.0)
            ),
            "best_global": best_global,
            "best_family": best_family,
            "configurations": configurations,
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
                "union_script": str(union_path),
                "union_script_sha256": sha256_file(union_path),
            },
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "event": "r48_fit_spectral_block_calibrator_complete",
                    "best_global": {
                        key: best_global[key]
                        for key in (
                            "blocks",
                            "smoothing_width",
                            "gain_cap",
                            "correction_scale",
                            "aggregate",
                            "absolute_goal",
                            "worst_record",
                        )
                    },
                    "best_family": {
                        key: best_family[key]
                        for key in (
                            "blocks",
                            "smoothing_width",
                            "gain_cap",
                            "correction_scale",
                            "aggregate",
                            "absolute_goal",
                            "worst_record",
                        )
                    },
                    "output": str(output),
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
