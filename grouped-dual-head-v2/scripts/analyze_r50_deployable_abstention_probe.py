#!/usr/bin/env python3
"""Deployable difficulty detector / abstention probe for a possible R50.

R49 falsified spatial gain transfer: truth-derived per-group gains decorrelate
below the 50 m fit grid spacing, so no fit-distribution densification can close
the 0.05 development maximum gate.  The remaining sanctioned route is selective
prediction: a deployable difficulty score with an abstention threshold chosen on
fit truth only, evaluated for transfer on the opened group-disjoint development
set.  Features use exclusively deployment-available quantities (parent spectrum,
static medium channels, source parameters).  Truth-derived quantities
(residuals, unselected error, target totals) are used only for labels on fit and
for diagnosis on development.  R29B, final validation, test data, and manuscript
files remain frozen.
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


def dct_band_masks(retained: int) -> list[np.ndarray]:
    order = np.add.outer(np.arange(retained), np.arange(retained))
    return [
        order < retained // 4,
        (order >= retained // 4) & (order < retained // 2),
        order >= retained // 2,
    ]


FEATURE_NAMES = [
    "log_parent_weighted_energy",
    "parent_band_low_fraction",
    "parent_band_mid_fraction",
    "parent_band_high_fraction",
    "parent_frequency_centroid_hz",
    "parent_top_third_frequency_fraction",
    "static0_high_fraction",
    "static3_high_fraction",
    "static4_high_fraction",
    "static5_high_fraction",
    "static6_high_fraction",
    "source_f0_hz",
    "source_t0_s",
]


def record_features(
    handle, local_index: int, weights: np.ndarray, frequency_hz: np.ndarray,
    band_masks: list[np.ndarray],
) -> np.ndarray:
    base = complex_channels(handle["base_dct_norm"][local_index])
    scale = np.asarray(
        handle["frequency_scale"][local_index], dtype=np.float64
    )[:, None, None]
    base = base * scale
    power = np.abs(base) ** 2
    per_frequency = np.sum(power, axis=(1, 2), dtype=np.float64)
    weighted_total = float(np.sum(weights * per_frequency, dtype=np.float64))
    weighted_total = max(weighted_total, 1.0e-30)
    band_fractions = [
        float(
            np.sum(weights * np.sum(power[:, mask], axis=1), dtype=np.float64)
            / weighted_total
        )
        for mask in band_masks
    ]
    frequency_weighted = weights * per_frequency
    centroid = float(
        np.sum(frequency_hz * frequency_weighted, dtype=np.float64)
        / max(np.sum(frequency_weighted, dtype=np.float64), 1.0e-30)
    )
    top_third = frequency_hz >= np.percentile(frequency_hz, 200.0 / 3.0)
    top_fraction = float(
        np.sum(frequency_weighted[top_third], dtype=np.float64)
        / max(np.sum(frequency_weighted, dtype=np.float64), 1.0e-30)
    )
    static = np.asarray(handle["static_dct_norm"][local_index], dtype=np.float64)
    static_features = []
    for channel in (0, 3, 4, 5, 6):
        channel_power = static[channel] ** 2
        total = max(float(np.sum(channel_power, dtype=np.float64)), 1.0e-30)
        static_features.append(
            float(np.sum(channel_power[band_masks[2]], dtype=np.float64) / total)
        )
    return np.asarray(
        [
            math.log(weighted_total),
            band_fractions[0],
            band_fractions[1],
            band_fractions[2],
            centroid,
            top_fraction,
            *static_features,
            float(handle["source_f0_hz"][local_index]),
            float(handle["source_t0_s"][local_index]),
        ],
        dtype=np.float64,
    )


def parent_rel_l2(handle, local_index: int, weights: np.ndarray) -> float:
    residual = complex_channels(handle["residual_dct_norm"][local_index])
    scale = np.asarray(
        handle["frequency_scale"][local_index], dtype=np.float64
    )[:, None, None]
    residual = residual * scale
    retained = float(
        np.sum(
            weights * np.sum(np.abs(residual) ** 2, axis=(1, 2), dtype=np.float64),
            dtype=np.float64,
        )
    )
    unselected = float(handle["base_error_square_unselected"][local_index])
    total = float(handle["target_square_total"][local_index])
    return math.sqrt((unselected + retained) / max(total, 1.0e-30))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        rank = np.empty_like(order, dtype=np.float64)
        rank[order] = np.arange(len(values), dtype=np.float64)
        unique, inverse, counts = np.unique(
            values, return_inverse=True, return_counts=True
        )
        cumulative = np.cumsum(counts) - counts
        mean_rank = cumulative + (counts - 1) / 2.0
        return mean_rank[inverse]

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = math.sqrt(float(np.sum(ra**2)) * float(np.sum(rb**2)))
    return float(np.sum(ra * rb) / max(denominator, 1.0e-30))


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
    r40 = load_module(r40_path, "r50_abstain_r40")
    union_module = load_module(union_path, "r50_abstain_union")
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
    ridge_lambda = 1.0
    fit_abstain_label = 0.045
    try:
        if str(fit.component_selection_sha256[0]) != str(holdout.selection_sha256):
            raise RuntimeError("base fit/holdout selection mismatch")
        if set(fit.group_ids) & set(holdout.group_ids):
            raise RuntimeError("fit/holdout group leakage")
        if not np.array_equal(fit.frequency_indices, holdout.frequency_indices):
            raise RuntimeError("fit/holdout frequency mismatch")

        weights = r40.rfft_weights(r40.TIME_COUNT)[holdout.frequency_indices].astype(
            np.float64
        )
        frequency_hz = np.fft.rfftfreq(r40.TIME_COUNT, d=float(fit.stored_dt_s))[
            holdout.frequency_indices
        ].astype(np.float64)
        band_masks = dct_band_masks(int(fit.retained))

        fit_features, fit_labels, fit_meta = [], [], []
        for position, (file_index, local_index) in enumerate(fit.records):
            handle = fit.handles[file_index]
            fit_features.append(
                record_features(handle, local_index, weights, frequency_hz, band_masks)
            )
            fit_labels.append(parent_rel_l2(handle, local_index, weights))
            fit_meta.append(
                {
                    "sample_id": str(fit.sample_ids[position]),
                    "group_id": str(fit.group_ids[position]),
                    "family": str(fit.families[position]),
                }
            )
        fit_features = np.asarray(fit_features)
        fit_labels = np.asarray(fit_labels)

        dev_features, dev_labels, dev_meta = [], [], []
        for position, (file_index, local_index) in enumerate(holdout.records):
            handle = holdout.handles[file_index]
            dev_features.append(
                record_features(handle, local_index, weights, frequency_hz, band_masks)
            )
            dev_labels.append(parent_rel_l2(handle, local_index, weights))
            dev_meta.append(
                {
                    "sample_id": str(holdout.sample_ids[position]),
                    "group_id": str(holdout.group_ids[position]),
                    "family": str(holdout.families[position]),
                }
            )
        dev_features = np.asarray(dev_features)
        dev_labels = np.asarray(dev_labels)

        mean = fit_features.mean(axis=0)
        std = np.maximum(fit_features.std(axis=0), 1.0e-12)
        fit_standardized = (fit_features - mean) / std
        dev_standardized = (dev_features - mean) / std

        targets = np.log(np.maximum(fit_labels, 1.0e-6))
        design = np.concatenate(
            [fit_standardized, np.ones((len(fit_standardized), 1))], axis=1
        )
        regularizer = ridge_lambda * np.eye(design.shape[1])
        regularizer[-1, -1] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + regularizer, design.T @ targets
        )
        fit_scores = design @ coefficients
        dev_scores = (
            np.concatenate(
                [dev_standardized, np.ones((len(dev_standardized), 1))], axis=1
            )
            @ coefficients
        )

        per_feature = [
            {
                "feature": FEATURE_NAMES[index],
                "fit_spearman_vs_parent": spearman(
                    fit_features[:, index], fit_labels
                ),
                "dev_spearman_vs_parent": spearman(
                    dev_features[:, index], dev_labels
                ),
                "ridge_coefficient": float(coefficients[index]),
            }
            for index in range(len(FEATURE_NAMES))
        ]

        # Fit-only threshold: abstain everything the detector must flag so that
        # all fit records with parent rel L2 above the label line are excluded.
        must_flag = fit_labels > fit_abstain_label
        if not np.any(must_flag):
            raise RuntimeError("no fit records above abstention label line")
        threshold = float(np.min(fit_scores[must_flag]))
        fit_abstained = fit_scores >= threshold
        dev_abstained = dev_scores >= threshold
        kept_dev = dev_labels[~dev_abstained]

        def coverage_report(abstained, labels):
            kept = labels[~abstained]
            return {
                "abstention_rate": float(np.mean(abstained)),
                "kept_count": int(len(kept)),
                "kept_mean": float(kept.mean()) if len(kept) else None,
                "kept_max": float(kept.max()) if len(kept) else None,
                "kept_max_lte_0p05": bool(kept.max() <= 0.05) if len(kept) else None,
                "missed_over_gate": int(np.sum(labels[~abstained] > 0.05)),
            }

        # Oracle ranking diagnostics on development.
        order = np.argsort(-dev_scores)
        ranked = [
            {
                "rank": int(rank + 1),
                "sample_id": dev_meta[index]["sample_id"],
                "group_id": dev_meta[index]["group_id"],
                "family": dev_meta[index]["family"],
                "detector_score": float(dev_scores[index]),
                "parent_rel_l2": float(dev_labels[index]),
            }
            for rank, index in enumerate(order)
        ]
        over_gate_ranks = [
            row["rank"] for row in ranked if row["parent_rel_l2"] > 0.05
        ]
        minimal_abstention_count = max(over_gate_ranks) if over_gate_ranks else 0

        payload = {
            "schema": "r50_deployable_abstention_probe_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "feature_policy": (
                "parent spectrum statistics, static medium channels, and"
                " source parameters only; truth-derived quantities are"
                " excluded from features and used solely as fit labels and"
                " development diagnosis"
            ),
            "ridge_lambda": ridge_lambda,
            "fit_abstain_label": fit_abstain_label,
            "fit_record_count": len(fit_labels),
            "development_record_count": len(dev_labels),
            "fit_development_group_overlap": 0,
            "detector": {
                "fit_spearman": spearman(fit_scores, fit_labels),
                "dev_spearman": spearman(dev_scores, dev_labels),
                "coefficients": {
                    FEATURE_NAMES[index]: float(coefficients[index])
                    for index in range(len(FEATURE_NAMES))
                },
                "intercept": float(coefficients[-1]),
            },
            "per_feature": per_feature,
            "fit_only_threshold": {
                "threshold": threshold,
                "fit": coverage_report(fit_abstained, fit_labels),
                "development": coverage_report(dev_abstained, dev_labels),
            },
            "development_ranking": ranked,
            "development_over_gate_ranks": over_gate_ranks,
            "development_minimal_abstention_count_to_cover": minimal_abstention_count,
            "data_boundary": {
                "fit_truth_used_for_labels": True,
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
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "event": "r50_deployable_abstention_probe_complete",
                    "detector_fit_spearman": payload["detector"]["fit_spearman"],
                    "detector_dev_spearman": payload["detector"]["dev_spearman"],
                    "fit_only_threshold": payload["fit_only_threshold"],
                    "development_over_gate_ranks": over_gate_ranks,
                    "development_minimal_abstention_count_to_cover": minimal_abstention_count,
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
