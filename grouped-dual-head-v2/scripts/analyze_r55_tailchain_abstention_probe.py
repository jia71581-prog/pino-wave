"""Deployable difficulty detector / abstention probe for the tail chain.

R50's probe consumed the R40/R46 DCT cache (base_dct_norm, residual_dct_norm,
frequency_scale, static_dct_norm).  The tail chain (r38 full-coverage cache) is
real-space instead: coarse_norm, truth_norm, static_features, field_scale.  This
script re-derives the same selective-prediction rule against those keys.

Feature policy is unchanged from R50: only deployable quantities enter the
design matrix -- parent (coarse) field statistics, static medium channels and
source parameters.  Truth-derived quantities are used solely as fit labels and
as development diagnosis.  truth_frame_energy_{mean,max}_norm are truth-derived
and are therefore excluded from features.

Fit shards hold 64 time frames while holdout shards hold 401, so every temporal
feature is sampled at fixed relative fractions of the record rather than at
absolute frame indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

FRACTIONS = (0.25, 0.5, 0.75, 1.0)

FEATURE_NAMES = [
    "log_parent_energy",
    "parent_late_energy_fraction",
    "parent_energy_growth_ratio",
    "parent_spatial_spread_late",
    "parent_edge_energy_fraction_late",
    "parent_peak_to_mean_late",
    "static0_std",
    "static0_gradient_mean",
    "static3_std",
    "static4_std",
    "static5_std",
    "static6_std",
    "source_f0_hz",
    "source_t0_s",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_slots(n_frames: int) -> list[int]:
    return [min(n_frames - 1, max(0, int(round(f * (n_frames - 1))))) for f in FRACTIONS]


def record_features(handle, index: int) -> np.ndarray:
    coarse = handle["coarse_norm"]
    n_frames = coarse.shape[1]
    slots = frame_slots(n_frames)
    frames = np.asarray(coarse[index, slots], dtype=np.float64)

    per_frame_energy = np.sum(frames**2, axis=(1, 2))
    total = max(float(np.sum(per_frame_energy)), 1.0e-30)
    late = frames[-1]
    early_energy = max(float(per_frame_energy[0]), 1.0e-30)

    late_power = late**2
    late_total = max(float(np.sum(late_power)), 1.0e-30)
    ny, nx = late.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy = float(np.sum(yy * late_power) / late_total)
    cx = float(np.sum(xx * late_power) / late_total)
    spread = math.sqrt(
        float(np.sum(((yy - cy) ** 2 + (xx - cx) ** 2) * late_power) / late_total)
    )
    border = max(1, ny // 10)
    edge_mask = np.zeros_like(late_power, dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True
    edge_fraction = float(np.sum(late_power[edge_mask]) / late_total)
    late_abs_mean = max(float(np.mean(np.abs(late))), 1.0e-30)
    peak_to_mean = float(np.max(np.abs(late)) / late_abs_mean)

    static = np.asarray(handle["static_features"][index], dtype=np.float64)
    gy, gx = np.gradient(static[0])
    static_feats = [
        float(np.std(static[0])),
        float(np.mean(np.sqrt(gy**2 + gx**2))),
        float(np.std(static[3])),
        float(np.std(static[4])),
        float(np.std(static[5])),
        float(np.std(static[6])),
    ]

    return np.asarray(
        [
            math.log(total),
            float(per_frame_energy[-1] / total),
            float(per_frame_energy[-1] / early_energy),
            spread,
            edge_fraction,
            peak_to_mean,
            *static_feats,
            float(handle["source_f0_hz"][index]),
            float(handle["source_t0_s"][index]),
        ],
        dtype=np.float64,
    )


def parent_rel_l2(handle, index: int) -> float:
    baseline = float(handle["baseline_error_square_norm"][index])
    target = max(float(handle["target_square_norm"][index]), 1.0e-30)
    return math.sqrt(max(baseline, 0.0) / target)


def decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def collect(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features, labels, meta = [], [], []
    for path in paths:
        with h5py.File(path, "r") as handle:
            for index in range(handle["family"].shape[0]):
                features.append(record_features(handle, index))
                labels.append(parent_rel_l2(handle, index))
                meta.append(
                    {
                        "shard": path.name,
                        "sample_id": decode(handle["sample_id"][index]),
                        "group_id": decode(handle["group_id"][index]),
                        "family": decode(handle["family"][index]),
                    }
                )
    return np.asarray(features), np.asarray(labels), meta


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = math.sqrt(float(np.sum(ra**2) * np.sum(rb**2)))
    return float(np.sum(ra * rb) / denominator) if denominator > 0 else float("nan")


def coverage_report(abstained: np.ndarray, labels: np.ndarray, gate: float) -> dict:
    kept = labels[~abstained]
    return {
        "abstention_rate": float(np.mean(abstained)) if len(labels) else float("nan"),
        "kept_count": int(len(kept)),
        "kept_mean": float(np.mean(kept)) if len(kept) else float("nan"),
        "kept_max": float(np.max(kept)) if len(kept) else float("nan"),
        "kept_max_lte_gate": bool(len(kept) and float(np.max(kept)) <= gate),
        "missed_over_gate": int(np.sum(kept > gate)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=float, default=0.05)
    parser.add_argument("--fit-abstain-label", type=float, default=0.045)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--conformal-quantile", type=float, default=99.0)
    args = parser.parse_args()

    fit_features, fit_labels, fit_meta = collect(list(args.fit_cache))
    dev_features, dev_labels, dev_meta = collect(list(args.holdout_cache))

    fit_groups = {row["group_id"] for row in fit_meta}
    dev_groups = {row["group_id"] for row in dev_meta}

    mean = fit_features.mean(axis=0)
    std = np.maximum(fit_features.std(axis=0), 1.0e-12)
    fit_standardized = (fit_features - mean) / std
    dev_standardized = (dev_features - mean) / std

    targets = np.log(np.maximum(fit_labels, 1.0e-6))
    design = np.concatenate([fit_standardized, np.ones((len(fit_standardized), 1))], axis=1)
    regularizer = args.ridge_lambda * np.eye(design.shape[1])
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)

    fit_scores = design @ coefficients
    dev_design = np.concatenate(
        [dev_standardized, np.ones((len(dev_standardized), 1))], axis=1
    )
    dev_scores = dev_design @ coefficients

    # Rule A: fit-only hard threshold (R50 headline rule).
    must_flag = fit_labels > args.fit_abstain_label
    if bool(np.any(must_flag)):
        threshold = float(np.min(fit_scores[must_flag]))
        rule_a = {
            "threshold": threshold,
            "fit": coverage_report(fit_scores >= threshold, fit_labels, args.gate),
            "development": coverage_report(dev_scores >= threshold, dev_labels, args.gate),
        }
    else:
        rule_a = {"threshold": None, "reason": "no fit record above fit_abstain_label"}

    # Rule B: fit-only conformal margin on the multiplicative residual.
    fit_predicted = np.exp(fit_scores)
    ratio = np.maximum(fit_labels, 1.0e-12) / np.maximum(fit_predicted, 1.0e-12)
    margin = float(np.percentile(ratio, args.conformal_quantile))
    dev_predicted = np.exp(dev_scores)
    rule_b = {
        "conformal_quantile": args.conformal_quantile,
        "margin": margin,
        "abstain_if": "predicted * margin >= gate",
        "fit": coverage_report(fit_predicted * margin >= args.gate, fit_labels, args.gate),
        "development": coverage_report(
            dev_predicted * margin >= args.gate, dev_labels, args.gate
        ),
    }

    order = np.argsort(-dev_scores)
    ranking = [
        {
            "rank": int(position),
            "sample_id": dev_meta[int(index)]["sample_id"],
            "family": dev_meta[int(index)]["family"],
            "detector_score": float(dev_scores[int(index)]),
            "predicted_rel_l2": float(dev_predicted[int(index)]),
            "parent_rel_l2": float(dev_labels[int(index)]),
        }
        for position, index in enumerate(order)
    ]

    per_family = {}
    for family in sorted({row["family"] for row in dev_meta}):
        mask = np.asarray([row["family"] == family for row in dev_meta])
        per_family[family] = {
            "count": int(np.sum(mask)),
            "parent_max": float(np.max(dev_labels[mask])),
            "conformal": coverage_report(
                (dev_predicted * margin >= args.gate)[mask], dev_labels[mask], args.gate
            ),
        }

    payload = {
        "schema": "r55_tailchain_abstention_probe_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": args.gate,
        "ridge_lambda": args.ridge_lambda,
        "fit_abstain_label": args.fit_abstain_label,
        "feature_names": FEATURE_NAMES,
        "temporal_sampling_fractions": list(FRACTIONS),
        "feature_policy": (
            "parent (coarse) field statistics, static medium channels and source "
            "parameters only; truth-derived quantities including "
            "truth_frame_energy_{mean,max}_norm are excluded from features and used "
            "solely as fit labels and development diagnosis"
        ),
        "counts": {"fit": int(len(fit_labels)), "development": int(len(dev_labels))},
        "fit_development_group_overlap": int(len(fit_groups & dev_groups)),
        "detector": {
            "intercept": float(coefficients[-1]),
            "feature_mean": [float(v) for v in mean],
            "feature_std": [float(v) for v in std],
            "coefficients": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, coefficients[:-1])
            },
            "fit_spearman": spearman(fit_scores, fit_labels),
            "dev_spearman": spearman(dev_scores, dev_labels),
        },
        "fit_only_threshold": rule_a,
        "fit_only_conformal": rule_b,
        "per_family_development": per_family,
        "development_ranking": ranking,
        "data_boundary": {
            "final_validation_opened": False,
            "test_id_opened": False,
            "fit_truth_used_for_labels": True,
            "opened_group_disjoint_development_used_for_diagnosis": True,
            "paper_modified": False,
        },
        "evidence": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "fit_cache": [str(p.resolve()) for p in args.fit_cache],
            "holdout_cache": [str(p.resolve()) for p in args.holdout_cache],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
