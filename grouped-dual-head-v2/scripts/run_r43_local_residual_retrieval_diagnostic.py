#!/usr/bin/env python3
"""Opened-development diagnostic for local train-residual retrieval.

The method retrieves group-disjoint fit records using only deployable source,
wavelet, family, and parent-wavefield information.  Retrieved retained-DCT
residuals are source-shifted, aligned to the target parent spectrum, and mixed.
The opened development set selects one global configuration; no per-record
truth-derived switch or coefficient is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.fft import dctn, idctn


COORDINATE_PATTERN = re.compile(r":x(-?[0-9.]+):z(-?[0-9.]+)$")


def load_r40(path: Path):
    spec = importlib.util.spec_from_file_location("r40_retrieval", path)
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


def parse_coordinates(group_id: str) -> tuple[float, float] | None:
    match = COORDINATE_PATTERN.search(group_id)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def translate_zero(value: np.ndarray, shift_z: int, shift_x: int) -> np.ndarray:
    result = np.zeros_like(value)
    height, width = value.shape[-2:]
    source_z0 = max(-int(shift_z), 0)
    source_z1 = min(height - int(shift_z), height)
    target_z0 = max(int(shift_z), 0)
    target_z1 = min(height + int(shift_z), height)
    source_x0 = max(-int(shift_x), 0)
    source_x1 = min(width - int(shift_x), width)
    target_x0 = max(int(shift_x), 0)
    target_x1 = min(width + int(shift_x), width)
    if source_z1 > source_z0 and source_x1 > source_x0:
        result[..., target_z0:target_z1, target_x0:target_x1] = value[
            ..., source_z0:source_z1, source_x0:source_x1
        ]
    return result


def shift_coefficients(
    coefficients: np.ndarray,
    *,
    grid: int,
    retained: int,
    shift_z: int,
    shift_x: int,
) -> np.ndarray:
    padded = np.zeros((coefficients.shape[0], grid, grid), dtype=np.complex64)
    padded[:, :retained, :retained] = coefficients
    spatial = idctn(padded.real, type=2, norm="ortho", axes=(-2, -1)) + 1j * idctn(
        padded.imag, type=2, norm="ortho", axes=(-2, -1)
    )
    shifted = translate_zero(spatial, shift_z, shift_x)
    return (
        dctn(shifted.real, type=2, norm="ortho", axes=(-2, -1))
        + 1j * dctn(shifted.imag, type=2, norm="ortho", axes=(-2, -1))
    )[:, :retained, :retained].astype(np.complex64)


def align_scalar(
    neighbor_base: np.ndarray,
    target_base: np.ndarray,
    *,
    maximum_magnitude: float,
) -> np.ndarray:
    numerator = np.sum(
        np.conj(neighbor_base.astype(np.complex128))
        * target_base.astype(np.complex128),
        axis=(1, 2),
    )
    denominator = np.sum(
        np.abs(neighbor_base.astype(np.complex128)) ** 2, axis=(1, 2)
    )
    alpha = np.zeros_like(numerator)
    valid = denominator > 1.0e-20
    alpha[valid] = numerator[valid] / denominator[valid]
    magnitude = np.abs(alpha)
    clipped = np.minimum(magnitude, float(maximum_magnitude))
    alpha = np.where(magnitude > 0.0, alpha * clipped / np.maximum(magnitude, 1.0e-30), 0.0)
    return alpha.astype(np.complex64)


def retained_coefficients(r40, handle, row: int, name: str) -> np.ndarray:
    normalized = r40.channels_to_complex(
        np.asarray(handle[name][row], dtype=np.float32)
    )
    scale = np.asarray(handle["frequency_scale"][row], dtype=np.float32)
    return (normalized * scale[:, None, None]).astype(np.complex64)


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidate = np.asarray([row["candidate_rel_l2"] for row in rows], dtype=np.float64)
    parent = np.asarray([row["parent_rel_l2"] for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "candidate_mean": float(candidate.mean()),
        "candidate_max": float(candidate.max()),
        "candidate_median": float(np.median(candidate)),
        "parent_mean": float(parent.mean()),
        "parent_max": float(parent.max()),
        "passed": bool(candidate.mean() <= 0.05 and candidate.max() <= 0.05),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-spacing-m", type=float, default=50.0)
    parser.add_argument("--source-distance-scale-m", type=float, default=100.0)
    parser.add_argument("--f0-scale-hz", type=float, default=2.0)
    parser.add_argument("--t0-scale-s", type=float, default=0.01)
    parser.add_argument("--maximum-alpha-magnitude", type=float, default=2.0)
    parser.add_argument("--group-diverse", action="store_true")
    parser.add_argument("--include-low-gates", action="store_true")
    parser.add_argument("--include-source-shifted", action="store_true")
    args = parser.parse_args()

    r40 = load_r40(args.r40_script.resolve())
    fit = r40.FrequencyCacheCollection(args.fit_cache, expected_subset="fit")
    holdout = r40.FrequencyCacheCollection(
        args.holdout_cache, expected_subset="holdout"
    )
    try:
        if fit.selection_sha256 != holdout.selection_sha256:
            raise RuntimeError("fit/holdout selection mismatch")
        if set(fit.group_ids) & set(holdout.group_ids):
            raise RuntimeError("fit/holdout group leakage")
        manifest = json.loads(args.curriculum_manifest.read_text(encoding="utf-8"))
        if manifest.get("schema") != "r42_train_only_hard_record_curriculum_v1":
            raise RuntimeError("unexpected fit-error manifest")
        if str(manifest.get("selection_sha256")) != str(fit.selection_sha256):
            raise RuntimeError("manifest/cache selection mismatch")
        manifest_records = manifest["records"]
        if len(manifest_records) != len(fit.records):
            raise RuntimeError("manifest/cache record-count mismatch")

        fit_meta = []
        for position, (file_index, local_row) in enumerate(fit.records):
            handle = fit.handles[file_index]
            record = manifest_records[position]
            sample_id = str(handle["sample_id"].asstr()[local_row])
            group_id = str(handle["group_id"].asstr()[local_row])
            family = str(handle["family"].asstr()[local_row])
            if sample_id != record["sample_id"]:
                raise RuntimeError("manifest/cache sample mismatch")
            fit_meta.append(
                {
                    "position": position,
                    "file_index": file_index,
                    "local_row": local_row,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "family": family,
                    "coordinates": parse_coordinates(group_id),
                    "f0": float(handle["source_f0_hz"][local_row]),
                    "t0": float(handle["source_t0_s"][local_row]),
                    "base_error": float(record["base_record_rel_l2"]),
                }
            )
        family_fit = defaultdict(list)
        for record in fit_meta:
            if record["coordinates"] is not None:
                family_fit[record["family"]].append(record)

        wavelet_weights = (0.0, 0.25, 1.0, 4.0)
        neighbor_counts = (1, 2, 4)
        gate_thresholds = (
            (0.0, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05)
            if args.include_low_gates
            else (0.03, 0.035, 0.04, 0.045, 0.05)
        )
        correction_scales = (0.25, 0.5, 0.75, 1.0)
        variants = (
            ("unshifted_scalar", "source_shifted_scalar")
            if args.include_source_shifted
            else ("unshifted_scalar",)
        )
        configuration_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        configuration_meta: dict[str, dict[str, Any]] = {}
        identity_rows = []
        indices = holdout.frequency_indices
        fft_weight = r40.rfft_weights(r40.TIME_COUNT)[indices]
        retained = int(holdout.retained)

        for file_index, local_row in holdout.records:
            handle = holdout.handles[file_index]
            sample_id = str(handle["sample_id"].asstr()[local_row])
            group_id = str(handle["group_id"].asstr()[local_row])
            family = str(handle["family"].asstr()[local_row])
            coordinates = parse_coordinates(group_id)
            f0 = float(handle["source_f0_hz"][local_row])
            t0 = float(handle["source_t0_s"][local_row])
            target_base_coeff = retained_coefficients(
                r40, handle, local_row, "base_dct_norm"
            )
            base_selected = r40.channels_to_complex(
                np.asarray(handle["base_spectrum_selected"][local_row], dtype=np.float32)
            )
            truth_selected = r40.channels_to_complex(
                np.asarray(handle["truth_spectrum_selected"][local_row], dtype=np.float32)
            )
            grid = int(base_selected.shape[-1])
            unselected = float(handle["base_error_square_unselected"][local_row])
            target_square = float(handle["target_square_total"][local_row])
            parent_selected_square = float(
                np.sum(
                    fft_weight
                    * np.sum(
                        np.abs(base_selected - truth_selected) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                )
            )
            parent_rel = math.sqrt(
                (unselected + parent_selected_square) / max(target_square, 1.0e-30)
            )
            identity_row = {
                "sample_id": sample_id,
                "group_id": group_id,
                "family": family,
                "candidate_rel_l2": parent_rel,
                "parent_rel_l2": parent_rel,
                "predicted_risk": None,
                "neighbors": [],
            }
            identity_rows.append(identity_row)
            aligned_neighbor_cache: dict[tuple[str, int], np.ndarray] = {}

            pools = family_fit.get(family, []) if coordinates is not None else []
            for wavelet_weight in wavelet_weights:
                ranked = []
                if pools:
                    target_x, target_z = coordinates
                    for neighbor in pools:
                        neighbor_x, neighbor_z = neighbor["coordinates"]
                        source_term = (
                            (target_x - neighbor_x) ** 2
                            + (target_z - neighbor_z) ** 2
                        ) / float(args.source_distance_scale_m) ** 2
                        wavelet_term = (
                            ((f0 - neighbor["f0"]) / float(args.f0_scale_hz)) ** 2
                            + ((t0 - neighbor["t0"]) / float(args.t0_scale_s)) ** 2
                        )
                        distance = source_term + float(wavelet_weight) * wavelet_term
                        ranked.append((distance, neighbor))
                    ranked.sort(key=lambda item: (item[0], item[1]["sample_id"]))

                for neighbor_count in neighbor_counts:
                    if args.group_diverse:
                        selected_neighbors = []
                        selected_groups = set()
                        for item in ranked:
                            neighbor_group = item[1]["group_id"]
                            if neighbor_group in selected_groups:
                                continue
                            selected_neighbors.append(item)
                            selected_groups.add(neighbor_group)
                            if len(selected_neighbors) == int(neighbor_count):
                                break
                    else:
                        selected_neighbors = ranked[: int(neighbor_count)]
                    if selected_neighbors:
                        distance_weights = np.asarray(
                            [1.0 / max(item[0], 1.0e-4) for item in selected_neighbors],
                            dtype=np.float64,
                        )
                        distance_weights /= distance_weights.sum()
                        predicted_risk = float(
                            sum(
                                weight * item[1]["base_error"]
                                for weight, item in zip(distance_weights, selected_neighbors)
                            )
                        )
                    else:
                        distance_weights = np.asarray([], dtype=np.float64)
                        predicted_risk = 0.0

                    for variant in variants:
                        correction_coeff = np.zeros(
                            (len(indices), retained, retained), dtype=np.complex64
                        )
                        neighbor_ids = []
                        for weight, (distance, neighbor) in zip(
                            distance_weights, selected_neighbors
                        ):
                            cache_key = (variant, int(neighbor["position"]))
                            aligned_neighbor = aligned_neighbor_cache.get(cache_key)
                            if aligned_neighbor is None:
                                neighbor_handle = fit.handles[neighbor["file_index"]]
                                neighbor_row = neighbor["local_row"]
                                neighbor_base = retained_coefficients(
                                    r40, neighbor_handle, neighbor_row, "base_dct_norm"
                                )
                                neighbor_residual = retained_coefficients(
                                    r40,
                                    neighbor_handle,
                                    neighbor_row,
                                    "residual_dct_norm",
                                )
                                if variant == "source_shifted_scalar":
                                    target_x, target_z = coordinates
                                    neighbor_x, neighbor_z = neighbor["coordinates"]
                                    shift_x = int(
                                        round(
                                            (target_x - neighbor_x)
                                            / float(args.grid_spacing_m)
                                        )
                                    )
                                    shift_z = int(
                                        round(
                                            (target_z - neighbor_z)
                                            / float(args.grid_spacing_m)
                                        )
                                    )
                                    neighbor_base = shift_coefficients(
                                        neighbor_base,
                                        grid=grid,
                                        retained=retained,
                                        shift_z=shift_z,
                                        shift_x=shift_x,
                                    )
                                    neighbor_residual = shift_coefficients(
                                        neighbor_residual,
                                        grid=grid,
                                        retained=retained,
                                        shift_z=shift_z,
                                        shift_x=shift_x,
                                    )
                                alpha = align_scalar(
                                    neighbor_base,
                                    target_base_coeff,
                                    maximum_magnitude=float(
                                        args.maximum_alpha_magnitude
                                    ),
                                )
                                aligned_neighbor = (
                                    alpha[:, None, None] * neighbor_residual
                                ).astype(np.complex64)
                                aligned_neighbor_cache[cache_key] = aligned_neighbor
                            correction_coeff += float(weight) * aligned_neighbor
                            neighbor_ids.append(
                                {
                                    "sample_id": neighbor["sample_id"],
                                    "group_id": neighbor["group_id"],
                                    "distance": float(distance),
                                    "weight": float(weight),
                                    "base_record_rel_l2": neighbor["base_error"],
                                }
                            )
                        padded = np.zeros(
                            (len(indices), grid, grid), dtype=np.complex64
                        )
                        padded[:, :retained, :retained] = correction_coeff
                        correction_spatial = idctn(
                            padded.real, type=2, norm="ortho", axes=(-2, -1)
                        ) + 1j * idctn(
                            padded.imag, type=2, norm="ortho", axes=(-2, -1)
                        )
                        for gate_threshold in gate_thresholds:
                            gate_open = bool(predicted_risk >= gate_threshold)
                            for correction_scale in correction_scales:
                                applied_scale = (
                                    float(correction_scale) if gate_open else 0.0
                                )
                                selected_square = float(
                                    np.sum(
                                        fft_weight
                                        * np.sum(
                                            np.abs(
                                                base_selected
                                                + applied_scale * correction_spatial
                                                - truth_selected
                                            )
                                            ** 2,
                                            axis=(1, 2),
                                            dtype=np.float64,
                                        )
                                    )
                                )
                                candidate_rel = math.sqrt(
                                    (unselected + selected_square)
                                    / max(target_square, 1.0e-30)
                                )
                                key_payload = {
                                    "variant": variant,
                                    "wavelet_weight": float(wavelet_weight),
                                    "neighbor_count": int(neighbor_count),
                                    "gate_threshold": float(gate_threshold),
                                    "correction_scale": float(correction_scale),
                                }
                                key = json.dumps(key_payload, sort_keys=True)
                                configuration_meta[key] = key_payload
                                configuration_rows[key].append(
                                    {
                                        "sample_id": sample_id,
                                        "group_id": group_id,
                                        "family": family,
                                        "candidate_rel_l2": candidate_rel,
                                        "parent_rel_l2": parent_rel,
                                        "predicted_risk": predicted_risk,
                                        "gate_open": gate_open,
                                        "neighbors": neighbor_ids,
                                    }
                                )

        identity_summary = summarize(identity_rows)
        candidates = []
        for key, rows in configuration_rows.items():
            if len(rows) != len(identity_rows):
                raise RuntimeError("configuration row coverage mismatch")
            summary = summarize(rows)
            candidates.append(
                {
                    "configuration": configuration_meta[key],
                    "summary": summary,
                    "score": float(summary["candidate_max"])
                    + 0.1 * float(summary["candidate_mean"]),
                    "rows": rows,
                }
            )
        identity_candidate = {
            "configuration": {"variant": "identity"},
            "summary": identity_summary,
            "score": float(identity_summary["candidate_max"])
            + 0.1 * float(identity_summary["candidate_mean"]),
            "rows": identity_rows,
        }
        candidates.append(identity_candidate)
        candidates.sort(
            key=lambda item: (
                item["score"],
                item["summary"]["candidate_max"],
                json.dumps(item["configuration"], sort_keys=True),
            )
        )
        best = candidates[0]
        payload = {
            "schema": "r43_local_train_residual_retrieval_opened_development_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "role": "opened_group_disjoint_development_model_selection_only",
            "selection_sha256": fit.selection_sha256,
            "fit_record_count": len(fit.records),
            "holdout_record_count": len(holdout.records),
            "fit_holdout_group_overlap": 0,
            "grid": {
                "wavelet_weights": list(wavelet_weights),
                "neighbor_counts": list(neighbor_counts),
                "gate_thresholds": list(gate_thresholds),
                "correction_scales": list(correction_scales),
                "variants": list(variants),
                "group_diverse": bool(args.group_diverse),
                "include_low_gates": bool(args.include_low_gates),
            },
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "identity": identity_candidate,
            "best": best,
            "top_candidates": candidates[:20],
            "absolute_goal": {
                "mean_lte_0p05": bool(best["summary"]["candidate_mean"] <= 0.05),
                "max_lte_0p05": bool(best["summary"]["candidate_max"] <= 0.05),
                "passed": bool(best["summary"]["passed"]),
            },
            "data_boundary": {
                "fit_truth_used": True,
                "opened_development_holdout_used": True,
                "r29b_opened": False,
                "final_validation_opened": False,
                "test_id_opened": False,
                "paper_modified": False,
            },
        }
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "event": "r43_complete",
                    "best_configuration": best["configuration"],
                    "best_summary": best["summary"],
                    "absolute_goal_passed": payload["absolute_goal"]["passed"],
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        fit.close()
        holdout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
