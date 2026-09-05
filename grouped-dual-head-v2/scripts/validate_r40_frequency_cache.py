#!/usr/bin/env python3
"""Validate R40 holdout cache reconstruction and its representation oracle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.fft import idctn


def as_complex(value: np.ndarray) -> np.ndarray:
    if value.ndim != 4 or value.shape[1] != 2:
        raise ValueError(f"expected [frequency, 2, height, width], got {value.shape}")
    return value[:, 0].astype(np.float32) + 1j * value[:, 1].astype(np.float32)


def rfft_weights(time_count: int) -> np.ndarray:
    result = np.full(time_count // 2 + 1, 2.0, dtype=np.float64)
    result[0] = 1.0
    if time_count % 2 == 0:
        result[-1] = 1.0
    return result


def validate(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema", "")) not in {
            "r40_frequency_residual_cache_v1",
            "r40_frequency_residual_cache_v2",
        }:
            raise RuntimeError("unexpected cache schema")
        if str(handle.attrs.get("status", "")) != "complete":
            raise RuntimeError("cache is not complete")
        if "base_spectrum_selected" not in handle:
            result = {
                "path": str(path.resolve()),
                "schema": str(handle.attrs["schema"]),
                "subset": str(handle.attrs["subset"]),
                "record_count": len(handle["sample_id"]),
                "static_dct_scale_present": "static_dct_scale" in handle,
            }
            if "static_dct_scale" in handle:
                scale = np.asarray(handle["static_dct_scale"][:], dtype=np.float32)
                result.update(
                    {
                        "static_dct_scale_shape": list(scale.shape),
                        "static_dct_scale_min": float(scale.min()),
                        "static_dct_scale_max": float(scale.max()),
                        "static_dct_scale_finite": bool(np.isfinite(scale).all()),
                    }
                )
            return result

        indices = np.asarray(handle["frequency_indices"], dtype=np.int64)
        time_count = (int(indices.max()) + 1) * 2
        # R40 currently uses the frozen 401-frame source dataset.
        if time_count != 400:
            time_count = 401
        weights = rfft_weights(time_count)[indices]
        retained = int(handle["base_dct_norm"].shape[-1])
        grid = int(handle["base_spectrum_selected"].shape[-1])
        rows = []
        for row in range(len(handle["sample_id"])):
            base = as_complex(np.asarray(handle["base_spectrum_selected"][row]))
            truth = as_complex(np.asarray(handle["truth_spectrum_selected"][row]))
            residual_norm = as_complex(np.asarray(handle["residual_dct_norm"][row]))
            scale = np.asarray(handle["frequency_scale"][row], dtype=np.float32)
            residual_coeff = residual_norm * scale[:, None, None]
            padded = np.zeros((len(indices), grid, grid), dtype=np.complex64)
            padded[:, :retained, :retained] = residual_coeff
            correction = idctn(
                padded.real, type=2, norm="ortho", axes=(-2, -1)
            ) + 1j * idctn(
                padded.imag, type=2, norm="ortho", axes=(-2, -1)
            )

            selected_base_square = float(
                np.sum(
                    weights
                    * np.sum(np.abs(base - truth) ** 2, axis=(1, 2), dtype=np.float64)
                )
            )
            selected_oracle_square = float(
                np.sum(
                    weights
                    * np.sum(
                        np.abs(base + correction - truth) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                )
            )
            unselected = float(handle["base_error_square_unselected"][row])
            target = float(handle["target_square_total"][row])
            rows.append(
                {
                    "sample_id": handle["sample_id"].asstr()[row],
                    "base_rel_l2": math.sqrt((unselected + selected_base_square) / target),
                    "representation_oracle_rel_l2": math.sqrt(
                        (unselected + selected_oracle_square) / target
                    ),
                    "unselected_error_fraction": unselected
                    / max(unselected + selected_base_square, 1.0e-30),
                    "retained": retained,
                    "frequency_count": len(indices),
                    "base_dct_norm_max_abs": float(
                        np.max(np.abs(np.asarray(handle["base_dct_norm"][row], dtype=np.float32)))
                    ),
                    "residual_dct_norm_max_abs": float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    handle["residual_dct_norm"][row], dtype=np.float32
                                )
                            )
                        )
                    ),
                    "residual_dct_norm_finite": bool(
                        np.isfinite(handle["residual_dct_norm"][row]).all()
                    ),
                }
            )
    return {"path": str(path.resolve()), "records": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.cache), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
