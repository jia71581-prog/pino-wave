#!/usr/bin/env python3
"""Read-only stability preflight for the v23b coarse LWC-84 audit."""

from __future__ import annotations

import json
import math

import h5py
import numpy as np


SOURCE_H5 = (
    "/data/jiayh/data/"
    "acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
)
FAMILIES = {"uniform", "layered", "marmousi"}


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def main() -> None:
    with h5py.File(SOURCE_H5, "r", swmr=True) as handle:
        selected = [
            index
            for index, (split, family) in enumerate(
                zip(handle["split"], handle["medium_type"], strict=True)
            )
            if _decode(split) == "validation" and _decode(family) in FAMILIES
        ]
        vmax_mps = max(
            float(np.max(handle["velocity_mps"][index])) for index in selected
        )
    axis_spectral_radius = 2048.0 / 315.0
    inverse_spacing_square_sum = 2.0 / (10.0**2)

    def qmax(dt_s: float) -> float:
        return (
            dt_s**2
            * vmax_mps**2
            * axis_spectral_radius
            * inverse_spacing_square_sum
        )

    payload = {
        "records": len(selected),
        "vmax_mps": vmax_mps,
        "qmax_dt_0p5ms": qmax(5.0e-4),
        "qmax_dt_0p25ms": qmax(2.5e-4),
        "qmax_dt_0p125ms": qmax(1.25e-4),
        "dt_stability_limit_s": math.sqrt(
            1.0
            / (
                vmax_mps**2
                * axis_spectral_radius
                * inverse_spacing_square_sum
            )
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
