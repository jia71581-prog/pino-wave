#!/usr/bin/env python3
"""Prepare the published high-resolution Marmousi-1 Vp file for z-by-x cropping."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SOURCE_DOI = "10.5281/zenodo.16114161"
SOURCE_URL = "https://zenodo.org/records/16114161"
SOURCE_MD5 = "3603290793a5d870fb1290301bc68dbd"
SOURCE_SHAPE_XZ = (2301, 751)
SOURCE_SPACING_M = 4.0


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_input = args.raw_input.expanduser().resolve()
    output_npy = args.output_npy.expanduser().resolve()
    provenance_json = args.provenance_json.expanduser().resolve()
    if output_npy.exists() or provenance_json.exists():
        raise FileExistsError("refusing to overwrite prepared Marmousi-1 assets")
    if _digest(raw_input, "md5") != SOURCE_MD5:
        raise ValueError("downloaded Marmousi-1 file does not match the published MD5")

    source_xz = np.fromfile(raw_input, dtype=np.float32)
    expected_size = int(np.prod(SOURCE_SHAPE_XZ))
    if source_xz.size != expected_size:
        raise ValueError(f"expected {expected_size} float32 values, got {source_xz.size}")
    source_xz = source_xz.reshape(SOURCE_SHAPE_XZ)
    velocity_zx = np.ascontiguousarray(source_xz.T)
    if not np.isfinite(velocity_zx).all() or float(velocity_zx.min()) <= 0.0:
        raise ValueError("Marmousi-1 velocity must be positive and finite")

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    partial = output_npy.with_name(f".{output_npy.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            np.save(handle, velocity_zx, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output_npy)
    finally:
        partial.unlink(missing_ok=True)

    provenance = {
        "status": "complete",
        "model": "Marmousi-1 P-wave velocity",
        "source_doi": SOURCE_DOI,
        "source_url": SOURCE_URL,
        "license": "CC-BY-4.0",
        "raw_input": str(raw_input),
        "raw_input_md5": SOURCE_MD5,
        "raw_input_sha256": _digest(raw_input, "sha256"),
        "raw_layout": "float32 C-order [horizontal_x, vertical_z]",
        "raw_shape_xz": list(SOURCE_SHAPE_XZ),
        "prepared_output": str(output_npy),
        "prepared_output_sha256": _digest(output_npy, "sha256"),
        "prepared_layout": "float32 NumPy [vertical_z, horizontal_x]",
        "prepared_shape_zx": [int(value) for value in velocity_zx.shape],
        "dx_m": SOURCE_SPACING_M,
        "dz_m": SOURCE_SPACING_M,
        "physical_extent_m": [
            float((velocity_zx.shape[1] - 1) * SOURCE_SPACING_M),
            float((velocity_zx.shape[0] - 1) * SOURCE_SPACING_M),
        ],
        "velocity_range_mps": [float(velocity_zx.min()), float(velocity_zx.max())],
        "deepest_velocity_le_1500_5_m": float(
            np.max(np.where(velocity_zx <= 1500.5)[0]) * SOURCE_SPACING_M
        ),
        "transform": "transpose_only_no_geological_interpolation",
    }
    provenance_json.parent.mkdir(parents=True, exist_ok=True)
    provenance_json.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
