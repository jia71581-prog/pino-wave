#!/usr/bin/env python3
"""Build the V3-format normalization JSON for the r49 direct48 P_bg pilot.

Reads ONLY velocity rows (an allowed model input) to compute the deterministic
velocity center/scale with the project's ``filtered_train_robust_percentile_v3``
rule.  The pressure scale is pinned by the scientific contract to the global
value ``1.894229157173348e-08`` (the pilot never derives a per-record or
per-frequency target normalization).  The JSON is bound to the train manifest
digest and written atomically; an existing file is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grouped_ufno_mionet_v3.data.index import build_manifest  # noqa: E402

DEFAULT_PRESSURE_SCALE_PA = 1.894229157173348e-08
SOURCE_SCALES = (2000.0, 2000.0, 50.0, 1.2, 1.0)
ALGORITHM = "direct48_pbg_pilot_v1"


def build_normalization_payload(
    source_h5: Path,
    *,
    pressure_scale_pa: float,
) -> dict[str, object]:
    manifest = build_manifest(source_h5)
    train_count = int(manifest.counts_after.get("train", 0))
    if train_count <= 0:
        raise ValueError("train census is empty")
    with h5py.File(str(source_h5), "r", swmr=True) as h5:
        velocity = np.asarray(h5["velocity_mps"][:], dtype=np.float64).reshape(-1)
    if velocity.size == 0 or not np.isfinite(velocity).all():
        raise ValueError("velocity rows are empty or non-finite")
    center = float(np.median(velocity))
    q25, q75 = np.percentile(velocity, [25.0, 75.0])
    scale = float(max(q75 - q25, float(np.std(velocity)), 1.0))
    if not np.isfinite(pressure_scale_pa) or float(pressure_scale_pa) <= 0.0:
        raise ValueError("pressure scale must be finite and positive")
    return {
        "velocity_center_mps": center,
        "velocity_scale_mps": scale,
        "pressure_scale_pa": float(pressure_scale_pa),
        "source_scales": list(SOURCE_SCALES),
        "train_manifest_sha256": manifest.digest,
        "allowed_medium_types": ["uniform", "layered", "marmousi"],
        "record_count": train_count,
        "algorithm": ALGORITHM,
    }


def write_atomic(payload: dict[str, object], destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-h5",
        default=str(
            ROOT / "artifacts/pbg_factorized_fno_marm700_eikonal_v3/dataset_v1.h5"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "artifacts/pbg_factorized_fno_marm700_eikonal_v3"
            / "normalization_v3_direct48_pbg.json"
        ),
    )
    parser.add_argument("--pressure-scale-pa", type=float, default=DEFAULT_PRESSURE_SCALE_PA)
    args = parser.parse_args(argv)
    payload = build_normalization_payload(
        Path(args.source_h5).resolve(), pressure_scale_pa=args.pressure_scale_pa
    )
    output = write_atomic(payload, Path(args.output).resolve())
    print(json.dumps({"output": str(output), "algorithm": ALGORITHM, **payload}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
