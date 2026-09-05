#!/usr/bin/env python3
"""Build one E1d eikonal/energy shard aligned to one E1 background cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.eikonal import (  # noqa: E402
    debiased_grid_eikonal_travel_time,
)
from saved_time_phase_operator_v4.exterior_cpml import (  # noqa: E402
    ExteriorCPMLContract,
    extend_velocity_to_exterior,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError(args.output)
    contract = ExteriorCPMLContract()
    started = time.time()
    partial = args.output.with_name(f"{args.output.name}.partial.{os.getpid()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(args.cache, "r", swmr=True) as source, h5py.File(partial, "x") as out:
            if source.attrs.get("schema") != "transfer_dg_wfp_e1_cache_v1" or source.attrs.get("status") != "complete":
                raise RuntimeError("invalid E1 cache")
            count = len(source["sample_id"])
            active = np.logical_or(
                np.asarray(source["profiles/active_x"], dtype=bool),
                np.asarray(source["profiles/active_z"], dtype=bool),
            )
            out.attrs["schema"] = "transfer_dg_wfp_e1d_travel_v1"
            out.attrs["status"] = "building"
            out.attrs["source_cache"] = str(args.cache.resolve())
            out.attrs["source_cache_sha256"] = sha256(args.cache)
            out.attrs["travel_method"] = "radius4_grid_eikonal_constant_graph_debiased_fractional_source"
            out.attrs["cpml_layers"] = 20
            out.attrs["validation_opened"] = False
            out.attrs["test_id_opened"] = False
            text = h5py.string_dtype("utf-8")
            out.create_dataset("sample_id", shape=(count,), dtype=text)
            out.create_dataset("source_index", shape=(count,), dtype=np.int64)
            physical_ds = out.create_dataset(
                "travel_physical_s", shape=(count, 201, 201), dtype=np.float32,
                chunks=(1, 201, 201), compression="lzf",
            )
            exterior_ds = out.create_dataset(
                "travel_exterior_s", shape=(count, 221, 241), dtype=np.float32,
                chunks=(1, 221, 241), compression="lzf",
            )
            physical_energy = out.create_dataset("physical_total_square", shape=(count,), dtype=np.float64)
            auxiliary_energy = out.create_dataset("auxiliary_total_square", shape=(count,), dtype=np.float64)
            graph_uniform_error = []
            for local in range(count):
                sample_id = str(source["sample_id"].asstr()[local])
                velocity = np.asarray(source["velocity_bg_saved_mps"][local], dtype=np.float32)
                sx, sz = (float(value) for value in source["source_parameters"][local, :2])
                source_z = int(np.clip(round(sz / 10.0), 0, 200))
                source_x = int(np.clip(round(sx / 10.0), 0, 200))
                local_speed = float(velocity[source_z, source_x])
                physical = debiased_grid_eikonal_travel_time(
                    velocity,
                    source_coordinates_m=[(sz, sx)],
                    dx_m=10.0,
                    dz_m=10.0,
                )[0]
                exterior_velocity = np.asarray(
                    extend_velocity_to_exterior(velocity, contract), dtype=np.float32
                )
                exterior = debiased_grid_eikonal_travel_time(
                    exterior_velocity,
                    source_coordinates_m=[(sz, sx)],
                    dx_m=10.0,
                    dz_m=10.0,
                    x0_m=-200.0,
                )[0]
                if str(source["family"].asstr()[local]) == "uniform":
                    x = np.arange(201, dtype=np.float32) * 10.0
                    z = np.arange(201, dtype=np.float32) * 10.0
                    xx, zz = np.meshgrid(x, z)
                    exact = np.sqrt((xx - sx) ** 2 + (zz - sz) ** 2) / local_speed
                    graph_uniform_error.append(float(np.max(np.abs(physical - exact))))
                physical_coeff = np.asarray(source["physical_coeff_norm"][local], dtype=np.float32)
                auxiliary_coeff = np.asarray(source["auxiliary_coeff_norm"][local], dtype=np.float32)
                out["sample_id"][local] = sample_id
                out["source_index"][local] = int(source["source_index"][local])
                physical_ds[local] = physical
                exterior_ds[local] = exterior
                physical_energy[local] = np.square(physical_coeff.astype(np.float64)).sum()
                auxiliary_energy[local] = np.square(
                    auxiliary_coeff[..., active].astype(np.float64)
                ).sum()
                print(json.dumps({
                    "event": "travel_record", "record": local + 1, "of": count,
                    "sample_id": sample_id, "elapsed_s": round(time.time() - started, 2),
                }), flush=True)
            out.attrs["uniform_graph_max_abs_error_s"] = max(graph_uniform_error, default=0.0)
            out.attrs["status"] = "complete"
            out.flush()
        os.replace(partial, args.output)
    finally:
        partial.unlink(missing_ok=True)
    summary = {
        "schema": "transfer_dg_wfp_e1d_travel_summary_v1",
        "status": "complete",
        "source_cache": str(args.cache.resolve()),
        "source_cache_sha256": sha256(args.cache),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "record_count": count,
        "elapsed_s": time.time() - started,
        "validation_opened": False,
        "test_id_opened": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
