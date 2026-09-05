#!/usr/bin/env python3
"""Build an audited no-water Marmousi sample for current-checkpoint visualization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

sys.path.insert(0, "src")
from fno_acoustic.data_generation.config import (  # noqa: E402
    boundaries_from_config,
    grid_from_config,
    load_config,
    resolve_config,
)
from fno_acoustic.data_generation.model_marmousi import _sha256_file  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402
from fno_acoustic.data_generation.velocity_models_lwc84 import (  # noqa: E402
    load_marmousi_crop,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saved_time_phase_operator_v4.hybrid_travel import straight_ray_grid_numpy  # noqa: E402


SCHEMA = "current_helmholtz_no_water_marmousi_visualization_v2"


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _upsample_saved_velocity(velocity_mps: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = torch.as_tensor(velocity_mps[None, None], dtype=torch.float32)
    return (
        torch.nn.functional.interpolate(
            value,
            size=shape,
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        .numpy()
        .astype(np.float64)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--velocity-file", type=Path)
    parser.add_argument("--velocity-sha256")
    parser.add_argument("--source-spacing-m", type=float)
    parser.add_argument("--source-provenance-json", type=Path)
    parser.add_argument("--source-model-name", default="Marmousi-derived")
    parser.add_argument("--sample-id-prefix", default="derived_marmousi_nowater")
    parser.add_argument("--group-id-prefix", default="derived:marmousi")
    parser.add_argument(
        "--selection-basis",
        default=(
            "no water; strong lateral contrast; dipping thin layers; faulted "
            "interfaces; high-velocity lens"
        ),
    )
    parser.add_argument("--crop-x0-m", type=float, default=5100.0)
    parser.add_argument("--crop-z0-m", type=float, default=1800.0)
    parser.add_argument("--source-x-m", type=float, default=1000.0)
    parser.add_argument("--source-z-m", type=float, default=100.0)
    parser.add_argument("--source-f0-hz", type=float, default=15.0)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--background-sigma-cells", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite visualization sample: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = resolve_config(load_config(str(args.frozen_config.expanduser().resolve())))
    using_override = args.velocity_file is not None
    velocity_file = (
        args.velocity_file.expanduser().resolve()
        if using_override
        else Path(str(cfg["marmousi"]["velocity_file"])).expanduser().resolve()
    )
    source_hash = _sha256_file(velocity_file)
    expected_source_hash = (
        str(args.velocity_sha256)
        if using_override
        else str(cfg["marmousi"]["sha256"])
    )
    if using_override and not args.velocity_sha256:
        raise ValueError("--velocity-sha256 is required with --velocity-file")
    if source_hash != expected_source_hash:
        raise ValueError("Marmousi source hash does not match its declared identity")
    source_spacing_m = (
        float(args.source_spacing_m)
        if using_override and args.source_spacing_m is not None
        else float(cfg["marmousi"]["source_dx_m"])
    )
    if using_override and args.source_spacing_m is None:
        raise ValueError("--source-spacing-m is required with --velocity-file")
    source_provenance_json = (
        args.source_provenance_json.expanduser().resolve()
        if args.source_provenance_json is not None
        else None
    )
    source_provenance_sha256 = (
        _sha256_file(source_provenance_json) if source_provenance_json is not None else ""
    )
    grid = grid_from_config(cfg)
    boundaries = boundaries_from_config(cfg)
    with h5py.File(args.source_h5.expanduser().resolve(), "r", swmr=True) as source:
        time_s = np.asarray(source["time_s"][:], dtype=np.float64)
        x_m = np.asarray(source["x_m"][:], dtype=np.float64)
        z_m = np.asarray(source["z_m"][:], dtype=np.float64)
        source_manifest_sha256 = str(source.attrs.get("manifest_sha256", ""))

    velocity_fine, crop_metadata = load_marmousi_crop(
        velocity_file,
        grid=grid,
        source_dx_m=source_spacing_m,
        source_dz_m=source_spacing_m,
        source_unit="m/s" if using_override else str(cfg["marmousi"]["velocity_unit"]),
        crop_x0_m=float(args.crop_x0_m),
        crop_z0_m=float(args.crop_z0_m),
        interpolation="linear" if using_override else str(cfg["marmousi"]["interpolation"]),
    )
    if np.any(velocity_fine <= 1500.5):
        raise ValueError("selected Marmousi crop still contains 1500 m/s water")
    if float(np.mean(velocity_fine[-41:])) <= float(np.mean(velocity_fine[:41])):
        raise ValueError("selected Marmousi crop lacks the expected positive depth trend")
    f0_hz = float(args.source_f0_hz)
    t0_s = 1.5 / f0_hz
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=boundaries,
        dt_s=float(cfg["time"]["dt_used_s"]),
        output_times_s=time_s,
        c_ref_mps=6750.0,
        device=args.device if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        kappa_max=float(cfg["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(cfg["boundaries"]["minimum_frequency_hz"]),
    )
    target_result = solver.simulate(
        velocity_fine[None],
        source_x_m=float(args.source_x_m),
        source_z_m=float(args.source_z_m),
        source_f0_hz=f0_hz,
        source_t0_s=t0_s,
        source_amplitude=float(args.source_amplitude),
    )
    velocity_saved = np.asarray(target_result.velocity_saved_mps[0], dtype=np.float32)
    target = np.asarray(target_result.wavefield[0], dtype=np.float32)
    source_map = np.asarray(target_result.source_map_saved[0], dtype=np.float32)
    sigma = float(args.background_sigma_cells)
    smoothed_saved = gaussian_filter(velocity_saved.astype(np.float64), sigma=sigma, mode="nearest")
    background_velocity_fine = _upsample_saved_velocity(
        smoothed_saved.astype(np.float32), (grid.nz, grid.nx)
    )
    background_result = solver.simulate(
        background_velocity_fine[None],
        source_x_m=float(args.source_x_m),
        source_z_m=float(args.source_z_m),
        source_f0_hz=f0_hz,
        source_t0_s=t0_s,
        source_amplitude=float(args.source_amplitude),
    )
    background = np.asarray(background_result.wavefield[0], dtype=np.float32)
    travel_time = straight_ray_grid_numpy(
        velocity_saved,
        source_x_m=np.asarray([args.source_x_m], dtype=np.float32),
        source_z_m=np.asarray([args.source_z_m], dtype=np.float32),
        x_m=x_m.astype(np.float32),
        z_m=z_m.astype(np.float32),
        samples=12,
    )[0]
    expected_shape = (len(time_s), len(z_m), len(x_m))
    if target.shape != expected_shape or background.shape != expected_shape:
        raise RuntimeError("custom Marmousi wavefield has an unexpected shape")
    if velocity_saved.shape != (len(z_m), len(x_m)):
        raise RuntimeError("custom Marmousi saved velocity has an unexpected shape")
    if not all(np.isfinite(value).all() for value in (target, background, travel_time)):
        raise RuntimeError("custom Marmousi artifact contains non-finite values")

    sample_id = (
        f"{args.sample_id_prefix}_x{int(args.crop_x0_m)}_"
        f"z{int(args.crop_z0_m)}_f{f0_hz:g}"
    )
    source_parameters = np.asarray(
        [args.source_x_m, args.source_z_m, f0_hz, t0_s, args.source_amplitude],
        dtype=np.float32,
    )
    content_sha256 = _array_sha256(
        velocity_saved,
        target,
        background,
        travel_time,
        source_map,
        source_parameters,
    )
    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    try:
        with h5py.File(partial, "x") as handle:
            handle.attrs["schema"] = SCHEMA
            handle.attrs["status"] = "building"
            handle.attrs["sample_id"] = sample_id
            handle.attrs["group_id"] = (
                f"{args.group_id_prefix}:x{args.crop_x0_m:.1f}:z{args.crop_z0_m:.1f}"
            )
            handle.attrs["source_model_name"] = str(args.source_model_name)
            handle.attrs["source_velocity_file"] = str(velocity_file)
            handle.attrs["source_velocity_sha256"] = source_hash
            handle.attrs["source_provenance_json"] = (
                str(source_provenance_json) if source_provenance_json is not None else ""
            )
            handle.attrs["source_provenance_sha256"] = source_provenance_sha256
            handle.attrs["source_manifest_sha256"] = source_manifest_sha256
            handle.attrs["frozen_config"] = str(args.frozen_config.expanduser().resolve())
            handle.attrs["frozen_config_sha256"] = str(cfg["config_sha256"])
            handle.attrs["crop_x0_m"] = float(args.crop_x0_m)
            handle.attrs["crop_z0_m"] = float(args.crop_z0_m)
            handle.attrs["background_sigma_saved_cells"] = sigma
            handle.attrs["water_fraction_le_1500_5"] = float(
                np.mean(velocity_saved <= 1500.5)
            )
            handle.attrs["selection_basis"] = str(args.selection_basis)
            handle.attrs["content_sha256"] = content_sha256
            handle.create_dataset("time_s", data=time_s)
            handle.create_dataset("x_m", data=x_m)
            handle.create_dataset("z_m", data=z_m)
            handle.create_dataset("source_parameters", data=source_parameters)
            handle.create_dataset("source_map", data=source_map, compression="gzip", compression_opts=1)
            handle.create_dataset("velocity_mps", data=velocity_saved, compression="gzip", compression_opts=1)
            handle.create_dataset("travel_time_s", data=travel_time, compression="gzip", compression_opts=1)
            handle.create_dataset(
                "wavefield_target",
                data=target,
                chunks=(1, len(z_m), len(x_m)),
                compression="gzip",
                compression_opts=1,
            )
            handle.create_dataset(
                "background_pbg",
                data=background,
                chunks=(1, len(z_m), len(x_m)),
                compression="gzip",
                compression_opts=1,
            )
            handle.attrs["status"] = "complete"
            handle.flush()
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    report = {
        "status": "complete",
        "output": str(output),
        "sample_id": sample_id,
        "crop_metadata": crop_metadata,
        "source_model_name": str(args.source_model_name),
        "source_provenance_json": (
            str(source_provenance_json) if source_provenance_json is not None else ""
        ),
        "source_provenance_sha256": source_provenance_sha256,
        "velocity_range_mps": [float(velocity_saved.min()), float(velocity_saved.max())],
        "water_fraction_le_1500_5": float(np.mean(velocity_saved <= 1500.5)),
        "content_sha256": content_sha256,
        "size_mb": output.stat().st_size / 1.0e6,
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
