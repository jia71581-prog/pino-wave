#!/usr/bin/env python3
"""Generate one GPU shard of sigma-2 Background-WFP and exterior-CPML labels."""
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
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from fno_acoustic.data_generation.config import (  # noqa: E402
    boundaries_from_config,
    grid_from_config,
    load_config,
)
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402
from saved_time_phase_operator_v4.exterior_cpml import (  # noqa: E402
    PROFILE_FIELDS,
    build_saved_exterior_cpml_profiles,
    contract_from_dataset_config,
)


PHYSICAL_SCALE = 1.0e-8
AUXILIARY_SCALES = np.asarray(
    [1.0e-8, 1.0e-8, 1.0e-11, 1.0e-11, 1.0e-11, 1.0e-11,
     1.0e-13, 1.0e-13, 1.0e-13, 1.0e-13],
    dtype=np.float32,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex_channels(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    return np.stack((array.real, array.imag), axis=2).reshape(
        array.shape[0], 2 * array.shape[1], *array.shape[2:]
    ).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--device", required=True)
    parser.add_argument("--frequency-count", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".summary.json").exists():
        raise FileExistsError(f"refusing to overwrite WFP cache shard: {args.output}")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index is outside shard count")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "transfer_dg_wfp_e1_manifest_v1":
        raise RuntimeError("unexpected E1 manifest schema")
    if manifest.get("split") != "train" or manifest.get("validation_opened") or manifest.get("test_id_opened"):
        raise RuntimeError("WFP cache generation is restricted to train records")
    rows = manifest["records"][args.shard_index :: args.shard_count]
    if args.max_records > 0:
        rows = rows[: args.max_records]
    if not rows:
        raise RuntimeError("empty WFP cache shard")
    source_path = Path(manifest["source_h5"])
    frozen_path = source_path.parent / "frozen_config.yaml"
    config = load_config(frozen_path)
    contract = contract_from_dataset_config(config)
    if contract.cpml_layers != 20:
        raise RuntimeError("WFP E1 requires exactly 20 saved CPML layers")
    frequency_count = int(args.frequency_count)
    if not 1 <= frequency_count <= 201:
        raise ValueError("frequency count must lie in [1,201]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA cache generation requested without CUDA")

    full_time = None
    with h5py.File(source_path, "r", swmr=True) as source:
        full_time = np.asarray(source["time_s"], dtype=np.float64)
    if len(full_time) != 401:
        raise RuntimeError("WFP E1 requires the complete 401-frame time axis")
    solver = LWC84CPMLSolver(
        grid=grid_from_config(config),
        boundaries=boundaries_from_config(config),
        dt_s=float(config["time"]["dt_used_s"]),
        output_times_s=full_time,
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float32,
        kappa_max=float(config["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(config["boundaries"]["minimum_frequency_hz"]),
        output_restriction_factor=2,
    )
    profiles = build_saved_exterior_cpml_profiles(
        contract, c_ref_mps=6750.0, device="cpu", dtype=torch.float32
    )
    active = (
        profiles.active_x.detach().cpu().numpy()
        | profiles.active_z.detach().cpu().numpy()
    )
    count = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.name}.partial.{os.getpid()}")
    started = time.time()
    maximum_normalized = 0.0
    try:
        with h5py.File(source_path, "r", swmr=True) as source, h5py.File(partial, "x") as out:
            out.attrs["schema"] = "transfer_dg_wfp_e1_cache_v1"
            out.attrs["status"] = "building"
            out.attrs["manifest_selection_sha256"] = manifest["selection_sha256"]
            out.attrs["source_manifest_sha256"] = manifest["source_manifest_sha256"]
            out.attrs["shard_index"] = args.shard_index
            out.attrs["shard_count"] = args.shard_count
            out.attrs["frequency_count"] = frequency_count
            out.attrs["background_sigma_saved_cells"] = float(args.sigma)
            out.attrs["physical_scale"] = PHYSICAL_SCALE
            out.attrs["validation_opened"] = False
            out.attrs["test_id_opened"] = False
            text_dtype = h5py.string_dtype("utf-8")
            for name in ("sample_id", "group_id", "family", "role"):
                out.create_dataset(name, shape=(count,), dtype=text_dtype)
            out.create_dataset("source_index", shape=(count,), dtype=np.int64)
            out.create_dataset("source_parameters", shape=(count, 4), dtype=np.float32)
            out.create_dataset("time_s", data=full_time)
            out.create_dataset("frequency_hz", data=np.fft.rfftfreq(401, d=0.0025)[:frequency_count])
            out.create_dataset("auxiliary_scales", data=AUXILIARY_SCALES)
            velocity_ds = out.create_dataset(
                "velocity_bg_saved_mps", shape=(count, 201, 201), dtype=np.float32,
                chunks=(1, 201, 201), compression="lzf",
            )
            source_ds = out.create_dataset(
                "source_map", shape=(count, 201, 201), dtype=np.float32,
                chunks=(1, 201, 201), compression="lzf",
            )
            wavelet_ds = out.create_dataset("source_wavelet", shape=(count, 401), dtype=np.float32)
            physical_ds = out.create_dataset(
                "physical_coeff_norm", shape=(count, frequency_count, 2, 201, 201),
                dtype=np.float16, chunks=(1, 1, 2, 201, 201), compression="lzf",
            )
            auxiliary_ds = out.create_dataset(
                "auxiliary_coeff_norm", shape=(count, frequency_count, 10, 221, 241),
                dtype=np.float16, chunks=(1, 1, 10, 221, 241), compression="lzf",
            )
            profile_group = out.create_group("profiles")
            for name in PROFILE_FIELDS:
                profile_group.create_dataset(name, data=getattr(profiles, name).cpu().numpy())

            for local, row in enumerate(rows):
                index = int(row["source_index"])
                family = str(row["family"])
                stored_velocity = np.asarray(source["velocity_mps"][index], dtype=np.float32)
                if family == "uniform":
                    background_saved = stored_velocity
                else:
                    background_saved = gaussian_filter(
                        stored_velocity, sigma=float(args.sigma), mode="nearest"
                    ).astype(np.float32)
                fine_velocity = F.interpolate(
                    torch.from_numpy(background_saved)[None, None],
                    size=(401, 401), mode="bilinear", align_corners=True,
                )[0, 0].numpy().astype(np.float32)
                parameters = {
                    "source_x_m": float(source["source_x_m"][index]),
                    "source_z_m": float(source["source_z_m"][index]),
                    "source_f0_hz": float(source["source_f0_hz"][index]),
                    "source_t0_s": float(source["source_t0_s"][index]),
                    "source_amplitude": float(source["source_amplitude"][index]),
                }
                result = solver.simulate(
                    fine_velocity, **parameters, capture_exterior_auxiliary=True
                )
                auxiliary = result.exterior_auxiliary
                if auxiliary is None:
                    raise RuntimeError("missing exterior auxiliary state")
                physical_fft = np.fft.rfft(result.wavefield[0], axis=0, norm="ortho")[:frequency_count]
                physical_channels = np.stack(
                    (physical_fft.real, physical_fft.imag), axis=1
                ).astype(np.float32) / PHYSICAL_SCALE

                exterior_fields = np.stack(
                    (
                        auxiliary.pressure_extended_saved[0],
                        auxiliary.psi_x_before_step[0],
                        auxiliary.psi_z_before_step[0],
                        auxiliary.phi_x_before_step[0],
                        auxiliary.phi_z_before_step[0],
                    ),
                    axis=1,
                )
                exterior_fields[..., ~active] = 0.0
                auxiliary_fft = np.fft.rfft(exterior_fields, axis=0, norm="ortho")[:frequency_count]
                auxiliary_channels = _complex_channels(auxiliary_fft)
                auxiliary_channels /= AUXILIARY_SCALES[None, :, None, None]
                current_maximum = max(
                    float(np.max(np.abs(physical_channels))),
                    float(np.max(np.abs(auxiliary_channels))),
                )
                if not np.isfinite(current_maximum) or current_maximum >= 60000.0:
                    raise FloatingPointError(f"normalized WFP target exceeds fp16: {current_maximum}")
                maximum_normalized = max(maximum_normalized, current_maximum)
                out["sample_id"][local] = row["sample_id"]
                out["group_id"][local] = row["group_id"]
                out["family"][local] = family
                out["role"][local] = row["role"]
                out["source_index"][local] = index
                out["source_parameters"][local] = (
                    parameters["source_x_m"], parameters["source_z_m"],
                    parameters["source_f0_hz"], parameters["source_t0_s"],
                )
                velocity_ds[local] = background_saved
                source_ds[local] = np.asarray(source["source_map"][index], dtype=np.float32)
                wavelet_ds[local] = result.source_wavelet[0]
                physical_ds[local] = physical_channels.astype(np.float16)
                auxiliary_ds[local] = auxiliary_channels.astype(np.float16)
                out.flush()
                print(json.dumps({
                    "event": "cache_record", "shard": args.shard_index,
                    "record": local + 1, "of": count, "sample_id": row["sample_id"],
                    "family": family, "elapsed_s": round(time.time() - started, 1),
                    "maximum_normalized": current_maximum,
                }), flush=True)
            out.attrs["maximum_normalized"] = maximum_normalized
            out.attrs["status"] = "complete"
            out.flush()
        os.replace(partial, args.output)
    finally:
        partial.unlink(missing_ok=True)

    summary = {
        "schema": "transfer_dg_wfp_e1_cache_summary_v1",
        "status": "complete",
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": _sha256(args.output),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "selection_sha256": manifest["selection_sha256"],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "record_count": count,
        "family_counts": {
            family: sum(row["family"] == family for row in rows)
            for family in ("uniform", "layered", "anomaly", "marmousi")
        },
        "role_counts": {
            role: sum(row["role"] == role for row in rows)
            for role in ("fit", "holdout")
        },
        "frequency_count": frequency_count,
        "maximum_normalized": maximum_normalized,
        "elapsed_s": time.time() - started,
        "validation_opened": False,
        "test_id_opened": False,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
