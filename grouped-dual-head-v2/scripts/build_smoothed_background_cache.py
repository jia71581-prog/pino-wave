"""Build the SMOOTHED-VELOCITY BACKGROUND field cache P_bg for Direction A+1.

Direction A+ verdict (results/temporal_frequency_helmholtz_study.md sec 12-13):
running the LWC84 solver on a Gaussian-SMOOTHED velocity gives a background field
P_bg whose residual scat = wf - P_bg reconstructs to <1% (all families, all time bins,
including late) -- it fixes the late blow-up of the uniform-c0 incident field WITHOUT
Direction B's (falsified) time-harmonic premise. The learnable network then only needs
the smooth low-rank scattering residual on top of the physical background.

encode_pressure is LINEAR (value / (pressure_scale * amplitude)), so in normalized
space  encode(wf) - encode(P_bg) = encode(scat): the model learns encode(scat) and the
coarse field is  encode(P_bg) + synthesis.  This builder stores PHYSICAL P_bg; the model
side encodes it with the same per-record amplitude as the target.

Output schema = lwc84_multifidelity_teacher_v1 (so the existing NumericalTeacherCache
loads it unchanged), but:
  * velocity is Gaussian-smoothed (sigma in SAVED-grid cells) before the solve;
  * By default all 401 saved frames are stored. ``--time-count`` can instead create a
    deterministic sparse exact-time cache for large training pools.

Large caches are streamed directly to HDF5 one record at a time. ``--all-non-anomaly``
uses the manifest-compatible three-family census, and ``--exclude-cache`` lets a sparse
extension reuse an existing full-time cache without recomputing or overwriting it.

Usage:
  python3 scripts/build_smoothed_background_cache.py \
      --records 0 420 1540 2100 \
      --sigma 4.0 \
      --out /home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/background_pbg_sigma4_pilot.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "src")
from fno_acoustic.data_generation.config import (  # noqa: E402
    load_config, resolve_config, grid_from_config, boundaries_from_config, time_from_config,
)
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saved_time_phase_operator_v4.multifidelity import fixed_teacher_time_indices  # noqa: E402

DATA_DIR = "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1"
H5 = f"{DATA_DIR}/dataset_v1.h5"
FROZEN = f"{DATA_DIR}/frozen_config.yaml"
SCHEMA = "lwc84_multifidelity_teacher_v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    selection = ap.add_mutually_exclusive_group(required=True)
    selection.add_argument("--records", type=int, nargs="+",
                           help="dataset source indices to build P_bg for")
    selection.add_argument(
        "--all-non-anomaly",
        action="store_true",
        help="build every uniform/layered/marmousi sample across every source split",
    )
    ap.add_argument("--sigma", type=float, default=4.0,
                    help="Gaussian smoothing sigma in SAVED-grid (201, dx=10m) cells")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="records solved together on one device; source parameters remain per-record",
    )
    ap.add_argument("--source-h5", default=H5)
    ap.add_argument("--frozen-config", default=FROZEN)
    ap.add_argument(
        "--exclude-cache",
        action="append",
        default=[],
        help="skip sample IDs already stored in this complete cache; repeatable",
    )
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--time-count",
        type=int,
        default=401,
        help="fixed exact saved-time pool size spanning indices 0..400",
    )
    ap.add_argument("--selfcheck-uniform", action="store_true",
                    help="assert uniform-family P_bg reproduces the stored field bit-exactly")
    args = ap.parse_args()

    if int(args.shard_count) <= 0 or not 0 <= int(args.shard_index) < int(args.shard_count):
        ap.error("--shard-index must lie in [0, --shard-count)")
    if not np.isfinite(float(args.sigma)) or float(args.sigma) <= 0.0:
        ap.error("--sigma must be finite and positive")
    if int(args.batch_size) <= 0:
        ap.error("--batch-size must be positive")

    cfg = resolve_config(load_config(str(args.frozen_config)))
    grid = grid_from_config(cfg); bnd = boundaries_from_config(cfg); tg = time_from_config(cfg)
    dt_used = float(cfg["time"]["dt_used_s"])
    source_path = Path(args.source_h5).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite background cache: {out_path}")

    with h5py.File(source_path, "r", swmr=True) as source:
        full_time_s = np.asarray(source["time_s"][:], dtype=np.float64)
        time_indices = np.asarray(
            fixed_teacher_time_indices(
                stored_time_count=len(full_time_s),
                count=int(args.time_count),
            ),
            dtype=np.int64,
        )
        time_s = full_time_s[time_indices]
        sample_ids_all = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["sample_id"][:]
        ]
        medium_all = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["medium_type"][:]
        ]
        if args.all_non_anomaly:
            records = [
                index
                for index, medium in enumerate(medium_all)
                if medium.split("_")[0] in {"uniform", "layered", "marmousi"}
            ]
        else:
            records = list(dict.fromkeys(int(value) for value in args.records))

        if any(index < 0 or index >= len(sample_ids_all) for index in records):
            raise IndexError("requested source index is outside the dataset")
        excluded_sample_ids: set[str] = set()
        for cache_value in args.exclude_cache:
            cache_path = Path(cache_value).expanduser().resolve()
            with h5py.File(cache_path, "r", swmr=True) as cache:
                if str(cache.attrs.get("schema", "")) != SCHEMA or str(
                    cache.attrs.get("status", "")
                ) != "complete":
                    raise ValueError(f"exclude cache is not complete: {cache_path}")
                excluded_sample_ids.update(
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in cache["sample_id"][:]
                )

        records = sorted(
            index for index in records if sample_ids_all[index] not in excluded_sample_ids
        )
        records = records[int(args.shard_index) :: int(args.shard_count)]
        if not records:
            raise ValueError("background-cache selection is empty after exclusions/sharding")
        sample_ids = [sample_ids_all[index] for index in records]
        if len(set(sample_ids)) != len(sample_ids):
            raise RuntimeError("duplicate sample_ids among requested records")
        source_manifest_sha256 = str(source.attrs.get("manifest_sha256", ""))

    device = args.device if torch.cuda.is_available() else "cpu"
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=bnd,
        dt_s=dt_used,
        output_times_s=time_s,
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float32,
        kappa_max=float(cfg["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(cfg["boundaries"]["minimum_frequency_hz"]),
    )

    nt = len(time_s)
    nz_saved, nx_saved = 201, 201
    nz_fine, nx_fine = grid.nz, grid.nx

    def up(v):
        t = torch.as_tensor(v[None, None], dtype=torch.float32)
        return torch.nn.functional.interpolate(
            t, size=(nz_fine, nx_fine), mode="bilinear", align_corners=True
        )[0, 0].numpy().astype(np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_indices = np.asarray(records, dtype=np.int64)
    text_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(source_path, "r", swmr=True) as source, h5py.File(out_path, "x") as h:
        h.attrs["schema"] = SCHEMA
        h.attrs["status"] = "building"
        h.attrs["source_manifest_sha256"] = source_manifest_sha256
        h.attrs["background_sigma_saved_cells"] = float(args.sigma)
        h.attrs["background_kind"] = "gaussian_smoothed_velocity_lwc84_solve"
        h.attrs["selection"] = "all_non_anomaly" if args.all_non_anomaly else "explicit_records"
        h.attrs["excluded_cache_count"] = len(args.exclude_cache)
        h.attrs["shard_count"] = int(args.shard_count)
        h.attrs["shard_index"] = int(args.shard_index)
        h.create_dataset("sample_id", data=np.asarray(sample_ids, dtype=object), dtype=text_dtype)
        h.create_dataset("source_index", data=source_indices)
        h.create_dataset("time_indices", data=time_indices)
        h.create_dataset("time_s", data=time_s)
        wavefield = h.create_dataset(
            "wavefield",
            shape=(len(records), nt, nz_saved, nx_saved),
            dtype=np.float32,
            chunks=(1, 1, nz_saved, nx_saved),
            compression="gzip",
            compression_opts=1,
        )
        batch_size = int(args.batch_size)
        for batch_start in range(0, len(records), batch_size):
            batch_records = records[batch_start : batch_start + batch_size]
            families = [medium_all[ridx].split("_")[0] for ridx in batch_records]
            velocities_saved = [
                np.asarray(source["velocity_mps"][ridx], dtype=np.float64)
                for ridx in batch_records
            ]
            velocities_fine = np.stack(
                [
                    up(vel)
                    if family == "uniform"
                    else up(
                        gaussian_filter(
                            vel, sigma=float(args.sigma), mode="nearest"
                        )
                    )
                    for vel, family in zip(
                        velocities_saved, families, strict=True
                    )
                ],
                axis=0,
            )
            record_array = np.asarray(batch_records, dtype=np.int64)
            res = solver.simulate(
                velocities_fine,
                source_x_m=np.asarray(source["source_x_m"][record_array]),
                source_z_m=np.asarray(source["source_z_m"][record_array]),
                source_f0_hz=np.asarray(source["source_f0_hz"][record_array]),
                source_t0_s=np.asarray(source["source_t0_s"][record_array]),
                source_amplitude=np.asarray(source["source_amplitude"][record_array]),
            )
            batch_pbg = np.asarray(res.wavefield, dtype=np.float32)
            expected_shape = (len(batch_records), nt, nz_saved, nx_saved)
            if batch_pbg.shape != expected_shape:
                raise RuntimeError(f"unexpected P_bg shape {batch_pbg.shape}")
            for local, (ridx, fam) in enumerate(
                zip(batch_records, families, strict=True)
            ):
                row = batch_start + local
                pbg = batch_pbg[local]
                if args.selfcheck_uniform and fam == "uniform":
                    target = np.asarray(
                        source["wavefield"][ridx, time_indices, :, :],
                        dtype=np.float64,
                    )
                    rel = float(
                        np.linalg.norm(pbg - target)
                        / (np.linalg.norm(target) + 1e-30)
                    )
                    if rel > 1e-4:
                        raise AssertionError(
                            f"uniform self-check failed: rel-L2={rel:.2e}"
                        )
                    print(
                        f"  [selfcheck] uniform #{ridx} P_bg==stored rel-L2={rel:.2e}",
                        flush=True,
                    )
                wavefield[row] = pbg
                print(
                    f"  built P_bg {row + 1}/{len(records)} source #{ridx} "
                    f"({fam}) sigma={args.sigma} frames={nt} batch={len(batch_records)}",
                    flush=True,
                )
            h.flush()
        h.attrs["status"] = "complete"
        h.flush()
    print(f"wrote {out_path}  records={len(records)} frames={nt} batch={args.batch_size} "
          f"size={out_path.stat().st_size/1e6:.1f}MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
