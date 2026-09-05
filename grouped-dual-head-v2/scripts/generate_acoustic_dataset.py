#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time as walltime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fno_acoustic.data_generation.config import artifact_root, boundaries_from_config, grid_from_config, load_config, shard_root, time_from_config
from fno_acoustic.data_generation.cfl import CFLPlan, compute_cfl_plan
from fno_acoustic.data_generation.gpu_backend import check_cuda_preflight
from fno_acoustic.data_generation.gpu_uniform_analytic import simulate_uniform_gpu_analytic_compatible
from fno_acoustic.data_generation.hdf5_writer import AtomicShardWriter, IncrementalHDF5Writer
from fno_acoustic.data_generation.model_layered import generate_layered_model
from fno_acoustic.data_generation.model_uniform import uniform_velocity
from fno_acoustic.data_generation.source import bilinear_point_source, source_time_function
from fno_acoustic.data_generation.split_manifest import build_manifest
from fno_acoustic.data_generation.pipeline_lwc84 import (
    assert_production_gate,
    generate_production_worker,
    generate_smoke_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v3 acoustic dataset shards with the CUDA teacher backend.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--preset", choices=("unit", "smoke", "production"), default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--output", default=None)
    parser.add_argument("--confirm-production", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--splits", default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--frozen-manifest", action="store_true")
    parser.add_argument("--log-every", type=int, default=1, help="Print progress after every N newly generated samples.")
    parser.add_argument("--batch-size", type=int, default=None, help="Numeric teacher batch size. Defaults to the largest configured 5m candidate.")
    parser.add_argument("--medium-types", default=None)
    parser.add_argument("--sample-ids", default=None)
    return parser.parse_args()


def _velocity_for(row: dict, grid):
    if row["category"] == "uniform":
        return uniform_velocity(grid, 1500.0 + (row["base_model_id"] % 4000))
    velocity, _meta = generate_layered_model(grid, seed=int(row["base_model_id"]))
    return velocity


def production_cfl_plan(config: dict, grid, time) -> CFLPlan:
    velocity_range = config["models"]["uniform"]["velocity_mps"]
    c_global_max_mps = float(max(velocity_range))
    return compute_cfl_plan(
        c_global_max_mps=c_global_max_mps,
        grid=grid,
        time=time,
        cfl_axis_safety=float(config["time"]["cfl_axis_safety"]),
        symbol_limit_fraction=float(config["time"]["symbol_limit_fraction"]),
        fine_dx_m=float(config["time"]["fine_truth_grid"]["dx_m"]),
        fine_dz_m=float(config["time"]["fine_truth_grid"]["dz_m"]),
    )


def quality_from_cfl_plan(plan: CFLPlan) -> dict[str, float | int]:
    return {
        "cfl_axis": float(plan.cfl_axis_5m),
        "truth_internal_dx_m": 5.0,
        "truth_internal_dt_s": float(plan.dt_internal_5m_s),
        "truth_internal_n_substeps": int(plan.n_substeps_5m),
    }


def output_path_for_generation(config: dict, *, profile: str, splits: set[str]) -> tuple[Path, str]:
    if profile == "core_v3_gpu" and splits == {"train", "validation", "test"}:
        return Path(config["paths"]["final_hdf5"]), "final_hdf5"
    return shard_root(config) / f"{profile}_{'_'.join(sorted(splits))}_00000.h5", "shard"


def select_generation_batch_size(config: dict, requested_batch_size: int | None) -> int:
    if requested_batch_size is not None:
        if int(requested_batch_size) < 1:
            raise ValueError("batch size must be positive")
        return int(requested_batch_size)
    candidates = [int(v) for v in config["compute"].get("batch_candidates_5m", [1])]
    return max(candidates) if candidates else 1


def format_progress_log(
    *,
    index: int,
    total: int,
    completed: int,
    generated_this_run: int,
    skipped: int,
    row: dict,
    elapsed_s: float,
    sample_elapsed_s: float,
) -> str:
    percent = 100.0 * float(completed) / float(total) if total else 0.0
    rate = float(generated_this_run) / float(elapsed_s) if elapsed_s > 0.0 and generated_this_run > 0 else 0.0
    remaining = max(int(total) - int(completed), 0)
    eta_s = float(remaining) / rate if rate > 0.0 else None
    eta = f"{eta_s:.1f}s" if eta_s is not None else "unknown"
    return (
        f"progress sample={completed}/{total} ({percent:.3f}%) "
        f"idx={index} category={row['category']} split={row['split']} role={row['frequency_role']} "
        f"f0={float(row['f0_hz']):.3f}Hz generated_this_run={generated_this_run} skipped={skipped} "
        f"sample_s={sample_elapsed_s:.3f} elapsed={elapsed_s:.1f}s rate={rate:.3f} samples/s eta={eta}"
    )


def simulate_numeric_rows_wavefield(
    rows: list[dict],
    velocities: list[np.ndarray],
    *,
    backend,
    grid,
    time,
    boundaries,
    dt_internal_s: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if hasattr(backend, "prepare_velocity_batch"):
        prepared = backend.prepare_velocity_batch(np.stack(velocities, axis=0), grid=grid, boundaries=boundaries, dt_s=float(dt_internal_s))
        result = backend.simulate_batch(
            prepared,
            source_xy_m=np.asarray([row["source_xy_m"] for row in rows], dtype=np.float64),
            f0_hz=np.asarray([row["f0_hz"] for row in rows], dtype=np.float32),
            output_times_s=time.t_s,
        )
        wavefield = result.wavefield
        if hasattr(wavefield, "detach"):
            wavefield = wavefield.detach().to("cpu").numpy()
        return np.asarray(wavefield, dtype=np.float32), {"truth_backend": backend.backend_name, **dict(result.metrics)}

    wavefields: list[np.ndarray] = []
    elapsed = 0.0
    for row, velocity in zip(rows, velocities):
        wavefield, metrics = simulate_row_wavefield(
            row,
            velocity,
            backend=backend,
            grid=grid,
            time=time,
            boundaries=boundaries,
            dt_internal_s=dt_internal_s,
            device=str(getattr(backend, "device", "cuda")),
        )
        wavefields.append(wavefield)
        elapsed += float(metrics.get("elapsed_s", 0.0))
    return np.stack(wavefields, axis=0), {"truth_backend": getattr(backend, "backend_name", "unknown"), "elapsed_s": elapsed}


def simulate_row_wavefield(
    row: dict,
    velocity: np.ndarray,
    *,
    backend,
    grid,
    time,
    boundaries,
    dt_internal_s: float,
    device: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if row["category"] == "uniform":
        wavefield = simulate_uniform_gpu_analytic_compatible(
            grid,
            time,
            c_mps=float(np.mean(velocity)),
            source_xy_m=(float(row["source_xy_m"][0]), float(row["source_xy_m"][1])),
            f0_hz=float(row["f0_hz"]),
            device=device,
        )
        return wavefield, {"truth_backend": "gpu_analytic_halfspace"}
    prepared = backend.prepare_velocity(velocity, grid=grid, boundaries=boundaries, dt_s=float(dt_internal_s))
    result = backend.simulate_batch(
        prepared,
        source_xy_m=np.asarray([row["source_xy_m"]], dtype=np.float64),
        f0_hz=np.asarray([row["f0_hz"]], dtype=np.float32),
        output_times_s=time.t_s,
    )
    wavefield = result.wavefield[0]
    if hasattr(wavefield, "detach"):
        wavefield = wavefield.detach().to("cpu").numpy()
    return np.asarray(wavefield, dtype=np.float32), {"truth_backend": backend.backend_name, **dict(result.metrics)}


def _write_blocked(config: dict, report: dict) -> None:
    path = artifact_root(config) / "implementation_status.json"
    payload = {
        "v3_toolchain_implemented": True,
        "bootstrap_gate_passed": None,
        "manifest_dry_run_passed": None,
        "cuda_preflight_passed": False,
        "smoke_generated": False,
        "pilot_generated": False,
        "core_generated": False,
        "final_hdf5_exists": Path(config["paths"]["final_hdf5"]).exists(),
        "blocking_reason": "external_gpu_occupancy_or_insufficient_vram",
        "preflight": report,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if str(config.get("schema_version", "")).startswith("acoustic-lwc84"):
        preset = args.preset or "unit"
        if preset in {"unit", "smoke"}:
            output = Path(args.output or f"artifacts/dataset_{preset}_lwc84")
            payload = generate_smoke_dataset(
                config,
                preset=preset,
                device=args.device,
                num_samples=args.num_samples,
                output=output,
                resume=args.resume,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        assert_production_gate(
            config, confirm=args.confirm_production, device=args.device
        )
        if not args.output or not args.manifest:
            raise ValueError("production requires --output and --manifest")
        payload = generate_production_worker(
            config,
            output=args.output,
            manifest_path=args.manifest,
            device=args.device,
            worker_rank=args.worker_rank,
            num_workers=args.num_workers,
            resume=args.resume,
            batch_size=args.batch_size,
            splits={value.strip() for value in args.splits.split(",") if value.strip()},
            medium_types=(
                {value.strip() for value in args.medium_types.split(",") if value.strip()}
                if args.medium_types
                else None
            ),
            sample_ids=(
                {value.strip() for value in args.sample_ids.split(",") if value.strip()}
                if args.sample_ids
                else None
            ),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not args.profile:
        raise ValueError("legacy dataset generation requires --profile")
    from fno_acoustic.data_generation.high_order_teacher import TorchCompiledAcousticBackend

    if args.no_cpu_fallback and args.device != "cuda":
        raise RuntimeError("CPU fallback is forbidden for v3 GPU production")
    report = check_cuda_preflight(
        require_cuda=args.device == "cuda",
        min_free_vram_gib=float(config["compute"]["min_free_vram_gib"]),
        no_cpu_fallback=args.no_cpu_fallback,
    )
    if args.device == "cuda" and int(report["exit_code"]) != 0:
        _write_blocked(config, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 3
    grid = grid_from_config(config)
    time = time_from_config(config)
    boundaries = boundaries_from_config(config)
    cfl_plan = production_cfl_plan(config, grid, time)
    rows = build_manifest(config, profile=args.profile, enabled_categories=None)
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    rows = [row for row in rows if row["split"] in splits]
    if args.limit_batches is not None:
        rows = rows[: max(0, int(args.limit_batches))]
    if not rows:
        return 0
    backend = TorchCompiledAcousticBackend(grid=grid, time=time, device=args.device, n_substeps=cfl_plan.n_substeps_5m)
    output_path, output_mode = output_path_for_generation(config, profile=args.profile, splits=splits)
    batch_size = select_generation_batch_size(config, args.batch_size)
    root_attrs = {
        "production_device": "cuda",
        "gpu_backend": backend.backend_name,
        "cpu_fallback": False,
        "cfl_plan": json.dumps(cfl_plan.as_dict(), sort_keys=True),
        "solver_orders": json.dumps(backend.solver_orders, sort_keys=True),
        "sample_count": len(rows),
        "generation_batch_size": int(batch_size),
    }
    if output_mode == "final_hdf5":
        writer = IncrementalHDF5Writer(
            output_path,
            sample_count=len(rows),
            grid=grid,
            time=time,
            root_attrs=root_attrs | {"write_mode": "incremental_final_hdf5"},
            resume=args.resume,
        )
    else:
        writer = AtomicShardWriter(output_path, sample_count=len(rows), grid=grid, time=time, root_attrs=root_attrs)
    base_quality = quality_from_cfl_plan(cfl_plan)
    skipped = 0
    generated_this_run = 0
    initial_completed = writer.completed_count() if hasattr(writer, "completed_count") else 0
    started_at = walltime.monotonic()
    log_every = max(1, int(args.log_every))
    print(
        f"progress_start output={output_path} mode={output_mode} total={len(rows)} "
        f"completed={initial_completed} remaining={max(len(rows) - initial_completed, 0)} resume={bool(args.resume)} batch_size={batch_size}",
        flush=True,
    )
    i = 0
    while i < len(rows):
        row = rows[i]
        if args.resume and hasattr(writer, "is_complete") and writer.is_complete(i):
            skipped += 1
            i += 1
            continue
        sample_started_at = walltime.monotonic()
        if row["category"] == "uniform":
            batch_indices = [i]
            batch_rows = [row]
        else:
            batch_indices = []
            batch_rows = []
            while i < len(rows) and len(batch_rows) < batch_size:
                candidate = rows[i]
                if args.resume and hasattr(writer, "is_complete") and writer.is_complete(i):
                    skipped += 1
                    i += 1
                    continue
                if candidate["category"] == "uniform":
                    break
                batch_indices.append(i)
                batch_rows.append(candidate)
                i += 1
        if not batch_rows:
            continue
        velocities = [_velocity_for(batch_row, grid) for batch_row in batch_rows]
        if len(batch_rows) == 1 and batch_rows[0]["category"] == "uniform":
            wavefield, metrics = simulate_row_wavefield(
                batch_rows[0],
                velocities[0],
                backend=backend,
                grid=grid,
                time=time,
                boundaries=boundaries,
                dt_internal_s=cfl_plan.dt_internal_5m_s,
                device=args.device,
            )
            wavefields = np.asarray(wavefield, dtype=np.float32)[None, ...]
        else:
            wavefields, metrics = simulate_numeric_rows_wavefield(
                batch_rows,
                velocities,
                backend=backend,
                grid=grid,
                time=time,
                boundaries=boundaries,
                dt_internal_s=cfl_plan.dt_internal_5m_s,
            )
        batch_elapsed = walltime.monotonic() - sample_started_at
        solver_elapsed = float(metrics.get("elapsed_s", batch_elapsed))
        for local_idx, (sample_index, batch_row, velocity) in enumerate(zip(batch_indices, batch_rows, velocities)):
            source = bilinear_point_source(
                batch_row["source_xy_m"][0],
                batch_row["source_xy_m"][1],
                nx=grid.nx,
                nz=grid.nz,
                dx_m=grid.dx_m,
                dz_m=grid.dz_m,
            )
            writer.write_sample(
                sample_index,
                row=batch_row,
                velocity_mps=velocity,
                wavefield=wavefields[local_idx],
                source=source,
                source_wavelet_out=source_time_function(time.t_s, batch_row["f0_hz"]).astype(np.float32),
                quality={
                    "c_min_mps": float(np.min(velocity)),
                    "c_max_mps": float(np.max(velocity)),
                    "passed": True,
                    "gpu_batch_size": len(batch_rows),
                    "gpu_solver_elapsed_s": solver_elapsed / max(len(batch_rows), 1),
                    **base_quality,
                },
            )
        generated_this_run += len(batch_rows)
        completed_now = initial_completed + generated_this_run
        if generated_this_run == 1 or generated_this_run % log_every == 0 or completed_now == len(rows):
            print(
                format_progress_log(
                    index=batch_indices[-1],
                    total=len(rows),
                    completed=completed_now,
                    generated_this_run=generated_this_run,
                    skipped=skipped,
                    row=batch_rows[-1],
                    elapsed_s=walltime.monotonic() - started_at,
                    sample_elapsed_s=batch_elapsed / max(len(batch_rows), 1),
                ),
                flush=True,
            )
        if len(batch_rows) == 1 and batch_rows[0]["category"] == "uniform":
            i += 1
    completed = writer.completed_count() if hasattr(writer, "completed_count") else len(rows)
    final = writer.close_and_commit()
    payload_key = "generated_hdf5" if output_mode == "final_hdf5" else "generated_shard"
    print(json.dumps({payload_key: str(final), "sample_count": len(rows), "completed_count": completed, "skipped_count": skipped}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
