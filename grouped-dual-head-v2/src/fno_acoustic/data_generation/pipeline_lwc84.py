from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import h5py

from .cfl import plan_lwc84_timestep
from .config import boundaries_from_config, grid_from_config, time_from_config
from .grid import AcousticGrid, BoundaryConfig
from .hdf5_lwc84 import LWC84ShardWriter, validate_lwc84_shard
from .holdout_audit import (
    audit_candidate_split_disjointness,
    audit_historical_manifest_disjointness,
)
from .lwc84_manifest import (
    MEDIUM_IDS,
    SPLIT_IDS,
    build_lwc84_manifest,
    configured_lwc84_counts,
    validate_lwc84_manifest,
)
from .model_marmousi import _load_velocity, inventory_marmousi, select_full_marmousi_candidate
from .solver_lwc84 import LWC84CPMLSolver
from .velocity_models_lwc84 import (
    generate_anomaly_velocity,
    generate_layered_velocity,
    generate_uniform_velocity,
    load_marmousi_crop,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _decode_hdf5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def validate_production_shard_binding(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    split: str,
    config_sha256: str,
    manifest_sha256: str,
    marmousi_sha256: str,
) -> None:
    """Reject a valid but unrelated shard before it can be reused."""
    path = Path(path)
    expected_ids = [str(row["sample_id"]) for row in rows]
    with h5py.File(path, "r") as h5:
        actual_ids = [_decode_hdf5_text(value) for value in h5["sample_id"][:]]
        expected_attrs = {
            "split": split,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "marmousi_sha256": marmousi_sha256,
        }
        mismatched = {
            name: (_decode_hdf5_text(h5.attrs.get(name, "<missing>")), expected)
            for name, expected in expected_attrs.items()
            if _decode_hdf5_text(h5.attrs.get(name, "<missing>")) != expected
        }
    if actual_ids != expected_ids:
        raise ValueError(
            f"production shard sample binding mismatch for {path}: "
            f"actual={actual_ids[:3]}, expected={expected_ids[:3]}"
        )
    if mismatched:
        raise ValueError(f"production shard provenance binding mismatch for {path}: {mismatched}")


def _git_commit(project_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, capture_output=True, check=False
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def _existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def storage_budget(config: dict[str, Any], *, sample_count: int | None = None) -> dict[str, Any]:
    if sample_count is None:
        sample_count = int(configured_lwc84_counts(config)["sample_count"])
    if int(sample_count) <= 0:
        raise ValueError("storage budget requires at least one configured sample")
    nz = int(config["storage_grid"]["nz"])
    nx = int(config["storage_grid"]["nx"])
    nt = int(config["time"]["nt_out"])
    wavefield = int(sample_count) * nt * nz * nx * np.dtype("float32").itemsize
    velocity_and_source = int(sample_count) * 2 * nz * nx * np.dtype("float32").itemsize
    wavelet = int(sample_count) * nt * np.dtype("float32").itemsize
    metadata = int(sample_count) * 1024
    uncompressed = wavefield + velocity_and_source + wavelet + metadata
    # The default remains conservative. A frozen config may provide a narrower
    # empirical interval only when it is backed by same-protocol completed data.
    compression_range = config["storage"].get(
        "estimated_compression_ratio_range", [0.55, 0.85]
    )
    if len(compression_range) != 2:
        raise ValueError("storage estimated compression ratio range needs two values")
    ratio_low, ratio_high = (float(value) for value in compression_range)
    if not 0.0 < ratio_low <= ratio_high <= 1.0:
        raise ValueError("storage estimated compression ratios must satisfy 0 < low <= high <= 1")
    compressed_low = int(uncompressed * ratio_low)
    compressed_high = int(uncompressed * ratio_high)
    samples_per_shard = max(int(config["storage"]["samples_per_shard"]), 1)
    temporary = int(math.ceil(uncompressed / int(sample_count) * samples_per_shard))
    safety_factor = float(config["production"]["require_free_disk_safety_factor"])
    required_free = int((compressed_high + temporary) * safety_factor)
    return {
        "sample_count_including_ood": int(sample_count),
        "saved_shape_per_sample_NTZX": [1, nt, nz, nx],
        "wavefield_uncompressed_bytes": wavefield,
        "static_and_metadata_uncompressed_bytes": uncompressed - wavefield,
        "total_uncompressed_bytes": uncompressed,
        "estimated_lzf_bytes_range": [compressed_low, compressed_high],
        "estimated_lzf_ratio_range": [ratio_low, ratio_high],
        "compression_estimate_evidence": config["storage"].get(
            "compression_estimate_evidence", "conservative_default"
        ),
        "estimated_temporary_bytes": temporary,
        "temporary_space_assumption": "one concurrent uncompressed partial shard",
        "safety_factor": safety_factor,
        "required_free_bytes": required_free,
    }


def validated_lwc84_time_plan(config: dict[str, Any]):
    """Build the LWC time plan and reject internally inconsistent frozen configs."""
    vmax_upper = max(
        float(config["models"]["uniform"]["velocity_mps"][1]),
        float(config["models"]["layered"]["velocity_mps"][1]),
        float(config["models"]["anomaly"]["background_velocity_mps"][1])
        * (1.0 + float(config["models"]["anomaly"]["relative_contrast_abs"][1])),
    )
    time_config = config["time"]
    plan = plan_lwc84_timestep(
        global_vmax_mps=vmax_upper,
        dx_m=float(config["grid"]["dx_m"]),
        dz_m=float(config["grid"]["dz_m"]),
        dt_requested_s=float(time_config["dt_requested_s"]),
        output_interval_s=float(time_config["dt_out_s"]),
        snapshot_stride_requested=int(time_config["snapshot_stride_requested"]),
        engineering_cfl_limit=float(time_config["engineering_cfl_limit"]),
        lwc_q_safety_limit=float(time_config["lwc_q_safety_limit"]),
    )
    configured_dt = float(time_config["dt_used_s"])
    configured_stride = int(time_config["snapshot_stride"])
    if not math.isclose(configured_dt, plan.dt_used_s, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(
            f"time.dt_used_s={configured_dt} differs from the aligned LWC plan {plan.dt_used_s}"
        )
    if configured_stride != plan.snapshot_stride:
        raise ValueError(
            "time.snapshot_stride differs from the aligned LWC plan: "
            f"{configured_stride} != {plan.snapshot_stride}"
        )
    expected_t_end = (int(time_config["nt_out"]) - 1) * float(time_config["dt_out_s"])
    if not math.isclose(
        float(time_config["t_end_s"]), expected_t_end, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("time.t_end_s is inconsistent with nt_out and dt_out_s")
    return plan


def _write_marmousi_extent_figure(inventory, output: Path, config: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not inventory:
        return
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
         "font.size": 9, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"}
    )
    fig, axes = plt.subplots(1, len(inventory), figsize=(8.0 * len(inventory), 3.2), squeeze=False)
    for axis, row in zip(axes[0], inventory, strict=True):
        array, _ = _load_velocity(Path(row.path))
        extent = [0.0, (array.shape[1] - 1) * float(row.dx_m), (array.shape[0] - 1) * float(row.dz_m), 0.0]
        image = axis.imshow(
            np.asarray(array),
            extent=extent,
            cmap="turbo",
            vmin=float(row.velocity_min_mps),
            vmax=float(row.velocity_max_mps),
            aspect="equal",
        )
        crop_x0 = float(config["marmousi"]["ood_crop_x0_m"])
        crop_z0 = float(config["marmousi"]["ood_crop_z0_m"])
        axis.add_patch(
            Rectangle(
                (crop_x0, crop_z0),
                2000.0,
                2000.0,
                fill=False,
                edgecolor="#D55E00",
                linewidth=1.8,
                linestyle="--",
                label="2 km × 2 km OOD crop",
            )
        )
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_title(
            f"Full Marmousi-1: {extent[1] / 1000.0:.1f} km × {extent[2] / 1000.0:.1f} km\n"
            f"{row.shape[0]}×{row.shape[1]} nodes, {float(row.dx_m):g} m spacing"
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)")
        axis.legend(loc="lower right", framealpha=0.9)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="P-wave velocity (m/s)", shrink=0.78)
    fig.savefig(output / "marmousi_extent_audit.pdf")
    fig.savefig(output / "marmousi_extent_audit.png")
    plt.close(fig)


def plan_dataset(config: dict[str, Any], *, output: str | Path) -> tuple[dict[str, Any], int]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inventory = inventory_marmousi(Path(config["marmousi"]["root_dir"]))
    inventory_payload = [row.as_dict() for row in inventory]
    (output / "marmousi_inventory.json").write_text(
        json.dumps(inventory_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sample_count = int(configured_lwc84_counts(config)["sample_count"])
    budget = storage_budget(config, sample_count=sample_count)
    disk = shutil.disk_usage(_existing_parent(output))
    time_plan = validated_lwc84_time_plan(config)
    vmax_upper = time_plan.global_vmax_mps
    total_steps = int(round(float(config["time"]["t_end_s"]) / time_plan.dt_used_s)) * sample_count
    report: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "grid": {
            "solver": [401, 401],
            "saved": [201, 201],
            "saved_time_frames": int(config["time"]["nt_out"]),
        },
        "marmousi_inventory": inventory_payload,
        "global_vmax_safe_upper_bound_mps": vmax_upper,
        "time_plan": time_plan.as_dict(),
        "total_solver_steps": total_steps,
        "configured_sample_count": sample_count,
        "storage_budget": budget,
        "disk": {"path": str(_existing_parent(output)), "free_bytes": int(disk.free), "total_bytes": int(disk.total)},
        "disk_safety_passed": bool(disk.free >= budget["required_free_bytes"]),
    }
    if sample_count == 4003:
        report["total_solver_steps_for_4003_samples"] = total_steps
    try:
        selected = select_full_marmousi_candidate(inventory, config["marmousi"])
    except Exception as exc:
        report.update(
            {
                "status": "BLOCKED_MARMOUSI_EXTENT",
                "blocking_reason": str(exc),
                "production_ready": False,
                "manifest_written": False,
            }
        )
        (output / "plan_summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report, 3
    _write_marmousi_extent_figure([selected], output, config)
    rows = build_lwc84_manifest(
        config,
        marmousi_geometry={
            "shape": selected.shape,
            "dx_m": selected.dx_m,
            "dz_m": selected.dz_m,
            "sha256": selected.sha256,
        },
    )
    manifest_summary = validate_lwc84_manifest(rows, config)
    internal_split_audit = audit_candidate_split_disjointness(rows)
    (output / "internal_split_overlap_audit.json").write_text(
        json.dumps(internal_split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    frozen = output / "frozen_config.yaml"
    frozen.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    historical_manifests = config["dataset"].get("historical_manifests")
    holdout_audit = None
    if historical_manifests is not None:
        holdout_audit = audit_historical_manifest_disjointness(rows, historical_manifests)
        (output / "historical_overlap_audit.json").write_text(
            json.dumps(holdout_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    overlap_passed = bool(internal_split_audit["passed"]) and (
        holdout_audit is None or bool(holdout_audit["passed"])
    )
    ready = bool(report["disk_safety_passed"] and overlap_passed)
    status = "READY" if ready else (
        "BLOCKED_HISTORICAL_OVERLAP" if not overlap_passed else "BLOCKED_DISK"
    )
    report.update(
        {
            "status": status,
            "production_ready": ready,
            "manifest_written": True,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "manifest_summary": manifest_summary,
            "selected_marmousi": selected.as_dict(),
            "historical_overlap_audit": holdout_audit,
            "internal_split_overlap_audit": internal_split_audit,
        }
    )
    (output / "plan_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report, 0 if report["production_ready"] else 3


def _smoke_problem(config: dict[str, Any], preset: str) -> tuple[AcousticGrid, BoundaryConfig, np.ndarray, float]:
    if preset == "unit":
        grid = AcousticGrid(nx=41, nz=41, dx_m=5.0, dz_m=5.0, lx_m=200.0, lz_m=200.0, centering="node")
        boundaries = BoundaryConfig(npml=8, cpml_target_reflection=1.0e-8, cpml_polynomial_order=3)
    else:
        grid = grid_from_config(config)
        boundaries = boundaries_from_config(config)
    requested_t_end = 0.20
    configured_times = time_from_config(config).t_s
    output_times = configured_times[configured_times <= requested_t_end + 1.0e-12]
    if output_times.size < 2:
        raise ValueError("smoke output schedule must contain at least two configured frames")
    t_end = float(output_times[-1])
    return grid, boundaries, output_times, t_end


def _sample_velocity(grid: AcousticGrid, index: int, seed: int) -> tuple[np.ndarray, str, dict[str, Any]]:
    kind = ("uniform", "layered", "anomaly")[index % 3]
    if kind == "uniform":
        velocity, metadata = generate_uniform_velocity(grid, seed=seed)
    elif kind == "layered":
        velocity, metadata = generate_layered_velocity(
            grid, seed=seed, minimum_thickness_m=min(150.0, float(grid.lz_m) / 6.0)
        )
    else:
        velocity, metadata = generate_anomaly_velocity(grid, seed=seed)
    return velocity, kind, metadata


def generate_smoke_dataset(
    config: dict[str, Any],
    *,
    preset: str,
    device: str,
    num_samples: int,
    output: str | Path,
    resume: bool,
) -> dict[str, Any]:
    if preset not in {"unit", "smoke"}:
        raise ValueError("generate_smoke_dataset only accepts unit or smoke")
    if preset == "smoke" and device != "cuda":
        raise ValueError("smoke preset requires CUDA; it never silently falls back to CPU")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback was used")
    if int(num_samples) <= 0 or int(num_samples) > 8:
        raise ValueError("unit/smoke num_samples must be between 1 and 8")
    output = Path(output)
    shard = output / "shards" / f"train-{preset}-00000.h5"
    if shard.exists():
        if not resume:
            raise FileExistsError(shard)
        summary = validate_lwc84_shard(shard, strict=True)
        return {"status": "REUSED_COMPLETE_SHARD", **summary}
    grid, boundaries, output_times, _ = _smoke_problem(config, preset)
    time_plan = validated_lwc84_time_plan(config)
    dt = time_plan.dt_used_s
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=boundaries,
        dt_s=dt,
        output_times_s=output_times,
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float64 if device == "cpu" else torch.float32,
        kappa_max=float(config["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(config["boundaries"]["minimum_frequency_hz"]),
    )
    project_root = Path(config["paths"]["project_root"])
    attrs = {
        "dt_requested_s": float(config["time"]["dt_requested_s"]),
        "dt_used_s": dt,
        "snapshot_stride": time_plan.snapshot_stride,
        "forward_protocol": str(config["schema_version"]),
        "config_sha256": config["config_sha256"],
        "manifest_sha256": "smoke-no-production-manifest",
        "marmousi_sha256": str(config["marmousi"]["sha256"]),
        "git_commit": _git_commit(project_root),
        "software_environment": json.dumps(
            {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": device},
            sort_keys=True,
        ),
        "generation_preset": preset,
        "production_device": device,
        "solver_grid_shape": json.dumps([grid.nz, grid.nx]),
        "saved_grid_shape": json.dumps([(grid.nz + 1) // 2, (grid.nx + 1) // 2]),
    }
    writer = LWC84ShardWriter(
        shard,
        sample_count=int(num_samples),
        split="train",
        time_s=output_times,
        x_m=grid.x_m[::2],
        z_m=grid.z_m[::2],
        attrs=attrs,
        resume=resume,
    )
    generated_rows: list[dict[str, Any]] = []
    for index in range(int(num_samples)):
        if writer.is_complete(index):
            continue
        seed = int(config["seed"]) + index * 104729
        velocity, medium_type, medium_metadata = _sample_velocity(grid, index, seed)
        source_x = 0.5 * grid.lx_m + (index % 2 - 0.5) * 0.1 * grid.lx_m
        source_z = max(50.0, 0.1 * grid.lz_m)
        source_z = min(source_z, grid.lz_m - 4.0 * grid.dz_m)
        f0 = (10.0, 15.0, 25.0, 12.0)[index % 4]
        result = solver.simulate(
            velocity,
            source_x_m=source_x,
            source_z_m=source_z,
            source_f0_hz=f0,
            source_amplitude=1.0,
        )
        metrics = result.metrics[0]
        writer.write_sample(
            index,
            velocity_mps=result.velocity_saved_mps[0],
            wavefield=result.wavefield[0],
            source_map=result.source_map_saved[0],
            source_wavelet=result.source_wavelet[0],
            source_x_m=source_x,
            source_z_m=source_z,
            source_f0_hz=f0,
            source_t0_s=1.5 / f0,
            source_amplitude=1.0,
            medium_type=medium_type,
            sample_id=f"{preset}-{index:03d}",
            group_id=f"{preset}:{medium_type}:{index:03d}",
            seed=seed,
            vmin_mps=metrics["vmin_mps"],
            vmax_mps=metrics["vmax_mps"],
            cfl_2d=metrics["cfl_2d"],
            lwc_qmax=metrics["lwc_qmax"],
            dt_used_s=metrics["dt_used_s"],
            crop_x0_m=np.nan,
            crop_z0_m=np.nan,
            qc_max_abs=metrics["qc_max_abs"],
            qc_final_energy_ratio=metrics["qc_final_energy_ratio"],
            qc_status="passed" if metrics["finite"] else "failed",
        )
        generated_rows.append({"sample_id": f"{preset}-{index:03d}", "medium": medium_metadata})
    final = writer.commit()
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_manifest.json").write_text(
        json.dumps(generated_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "COMPLETE",
        "preset": preset,
        "device": device,
        "sample_count": int(num_samples),
        "shard": str(final.resolve()),
        "validation": validate_lwc84_shard(final, strict=True),
    }


def assert_production_gate(config: dict[str, Any], *, confirm: str | None, device: str) -> None:
    expected = str(config["production"]["confirm_token"])
    if confirm != expected:
        raise PermissionError(f"production requires --confirm-production {expected}")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production requires CUDA and forbids CPU fallback")
    inventory = inventory_marmousi(Path(config["marmousi"]["root_dir"]))
    select_full_marmousi_candidate(inventory, config["marmousi"])


def _production_velocity(config: dict[str, Any], grid: AcousticGrid, row: dict[str, Any]) -> np.ndarray:
    medium = str(row["medium_type"])
    parameters = row["medium_parameters"]
    if medium == "uniform":
        return np.full((grid.nz, grid.nx), float(parameters["velocity_mps"]), dtype=np.float32)
    if medium == "layered":
        fixed_keys = {"upper_velocity_mps", "lower_velocity_mps", "interface_z_m"}
        if fixed_keys.issubset(parameters):
            velocity_column = np.where(
                grid.z_m[:, None] < float(parameters["interface_z_m"]),
                float(parameters["upper_velocity_mps"]),
                float(parameters["lower_velocity_mps"]),
            )
            return np.broadcast_to(velocity_column, (grid.nz, grid.nx)).astype(np.float32, copy=True)
        return generate_layered_velocity(grid, seed=int(parameters["seed"]))[0]
    if medium == "anomaly":
        return generate_anomaly_velocity(grid, seed=int(parameters["seed"]))[0]
    if medium == "marmousi":
        velocity = load_marmousi_crop(
            config["marmousi"]["velocity_file"],
            grid=grid,
            source_dx_m=float(config["marmousi"]["source_dx_m"]),
            source_dz_m=float(config["marmousi"]["source_dz_m"]),
            source_unit=str(config["marmousi"]["velocity_unit"]),
            crop_x0_m=float(row["crop_x0_m"]),
            crop_z0_m=float(row["crop_z0_m"]),
            interpolation=str(config["marmousi"]["interpolation"]),
        )[0]
        minimum_exclusive = config["marmousi"].get(
            "minimum_crop_velocity_exclusive_mps"
        )
        if minimum_exclusive is not None and np.any(
            velocity <= float(minimum_exclusive)
        ):
            raise ValueError(
                "Marmousi crop violates the configured no-water velocity threshold"
            )
        return velocity
    raise ValueError(f"unsupported production medium_type={medium}")


def prepare_production_batch(
    rows: list[dict[str, Any]], velocities: list[np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not rows or len(rows) != len(velocities):
        raise ValueError("production batch rows and velocities must have the same positive length")
    velocity_batch = np.stack([np.asarray(value, dtype=np.float32) for value in velocities], axis=0)
    parameters = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in (
            "source_x_m",
            "source_z_m",
            "source_f0_hz",
            "source_t0_s",
            "source_amplitude",
        )
    }
    return velocity_batch, parameters


def production_batch_size(config: dict[str, Any], *, requested: int | None = None) -> int:
    value = requested
    if value is None:
        value = config["production"].get("batch_size", config["storage"]["samples_per_shard"])
    return max(1, int(value))


def group_partitions_for_batch(
    partitions: list[tuple[str, int, list[dict[str, Any]]]], *, batch_size: int
) -> list[list[tuple[str, int, list[dict[str, Any]]]]]:
    groups: list[list[tuple[str, int, list[dict[str, Any]]]]] = []
    current: list[tuple[str, int, list[dict[str, Any]]]] = []
    current_count = 0
    for partition in partitions:
        count = len(partition[2])
        if current and current_count + count > int(batch_size):
            groups.append(current)
            current, current_count = [], 0
        current.append(partition)
        current_count += count
    if current:
        groups.append(current)
    return groups


def generate_production_worker(
    config: dict[str, Any],
    *,
    output: str | Path,
    manifest_path: str | Path,
    device: str,
    worker_rank: int,
    num_workers: int,
    resume: bool,
    batch_size: int | None = None,
    splits: set[str] | None = None,
    medium_types: set[str] | None = None,
    sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    if device != "cuda":
        raise RuntimeError("production worker requires CUDA")
    output = Path(output)
    manifest_path = Path(manifest_path)
    all_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_hash = sha256_file(manifest_path)
    rows = all_rows
    if splits is not None:
        unknown_splits = set(splits) - set(SPLIT_IDS)
        if unknown_splits:
            raise ValueError(f"unknown production splits: {sorted(unknown_splits)}")
        rows = [row for row in rows if str(row["split"]) in splits]
    if medium_types is not None:
        unknown = set(medium_types) - set(MEDIUM_IDS)
        if unknown:
            raise ValueError(f"unknown production medium types: {sorted(unknown)}")
        rows = [row for row in rows if str(row["medium_type"]) in medium_types]
    if sample_ids is not None:
        available = {str(row["sample_id"]) for row in rows}
        missing = set(sample_ids) - available
        if missing:
            raise ValueError(f"requested production sample IDs are absent: {sorted(missing)}")
        rows = [row for row in rows if str(row["sample_id"]) in sample_ids]
    grid = grid_from_config(config)
    boundaries = boundaries_from_config(config)
    output_times = time_from_config(config).t_s
    time_plan = validated_lwc84_time_plan(config)
    dt = time_plan.dt_used_s
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=boundaries,
        dt_s=dt,
        output_times_s=output_times,
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float32,
        kappa_max=float(config["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(config["boundaries"]["minimum_frequency_hz"]),
    )
    shard_size = int(config["storage"]["samples_per_shard"])
    partitions: list[tuple[str, int, list[dict[str, Any]]]] = []
    for split in ("train", "validation", "test_id", "ood_canonical"):
        split_rows = [row for row in rows if row["split"] == split]
        for ordinal, start in enumerate(range(0, len(split_rows), shard_size)):
            partitions.append((split, ordinal, split_rows[start : start + shard_size]))
    assigned = [item for global_ordinal, item in enumerate(partitions) if global_ordinal % int(num_workers) == int(worker_rank)]
    completed: list[str] = []
    reused: list[str] = []
    project_root = Path(config["paths"]["project_root"])
    common_attrs = {
        "dt_requested_s": float(config["time"]["dt_requested_s"]),
        "dt_used_s": dt,
        "snapshot_stride": time_plan.snapshot_stride,
        "forward_protocol": str(config["schema_version"]),
        "config_sha256": config["config_sha256"],
        "manifest_sha256": manifest_hash,
        "marmousi_sha256": str(config["marmousi"]["sha256"]),
        "git_commit": _git_commit(project_root),
        "software_environment": json.dumps(
            {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": device},
            sort_keys=True,
        ),
        "generation_preset": "production",
        "production_device": "cuda",
        "cpu_fallback": False,
        "solver_grid_shape": json.dumps([grid.nz, grid.nx]),
        "saved_grid_shape": json.dumps([(grid.nz + 1) // 2, (grid.nx + 1) // 2]),
    }
    failure_log = output / "failures" / f"worker-{int(worker_rank):03d}.jsonl"
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    batch_size = production_batch_size(config, requested=batch_size)
    common_attrs["production_batch_size"] = batch_size
    for partition_group in group_partitions_for_batch(assigned, batch_size=batch_size):
        open_writers: list[tuple[LWC84ShardWriter, list[dict[str, Any]]]] = []
        try:
            for split, ordinal, shard_rows in partition_group:
                shard_path = output / "shards" / split / f"{split}-{ordinal:05d}.h5"
                if shard_path.exists():
                    if not resume:
                        raise FileExistsError(shard_path)
                    validate_lwc84_shard(shard_path, strict=True)
                    validate_production_shard_binding(
                        shard_path,
                        shard_rows,
                        split=split,
                        config_sha256=str(config["config_sha256"]),
                        manifest_sha256=manifest_hash,
                        marmousi_sha256=str(config["marmousi"]["sha256"]),
                    )
                    reused.append(str(shard_path.resolve()))
                    continue
                writer = LWC84ShardWriter(
                    shard_path,
                    sample_count=len(shard_rows),
                    split=split,
                    time_s=output_times,
                    x_m=grid.x_m[::2],
                    z_m=grid.z_m[::2],
                    attrs=common_attrs,
                    resume=resume,
                    expected_sample_ids=[str(row["sample_id"]) for row in shard_rows],
                )
                open_writers.append((writer, shard_rows))
            pending_items = [
                (writer, local_index, row)
                for writer, shard_rows in open_writers
                for local_index, row in enumerate(shard_rows)
                if not writer.is_complete(local_index)
            ]
            for batch_start in range(0, len(pending_items), batch_size):
                batch_items = pending_items[batch_start : batch_start + batch_size]
                batch_rows = [item[2] for item in batch_items]
                try:
                    velocities = [_production_velocity(config, grid, row) for row in batch_rows]
                    velocity_batch, source_parameters = prepare_production_batch(batch_rows, velocities)
                    result = solver.simulate(velocity_batch, **source_parameters)
                    for result_index, (writer, local_index, row) in enumerate(batch_items):
                        metrics = result.metrics[result_index]
                        writer.write_sample(
                            local_index,
                            velocity_mps=result.velocity_saved_mps[result_index],
                            wavefield=result.wavefield[result_index],
                            source_map=result.source_map_saved[result_index],
                            source_wavelet=result.source_wavelet[result_index],
                            source_x_m=row["source_x_m"],
                            source_z_m=row["source_z_m"],
                            source_f0_hz=row["source_f0_hz"],
                            source_t0_s=row["source_t0_s"],
                            source_amplitude=row["source_amplitude"],
                            medium_type=row["medium_type"],
                            sample_id=row["sample_id"],
                            group_id=row["group_id"],
                            seed=row["seed"],
                            vmin_mps=metrics["vmin_mps"],
                            vmax_mps=metrics["vmax_mps"],
                            cfl_2d=metrics["cfl_2d"],
                            lwc_qmax=metrics["lwc_qmax"],
                            dt_used_s=metrics["dt_used_s"],
                            crop_x0_m=np.nan if row["crop_x0_m"] is None else row["crop_x0_m"],
                            crop_z0_m=np.nan if row["crop_z0_m"] is None else row["crop_z0_m"],
                            qc_max_abs=metrics["qc_max_abs"],
                            qc_final_energy_ratio=metrics["qc_final_energy_ratio"],
                            qc_status="passed" if metrics["finite"] else "failed",
                        )
                except Exception as exc:
                    with failure_log.open("a", encoding="utf-8") as stream:
                        for row in batch_rows:
                            stream.write(
                                json.dumps(
                                    {"sample_id": row["sample_id"], "seed": row["seed"], "error": repr(exc)},
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                    raise
            for writer, shard_rows in open_writers:
                committed_path = writer.commit()
                validate_production_shard_binding(
                    committed_path,
                    shard_rows,
                    split=str(shard_rows[0]["split"]),
                    config_sha256=str(config["config_sha256"]),
                    manifest_sha256=manifest_hash,
                    marmousi_sha256=str(config["marmousi"]["sha256"]),
                )
                completed.append(str(committed_path.resolve()))
        except Exception:
            for writer, _ in open_writers:
                try:
                    writer.close_partial()
                except Exception:
                    pass
            raise
    return {
        "status": "COMPLETE",
        "worker_rank": int(worker_rank),
        "num_workers": int(num_workers),
        "completed_shards": completed,
        "reused_shards": reused,
    }
