#!/usr/bin/env python3
"""Build and audit the train-only Transfer DG exterior-CPML E0 pilot cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from fno_acoustic.data_generation.config import (  # noqa: E402
    boundaries_from_config,
    grid_from_config,
    load_config,
)
from fno_acoustic.data_generation.cpml import build_cfs_cpml_profiles  # noqa: E402
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402
from saved_time_phase_operator_v4.exterior_cpml import (  # noqa: E402
    PROFILE_FIELDS,
    build_saved_exterior_cpml_profiles,
    contract_from_dataset_config,
    crop_physical_domain,
    profile_sampling_report,
)


DEFAULT_DATA = Path(
    "/data/jiayh/data/"
    "acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
)
DEFAULT_FROZEN = DEFAULT_DATA.parent / "frozen_config.yaml"
DEFAULT_PREREG = ROOT / "results/transfer_dg_exterior_cpml_e0_preregistration_20260902.json"
DEFAULT_CACHE = ROOT / "results/transfer_dg_exterior_cpml_e0_20260902.h5"
DEFAULT_RESULT = ROOT / "results/transfer_dg_exterior_cpml_e0_20260902.json"
DEFAULT_TERMINAL = ROOT / "results/transfer_dg_exterior_cpml_e0_terminal_20260902.json"
DEFAULT_FRAMES = (0, 40, 80, 120, 160, 200, 240, 320, 400)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.linalg.norm(difference) / max(np.linalg.norm(target), 1.0e-30))


def _maximum_absolute(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _write_cache(
    path: Path,
    *,
    metadata: dict[str, object],
    arrays: dict[str, np.ndarray],
    profiles: dict[str, np.ndarray],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite E0 cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        with h5py.File(partial, "x") as handle:
            handle.attrs["schema"] = "transfer_dg_exterior_cpml_e0_cache_v1"
            handle.attrs["status"] = "complete"
            handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            for name, value in arrays.items():
                array = np.asarray(value)
                chunks = None
                if array.ndim >= 3:
                    chunks = (1,) * (array.ndim - 2) + array.shape[-2:]
                handle.create_dataset(
                    name,
                    data=array,
                    chunks=chunks,
                    compression="gzip" if array.ndim >= 2 else None,
                    compression_opts=1 if array.ndim >= 2 else None,
                )
            group = handle.create_group("profiles")
            for name, value in profiles.items():
                group.create_dataset(name, data=np.asarray(value), compression="gzip", compression_opts=1)
            handle.flush()
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--frozen-config", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument("--source-index", type=int, default=6)
    parser.add_argument("--frame-indices", type=int, nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    for path in (args.source_h5, args.frozen_config, args.preregistration):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.cache, args.result, args.terminal):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite E0 artifact: {path}")

    config = load_config(args.frozen_config)
    contract = contract_from_dataset_config(config)
    if contract.cpml_layers != 20 or contract.top != "free_surface_dirichlet":
        raise RuntimeError("E0 requires 20 exterior CPML layers and a free top")
    if (contract.left, contract.right, contract.bottom) != ("cpml", "cpml", "cpml"):
        raise RuntimeError("E0 requires left/right/bottom CPML")

    frame_indices = np.asarray(args.frame_indices, dtype=np.int64)
    if (
        frame_indices.ndim != 1
        or len(frame_indices) < 2
        or frame_indices[0] != 0
        or np.any(np.diff(frame_indices) <= 0)
    ):
        raise ValueError("frame indices must start at zero and increase strictly")

    source_index = int(args.source_index)
    with h5py.File(args.source_h5, "r", swmr=True) as source:
        if source_index < 0 or source_index >= len(source["sample_id"]):
            raise IndexError("source index is outside the dataset")
        split = _text(source["split"][source_index])
        family = _text(source["medium_type"][source_index]).split("_")[0]
        sample_id = _text(source["sample_id"][source_index])
        if split != "train" or family != "uniform":
            raise RuntimeError("E0 pilot is restricted to one train/uniform record")
        full_time = np.asarray(source["time_s"][:], dtype=np.float64)
        if frame_indices[-1] >= len(full_time):
            raise IndexError("frame selection exceeds stored time axis")
        output_times = full_time[frame_indices]
        stored_velocity = np.asarray(source["velocity_mps"][source_index], dtype=np.float32)
        if float(np.ptp(stored_velocity)) != 0.0:
            raise RuntimeError("selected uniform record is not spatially constant")
        velocity_value = float(stored_velocity[0, 0])
        fine_velocity = np.full((401, 401), velocity_value, dtype=np.float32)
        source_parameters = {
            "source_x_m": float(source["source_x_m"][source_index]),
            "source_z_m": float(source["source_z_m"][source_index]),
            "source_f0_hz": float(source["source_f0_hz"][source_index]),
            "source_t0_s": float(source["source_t0_s"][source_index]),
            "source_amplitude": float(source["source_amplitude"][source_index]),
        }
        stored_truth = np.asarray(
            source["wavefield"][source_index, frame_indices], dtype=np.float32
        )
        sample_sha256 = _text(source["sample_sha256"][source_index])
        manifest_sha256 = str(source.attrs["manifest_sha256"])

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("E0 GPU replay requested but CUDA is unavailable")
    device = torch.device(args.device)
    solver = LWC84CPMLSolver(
        grid=grid_from_config(config),
        boundaries=boundaries_from_config(config),
        dt_s=float(config["time"]["dt_used_s"]),
        output_times_s=output_times,
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float32,
        kappa_max=float(config["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(config["boundaries"]["minimum_frequency_hz"]),
        output_restriction_factor=2,
    )
    replay = solver.simulate(
        fine_velocity,
        **source_parameters,
        capture_exterior_auxiliary=True,
    )
    auxiliary = replay.exterior_auxiliary
    if auxiliary is None:
        raise RuntimeError("solver did not return requested exterior auxiliary state")

    saved_profiles = build_saved_exterior_cpml_profiles(
        contract, c_ref_mps=6750.0, device="cpu", dtype=torch.float64
    )
    fine_profiles = build_cfs_cpml_profiles(
        grid_from_config(config),
        boundaries_from_config(config),
        dt_s=contract.internal_dt_s,
        c_ref_mps=6750.0,
        target_reflection=contract.target_reflection,
        polynomial_order=contract.polynomial_order,
        kappa_max=contract.kappa_max,
        minimum_frequency_hz=contract.minimum_frequency_hz,
        device="cpu",
        dtype=torch.float64,
    )
    profile_errors = profile_sampling_report(saved_profiles, fine_profiles)

    replay_physical = replay.wavefield[0]
    pressure_extended = auxiliary.pressure_extended_saved[0]
    pressure_crop = np.asarray(crop_physical_domain(pressure_extended, contract))
    memories = {
        "psi_x_before_step": auxiliary.psi_x_before_step[0],
        "psi_z_before_step": auxiliary.psi_z_before_step[0],
        "phi_x_before_step": auxiliary.phi_x_before_step[0],
        "phi_z_before_step": auxiliary.phi_z_before_step[0],
    }
    memory_max_by_frame = np.stack(
        [np.max(np.abs(value), axis=(-2, -1)) for value in memories.values()], axis=1
    )
    active = (
        saved_profiles.active_x.detach().cpu().numpy()
        | saved_profiles.active_z.detach().cpu().numpy()
    )
    cpml_pressure_max_by_frame = np.max(
        np.abs(pressure_extended[:, active]), axis=1
    )

    replay_relative_l2 = _relative_l2(replay_physical, stored_truth)
    crop_relative_l2 = _relative_l2(pressure_crop, replay_physical)
    top_max = _maximum_absolute(pressure_extended[:, 0, :])
    outer_max = max(
        _maximum_absolute(pressure_extended[:, -1, :]),
        _maximum_absolute(pressure_extended[:, :, 0]),
        _maximum_absolute(pressure_extended[:, :, -1]),
    )
    arrays_finite = all(
        np.isfinite(value).all()
        for value in (
            replay_physical,
            stored_truth,
            pressure_extended,
            auxiliary.velocity_extended_saved_mps,
            *memories.values(),
        )
    )
    maximum_profile_error = max(profile_errors.values())
    late_memory_max = float(memory_max_by_frame[-1].max())
    gates = {
        "geometry_221x241": list(pressure_extended.shape[-2:]) == [221, 241],
        "saved_cpml_layers_20": contract.cpml_layers == 20,
        "free_top_three_absorbing_sides": contract.top == "free_surface_dirichlet"
        and (contract.left, contract.right, contract.bottom) == ("cpml", "cpml", "cpml"),
        "profile_sampling_lte_1e_6": maximum_profile_error <= 1.0e-6,
        "physical_replay_relative_l2_lte_1e_4": replay_relative_l2 <= 1.0e-4,
        "all_arrays_finite": bool(arrays_finite),
        "top_pressure_exactly_zero": top_max == 0.0,
        "outer_pressure_exactly_zero": outer_max == 0.0,
        "late_cpml_memory_nonzero": late_memory_max > 0.0,
        "validation_not_opened": True,
        "test_id_not_opened": True,
    }
    accepted = all(gates.values())

    metadata = {
        "candidate": "transfer_dg_exterior_cpml_e0_20260902",
        "claim_scope": "train-only exterior CPML numerical contract",
        "contract": contract.as_dict(),
        "sample_id": sample_id,
        "sample_sha256": sample_sha256,
        "source_index": source_index,
        "split": split,
        "family": family,
        "source_parameters": source_parameters,
        "frame_indices": frame_indices.tolist(),
        "memory_alignment": auxiliary.memory_alignment,
        "validation_opened": False,
        "test_id_opened": False,
    }
    profile_arrays = {
        name: getattr(saved_profiles, name).detach().cpu().numpy()
        for name in PROFILE_FIELDS
    }
    _write_cache(
        args.cache,
        metadata=metadata,
        arrays={
            "frame_indices": frame_indices,
            "time_s": output_times,
            "physical_pressure_stored": stored_truth,
            "physical_pressure_replay": replay_physical,
            "pressure_extended_saved": pressure_extended,
            "velocity_extended_saved_mps": auxiliary.velocity_extended_saved_mps[0],
            "cpml_pressure_max_by_frame": cpml_pressure_max_by_frame,
            "memory_max_by_frame": memory_max_by_frame,
            **memories,
        },
        profiles=profile_arrays,
    )

    result = {
        "schema": "transfer_dg_exterior_cpml_e0_result_v1",
        "status": "accepted" if accepted else "rejected",
        "decision": "accepted_e0" if accepted else "rejected_e0",
        "claim_scope": metadata["claim_scope"],
        "contract": contract.as_dict(),
        "record": {
            "source_index": source_index,
            "sample_id": sample_id,
            "sample_sha256": sample_sha256,
            "split": split,
            "family": family,
            "source_parameters": source_parameters,
            "frame_indices": frame_indices.tolist(),
        },
        "metrics": {
            "physical_replay_relative_l2": replay_relative_l2,
            "physical_crop_from_extended_relative_l2": crop_relative_l2,
            "maximum_profile_sampling_error": maximum_profile_error,
            "profile_sampling_errors": profile_errors,
            "top_pressure_max_abs": top_max,
            "outer_pressure_max_abs": outer_max,
            "late_cpml_memory_max_abs": late_memory_max,
            "cpml_pressure_max_by_frame": cpml_pressure_max_by_frame.tolist(),
            "memory_max_by_frame": memory_max_by_frame.tolist(),
            "solver_compute_elapsed_s": replay.metrics[0]["compute_elapsed_s"],
        },
        "gates": gates,
        "bindings": {
            "source_h5": str(args.source_h5.resolve()),
            "source_h5_sha256": _sha256(args.source_h5),
            "manifest_sha256": manifest_sha256,
            "frozen_config": str(args.frozen_config.resolve()),
            "frozen_config_sha256": _sha256(args.frozen_config),
            "preregistration": str(args.preregistration.resolve()),
            "preregistration_sha256": _sha256(args.preregistration),
            "implementation": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (
                    ROOT / "saved_time_phase_operator_v4/exterior_cpml.py",
                    ROOT / "saved_time_phase_operator_v4/dg_interface.py",
                    ROOT / "src/fno_acoustic/data_generation/solver_lwc84.py",
                    Path(__file__).resolve(),
                )
            },
        },
        "cache": str(args.cache.resolve()),
        "cache_sha256": _sha256(args.cache),
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(result, args.result)
    terminal = {
        "schema": "transfer_dg_exterior_cpml_e0_terminal_v1",
        "status": "complete" if accepted else "rejected",
        "decision": result["decision"],
        "result": str(args.result.resolve()),
        "result_sha256": _sha256(args.result),
        "cache": str(args.cache.resolve()),
        "cache_sha256": result["cache_sha256"],
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(terminal, args.terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
