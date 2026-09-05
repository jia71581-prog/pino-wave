#!/usr/bin/env python3
"""Sealed fixed-frequency Marmousi source-position predictions.

The prediction stage is deliberately input-only: it reads the preregistered
velocity slices and coordinate axes, synthesizes every source at the same
frequency, and seals model outputs before any reference wavefield is opened or
generated.  Reference generation and scoring are separate stages so that a
failed or surprising case cannot influence case selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fno_acoustic.data_generation.source import bilinear_point_source
from fno_acoustic.data_generation.config import (
    boundaries_from_config,
    grid_from_config,
    load_config,
    time_from_config,
)
from fno_acoustic.data_generation.pipeline_lwc84 import validated_lwc84_time_plan
from fno_acoustic.data_generation.restriction import restrict_nodal_2x
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from fno_acoustic.data_generation.velocity_models_lwc84 import load_marmousi_crop
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.confirmatory_metrics import complete_transient_metrics
from saved_time_phase_operator_v4.evaluation import sha256_file, time_axis_sha256
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import _load_context, _load_parent_model


PROTOCOL_SCHEMA = "marmousi_fixed_frequency_source_position_control_v1"
PREDICTION_SCHEMA = "sealed_marmousi_fixed_frequency_source_position_predictions_v1"
REFERENCE_SCHEMA = "sealed_marmousi_fixed_frequency_source_position_references_v1"
SCORE_SCHEMA = "marmousi_fixed_frequency_source_position_scores_v1"


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected source-position protocol schema")
    fixed = payload["fixed_source_parameters"]
    f0 = float(fixed["source_f0_hz"])
    t0 = float(fixed["source_t0_s"])
    if f0 <= 0.0 or not np.isclose(t0, 1.5 / f0, rtol=0.0, atol=1.0e-15):
        raise ValueError("the fixed source onset no longer equals 1.5/f0")
    if float(fixed["source_amplitude"]) != 1.0:
        raise ValueError("the locked source amplitude changed")
    scope = payload.get("generalization_scope", {})
    if (
        scope.get("varied_variable") != "source position (x_m, z_m) only"
        or scope.get("source_frequency_generalization_in_scope") is not False
        or scope.get("frequency_sweep_permitted") is not False
    ):
        raise ValueError("the protocol no longer isolates source-position generalization")
    slices = payload["velocity_panel"]["slices"]
    positions = payload["positions_m"]
    if len(slices) != 30 or len(positions) != 8:
        raise ValueError("the locked protocol must contain 30 slices and eight positions")
    ranks = [int(row["rank"]) for row in slices]
    if ranks != list(range(1, 31)):
        raise ValueError("the locked protocol must retain the complete ranked slice census")
    qualitative = [row for row in slices if row.get("qualitative_main_figure") is True]
    if len(qualitative) != 1 or int(qualitative[0]["rank"]) != 15:
        raise ValueError("the qualitative source-position slice is no longer rank 15")
    case_ids = [str(row["case_id"]) for row in positions]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("source-position case identifiers must be unique")
    if int(payload["evaluation"]["record_count"]) != len(slices) * len(positions):
        raise ValueError("locked record count does not match slices times positions")
    return payload


def locked_cases(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the immutable slice-position Cartesian product."""

    fixed = protocol["fixed_source_parameters"]
    rows: list[dict[str, Any]] = []
    for velocity_slice in protocol["velocity_panel"]["slices"]:
        for position in protocol["positions_m"]:
            rows.append(
                {
                    "record_id": f"marm_r{int(velocity_slice['rank']):02d}_{position['case_id']}",
                    "slice_rank": int(velocity_slice["rank"]),
                    "source_sample_id": str(velocity_slice["source_sample_id"]),
                    "source_group_id": str(velocity_slice["source_group_id"]),
                    "case_id": str(position["case_id"]),
                    "role": str(position["role"]),
                    "nearest_train_position_distance_m": float(
                        position["nearest_train_position_distance_m"]
                    ),
                    "nearest_train_position_distance_domain_diagonal": float(
                        position["nearest_train_position_distance_domain_diagonal"]
                    ),
                    "source_parameters": [
                        float(position["x_m"]),
                        float(position["z_m"]),
                        float(fixed["source_f0_hz"]),
                        float(fixed["source_t0_s"]),
                        float(fixed["source_amplitude"]),
                    ],
                }
            )
    if len(rows) != int(protocol["evaluation"]["record_count"]):
        raise AssertionError("protocol expansion changed record count")
    return rows


def _load_config_context(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    base, manifest, parent_identity = _load_context(config)
    return config, base, manifest, parent_identity


def _manifest_record_by_sample_id(manifest, sample_id: str):
    matches = [record for record in manifest.records if record.sample_id == sample_id]
    if len(matches) != 1:
        raise ValueError(f"locked sample ID has {len(matches)} manifest matches: {sample_id}")
    record = matches[0]
    if record.split != "validation" or record.medium_type != "marmousi":
        raise ValueError("locked velocity slice is not validation Marmousi")
    return record


def load_locked_inputs(source_h5: Path, manifest, protocol: Mapping[str, Any]):
    """Read velocity and axes only; the HDF5 wavefield dataset is untouched."""

    expected_path = Path(protocol["velocity_panel"]["source_h5"]).resolve()
    if source_h5.resolve() != expected_path:
        raise ValueError("model data source and locked protocol HDF5 disagree")
    inputs: dict[int, dict[str, Any]] = {}
    with h5py.File(source_h5, "r", swmr=True) as handle:
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        x_m = np.asarray(handle["x_m"][:], dtype=np.float64)
        z_m = np.asarray(handle["z_m"][:], dtype=np.float64)
        if time_s.shape != (401,) or x_m.shape != (201,) or z_m.shape != (201,):
            raise ValueError("locked source-position geometry changed")
        if not np.allclose(time_s[[80, 240, 400]], [0.2, 0.6, 1.0]):
            raise ValueError("locked snapshot times changed")
        for locked in protocol["velocity_panel"]["slices"]:
            record = _manifest_record_by_sample_id(manifest, str(locked["source_sample_id"]))
            velocity = np.asarray(handle["velocity_mps"][record.source_index], dtype=np.float32)
            actual_hash = array_sha256(velocity)
            if actual_hash != str(locked["saved_velocity_bytes_sha256"]):
                raise ValueError(f"locked velocity hash mismatch at rank {locked['rank']}")
            inputs[int(locked["rank"])] = {
                "velocity_mps": velocity,
                "velocity_sha256": actual_hash,
                "source_index": int(record.source_index),
                "sample_id": str(record.sample_id),
                "group_id": str(record.group_id),
            }
    return inputs, time_s, x_m, z_m


def load_verified_prediction_manifest(
    path: Path, protocol_path: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify the immutable prediction seal before any reference generation."""

    payload = json.loads(path.read_text(encoding="utf8"))
    if payload.get("schema") != PREDICTION_SCHEMA or payload.get("status") != "complete":
        raise ValueError("prediction manifest is not a complete position-control seal")
    if payload.get("truth_wavefield_access") is not False:
        raise ValueError("prediction stage did not preserve the input-only contract")
    if payload.get("source_frequency_varied") is not False:
        raise ValueError("prediction seal varies source frequency")
    if payload.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("prediction seal is bound to a different protocol")
    fixed_f0 = float(payload["fixed_source_parameters"]["source_f0_hz"])
    records = payload.get("records", [])
    paths: dict[str, Path] = {}
    for row in records:
        if float(row["source_parameters"][2]) != fixed_f0:
            raise ValueError("a sealed prediction has a nonfixed source frequency")
        prediction_path = Path(row["prediction_path"])
        if sha256_file(prediction_path) != row["prediction_sha256"]:
            raise ValueError(f"prediction seal mismatch: {row['record_id']}")
        paths[str(row["record_id"])] = prediction_path
    protocol = load_protocol(protocol_path)
    expected = {row["record_id"] for row in locked_cases(protocol)}
    if set(paths) != expected or len(records) != len(expected):
        raise ValueError("prediction seal does not exactly cover the locked cases")
    for row in payload.get("inputs", []):
        input_path = Path(row["input_path"])
        if sha256_file(input_path) != row["input_file_sha256"]:
            raise ValueError(f"input seal mismatch at slice rank {row['slice_rank']}")
    return payload, paths


def _validate_reference_binding(
    protocol: Mapping[str, Any], config: Mapping[str, Any], source_h5: Path
) -> None:
    reference = protocol["reference_solver"]
    if str(config["config_sha256"]) != str(reference["canonical_config_sha256"]):
        raise ValueError("frozen LWC configuration hash disagrees with the protocol")
    with h5py.File(source_h5, "r", swmr=True) as handle:
        expected = {
            "config_sha256": reference["canonical_config_sha256"],
            "manifest_sha256": reference["dataset_manifest_sha256"],
            "marmousi_sha256": reference["marmousi_original_sha256"],
            "git_commit": reference["generator_git_commit"],
        }
        for name, value in expected.items():
            actual = handle.attrs[name]
            if isinstance(actual, bytes):
                actual = actual.decode()
            if str(actual) != str(value):
                raise ValueError(f"HDF5 {name} disagrees with the reference protocol")
        numeric = {
            "dt_used_s": float(config["time"]["dt_used_s"]),
            "dt_requested_s": float(config["time"]["dt_requested_s"]),
            "snapshot_stride": int(config["time"]["snapshot_stride"]),
        }
        for name, value in numeric.items():
            if not np.isclose(float(handle.attrs[name]), float(value), rtol=0.0, atol=1.0e-15):
                raise ValueError(f"HDF5 {name} disagrees with the frozen reference config")


def reconstruct_locked_fine_velocities(
    protocol: Mapping[str, Any], config: Mapping[str, Any], source_h5: Path
) -> dict[int, np.ndarray]:
    """Rebuild each 5 m Marmousi crop and prove its 10 m input is exact."""

    reference = protocol["reference_solver"]
    prepared_path = Path(reference["local_prepared_marmousi_npy"])
    if sha256_file(prepared_path) != reference["local_prepared_marmousi_npy_sha256"]:
        raise ValueError("prepared Marmousi file hash mismatch")
    grid = grid_from_config(dict(config))
    fine: dict[int, np.ndarray] = {}
    with h5py.File(source_h5, "r", swmr=True) as handle:
        sample_ids = np.asarray(handle["sample_id"].asstr()[:], dtype=str)
        for locked in protocol["velocity_panel"]["slices"]:
            matches = np.flatnonzero(sample_ids == str(locked["source_sample_id"]))
            if len(matches) != 1:
                raise ValueError("locked source sample ID is not unique in HDF5")
            crop, _ = load_marmousi_crop(
                prepared_path,
                grid=grid,
                source_dx_m=float(config["marmousi"]["source_dx_m"]),
                source_dz_m=float(config["marmousi"]["source_dz_m"]),
                source_unit="m/s",
                crop_x0_m=float(locked["crop_x0_m"]),
                crop_z0_m=float(locked["crop_z0_m"]),
                interpolation=str(config["marmousi"]["interpolation"]),
            )
            saved = restrict_nodal_2x(crop).astype(np.float32, copy=False)
            stored = np.asarray(handle["velocity_mps"][int(matches[0])], dtype=np.float32)
            if not np.array_equal(saved, stored):
                raise ValueError(f"fine-grid reconstruction mismatch at rank {locked['rank']}")
            if array_sha256(saved) != str(locked["saved_velocity_bytes_sha256"]):
                raise ValueError("reconstructed velocity hash differs from the protocol")
            fine[int(locked["rank"])] = crop
    return fine


def _reference_solver(config: Mapping[str, Any], device: torch.device) -> LWC84CPMLSolver:
    plan = validated_lwc84_time_plan(dict(config))
    return LWC84CPMLSolver(
        grid=grid_from_config(dict(config)),
        boundaries=boundaries_from_config(dict(config)),
        dt_s=plan.dt_used_s,
        output_times_s=time_from_config(dict(config)).t_s,
        c_ref_mps=6750.0,
        device=device,
        dtype=torch.float32,
        kappa_max=float(config["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(config["boundaries"]["minimum_frequency_hz"]),
        output_restriction_factor=2,
    )


def reproduce_stored_reference(
    *,
    solver: LWC84CPMLSolver,
    source_h5: Path,
    fine_velocity: np.ndarray,
    sample_id: str,
    tolerance: float,
) -> dict[str, Any]:
    """Recompute one original record before accepting newly generated truth."""

    with h5py.File(source_h5, "r", swmr=True) as handle:
        ids = np.asarray(handle["sample_id"].asstr()[:], dtype=str)
        matches = np.flatnonzero(ids == sample_id)
        if len(matches) != 1:
            raise ValueError("reproduction sample ID is not unique")
        index = int(matches[0])
        params = {
            "source_x_m": float(handle["source_x_m"][index]),
            "source_z_m": float(handle["source_z_m"][index]),
            "source_f0_hz": float(handle["source_f0_hz"][index]),
            "source_t0_s": float(handle["source_t0_s"][index]),
            "source_amplitude": float(handle["source_amplitude"][index]),
        }
        stored = np.asarray(handle["wavefield"][index], dtype=np.float32)
    result = solver.simulate(fine_velocity, **params)
    actual = result.wavefield[0]
    difference = np.linalg.norm((actual - stored).astype(np.float64).ravel())
    denominator = max(np.linalg.norm(stored.astype(np.float64).ravel()), 1.0e-8)
    relative_l2 = float(difference / denominator)
    if relative_l2 > float(tolerance):
        raise RuntimeError(
            f"LWC stored-reference reproduction failed: {relative_l2} > {tolerance}"
        )
    return {
        "sample_id": sample_id,
        "relative_l2": relative_l2,
        "required_max_relative_l2": float(tolerance),
        "passed": True,
        "original_source_parameters": [params[name] for name in (
            "source_x_m", "source_z_m", "source_f0_hz", "source_t0_s", "source_amplitude"
        )],
    }


def generate_references(
    *,
    prediction_manifest_path: Path,
    protocol_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    reproduction_tolerance: float,
) -> dict[str, Any]:
    """Generate fixed-frequency truth only after verifying prediction seals."""

    reference_manifest_path = output_dir / "reference_manifest.json"
    if reference_manifest_path.exists():
        raise FileExistsError("reference seal already exists; refusing overwrite")
    if batch_size <= 0:
        raise ValueError("reference batch size must be positive")
    protocol = load_protocol(protocol_path)
    prediction_manifest, _ = load_verified_prediction_manifest(
        prediction_manifest_path, protocol_path
    )
    reference = protocol["reference_solver"]
    frozen_config_path = Path(reference["frozen_config"])
    config = load_config(frozen_config_path)
    source_h5 = Path(protocol["velocity_panel"]["source_h5"])
    _validate_reference_binding(protocol, config, source_h5)
    fine = reconstruct_locked_fine_velocities(protocol, config, source_h5)
    solver = _reference_solver(config, device)
    qualitative = next(
        row for row in protocol["velocity_panel"]["slices"]
        if row.get("qualitative_main_figure") is True
    )
    reproduction = reproduce_stored_reference(
        solver=solver,
        source_h5=source_h5,
        fine_velocity=fine[int(qualitative["rank"])],
        sample_id=str(qualitative["source_sample_id"]),
        tolerance=float(reproduction_tolerance),
    )

    cases = locked_cases(protocol)
    fixed_f0 = float(protocol["fixed_source_parameters"]["source_f0_hz"])
    entries: list[dict[str, Any]] = []
    for rank in sorted(fine):
        rank_cases = [row for row in cases if row["slice_rank"] == rank]
        for start in range(0, len(rank_cases), int(batch_size)):
            block = rank_cases[start : start + int(batch_size)]
            velocity_batch = np.repeat(fine[rank][None], len(block), axis=0)
            parameters = np.asarray([row["source_parameters"] for row in block], dtype=np.float64)
            if not np.all(parameters[:, 2] == fixed_f0):
                raise AssertionError("reference block varies source frequency")
            result = solver.simulate(
                velocity_batch,
                source_x_m=parameters[:, 0],
                source_z_m=parameters[:, 1],
                source_f0_hz=parameters[:, 2],
                source_t0_s=parameters[:, 3],
                source_amplitude=parameters[:, 4],
            )
            for local_index, row in enumerate(block):
                target = result.wavefield[local_index]
                if target.shape != (401, 201, 201) or not np.isfinite(target).all():
                    raise RuntimeError(f"invalid generated reference for {row['record_id']}")
                path = output_dir / "references" / f"{row['record_id']}.npz"
                _atomic_npz(
                    path,
                    target_tzx=target.astype(np.float32, copy=False),
                    source_map_zx=result.source_map_saved[local_index].astype(np.float32),
                    source_parameters=parameters[local_index],
                )
                entries.append(
                    {
                        **row,
                        "reference_path": str(path.resolve()),
                        "reference_sha256": sha256_file(path),
                        "solver_metrics": result.metrics[local_index],
                    }
                )
    entries.sort(key=lambda row: (row["slice_rank"], row["case_id"]))
    payload = {
        "schema": REFERENCE_SCHEMA,
        "status": "complete",
        "prediction_seal_verified_before_truth_generation": True,
        "prediction_manifest": str(prediction_manifest_path.resolve()),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "prediction_checkpoint_sha256": prediction_manifest["checkpoint_sha256"],
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "frozen_config": str(frozen_config_path.resolve()),
        "canonical_config_sha256": config["config_sha256"],
        "fixed_source_frequency_hz": fixed_f0,
        "source_frequency_varied": False,
        "solver_reproduction": reproduction,
        "records": entries,
    }
    if len(entries) != int(protocol["evaluation"]["record_count"]):
        raise AssertionError("reference record count changed")
    _atomic_json(payload, reference_manifest_path)
    return payload


def load_verified_reference_manifest(
    path: Path, protocol_path: Path, prediction_manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if payload.get("schema") != REFERENCE_SCHEMA or payload.get("status") != "complete":
        raise ValueError("reference manifest is not complete")
    if payload.get("prediction_seal_verified_before_truth_generation") is not True:
        raise ValueError("reference generation did not verify the prediction seal first")
    if payload.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("reference manifest is bound to a different protocol")
    origin_prediction_path = Path(payload.get("prediction_manifest", ""))
    if (
        not origin_prediction_path.is_file()
        or payload.get("prediction_manifest_sha256")
        != sha256_file(origin_prediction_path)
    ):
        raise ValueError("the prediction-before-truth provenance seal is unavailable or changed")
    candidate_prediction = json.loads(prediction_manifest_path.read_text(encoding="utf8"))
    if candidate_prediction.get("protocol_sha256") != payload.get("protocol_sha256"):
        raise ValueError("candidate prediction and shared reference use different protocols")
    if payload.get("source_frequency_varied") is not False:
        raise ValueError("reference source frequency was varied")
    if payload.get("solver_reproduction", {}).get("passed") is not True:
        raise ValueError("reference solver reproduction gate did not pass")
    paths: dict[str, Path] = {}
    for row in payload.get("records", []):
        reference_path = Path(row["reference_path"])
        if sha256_file(reference_path) != row["reference_sha256"]:
            raise ValueError(f"reference seal mismatch: {row['record_id']}")
        paths[str(row["record_id"])] = reference_path
    expected = {row["record_id"] for row in locked_cases(load_protocol(protocol_path))}
    if set(paths) != expected:
        raise ValueError("reference seal does not exactly cover the locked cases")
    return payload, paths


def _summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(rows[0]["metrics"])
    if any(sorted(row["metrics"]) != names for row in rows):
        raise ValueError("score rows have inconsistent metric keys")
    result: dict[str, Any] = {"record_count": len(rows)}
    for name in names:
        values = np.asarray([row["metrics"][name] for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def cluster_bootstrap_by_slice(
    rows: list[dict[str, Any]], *, repetitions: int = 10_000, seed: int = 20260813
) -> dict[str, Any]:
    """Bootstrap velocity-slice means, never individual source-position records."""

    if repetitions < 100:
        raise ValueError("cluster bootstrap requires at least 100 repetitions")
    ranks = sorted({int(row["slice_rank"]) for row in rows})
    if len(ranks) < 2:
        raise ValueError("cluster bootstrap requires at least two velocity slices")
    names = sorted(rows[0]["metrics"])
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, len(ranks), size=(int(repetitions), len(ranks)))
    result: dict[str, Any] = {
        "independent_unit": "velocity_slice",
        "independent_velocity_slice_count": len(ranks),
        "source_positions_per_slice": sorted(
            {sum(int(row["slice_rank"]) == rank for row in rows) for rank in ranks}
        ),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "interval": "percentile 95% confidence interval over velocity-slice means",
        "metrics": {},
    }
    for name in names:
        slice_means = np.asarray(
            [
                np.mean(
                    [float(row["metrics"][name]) for row in rows if int(row["slice_rank"]) == rank]
                )
                for rank in ranks
            ],
            dtype=np.float64,
        )
        bootstrap = slice_means[draw].mean(axis=1)
        result["metrics"][name] = {
            "mean_of_slice_means": float(slice_means.mean()),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "slice_means": [float(value) for value in slice_means],
        }
    return result


def _render_position_figures(
    *,
    protocol: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    predictions: Mapping[str, Path],
    references: Mapping[str, Path],
    output_dir: Path,
) -> list[dict[str, str]]:
    """Render the preregistered main-slice source map, snapshots, and gathers."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 6.5,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    main_slice = next(
        row for row in protocol["velocity_panel"]["slices"]
        if row.get("qualitative_main_figure") is True
    )
    rank = int(main_slice["rank"])
    main_rows = [row for row in locked_cases(protocol) if int(row["slice_rank"]) == rank]
    position_order = {row["case_id"]: i for i, row in enumerate(protocol["positions_m"])}
    main_rows.sort(key=lambda row: position_order[row["case_id"]])
    input_row = next(row for row in prediction_manifest["inputs"] if int(row["slice_rank"]) == rank)
    with np.load(input_row["input_path"], allow_pickle=False) as archive:
        velocity = np.asarray(archive["velocity_mps"], dtype=np.float32)
        x_m = np.asarray(archive["x_m"], dtype=np.float64)
        z_m = np.asarray(archive["z_m"], dtype=np.float64)
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
    extent = [x_m[0] / 1000.0, x_m[-1] / 1000.0, z_m[-1] / 1000.0, z_m[0] / 1000.0]
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, str]] = []

    fig, axis = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    image = axis.imshow(velocity, cmap="viridis", extent=extent, origin="upper", aspect="equal")
    for index, row in enumerate(main_rows, 1):
        source = row["source_parameters"]
        color = "#0072B2" if row["role"] == "interpolation" else "#D55E00"
        axis.scatter(source[0] / 1000.0, source[1] / 1000.0, s=95, marker="*", color=color)
        axis.text(source[0] / 1000.0 + 0.025, source[1] / 1000.0 + 0.025, f"S{index}")
    axis.set(
        xlabel="x (km)",
        ylabel="z (km)",
        title="Held-out Marmousi slice; all sources fixed at 19 Hz",
    )
    fig.colorbar(image, ax=axis, label="P-wave velocity (m s$^{-1}$)")
    path = figure_dir / "marmousi_fixed19hz_source_positions.pdf"
    fig.savefig(path)
    plt.close(fig)
    artifacts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})

    snapshot_indices = [int(value) for value in protocol["evaluation"]["snapshot_indices"]]
    for time_index in snapshot_indices:
        snapshot_panels: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for row in main_rows:
            with np.load(predictions[row["record_id"]], allow_pickle=False) as archive:
                prediction = np.asarray(archive["prediction_tzx"][time_index], dtype=np.float32)
            with np.load(references[row["record_id"]], allow_pickle=False) as archive:
                target = np.asarray(archive["target_tzx"][time_index], dtype=np.float32)
            snapshot_panels.append((target, prediction, prediction - target))
        pressure_limit = max(
            1.0e-20,
            *(float(np.abs(panel).max()) for pair in snapshot_panels for panel in pair[:2]),
        )
        error_limit = max(
            1.0e-20, *(float(np.abs(pair[2]).max()) for pair in snapshot_panels)
        )
        # Author at final IEEE two-column width.  Shared colour bars make the common
        # amplitude scaling visible rather than leaving it only in the caption.
        fig, axes = plt.subplots(3, len(main_rows), figsize=(7.15, 3.15))
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.86,
                            wspace=0.08, hspace=0.10)
        pressure_image = None
        error_image = None
        for column, (row, (target, prediction, error)) in enumerate(zip(main_rows, snapshot_panels)):
            for panel_index, (panel, label, limit) in enumerate(
                ((target, "reference", pressure_limit), (prediction, "prediction", pressure_limit), (error, "error", error_limit))
            ):
                rendered = axes[panel_index, column].imshow(
                    panel,
                    cmap="seismic",
                    vmin=-limit,
                    vmax=limit,
                    extent=extent,
                    origin="upper",
                    aspect="equal",
                )
                if panel_index < 2:
                    pressure_image = rendered
                else:
                    error_image = rendered
                if column == 0:
                    axes[panel_index, column].set_ylabel(f"{label}\nz (km)", fontsize=6.2)
                else:
                    axes[panel_index, column].set_yticks([])
                if panel_index == 2:
                    axes[panel_index, column].set_xlabel("x (km)", fontsize=6.2)
                else:
                    axes[panel_index, column].set_xticks([])
                axes[panel_index, column].tick_params(labelsize=5.5, length=2)
            position_role = "in" if row["role"] == "interpolation" else "out"
            axes[0, column].set_title(
                f"S{column + 1} ({position_role})\n"
                f"({row['source_parameters'][0] / 1000.0:.2f}, "
                f"{row['source_parameters'][1] / 1000.0:.2f}) km",
                fontsize=5.6,
                pad=2,
            )
        assert pressure_image is not None and error_image is not None
        pressure_cax = fig.add_axes([0.16, 0.055, 0.29, 0.025])
        error_cax = fig.add_axes([0.59, 0.055, 0.29, 0.025])
        pressure_bar = fig.colorbar(pressure_image, cax=pressure_cax, orientation="horizontal")
        error_bar = fig.colorbar(error_image, cax=error_cax, orientation="horizontal")
        # Compact symbols beside the bars avoid colliding with the eight x labels;
        # the caption defines the shared wavefield and signed-error scales.
        fig.text(0.145, 0.066, "$p$", ha="right", va="center", fontsize=6.2)
        fig.text(0.575, 0.066, "$\\Delta p$", ha="right", va="center", fontsize=6.2)
        pressure_bar.ax.tick_params(labelsize=5.2, length=2)
        error_bar.ax.tick_params(labelsize=5.2, length=2)
        fig.suptitle(
            f"Fixed 19-Hz source-position control at $t={time_s[time_index]:.1f}$ s",
            fontsize=7.2, y=0.965,
        )
        path = figure_dir / f"marmousi_fixed19hz_snapshots_t{time_index:03d}.pdf"
        fig.savefig(path)
        plt.close(fig)
        artifacts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})

    receiver = protocol["evaluation"]["receiver_x_indices"]
    receiver_x = np.arange(
        int(receiver["start"]), int(receiver["stop_inclusive"]) + 1, int(receiver["step"])
    )
    receiver_z = int(protocol["evaluation"]["receiver_z_index"])
    gather_panels: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for row in main_rows:
        with np.load(predictions[row["record_id"]], allow_pickle=False) as archive:
            prediction = np.asarray(archive["prediction_tzx"][:, receiver_z, receiver_x].T, dtype=np.float32)
        with np.load(references[row["record_id"]], allow_pickle=False) as archive:
            target = np.asarray(archive["target_tzx"][:, receiver_z, receiver_x].T, dtype=np.float32)
        gather_panels.append((target, prediction, prediction - target))
    pressure_limit = max(
        1.0e-20, *(float(np.abs(panel).max()) for pair in gather_panels for panel in pair[:2])
    )
    error_limit = max(1.0e-20, *(float(np.abs(pair[2]).max()) for pair in gather_panels))
    fig, axes = plt.subplots(3, len(main_rows), figsize=(7.15, 3.15))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.86,
                        wspace=0.08, hspace=0.10)
    gather_extent = [time_s[0], time_s[-1], x_m[receiver_x[-1]] / 1000.0, x_m[receiver_x[0]] / 1000.0]
    pressure_image = None
    error_image = None
    for column, (row, (target, prediction, error)) in enumerate(zip(main_rows, gather_panels)):
        for panel_index, (panel, label, limit) in enumerate(
            ((target, "reference", pressure_limit), (prediction, "prediction", pressure_limit), (error, "error", error_limit))
        ):
            rendered = axes[panel_index, column].imshow(
                panel,
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
                extent=gather_extent,
                origin="upper",
                aspect="auto",
            )
            if panel_index < 2:
                pressure_image = rendered
            else:
                error_image = rendered
            if column == 0:
                axes[panel_index, column].set_ylabel(f"{label}\nreceiver x (km)", fontsize=6.2)
            else:
                axes[panel_index, column].set_yticks([])
            if panel_index == 2:
                axes[panel_index, column].set_xticks([0.5], ["0.5"])
            else:
                axes[panel_index, column].set_xticks([])
            axes[panel_index, column].tick_params(labelsize=5.5, length=2)
        position_role = "in" if row["role"] == "interpolation" else "out"
        axes[0, column].set_title(f"S{column + 1} ({position_role})", fontsize=5.8, pad=2)
    assert pressure_image is not None and error_image is not None
    pressure_cax = fig.add_axes([0.16, 0.055, 0.29, 0.025])
    error_cax = fig.add_axes([0.59, 0.055, 0.29, 0.025])
    pressure_bar = fig.colorbar(pressure_image, cax=pressure_cax, orientation="horizontal")
    error_bar = fig.colorbar(error_image, cax=error_cax, orientation="horizontal")
    pressure_bar.ax.tick_params(labelsize=5.2, length=2)
    error_bar.ax.tick_params(labelsize=5.2, length=2)
    fig.text(0.145, 0.066, "$p$", ha="right", va="center", fontsize=6.2)
    fig.text(0.575, 0.066, "$\\Delta p$", ha="right", va="center", fontsize=6.2)
    fig.text(0.515, 0.125, "time (0--1 s)", ha="center", va="center", fontsize=6.2)
    fig.suptitle("Shallow receiver-line gathers ($z=40$ m), fixed 19 Hz",
                 fontsize=7.2, y=0.965)
    path = figure_dir / "marmousi_fixed19hz_receiver_gathers.pdf"
    fig.savefig(path)
    plt.close(fig)
    artifacts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    return artifacts


def score(
    *,
    prediction_manifest_path: Path,
    reference_manifest_path: Path,
    protocol_path: Path,
    output_dir: Path,
    compute_komega: bool,
) -> dict[str, Any]:
    """Verify both seals and score every locked source position."""

    score_path = output_dir / "score_summary.json"
    if score_path.exists():
        raise FileExistsError("score summary already exists; refusing overwrite")
    protocol = load_protocol(protocol_path)
    prediction_manifest, predictions = load_verified_prediction_manifest(
        prediction_manifest_path, protocol_path
    )
    reference_manifest, references = load_verified_reference_manifest(
        reference_manifest_path, protocol_path, prediction_manifest_path
    )
    evaluation = protocol["evaluation"]
    receiver = evaluation["receiver_x_indices"]
    receiver_x_indices = list(
        range(int(receiver["start"]), int(receiver["stop_inclusive"]) + 1, int(receiver["step"]))
    )
    time_s = np.linspace(
        float(protocol["reference_solver"]["time_range_s"][0]),
        float(protocol["reference_solver"]["time_range_s"][1]),
        int(protocol["reference_solver"]["saved_times"]),
        dtype=np.float32,
    )
    onset_index = int(np.searchsorted(
        time_s, float(protocol["fixed_source_parameters"]["source_t0_s"]), side="left"
    ))
    rows: list[dict[str, Any]] = []
    case_lookup = {row["record_id"]: row for row in locked_cases(protocol)}
    for record_id in sorted(predictions):
        with np.load(predictions[record_id], allow_pickle=False) as archive:
            prediction = np.asarray(archive["prediction_tzx"], dtype=np.float32)
            prediction_source = np.asarray(archive["source_parameters"], dtype=np.float64)
        with np.load(references[record_id], allow_pickle=False) as archive:
            target = np.asarray(archive["target_tzx"], dtype=np.float32)
            target_source = np.asarray(archive["source_parameters"], dtype=np.float64)
        expected_source = np.asarray(case_lookup[record_id]["source_parameters"], dtype=np.float64)
        if not np.array_equal(prediction_source, expected_source) or not np.array_equal(
            target_source, expected_source
        ):
            raise ValueError(f"source parameters disagree for {record_id}")
        metrics = complete_transient_metrics(
            torch.from_numpy(prediction),
            torch.from_numpy(target),
            torch.from_numpy(time_s),
            onset_index=onset_index,
            receiver_z_index=int(evaluation["receiver_z_index"]),
            receiver_x_indices=receiver_x_indices,
            compute_komega=bool(compute_komega),
        )
        rows.append({**case_lookup[record_id], "metrics": metrics})
    by_role = {
        role: _summarize_metric_rows([row for row in rows if row["role"] == role])
        for role in sorted({row["role"] for row in rows})
    }
    by_slice = {
        str(rank): _summarize_metric_rows([row for row in rows if row["slice_rank"] == rank])
        for rank in sorted({row["slice_rank"] for row in rows})
    }
    cluster_bootstrap = cluster_bootstrap_by_slice(rows)
    figure_artifacts = _render_position_figures(
        protocol=protocol,
        prediction_manifest=prediction_manifest,
        predictions=predictions,
        references=references,
        output_dir=output_dir,
    )
    payload = {
        "schema": SCORE_SCHEMA,
        "status": "complete",
        "source_generalization_variable": "position_only",
        "fixed_source_frequency_hz": float(protocol["fixed_source_parameters"]["source_f0_hz"]),
        "frequency_generalization_claim_permitted": False,
        "protocol_sha256": sha256_file(protocol_path),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "shared_reference_origin_prediction_manifest_sha256": reference_manifest[
            "prediction_manifest_sha256"
        ],
        "checkpoint_sha256": prediction_manifest["checkpoint_sha256"],
        "compute_komega": bool(compute_komega),
        "overall": _summarize_metric_rows(rows),
        "by_position_role": by_role,
        "by_velocity_slice": by_slice,
        "cluster_bootstrap": cluster_bootstrap,
        "figure_artifacts": figure_artifacts,
        "records": rows,
    }
    _atomic_json(payload, score_path)
    return payload


@torch.inference_mode()
def predict(
    *,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_identity_path: Path,
    protocol_path: Path,
    output_dir: Path,
    device: torch.device,
    time_block: int,
) -> dict[str, Any]:
    manifest_path = output_dir / "prediction_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("prediction seal already exists; refusing overwrite")
    protocol = load_protocol(protocol_path)
    cases = locked_cases(protocol)
    config, base, manifest, parent_identity = _load_config_context(config_path)
    identity = json.loads(checkpoint_identity_path.read_text(encoding="utf8"))
    if identity.get("manifest_digest") != manifest.digest:
        raise ValueError("checkpoint identity and active manifest disagree")
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    metadata = load_checkpoint(
        checkpoint_path,
        model=model,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=str(identity["run_digest"]),
        map_location=device,
    )
    model.eval()
    normalizer = load_normalizer(base, manifest.digest)
    inputs, time_s, x_m, z_m = load_locked_inputs(
        Path(base.data.source_h5), manifest, protocol
    )
    time_tensor = torch.from_numpy(time_s.astype(np.float32)).to(device)
    x_tensor = torch.from_numpy(x_m.astype(np.float32)).to(device)
    z_tensor = torch.from_numpy(z_m.astype(np.float32)).to(device)

    input_entries: list[dict[str, Any]] = []
    prediction_entries: list[dict[str, Any]] = []
    by_rank = {rank: [row for row in cases if row["slice_rank"] == rank] for rank in inputs}
    for rank in sorted(inputs):
        item = inputs[rank]
        velocity_path = output_dir / "inputs" / f"marm_r{rank:02d}_velocity.npz"
        _atomic_npz(
            velocity_path,
            velocity_mps=item["velocity_mps"],
            time_s=time_s,
            x_m=x_m,
            z_m=z_m,
        )
        input_entries.append(
            {
                "slice_rank": rank,
                "source_index": item["source_index"],
                "source_sample_id": item["sample_id"],
                "source_group_id": item["group_id"],
                "velocity_array_sha256": item["velocity_sha256"],
                "input_path": str(velocity_path.resolve()),
                "input_file_sha256": sha256_file(velocity_path),
            }
        )
        medium = model.encode_medium(
            torch.from_numpy(item["velocity_mps"])[None, None].to(device), normalizer
        )
        for case in by_rank[rank]:
            source = torch.tensor(case["source_parameters"], dtype=torch.float32, device=device)[None]
            point = bilinear_point_source(
                float(source[0, 0]),
                float(source[0, 1]),
                nx=len(x_m),
                nz=len(z_m),
                dx_m=float(x_m[1] - x_m[0]),
                dz_m=float(z_m[1] - z_m[0]),
                centering="node",
            )
            source_map = torch.from_numpy(point.source_map)[None, None].to(device)
            prepared = model.prepare_sources(
                medium,
                source,
                source_map,
                normalizer,
                record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
            )
            normalized = model.dense_normalized(
                prepared,
                time_tensor,
                x_m=x_tensor,
                z_m=z_tensor,
                time_block=int(time_block),
            )
            prediction = (
                normalizer.decode_pressure(normalized.float(), source[:, 4])
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if prediction.shape != (401, 201, 201) or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid prediction for {case['record_id']}")
            prediction_path = output_dir / "predictions" / f"{case['record_id']}.npz"
            _atomic_npz(
                prediction_path,
                prediction_tzx=prediction,
                source_map_zx=point.source_map,
                source_parameters=np.asarray(case["source_parameters"], dtype=np.float64),
            )
            prediction_entries.append(
                {
                    **case,
                    "prediction_path": str(prediction_path.resolve()),
                    "prediction_sha256": sha256_file(prediction_path),
                }
            )

    prediction_entries.sort(key=lambda row: (row["slice_rank"], row["case_id"]))
    payload = {
        "schema": PREDICTION_SCHEMA,
        "status": "complete",
        "truth_wavefield_access": False,
        "source_frequency_varied": False,
        "input_hdf5_datasets_accessed": ["time_s", "velocity_mps", "x_m", "z_m"],
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "config": str(config_path.resolve()),
        "config_file_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(metadata.epoch),
        "checkpoint_global_step": int(metadata.global_step),
        "checkpoint_identity": str(checkpoint_identity_path.resolve()),
        "checkpoint_identity_sha256": sha256_file(checkpoint_identity_path),
        "model_config_digest": str(identity["run_digest"]),
        "manifest_digest": manifest.digest,
        "time_axis_sha256": time_axis_sha256(time_s),
        "fixed_source_parameters": protocol["fixed_source_parameters"],
        "inputs": input_entries,
        "records": prediction_entries,
    }
    if len(payload["records"]) != int(protocol["evaluation"]["record_count"]):
        raise AssertionError("prediction record count changed")
    _atomic_json(payload, manifest_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--config", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--checkpoint-identity", type=Path, required=True)
    predict_parser.add_argument("--protocol", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser.add_argument("--device", default="cuda:0")
    predict_parser.add_argument("--time-block", type=int, default=16)
    reference_parser = subparsers.add_parser("generate-reference")
    reference_parser.add_argument("--prediction-manifest", type=Path, required=True)
    reference_parser.add_argument("--protocol", type=Path, required=True)
    reference_parser.add_argument("--output-dir", type=Path, required=True)
    reference_parser.add_argument("--device", default="cuda:0")
    reference_parser.add_argument("--batch-size", type=int, default=2)
    reference_parser.add_argument("--reproduction-tolerance", type=float, default=1.0e-5)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--prediction-manifest", type=Path, required=True)
    score_parser.add_argument("--reference-manifest", type=Path, required=True)
    score_parser.add_argument("--protocol", type=Path, required=True)
    score_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser.add_argument("--skip-komega", action="store_true")
    args = parser.parse_args()
    if args.command == "predict":
        if not str(args.device).startswith("cuda"):
            raise ValueError("full prediction is GPU-only; use unit tests for CPU checks")
        payload = predict(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            checkpoint_identity_path=args.checkpoint_identity,
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            device=torch.device(args.device),
            time_block=args.time_block,
        )
    elif args.command == "generate-reference":
        if not str(args.device).startswith("cuda"):
            raise ValueError("full LWC reference generation is GPU-only")
        payload = generate_references(
            prediction_manifest_path=args.prediction_manifest,
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            device=torch.device(args.device),
            batch_size=args.batch_size,
            reproduction_tolerance=args.reproduction_tolerance,
        )
    else:
        payload = score(
            prediction_manifest_path=args.prediction_manifest,
            reference_manifest_path=args.reference_manifest,
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            compute_komega=not args.skip_komega,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "array_sha256",
    "cluster_bootstrap_by_slice",
    "generate_references",
    "load_locked_inputs",
    "load_protocol",
    "load_verified_prediction_manifest",
    "load_verified_reference_manifest",
    "locked_cases",
    "predict",
    "reconstruct_locked_fine_velocities",
    "score",
]
