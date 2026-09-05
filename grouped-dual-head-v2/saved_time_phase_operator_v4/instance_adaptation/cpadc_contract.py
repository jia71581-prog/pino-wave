"""Immutable numerical and implementation identities for CPADC.

The CPADC PDE loss is evaluated on saved 10 m / 2.5 ms fields, whereas the
registered teacher was advanced on a 5 m / 0.125 ms grid and then restricted.
This module proves the intended saved-grid CPML equivalence at launch time and
records the exact code bytes used to train and deploy a basis checkpoint.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import h5py
import yaml


DATASET_CONTRACT_SCHEMA = "cpadc_dataset_numerical_contract_v1"

_IMPLEMENTATION_FILES = (
    "saved_time_phase_operator_v4/muon.py",
    "saved_time_phase_operator_v4/data.py",
    "saved_time_phase_operator_v4/operator.py",
    "saved_time_phase_operator_v4/decoder.py",
    "saved_time_phase_operator_v4/features.py",
    "saved_time_phase_operator_v4/full_support.py",
    "saved_time_phase_operator_v4/local_field.py",
    "saved_time_phase_operator_v4/spectral.py",
    "grouped_ufno_mionet_v3/data/index.py",
    "scripts/train_causal_defect_basis.py",
    "scripts/run_causal_defect_adaptation.py",
    "scripts/calibrate_causal_defect_risk.py",
    "scripts/evaluate_v5_instance_adaptation.py",
    "scripts/merge_causal_defect_evaluations.py",
    "scripts/supervise_target10_instance_finetune.py",
    "scripts/run_v5_instance_adaptation.py",
    "scripts/train_meta_hypernet.py",
    "scripts/train_residual_iterator.py",
    "scripts/train_v5_feature_meta.py",
    "scripts/train_v5_residual_meta.py",
)

_IMPLEMENTATION_TREES = (
    "saved_time_phase_operator_v4/instance_adaptation",
)

_NUMERICAL_IMPLEMENTATION_FILES = (
    "src/fno_acoustic/data_generation/cpml.py",
    "src/fno_acoustic/data_generation/free_surface.py",
    "src/fno_acoustic/data_generation/grid.py",
    "src/fno_acoustic/data_generation/lwc84.py",
    "src/fno_acoustic/data_generation/restriction.py",
    "src/fno_acoustic/data_generation/ricker.py",
    "src/fno_acoustic/data_generation/source.py",
    "src/fno_acoustic/data_generation/stencils.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def cpadc_implementation_digests(project_root: str | Path) -> dict[str, str]:
    """Return stable SHA-256 identities for all target-affecting CPADC code."""

    root = Path(project_root).expanduser().resolve()
    result: dict[str, str] = {}
    relatives = set(_IMPLEMENTATION_FILES) | set(_NUMERICAL_IMPLEMENTATION_FILES)
    for tree in _IMPLEMENTATION_TREES:
        directory = root / tree
        if not directory.is_dir():
            raise FileNotFoundError(f"missing CPADC implementation tree: {directory}")
        relatives.update(
            str(path.relative_to(root)) for path in directory.rglob("*.py")
        )
    for relative in sorted(relatives):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing CPADC implementation file: {path}")
        result[relative] = sha256_file(path)
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"dataset {name} must be a mapping")
    return value


def _require_close(name: str, actual: object, expected: object) -> float:
    actual_value = float(actual)
    expected_value = float(expected)
    if not (
        math.isfinite(actual_value)
        and math.isfinite(expected_value)
        and math.isclose(
            actual_value, expected_value, rel_tol=1.0e-9, abs_tol=1.0e-12
        )
    ):
        raise ValueError(
            f"CPADC numerical contract mismatch for {name}: "
            f"{actual_value} != {expected_value}"
        )
    return actual_value


def _resolved_config_digest(config: Mapping[str, object]) -> str:
    payload = dict(config)
    payload.pop("config_sha256", None)
    paths = dict(payload.get("paths", {}) or {})
    for key, value in tuple(paths.items()):
        if value is not None:
            paths[key] = str(Path(str(value)).expanduser())
    payload["paths"] = paths
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_dataset_cpml_contract(
    source_h5: str | Path,
    *,
    saved_grid_cpml: Mapping[str, object],
    saved_dt_s: float,
    saved_dx_m: float,
    saved_dz_m: float,
) -> dict[str, object]:
    """Validate the saved-grid CPML against the immutable teacher contract.

    The saved-grid closure is not claimed to invert the fine states or the
    binomial restriction.  It does, however, have to preserve physical PML
    thickness, CFS coefficients, the internal time step/count, and all boundary
    conventions exactly.  Any mismatch aborts before training or evaluation.
    """

    source = Path(source_h5).expanduser().resolve()
    frozen_path = source.parent / "frozen_config.yaml"
    plan_path = source.parent / "plan_summary.json"
    for path in (source, frozen_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    frozen_raw = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    plan_raw = json.loads(plan_path.read_text(encoding="utf-8"))
    frozen = _mapping(frozen_raw, name="frozen config")
    plan = _mapping(plan_raw, name="plan summary")
    grid = _mapping(frozen.get("grid"), name="solver grid")
    storage = _mapping(frozen.get("storage_grid"), name="storage grid")
    time = _mapping(frozen.get("time"), name="time contract")
    boundary = _mapping(frozen.get("boundaries"), name="boundary contract")
    cpml = _mapping(saved_grid_cpml, name="saved-grid CPML")

    resolved_config_sha = _resolved_config_digest(frozen)
    registered_config_sha = str(frozen.get("config_sha256", ""))
    if not registered_config_sha or registered_config_sha != resolved_config_sha:
        raise ValueError("frozen dataset config SHA-256 is invalid")

    with h5py.File(source, "r", swmr=True) as handle:
        attrs = handle.attrs
        h5_config_sha = str(attrs.get("config_sha256", ""))
        h5_manifest_sha = str(attrs.get("manifest_sha256", ""))
        h5_schema = str(attrs.get("schema_version", ""))
        h5_cpml = str(attrs.get("cpml", ""))
        h5_restriction = str(attrs.get("restriction", ""))
        h5_saved_dx = float(attrs.get("saved_dx_m", attrs.get("dx_m", math.nan)))
        h5_saved_dz = float(attrs.get("saved_dz_m", attrs.get("dz_m", math.nan)))
        h5_internal_dt = float(attrs.get("dt_used_s", math.nan))
        h5_output_dt = float(attrs.get("dt_output_s", math.nan))
        h5_snapshot_stride = int(attrs.get("snapshot_stride", -1))
        h5_saved_shape = json.loads(str(attrs.get("saved_grid_shape", "[]")))
    if h5_config_sha != registered_config_sha or not h5_manifest_sha:
        raise ValueError("source HDF5 is not bound to its frozen config/manifest")

    teacher_schema = str(frozen.get("schema_version", ""))
    if (
        not teacher_schema.startswith("acoustic-lwc84-cpml-401-to-201-")
        or h5_schema != "acoustic_lwc84_401_to_201_v1"
        or str(plan.get("schema_version", "")) != teacher_schema
    ):
        raise ValueError("unsupported LWC84 teacher schema")
    if (
        str(grid.get("centering")) != "node"
        or [int(grid.get("nz", -1)), int(grid.get("nx", -1))] != [401, 401]
        or str(storage.get("centering")) != "node"
        or [int(storage.get("nz", -1)), int(storage.get("nx", -1))]
        != [201, 201]
        or list(h5_saved_shape) != [201, 201]
        or str(storage.get("restriction"))
        != "binomial5_lowpass_then_decimate2"
        or h5_restriction
        != "separable_binomial5_lowpass_then_nodal_decimate2"
    ):
        raise ValueError("CPADC requires the registered 401-to-201 nodal restriction")
    if (
        str(boundary.get("top")) != "free_surface_dirichlet"
        or any(
            str(boundary.get(side)) != "cpml"
            for side in ("left", "right", "bottom")
        )
        or not bool(boundary.get("cpml_outside_physical_domain"))
        or str(boundary.get("alpha_max_definition"))
        != "pi_times_minimum_frequency"
        or "left,right,bottom; no top CPML" not in h5_cpml
    ):
        raise ValueError("teacher is not the registered three-sided CFS-CPML")

    solver_dx = _require_close("teacher dx", grid.get("dx_m"), 5.0)
    solver_dz = _require_close("teacher dz", grid.get("dz_m"), 5.0)
    storage_dx = _require_close("saved dx", storage.get("dx_m"), saved_dx_m)
    storage_dz = _require_close("saved dz", storage.get("dz_m"), saved_dz_m)
    _require_close("HDF5 saved dx", h5_saved_dx, storage_dx)
    _require_close("HDF5 saved dz", h5_saved_dz, storage_dz)
    teacher_internal_dt = _require_close(
        "teacher internal dt", time.get("dt_used_s"), cpml.get("internal_dt_s")
    )
    _require_close("HDF5 internal dt", h5_internal_dt, teacher_internal_dt)
    output_dt = _require_close("saved dt", time.get("dt_out_s"), saved_dt_s)
    _require_close("HDF5 saved dt", h5_output_dt, output_dt)

    teacher_substeps = int(time.get("snapshot_stride", -1))
    saved_substeps = int(cpml.get("internal_substeps_per_saved_frame", -2))
    if (
        int(time.get("nt_out", -1)) != 401
        or teacher_substeps != saved_substeps
        or h5_snapshot_stride != teacher_substeps
        or teacher_substeps != 20
    ):
        raise ValueError("saved CPML internal substep contract is inconsistent")
    _require_close(
        "internal steps per saved frame",
        teacher_internal_dt * teacher_substeps,
        output_dt,
    )

    teacher_npml = int(boundary.get("npml", -1))
    saved_npml = int(cpml.get("npml", -2))
    teacher_thickness_x = teacher_npml * solver_dx
    teacher_thickness_z = teacher_npml * solver_dz
    saved_thickness_x = saved_npml * storage_dx
    saved_thickness_z = saved_npml * storage_dz
    _require_close("CPML physical x thickness", saved_thickness_x, teacher_thickness_x)
    _require_close("CPML physical z thickness", saved_thickness_z, teacher_thickness_z)
    if teacher_npml != 40 or saved_npml != 20:
        raise ValueError("CPADC CPML must preserve 40x5 m as 20x10 m")

    _require_close(
        "CPML target reflection",
        cpml.get("target_reflection"),
        boundary.get("cpml_target_reflection"),
    )
    if int(cpml.get("polynomial_order", -1)) != int(
        boundary.get("cpml_polynomial_order", -2)
    ):
        raise ValueError("CPML polynomial order differs from the teacher")
    _require_close("CPML kappa max", cpml.get("kappa_max"), boundary.get("kappa_max"))
    _require_close(
        "CPML minimum frequency",
        cpml.get("minimum_frequency_hz"),
        boundary.get("minimum_frequency_hz"),
    )
    c_ref = _require_close(
        "CPML reference speed",
        cpml.get("c_ref_mps"),
        plan.get("global_vmax_safe_upper_bound_mps"),
    )
    if (
        str(cpml.get("memory_time_integration"))
        != "internal_substep_exact_linear_causal_vectorized"
        or str(cpml.get("exterior_initialization")) != "zero"
        or str(cpml.get("physical_state_injection"))
        != "hard_each_saved_frame"
    ):
        raise ValueError("saved-grid CPML closure is not the registered causal closure")

    return {
        "schema": DATASET_CONTRACT_SCHEMA,
        "source_h5": str(source),
        "source_h5_byte_count": int(source.stat().st_size),
        "source_h5_sha256": sha256_file(source),
        "source_manifest_sha256": h5_manifest_sha,
        "dataset_config": str(frozen_path.resolve()),
        "dataset_config_file_sha256": sha256_file(frozen_path),
        "dataset_config_sha256": registered_config_sha,
        "plan_summary": str(plan_path.resolve()),
        "plan_summary_sha256": sha256_file(plan_path),
        "teacher_schema": teacher_schema,
        "teacher_hdf5_schema": h5_schema,
        "solver_grid_shape": [401, 401],
        "saved_grid_shape": [201, 201],
        "solver_spacing_m": [solver_dz, solver_dx],
        "saved_spacing_m": [storage_dz, storage_dx],
        "saved_dt_s": output_dt,
        "internal_dt_s": teacher_internal_dt,
        "internal_substeps_per_saved_frame": teacher_substeps,
        "fine_cpml_cells": teacher_npml,
        "saved_cpml_cells": saved_npml,
        "cpml_physical_thickness_m": teacher_thickness_x,
        "cpml_reference_speed_mps": c_ref,
        "cpml_target_reflection": float(cpml["target_reflection"]),
        "cpml_polynomial_order": int(cpml["polynomial_order"]),
        "cpml_kappa_max": float(cpml["kappa_max"]),
        "cpml_minimum_frequency_hz": float(cpml["minimum_frequency_hz"]),
        "top_boundary": "free_surface_dirichlet",
        "absorbing_boundaries": ["left", "right", "bottom"],
        "restriction": "binomial5_lowpass_then_nodal_decimate2",
        "saved_source_contract": (
            "direct_saved_grid_bilinear_unit_mass_reinjection"
        ),
        "exact_fine_source_restriction": False,
        "saved_grid_closure": dict(cpml),
        "exact_fine_state_reconstruction": False,
    }


_DATASET_SPECIFIC_CONTRACT_KEYS = {
    "source_h5",
    "source_h5_byte_count",
    "source_h5_sha256",
    "source_manifest_sha256",
    "dataset_config",
    "dataset_config_file_sha256",
    "dataset_config_sha256",
    "plan_summary",
    "plan_summary_sha256",
    "teacher_schema",
}


def dataset_numerical_protocol_identity(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """Strip dataset identity while retaining every numerical/CPML invariant."""
    if str(contract.get("schema", "")) != DATASET_CONTRACT_SCHEMA:
        raise ValueError("unsupported CPADC dataset numerical contract schema")
    identity = {
        str(key): value
        for key, value in contract.items()
        if str(key) not in _DATASET_SPECIFIC_CONTRACT_KEYS
    }
    if not identity:
        raise ValueError("CPADC numerical protocol identity is empty")
    return identity


def same_dataset_numerical_protocol(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    return dataset_numerical_protocol_identity(left) == dataset_numerical_protocol_identity(
        right
    )


__all__ = [
    "DATASET_CONTRACT_SCHEMA",
    "cpadc_implementation_digests",
    "dataset_numerical_protocol_identity",
    "same_dataset_numerical_protocol",
    "sha256_file",
    "validate_dataset_cpml_contract",
]
