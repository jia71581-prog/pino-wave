#!/usr/bin/env python3
"""Run guarded CPADC adaptation, seal predictions, then evaluate future truth."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.bridge import make_onset_bridge
from saved_time_phase_operator_v4.instance_adaptation.cpadc_contract import (
    cpadc_implementation_digests,
    same_dataset_numerical_protocol,
    validate_dataset_cpml_contract,
)
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.defect_correction import (
    CPADC_SCHEMA_VERSION,
    CausalErrorBasisGenerator,
    DefectCorrectionWeights,
    causal_observation_probe_basis,
    solve_causal_defect_correction,
)
from saved_time_phase_operator_v4.instance_adaptation.forced_defect import (
    lwc84_discrete_defect,
    saved_grid_cpml_config,
)
from saved_time_phase_operator_v4.instance_adaptation.losses import (
    build_rad_physics_points,
    build_fixed_physics_points,
)
from scripts.evaluate_v5_instance_adaptation import evaluate_after_adaptation, write_report
from scripts.run_v5_instance_adaptation import (
    _load_background_provider,
    _load_parent,
    resolve_saved_time_parent_config,
    validate_external_evaluation_contract,
    _predict_parent,
    _read_future_truth,
    _write_manifest,
    build_instance_manifest,
)
from scripts.train_saved_time_v4_full_support import _load_context
from scripts.train_v5_residual_meta import _sha256
from scripts.train_v5_feature_meta import build_balanced_meta_episodes


def build_risk_calibration_manifest(
    manifest,
    *,
    basis_per_family: int,
    basis_seed: int,
    calibration_per_family: int,
    calibration_seed: int,
):
    """Select train-only calibration records disjoint from basis meta-training."""

    basis_rows = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=int(basis_per_family),
        seed=int(basis_seed),
    )
    excluded = {row.sample_id for row in basis_rows}
    allowed = {
        row.sample_id
        for row in manifest.records
        if row.split == "train" and row.sample_id not in excluded
    }
    rows = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=int(calibration_per_family),
        seed=int(calibration_seed),
        allowed_sample_ids=allowed,
    )
    if excluded.intersection(row.sample_id for row in rows):
        raise RuntimeError("risk calibration overlaps basis meta-training episodes")
    return rows


def resolve_basis_training_per_family(
    config: dict[str, object], override: int | None
) -> int:
    """Resolve the actual basis-training episode count for exclusion.

    Short pilots may override ``cpadc.per_family`` on the training CLI.  Their
    disjoint train evaluation must therefore supply the same effective count
    instead of silently excluding the production-config count.
    """

    value = (
        int((config.get("cpadc", {}) or {}).get("per_family", 0))
        if override is None
        else int(override)
    )
    if value <= 0:
        raise ValueError("basis training per-family count must be positive")
    return value


def build_complete_evaluation_manifest(manifest, *, split: str):
    """Return one complete registered non-anomaly evaluation split."""

    normalized = str(split)
    if normalized not in {"validation", "test_id"}:
        raise ValueError("complete CPADC evaluation split must be validation or test_id")
    rows = tuple(record for record in manifest.records if record.split == normalized)
    if not rows or {record.medium_type for record in rows} != set(
        ALLOWED_MEDIUM_TYPES
    ):
        raise ValueError(
            f"complete {normalized} manifest must contain all medium families"
        )
    return rows


def _nearest_rank_percentile(values: list[float], fraction: float) -> float:
    """Return a deterministic nearest-rank latency percentile."""

    if not values:
        raise ValueError("runtime percentile requires at least one value")
    quantile = float(fraction)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("runtime percentile must lie in [0,1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _synchronize_device_for_timing(device: torch.device) -> None:
    """Close the CUDA queue before reading a wall-clock deployment timer."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _promotion_gate(
    reports: list[dict[str, object]],
    gate: dict[str, object],
    *,
    all_validation: bool,
) -> dict[str, object]:
    """Aggregate the sealed blind evaluation into a reproducible promotion gate."""

    if not reports:
        raise ValueError("CPADC promotion requires at least one sealed report")
    minimum_improvement = float(gate.get("minimum_blind_mean_improvement", 0.01))
    minimum_nonworse = float(gate.get("minimum_nonworse_fraction", 0.90))
    maximum_runtime = float(gate.get("maximum_runtime_s", 5.0))
    maximum_p95_adaptation_runtime_raw = gate.get(
        "maximum_p95_adaptation_runtime_s"
    )
    maximum_p95_adaptation_runtime = (
        None
        if maximum_p95_adaptation_runtime_raw is None
        else float(maximum_p95_adaptation_runtime_raw)
    )
    minimum_speedup_raw = gate.get(
        "minimum_end_to_end_speedup_vs_traditional"
    )
    traditional_runtime_raw = gate.get(
        "traditional_solver_reference_runtime_s"
    )
    minimum_speedup = (
        None if minimum_speedup_raw is None else float(minimum_speedup_raw)
    )
    traditional_runtime = (
        None
        if traditional_runtime_raw is None
        else float(traditional_runtime_raw)
    )
    require_family_safe = bool(gate.get("require_every_family_nonworse", True))
    maximum_mean_error_raw = gate.get(
        "maximum_mean_adapted_future_relative_l2"
    )
    maximum_family_mean_error_raw = gate.get(
        "maximum_family_mean_adapted_future_relative_l2"
    )
    maximum_mean_error = (
        None
        if maximum_mean_error_raw is None
        else float(maximum_mean_error_raw)
    )
    maximum_family_mean_error = (
        None
        if maximum_family_mean_error_raw is None
        else float(maximum_family_mean_error_raw)
    )
    if not 0.0 <= minimum_nonworse <= 1.0:
        raise ValueError("minimum_nonworse_fraction must lie in [0,1]")
    if not math.isfinite(maximum_runtime) or maximum_runtime <= 0.0:
        raise ValueError("maximum_runtime_s must be positive")
    if maximum_p95_adaptation_runtime is not None and (
        not math.isfinite(maximum_p95_adaptation_runtime)
        or maximum_p95_adaptation_runtime <= 0.0
    ):
        raise ValueError("maximum_p95_adaptation_runtime_s must be positive")
    if (minimum_speedup is None) != (traditional_runtime is None):
        raise ValueError(
            "minimum speedup and traditional runtime reference must be configured together"
        )
    if minimum_speedup is not None and (
        not math.isfinite(minimum_speedup) or minimum_speedup <= 0.0
    ):
        raise ValueError(
            "minimum_end_to_end_speedup_vs_traditional must be positive"
        )
    if traditional_runtime is not None and (
        not math.isfinite(traditional_runtime) or traditional_runtime <= 0.0
    ):
        raise ValueError("traditional_solver_reference_runtime_s must be positive")
    if maximum_mean_error is not None and (
        not math.isfinite(maximum_mean_error) or maximum_mean_error <= 0.0
    ):
        raise ValueError(
            "maximum_mean_adapted_future_relative_l2 must be positive"
        )
    if maximum_family_mean_error is not None and (
        not math.isfinite(maximum_family_mean_error)
        or maximum_family_mean_error <= 0.0
    ):
        raise ValueError(
            "maximum_family_mean_adapted_future_relative_l2 must be positive"
        )

    rows: list[dict[str, object]] = []
    for report in reports:
        parent_error = float(report["parent_future_fullfield_relative_l2"])
        adapted_error = float(report["future_fullfield_relative_l2"])
        adaptation = dict(report["adaptation"])
        adaptation_elapsed_s = float(adaptation["adaptation_elapsed_s"])
        total_inference_elapsed_s = float(
            adaptation.get("total_inference_elapsed_s", adaptation_elapsed_s)
        )
        truth_squared_norm = float(report.get("future_truth_squared_norm", 0.0))
        adapted_squared_error = float(
            report.get("future_adapted_squared_error", -1.0)
        )
        parent_squared_error = float(
            report.get("future_parent_squared_error", -1.0)
        )
        if not all(
            math.isfinite(value)
            for value in (
                parent_error,
                adapted_error,
                adaptation_elapsed_s,
                total_inference_elapsed_s,
                truth_squared_norm,
                adapted_squared_error,
                parent_squared_error,
            )
        ):
            raise ValueError("CPADC promotion metrics must be finite")
        if parent_error < 0.0 or adapted_error < 0.0:
            raise ValueError("CPADC relative L2 metrics must be nonnegative")
        if adaptation_elapsed_s < 0.0 or total_inference_elapsed_s < 0.0:
            raise ValueError("CPADC runtime metrics must be nonnegative")
        if (
            truth_squared_norm <= 0.0
            or adapted_squared_error < 0.0
            or parent_squared_error < 0.0
        ):
            raise ValueError("CPADC exact future-field energy metrics are missing")
        improvement = (parent_error - adapted_error) / max(parent_error, 1.0e-12)
        rows.append(
            {
                "family": str(report["medium_type"]),
                "parent_error": parent_error,
                "adapted_error": adapted_error,
                "truth_squared_norm": truth_squared_norm,
                "adapted_squared_error": adapted_squared_error,
                "parent_squared_error": parent_squared_error,
                "relative_improvement": improvement,
                "nonworse": adapted_error <= parent_error * (1.0 + 1.0e-9),
                "adaptation_elapsed_s": adaptation_elapsed_s,
                "total_inference_elapsed_s": total_inference_elapsed_s,
                "cuda_synchronized_timing": bool(
                    adaptation.get("cuda_synchronized_timing", False)
                ),
                "future_truth_used": bool(adaptation["future_truth_used"]),
                "accepted": bool(adaptation["accepted"]),
                "coefficient_finetune_cpu_only": bool(
                    adaptation.get("coefficient_finetune_cpu_only", False)
                ),
                "coefficient_finetune_device": str(
                    adaptation.get("coefficient_finetune_device", "")
                ),
                "coefficient_objective_device": str(
                    adaptation.get("coefficient_objective_device", "")
                ),
                "correction_materialization_device": str(
                    adaptation.get("correction_materialization_device", "")
                ),
                "synthetic_bridge_device": str(
                    adaptation.get("synthetic_bridge_device", "")
                ),
                "instance_trainable_path_cpu_only": bool(
                    adaptation.get("instance_trainable_path_cpu_only", False)
                ),
                "all_saved_time_indices": int(
                    report.get("all_saved_time_indices", -1)
                ),
            }
        )
    mean_improvement = sum(float(row["relative_improvement"]) for row in rows) / len(rows)
    mean_adapted_error = sum(float(row["adapted_error"]) for row in rows) / len(rows)
    aggregate_truth_energy = sum(float(row["truth_squared_norm"]) for row in rows)
    aggregate_adapted_error = math.sqrt(
        sum(float(row["adapted_squared_error"]) for row in rows)
        / max(aggregate_truth_energy, 1.0e-30)
    )
    aggregate_parent_error = math.sqrt(
        sum(float(row["parent_squared_error"]) for row in rows)
        / max(aggregate_truth_energy, 1.0e-30)
    )
    nonworse_fraction = sum(bool(row["nonworse"]) for row in rows) / len(rows)
    maximum_observed_runtime = max(float(row["adaptation_elapsed_s"]) for row in rows)
    adaptation_runtimes = [float(row["adaptation_elapsed_s"]) for row in rows]
    total_runtimes = [float(row["total_inference_elapsed_s"]) for row in rows]
    mean_adaptation_runtime = sum(adaptation_runtimes) / len(adaptation_runtimes)
    p95_adaptation_runtime = _nearest_rank_percentile(adaptation_runtimes, 0.95)
    mean_total_runtime = sum(total_runtimes) / len(total_runtimes)
    p50_total_runtime = _nearest_rank_percentile(total_runtimes, 0.50)
    p95_total_runtime = _nearest_rank_percentile(total_runtimes, 0.95)
    maximum_total_runtime = max(total_runtimes)
    synchronized_timing = all(
        bool(row["cuda_synchronized_timing"]) for row in rows
    )
    mean_speedup = (
        None
        if traditional_runtime is None
        else traditional_runtime / max(mean_total_runtime, 1.0e-12)
    )
    p95_speedup = (
        None
        if traditional_runtime is None
        else traditional_runtime / max(p95_total_runtime, 1.0e-12)
    )
    family_metrics: dict[str, dict[str, object]] = {}
    for family in ALLOWED_MEDIUM_TYPES:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        parent_mean = sum(float(row["parent_error"]) for row in family_rows) / len(family_rows)
        adapted_mean = sum(float(row["adapted_error"]) for row in family_rows) / len(family_rows)
        family_truth_energy = sum(
            float(row["truth_squared_norm"]) for row in family_rows
        )
        adapted_aggregate = math.sqrt(
            sum(float(row["adapted_squared_error"]) for row in family_rows)
            / max(family_truth_energy, 1.0e-30)
        )
        parent_aggregate = math.sqrt(
            sum(float(row["parent_squared_error"]) for row in family_rows)
            / max(family_truth_energy, 1.0e-30)
        )
        family_metrics[family] = {
            "record_count": len(family_rows),
            "mean_parent_future_relative_l2": parent_mean,
            "mean_adapted_future_relative_l2": adapted_mean,
            "aggregate_parent_future_relative_l2": parent_aggregate,
            "aggregate_adapted_future_relative_l2": adapted_aggregate,
            "relative_improvement": (parent_mean - adapted_mean) / max(parent_mean, 1.0e-12),
            "nonworse": adapted_aggregate <= parent_aggregate * (1.0 + 1.0e-9),
        }
    every_family_nonworse = (
        len(family_metrics) == len(ALLOWED_MEDIUM_TYPES)
        and all(bool(value["nonworse"]) for value in family_metrics.values())
    )
    every_family_within_accuracy_target = (
        maximum_family_mean_error is None
        or (
            len(family_metrics) == len(ALLOWED_MEDIUM_TYPES)
            and all(
                float(value["aggregate_adapted_future_relative_l2"])
                <= maximum_family_mean_error
                for value in family_metrics.values()
            )
        )
    )
    checks = {
        "full_validation_protocol": bool(
            all_validation
            and all(int(row["all_saved_time_indices"]) == 401 for row in rows)
        ),
        "cpu_coefficient_finetune": all(
            bool(row["coefficient_finetune_cpu_only"])
            and str(row["coefficient_finetune_device"]) == "cpu"
            and str(row["coefficient_objective_device"]) == "cpu"
            and str(row["correction_materialization_device"]) == "cpu"
            and str(row["synthetic_bridge_device"]) == "cpu"
            and bool(row["instance_trainable_path_cpu_only"])
            for row in rows
        ),
        "future_truth_sealed": not any(bool(row["future_truth_used"]) for row in rows),
        "blind_mean_improvement": mean_improvement >= minimum_improvement,
        "nonworse_fraction": nonworse_fraction >= minimum_nonworse,
        "online_runtime": maximum_observed_runtime <= maximum_runtime,
        "online_runtime_p95": (
            maximum_p95_adaptation_runtime is None
            or p95_adaptation_runtime <= maximum_p95_adaptation_runtime
        ),
        "runtime_measurement_synchronized": (
            minimum_speedup is None or synchronized_timing
        ),
        "end_to_end_speedup_mean": (
            minimum_speedup is None
            or (
                mean_speedup is not None
                and mean_speedup >= minimum_speedup
            )
        ),
        "end_to_end_speedup_p95": (
            minimum_speedup is None
            or (
                p95_speedup is not None
                and p95_speedup >= minimum_speedup
            )
        ),
        "family_safe": every_family_nonworse if require_family_safe else True,
        "absolute_mean_accuracy": (
            maximum_mean_error is None
            or aggregate_adapted_error <= maximum_mean_error
        ),
        "absolute_family_accuracy": every_family_within_accuracy_target,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "record_count": len(rows),
        "acceptance_fraction": sum(bool(row["accepted"]) for row in rows) / len(rows),
        "mean_relative_improvement": mean_improvement,
        "mean_adapted_future_relative_l2": mean_adapted_error,
        "aggregate_parent_future_relative_l2": aggregate_parent_error,
        "aggregate_adapted_future_relative_l2": aggregate_adapted_error,
        "nonworse_fraction": nonworse_fraction,
        "maximum_adaptation_elapsed_s": maximum_observed_runtime,
        "mean_adaptation_elapsed_s": mean_adaptation_runtime,
        "p95_adaptation_elapsed_s": p95_adaptation_runtime,
        "mean_total_inference_elapsed_s": mean_total_runtime,
        "p50_total_inference_elapsed_s": p50_total_runtime,
        "p95_total_inference_elapsed_s": p95_total_runtime,
        "maximum_total_inference_elapsed_s": maximum_total_runtime,
        "mean_end_to_end_speedup_vs_traditional": mean_speedup,
        "p95_end_to_end_speedup_vs_traditional": p95_speedup,
        "thresholds": {
            "minimum_blind_mean_improvement": minimum_improvement,
            "minimum_nonworse_fraction": minimum_nonworse,
            "maximum_runtime_s": maximum_runtime,
            "maximum_p95_adaptation_runtime_s": (
                maximum_p95_adaptation_runtime
            ),
            "minimum_end_to_end_speedup_vs_traditional": minimum_speedup,
            "traditional_solver_reference_runtime_s": traditional_runtime,
            "require_every_family_nonworse": require_family_safe,
            "maximum_mean_adapted_future_relative_l2": maximum_mean_error,
            "maximum_family_mean_adapted_future_relative_l2": (
                maximum_family_mean_error
            ),
        },
        "families": family_metrics,
    }


def _load_basis(
    path: str | Path,
    *,
    parent_checkpoint: str | Path,
    manifest_digest: str,
    device: torch.device,
    dataset_numerical_contract: dict[str, object] | None = None,
    implementation_digests: dict[str, str] | None = None,
    basis_manifest_digest: str | None = None,
    allow_evaluation_dataset_mismatch: bool = False,
) -> tuple[CausalErrorBasisGenerator, dict[str, object]]:
    checkpoint = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    schema_version = int(payload.get("schema_version", 0))
    if schema_version not in (4, 5, 6, 7, 8, 9, CPADC_SCHEMA_VERSION):
        raise ValueError("CPADC checkpoint schema mismatch")
    if payload.get("schema") != "causal_physics_aligned_defect_correction_v1":
        raise ValueError("CPADC checkpoint type mismatch")
    solve_contract = payload.get("online_solve_contract") or {}
    solve_name = solve_contract.get("name")
    if solve_name not in {
        "ridge_direction_learned_energy_ball_projection_v1",
        "ridge_direction_calibrated_strength_abstention_v1",
        "ridge_direction_family_calibrated_strength_abstention_v1",
        "cpu_causal_observation_bridge_ridge_v2",
        "cpu_causal_observation_bridge_calibrated_ridge_v2",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v2",
        "cpu_causal_observation_bridge_ridge_v3",
        "cpu_causal_observation_bridge_calibrated_ridge_v3",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v3",
        "cpu_causal_observation_bridge_ridge_v4",
        "cpu_causal_observation_bridge_calibrated_ridge_v4",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v4",
        "cpu_causal_observation_bridge_ridge_v5",
        "cpu_causal_observation_bridge_calibrated_ridge_v5",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v5",
        "cpu_causal_sparse_defect_observation_bridge_ridge_v6",
        "cpu_causal_sparse_defect_observation_bridge_calibrated_ridge_v6",
        "cpu_causal_sparse_defect_observation_bridge_family_calibrated_ridge_v6",
        "cpu_causal_weak_sparse_defect_observation_bridge_ridge_v7",
        "cpu_causal_weak_sparse_defect_observation_bridge_calibrated_ridge_v7",
        "cpu_causal_weak_sparse_defect_observation_bridge_family_calibrated_ridge_v7",
    }:
        raise ValueError("CPADC checkpoint online solve contract mismatch")
    strength_floor = float(
        solve_contract.get("minimum_unconstrained_correction_ratio", 0.0)
    )
    if strength_floor < 0.0:
        raise ValueError("CPADC calibrated strength floor is invalid")
    strength_floors_by_family: dict[str, float] = {}
    if solve_name in {
        "ridge_direction_family_calibrated_strength_abstention_v1",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v2",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v3",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v4",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v5",
        "cpu_causal_sparse_defect_observation_bridge_family_calibrated_ridge_v6",
        "cpu_causal_weak_sparse_defect_observation_bridge_family_calibrated_ridge_v7",
    }:
        raw_floors = solve_contract.get(
            "minimum_unconstrained_correction_ratio_by_family"
        )
        if not isinstance(raw_floors, dict) or set(raw_floors) != set(
            ALLOWED_MEDIUM_TYPES
        ):
            raise ValueError("CPADC family-calibrated strength floors are incomplete")
        strength_floors_by_family = {
            family: float(raw_floors[family]) for family in ALLOWED_MEDIUM_TYPES
        }
        if not all(
            value >= 0.0 and torch.isfinite(torch.tensor(value)).item()
            for value in strength_floors_by_family.values()
        ):
            raise ValueError("CPADC family-calibrated strength floor is invalid")
    if solve_name in {
        "ridge_direction_calibrated_strength_abstention_v1",
        "ridge_direction_family_calibrated_strength_abstention_v1",
        "cpu_causal_observation_bridge_calibrated_ridge_v2",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v2",
        "cpu_causal_observation_bridge_calibrated_ridge_v3",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v3",
        "cpu_causal_observation_bridge_calibrated_ridge_v4",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v4",
        "cpu_causal_observation_bridge_calibrated_ridge_v5",
        "cpu_causal_observation_bridge_family_calibrated_ridge_v5",
        "cpu_causal_sparse_defect_observation_bridge_calibrated_ridge_v6",
        "cpu_causal_sparse_defect_observation_bridge_family_calibrated_ridge_v6",
        "cpu_causal_weak_sparse_defect_observation_bridge_calibrated_ridge_v7",
        "cpu_causal_weak_sparse_defect_observation_bridge_family_calibrated_ridge_v7",
    }:
        calibration = payload.get("risk_calibration") or {}
        if calibration.get("future_truth_scope") != "disjoint_train_split_only":
            raise ValueError("CPADC risk calibration provenance mismatch")
        if int(calibration.get("record_count", 0)) <= 0:
            raise ValueError("CPADC risk calibration manifest is empty")
    defect_contract = payload.get("defect_contract") or {}
    if schema_version == CPADC_SCHEMA_VERSION:
        cpml_contract = saved_grid_cpml_config(defect_contract.get("cpml"))
        offline_physics = dict(payload.get("offline_physics_training") or {})
        if (
            defect_contract.get("name")
            != "source_consistent_saved_grid_lwc84_cfs_cpml_v3"
            or int(defect_contract.get("spatial_halo_cells", -1)) != 0
            or defect_contract.get("exact_fine_generator_residual") is not False
            or tuple(defect_contract.get("cpml_memory_variables", ()))
            != ("psi_x", "psi_z", "phi_x", "phi_z")
            or defect_contract.get("top_boundary")
            != "free_surface_dirichlet"
            or defect_contract.get("cpml_memory_reconstructed_causally")
            is not True
            or defect_contract.get("exterior_pressure_closure")
            != "zero_unavailable_state"
            or defect_contract.get("saved_source_contract")
            != "direct_saved_grid_bilinear_unit_mass_reinjection"
            or defect_contract.get("exact_fine_source_restriction") is not False
            or cpml_contract is None
            or cpml_contract.memory_time_integration
            != "internal_substep_exact_linear_causal_vectorized"
            or offline_physics.get("operator")
            != "lwc84_cfs_cpml_saved_grid_arrival_window_v4"
            or int(offline_physics.get("time_count", -1)) != 64
            or int(offline_physics.get("evaluations_per_candidate", -1)) != 1
            or offline_physics.get("window_strategy")
            != "nearest_side_cpml_peak_arrival_causal_prefix_v1"
            or offline_physics.get("causal_prefix_memory_warmup") is not True
            or offline_physics.get("detached_prefix_before_loss_window") is not True
            or int(offline_physics.get("prearrival_frames", -1)) != 8
            or int(offline_physics.get("boundary_band_cells", -1)) != 12
            or float(offline_physics.get("boundary_weight", -1.0)) != 4.0
        ):
            raise ValueError("CPADC checkpoint CPML defect contract mismatch")
        checkpoint_dataset_contract = dict(
            payload.get("dataset_numerical_contract") or {}
        )
        dataset_contract_matches = bool(
            dataset_numerical_contract is not None
            and (
                checkpoint_dataset_contract == dict(dataset_numerical_contract)
                or (
                    allow_evaluation_dataset_mismatch
                    and same_dataset_numerical_protocol(
                        checkpoint_dataset_contract,
                        dict(dataset_numerical_contract),
                    )
                )
            )
        )
        if not dataset_contract_matches:
            raise ValueError("CPADC checkpoint dataset numerical contract mismatch")
        if (
            implementation_digests is None
            or dict(payload.get("implementation_digests") or {})
            != dict(implementation_digests)
        ):
            raise ValueError("CPADC checkpoint implementation digest mismatch")
    elif schema_version == 9:
        cpml_contract = saved_grid_cpml_config(defect_contract.get("cpml"))
        offline_physics = dict(payload.get("offline_physics_training") or {})
        if (
            defect_contract.get("name")
            != "source_consistent_saved_grid_lwc84_cfs_cpml_v3"
            or int(defect_contract.get("spatial_halo_cells", -1)) != 0
            or defect_contract.get("exact_fine_generator_residual") is not False
            or tuple(defect_contract.get("cpml_memory_variables", ()))
            != ("psi_x", "psi_z", "phi_x", "phi_z")
            or defect_contract.get("top_boundary")
            != "free_surface_dirichlet"
            or defect_contract.get("cpml_memory_reconstructed_causally")
            is not True
            or defect_contract.get("exterior_pressure_closure")
            != "zero_unavailable_state"
            or cpml_contract is None
            or cpml_contract.memory_time_integration
            != "internal_substep_exact_linear_causal_vectorized"
            or offline_physics.get("operator")
            != "lwc84_cfs_cpml_saved_grid_arrival_window_v4"
            or int(offline_physics.get("time_count", -1)) != 64
            or int(offline_physics.get("evaluations_per_candidate", -1)) != 1
            or offline_physics.get("window_strategy")
            != "nearest_side_cpml_peak_arrival_causal_prefix_v1"
            or offline_physics.get("causal_prefix_memory_warmup") is not True
            or offline_physics.get("detached_prefix_before_loss_window") is not True
            or int(offline_physics.get("prearrival_frames", -1)) != 8
            or int(offline_physics.get("boundary_band_cells", -1)) != 12
            or float(offline_physics.get("boundary_weight", -1.0)) != 4.0
        ):
            raise ValueError("legacy schema-9 CPADC CPML defect contract mismatch")
    elif schema_version == 8:
        cpml_contract = saved_grid_cpml_config(defect_contract.get("cpml"))
        if (
            defect_contract.get("name")
            != "source_consistent_saved_grid_lwc84_cfs_cpml_v3"
            or int(defect_contract.get("spatial_halo_cells", -1)) != 0
            or cpml_contract is None
            or cpml_contract.memory_time_integration
            != "internal_substep_exact_linear_causal_vectorized"
        ):
            raise ValueError("legacy CPADC linear CPML defect contract mismatch")
    elif schema_version == 7:
        cpml_contract = saved_grid_cpml_config(defect_contract.get("cpml"))
        if (
            defect_contract.get("name")
            != "source_consistent_saved_grid_lwc84_cfs_cpml_v2"
            or int(defect_contract.get("spatial_halo_cells", -1)) != 0
            or cpml_contract is None
            or cpml_contract.memory_time_integration
            != "internal_substep_exact_zoh_vectorized"
        ):
            raise ValueError("legacy CPADC CPML defect contract mismatch")
    elif (
        defect_contract.get("name")
        != "source_consistent_effective_saved_grid_lwc84_v1"
        or int(defect_contract.get("spatial_halo_cells", 0)) != 8
        or defect_contract.get("exact_fine_generator_residual") is not False
    ):
        raise ValueError("legacy CPADC checkpoint defect contract mismatch")
    expected_basis_manifest = str(basis_manifest_digest or manifest_digest)
    if payload.get("manifest_digest") != expected_basis_manifest:
        raise ValueError("CPADC checkpoint manifest digest mismatch")
    expected_parent = Path(parent_checkpoint).expanduser().resolve()
    if Path(str(payload.get("parent_checkpoint", ""))).resolve() != expected_parent:
        raise ValueError("CPADC checkpoint is bound to a different parent")
    expected_parent_sha = str(payload.get("parent_checkpoint_sha256", ""))
    if not expected_parent_sha or _sha256(expected_parent) != expected_parent_sha:
        raise ValueError("CPADC parent checkpoint hash mismatch")
    generator = CausalErrorBasisGenerator(
        rank=int(payload["basis_rank"]),
        phase_rank=int(payload["phase_rank"]),
        width=int(payload.get("basis_width", 32)),
        ramp_steps=int(payload["causal_ramp_steps"]),
    ).to(device)
    generator.load_state_dict(payload["basis_state"], strict=True)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "parent_checkpoint": str(expected_parent),
        "parent_checkpoint_sha256": expected_parent_sha,
        "basis_rank": generator.rank,
        "phase_rank": generator.phase_rank,
        "epoch": payload.get("epoch"),
        "defect_contract": defect_contract,
        "online_solve_contract": solve_contract,
        "offline_physics_training": dict(
            payload.get("offline_physics_training") or {}
        ),
        "dataset_numerical_contract": dict(
            payload.get("dataset_numerical_contract") or {}
        ),
        "implementation_digests": dict(
            payload.get("implementation_digests") or {}
        ),
        "observed_probe_design": bool(
            solve_contract.get("observed_probe_design", False)
        ),
        "minimum_unconstrained_correction_ratio": strength_floor,
        "minimum_unconstrained_correction_ratio_by_family": (
            strength_floors_by_family
        ),
        "schema_version": schema_version,
        "basis_manifest_digest": expected_basis_manifest,
        "evaluation_manifest_digest": str(manifest_digest),
        "external_evaluation_dataset": bool(allow_evaluation_dataset_mismatch),
    }


def run(
    config_path: str | Path,
    *,
    basis_checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    adaptation_device_name: str = "cpu",
    sample_ids: tuple[str, ...] | None = None,
    per_family: int = 3,
    all_validation: bool = False,
    all_test_id: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    calibration_per_family: int | None = None,
    calibration_seed: int | None = None,
    basis_training_per_family: int | None = None,
    save_fields: bool = True,
    parent_checkpoint: str | Path | None = None,
    parent_checkpoint_identity: str | Path | None = None,
    travel_time_h5: str | Path | None = None,
) -> list[dict[str, object]]:
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("CPADC config must contain a mapping")
    if parent_checkpoint is not None:
        config["parent_checkpoint"] = str(Path(parent_checkpoint).resolve())
    if parent_checkpoint_identity is not None:
        config["parent_checkpoint_identity"] = str(
            Path(parent_checkpoint_identity).resolve()
        )
    if travel_time_h5 is not None:
        config["travel_time_h5"] = str(Path(travel_time_h5).resolve())
    manifest = build_manifest(config["source_h5"])
    external_evaluation = None
    saved_time_config = resolve_saved_time_parent_config(config)
    if saved_time_config is not None:
        _, training_manifest, _ = _load_context(saved_time_config)
        external_evaluation = validate_external_evaluation_contract(
            config, training_manifest, manifest
        )
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    if all_validation and all_test_id:
        raise ValueError("full validation and full test_id are mutually exclusive")
    complete_evaluation_split = (
        "validation" if all_validation else "test_id" if all_test_id else None
    )
    if complete_evaluation_split is not None and calibration_per_family is not None:
        raise ValueError("complete evaluation and risk calibration are mutually exclusive")
    if basis_training_per_family is not None and calibration_per_family is None:
        raise ValueError(
            "basis training per-family override requires train-only calibration"
        )
    if shard_count > 1 and not (
        complete_evaluation_split is not None or calibration_per_family is not None
    ):
        raise ValueError("sharded evaluation requires a complete selection protocol")
    selection_split = "validation"
    resolved_basis_training_per_family = None
    if calibration_per_family is not None:
        resolved_basis_training_per_family = resolve_basis_training_per_family(
            config, basis_training_per_family
        )
        if int(calibration_per_family) <= 0:
            raise ValueError("risk calibration episode counts must be positive")
        full_rows = build_risk_calibration_manifest(
            manifest,
            basis_per_family=resolved_basis_training_per_family,
            basis_seed=int(config.get("seed", 372)),
            calibration_per_family=int(calibration_per_family),
            calibration_seed=(
                int(config.get("seed", 372)) + 10_000
                if calibration_seed is None
                else int(calibration_seed)
            ),
        )
        rows = full_rows[shard_index::shard_count]
        selection_split = "train"
    elif complete_evaluation_split is not None:
        full_rows = build_complete_evaluation_manifest(
            manifest, split=complete_evaluation_split
        )
        rows = full_rows[shard_index::shard_count]
        selection_split = complete_evaluation_split
    elif sample_ids:
        by_sample = {
            row.sample_id: row for row in manifest.records if row.split == "validation"
        }
        missing = tuple(value for value in sample_ids if value not in by_sample)
        if missing:
            raise ValueError(f"selected validation samples are missing: {missing[:3]}")
        rows = tuple(by_sample[value] for value in sample_ids)
    else:
        rows = build_instance_manifest(
            manifest,
            seed=int(config.get("seed", 372)),
            per_family=int(per_family),
        )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_manifest(rows, output / "instance_manifest.json")
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    adaptation_device = torch.device(adaptation_device_name)
    if adaptation_device.type != "cpu":
        raise ValueError("CPADC instance adaptation is registered as CPU-only")
    parent, normalizer = _load_parent(config, manifest, device)
    provider = _load_background_provider(config, (row.sample_id for row in rows))
    dx = float(config.get("dx_m", 10.0))
    dz = float(config.get("dz_m", 10.0))
    cpml_config = saved_grid_cpml_config(config.get("pde_cpml"))
    if cpml_config is None:
        raise ValueError("CPADC deployment requires an enabled pde_cpml contract")
    dataset_numerical_contract = validate_dataset_cpml_contract(
        config["source_h5"],
        saved_grid_cpml=cpml_config.as_dict(),
        saved_dt_s=float(config.get("dt_s", 0.0025)),
        saved_dx_m=dx,
        saved_dz_m=dz,
    )
    implementation_digests = cpadc_implementation_digests(ROOT)
    generator, basis_info = _load_basis(
        basis_checkpoint,
        parent_checkpoint=config["parent_checkpoint"],
        manifest_digest=manifest.digest,
        device=device,
        dataset_numerical_contract=dataset_numerical_contract,
        implementation_digests=implementation_digests,
        basis_manifest_digest=(
            str(external_evaluation["training_manifest_digest"])
            if external_evaluation and bool(external_evaluation["external"])
            else manifest.digest
        ),
        allow_evaluation_dataset_mismatch=bool(
            external_evaluation and external_evaluation["external"]
        ),
    )
    schema_version = int(basis_info["schema_version"])
    solve_contract = dict(basis_info["online_solve_contract"])
    solve_name = str(solve_contract.get("name", ""))
    dataset = GuardedOnsetDataset(
        config["source_h5"],
        manifest,
        split=selection_split,
        sample_ids=tuple(row.sample_id for row in rows),
        travel_time_h5=config.get("travel_time_h5"),
    )
    gate = config.get("deployment_gate", {}) or {}
    inner_raw = config.get("online_inner_weights", {}) or {}
    inner_weights = DefectCorrectionWeights(
        defect=float(inner_raw.get("defect", 0.0)),
        observed=float(inner_raw.get("observed", 0.0)),
        bridge=float(inner_raw.get("bridge", 0.5)),
        prior=float(inner_raw.get("prior", 1.0e-4)),
    )
    if schema_version == CPADC_SCHEMA_VERSION and (
        solve_name not in {
            "cpu_causal_observation_bridge_ridge_v5",
            "cpu_causal_observation_bridge_calibrated_ridge_v5",
            "cpu_causal_observation_bridge_family_calibrated_ridge_v5",
            "cpu_causal_sparse_defect_observation_bridge_ridge_v6",
            "cpu_causal_sparse_defect_observation_bridge_calibrated_ridge_v6",
            "cpu_causal_sparse_defect_observation_bridge_family_calibrated_ridge_v6",
            "cpu_causal_weak_sparse_defect_observation_bridge_ridge_v7",
            "cpu_causal_weak_sparse_defect_observation_bridge_calibrated_ridge_v7",
            "cpu_causal_weak_sparse_defect_observation_bridge_family_calibrated_ridge_v7",
        }
        or solve_contract.get("coefficient_solve_device") != "cpu"
        or solve_contract.get("coefficient_objective_device") != "cpu"
        or solve_contract.get("correction_materialization_device") != "cpu"
        or solve_contract.get("synthetic_bridge_device") != "cpu"
        or float(solve_contract.get("online_defect_weight", -1.0))
        != float(inner_weights.defect)
        or float(solve_contract.get("online_observed_weight", -1.0))
        != float(inner_weights.observed)
        or float(solve_contract.get("online_bridge_weight", -1.0))
        != float(inner_weights.bridge)
        or (
            "online_prior_weight" in solve_contract
            and float(solve_contract["online_prior_weight"])
            != float(inner_weights.prior)
        )
        or solve_contract.get("offline_physics_operator")
        != "lwc84_cfs_cpml_saved_grid_arrival_window_v4"
    ):
        raise ValueError("CPADC online CPU solve contract mismatch")
    sparse_online_defect = solve_name in {
        "cpu_causal_sparse_defect_observation_bridge_ridge_v6",
        "cpu_causal_sparse_defect_observation_bridge_calibrated_ridge_v6",
        "cpu_causal_sparse_defect_observation_bridge_family_calibrated_ridge_v6",
        "cpu_causal_weak_sparse_defect_observation_bridge_ridge_v7",
        "cpu_causal_weak_sparse_defect_observation_bridge_calibrated_ridge_v7",
        "cpu_causal_weak_sparse_defect_observation_bridge_family_calibrated_ridge_v7",
    }
    weak_sparse_online_defect = solve_name in {
        "cpu_causal_weak_sparse_defect_observation_bridge_ridge_v7",
        "cpu_causal_weak_sparse_defect_observation_bridge_calibrated_ridge_v7",
        "cpu_causal_weak_sparse_defect_observation_bridge_family_calibrated_ridge_v7",
    }
    online_defect_test_function_width = int(
        solve_contract.get("online_defect_test_function_width", 1)
    )
    if schema_version == CPADC_SCHEMA_VERSION:
        sparse_contract_valid = (
            float(inner_weights.defect) > 0.0
            and solve_contract.get("online_defect_design") == "sparse_interior"
            and int(solve_contract.get("online_defect_time_order", -1)) == 2
            and int(solve_contract.get("online_physics_point_count", 0)) > 0
            and solve_contract.get("online_physics_sampling") == "rad"
            and online_defect_test_function_width >= 1
            and online_defect_test_function_width % 2 == 1
            and (
                solve_contract.get("online_defect_test_function_normalization")
                == "discrete_l2_unit"
                or (
                    not weak_sparse_online_defect
                    and online_defect_test_function_width == 1
                    and "online_defect_test_function_normalization"
                    not in solve_contract
                )
            )
            and (
                (weak_sparse_online_defect and online_defect_test_function_width > 1)
                or (
                    not weak_sparse_online_defect
                    and online_defect_test_function_width == 1
                )
            )
            and int(solve_contract.get("online_spatial_stride", 0)) > 0
            and float(solve_contract.get("online_prior_weight", -1.0))
            == float(inner_weights.prior)
            and math.isfinite(float(solve_contract.get("online_rad_k", math.nan)))
            and float(solve_contract.get("online_rad_k", -1.0)) >= 0.0
            and math.isfinite(float(solve_contract.get("online_rad_c", math.nan)))
            and float(solve_contract.get("online_rad_c", -1.0)) >= 0.0
            and math.isfinite(
                float(solve_contract.get("online_rad_time_tilt", math.nan))
            )
        )
        legacy_contract_valid = (
            float(inner_weights.defect) == 0.0 and not sparse_online_defect
        )
        if not (
            (sparse_online_defect and sparse_contract_valid)
            or legacy_contract_valid
        ):
            raise ValueError("CPADC online defect contract mismatch")
    configured_observed_probe = float(inner_weights.observed) > 0.0
    if configured_observed_probe != bool(basis_info["observed_probe_design"]):
        raise ValueError(
            "CPADC observed-probe solve does not match the checkpoint training contract"
        )
    pressure_scale = float(normalizer.metadata.pressure_scale_pa)
    if cpml_config.as_dict() != dict(basis_info["defect_contract"].get("cpml", {})):
        raise ValueError("CPADC config and checkpoint CPML contracts differ")
    configured_offline_physics = config.get("outer_weights", {}) or {}
    checkpoint_offline_physics = basis_info.get("offline_physics_training", {}) or {}
    if schema_version == CPADC_SCHEMA_VERSION and (
        int(configured_offline_physics.get("physics_time_count", -1))
        != int(checkpoint_offline_physics.get("time_count", -2))
        or str(configured_offline_physics.get("physics_window_strategy", ""))
        != str(checkpoint_offline_physics.get("window_strategy", "missing"))
        or int(configured_offline_physics.get("physics_prearrival_frames", -1))
        != int(checkpoint_offline_physics.get("prearrival_frames", -2))
        or int(configured_offline_physics.get("physics_boundary_band_cells", -1))
        != int(checkpoint_offline_physics.get("boundary_band_cells", -2))
        or float(configured_offline_physics.get("physics_boundary_weight", -1.0))
        != float(checkpoint_offline_physics.get("boundary_weight", -2.0))
    ):
        raise ValueError("CPADC config and checkpoint offline physics contracts differ")
    reports: list[dict[str, object]] = []
    try:
        for record in dataset:
            record.audit.read(record.observed_indices)
            _synchronize_device_for_timing(device)
            record_started = time.perf_counter()
            velocity = record.velocity_mps.to(device).unsqueeze(0)
            source = record.source_parameters.to(device).unsqueeze(0)
            source_map = record.source_map.to(device).unsqueeze(0)
            times = record.time_s.to(device)
            observed = normalizer.encode_pressure(
                record.observed_wavefield.to(device).unsqueeze(0), source[:, 4]
            )
            parent_normalized = _predict_parent(
                parent,
                normalizer,
                record,
                device,
                normalized=True,
                background_provider=provider,
                time_block=int(config.get("deployment_time_block", 32)),
            )
            _synchronize_device_for_timing(device)
            adaptation_started = time.perf_counter()
            adaptation_velocity = velocity.to(adaptation_device)
            adaptation_source = source.to(adaptation_device)
            adaptation_source_map = source_map.to(adaptation_device)
            adaptation_times = times.to(adaptation_device)
            adaptation_observed_physical = (
                record.observed_wavefield.to(adaptation_device).unsqueeze(0)
            )
            bridge = make_onset_bridge(
                adaptation_velocity,
                adaptation_source,
                adaptation_observed_physical,
                record.observed_indices,
                adaptation_times,
                source_map=adaptation_source_map,
                steps=int(config.get("bridge_steps", 4)),
                dx_m=dx,
                dz_m=dz,
                device=adaptation_device,
            )
            if not bridge.valid:
                raise RuntimeError(
                    f"synthetic bridge failed for {record.sample_id}: {bridge.failure_reason}"
                )
            bridge_normalized = normalizer.encode_pressure(
                bridge.frames, adaptation_source[:, 4]
            )
            dt = float((times[1:] - times[:-1]).mean())
            field_scale_pa = source[:, 4] * pressure_scale
            with torch.no_grad():
                strength_floor_by_family = dict(
                    basis_info[
                        "minimum_unconstrained_correction_ratio_by_family"
                    ]
                )
                record_strength_floor = float(
                    strength_floor_by_family.get(
                        record.medium_type,
                        basis_info["minimum_unconstrained_correction_ratio"],
                    )
                )
                # A cheap interior LWC-4 summary conditions the frozen basis.
                # The full CFS-CPML objective trained that basis offline; online
                # CPU adaptation does not re-evaluate CPML for every rank mode.
                parent_defect, defect_scale = lwc84_discrete_defect(
                    parent_normalized,
                    velocity,
                    dt=dt,
                    dx=dx,
                    dz=dz,
                    observed_indices=record.observed_indices,
                    source_parameters=source,
                    source_map=source_map,
                    time_s=times,
                    field_scale_pa=field_scale_pa,
                    normalize=False,
                    return_scale=True,
                )
                basis = generator(
                    parent_normalized,
                    velocity,
                    observed,
                    source,
                    times,
                    record.observed_indices,
                    parent_defect=parent_defect,
                    defect_scale=defect_scale,
                )
                # The frozen generator may execute on the deployment GPU, but
                # every instance-specific objective, optimized coefficient and
                # output correction is moved to and evaluated on CPU.
                adaptation_basis = basis.to(adaptation_device)
                adaptation_parent = parent_normalized.to(adaptation_device)
                adaptation_observed = observed.to(adaptation_device)
                adaptation_bridge = bridge_normalized
                adaptation_field_scale = field_scale_pa.to(adaptation_device)
                observed_probe_basis = (
                    None
                    if float(inner_weights.observed) == 0.0
                    else causal_observation_probe_basis(adaptation_basis)
                )
                point_seed = int(
                    solve_contract.get("online_physics_seed", 372)
                ) + int(
                    record.source_index
                )
                if sparse_online_defect:
                    point_generator = torch.Generator(
                        device=parent_defect.device
                    ).manual_seed(point_seed)
                    physics_points = build_rad_physics_points(
                        parent_defect,
                        record.observed_indices,
                        count=int(solve_contract["online_physics_point_count"]),
                        k=float(solve_contract["online_rad_k"]),
                        c=float(solve_contract["online_rad_c"]),
                        time_tilt=float(
                            solve_contract["online_rad_time_tilt"]
                        ),
                        generator=point_generator,
                    )
                else:
                    physics_points = build_fixed_physics_points(
                        len(times),
                        record.observed_indices,
                        count=1,
                        seed=point_seed,
                    )
                result = solve_causal_defect_correction(
                    adaptation_basis,
                    adaptation_parent,
                    adaptation_velocity,
                    adaptation_source,
                    adaptation_source_map,
                    adaptation_times,
                    record.observed_indices,
                    physics_points,
                    field_scale_pa=adaptation_field_scale,
                    observed_wavefield=adaptation_observed,
                    observed_design_basis=observed_probe_basis,
                    bridge_wavefield=adaptation_bridge,
                    bridge_indices=bridge.time_indices,
                    weights=inner_weights,
                    dt=dt,
                    dx=dx,
                    dz=dz,
                    spatial_stride=int(
                        solve_contract.get("online_spatial_stride", 4)
                    ),
                    rank_chunk_size=int(
                        (config.get("cpadc", {}) or {}).get("rank_chunk_size", 2)
                    ),
                    time_order=int(
                        solve_contract.get("online_defect_time_order", 4)
                    ),
                    defect_design=str(
                        solve_contract.get("online_defect_design", "materialized")
                    ),
                    defect_test_function_width=(
                        online_defect_test_function_width
                    ),
                    minimum_relative_improvement=float(
                        gate.get("minimum_causal_objective_improvement", 1.0e-3)
                    ),
                    maximum_condition_number=float(
                        gate.get("maximum_condition_number", 1.0e8)
                    ),
                    minimum_effective_design_rank=int(
                        gate.get("minimum_effective_design_rank", 1)
                    ),
                    maximum_correction_ratio=float(
                        gate.get("maximum_correction_ratio", 0.25)
                    ),
                    minimum_unconstrained_correction_ratio=float(
                        record_strength_floor
                    ),
                    cpml_config=None,
                    solve_device=adaptation_device,
                )
            _synchronize_device_for_timing(device)
            adaptation_elapsed = time.perf_counter() - adaptation_started
            adapted_field = normalizer.decode_pressure(
                result.field, adaptation_source[:, 4]
            )
            # Materialize deployment outputs before closing the end-to-end timer.
            # Disk serialization and the later opening of future truth remain outside
            # the compute-time protocol.
            adapted_field_cpu = adapted_field.detach().cpu()
            _synchronize_device_for_timing(device)
            total_elapsed = time.perf_counter() - record_started
            # The parent-only field is retained solely for the offline accuracy
            # comparison.  It is not a deployment output and is copied after the
            # end-to-end compute timer has closed.
            parent_field_cpu = normalizer.decode_pressure(
                parent_normalized, source[:, 4]
            ).detach().cpu()
            artifact = output / record.sample_id
            artifact.mkdir(parents=True, exist_ok=True)
            adaptation_payload = {
                "accepted": bool(result.accepted[0]),
                "rollback_reason": result.rollback_reasons[0],
                "objective_before": float(result.objective_before[0]),
                "objective_after": float(result.objective_after[0]),
                "condition_number": float(result.condition_number[0]),
                "effective_design_rank": int(result.effective_design_rank[0]),
                "design_row_count": int(result.design_row_count),
                "coefficient_solve_elapsed_s": float(
                    result.coefficient_solve_elapsed_s
                ),
                "correction_ratio": float(result.correction_ratio[0]),
                "unconstrained_correction_ratio": float(
                    result.unconstrained_correction_ratio[0]
                ),
                "projection_scale": float(result.projection_scale[0]),
                "trust_fraction": float(result.trust_fraction[0]),
                "observed_probe_design": configured_observed_probe,
                "observed_weight": float(inner_weights.observed),
                "effective_correction_ratio_limit": float(
                    result.effective_correction_ratio_limit[0]
                ),
                "minimum_unconstrained_correction_ratio": float(
                    record_strength_floor
                ),
                "adaptation_elapsed_s": adaptation_elapsed,
                "total_inference_elapsed_s": total_elapsed,
                "cuda_synchronized_timing": True,
                "parent_inference_device": str(device),
                "adaptation_device": result.coefficient_solve_device,
                # Only the instance-specific optimized variables are the ridge
                # coefficients below.  Their complete fit is on CPU; frozen
                # parent/basis feature evaluation stays on the deployment GPU.
                "adaptation_cpu_only": False,
                "coefficient_finetune_cpu_only": (
                    result.coefficient_solve_device == "cpu"
                    and result.coefficient_objective_device == "cpu"
                    and result.correction_materialization_device == "cpu"
                ),
                "coefficient_finetune_device": result.coefficient_solve_device,
                "coefficient_objective_device": (
                    result.coefficient_objective_device
                ),
                "correction_materialization_device": (
                    result.correction_materialization_device
                ),
                "synthetic_bridge_device": str(bridge.frames.device),
                "instance_trainable_path_cpu_only": (
                    result.coefficient_solve_device == "cpu"
                    and result.coefficient_objective_device == "cpu"
                    and result.correction_materialization_device == "cpu"
                    and bridge.frames.device.type == "cpu"
                ),
                "frozen_feature_device": str(device),
                "offline_pde_operator": "lwc84_cfs_cpml_saved_grid_arrival_window_v4",
                "online_defect_weight": float(inner_weights.defect),
                "online_defect_design": str(
                    solve_contract.get("online_defect_design", "materialized")
                ),
                "online_defect_time_order": int(
                    solve_contract.get("online_defect_time_order", 4)
                ),
                "online_defect_test_function_width": (
                    online_defect_test_function_width
                ),
                "online_defect_test_function_normalization": str(
                    solve_contract.get(
                        "online_defect_test_function_normalization", "missing"
                    )
                ),
                "online_physics_point_count": int(
                    solve_contract.get("online_physics_point_count", 1)
                ),
                "online_physics_sampling": str(
                    solve_contract.get("online_physics_sampling", "fixed")
                ),
                "online_rad_k": float(
                    solve_contract.get("online_rad_k", 1.0)
                ),
                "online_rad_c": float(
                    solve_contract.get("online_rad_c", 1.0)
                ),
                "online_rad_time_tilt": float(
                    solve_contract.get("online_rad_time_tilt", 1.5)
                ),
                "online_physics_seed": int(
                    solve_contract.get("online_physics_seed", 372)
                ),
                "online_prior_weight": float(inner_weights.prior),
                "online_spatial_stride": int(
                    solve_contract.get("online_spatial_stride", 4)
                ),
                "online_solve": "causal_observation_bridge_ridge_cpu_v5",
                "pde_cpml_contract": cpml_config.as_dict(),
                "runtime_protocol": (
                    "input_ready_to_401_frame_output_materialized_no_disk_io_v1"
                ),
                "parent_inference_time_block": int(
                    config.get("deployment_time_block", 32)
                ),
                "coefficient_count": int(result.coefficients.shape[1]),
                "coefficients": result.coefficients[0].detach().cpu(),
                "accessed_true_indices": tuple(record.audit.requested_indices),
                "future_truth_used": bool(record.audit.payload()["future_truth_used"]),
                "basis": basis_info,
            }
            sealed = {"adaptation": adaptation_payload}
            if save_fields:
                sealed.update(
                    {
                        "parent_field": parent_field_cpu,
                        "adapted_field": adapted_field_cpu,
                    }
                )
            # Seal all deployment outputs before opening any future frame.
            torch.save(sealed, artifact / "adaptation.pt")

            truth = _read_future_truth(config["source_h5"], record.source_index)
            report = evaluate_after_adaptation(
                {
                    "parent_field": parent_field_cpu,
                    "adapted_field": adapted_field_cpu,
                },
                truth.unsqueeze(0),
                observed_indices=(record.observed_indices,),
                families=(record.medium_type,),
                group_ids=(record.group_id,),
                sample_ids=(record.sample_id,),
                sealed=True,
            )
            report.update(
                {
                    "sample_id": record.sample_id,
                    "medium_type": record.medium_type,
                    "adaptation": adaptation_payload,
                }
            )
            write_report(report, artifact / "evaluation.json")
            reports.append(report)
    finally:
        dataset.close()
        if provider is not None:
            provider.close()
    # A shard is deliberately ineligible for promotion by itself.  Promotion is
    # evaluated only after the merge verifies exact coverage of the sealed
    # full-validation manifest.
    promotion = _promotion_gate(
        reports,
        dict(gate),
        all_validation=complete_evaluation_split is not None and shard_count == 1,
    )
    evaluation_shard = {
        "index": shard_index,
        "count": shard_count,
        "complete_evaluation_requested": complete_evaluation_split is not None,
        "evaluation_split": selection_split,
    }
    summary = {
        "records": reports,
        "family_count": {
            family: sum(item["medium_type"] == family for item in reports)
            for family in ALLOWED_MEDIUM_TYPES
        },
        "basis": basis_info,
        "future_truth_opened_only_after_seal": True,
        "evaluation_shard": evaluation_shard,
        # Retain the old key so existing validation merge tooling stays compatible.
        "validation_shard": {
            **evaluation_shard,
            "full_validation_requested": bool(all_validation),
        },
        "selection_split": selection_split,
        "risk_calibration_requested": calibration_per_family is not None,
        "basis_training_per_family": resolved_basis_training_per_family,
        "promotion_gate": promotion,
    }
    write_report(summary, output / "summary.json")
    write_report(
        {
            "status": "complete",
            "schema": "cpadc_same_protocol_terminal_v1",
            "same_protocol_validation_passed": bool(promotion["passed"]),
            "same_protocol_evaluation_passed": bool(promotion["passed"]),
            "evaluation_split": selection_split,
            "promotion_gate": promotion,
            "basis": basis_info,
            "claim": (
                "same-protocol CPADC gate passed"
                if promotion["passed"]
                else "evaluation complete; accuracy promotion not authorized"
            ),
        },
        output / "terminal.json",
    )
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--basis-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adaptation-device", default="cpu")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--parent-checkpoint-identity")
    parser.add_argument("--travel-time-h5")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--all-validation", action="store_true")
    parser.add_argument("--all-test-id", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--calibration-per-family", type=int)
    parser.add_argument("--calibration-seed", type=int)
    parser.add_argument("--basis-training-per-family", type=int)
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_exists": Path(args.config).is_file(),
                    "basis_checkpoint_exists": Path(args.basis_checkpoint).is_file(),
                    "future_truth_opened": False,
                    "allowed_true_snapshot_count": 2,
                    "gpu_launch_authorized": False,
                    "parent_inference_device": args.device,
                    "adaptation_device": args.adaptation_device,
                    "basis_training_per_family": args.basis_training_per_family,
                },
                sort_keys=True,
            )
        )
        return 0
    run(
        args.config,
        basis_checkpoint=args.basis_checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        adaptation_device_name=args.adaptation_device,
        sample_ids=None if args.sample_id is None else tuple(args.sample_id),
        per_family=args.per_family,
        all_validation=args.all_validation,
        all_test_id=args.all_test_id,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        calibration_per_family=args.calibration_per_family,
        calibration_seed=args.calibration_seed,
        basis_training_per_family=args.basis_training_per_family,
        save_fields=not args.no_fields,
        parent_checkpoint=args.parent_checkpoint,
        parent_checkpoint_identity=args.parent_checkpoint_identity,
        travel_time_h5=args.travel_time_h5,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
