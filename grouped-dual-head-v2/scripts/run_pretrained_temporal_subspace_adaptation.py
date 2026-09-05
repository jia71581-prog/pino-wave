#!/usr/bin/env python3
"""Train-only pilot for fast causal adaptation in a pretrained A3 subspace.

The frozen parent supplies 32 query-invariant spatial anchors and continuous
time coefficients.  Deployment solves only one gain per pretrained mode with
the existing source-consistent convex defect objective.  No future frame is
opened until the adapted prediction has been sealed on disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.bridge import make_onset_bridge
from saved_time_phase_operator_v4.instance_adaptation.data_guard import (
    GuardedOnsetDataset,
)
from saved_time_phase_operator_v4.instance_adaptation.defect_correction import (
    DefectCorrectionWeights,
    solve_causal_defect_correction,
)
from saved_time_phase_operator_v4.instance_adaptation.forced_defect import (
    lwc84_discrete_defect,
)
from saved_time_phase_operator_v4.instance_adaptation.losses import (
    build_rad_physics_points,
)
from saved_time_phase_operator_v4.instance_adaptation.pretrained_subspace import (
    TemporalLatentInputCapture,
    pretrained_temporal_latent_basis,
    pretrained_temporal_latent_probe_basis,
    temporal_latent_state_sha256,
    train_only_scalar_multiplier_oracle,
)
from scripts.evaluate_v5_instance_adaptation import (
    evaluate_after_adaptation,
    write_report,
)
from scripts.run_v5_instance_adaptation import (
    _load_background_provider,
    _load_parent,
    _predict_parent,
    _read_future_truth,
    _write_manifest,
)
from scripts.train_v5_feature_meta import build_balanced_meta_episodes
from scripts.train_v5_residual_meta import _sha256


PTLSA_SCHEMA = "pretrained_temporal_latent_subspace_adaptation_v1"
PTLSA_RISK_SCHEMA = "pretrained_temporal_subspace_norm_risk_calibration_v1"
PTLSA_RESCUE_RISK_SCHEMA = (
    "pretrained_temporal_subspace_linear_rescue_risk_calibration_v1"
)
PTLSA_RESCUE_MODEL_SCHEMA = "standardized_ridge_relative_improvement_v1"
PTLSA_RESCUE_FEATURES = (
    "solver_coefficient_l2_norm",
    "objective_relative_improvement",
)


def _checkpoint_exclusions(paths: tuple[str, ...]) -> set[str]:
    """Read train IDs used by prior calibration checkpoints without mutating them."""

    excluded: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        calibration = payload.get("risk_calibration") or {}
        excluded.update(str(value) for value in calibration.get("sample_ids", ()))
    return excluded


def build_train_pilot_manifest(
    manifest,
    *,
    per_family: int,
    seed: int,
    excluded_sample_ids: set[str],
    prior_basis_per_family: int,
    prior_basis_seed: int,
):
    """Select balanced train records disjoint from prior CPADC fitting/calibration."""

    prior_basis = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=int(prior_basis_per_family),
        seed=int(prior_basis_seed),
    )
    excluded = set(excluded_sample_ids)
    excluded.update(row.sample_id for row in prior_basis)
    allowed = {
        row.sample_id
        for row in manifest.records
        if row.split == "train" and row.sample_id not in excluded
    }
    rows = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=int(per_family),
        seed=int(seed),
        allowed_sample_ids=allowed,
    )
    if excluded.intersection(row.sample_id for row in rows):
        raise RuntimeError("PTLSA pilot overlaps prior fitting or calibration records")
    return rows, excluded


def _resolve_temporal_latent(parent: torch.nn.Module) -> torch.nn.Module:
    local_field = getattr(parent, "local_field", None)
    module = None if local_field is None else getattr(local_field, "temporal_latent", None)
    if module is None:
        raise ValueError("the selected parent has no A3 temporal-latent module")
    required = ("anchor_projection", "time_features", "time_trunk", "rank")
    if any(not hasattr(module, name) for name in required):
        raise ValueError("the parent temporal-latent module has an incompatible contract")
    return module


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_risk_calibration(
    path: str | Path,
    *,
    parent_identity: dict[str, object],
    enabled_families: set[str],
) -> dict[str, object]:
    checkpoint = Path(path).expanduser().resolve()
    payload = json.loads(checkpoint.read_text())
    schema = str(payload.get("schema", ""))
    if schema not in {PTLSA_RISK_SCHEMA, PTLSA_RESCUE_RISK_SCHEMA}:
        raise ValueError("PTLSA risk calibration schema mismatch")
    if payload.get("future_truth_scope") != "train_split_after_adaptation_seal_only":
        raise ValueError("PTLSA risk calibration provenance mismatch")
    if dict(payload.get("parent") or {}) != parent_identity:
        raise ValueError("PTLSA risk calibration parent identity mismatch")
    family = str(payload.get("family", ""))
    if family not in enabled_families:
        raise ValueError("PTLSA risk calibration family is not enabled")
    threshold = float(
        payload.get("maximum_coefficient_l2_norm", 0.0)
        if schema == PTLSA_RISK_SCHEMA
        else dict(payload.get("base_gate") or {}).get(
            "maximum_coefficient_l2_norm", 0.0
        )
    )
    if not torch.isfinite(torch.tensor(threshold)).item() or threshold <= 0.0:
        raise ValueError("PTLSA risk calibration threshold is invalid")
    if int(payload.get("record_count", 0)) <= 0:
        raise ValueError("PTLSA risk calibration is empty")
    identity: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "schema": schema,
        "family": family,
        "maximum_coefficient_l2_norm": threshold,
        "record_count": int(payload["record_count"]),
    }
    if schema == PTLSA_RESCUE_RISK_SCHEMA:
        rescue = dict(payload.get("rescue_model") or {})
        if rescue.get("schema") != PTLSA_RESCUE_MODEL_SCHEMA:
            raise ValueError("PTLSA rescue model schema mismatch")
        features = tuple(str(value) for value in rescue.get("features") or ())
        means = tuple(float(value) for value in rescue.get("feature_means") or ())
        scales = tuple(float(value) for value in rescue.get("feature_scales") or ())
        weights = tuple(float(value) for value in rescue.get("weights") or ())
        minimum_score = float(rescue.get("minimum_score", float("nan")))
        finite = lambda values: all(
            torch.isfinite(torch.tensor(value)).item() for value in values
        )
        if (
            features != PTLSA_RESCUE_FEATURES
            or len(means) != len(features)
            or len(scales) != len(features)
            or len(weights) != len(features) + 1
            or not finite(means)
            or not finite(scales)
            or not finite(weights)
            or any(value <= 0.0 for value in scales)
            or not torch.isfinite(torch.tensor(minimum_score)).item()
        ):
            raise ValueError("PTLSA rescue model parameters are invalid")
        identity["rescue_model"] = {
            "schema": PTLSA_RESCUE_MODEL_SCHEMA,
            "features": list(features),
            "feature_means": list(means),
            "feature_scales": list(scales),
            "weights": list(weights),
            "minimum_score": minimum_score,
        }
    return identity


def _risk_decision(
    risk_identity: dict[str, object] | None,
    *,
    family: str,
    solver_coefficient_l2_norm: float,
    objective_before: float,
    objective_after: float,
) -> tuple[bool, dict[str, object] | None]:
    """Apply a sealed online-only risk rule without consulting future truth."""

    if risk_identity is None or family != str(risk_identity["family"]):
        return True, None
    coefficient_norm = float(solver_coefficient_l2_norm)
    base_safe = coefficient_norm <= float(
        risk_identity["maximum_coefficient_l2_norm"]
    )
    decision: dict[str, object] = {
        "base_safe": base_safe,
        "solver_coefficient_l2_norm": coefficient_norm,
        "route": "base" if base_safe else "abstain",
    }
    if str(risk_identity["schema"]) == PTLSA_RISK_SCHEMA:
        return base_safe, decision

    rescue = dict(risk_identity["rescue_model"])
    objective_relative_improvement = (
        float(objective_before) - float(objective_after)
    ) / max(abs(float(objective_before)), 1.0e-12)
    raw_features = (coefficient_norm, objective_relative_improvement)
    means = tuple(float(value) for value in rescue["feature_means"])
    scales = tuple(float(value) for value in rescue["feature_scales"])
    weights = tuple(float(value) for value in rescue["weights"])
    score = weights[0] + sum(
        weight * ((value - mean) / scale)
        for weight, value, mean, scale in zip(
            weights[1:], raw_features, means, scales
        )
    )
    rescue_safe = score >= float(rescue["minimum_score"])
    decision.update(
        {
            "objective_relative_improvement": objective_relative_improvement,
            "rescue_safe": rescue_safe,
            "rescue_score": score,
            "route": "base" if base_safe else "rescue" if rescue_safe else "abstain",
        }
    )
    return base_safe or rescue_safe, decision


def _aggregate(reports: list[dict[str, object]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for report in reports:
        parent_error = float(report["parent_future_fullfield_relative_l2"])
        adapted_error = float(report["future_fullfield_relative_l2"])
        rows.append(
            {
                "sample_id": str(report["sample_id"]),
                "family": str(report["medium_type"]),
                "parent_error": parent_error,
                "adapted_error": adapted_error,
                "relative_improvement": (
                    (parent_error - adapted_error) / max(parent_error, 1.0e-12)
                ),
                "nonworse": adapted_error <= parent_error * (1.0 + 1.0e-9),
                "accepted": bool(dict(report["adaptation"])["accepted"]),
                "adaptation_elapsed_s": float(
                    dict(report["adaptation"])["adaptation_elapsed_s"]
                ),
            }
        )
    family_metrics: dict[str, dict[str, object]] = {}
    for family in ALLOWED_MEDIUM_TYPES:
        selected = [row for row in rows if row["family"] == family]
        if not selected:
            continue
        parent_mean = sum(float(row["parent_error"]) for row in selected) / len(selected)
        adapted_mean = sum(float(row["adapted_error"]) for row in selected) / len(selected)
        family_metrics[family] = {
            "record_count": len(selected),
            "mean_parent_error": parent_mean,
            "mean_adapted_error": adapted_mean,
            "ratio_of_mean_improvement": (
                (parent_mean - adapted_mean) / max(parent_mean, 1.0e-12)
            ),
            "nonworse_fraction": sum(bool(row["nonworse"]) for row in selected)
            / len(selected),
        }
    return {
        "record_count": len(rows),
        "mean_per_record_relative_improvement": sum(
            float(row["relative_improvement"]) for row in rows
        )
        / len(rows),
        "ratio_of_mean_improvement": (
            sum(float(row["parent_error"]) for row in rows)
            - sum(float(row["adapted_error"]) for row in rows)
        )
        / max(sum(float(row["parent_error"]) for row in rows), 1.0e-12),
        "nonworse_fraction": sum(bool(row["nonworse"]) for row in rows) / len(rows),
        "acceptance_fraction": sum(bool(row["accepted"]) for row in rows) / len(rows),
        "maximum_adaptation_elapsed_s": max(
            float(row["adaptation_elapsed_s"]) for row in rows
        ),
        "families": family_metrics,
        "records": rows,
    }


def _load_resumable_report(
    artifact: Path,
    record,
    *,
    parent_identity: dict[str, object],
    risk_identity: dict[str, object] | None,
    observed_probe_weight: float,
    enabled_families: set[str],
    causal_ramp_steps: int,
    maximum_correction_ratio: float,
) -> dict[str, object] | None:
    """Load one completely sealed/evaluated record without reopening truth."""

    adaptation_path = artifact / "adaptation.pt"
    evaluation_path = artifact / "evaluation.json"
    present = (adaptation_path.is_file(), evaluation_path.is_file())
    if present == (False, False):
        return None
    if present != (True, True):
        raise RuntimeError(
            f"incomplete resumable record {record.sample_id}; preserve or archive its "
            "artifact directory before retrying"
        )
    sealed = torch.load(adaptation_path, map_location="cpu", weights_only=False)
    report = json.loads(evaluation_path.read_text())
    sealed_adaptation = dict(sealed.get("adaptation") or {})
    report_adaptation = dict(report.get("adaptation") or {})
    expected = {
        "schema": PTLSA_SCHEMA,
        "sample_id": record.sample_id,
        "input_digest": record.input_digest,
        "parent": parent_identity,
        "risk_calibration": risk_identity,
        "observed_probe_weight": float(observed_probe_weight),
        "apply_families": tuple(sorted(enabled_families)),
        "causal_ramp_steps": int(causal_ramp_steps),
        "maximum_correction_ratio": float(maximum_correction_ratio),
        "accessed_true_indices": tuple(record.observed_indices),
        "future_truth_used": False,
    }
    for key, value in expected.items():
        for source, payload in (
            ("adaptation.pt", sealed_adaptation),
            ("evaluation.json", report_adaptation),
        ):
            actual = payload.get(key)
            if isinstance(value, tuple):
                actual = tuple(actual or ())
            if actual != value:
                raise RuntimeError(
                    f"resume identity mismatch for {record.sample_id} in {source}: {key}"
                )
    if str(report.get("sample_id")) != record.sample_id:
        raise RuntimeError(f"resume evaluation sample mismatch for {record.sample_id}")
    if str(report.get("medium_type")) != record.medium_type:
        raise RuntimeError(f"resume evaluation family mismatch for {record.sample_id}")
    return report


def run(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    device_name: str = "cuda",
    sample_ids: tuple[str, ...] | None = None,
    per_family: int = 1,
    pilot_seed: int = 20_372,
    exclude_checkpoints: tuple[str, ...] = (),
    save_fields: bool = True,
    scalar_oracle_diagnostic: bool = False,
    observed_probe_weight: float | None = None,
    apply_families: tuple[str, ...] | None = None,
    all_validation: bool = False,
    all_test_id: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    train_family: str | None = None,
    train_range_start: int = 0,
    train_range_stop: int | None = None,
    risk_calibration: str | Path | None = None,
    travel_time_h5: str | Path | None = None,
    parent_checkpoint: str | Path | None = None,
    resume: bool = False,
) -> list[dict[str, object]]:
    config_file = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_file.read_text())
    if not isinstance(config, dict):
        raise ValueError("PTLSA config must contain a mapping")
    if travel_time_h5 is not None:
        config["travel_time_h5"] = str(Path(travel_time_h5).expanduser().resolve())
    if parent_checkpoint is not None:
        config["parent_checkpoint"] = str(
            Path(parent_checkpoint).expanduser().resolve()
        )
    manifest = build_manifest(config["source_h5"])
    enabled_families = (
        set(ALLOWED_MEDIUM_TYPES)
        if apply_families is None
        else {str(value) for value in apply_families}
    )
    if not enabled_families or not enabled_families.issubset(ALLOWED_MEDIUM_TYPES):
        raise ValueError("apply families must be a nonempty subset of registered families")
    if shard_count < 1 or not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard index/count are invalid")
    if all_validation and all_test_id:
        raise ValueError("complete validation and test_id are mutually exclusive")
    complete_evaluation_split = (
        "validation" if all_validation else "test_id" if all_test_id else None
    )
    if complete_evaluation_split is not None and sample_ids:
        raise ValueError("complete evaluation cannot be combined with sample IDs")
    if complete_evaluation_split is not None and train_family is not None:
        raise ValueError("complete evaluation cannot use a train family range")
    if sample_ids and train_family is not None:
        raise ValueError("sample IDs and train family range are mutually exclusive")
    if complete_evaluation_split is None and (
        int(shard_index) != 0 or int(shard_count) != 1
    ):
        raise ValueError("sharding is reserved for complete evaluation")
    if complete_evaluation_split is not None and scalar_oracle_diagnostic:
        raise ValueError("train-only scalar oracle is forbidden on evaluation")
    excluded = _checkpoint_exclusions(exclude_checkpoints)
    cp_settings = config.get("cpadc", {}) or {}
    prior_basis_per_family = int(cp_settings.get("per_family", 32))
    prior_basis_seed = int(config.get("seed", 372))
    selection_split = complete_evaluation_split or "train"
    if complete_evaluation_split is not None:
        complete_rows = tuple(
            row for row in manifest.records if row.split == complete_evaluation_split
        )
        if not complete_rows or {row.medium_type for row in complete_rows} != set(
            ALLOWED_MEDIUM_TYPES
        ):
            raise ValueError("complete evaluation manifest is invalid")
        rows = complete_rows[int(shard_index) :: int(shard_count)]
    elif sample_ids:
        by_sample = {
            row.sample_id: row
            for row in manifest.records
            if row.split == "train"
        }
        missing = tuple(value for value in sample_ids if value not in by_sample)
        if missing:
            raise ValueError(f"selected train samples are missing: {missing[:3]}")
        prior_basis = build_balanced_meta_episodes(
            manifest,
            split="train",
            per_family=prior_basis_per_family,
            seed=prior_basis_seed,
        )
        excluded.update(row.sample_id for row in prior_basis)
        overlap = excluded.intersection(sample_ids)
        if overlap:
            raise ValueError(f"selected pilot samples are excluded: {sorted(overlap)[:3]}")
        rows = tuple(by_sample[value] for value in sample_ids)
    elif train_family is not None:
        family = str(train_family)
        if family not in ALLOWED_MEDIUM_TYPES:
            raise ValueError("train family is not registered")
        start = int(train_range_start)
        stop = int(train_range_stop) if train_range_stop is not None else start + 1
        if start < 0 or stop <= start:
            raise ValueError("train family range is invalid")
        complete_rows, excluded = build_train_pilot_manifest(
            manifest,
            per_family=stop,
            seed=int(pilot_seed),
            excluded_sample_ids=excluded,
            prior_basis_per_family=prior_basis_per_family,
            prior_basis_seed=prior_basis_seed,
        )
        family_rows = tuple(row for row in complete_rows if row.medium_type == family)
        rows = family_rows[start:stop]
        if len(rows) != stop - start:
            raise RuntimeError("train family range selection is incomplete")
    else:
        rows, excluded = build_train_pilot_manifest(
            manifest,
            per_family=int(per_family),
            seed=int(pilot_seed),
            excluded_sample_ids=excluded,
            prior_basis_per_family=prior_basis_per_family,
            prior_basis_seed=prior_basis_seed,
        )

    output = Path(output_dir).expanduser().resolve()
    if (output / "summary.json").exists() or (output / "terminal.json").exists():
        raise FileExistsError("refusing to overwrite an existing completed PTLSA run")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "instance_manifest.json"
    expected_manifest = [row.__dict__ for row in rows]
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("partial PTLSA output requires explicit --resume")
        if json.loads(manifest_path.read_text()) != expected_manifest:
            raise RuntimeError("resume manifest does not match the requested shard")
    else:
        _write_manifest(rows, manifest_path)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    parent, normalizer = _load_parent(config, manifest, device)
    provider = _load_background_provider(config, (row.sample_id for row in rows))
    temporal_latent = _resolve_temporal_latent(parent)
    temporal_hash = temporal_latent_state_sha256(temporal_latent)
    parent_checkpoint = Path(str(config["parent_checkpoint"])).expanduser().resolve()
    parent_identity = {
        "checkpoint": str(parent_checkpoint),
        "checkpoint_sha256": _sha256(parent_checkpoint),
        "manifest_digest": manifest.digest,
        "temporal_latent_state_sha256": temporal_hash,
        "temporal_latent_rank": int(temporal_latent.rank),
    }
    risk_identity = (
        None
        if risk_calibration is None
        else _load_risk_calibration(
            risk_calibration,
            parent_identity=parent_identity,
            enabled_families=enabled_families,
        )
    )
    dataset = GuardedOnsetDataset(
        config["source_h5"],
        manifest,
        split=selection_split,
        sample_ids=tuple(row.sample_id for row in rows),
        travel_time_h5=config.get("travel_time_h5"),
    )
    inner_raw = config.get("inner_weights", {}) or {}
    resolved_observed_weight = (
        float(inner_raw.get("observed", 0.0))
        if observed_probe_weight is None
        else float(observed_probe_weight)
    )
    if resolved_observed_weight < 0.0:
        raise ValueError("observed probe weight must be nonnegative")
    weights = DefectCorrectionWeights(
        defect=float(inner_raw.get("defect", 1.0)),
        observed=resolved_observed_weight,
        bridge=float(inner_raw.get("bridge", 0.5)),
        prior=float(inner_raw.get("prior", 1.0e-4)),
    )
    gate = config.get("deployment_gate", {}) or {}
    dx = float(config.get("dx_m", 10.0))
    dz = float(config.get("dz_m", 10.0))
    pressure_scale = float(normalizer.metadata.pressure_scale_pa)
    method_identity = {
        "schema": PTLSA_SCHEMA,
        "config": str(config_file),
        "config_sha256": _sha256(config_file),
        "coefficient_count": int(temporal_latent.rank),
        "observed_probe_weight": resolved_observed_weight,
        "defect_weight": float(weights.defect),
        "bridge_weight": float(weights.bridge),
        "prior_precision": float(weights.prior),
        "causal_ramp_steps": int(config.get("causal_ramp_steps", 4)),
        "maximum_correction_ratio": float(gate.get("maximum_correction_ratio", 0.10)),
        "apply_families": tuple(sorted(enabled_families)),
        "risk_calibration": risk_identity,
        "travel_time_h5": config.get("travel_time_h5"),
    }
    reports: list[dict[str, object]] = []
    resumed_record_count = 0
    try:
        for record in dataset:
            artifact = output / record.sample_id
            if resume:
                resumed_report = _load_resumable_report(
                    artifact,
                    record,
                    parent_identity=parent_identity,
                    risk_identity=risk_identity,
                    observed_probe_weight=resolved_observed_weight,
                    enabled_families=enabled_families,
                    causal_ramp_steps=int(config.get("causal_ramp_steps", 4)),
                    maximum_correction_ratio=float(
                        gate.get("maximum_correction_ratio", 0.10)
                    ),
                )
                if resumed_report is not None:
                    reports.append(resumed_report)
                    resumed_record_count += 1
                    continue
            record_started = time.perf_counter()
            record.audit.read(record.observed_indices)
            velocity = record.velocity_mps.to(device).unsqueeze(0)
            source = record.source_parameters.to(device).unsqueeze(0)
            source_map = record.source_map.to(device).unsqueeze(0)
            times = record.time_s.to(device)
            observed = normalizer.encode_pressure(
                record.observed_wavefield.to(device).unsqueeze(0), source[:, 4]
            )
            with TemporalLatentInputCapture(temporal_latent) as capture:
                parent_normalized = _predict_parent(
                    parent,
                    normalizer,
                    record,
                    device,
                    normalized=True,
                    background_provider=provider,
                )
            captured = capture.finalize(expected_time_s=times)
            _synchronize(device)
            adaptation_started = time.perf_counter()
            bridge = make_onset_bridge(
                velocity,
                source,
                record.observed_wavefield.to(device).unsqueeze(0),
                record.observed_indices,
                times,
                source_map=source_map,
                steps=int(config.get("bridge_steps", 4)),
                dx_m=dx,
                dz_m=dz,
                device=device,
            )
            if not bridge.valid:
                raise RuntimeError(
                    f"synthetic bridge failed for {record.sample_id}: "
                    f"{bridge.failure_reason}"
                )
            bridge_normalized = normalizer.encode_pressure(bridge.frames, source[:, 4])
            dt = float((times[1:] - times[:-1]).mean())
            field_scale_pa = source[:, 4] * pressure_scale
            with torch.no_grad():
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
                basis = pretrained_temporal_latent_basis(
                    temporal_latent,
                    captured,
                    parent_normalized,
                    observed,
                    record.observed_indices,
                    ramp_steps=int(config.get("causal_ramp_steps", 4)),
                )
                observed_probe_basis = (
                    None
                    if resolved_observed_weight == 0.0
                    else pretrained_temporal_latent_probe_basis(basis)
                )
                points = build_rad_physics_points(
                    parent_defect / defect_scale[:, None, None, None],
                    record.observed_indices,
                    count=int(cp_settings.get("physics_point_count", 128)),
                    k=float(config.get("rad_k", 1.0)),
                    c=float(config.get("rad_c", 1.0)),
                    time_tilt=float(config.get("rad_time_tilt", 1.5)),
                    generator=torch.Generator(device=device).manual_seed(
                        int(config.get("seed", 372)) + int(record.source_index)
                    ),
                )
                result = solve_causal_defect_correction(
                    basis,
                    parent_normalized,
                    velocity,
                    source,
                    source_map,
                    times,
                    record.observed_indices,
                    points,
                    field_scale_pa=field_scale_pa,
                    observed_wavefield=observed,
                    observed_design_basis=observed_probe_basis,
                    bridge_wavefield=bridge_normalized,
                    bridge_indices=bridge.time_indices,
                    weights=weights,
                    dt=dt,
                    dx=dx,
                    dz=dz,
                    spatial_stride=int(config.get("inner_spatial_stride", 4)),
                    rank_chunk_size=int(cp_settings.get("rank_chunk_size", 1)),
                    minimum_relative_improvement=float(
                        gate.get("minimum_causal_objective_improvement", 1.0e-3)
                    ),
                    maximum_condition_number=float(
                        gate.get("maximum_condition_number", 1.0e8)
                    ),
                    maximum_correction_ratio=float(
                        gate.get("maximum_correction_ratio", 0.10)
                    ),
                    minimum_unconstrained_correction_ratio=0.0,
                )
            _synchronize(device)
            adaptation_elapsed = time.perf_counter() - adaptation_started
            total_elapsed = time.perf_counter() - record_started
            family_enabled = record.medium_type in enabled_families
            solver_coefficient_l2_norm = float(result.coefficients[0].double().norm())
            risk_safe, risk_decision = _risk_decision(
                risk_identity,
                family=record.medium_type,
                solver_coefficient_l2_norm=solver_coefficient_l2_norm,
                objective_before=float(result.objective_before[0]),
                objective_after=float(result.objective_after[0]),
            )
            deployment_accepted = (
                bool(result.accepted[0]) and family_enabled and risk_safe
            )
            deployment_coefficients = (
                result.coefficients
                if deployment_accepted
                else torch.zeros_like(result.coefficients)
            )
            deployed_normalized = parent_normalized + basis.combine(
                deployment_coefficients
            )
            parent_field = normalizer.decode_pressure(parent_normalized, source[:, 4])
            adapted_field = normalizer.decode_pressure(deployed_normalized, source[:, 4])
            artifact.mkdir(parents=True, exist_ok=True)
            adaptation_payload = {
                "schema": PTLSA_SCHEMA,
                "sample_id": record.sample_id,
                "input_digest": record.input_digest,
                "accepted": deployment_accepted,
                "solver_accepted": bool(result.accepted[0]),
                "rollback_reason": (
                    "pre_registered_family_abstention"
                    if not family_enabled
                    else (
                        "calibrated_coefficient_norm_abstention"
                        if not risk_safe
                        and (
                            risk_identity is None
                            or str(risk_identity["schema"]) == PTLSA_RISK_SCHEMA
                        )
                        else "calibrated_linear_rescue_abstention"
                        if not risk_safe
                        else result.rollback_reasons[0]
                    )
                ),
                "objective_before": float(result.objective_before[0]),
                "objective_after": float(
                    result.objective_after[0]
                    if deployment_accepted
                    else result.objective_before[0]
                ),
                "condition_number": float(result.condition_number[0]),
                "correction_ratio": float(
                    result.correction_ratio[0] if deployment_accepted else 0.0
                ),
                "unconstrained_correction_ratio": float(
                    result.unconstrained_correction_ratio[0]
                ),
                "projection_scale": float(result.projection_scale[0]),
                "adaptation_elapsed_s": adaptation_elapsed,
                "total_inference_elapsed_s": total_elapsed,
                "coefficient_count": int(result.coefficients.shape[1]),
                "coefficients": deployment_coefficients[0].detach().cpu(),
                "solver_coefficients": result.coefficients[0].detach().cpu(),
                "solver_coefficient_l2_norm": solver_coefficient_l2_norm,
                "risk_calibration": risk_identity,
                "risk_decision": risk_decision,
                "accessed_true_indices": tuple(record.audit.requested_indices),
                "future_truth_used": bool(record.audit.payload()["future_truth_used"]),
                "parent": parent_identity,
                "causal_ramp_steps": int(config.get("causal_ramp_steps", 4)),
                "bridge_indices": bridge.time_indices,
                "maximum_correction_ratio": float(
                    gate.get("maximum_correction_ratio", 0.10)
                ),
                "observed_probe_weight": resolved_observed_weight,
                "apply_families": tuple(sorted(enabled_families)),
            }
            sealed: dict[str, object] = {"adaptation": adaptation_payload}
            if save_fields:
                sealed.update(
                    {
                        "parent_field": parent_field.detach().cpu(),
                        "adapted_field": adapted_field.detach().cpu(),
                    }
                )
            # This write is the deployment boundary.  Truth is opened only below.
            torch.save(sealed, artifact / "adaptation.pt")

            truth = _read_future_truth(config["source_h5"], record.source_index)
            report = evaluate_after_adaptation(
                {
                    "parent_field": parent_field.detach().cpu(),
                    "adapted_field": adapted_field.detach().cpu(),
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
            if scalar_oracle_diagnostic:
                oracle = train_only_scalar_multiplier_oracle(
                    parent_field.detach().cpu(),
                    adapted_field.detach().cpu(),
                    truth.unsqueeze(0),
                    record.observed_indices,
                )
                parent_error = float(oracle.parent_relative_l2[0])
                best_error = float(oracle.best_relative_l2[0])
                report["train_only_scalar_oracle"] = {
                    "future_truth_scope": "train_split_after_adaptation_seal_only",
                    "multiplier_grid": oracle.multiplier_grid.cpu().tolist(),
                    "relative_l2": oracle.relative_l2[0].cpu().tolist(),
                    "best_multiplier": float(oracle.best_multiplier[0]),
                    "parent_relative_l2": parent_error,
                    "best_relative_l2": best_error,
                    "best_relative_improvement": (
                        (parent_error - best_error) / max(parent_error, 1.0e-12)
                    ),
                }
            write_report(report, artifact / "evaluation.json")
            reports.append(report)
    finally:
        dataset.close()
        if provider is not None:
            provider.close()

    pilot_metrics = _aggregate(reports)
    summary = {
        "schema": PTLSA_SCHEMA,
        "selection_split": selection_split,
        "selection_seed": int(pilot_seed),
        "prior_basis_exclusion_count": prior_basis_per_family * len(ALLOWED_MEDIUM_TYPES),
        "total_exclusion_count": len(excluded),
        "parent": parent_identity,
        "method": method_identity,
        "resumed_record_count": resumed_record_count,
        "future_truth_opened_only_after_seal": True,
        "train_only_scalar_oracle_requested": bool(scalar_oracle_diagnostic),
        "apply_families": tuple(sorted(enabled_families)),
        "claim_authorized": False,
        "claim": (
            "complete validation shard; merge required for promotion"
            if complete_evaluation_split is not None
            else "train-only pilot; same-protocol accuracy promotion not authorized"
        ),
        "evaluation_shard": {
            "index": int(shard_index),
            "count": int(shard_count),
            "complete_evaluation_requested": complete_evaluation_split is not None,
            "evaluation_split": selection_split,
        },
        "pilot_metrics": pilot_metrics,
        "records": reports,
    }
    write_report(summary, output / "summary.json")
    write_report(
        {
            "status": "complete",
            "schema": PTLSA_SCHEMA,
            "selection_split": selection_split,
            "same_protocol_validation_passed": False,
            "claim_authorized": False,
            "claim": (
                "validation shard complete; exact-coverage merge required"
                if complete_evaluation_split is not None
                else "train-only pilot complete; validation remains unopened"
            ),
            "pilot_metrics": pilot_metrics,
            "parent": parent_identity,
            "method": method_identity,
        },
        output / "terminal.json",
    )
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--per-family", type=int, default=1)
    parser.add_argument("--pilot-seed", type=int, default=20_372)
    parser.add_argument("--exclude-checkpoint", action="append", default=[])
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--train-only-scalar-oracle", action="store_true")
    parser.add_argument("--observed-probe-weight", type=float)
    parser.add_argument("--apply-family", action="append")
    parser.add_argument("--all-validation", action="store_true")
    parser.add_argument("--all-test-id", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--train-family", choices=ALLOWED_MEDIUM_TYPES)
    parser.add_argument("--train-range-start", type=int, default=0)
    parser.add_argument("--train-range-stop", type=int)
    parser.add_argument("--risk-calibration")
    parser.add_argument("--travel-time-h5")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_exists": Path(args.config).is_file(),
                    "exclude_checkpoints_exist": all(
                        Path(value).is_file() for value in args.exclude_checkpoint
                    ),
                    "selection_split": (
                        "validation"
                        if args.all_validation
                        else "test_id"
                        if args.all_test_id
                        else "train"
                    ),
                    "future_truth_opened": False,
                    "allowed_true_snapshot_count": 2,
                    "gpu_launch_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    run(
        args.config,
        output_dir=args.output_dir,
        device_name=args.device,
        sample_ids=None if args.sample_id is None else tuple(args.sample_id),
        per_family=args.per_family,
        pilot_seed=args.pilot_seed,
        exclude_checkpoints=tuple(args.exclude_checkpoint),
        save_fields=not args.no_fields,
        scalar_oracle_diagnostic=args.train_only_scalar_oracle,
        observed_probe_weight=args.observed_probe_weight,
        apply_families=(
            None if args.apply_family is None else tuple(args.apply_family)
        ),
        all_validation=args.all_validation,
        all_test_id=args.all_test_id,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        train_family=args.train_family,
        train_range_start=args.train_range_start,
        train_range_stop=args.train_range_stop,
        risk_calibration=args.risk_calibration,
        travel_time_h5=args.travel_time_h5,
        parent_checkpoint=args.parent_checkpoint,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
