#!/usr/bin/env python3
"""Sealed three-family evaluation of efficient early-feature adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.bridge import make_onset_bridge
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.feature_modulation import (
    EarlyFeatureOnsetAdapter,
    energy_balanced_relative_loss,
    metric_aligned_relative_loss,
)
from saved_time_phase_operator_v4.instance_adaptation.visualization import (
    plot_receiver_comparison,
    plot_wavefield_comparison,
)
from scripts.evaluate_v5_instance_adaptation import (
    evaluate_after_adaptation,
    write_report,
)
from scripts.run_v5_instance_adaptation import (
    _load_parent,
    _predict_parent,
    _read_future_truth,
    build_instance_manifest,
    resolve_saved_time_parent_config,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_adapter(
    parent,
    checkpoint: str | Path,
    *,
    device: torch.device,
    expected_manifest_digest: str,
) -> tuple[EarlyFeatureOnsetAdapter, dict[str, object]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("adapter_kind") != "early_feature_meta":
        raise ValueError("checkpoint is not an early-feature meta adapter")
    if payload.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("feature adapter manifest digest mismatch")
    adapter = EarlyFeatureOnsetAdapter(
        parent,
        latent_dim=int(payload["latent_dim"]),
        lora_rank=int(payload["lora_rank"]),
        max_scale=float(payload["max_scale"]),
        max_bias=float(payload["max_bias"]),
    ).to(device)
    current = adapter.state_dict()
    state = payload.get("adapter_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("feature adapter checkpoint lacks adapter state")
    unknown = tuple(key for key in state if key not in current)
    if unknown:
        raise ValueError(f"feature adapter state contains unknown tensors: {unknown[:3]}")
    current.update({key: value.to(device) for key, value in state.items()})
    adapter.load_state_dict(current, strict=True)
    adapter.eval()
    return adapter, payload


def _predict_feature(
    adapter: EarlyFeatureOnsetAdapter,
    normalizer,
    record,
    device: torch.device,
    *,
    time_s: torch.Tensor,
    latent_delta: torch.Tensor | None = None,
    conditioner_wavefield: torch.Tensor | None = None,
    modulation_gate: torch.Tensor | None = None,
    time_block: int = 1,
) -> torch.Tensor:
    prepared, _ = adapter.prepare_sources(
        record.velocity_mps.to(device).unsqueeze(0),
        record.source_parameters.to(device).unsqueeze(0),
        record.source_map.to(device).unsqueeze(0),
        record.observed_wavefield.to(device).unsqueeze(0),
        normalizer,
        latent_delta=latent_delta,
        conditioner_wavefield=conditioner_wavefield,
        modulation_gate=modulation_gate,
    )
    dense_grid = adapter.parent.prepare_dense_grid(
        prepared,
        x_m=record.x_m.to(device),
        z_m=record.z_m.to(device),
        travel_time_s=(
            None
            if record.dense_travel_time_s is None
            else record.dense_travel_time_s.to(device).unsqueeze(0)
        ),
    )
    return adapter.parent.predict_wavefield(
        prepared,
        time_s.to(device),
        dense_grid=dense_grid,
        time_block=int(time_block),
    )


def _refine_latent_with_lwc84(
    adapter: EarlyFeatureOnsetAdapter,
    normalizer,
    record,
    device: torch.device,
    *,
    steps: int,
    learning_rate: float,
    bridge_steps: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Optimize only one tiny code using two true frames plus a synthetic bridge."""

    record.audit.read(record.observed_indices)
    bridge = make_onset_bridge(
        record.velocity_mps.to(device).unsqueeze(0),
        record.source_parameters.to(device).unsqueeze(0),
        record.observed_wavefield.to(device).unsqueeze(0),
        record.observed_indices,
        record.time_s.to(device),
        source_map=record.source_map.to(device).unsqueeze(0),
        steps=int(bridge_steps),
        device=device,
    )
    query_indices = tuple(record.observed_indices) + tuple(bridge.time_indices)
    query_time = record.time_s[list(query_indices)].to(device)
    observed = record.observed_wavefield.to(device).unsqueeze(0)
    latent_dim = int(adapter.conditioner.latent_dim)
    delta = torch.nn.Parameter(torch.zeros(1, latent_dim, device=device))
    for parameter in adapter.adapter_parameters():
        parameter.requires_grad_(False)

    def objective(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed_loss = energy_balanced_relative_loss(
            prediction[:, :2], observed, energy_floor_fraction=0.01
        )
        bridge_loss = prediction.sum() * 0.0
        if bridge.valid and bridge.frames.shape[1]:
            bridge_loss = energy_balanced_relative_loss(
                prediction[:, 2:], bridge.frames, energy_floor_fraction=0.01
            )
        total = observed_loss + 0.1 * bridge_loss + 1.0e-3 * delta.square().mean()
        return total, observed_loss, bridge_loss

    started = time.perf_counter()
    with torch.no_grad():
        baseline_prediction = _predict_feature(
            adapter, normalizer, record, device, time_s=query_time
        )
        _, baseline_observed, baseline_bridge = objective(baseline_prediction)
    optimizer = torch.optim.Adam((delta,), lr=float(learning_rate))
    history: list[dict[str, float | int]] = []
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict_feature(
            adapter,
            normalizer,
            record,
            device,
            time_s=query_time,
            latent_delta=delta,
        )
        loss, observed_loss, bridge_loss = objective(prediction)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite latent-refinement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((delta,), 1.0)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "observed_loss": float(observed_loss.detach()),
                "bridge_loss": float(bridge_loss.detach()),
            }
        )
    with torch.no_grad():
        candidate_prediction = _predict_feature(
            adapter,
            normalizer,
            record,
            device,
            time_s=query_time,
            latent_delta=delta,
        )
        _, candidate_observed, candidate_bridge = objective(candidate_prediction)
    finite = bool(torch.isfinite(delta).all())
    accepted = (
        finite
        and float(delta.norm()) <= 2.0
        and float(candidate_observed) <= float(baseline_observed) * 1.001
        and (
            not bridge.valid
            or float(candidate_bridge) <= float(baseline_bridge) * 1.001
        )
    )
    if not accepted:
        delta = torch.nn.Parameter(torch.zeros_like(delta))
    return delta.detach(), {
        "accepted": accepted,
        "steps": int(steps),
        "elapsed_s": time.perf_counter() - started,
        "trainable_parameter_count": latent_dim,
        "baseline_observed_loss": float(baseline_observed),
        "candidate_observed_loss": float(candidate_observed),
        "baseline_bridge_loss": float(baseline_bridge),
        "candidate_bridge_loss": float(candidate_bridge),
        "latent_norm": float(delta.norm()),
        "bridge_valid": bridge.valid,
        "bridge_failure_reason": bridge.failure_reason,
        "bridge_substeps_per_saved_step": bridge.substeps_per_saved_step,
        "bridge_provenance": bridge.provenance.__dict__,
        "history": history,
        "access_audit": record.audit.payload(),
    }


def _refine_latent_onset_only(
    adapter: EarlyFeatureOnsetAdapter,
    normalizer,
    record,
    device: torch.device,
    *,
    steps: int,
    learning_rate: float,
    conditioner_wavefield: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Fine-tune one latent code using exactly the two permitted true frames."""

    if int(steps) <= 0 or float(learning_rate) <= 0.0:
        raise ValueError("onset-only latent refinement settings must be positive")
    record.audit.read(record.observed_indices)
    query_time = record.time_s[list(record.observed_indices)].to(device)
    observed = record.observed_wavefield.to(device).unsqueeze(0)
    latent_dim = int(adapter.conditioner.latent_dim)
    delta = torch.nn.Parameter(torch.zeros(1, latent_dim, device=device))
    for parameter in adapter.adapter_parameters():
        parameter.requires_grad_(False)

    def objective(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observed_loss = energy_balanced_relative_loss(
            prediction, observed, energy_floor_fraction=0.01
        )
        total = observed_loss + 1.0e-3 * delta.square().mean()
        return total, observed_loss

    started = time.perf_counter()
    with torch.no_grad():
        baseline_prediction = _predict_feature(
            adapter,
            normalizer,
            record,
            device,
            time_s=query_time,
            conditioner_wavefield=conditioner_wavefield,
        )
        _, baseline_observed = objective(baseline_prediction)
    optimizer = torch.optim.Adam((delta,), lr=float(learning_rate))
    history: list[dict[str, float | int]] = []
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict_feature(
            adapter,
            normalizer,
            record,
            device,
            time_s=query_time,
            latent_delta=delta,
            conditioner_wavefield=conditioner_wavefield,
        )
        loss, observed_loss = objective(prediction)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite onset-only latent-refinement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((delta,), 1.0)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "observed_loss": float(observed_loss.detach()),
            }
        )
    with torch.no_grad():
        candidate_prediction = _predict_feature(
            adapter,
            normalizer,
            record,
            device,
            time_s=query_time,
            latent_delta=delta,
            conditioner_wavefield=conditioner_wavefield,
        )
        _, candidate_observed = objective(candidate_prediction)
    finite = bool(torch.isfinite(delta).all())
    accepted = (
        finite
        and float(delta.norm()) <= 2.0
        and float(candidate_observed) <= float(baseline_observed) * 1.001
    )
    if not accepted:
        delta = torch.nn.Parameter(torch.zeros_like(delta))
    return delta.detach(), {
        "mode": "onset_only_no_wavefield_propagator",
        "accepted": accepted,
        "steps": int(steps),
        "elapsed_s": time.perf_counter() - started,
        "trainable_parameter_count": latent_dim,
        "baseline_observed_loss": float(baseline_observed),
        "candidate_observed_loss": float(candidate_observed),
        "latent_norm": float(delta.norm()),
        "physical_wavefield_propagator": False,
        "synthetic_bridge": False,
        "history": history,
        "access_audit": record.audit.payload(),
    }


def _refine_residual_trust_gate(
    adapter: EarlyFeatureOnsetAdapter,
    normalizer,
    record,
    device: torch.device,
    *,
    steps: int,
    learning_rate: float,
    conditioner_wavefield: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Optimize a zero-origin trust gate and tiny latent on two true frames."""

    if int(steps) <= 0 or float(learning_rate) <= 0.0:
        raise ValueError("residual trust-gate settings must be positive")
    condition = torch.as_tensor(
        conditioner_wavefield, dtype=torch.float32, device=device
    )
    observed = record.observed_wavefield.to(device).unsqueeze(0)
    if condition.shape != observed.shape:
        raise ValueError("residual trust gate requires two parent residual frames")
    record.audit.read(record.observed_indices)
    query_time = record.time_s[list(record.observed_indices)].to(device)
    latent_dim = int(adapter.conditioner.latent_dim)
    delta = torch.nn.Parameter(torch.zeros(1, latent_dim, device=device))
    gate = torch.nn.Parameter(torch.zeros(1, device=device))
    for parameter in adapter.adapter_parameters():
        parameter.requires_grad_(False)

    def field_objective(prediction: torch.Tensor) -> torch.Tensor:
        return metric_aligned_relative_loss(prediction, observed)

    started = time.perf_counter()
    with torch.no_grad():
        baseline_prediction = _predict_feature(
            adapter,
            normalizer,
            record,
            device,
            time_s=query_time,
            latent_delta=delta,
            conditioner_wavefield=condition,
            modulation_gate=torch.zeros_like(gate),
        )
        baseline_observed = field_objective(baseline_prediction)
    optimizer = torch.optim.Adam((delta, gate), lr=float(learning_rate))
    history: list[dict[str, float | int]] = []
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        bounded_gate = gate.clamp(0.0, 1.0)
        prediction = _predict_feature(
            adapter,
            normalizer,
            record,
            device,
            time_s=query_time,
            latent_delta=delta,
            conditioner_wavefield=condition,
            modulation_gate=bounded_gate,
        )
        observed_loss = field_objective(prediction)
        loss = (
            observed_loss
            + 1.0e-3 * delta.square().mean()
            + 1.0e-4 * bounded_gate.square().mean()
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite residual trust-gate loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((delta, gate), 1.0)
        optimizer.step()
        with torch.no_grad():
            gate.clamp_(0.0, 1.0)
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "observed_loss": float(observed_loss.detach()),
                "gate": float(gate.detach()),
                "latent_norm": float(delta.detach().norm()),
            }
        )
    with torch.no_grad():
        candidate_prediction = _predict_feature(
            adapter,
            normalizer,
            record,
            device,
            time_s=query_time,
            latent_delta=delta,
            conditioner_wavefield=condition,
            modulation_gate=gate,
        )
        candidate_observed = field_objective(candidate_prediction)
    finite = bool(torch.isfinite(delta).all()) and bool(torch.isfinite(gate).all())
    accepted = (
        finite
        and float(delta.norm()) <= 2.0
        and 0.0 <= float(gate) <= 1.0
        and float(candidate_observed) <= float(baseline_observed) * 0.9999
    )
    if not accepted:
        delta = torch.nn.Parameter(torch.zeros_like(delta))
        gate = torch.nn.Parameter(torch.zeros_like(gate))
    return delta.detach(), gate.detach(), {
        "mode": "parent_residual_zero_origin_trust_gate",
        "accepted": accepted,
        "steps": int(steps),
        "elapsed_s": time.perf_counter() - started,
        "trainable_parameter_count": latent_dim + 1,
        "baseline_observed_loss": float(baseline_observed),
        "candidate_observed_loss": float(candidate_observed),
        "latent_norm": float(delta.norm()),
        "modulation_gate": float(gate),
        "physical_wavefield_propagator": False,
        "synthetic_bridge": False,
        "history": history,
        "access_audit": record.audit.payload(),
    }


def aggregate_future_energy(
    reports: list[dict[str, object]], report_key: str
) -> dict[str, object]:
    """Aggregate sealed future-field squared energies globally and by family."""

    def aggregate(selected: list[dict[str, object]]) -> dict[str, float]:
        truth = sum(
            float(item[report_key]["future_truth_squared_norm"])
            for item in selected
        )
        parent_error = sum(
            float(item[report_key]["future_parent_squared_error"])
            for item in selected
        )
        adapted_error = sum(
            float(item[report_key]["future_adapted_squared_error"])
            for item in selected
        )
        if truth <= 0.0:
            raise ValueError("future truth energy must be positive")
        parent = float(np.sqrt(parent_error / truth))
        adapted = float(np.sqrt(adapted_error / truth))
        return {
            "parent_relative_l2": parent,
            "adapted_relative_l2": adapted,
            "relative_improvement_fraction": (parent - adapted) / parent,
            "future_truth_squared_norm": truth,
            "parent_squared_error": parent_error,
            "adapted_squared_error": adapted_error,
        }

    if not reports:
        raise ValueError("cannot aggregate an empty feature-adaptation report")
    return {
        "global": aggregate(reports),
        "by_family": {
            family: aggregate(
                [item for item in reports if item["medium_type"] == family]
            )
            for family in ALLOWED_MEDIUM_TYPES
        },
    }


def _plot_velocity(record, output: Path) -> None:
    velocity = record.velocity_mps.squeeze(0).cpu().numpy()
    source = record.source_parameters.cpu().numpy()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    fig, axis = plt.subplots(figsize=(5.5, 4.2))
    image = axis.imshow(
        velocity,
        cmap="viridis",
        extent=(float(record.x_m[0]), float(record.x_m[-1]), float(record.z_m[-1]), float(record.z_m[0])),
        aspect="equal",
    )
    axis.scatter(
        [float(source[0])],
        [float(source[1])],
        marker="*",
        s=130,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.8,
        label="Source",
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_title(f"{record.medium_type.capitalize()} velocity model")
    axis.legend(frameon=False, loc="lower right")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Velocity (m/s)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def evaluate(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    device_name: str,
    per_family: int,
    latent_steps: int,
    latent_learning_rate: float,
    bridge_steps: int,
    selection_split: str = "validation",
    selection_seed: int | None = None,
    extra_excluded_group_ids=(),
    refinement_mode: str = "lwc84_bridge",
    summary_only: bool = False,
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    manifest = build_manifest(config["source_h5"])
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    parent, normalizer = _load_parent(config, manifest, device)
    adapter, adapter_payload = _load_adapter(
        parent,
        checkpoint,
        device=device,
        expected_manifest_digest=manifest.digest,
    )
    conditioner_input_kind = str(
        adapter_payload.get("conditioner_input", "raw_observed")
    )
    if conditioner_input_kind not in {"raw_observed", "parent_onset_residual"}:
        raise ValueError("unknown checkpoint conditioner input")
    selection_split = str(selection_split)
    if selection_split not in {"train", "validation"}:
        raise ValueError("feature evaluation split must be train or validation")
    training_group_ids = tuple(
        str(value) for value in adapter_payload.get("training_group_ids", ())
    )
    if selection_split == "train" and not training_group_ids:
        raise ValueError(
            "train-split feature gate requires checkpoint-bound training group IDs"
        )
    extra_excluded = tuple(str(value) for value in extra_excluded_group_ids)
    excluded_group_ids = tuple(dict.fromkeys(training_group_ids + extra_excluded))
    effective_selection_seed = (
        int(config.get("seed", 17)) + 991
        if selection_seed is None
        else int(selection_seed)
    )
    rows = build_instance_manifest(
        manifest,
        seed=effective_selection_seed,
        per_family=int(per_family),
        excluded_group_ids=excluded_group_ids,
        split=selection_split,
    )
    saved_time_parent = resolve_saved_time_parent_config(config)
    travel_time_h5 = config.get("travel_time_h5")
    if travel_time_h5 is None and saved_time_parent is not None:
        travel_time_h5 = saved_time_parent.get("travel_time_h5")
    dataset = GuardedOnsetDataset(
        config["source_h5"],
        manifest,
        split=selection_split,
        sample_ids=tuple(row.sample_id for row in rows),
        travel_time_h5=travel_time_h5,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_report(
        {
            "selection_split": selection_split,
            "selection_seed": effective_selection_seed,
            "excluded_group_ids": excluded_group_ids,
            "meta_training_group_ids": training_group_ids,
            "extra_excluded_group_ids": extra_excluded,
            "records": [row.__dict__ for row in rows],
        },
        output / "selection_manifest.json",
    )
    reports: list[dict[str, object]] = []
    try:
        for index in range(len(dataset)):
            record = dataset[index]
            parent_started = time.perf_counter()
            parent_field = _predict_parent(parent, normalizer, record, device)
            parent_elapsed = time.perf_counter() - parent_started
            conditioner_wavefield = None
            if conditioner_input_kind == "parent_onset_residual":
                onset_index = list(record.observed_indices)
                conditioner_wavefield = (
                    record.observed_wavefield.to(device).unsqueeze(0)
                    - parent_field[:, onset_index]
                ).detach()
            with torch.no_grad():
                zero_started = time.perf_counter()
                zero_field = _predict_feature(
                    adapter,
                    normalizer,
                    record,
                    device,
                    time_s=record.time_s.to(device),
                    conditioner_wavefield=conditioner_wavefield,
                )
                zero_elapsed = time.perf_counter() - zero_started
            modulation_gate = None
            if refinement_mode == "onset_only":
                delta, refinement = _refine_latent_onset_only(
                    adapter,
                    normalizer,
                    record,
                    device,
                    steps=int(latent_steps),
                    learning_rate=float(latent_learning_rate),
                    conditioner_wavefield=conditioner_wavefield,
                )
            elif refinement_mode == "residual_trust_gate":
                if conditioner_wavefield is None:
                    raise ValueError(
                        "residual trust gate requires a residual-conditioned checkpoint"
                    )
                delta, modulation_gate, refinement = _refine_residual_trust_gate(
                    adapter,
                    normalizer,
                    record,
                    device,
                    steps=int(latent_steps),
                    learning_rate=float(latent_learning_rate),
                    conditioner_wavefield=conditioner_wavefield,
                )
            elif refinement_mode == "lwc84_bridge":
                delta, refinement = _refine_latent_with_lwc84(
                    adapter,
                    normalizer,
                    record,
                    device,
                    steps=int(latent_steps),
                    learning_rate=float(latent_learning_rate),
                    bridge_steps=int(bridge_steps),
                )
            else:
                raise ValueError("unknown feature latent-refinement mode")
            with torch.no_grad():
                hybrid_started = time.perf_counter()
                hybrid_field = _predict_feature(
                    adapter,
                    normalizer,
                    record,
                    device,
                    time_s=record.time_s.to(device),
                    latent_delta=delta,
                    conditioner_wavefield=conditioner_wavefield,
                    modulation_gate=modulation_gate,
                )
                hybrid_elapsed = time.perf_counter() - hybrid_started
            artifact = output / record.sample_id
            artifact.mkdir(parents=True, exist_ok=True)
            sealed_payload = {
                "parent_field": parent_field.detach().cpu(),
                "zero_shot_field": zero_field.detach().cpu(),
                "hybrid_field": hybrid_field.detach().cpu(),
                "latent_delta": delta.cpu(),
                "modulation_gate": (
                    None if modulation_gate is None else modulation_gate.cpu()
                ),
                "conditioner_input": conditioner_input_kind,
                "refinement": refinement,
                "input_digest": record.input_digest,
                "timing_s": {
                    "parent": parent_elapsed,
                    "zero_shot": zero_elapsed,
                    "latent_refinement": refinement["elapsed_s"],
                    "hybrid_inference": hybrid_elapsed,
                },
            }
            if not summary_only:
                torch.save(sealed_payload, artifact / "sealed_predictions.pt")
            truth = _read_future_truth(config["source_h5"], record.source_index)
            common = {
                "target": truth.unsqueeze(0),
                "observed_indices": (record.observed_indices,),
                "families": (record.medium_type,),
                "group_ids": (record.group_id,),
                "sample_ids": (record.sample_id,),
                "sealed": True,
            }
            zero_report = evaluate_after_adaptation(
                {
                    "parent_field": parent_field.cpu(),
                    "adapted_field": zero_field.cpu(),
                },
                common.pop("target"),
                **common,
            )
            hybrid_report = evaluate_after_adaptation(
                {
                    "parent_field": parent_field.cpu(),
                    "adapted_field": hybrid_field.cpu(),
                },
                truth.unsqueeze(0),
                **common,
            )
            report = {
                "sample_id": record.sample_id,
                "medium_type": record.medium_type,
                "observed_indices": record.observed_indices,
                "conditioner_input": conditioner_input_kind,
                "input_digest": record.input_digest,
                "sealed_prediction_sha256": {
                    "parent": _tensor_sha256(parent_field),
                    "zero_shot": _tensor_sha256(zero_field),
                    "adapted": _tensor_sha256(hybrid_field),
                },
                "zero_shot": zero_report,
                "hybrid": hybrid_report,
                "refinement": refinement,
                "timing_s": sealed_payload["timing_s"],
            }
            write_report(report, artifact / "evaluation.json")
            if not summary_only:
                snapshots = (
                    record.observed_indices[1],
                    len(record.time_s) // 2,
                    len(record.time_s) - 1,
                )
                target_np = truth.numpy()
                parent_np = parent_field[0].detach().cpu().numpy()
                zero_np = zero_field[0].detach().cpu().numpy()
                hybrid_np = hybrid_field[0].detach().cpu().numpy()
                plot_wavefield_comparison(
                    target_np,
                    parent_np,
                    zero_np,
                    snapshots,
                    output=artifact / "wavefield_zero_shot",
                    title=f"{record.medium_type} · zero-shot early-feature adapter",
                )
                plot_wavefield_comparison(
                    target_np,
                    parent_np,
                    hybrid_np,
                    snapshots,
                    output=artifact / "wavefield_hybrid",
                    title=f"{record.medium_type} · {refinement_mode} latent refinement",
                )
                plot_receiver_comparison(
                    target_np,
                    parent_np,
                    hybrid_np,
                    record.time_s.numpy(),
                    hybrid_report["receiver_indices"],
                    output=artifact / "receiver_waveforms_hybrid",
                    title=record.medium_type,
                )
                _plot_velocity(record, artifact / "velocity_model")
            reports.append(report)
    finally:
        dataset.close()
    summary = {
        "adapter_checkpoint": str(checkpoint),
        "adapter_checkpoint_sha256": _sha256_file(checkpoint),
        "adapter_epoch": adapter_payload.get("epoch"),
        "selection_split": selection_split,
        "selection_seed": effective_selection_seed,
        "excluded_meta_training_group_ids": training_group_ids,
        "extra_excluded_group_ids": extra_excluded,
        "refinement_mode": refinement_mode,
        "conditioner_input": conditioner_input_kind,
        "summary_only": bool(summary_only),
        "records": reports,
        "family_count": {
            family: sum(item["medium_type"] == family for item in reports)
            for family in ALLOWED_MEDIUM_TYPES
        },
        "zero_shot_future_energy": aggregate_future_energy(reports, "zero_shot"),
        "adapted_future_energy": aggregate_future_energy(reports, "hybrid"),
        "paired_future_energy_wins_vs_parent": {
            "zero_shot": sum(
                float(item["zero_shot"]["future_adapted_squared_error"])
                < float(item["zero_shot"]["future_parent_squared_error"])
                for item in reports
            ),
            "adapted": sum(
                float(item["hybrid"]["future_adapted_squared_error"])
                < float(item["hybrid"]["future_parent_squared_error"])
                for item in reports
            ),
            "records": len(reports),
        },
        "future_truth_opened_only_after_seal": True,
    }
    summary_path = output / "summary.json"
    write_report(summary, summary_path)
    return summary_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-family", type=int, default=1)
    parser.add_argument("--latent-steps", type=int, default=6)
    parser.add_argument("--latent-learning-rate", type=float, default=0.05)
    parser.add_argument("--bridge-steps", type=int, default=4)
    parser.add_argument(
        "--selection-split", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--exclude-group-id", action="append", default=[])
    parser.add_argument(
        "--refinement-mode",
        choices=("onset_only", "residual_trust_gate", "lwc84_bridge"),
        default="lwc84_bridge",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    summary = evaluate(
        args.config,
        args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        per_family=args.per_family,
        latent_steps=args.latent_steps,
        latent_learning_rate=args.latent_learning_rate,
        bridge_steps=args.bridge_steps,
        selection_split=args.selection_split,
        selection_seed=args.selection_seed,
        extra_excluded_group_ids=args.exclude_group_id,
        refinement_mode=args.refinement_mode,
        summary_only=args.summary_only,
    )
    print(json.dumps({"summary": str(summary)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
