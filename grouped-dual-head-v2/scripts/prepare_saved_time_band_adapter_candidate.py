#!/usr/bin/env python3
"""Prepare an exact-V49-parent, high-band-safe adapter pilot."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.evaluation import sha256_file, time_axis_sha256


V49_ARTIFACT_NAME = "saved_time_v49_family_experts_structural_prior_pilot_r1"
V49_RUN_DIGEST = "f71dbac27c09bb94244c5866c8c02b370724a2a56904f2a66d0913c17a36cc63"
V49_CHECKPOINT_SHA256 = "3b8569790ae9f03883d3a9a1aed0c71f6138a611d9f33392b90ccd5914fa19bc"
V49_IDENTITY_SHA256 = "8d787f15fc6eb077309de46e297d2e4e71726e1a9561453b6a1e9f97f0b9b21b"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(value)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _shape_attribute(value, *, name: str) -> list[int]:
    try:
        parsed = json.loads(str(value))
        result = [int(item) for item in parsed]
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"dataset {name} is malformed") from error
    if len(result) != 2 or any(item <= 0 for item in result):
        raise ValueError(f"dataset {name} is malformed")
    return result


def validate_dataset_contract(path: str | Path) -> dict[str, object]:
    """Bind the unchanged LWC84 grid, boundary, source, time, and output contract."""

    source = Path(path).resolve()
    with h5py.File(source, "r") as handle:
        schema = str(handle.attrs.get("schema_version", ""))
        solver_shape = _shape_attribute(
            handle.attrs.get("solver_grid_shape", ""),
            name="solver_grid_shape",
        )
        saved_shape = _shape_attribute(
            handle.attrs.get("saved_grid_shape", ""),
            name="saved_grid_shape",
        )
        if schema != "acoustic_lwc84_401_to_201_v1":
            raise ValueError("dataset is not the registered LWC84 401-to-201 schema")
        if solver_shape != [401, 401] or saved_shape != [201, 201]:
            raise ValueError("dataset must preserve 401x401 solve and 201x201 output grids")
        free_surface = str(handle.attrs.get("free_surface", ""))
        cpml = str(handle.attrs.get("cpml", ""))
        if "z=0" not in free_surface or "no top" not in cpml.lower():
            raise ValueError("dataset does not preserve free-top and three-sided CPML")
        required = (
            "time_s",
            "source_x_m",
            "source_z_m",
            "source_f0_hz",
            "source_t0_s",
            "source_amplitude",
            "source_map",
            "velocity_mps",
            "wavefield",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"dataset source/wavefield contract is incomplete: {missing}")
        record_count = int(handle["wavefield"].shape[0])
        time_values = np.asarray(handle["time_s"], dtype=np.float64)
        if (
            record_count <= 0
            or time_values.ndim != 1
            or time_values.size <= 1
            or not np.isfinite(time_values).all()
            or not np.all(np.diff(time_values) > 0.0)
        ):
            raise ValueError("dataset stored-time axis is invalid")
        if tuple(handle["wavefield"].shape[1:]) != (
            time_values.size,
            201,
            201,
        ):
            raise ValueError("dataset wavefield is not [record,time,201,201]")
        if tuple(handle["velocity_mps"].shape) != (record_count, 201, 201):
            raise ValueError("dataset velocity model is not [record,201,201]")
        if tuple(handle["source_map"].shape) != (record_count, 201, 201):
            raise ValueError("dataset source map is not one map per record")
        if any(tuple(handle[name].shape) != (record_count,) for name in required[1:6]):
            raise ValueError("dataset must contain exactly one source tuple per record")
        object_names: list[str] = []
        handle.visit(object_names.append)
        receiver_names = [name for name in object_names if "receiver" in name.lower()]
        if receiver_names:
            raise ValueError("dataset contract unexpectedly contains receiver inputs")
        return {
            "dataset_h5": str(source),
            "schema_version": schema,
            "solver_grid_shape": solver_shape,
            "wavefield_shape": saved_shape,
            "stored_time_count": int(time_values.size),
            "time_axis_sha256": time_axis_sha256(time_values),
            "record_count": record_count,
            "one_source_per_record": True,
            "receiver_input": False,
            "free_surface": free_surface,
            "cpml": cpml,
        }


def validate_v49_parent_identity(
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Reject every parent except the registered high-band-safe V49 epoch family."""

    if (
        identity.get("schema") != "saved_time_v5_training_contract_recovery_v1"
        or identity.get("run_digest") != V49_RUN_DIGEST
    ):
        raise ValueError("band adapter parent must be the registered V49 run")
    config = identity.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("registered V49 identity has no config")
    if Path(str(config.get("artifact_dir", ""))).name != V49_ARTIFACT_NAME:
        raise ValueError("band adapter parent artifact is not V49")
    overrides = config.get("variant_overrides")
    expected_overrides = {
        "temporal_basis_rank": 96,
        "family_expert_rank": 16,
    }
    if not isinstance(overrides, Mapping) or dict(overrides) != expected_overrides:
        raise ValueError("V49 parent architecture is not the registered variant")
    if config.get("band_limited_adapter") is not None:
        raise ValueError("V49 parent must not already contain a band adapter")
    validation = config.get("validation")
    if (
        not isinstance(validation, Mapping)
        or int(validation.get("panel_records", 0)) != 48
        or int(validation.get("frames_per_record", 0)) != 32
        or str(config.get("time_policy", "")) != "appearance16"
    ):
        raise ValueError("V49 parent does not use the registered exact-time panel")
    if not identity.get("manifest_digest") or not identity.get("time_axis_sha256"):
        raise ValueError("V49 parent lacks manifest or exact-time provenance")
    return dict(config)


def build_band_adapter_pilot_config(
    parent_identity: Mapping[str, object],
    *,
    parent_checkpoint: str | Path,
    parent_checkpoint_identity: str | Path,
    artifact_dir: str | Path,
    band_adapter_rank: int = 16,
    band_adapter_architecture: str = "low_rank",
    band_adapter_spectral_rank: int = 32,
    band_adapter_modes: int = 32,
    band_adapter_full_depth: int = 4,
    band_adapter_coarse_depth: int = 2,
    band_adapter_activation_checkpointing: bool = True,
    band_adapter_preserve_high_band: bool = True,
    physical_microbatch_records: int = 12,
    macro_records: int = 12,
    effective_batch_records: int = 96,
    training_frames_per_record: int | None = None,
    dense_learning_rate: float = 1.0e-4,
    band_adapter_feature_learning_rate: float = 3.0e-4,
    band_adapter_output_learning_rate: float = 1.0e-5,
    extra_variant_overrides: Mapping[str, object] | None = None,
    restore_parent_optimizer: bool = False,
) -> dict[str, object]:
    """Create one staged, adapter-only V60/V61 pilot from exact V49 weights."""

    parent_config = validate_v49_parent_identity(parent_identity)
    rank = int(band_adapter_rank)
    architecture = str(band_adapter_architecture)
    adapter_spectral_rank = int(band_adapter_spectral_rank)
    adapter_modes = int(band_adapter_modes)
    adapter_full_depth = int(band_adapter_full_depth)
    adapter_coarse_depth = int(band_adapter_coarse_depth)
    physical = int(physical_microbatch_records)
    macro = int(macro_records)
    effective_batch = int(effective_batch_records)
    training_frames = (
        None
        if training_frames_per_record is None
        else int(training_frames_per_record)
    )
    dense_lr = float(dense_learning_rate)
    feature_lr = float(band_adapter_feature_learning_rate)
    output_lr = float(band_adapter_output_learning_rate)
    if isinstance(band_adapter_rank, bool) or not 1 <= rank <= 64:
        raise ValueError("band adapter rank must lie in [1, 64]")
    if architecture not in {
        "low_rank",
        "multiscale_spectral",
        "dynamic_multiscale_spectral",
    }:
        raise ValueError("band adapter architecture is unsupported")
    capacity = {
        "spectral rank": (band_adapter_spectral_rank, adapter_spectral_rank),
        "full depth": (band_adapter_full_depth, adapter_full_depth),
        "coarse depth": (band_adapter_coarse_depth, adapter_coarse_depth),
    }
    for name, (raw, resolved) in capacity.items():
        if isinstance(raw, bool) or not 1 <= resolved <= 128:
            raise ValueError(f"band adapter {name} must lie in [1, 128]")
    if (
        isinstance(band_adapter_modes, bool)
        or not 1 <= adapter_modes <= 101
    ):
        raise ValueError("band adapter modes must lie in [1, 101]")
    if not isinstance(band_adapter_activation_checkpointing, bool):
        raise ValueError("band adapter activation checkpointing must be boolean")
    if not isinstance(band_adapter_preserve_high_band, bool):
        raise ValueError("band adapter high-band preservation must be boolean")
    if physical <= 0 or physical > 24:
        raise ValueError("band adapter physical microbatch must lie in [1, 24]")
    if macro <= 0:
        raise ValueError("band adapter macro records must be positive")
    if physical > macro:
        raise ValueError("band adapter physical microbatch must lie within one macro")
    if effective_batch <= 0 or effective_batch % macro:
        raise ValueError(
            "band adapter effective batch must be positive and divisible by macro records"
        )
    macros_per_update = effective_batch // macro
    if macros_per_update % 4:
        raise ValueError(
            "band adapter macros per update must divide evenly across four-GPU DDP"
        )
    if training_frames is not None and not 16 <= training_frames <= 401:
        raise ValueError("band adapter training frames must lie in [16, 401]")
    if not math.isfinite(dense_lr) or dense_lr <= 0.0:
        raise ValueError("band adapter dense learning rate must be positive and finite")
    if architecture in {
        "multiscale_spectral",
        "dynamic_multiscale_spectral",
    } and any(
        not math.isfinite(value) or value <= 0.0
        for value in (feature_lr, output_lr)
    ):
        raise ValueError("band adapter split learning rates must be positive and finite")
    if extra_variant_overrides:
        raise ValueError("band adapter and other architecture expansions must be staged")
    if restore_parent_optimizer:
        raise ValueError("band adapter expansion cannot restore parent optimizer state")

    config = deepcopy(parent_config)
    config["parent_checkpoint"] = str(Path(parent_checkpoint).resolve())
    config["parent_checkpoint_identity"] = str(
        Path(parent_checkpoint_identity).resolve()
    )
    config["artifact_dir"] = str(Path(artifact_dir).resolve())
    overrides: dict[str, object] = {
        "temporal_basis_rank": 96,
        "family_expert_rank": 16,
        "band_adapter_rank": rank,
    }
    adapter_config: dict[str, object] = {
        "adapter_only": True,
        "cutoff_normalized_radius": 2.0 / 3.0,
    }
    if not band_adapter_preserve_high_band:
        overrides["band_adapter_preserve_high_band"] = False
        adapter_config["preserve_high_band"] = False
    if architecture in {
        "multiscale_spectral",
        "dynamic_multiscale_spectral",
    }:
        overrides.update(
            {
                "band_adapter_architecture": architecture,
                "band_adapter_spectral_rank": adapter_spectral_rank,
                "band_adapter_modes": adapter_modes,
                "band_adapter_full_depth": adapter_full_depth,
                "band_adapter_coarse_depth": adapter_coarse_depth,
                "band_adapter_activation_checkpointing": (
                    band_adapter_activation_checkpointing
                ),
            }
        )
        adapter_config["architecture"] = architecture
        if architecture == "dynamic_multiscale_spectral":
            adapter_config["physical_conditioning"] = {
                "medium": "raw_velocity_encoded_once",
                "source": "normalized_parameters",
            }
    config["variant_overrides"] = overrides
    config["band_limited_adapter"] = adapter_config
    transfer = dict(config.get("checkpoint_transfer", {}))
    for name in tuple(transfer):
        if name.startswith("allow_new_") or name == "allow_spectral_mode_expansion":
            transfer.pop(name)
    transfer.update(
        {
            "parent_residual_already_active": True,
            "parent_optimizer_state": False,
            "allow_new_band_adapter_parameters": True,
        }
    )
    config["checkpoint_transfer"] = transfer
    config["family_curriculum"] = {
        "stages": [
            {
                "epochs": 6,
                "macro_pattern": ["uniform", "layered", "marmousi"],
            }
        ]
    }
    config["epochs"] = 6
    config["family_gradient_weights"] = {
        "uniform": 1.0,
        "layered": 1.0,
        "marmousi": 1.0,
    }
    config["macro_records"] = macro
    config["macros_per_update"] = macros_per_update
    config["microbatch_records"] = physical
    if training_frames is not None:
        config["training_frames_per_record"] = training_frames
    config["time_appearance_offset"] = int(
        parent_config.get("time_appearance_offset", 0)
    ) + 1
    config.pop("schedule_epoch_offset", None)

    optimizer = dict(config["optimizer"])
    optimizer.update(
        {
            "dense_learning_rate": dense_lr,
            "weight_decay": 1.0e-6,
            "schedule_epoch_offset": 0,
            "schedule_total_epochs": 6,
            "warmup_epochs": 1,
        }
    )
    if architecture in {
        "multiscale_spectral",
        "dynamic_multiscale_spectral",
    }:
        optimizer.update(
            {
                "band_adapter_feature_learning_rate": feature_lr,
                "band_adapter_output_learning_rate": output_lr,
            }
        )
    config["optimizer"] = optimizer
    loss = dict(config["loss"])
    loss.update(
        {
            "delta": 0.0,
            "temporal_difference": 0.0,
            "spatial_gradient": 0.0,
            "spectrum": 0.05,
            "hard_causality": True,
            "hard_causality_lead_cycles": 1.0,
        }
    )
    config["loss"] = loss
    recovery = dict(config.get("residual_recovery", {}))
    recovery.update(
        {
            "activation_mode": "preserve",
            "absorb_temporal_basis_gate": False,
        }
    )
    config["residual_recovery"] = recovery
    family_experts = dict(config["family_experts"])
    family_experts["teacher_forced_routing"] = True
    family_experts["router_loss_weight"] = 0.0
    config["family_experts"] = family_experts
    gate = dict(config["gate"])
    gate.update({"pilot_epochs": 6, "external_evidence_gate": True})
    config["gate"] = gate
    return config


def _validate_parent_files(
    checkpoint: Path,
    identity_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    if sha256_file(checkpoint) != V49_CHECKPOINT_SHA256:
        raise ValueError("parent checkpoint SHA256 is not the registered V49 epoch 1")
    if sha256_file(identity_path) != V49_IDENTITY_SHA256:
        raise ValueError("parent identity SHA256 is not the registered V49 identity")
    identity = json.loads(identity_path.read_text())
    validate_v49_parent_identity(identity)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("epoch", -1)) != 1
        or str(payload.get("config_digest", "")) != V49_RUN_DIGEST
        or str(payload.get("manifest_digest", ""))
        != str(identity["manifest_digest"])
    ):
        raise ValueError("parent checkpoint metadata is not the registered V49 epoch 1")
    return identity, dict(payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--parent-checkpoint-identity", required=True)
    parser.add_argument("--dataset-h5", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--band-adapter-rank", type=int, default=16)
    parser.add_argument(
        "--band-adapter-architecture",
        choices=(
            "low_rank",
            "multiscale_spectral",
            "dynamic_multiscale_spectral",
        ),
        default="low_rank",
    )
    parser.add_argument("--band-adapter-spectral-rank", type=int, default=32)
    parser.add_argument("--band-adapter-modes", type=int, default=32)
    parser.add_argument("--band-adapter-full-depth", type=int, default=4)
    parser.add_argument("--band-adapter-coarse-depth", type=int, default=2)
    parser.add_argument("--physical-microbatch-records", type=int, default=12)
    parser.add_argument("--macro-records", type=int, default=12)
    parser.add_argument("--effective-batch-records", type=int, default=96)
    parser.add_argument("--training-frames-per-record", type=int)
    parser.add_argument("--dense-learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--band-adapter-feature-learning-rate", type=float, default=3.0e-4
    )
    parser.add_argument(
        "--band-adapter-output-learning-rate", type=float, default=1.0e-5
    )
    parser.add_argument(
        "--band-adapter-preserve-high-band",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    checkpoint = Path(args.parent_checkpoint).resolve()
    identity_path = Path(args.parent_checkpoint_identity).resolve()
    output_config = Path(args.output_config).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    report_path = Path(args.report).resolve()
    identity, payload = _validate_parent_files(checkpoint, identity_path)
    dataset = validate_dataset_contract(args.dataset_h5)
    if dataset["time_axis_sha256"] != identity["time_axis_sha256"]:
        raise ValueError("dataset exact-time axis does not match the V49 identity")
    config = build_band_adapter_pilot_config(
        identity,
        parent_checkpoint=checkpoint,
        parent_checkpoint_identity=identity_path,
        artifact_dir=artifact_dir,
        band_adapter_rank=int(args.band_adapter_rank),
        band_adapter_architecture=str(args.band_adapter_architecture),
        band_adapter_spectral_rank=int(args.band_adapter_spectral_rank),
        band_adapter_modes=int(args.band_adapter_modes),
        band_adapter_full_depth=int(args.band_adapter_full_depth),
        band_adapter_coarse_depth=int(args.band_adapter_coarse_depth),
        band_adapter_activation_checkpointing=True,
        band_adapter_preserve_high_band=bool(args.band_adapter_preserve_high_band),
        physical_microbatch_records=int(args.physical_microbatch_records),
        macro_records=int(args.macro_records),
        effective_batch_records=int(args.effective_batch_records),
        training_frames_per_record=args.training_frames_per_record,
        dense_learning_rate=float(args.dense_learning_rate),
        band_adapter_feature_learning_rate=float(
            args.band_adapter_feature_learning_rate
        ),
        band_adapter_output_learning_rate=float(
            args.band_adapter_output_learning_rate
        ),
    )
    _atomic_text(output_config, yaml.safe_dump(config, sort_keys=False))
    evidence = {
        "schema": "saved_time_band_adapter_parent_selection_v1",
        "status": "complete",
        "parent": {
            "name": V49_ARTIFACT_NAME,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": V49_CHECKPOINT_SHA256,
            "checkpoint_epoch": int(payload["epoch"]),
            "checkpoint_global_step": int(payload["global_step"]),
            "checkpoint_identity": str(identity_path),
            "checkpoint_identity_sha256": V49_IDENTITY_SHA256,
            "run_digest": V49_RUN_DIGEST,
            "manifest_digest": str(identity["manifest_digest"]),
            "time_axis_sha256": str(identity["time_axis_sha256"]),
        },
        "data_contract": dataset,
        "candidate": {
            "generated_config": str(output_config),
            "generated_config_sha256": sha256_file(output_config),
            "artifact_dir": str(artifact_dir),
            "band_adapter_rank": int(args.band_adapter_rank),
            "band_adapter_architecture": str(args.band_adapter_architecture),
            "band_adapter_spectral_rank": int(args.band_adapter_spectral_rank),
            "band_adapter_modes": int(args.band_adapter_modes),
            "band_adapter_full_depth": int(args.band_adapter_full_depth),
            "band_adapter_coarse_depth": int(args.band_adapter_coarse_depth),
            "band_adapter_activation_checkpointing": True,
            "band_adapter_preserve_high_band": bool(
                args.band_adapter_preserve_high_band
            ),
            "adapter_only": True,
            "effective_batch": int(config["macro_records"])
            * int(config["macros_per_update"]),
            "macro_records": int(config["macro_records"]),
            "macros_per_update": int(config["macros_per_update"]),
            "physical_microbatch_records": int(config["microbatch_records"]),
            "training_frames_per_record": int(
                config.get("training_frames_per_record", 16)
            ),
            "dense_learning_rate": float(
                config["optimizer"]["dense_learning_rate"]
            ),
            "pilot_epochs": int(config["gate"]["pilot_epochs"]),
            "exact_parent_identity": True,
        },
    }
    if "band_adapter_feature_learning_rate" in config["optimizer"]:
        evidence["candidate"].update(
            {
                "band_adapter_feature_learning_rate": float(
                    config["optimizer"]["band_adapter_feature_learning_rate"]
                ),
                "band_adapter_output_learning_rate": float(
                    config["optimizer"]["band_adapter_output_learning_rate"]
                ),
            }
        )
    if not math.isclose(
        float(config["band_limited_adapter"]["cutoff_normalized_radius"]),
        2.0 / 3.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("band adapter cutoff registration changed")
    _atomic_text(report_path, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V49_ARTIFACT_NAME",
    "V49_RUN_DIGEST",
    "build_band_adapter_pilot_config",
    "validate_dataset_contract",
    "validate_v49_parent_identity",
]
