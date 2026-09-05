#!/usr/bin/env python3
"""Offline first-order meta-training for the CPADC causal error basis.

The frozen parent is evaluated once per train episode.  Every inner episode
uses only the two guarded onset frames, a synthetic LWC-84 bridge, and the
source-consistent discrete defect to solve small correction coefficients.  The
train-only future field appears exclusively in the outer meta loss that updates
the shared basis generator.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.bridge import make_onset_bridge
from saved_time_phase_operator_v4.instance_adaptation.cpadc_contract import (
    cpadc_implementation_digests,
    validate_dataset_cpml_contract,
)
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.defect_correction import (
    CPADC_SCHEMA_VERSION,
    CausalErrorBasisGenerator,
    ConvexDefectCorrectionResult,
    DefectCorrectionWeights,
    FactorizedCausalBasis,
    MetaDefectLossWeights,
    causal_observation_probe_basis,
    meta_defect_correction_loss,
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
from saved_time_phase_operator_v4.muon import build_module_muon_adamw
from scripts.run_v5_instance_adaptation import _load_background_provider, _load_parent
from scripts.train_meta_hypernet import (
    _average_gradients,
    _broadcast_trainable_parameters,
    _distributed_context,
    _parent_normalized_at_times,
)
from scripts.train_residual_iterator import _weighted_epoch_indices
from scripts.train_v5_feature_meta import build_balanced_meta_episodes
from scripts.train_v5_residual_meta import _sha256


def _config_digest(config: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _family_weights(config: Mapping[str, object], key: str) -> dict[str, float]:
    raw = config.get(key, {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must map medium families to positive weights")
    unknown = set(str(value) for value in raw) - set(ALLOWED_MEDIUM_TYPES)
    if unknown:
        raise ValueError(f"{key} contains unknown families: {sorted(unknown)}")
    result = {
        family: float(raw.get(family, 1.0)) for family in ALLOWED_MEDIUM_TYPES
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in result.values()):
        raise ValueError(f"{key} weights must be finite and positive")
    return result


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        (prediction.float() - target.float()).flatten(1).norm(dim=1)
        / target.float().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    )


def _first_order_projected_coefficients(
    basis: FactorizedCausalBasis,
    inner: ConvexDefectCorrectionResult,
    *,
    maximum_correction_ratio: float,
) -> torch.Tensor:
    """Keep the ridge direction fixed while learning the causal trust radius.

    First-order CPADC intentionally stops gradients through the CPU ridge solve.
    The old implementation also stopped gradients through its energy-ball
    projection, which left ``trust_head`` frozen at initialization.  Recover the
    detached unconstrained ridge direction and reapply the projection with the
    live trust fraction.  This matches deployment in the forward pass while
    allowing the train-only outer loss to learn an instance-conditioned radius.
    """

    hard_limit = float(maximum_correction_ratio)
    if not math.isfinite(hard_limit) or hard_limit <= 0.0:
        raise ValueError("maximum correction ratio must be finite and positive")
    if not bool(inner.accepted.all()):
        raise ValueError("first-order trust projection requires accepted inner solves")
    device = basis.trust_fraction.device
    dtype = basis.trust_fraction.dtype
    stored_scale = inner.projection_scale.detach().to(device=device, dtype=dtype)
    raw_ratio = inner.unconstrained_correction_ratio.detach().to(
        device=device, dtype=dtype
    )
    if bool(torch.any(stored_scale <= 0.0)):
        raise ValueError("first-order trust projection received a zero ridge scale")
    raw_direction = (
        inner.coefficients.detach().to(device=device, dtype=dtype)
        / stored_scale[:, None]
    )
    live_limit = basis.trust_fraction * hard_limit
    live_scale = torch.minimum(
        torch.ones_like(raw_ratio),
        live_limit / raw_ratio.clamp_min(1.0e-12),
    )
    return raw_direction * live_scale[:, None]


def _cpml_physics_window(
    time_s: torch.Tensor,
    source_parameters: torch.Tensor,
    velocity_mps: torch.Tensor,
    observed_indices: tuple[int, int],
    *,
    dx_m: float,
    residual_count: int,
    prearrival_frames: int,
) -> tuple[int, int]:
    """Choose a causal window around the first possible three-sided CPML arrival."""

    times = torch.as_tensor(time_s).flatten()
    source = torch.as_tensor(source_parameters)
    velocity = torch.as_tensor(velocity_mps)
    if source.shape != (1, 5) or velocity.ndim != 4 or velocity.shape[0] != 1:
        raise ValueError("CPADC CPML window requires one source and one medium")
    count = int(residual_count)
    prearrival = int(prearrival_frames)
    if count < 1 or prearrival < 0 or times.numel() < count + 2:
        raise ValueError("CPADC CPML window settings are invalid")
    domain_x_m = (int(velocity.shape[-1]) - 1) * float(dx_m)
    source_x_m = float(source[0, 0].detach())
    side_distance_m = min(source_x_m, domain_x_m - source_x_m)
    if side_distance_m < 0.0:
        raise ValueError("source lies outside the physical CPML domain")
    peak_arrival_s = float(source[0, 3].detach()) + side_distance_m / max(
        float(velocity.detach().amax()), 1.0
    )
    arrival = int(
        torch.searchsorted(
            times.detach().cpu().double(),
            torch.tensor(peak_arrival_s, dtype=torch.float64),
        ).item()
    )
    earliest = int(observed_indices[1]) + 1
    start = max(earliest, arrival - prearrival)
    stop = min(int(times.numel()), start + count + 1)
    if stop - start < count + 1:
        start = max(earliest, stop - count - 1)
    if start < 1 or stop - start < 2:
        raise ValueError("CPADC CPML window has insufficient recurrence support")
    return start, stop


def _weighted_cpml_defect_loss(
    defect: torch.Tensor,
    *,
    boundary_band_cells: int,
    boundary_weight: float,
) -> torch.Tensor:
    """Average full-grid defect while upweighting left/right/bottom CPML interfaces."""

    value = torch.as_tensor(defect)
    band = int(boundary_band_cells)
    weight = float(boundary_weight)
    if value.ndim != 4 or band < 1 or 2 * band >= value.shape[-1] or band >= value.shape[-2]:
        raise ValueError("CPADC CPML boundary band is invalid")
    if not math.isfinite(weight) or weight < 1.0:
        raise ValueError("CPADC CPML boundary weight must be at least one")
    squared = value.square()
    boundary_mask = torch.zeros_like(squared, dtype=torch.bool)
    boundary_mask[..., :band] = True
    boundary_mask[..., -band:] = True
    boundary_mask[..., -band:, :] = True
    extra = weight - 1.0
    numerator = squared.sum() + extra * squared[boundary_mask].sum()
    denominator = squared.numel() + extra * int(boundary_mask.sum())
    return numerator / float(denominator)


def _spatial_patch(
    *,
    height: int,
    width: int,
    patch_size: int,
    sample_index: int,
    epoch: int,
    seed: int,
) -> tuple[slice, slice]:
    patch = min(int(patch_size), int(height), int(width))
    if patch <= 16:
        raise ValueError("CPADC patches must be at least 17x17")
    generator = torch.Generator().manual_seed(
        int(seed) + 1_000_003 * int(epoch) + 9_973 * int(sample_index)
    )
    z0 = int(torch.randint(0, height - patch + 1, (1,), generator=generator))
    x0 = int(torch.randint(0, width - patch + 1, (1,), generator=generator))
    return slice(z0, z0 + patch), slice(x0, x0 + patch)


def _cache_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value).detach().contiguous().cpu()


def _cpadc_adamw_parameter_groups(
    named_parameters,
    *,
    weight_decay: float,
    learning_rate: float | None = None,
    trust_learning_rate_scale: float = 1.0,
) -> tuple[dict[str, object], ...]:
    """Separate the low-dimensional trust controller from basis directions."""

    decay_value = float(weight_decay)
    if not math.isfinite(decay_value) or decay_value < 0.0:
        raise ValueError("CPADC weight decay must be finite and nonnegative")
    base_lr = None if learning_rate is None else float(learning_rate)
    trust_scale = float(trust_learning_rate_scale)
    if base_lr is not None and (not math.isfinite(base_lr) or base_lr <= 0.0):
        raise ValueError("CPADC learning rate must be finite and positive")
    if not math.isfinite(trust_scale) or trust_scale <= 0.0:
        raise ValueError("CPADC trust learning-rate scale must be finite and positive")
    buckets: dict[str, list[torch.nn.Parameter]] = {
        "decay": [],
        "no_decay": [],
        "trust_decay": [],
        "trust_no_decay": [],
    }
    seen: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"duplicate CPADC optimizer parameter: {name}")
        seen.add(identity)
        trust = str(name).startswith("trust_head.")
        decayed = parameter.ndim >= 2 and not str(name).endswith(".bias")
        key = ("trust_" if trust else "") + ("decay" if decayed else "no_decay")
        buckets[key].append(parameter)
    if not any(buckets.values()):
        raise ValueError("CPADC optimizer has no trainable parameters")
    groups: list[dict[str, object]] = []
    for name, parameters in buckets.items():
        if not parameters:
            continue
        group: dict[str, object] = {
            "params": tuple(parameters),
            "weight_decay": decay_value if name.endswith("decay") and not name.endswith("no_decay") else 0.0,
            "group_name": name,
        }
        if base_lr is not None:
            group["lr"] = base_lr * (trust_scale if name.startswith("trust_") else 1.0)
        groups.append(group)
    return tuple(groups)


def _cpadc_learning_rate_for_epoch(
    epoch: int,
    *,
    total_epochs: int,
    maximum_learning_rate: float,
    minimum_learning_rate: float,
    warmup_epochs: int,
) -> float:
    """Linearly warm up, then cosine-decay the CPADC outer optimizer."""

    current = int(epoch)
    total = int(total_epochs)
    warmup = int(warmup_epochs)
    maximum = float(maximum_learning_rate)
    minimum = float(minimum_learning_rate)
    if total <= 0 or current <= 0 or current > total:
        raise ValueError("CPADC epoch must lie inside the registered horizon")
    if warmup < 0 or warmup >= total:
        raise ValueError("CPADC warmup must be shorter than the training horizon")
    if not 0.0 < minimum <= maximum or not all(
        math.isfinite(value) for value in (minimum, maximum)
    ):
        raise ValueError("CPADC learning-rate bounds are invalid")
    if warmup and current <= warmup:
        return minimum + (maximum - minimum) * current / warmup
    decay_epochs = total - warmup
    if decay_epochs == 1:
        return maximum
    progress = (current - warmup - 1) / (decay_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (maximum - minimum) * cosine


def _checkpoint_payload(
    generator: CausalErrorBasisGenerator,
    *,
    config: Mapping[str, object],
    manifest_digest: str,
    parent_checkpoint: str | Path,
    parent_checkpoint_sha256: str,
    dataset_numerical_contract: Mapping[str, object],
    implementation_digests: Mapping[str, str],
    epoch: int,
    event: Mapping[str, object],
) -> dict[str, object]:
    parent_path = Path(parent_checkpoint).expanduser().resolve()
    online_weights = config.get("online_inner_weights", {}) or {}
    online_defect_weight = float(online_weights.get("defect", 0.0))
    online_defect_design = str(
        config.get("online_defect_design", "materialized")
    )
    online_test_width = int(config.get("online_defect_test_function_width", 1))
    if online_defect_weight > 0.0:
        online_solve_name = (
            "cpu_causal_weak_sparse_defect_observation_bridge_ridge_v7"
            if online_test_width > 1
            else "cpu_causal_sparse_defect_observation_bridge_ridge_v6"
        )
    else:
        online_solve_name = "cpu_causal_observation_bridge_ridge_v5"
    return {
        "schema": "causal_physics_aligned_defect_correction_v1",
        "schema_version": CPADC_SCHEMA_VERSION,
        "basis_state": generator.state_dict(),
        "basis_rank": generator.rank,
        "phase_rank": generator.phase_rank,
        "basis_width": generator.width,
        "causal_ramp_steps": generator.ramp_steps,
        "manifest_digest": str(manifest_digest),
        "config_digest": _config_digest(config),
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "dataset_numerical_contract": dict(dataset_numerical_contract),
        "implementation_digests": dict(implementation_digests),
        "epoch": int(epoch),
        "event": dict(event),
        "defect_contract": {
            "name": "source_consistent_saved_grid_lwc84_cfs_cpml_v3",
            "time_order": 4,
            "spatial_halo_cells": 0,
            "exact_fine_generator_residual": False,
            "closure_gap": (
                "20_internal_steps_plus_binomial5_restriction_plus_"
                "omitted_exterior_pressure_plus_saved_grid_source_reinjection"
            ),
            "cpml": saved_grid_cpml_config(config.get("pde_cpml")).as_dict(),
            "cpml_memory_variables": ("psi_x", "psi_z", "phi_x", "phi_z"),
            "top_boundary": "free_surface_dirichlet",
            "cpml_memory_reconstructed_causally": True,
            "exterior_pressure_closure": "zero_unavailable_state",
            "saved_source_contract": (
                "direct_saved_grid_bilinear_unit_mass_reinjection"
            ),
            "exact_fine_source_restriction": False,
        },
        "online_solve_contract": {
            "name": online_solve_name,
            "projection_uses_parent_rms": True,
            "trust_fraction_learned_offline": True,
            "observed_probe_design": bool(
                float(online_weights.get("observed", 0.0))
                > 0.0
            ),
            "online_defect_weight": online_defect_weight,
            "online_defect_design": online_defect_design,
            "online_defect_time_order": int(
                config.get("online_defect_time_order", 4)
            ),
            "online_defect_test_function_width": online_test_width,
            "online_defect_test_function_normalization": "discrete_l2_unit",
            "online_physics_point_count": int(
                config.get("online_physics_point_count", 1)
            ),
            "online_physics_sampling": str(
                config.get("online_physics_sampling", "fixed")
            ),
            "online_rad_k": float(config.get("rad_k", 1.0)),
            "online_rad_c": float(config.get("rad_c", 1.0)),
            "online_rad_time_tilt": float(
                config.get("rad_time_tilt", 1.5)
            ),
            "online_physics_seed": int(config.get("seed", 372)),
            "online_prior_weight": float(online_weights.get("prior", 1.0e-4)),
            "online_spatial_stride": int(config.get("inner_spatial_stride", 4)),
            "online_observed_weight": float(online_weights.get("observed", 0.0)),
            "online_bridge_weight": float(online_weights.get("bridge", 0.0)),
            "coefficient_solve_device": "cpu",
            "coefficient_objective_device": "cpu",
            "correction_materialization_device": "cpu",
            "synthetic_bridge_device": "cpu",
            "offline_physics_operator": "lwc84_cfs_cpml_saved_grid_arrival_window_v4",
        },
        "differentiable_inner_solve": bool(
            config.get("differentiate_inner_solve", True)
        ),
        "future_truth_used_only_for_outer_train_loss": True,
        "online_future_truth_used": False,
        "offline_physics_training": {
            "operator": "lwc84_cfs_cpml_saved_grid_arrival_window_v4",
            "weight": float(
                ((config.get("outer_weights", {}) or {}).get("physics", 0.0))
            ),
            "time_count": int(
                ((config.get("outer_weights", {}) or {}).get(
                    "physics_time_count", 64
                ))
            ),
            "evaluations_per_candidate": 1,
            "window_strategy": str(
                ((config.get("outer_weights", {}) or {}).get(
                    "physics_window_strategy",
                    "nearest_side_cpml_peak_arrival_causal_prefix_v1",
                ))
            ),
            "causal_prefix_memory_warmup": True,
            "detached_prefix_before_loss_window": True,
            "prearrival_frames": int(
                ((config.get("outer_weights", {}) or {}).get(
                    "physics_prearrival_frames", 8
                ))
            ),
            "boundary_band_cells": int(
                ((config.get("outer_weights", {}) or {}).get(
                    "physics_boundary_band_cells", 12
                ))
            ),
            "boundary_weight": float(
                ((config.get("outer_weights", {}) or {}).get(
                    "physics_boundary_weight", 4.0
                ))
            ),
        },
        "offline_optimizer": {
            "name": str((config.get("cpadc", {}) or {}).get(
                "optimizer", "muon_adamw"
            )),
            "muon_hidden_matrices_only": True,
            "scale_sensitive_heads": "adamw",
            "trust_learning_rate_scale": float(
                (config.get("cpadc", {}) or {}).get(
                    "trust_learning_rate_scale", 1.0
                )
            ),
            "cpu_online_solve_is_optimizer_free": True,
        },
    }


def train_causal_defect_basis(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    parent_checkpoint: str | Path | None = None,
    parent_checkpoint_identity: str | Path | None = None,
    device: str = "cuda",
    per_family: int | None = None,
    epochs: int | None = None,
    patch_size: int | None = None,
    rank: int | None = None,
    phase_rank: int | None = None,
    width: int | None = None,
    learning_rate: float | None = None,
    warmup_epochs: int | None = None,
    physics_point_count: int | None = None,
    rank_chunk_size: int | None = None,
    smoke: bool = False,
) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_file.read_text())
    if not isinstance(config, dict):
        raise ValueError("CPADC config must contain a mapping")
    if parent_checkpoint is not None:
        config["parent_checkpoint"] = str(Path(parent_checkpoint).expanduser().resolve())
    if parent_checkpoint_identity is not None:
        config["parent_checkpoint_identity"] = str(
            Path(parent_checkpoint_identity).expanduser().resolve()
        )
    settings = config.get("cpadc", {}) or {}
    if not isinstance(settings, Mapping):
        raise ValueError("cpadc config section must be a mapping")
    cpml_config = saved_grid_cpml_config(config.get("pde_cpml"))
    if cpml_config is None:
        raise ValueError("CPADC requires an enabled pde_cpml contract")
    per_family = int(settings.get("per_family", 8) if per_family is None else per_family)
    epochs = int(settings.get("epochs", 5) if epochs is None else epochs)
    patch_size = int(settings.get("patch_size", 96) if patch_size is None else patch_size)
    rank = int(settings.get("rank", 64) if rank is None else rank)
    phase_rank = int(
        settings.get("phase_rank", 16) if phase_rank is None else phase_rank
    )
    width = int(settings.get("width", 32) if width is None else width)
    learning_rate = float(
        settings.get("learning_rate", 2.0e-4)
        if learning_rate is None
        else learning_rate
    )
    minimum_learning_rate = float(
        settings.get("minimum_learning_rate", learning_rate)
    )
    warmup_epochs = int(
        settings.get("warmup_epochs", 0)
        if warmup_epochs is None
        else warmup_epochs
    )
    # Validate the effective schedule before loading the dataset or parent.  In
    # particular, short feasibility runs must explicitly disable a longer
    # production warmup rather than failing after expensive data preparation.
    _cpadc_learning_rate_for_epoch(
        1,
        total_epochs=epochs,
        maximum_learning_rate=learning_rate,
        minimum_learning_rate=minimum_learning_rate,
        warmup_epochs=warmup_epochs,
    )
    physics_point_count = int(
        settings.get("physics_point_count", 256)
        if physics_point_count is None
        else physics_point_count
    )
    rank_chunk_size = int(
        settings.get("rank_chunk_size", 2)
        if rank_chunk_size is None
        else rank_chunk_size
    )
    effective_settings = {
        "per_family": per_family,
        "epochs": epochs,
        "patch_size": patch_size,
        "rank": rank,
        "phase_rank": phase_rank,
        "width": width,
        "learning_rate": learning_rate,
        "minimum_learning_rate": minimum_learning_rate,
        "warmup_epochs": warmup_epochs,
        "physics_point_count": physics_point_count,
        "rank_chunk_size": rank_chunk_size,
        "smoke": bool(smoke),
        "optimizer": str(settings.get("optimizer", "muon_adamw")),
        "muon_lr_scale": float(settings.get("muon_lr_scale", 20.0)),
        "trust_learning_rate_scale": float(
            settings.get("trust_learning_rate_scale", 1.0)
        ),
    }
    config = dict(config)
    config["cpadc_effective"] = effective_settings
    source_h5 = Path(str(config["source_h5"])).expanduser()
    parent_path = Path(str(config["parent_checkpoint"])).expanduser()
    if not source_h5.is_file():
        raise FileNotFoundError(source_h5)
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    parent_checkpoint_sha256 = _sha256(parent_path)
    dataset_numerical_contract = validate_dataset_cpml_contract(
        source_h5,
        saved_grid_cpml=cpml_config.as_dict(),
        saved_dt_s=float(config.get("dt_s", 0.0025)),
        saved_dx_m=float(config.get("dx_m", 10.0)),
        saved_dz_m=float(config.get("dz_m", 10.0)),
    )
    implementation_digests = cpadc_implementation_digests(ROOT)

    manifest = build_manifest(source_h5)
    rank_id, world_size, target_device, initialized_here = _distributed_context(device)
    seed = int(config.get("seed", 372))
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    loss_family_weights = _family_weights(config, "family_loss_weights")
    sampling_family_weights = _family_weights(config, "family_sampling_weights")

    requested_per_family = max(1, int(per_family))
    if smoke:
        requested_per_family = max(1, math.ceil(world_size / len(ALLOWED_MEDIUM_TYPES)))
    episodes = list(
        build_balanced_meta_episodes(
            manifest,
            split="train",
            per_family=requested_per_family,
            seed=seed,
        )
    )
    if smoke:
        episodes = episodes[: max(1, world_size)]
    if len(episodes) % world_size:
        raise ValueError(
            "global CPADC episodes must be divisible by WORLD_SIZE: "
            f"episodes={len(episodes)}, world_size={world_size}"
        )
    local_episodes = episodes[rank_id::world_size]
    if not local_episodes:
        raise ValueError("this rank received no CPADC train episodes")

    parent, normalizer = _load_parent(config, manifest, target_device)
    background_provider = _load_background_provider(
        config, (episode.sample_id for episode in local_episodes)
    )
    dataset = GuardedOnsetDataset(
        source_h5,
        manifest,
        split="train",
        travel_time_h5=config.get("travel_time_h5"),
    )
    records_by_source = {
        record.source_index: index
        for index, record in enumerate(dataset.records.records)
    }
    cached: list[dict[str, object]] = []
    pressure_scale = float(normalizer.metadata.pressure_scale_pa)
    try:
        with h5py.File(source_h5, "r", swmr=True) as h5:
            for local_index, metadata in enumerate(local_episodes):
                record = dataset[records_by_source[metadata.source_index]]
                velocity = record.velocity_mps.to(target_device).unsqueeze(0)
                source = record.source_parameters.to(target_device).unsqueeze(0)
                source_map = record.source_map.to(target_device).unsqueeze(0)
                time_s = record.time_s.to(target_device)
                all_indices = torch.arange(len(time_s), dtype=torch.long)
                truth_physical = torch.as_tensor(
                    np.asarray(h5["wavefield"][metadata.source_index], dtype=np.float32),
                    device=target_device,
                ).unsqueeze(0)
                truth = normalizer.encode_pressure(truth_physical, source[:, 4])
                observed = normalizer.encode_pressure(
                    record.observed_wavefield.to(target_device).unsqueeze(0),
                    source[:, 4],
                )
                parent_field = _parent_normalized_at_times(
                    parent,
                    normalizer,
                    record,
                    target_device,
                    time_s,
                    frame_indices=all_indices,
                    background_provider=background_provider,
                    time_block=int(config.get("deployment_time_block", 32)),
                )
                bridge = make_onset_bridge(
                    velocity,
                    source,
                    record.observed_wavefield.to(target_device).unsqueeze(0),
                    record.observed_indices,
                    time_s,
                    source_map=source_map,
                    steps=int(config.get("bridge_steps", 4)),
                    dx_m=float(config.get("dx_m", 10.0)),
                    dz_m=float(config.get("dz_m", 10.0)),
                    device=target_device,
                )
                if not bridge.valid:
                    raise RuntimeError(
                        f"synthetic bridge failed for {record.sample_id}: {bridge.failure_reason}"
                    )
                bridge_normalized = normalizer.encode_pressure(
                    bridge.frames, source[:, 4]
                )
                cached.append(
                    {
                        "sample_index": local_index * world_size + rank_id,
                        "sample_id": record.sample_id,
                        "medium_type": record.medium_type,
                        "observed_indices": record.observed_indices,
                        "bridge_indices": bridge.time_indices,
                        "velocity": _cache_tensor(velocity),
                        "source": _cache_tensor(source),
                        "source_map": _cache_tensor(source_map),
                        "time_s": _cache_tensor(time_s),
                        "truth": _cache_tensor(truth),
                        "observed": _cache_tensor(observed),
                        "parent": _cache_tensor(parent_field),
                        "bridge": _cache_tensor(bridge_normalized),
                        "field_scale_pa": _cache_tensor(
                            source[:, 4] * pressure_scale
                        ),
                    }
                )
    finally:
        dataset.close()
        if background_provider is not None:
            background_provider.close()
    parent.to("cpu")
    del parent
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    generator = CausalErrorBasisGenerator(
        rank=int(rank),
        phase_rank=int(phase_rank),
        width=int(width),
        ramp_steps=int(config.get("causal_ramp_steps", 4)),
    ).to(target_device)
    trainable = list(generator.parameters())
    _broadcast_trainable_parameters(trainable, world_size=world_size)
    beta1 = float(settings.get("beta1", 0.9))
    beta2 = float(settings.get("beta2", 0.999))
    epsilon = float(settings.get("eps", 1.0e-8))
    weight_decay = float(settings.get("weight_decay", 1.0e-5))
    gradient_clip = float(settings.get("gradient_clip", 1.0))
    trust_learning_rate_scale = float(
        settings.get("trust_learning_rate_scale", 1.0)
    )
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("CPADC AdamW betas must lie in [0,1)")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("CPADC AdamW epsilon must be finite and positive")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0.0:
        raise ValueError("CPADC gradient clip must be finite and positive")
    if (
        not math.isfinite(trust_learning_rate_scale)
        or trust_learning_rate_scale <= 0.0
    ):
        raise ValueError("CPADC trust learning-rate scale must be finite and positive")
    optimizer_name = str(settings.get("optimizer", "muon_adamw"))
    if optimizer_name == "muon_adamw":
        if trust_learning_rate_scale != 1.0:
            raise ValueError(
                "trust learning-rate scaling currently requires the adamw optimizer"
            )
        optimizer = build_module_muon_adamw(
            generator,
            adamw_lr=float(learning_rate),
            muon_lr_scale=float(settings.get("muon_lr_scale", 20.0)),
            weight_decay=weight_decay,
            momentum=float(settings.get("muon_momentum", 0.95)),
            nesterov=bool(settings.get("muon_nesterov", True)),
            ns_steps=int(settings.get("muon_ns_steps", 5)),
            adamw_betas=(beta1, beta2),
            adamw_eps=epsilon,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            _cpadc_adamw_parameter_groups(
                generator.named_parameters(),
                weight_decay=weight_decay,
                learning_rate=float(learning_rate),
                trust_learning_rate_scale=trust_learning_rate_scale,
            ),
            lr=float(learning_rate),
            betas=(beta1, beta2),
            eps=epsilon,
        )
    else:
        raise ValueError("CPADC optimizer must be adamw or muon_adamw")
    initial_group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    inner_raw = config.get("online_inner_weights", {}) or {}
    outer_raw = config.get("outer_weights", {}) or {}
    inner_weights = DefectCorrectionWeights(
        defect=float(inner_raw.get("defect", 1.0)),
        observed=float(inner_raw.get("observed", 0.0)),
        bridge=float(inner_raw.get("bridge", 0.5)),
        prior=float(inner_raw.get("prior", 1.0e-4)),
    )
    online_defect_design = str(
        config.get("online_defect_design", "materialized")
    ).strip().lower()
    online_defect_time_order = int(config.get("online_defect_time_order", 4))
    online_defect_test_function_width = int(
        config.get("online_defect_test_function_width", 1)
    )
    online_physics_point_count = int(config.get("online_physics_point_count", 1))
    online_physics_sampling = str(
        config.get("online_physics_sampling", "fixed")
    ).strip().lower()
    if (
        online_defect_test_function_width < 1
        or online_defect_test_function_width % 2 == 0
    ):
        raise ValueError(
            "online_defect_test_function_width must be a positive odd integer"
        )
    if float(inner_weights.defect) > 0.0 and (
        online_defect_design != "sparse_interior"
        or online_defect_time_order != 2
        or online_physics_point_count <= 0
        or online_physics_sampling != "rad"
    ):
        raise ValueError(
            "online CPADC defect requires sparse_interior, time_order=2, "
            "positive point count, and RAD sampling"
        )
    outer_weights = MetaDefectLossWeights(
        full_field=float(outer_raw.get("full_field", 1.0)),
        late=float(outer_raw.get("late", 0.5)),
        temporal_difference=float(outer_raw.get("temporal_difference", 0.1)),
        spectrum=float(outer_raw.get("spectrum", 0.1)),
        coefficient=float(outer_raw.get("coefficient", 1.0e-5)),
    )
    outer_physics_weight = float(outer_raw.get("physics", 0.0))
    outer_physics_time_count = int(outer_raw.get("physics_time_count", 64))
    outer_physics_window_strategy = str(
        outer_raw.get(
            "physics_window_strategy",
            "nearest_side_cpml_peak_arrival_causal_prefix_v1",
        )
    )
    outer_physics_prearrival_frames = int(
        outer_raw.get("physics_prearrival_frames", 8)
    )
    outer_physics_boundary_band = int(
        outer_raw.get("physics_boundary_band_cells", 12)
    )
    outer_physics_boundary_weight = float(
        outer_raw.get("physics_boundary_weight", 4.0)
    )
    if outer_physics_weight < 0.0 or outer_physics_time_count < 3:
        raise ValueError("CPADC outer physics settings are invalid")
    if (
        outer_physics_window_strategy
        != "nearest_side_cpml_peak_arrival_causal_prefix_v1"
    ):
        raise ValueError("unsupported CPADC CPML physics window strategy")

    output = Path(output_dir).expanduser().resolve()
    if rank_id == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_identity.json").write_text(
            json.dumps(
                {
                    "schema": "cpadc_run_identity_v1",
                    "schema_version": CPADC_SCHEMA_VERSION,
                    "config": str(config_file),
                    "config_digest": _config_digest(config),
                    "manifest_digest": manifest.digest,
                    "dataset_numerical_contract": dataset_numerical_contract,
                    "implementation_digests": implementation_digests,
                    "parent_checkpoint": str(parent_path.resolve()),
                    "rank": int(rank),
                    "phase_rank": int(phase_rank),
                    "first_order_inner_solve": not bool(
                        config.get("differentiate_inner_solve", True)
                    ),
                    "differentiable_inner_solve": bool(
                        config.get("differentiate_inner_solve", True)
                    ),
                    "defect_contract": "source_consistent_saved_grid_lwc84_cfs_cpml_v3",
                    "exact_fine_generator_residual": False,
                    "effective_settings": effective_settings,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if world_size > 1:
        dist.barrier()
    metrics_path = output / "metrics.jsonl"
    best_error = math.inf
    best_improvement = -math.inf
    best_epoch = 0
    started = time.perf_counter()
    dt_default = float(config.get("dt_s", 0.0025))
    dx = float(config.get("dx_m", 10.0))
    dz = float(config.get("dz_m", 10.0))

    try:
        for epoch in range(int(epochs)):
            generator.train()
            scheduled_learning_rate = _cpadc_learning_rate_for_epoch(
                epoch + 1,
                total_epochs=int(epochs),
                maximum_learning_rate=float(learning_rate),
                minimum_learning_rate=minimum_learning_rate,
                warmup_epochs=warmup_epochs,
            )
            schedule_factor = scheduled_learning_rate / float(learning_rate)
            for parameter_group, initial_group_lr in zip(
                optimizer.param_groups, initial_group_lrs, strict=True
            ):
                parameter_group["lr"] = initial_group_lr * schedule_factor
            schedule = _weighted_epoch_indices(
                cached,
                family_weights=sampling_family_weights,
                seed=seed,
                epoch=epoch,
            )
            local_loss = 0.0
            local_before = 0.0
            local_after = 0.0
            local_accepted = 0.0
            local_physics = 0.0
            local_trust = 0.0
            local_unconstrained_ratio = 0.0
            local_projection_scale = 0.0
            local_trust_gradient_norm = 0.0
            local_oracle_step = 0.0
            local_oracle_error = 0.0
            local_positive_oracle = 0.0
            local_directional_cosine = 0.0
            local_count = 0
            for cached_index in schedule:
                sample = cached[cached_index]
                parent_full = sample["parent"].to(target_device)
                truth_full = sample["truth"].to(target_device)
                height, spatial_width = parent_full.shape[-2:]
                if int(patch_size) < max(height, spatial_width):
                    raise ValueError(
                        "CPML-aware CPADC must use the complete physical grid; "
                        f"patch_size={patch_size}, grid={height}x{spatial_width}"
                    )
                z_slice, x_slice = slice(0, height), slice(0, spatial_width)
                parent_field = parent_full[..., z_slice, x_slice]
                truth = truth_full[..., z_slice, x_slice]
                velocity = sample["velocity"].to(target_device)[..., z_slice, x_slice]
                source = sample["source"].to(target_device)
                source_map = sample["source_map"].to(target_device)[..., z_slice, x_slice]
                observed = sample["observed"].to(target_device)[..., z_slice, x_slice]
                bridge = sample["bridge"].to(target_device)[..., z_slice, x_slice]
                times = sample["time_s"].to(target_device)
                field_scale_pa = sample["field_scale_pa"].to(target_device)
                observed_indices = tuple(int(value) for value in sample["observed_indices"])
                step_dt = (
                    float((times[1:] - times[:-1]).mean())
                    if len(times) > 1
                    else dt_default
                )
                with torch.no_grad():
                    # The generator and deployed CPU solve share this cheap
                    # interior LWC-4 summary.  Full CFS-CPML is applied once to
                    # the combined candidate below, never once per rank mode.
                    parent_defect, defect_scale = lwc84_discrete_defect(
                        parent_field,
                        velocity,
                        dt=step_dt,
                        dx=dx,
                        dz=dz,
                        observed_indices=observed_indices,
                        source_parameters=source,
                        source_map=source_map,
                        time_s=times,
                        field_scale_pa=field_scale_pa,
                        normalize=False,
                        return_scale=True,
                    )
                    point_seed = (
                        seed + epoch * 1_000_003 + int(sample["sample_index"])
                    )
                    if float(inner_weights.defect) > 0.0:
                        point_generator = torch.Generator(
                            device=parent_defect.device
                        ).manual_seed(point_seed)
                        physics_points = build_rad_physics_points(
                            parent_defect,
                            observed_indices,
                            count=online_physics_point_count,
                            k=float(config.get("rad_k", 1.0)),
                            c=float(config.get("rad_c", 1.0)),
                            time_tilt=float(config.get("rad_time_tilt", 1.5)),
                            generator=point_generator,
                        )
                    else:
                        physics_points = build_fixed_physics_points(
                            len(times),
                            observed_indices,
                            count=1,
                            seed=point_seed,
                        )
                basis = generator(
                    parent_field,
                    velocity,
                    observed,
                    source,
                    times,
                    observed_indices,
                    parent_defect=parent_defect,
                    defect_scale=defect_scale,
                )
                differentiate_inner = bool(
                    config.get("differentiate_inner_solve", True)
                )
                inner_context = (
                    torch.enable_grad() if differentiate_inner else torch.no_grad()
                )
                with inner_context:
                    solve_basis = (
                        basis if differentiate_inner else basis.detached()
                    )
                    observed_probe_basis = (
                        None
                        if float(inner_weights.observed) == 0.0
                        else causal_observation_probe_basis(solve_basis)
                    )
                    inner = solve_causal_defect_correction(
                        solve_basis,
                        parent_field,
                        velocity,
                        source,
                        source_map,
                        times,
                        observed_indices,
                        physics_points,
                        field_scale_pa=field_scale_pa,
                        observed_wavefield=observed,
                        observed_design_basis=observed_probe_basis,
                        bridge_wavefield=bridge,
                        bridge_indices=sample["bridge_indices"],
                        weights=inner_weights,
                        dt=step_dt,
                        dx=dx,
                        dz=dz,
                        spatial_stride=int(config.get("inner_spatial_stride", 4)),
                        rank_chunk_size=int(rank_chunk_size),
                        time_order=online_defect_time_order,
                        defect_design=online_defect_design,
                        defect_test_function_width=(
                            online_defect_test_function_width
                        ),
                        cpml_config=None,
                        solve_device="cpu",
                        minimum_relative_improvement=0.0,
                        maximum_condition_number=float(
                            config.get("training_maximum_condition_number", 1.0e30)
                        ),
                        minimum_effective_design_rank=int(
                            config.get("training_minimum_effective_design_rank", 1)
                        ),
                        maximum_correction_ratio=float(
                            config.get("training_maximum_correction_ratio", 1.0e30)
                        ),
                    )
                    if not bool(inner.accepted.all()):
                        raise FloatingPointError(
                            "CPADC inner solve was numerically rejected during offline "
                            "training after correction-energy projection"
                        )
                coefficients = (
                    inner.coefficients
                    if differentiate_inner
                    else _first_order_projected_coefficients(
                        basis,
                        inner,
                        maximum_correction_ratio=float(
                            config.get(
                                "training_maximum_correction_ratio", 1.0e30
                            )
                        ),
                    )
                )
                candidate = parent_field + basis.combine(
                    coefficients.to(parent_field.device)
                )
                # Train-only line-search diagnostic.  This does not alter the
                # deployed coefficients or the gradient path.  It asks whether
                # the causal CPADC direction is useful but over-stepped (oracle
                # alpha in (0,1)), or directionally wrong (oracle alpha = 0).
                with torch.no_grad():
                    correction = (candidate - parent_field).detach().flatten(1)
                    parent_residual = (parent_field - truth).detach().flatten(1)
                    correction_energy = correction.square().sum(dim=1)
                    descent_inner = -(parent_residual * correction).sum(dim=1)
                    oracle_step = torch.clamp(
                        descent_inner / correction_energy.clamp_min(1.0e-30),
                        min=0.0,
                        max=1.0,
                    )
                    oracle_candidate = parent_field + oracle_step.view(
                        -1, 1, 1, 1
                    ) * (candidate.detach() - parent_field)
                    oracle_error = _relative_l2(oracle_candidate, truth)
                    residual_norm = parent_residual.square().sum(dim=1).sqrt()
                    correction_norm = correction_energy.sqrt()
                    directional_cosine = descent_inner / (
                        residual_norm * correction_norm
                    ).clamp_min(1.0e-30)
                terms = meta_defect_correction_loss(
                    candidate,
                    truth,
                    coefficients,
                    weights=outer_weights,
                    late_start_fraction=float(config.get("late_start_fraction", 0.5)),
                    future_start_index=int(observed_indices[1]) + 1,
                )
                physics_start, physics_stop = _cpml_physics_window(
                    times,
                    source,
                    velocity,
                    observed_indices,
                    dx_m=dx,
                    residual_count=min(
                        int(outer_physics_time_count), candidate.shape[1] - 2
                    ),
                    prearrival_frames=outer_physics_prearrival_frames,
                )
                gradient_prefix = max(0, physics_start - 1)
                physics_defect = lwc84_discrete_defect(
                    candidate[:, :physics_stop],
                    velocity,
                    dt=step_dt,
                    dx=dx,
                    dz=dz,
                    observed_indices=(-1, physics_start - 1),
                    source_parameters=source,
                    source_map=source_map,
                    time_s=times[:physics_stop],
                    field_scale_pa=field_scale_pa,
                    normalize=True,
                    cpml_config=cpml_config,
                    cpml_gradient_start_index=gradient_prefix,
                )
                physics_loss = _weighted_cpml_defect_loss(
                    physics_defect,
                    boundary_band_cells=outer_physics_boundary_band,
                    boundary_weight=outer_physics_boundary_weight,
                )
                family_weight = loss_family_weights[str(sample["medium_type"])]
                loss = (
                    terms["total"] + outer_physics_weight * physics_loss
                ) * float(family_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite CPADC outer loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _average_gradients(trainable, world_size=world_size)
                trust_gradient_norm = math.sqrt(
                    sum(
                        float(parameter.grad.detach().float().square().sum())
                        for parameter in generator.trust_head.parameters()
                        if parameter.grad is not None
                    )
                )
                torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
                optimizer.step()

                local_loss += float(loss.detach())
                local_before += float(_relative_l2(parent_field, truth).mean())
                local_after += float(_relative_l2(candidate.detach(), truth).mean())
                local_accepted += float(inner.accepted.float().mean())
                local_physics += float(physics_loss.detach())
                local_trust += float(basis.trust_fraction.detach().mean())
                local_unconstrained_ratio += float(
                    inner.unconstrained_correction_ratio.detach().mean()
                )
                local_projection_scale += float(
                    inner.projection_scale.detach().mean()
                )
                local_trust_gradient_norm += trust_gradient_norm
                local_oracle_step += float(oracle_step.mean())
                local_oracle_error += float(oracle_error.mean())
                local_positive_oracle += float((oracle_step > 0.0).float().mean())
                local_directional_cosine += float(directional_cosine.mean())
                local_count += 1

            totals = torch.tensor(
                [
                    local_loss,
                    local_before,
                    local_after,
                    local_accepted,
                    local_physics,
                    local_trust,
                    local_unconstrained_ratio,
                    local_projection_scale,
                    local_trust_gradient_norm,
                    local_oracle_step,
                    local_oracle_error,
                    local_positive_oracle,
                    local_directional_cosine,
                    float(local_count),
                ],
                dtype=torch.float64,
                device=target_device,
            )
            if world_size > 1:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            count = max(float(totals[13]), 1.0)
            mean_loss = float(totals[0] / count)
            mean_before = float(totals[1] / count)
            mean_after = float(totals[2] / count)
            accepted_fraction = float(totals[3] / count)
            mean_physics_loss = float(totals[4] / count)
            mean_trust_fraction = float(totals[5] / count)
            mean_unconstrained_ratio = float(totals[6] / count)
            mean_projection_scale = float(totals[7] / count)
            mean_trust_gradient_norm = float(totals[8] / count)
            mean_oracle_step = float(totals[9] / count)
            mean_oracle_error = float(totals[10] / count)
            positive_oracle_fraction = float(totals[11] / count)
            mean_directional_cosine = float(totals[12] / count)
            relative_improvement = (
                (mean_before - mean_after) / max(mean_before, 1.0e-12)
            )
            event = {
                "event": "epoch_complete",
                "epoch": epoch + 1,
                "epochs": int(epochs),
                "mean_outer_loss": mean_loss,
                "mean_train_patch_relative_l2_before": mean_before,
                "mean_train_patch_relative_l2_after": mean_after,
                "relative_improvement": relative_improvement,
                "inner_acceptance_fraction": accepted_fraction,
                "mean_lwc84_cfs_cpml_loss": mean_physics_loss,
                "mean_trust_fraction": mean_trust_fraction,
                "mean_unconstrained_correction_ratio": mean_unconstrained_ratio,
                "mean_projection_scale": mean_projection_scale,
                "mean_trust_gradient_norm": mean_trust_gradient_norm,
                "mean_oracle_step_fraction": mean_oracle_step,
                "mean_oracle_train_patch_relative_l2": mean_oracle_error,
                "oracle_relative_improvement": (
                    (mean_before - mean_oracle_error)
                    / max(mean_before, 1.0e-12)
                ),
                "positive_oracle_step_fraction": positive_oracle_fraction,
                "mean_descent_direction_cosine": mean_directional_cosine,
                "offline_physics_weight": outer_physics_weight,
                "offline_physics_time_count": outer_physics_time_count,
                "offline_physics_window_strategy": outer_physics_window_strategy,
                "offline_physics_prearrival_frames": outer_physics_prearrival_frames,
                "offline_physics_boundary_band_cells": outer_physics_boundary_band,
                "offline_physics_boundary_weight": outer_physics_boundary_weight,
                "coefficient_solve_device": "cpu",
                "elapsed_seconds": time.perf_counter() - started,
                "world_size": world_size,
                "rank": int(rank),
                "phase_rank": int(phase_rank),
                "learning_rate": scheduled_learning_rate,
                "optimizer": optimizer_name,
                "optimizer_group_learning_rates": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "future_truth_scope": "outer_train_loss_only",
            }
            if rank_id == 0:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event, sort_keys=True), flush=True)
                payload = _checkpoint_payload(
                    generator,
                    config=config,
                    manifest_digest=manifest.digest,
                    parent_checkpoint=parent_path,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    dataset_numerical_contract=dataset_numerical_contract,
                    implementation_digests=implementation_digests,
                    epoch=epoch + 1,
                    event=event,
                )
                torch.save(payload, output / "latest.pt")
                # Patches vary by epoch, so absolute L2 values are not directly
                # comparable.  Select on paired improvement over the frozen
                # parent evaluated on the exact same epoch/task patches.
                if relative_improvement > best_improvement:
                    best_improvement = relative_improvement
                    best_error = mean_after
                    best_epoch = epoch + 1
                    torch.save(payload, output / "best.pt")
        if rank_id == 0:
            terminal = {
                "status": "complete",
                "schema": "cpadc_training_terminal_v1",
                "best_epoch": best_epoch,
                "best_train_patch_relative_l2": best_error,
                "best_train_patch_relative_improvement": best_improvement,
                "checkpoint": str((output / "best.pt").resolve()),
                "checkpoint_sha256": _sha256(output / "best.pt"),
                "same_protocol_validation_passed": False,
                "claim": "offline training complete; accuracy not yet validated",
            }
            (output / "terminal.json").write_text(
                json.dumps(terminal, indent=2, sort_keys=True) + "\n"
            )
        if world_size > 1:
            dist.barrier()
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
    return output / "best.pt"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--parent-checkpoint-identity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-family", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--phase-rank", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--physics-point-count", type=int)
    parser.add_argument("--rank-chunk-size", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        config = yaml.safe_load(Path(args.config).read_text())
        parent = args.parent_checkpoint or config.get("parent_checkpoint")
        payload = vars(args).copy()
        payload.update(
            {
                "config_exists": Path(args.config).is_file(),
                "source_h5_exists": Path(str(config.get("source_h5", ""))).is_file(),
                "parent_checkpoint": parent,
                "parent_checkpoint_exists": bool(parent) and Path(str(parent)).is_file(),
                "gpu_launch_authorized": False,
            }
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    train_causal_defect_basis(
        args.config,
        output_dir=args.output_dir,
        parent_checkpoint=args.parent_checkpoint,
        parent_checkpoint_identity=args.parent_checkpoint_identity,
        device=args.device,
        per_family=args.per_family,
        epochs=args.epochs,
        patch_size=args.patch_size,
        rank=args.rank,
        phase_rank=args.phase_rank,
        width=args.width,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        physics_point_count=args.physics_point_count,
        rank_chunk_size=args.rank_chunk_size,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
