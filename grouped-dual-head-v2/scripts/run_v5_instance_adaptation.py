#!/usr/bin/env python3
"""Run guarded V5 adaptation and sealed full-field evaluation."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from collections.abc import Mapping

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, V3DataManifest, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.adapters import (
    ADAPTER_SCHEMA_VERSION,
    OnsetAdaptedV5,
)
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.chonknoris import (
    ReducedChonknorisConfig,
)
from saved_time_phase_operator_v4.instance_adaptation.trainer import adapt_instance
from saved_time_phase_operator_v4.instance_adaptation.losses import PhysicsSampling
from saved_time_phase_operator_v4.instance_adaptation.visualization import (
    plot_receiver_comparison,
    plot_wavefield_comparison,
)
from scripts.evaluate_v5_instance_adaptation import evaluate_after_adaptation, write_report
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_probe import _model
from scripts.train_saved_time_v4_full_support import _load_context, _load_parent_model
from saved_time_phase_operator_v4.probe import ProbeVariant


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_contract_path(
    contract: Mapping[str, object],
    *,
    path_key: str,
    digest_key: str,
    label: str,
) -> Path:
    path = Path(str(contract.get(path_key, ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"external evaluation {label} is missing")
    expected = str(contract.get(digest_key, ""))
    if len(expected) != 64 or _sha256_file(path) != expected:
        raise ValueError(f"external evaluation {label} hash binding mismatch")
    return path


def _validate_vds_shard_inventory(
    contract: Mapping[str, object], evaluation_dataset: Path
) -> int:
    raw = contract.get("evaluation_vds_shards")
    if not isinstance(raw, list) or not raw:
        raise ValueError("external evaluation VDS shard inventory is missing")
    with h5py.File(evaluation_dataset, "r", swmr=True) as handle:
        try:
            mapped = [
                str(Path(str(value)).resolve())
                for value in json.loads(str(handle.attrs["vds_source_shards"]))
            ]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("external evaluation VDS shard mapping is invalid") from error
    registered: list[str] = []
    for raw_item in raw:
        if not isinstance(raw_item, Mapping):
            raise ValueError("external evaluation VDS shard entry is invalid")
        item = dict(raw_item)
        shard = Path(str(item.get("path", ""))).resolve()
        sidecar = Path(str(item.get("sidecar", ""))).resolve()
        registered.append(str(shard))
        if not shard.is_file() or int(item.get("byte_count", -1)) != shard.stat().st_size:
            raise ValueError("external evaluation VDS shard size binding mismatch")
        if not sidecar.is_file() or _sha256_file(sidecar) != str(
            item.get("sidecar_sha256", "")
        ):
            raise ValueError("external evaluation VDS shard sidecar binding mismatch")
        if sidecar.read_text(encoding="utf-8").strip() != str(item.get("sha256", "")):
            raise ValueError("external evaluation VDS shard digest binding mismatch")
        if _sha256_file(shard) != str(item.get("sha256", "")):
            raise ValueError("external evaluation VDS shard content binding mismatch")
    if registered != mapped:
        raise ValueError("external evaluation VDS shard inventory differs from mapping")
    return len(registered)


def build_instance_manifest(
    manifest: V3DataManifest,
    *,
    seed: int = 17,
    per_family: int = 3,
    excluded_group_ids=(),
    split: str = "validation",
):
    """Select distinct media from one split with deterministic family balance."""
    selection_split = str(split)
    if selection_split not in {"train", "validation"}:
        raise ValueError("instance selection split must be train or validation")
    rng = np.random.default_rng(int(seed))
    excluded = {str(value) for value in excluded_group_ids}
    selected = []
    for family in ALLOWED_MEDIUM_TYPES:
        records = [
            record
            for record in manifest.records
            if record.split == selection_split
            and record.medium_type == family
            and record.group_id not in excluded
        ]
        rng.shuffle(records)
        seen_groups: set[str] = set()
        for record in records:
            if record.group_id in seen_groups:
                continue
            selected.append(record)
            seen_groups.add(record.group_id)
            if len([item for item in selected if item.medium_type == family]) == int(per_family):
                break
        count = sum(item.medium_type == family for item in selected)
        if count != int(per_family):
            raise ValueError(
                f"{selection_split} family {family} has only {count} distinct media"
            )
    if len({item.group_id for item in selected}) != len(selected):
        raise ValueError("instance manifest contains duplicate media groups")
    return tuple(selected)


def build_all_validation_manifest(manifest: V3DataManifest):
    """Return the complete validation split in manifest order."""

    rows = tuple(record for record in manifest.records if record.split == "validation")
    if not rows or {record.medium_type for record in rows} != set(ALLOWED_MEDIUM_TYPES):
        raise ValueError("complete validation manifest must contain all medium families")
    return rows


def resolve_saved_time_parent_config(config) -> dict[str, object] | None:
    """Bind an adapter run to a selected checkpoint of a saved-time operator."""

    path = config.get("parent_operator_config")
    if path is None:
        return None
    if not config.get("parent_checkpoint") or not config.get("parent_checkpoint_identity"):
        raise ValueError(
            "saved-time adaptation requires parent checkpoint and run identity"
        )
    resolved = yaml.safe_load(Path(str(path)).read_text())
    if not isinstance(resolved, dict):
        raise ValueError("parent operator config must contain a mapping")
    resolved["parent_checkpoint"] = str(config["parent_checkpoint"])
    resolved["parent_checkpoint_identity"] = str(
        config["parent_checkpoint_identity"]
    )
    transfer = dict(resolved.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    resolved["checkpoint_transfer"] = transfer
    return resolved


def _apply_normalization_override(config, base):
    """Point the parent's normalizer at an explicit JSON when the config asks.

    The shared processed normalization JSON can drift from the parent checkpoint
    (e.g. a later ablation re-wrote it under a different manifest digest).  A
    ``normalization_json`` override lets a run bind to the normalizer that
    actually matches the frozen parent without touching the shared file.
    """
    override = config.get("normalization_json")
    if not override:
        return base
    import dataclasses

    return dataclasses.replace(
        base, data=dataclasses.replace(base.data, normalization_json=str(override))
    )


def _apply_parent_correction_policy(model, config) -> dict[str, float | str]:
    """Preserve a learned correction amplitude unless the branch is disabled.

    ``dense_decoder.correction_scale`` is a trained checkpoint parameter, not a
    Boolean multiplier.  The legacy configuration field accepts 1 to enable the
    learned correction and 0 for a coarse-only ablation.  Enabling must therefore
    leave the loaded checkpoint value untouched instead of replacing it with 1.
    """

    enabled = float(config.get("parent_correction_scale", 1.0))
    if enabled not in (0.0, 1.0):
        raise ValueError("saved-time parent correction scale must be 0 or 1")
    checkpoint_scale = float(model.dense_decoder.correction_scale.detach())
    if enabled == 0.0:
        with torch.no_grad():
            model.dense_decoder.correction_scale.zero_()
        return {
            "policy": "disabled_coarse_only",
            "checkpoint_scale": checkpoint_scale,
            "effective_scale": 0.0,
        }
    return {
        "policy": "enabled_preserve_checkpoint",
        "checkpoint_scale": checkpoint_scale,
        "effective_scale": checkpoint_scale,
    }


def validate_external_evaluation_contract(
    config: Mapping[str, object],
    training_manifest: V3DataManifest,
    evaluation_manifest: V3DataManifest,
) -> dict[str, object]:
    """Authorize a checkpoint on a new, hash-bound, evaluation-only dataset."""
    if training_manifest.digest == evaluation_manifest.digest:
        return {
            "external": False,
            "training_manifest_digest": training_manifest.digest,
            "evaluation_manifest_digest": evaluation_manifest.digest,
        }
    raw = config.get("external_evaluation_contract")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "external evaluation dataset requires external_evaluation_contract"
        )
    contract = dict(raw)
    role = str(contract.get("role", ""))
    if role not in {"frozen_validation", "independent_test_id"}:
        raise ValueError("external evaluation role is not frozen validation/test_id")
    if (
        str(contract.get("training_manifest_digest", ""))
        != training_manifest.digest
        or str(contract.get("evaluation_manifest_digest", ""))
        != evaluation_manifest.digest
    ):
        raise ValueError("external evaluation manifest binding mismatch")
    if contract.get("evaluation_only") is not True or contract.get("training_forbidden") is not True:
        raise ValueError("external dataset must be registered evaluation-only")
    for name in ("time_s", "x_m", "z_m"):
        if tuple(getattr(training_manifest, name)) != tuple(
            getattr(evaluation_manifest, name)
        ):
            raise ValueError(f"external evaluation coordinate protocol differs: {name}")

    for name in ("sample_id", "group_id", "sample_sha256"):
        training_values = {str(getattr(row, name)) for row in training_manifest.records}
        evaluation_values = {
            str(getattr(row, name)) for row in evaluation_manifest.records
        }
        if training_values & evaluation_values:
            raise ValueError(f"external evaluation {name} overlaps training data")

    registered_dataset = Path(str(contract.get("evaluation_dataset", ""))).resolve()
    if registered_dataset != Path(evaluation_manifest.source_path).resolve():
        raise ValueError("external evaluation registered dataset path mismatch")
    shard_count = _validate_vds_shard_inventory(contract, registered_dataset)
    audit_path = _bound_contract_path(
        contract,
        path_key="postgeneration_sample_sha256_audit",
        digest_key="postgeneration_sample_sha256_audit_sha256",
        label="sample-hash audit",
    )
    audit = json.loads(audit_path.read_text())
    if not (
        isinstance(audit, dict)
        and audit.get("status") == "passed"
        and audit.get("passed") is True
        and int(audit.get("intersection_count", -1)) == 0
        and audit.get("future_truth_opened_by_evaluator") is False
        and Path(str(audit.get("candidate_dataset", ""))).resolve()
        == Path(evaluation_manifest.source_path).resolve()
    ):
        raise ValueError("external evaluation sample-hash audit did not pass")
    pretruth_path = _bound_contract_path(
        contract,
        path_key="pretruth_historical_overlap_audit",
        digest_key="pretruth_historical_overlap_audit_sha256",
        label="pre-truth audit",
    )
    pretruth = json.loads(pretruth_path.read_text())
    intersection_counts = (
        pretruth.get("intersection_counts", {}) if isinstance(pretruth, dict) else {}
    )
    if not (
        isinstance(pretruth, dict)
        and pretruth.get("status") == "passed"
        and pretruth.get("passed") is True
        and intersection_counts
        and all(int(value) == 0 for value in intersection_counts.values())
    ):
        raise ValueError("external evaluation pre-truth overlap audit did not pass")
    internal_path = _bound_contract_path(
        contract,
        path_key="internal_split_overlap_audit",
        digest_key="internal_split_overlap_audit_sha256",
        label="internal-split audit",
    )
    internal = json.loads(internal_path.read_text())
    pairwise = (
        internal.get("pairwise_intersection_counts", {})
        if isinstance(internal, dict)
        else {}
    )
    if not (
        isinstance(internal, dict)
        and internal.get("status") == "passed"
        and internal.get("passed") is True
        and pairwise
        and all(
            int(count) == 0
            for intersections in pairwise.values()
            for count in intersections.values()
        )
    ):
        raise ValueError("external evaluation internal-split audit did not pass")
    travel_path = _bound_contract_path(
        contract,
        path_key="travel_time_h5",
        digest_key="travel_time_h5_sha256",
        label="travel-time cache",
    )
    configured_travel = Path(str(config.get("travel_time_h5", ""))).resolve()
    if configured_travel != travel_path:
        raise ValueError("external evaluation configured travel cache is not registered")
    with h5py.File(travel_path, "r", swmr=True) as handle:
        travel_source = Path(str(handle.attrs.get("source_h5", ""))).resolve()
        travel_content_sha256 = str(handle.attrs.get("content_sha256", ""))
        travel_schema = str(handle.attrs.get("schema", ""))
    if travel_source != Path(evaluation_manifest.source_path).resolve():
        raise ValueError("external evaluation travel cache source binding mismatch")
    if (
        travel_content_sha256
        != str(contract.get("travel_time_content_sha256", ""))
        or travel_schema != str(contract.get("travel_time_schema", ""))
    ):
        raise ValueError("external evaluation travel cache metadata binding mismatch")
    return {
        "external": True,
        "role": role,
        "training_manifest_digest": training_manifest.digest,
        "evaluation_manifest_digest": evaluation_manifest.digest,
        "sample_hash_audit": str(audit_path.resolve()),
        "pretruth_overlap_audit": str(pretruth_path.resolve()),
        "internal_split_overlap_audit": str(internal_path.resolve()),
        "travel_time_h5": str(travel_path.resolve()),
        "travel_time_h5_sha256": str(contract["travel_time_h5_sha256"]),
        "travel_time_content_sha256": travel_content_sha256,
        "evaluation_vds_shard_count": shard_count,
    }


def _load_parent(config, manifest, device):
    parent_kind = str(config.get("parent_kind", "saved_time_v4"))
    if parent_kind == "helmholtz_g3":
        identity_path = Path(str(config["parent_run_identity"]))
        identity = json.loads(identity_path.read_text())
        if identity.get("schema") != "helmholtz_g3_heldout_v1":
            raise ValueError("Phase4b parent identity has an incompatible schema")
        if identity.get("manifest_digest") != manifest.digest:
            raise ValueError("Phase4b parent and adapter manifest digest mismatch")

        from scripts.diagnose_capacity_ladder_overfit import build_base_config

        base = build_base_config(int(identity["width"]))
        base = _apply_normalization_override(config, base)
        variant = ProbeVariant(
            depth=int(identity["dense_depth"]),
            use_local_phase=True,
            spectral_rank=int(identity["dense_spectral_rank"]),
            modes=int(identity["dense_modes"]),
            temporal_basis_rank=0,
            family_expert_rank=0,
            local_field=True,
            local_field_residual=False,
            local_field_helmholtz_synthesis=True,
            local_field_helmholtz_synthesis_frequencies=int(
                identity["helmholtz_frequencies"]
            ),
            local_field_helmholtz_synthesis_wkb_phase=bool(
                identity["helmholtz_wkb_phase"]
            ),
            local_field_helmholtz_synthesis_rank=int(identity["helmholtz_rank"]),
            local_field_helmholtz_synthesis_late_rank=int(
                identity["helmholtz_late_rank"]
            ),
            local_field_helmholtz_synthesis_late_frequencies=int(
                identity["helmholtz_late_frequencies"]
            ),
            local_field_helmholtz_spectral_bypass=bool(
                identity["helmholtz_spectral_bypass"]
            ),
            local_field_helmholtz_spectral_bypass_per_branch=bool(
                identity["helmholtz_spectral_bypass_per_branch"]
            ),
            local_field_helmholtz_background_conditioning=bool(
                identity["helmholtz_background_conditioning"]
            ),
            local_field_helmholtz_background_sigma_cells=float(
                identity["helmholtz_background_sigma_cells"]
            ),
        )
        model = _model(base, manifest, variant).to(device)
        checkpoint = Path(str(config["parent_checkpoint"]))
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("manifest_digest") != manifest.digest:
            raise ValueError("Phase4b checkpoint manifest digest mismatch")
        if payload.get("config_digest") != identity.get("run_digest"):
            raise ValueError("Phase4b checkpoint and run identity digest mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        return model, load_normalizer(base, manifest.digest)
    if parent_kind != "saved_time_v4":
        raise ValueError(f"unsupported parent_kind: {parent_kind}")
    saved_time_config = resolve_saved_time_parent_config(config)
    if saved_time_config is not None:
        base, active_manifest, parent_identity = _load_context(saved_time_config)
        validate_external_evaluation_contract(config, active_manifest, manifest)
        model = _load_parent_model(
            saved_time_config,
            base,
            active_manifest,
            parent_identity,
            device,
        )
        _apply_parent_correction_policy(model, config)
        model.eval()
        base = _apply_normalization_override(config, base)
        return model, load_normalizer(base, active_manifest.digest)
    base = V3Config.from_yaml(config["base_config"])
    model = _model(base, manifest, ProbeVariant(depth=8, use_local_phase=True)).to(device)
    payload = torch.load(config["parent_checkpoint"], map_location=device, weights_only=False)
    if payload.get("manifest_digest") != manifest.digest:
        raise ValueError("parent checkpoint manifest digest mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    base = _apply_normalization_override(config, base)
    return model, load_normalizer(base, manifest.digest)


def _load_background_provider(config, sample_ids=()):
    cache = config.get("background_cache")
    if not cache:
        return None
    from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider

    provider = BackgroundFieldProvider(str(cache))
    requested = tuple(str(value) for value in sample_ids)
    if requested and not provider.covers(requested):
        missing = [value for value in requested if value not in set(provider.sample_ids)]
        raise ValueError(
            f"background cache misses {len(missing)} requested records: {missing[:3]}"
        )
    return provider


def _load_conditioner(
    adapter: OnsetAdaptedV5,
    checkpoint: str | Path,
    device: torch.device,
    *,
    expected_manifest_digest: str,
    expected_parent_checkpoint: str | Path | None = None,
    require_parent_binding: bool = False,
) -> dict[str, object]:
    """Load an offline-trained onset conditioner without unfreezing the parent."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(payload.get("adapter_schema_version", 0)) != ADAPTER_SCHEMA_VERSION:
        raise ValueError(
            "adapter checkpoint schema is incompatible; retrain the onset adapter"
        )
    state = payload.get("conditioner_state")
    if not isinstance(state, dict):
        raise ValueError("conditioner checkpoint lacks conditioner_state")
    checkpoint_manifest = payload.get("manifest_digest")
    if checkpoint_manifest not in (None, expected_manifest_digest):
        raise ValueError("conditioner checkpoint manifest digest mismatch")
    bound_parent = payload.get("parent_checkpoint")
    if require_parent_binding and not bound_parent:
        raise ValueError("conditioner checkpoint lacks required parent binding")
    if bound_parent and expected_parent_checkpoint is not None:
        if Path(str(bound_parent)).resolve() != Path(str(expected_parent_checkpoint)).resolve():
            raise ValueError("conditioner checkpoint is bound to a different parent")
    adapter.conditioner.load_state_dict(state, strict=True)
    residual_state = payload.get("residual_state")
    if isinstance(residual_state, dict):
        adapter.residual.load_state_dict(residual_state, strict=True)
    chonknoris_state = payload.get("chonknoris_state")
    if isinstance(chonknoris_state, dict):
        adapter.chonknoris.load_state_dict(chonknoris_state, strict=True)
        adapter._chonknoris_pretrained = True
    return {
        "checkpoint": str(checkpoint),
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "manifest_digest": payload.get("manifest_digest"),
        "source_h5_sha256": payload.get("source_h5_sha256"),
        "residual_state_loaded": isinstance(residual_state, dict),
        "chonknoris_state_loaded": isinstance(chonknoris_state, dict),
        "future_truth_used_only_for_train_episode": payload.get(
            "future_truth_used_only_for_train_episode", False
        ),
        "parent_checkpoint": bound_parent,
        "parent_checkpoint_sha256": payload.get("parent_checkpoint_sha256"),
    }


def _predict_parent(
    model,
    normalizer,
    record,
    device,
    *,
    normalized: bool = False,
    background_provider=None,
    time_block: int = 1,
):
    block = int(time_block)
    if block <= 0:
        raise ValueError("parent inference time_block must be positive")
    velocity = record.velocity_mps.to(device).unsqueeze(0)
    source = record.source_parameters.to(device).unsqueeze(0)
    source_map = record.source_map.to(device).unsqueeze(0)
    prepared = model.prepare_sources(
        model.encode_medium(velocity, normalizer), source, source_map, normalizer,
        record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
    )
    dense_grid = model.prepare_dense_grid(
        prepared,
        x_m=record.x_m.to(device),
        z_m=record.z_m.to(device),
        travel_time_s=(
            None
            if record.dense_travel_time_s is None
            else record.dense_travel_time_s.to(device).unsqueeze(0)
        ),
    )
    with torch.no_grad():
        if normalized:
            # O(1) normalized space: decoded pressure on this dataset is ~1e-9,
            # which makes an L2 residual fit numerically ill-conditioned.  The
            # deployment LoRA is trained and gated in normalized space and the
            # field is decoded only for sealed evaluation.
            field = model.dense_normalized(
                prepared,
                record.time_s.to(device),
                dense_grid=dense_grid,
                time_block=block,
            )
            if background_provider is not None:
                # A+1 hybrid solver: the model predicts the normalized SCATTERING
                # residual; the full parent field = residual + encode(P_bg).  Add the
                # (fixed, physical) smoothed-velocity background back so the adapter
                # sees the true A+1 field.  encode_pressure is linear -> exact.
                frame_idx = torch.arange(
                    record.time_s.shape[0], device="cpu"
                ).unsqueeze(0)
                pbg_phys = background_provider.physical(
                    [record.sample_id], frame_idx, device=device, dtype=field.dtype,
                )
                pbg_norm = normalizer.encode_pressure(pbg_phys, source[:, 4])
                field = field + pbg_norm
            return field
        field = model.predict_wavefield(
            prepared,
            record.time_s.to(device),
            dense_grid=dense_grid,
            time_block=block,
        )
        if background_provider is not None:
            frame_idx = torch.arange(record.time_s.shape[0], device="cpu").unsqueeze(0)
            field = field + background_provider.physical(
                [record.sample_id], frame_idx, device=device, dtype=field.dtype
            )
        return field


def _read_future_truth(source_h5: str | Path, source_index: int) -> torch.Tensor:
    """Open later truth only after the adaptation artifact is sealed."""
    with h5py.File(source_h5, "r", swmr=True) as handle:
        return torch.from_numpy(np.asarray(handle["wavefield"][int(source_index)], dtype=np.float32))


def _write_manifest(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True) + "\n")


def run(
    config_path: str | Path,
    *,
    device_name: str,
    output_dir: str | Path,
    pilot: bool = False,
    parent_checkpoint: str | Path | None = None,
    conditioner_checkpoint: str | Path | None = None,
    sample_ids=None,
    selection_split: str = "validation",
    selection_seed: int | None = None,
    per_family: int | None = None,
    save_fields: bool = True,
    write_plots: bool = True,
    deployment_lora: bool = False,
    hard_project_onset: bool | None = None,
):
    config = yaml.safe_load(Path(config_path).read_text())
    if parent_checkpoint is not None:
        config["parent_checkpoint"] = str(parent_checkpoint)
    if conditioner_checkpoint is not None:
        config["conditioner_checkpoint"] = str(conditioner_checkpoint)
    if hard_project_onset is not None:
        config["hard_project_onset"] = bool(hard_project_onset)
    manifest = build_manifest(config["source_h5"])
    selection_split = str(selection_split)
    if selection_split not in {"train", "validation"}:
        raise ValueError("instance selection split must be train or validation")
    if sample_ids is None:
        rows = build_instance_manifest(
            manifest,
            seed=(
                int(config.get("seed", 17))
                if selection_seed is None
                else int(selection_seed)
            ),
            per_family=(
                (1 if pilot else 3) if per_family is None else int(per_family)
            ),
            split=selection_split,
        )
    else:
        by_sample = {
            row.sample_id: row
            for row in manifest.records
            if row.split == selection_split
        }
        requested = tuple(str(value) for value in sample_ids)
        missing = tuple(value for value in requested if value not in by_sample)
        if missing:
            raise ValueError(
                f"selected {selection_split} sample IDs are missing: {missing[:3]}"
            )
        rows = tuple(by_sample[value] for value in requested)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    _write_manifest(rows, output / "instance_manifest.json")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    if deployment_lora and not config.get("conditioner_checkpoint"):
        raise ValueError(
            "deployment LoRA freezes the meta network; a conditioner_checkpoint "
            "(meta-trained hypernetwork) is required"
        )
    model, normalizer = _load_parent(config, manifest, device)
    background_provider = _load_background_provider(
        config, (row.sample_id for row in rows)
    )
    sample_ids = tuple(row.sample_id for row in rows)
    # Adaptive PDE collocation sampling (SOTA RAD + causal time tilt).  Absent /
    # empty config block reproduces the historical uniform-random points.
    _sampling_cfg = dict(config.get("physics_sampling") or {})
    physics_sampling = PhysicsSampling(
        method=str(_sampling_cfg.get("method", "uniform")),
        k=float(_sampling_cfg.get("k", 1.0)),
        c=float(_sampling_cfg.get("c", 1.0)),
        time_tilt=float(_sampling_cfg.get("time_tilt", 1.5)),
        resample_every=int(_sampling_cfg.get("resample_every", 4)),
        hard_quantile=float(_sampling_cfg.get("hard_quantile", 0.9)),
        release_quantile=float(_sampling_cfg.get("release_quantile", 0.7)),
        uniform_fraction=float(_sampling_cfg.get("uniform_fraction", 0.2)),
        rams_fraction=float(_sampling_cfg.get("rams_fraction", 0.2)),
        rams_steps=int(_sampling_cfg.get("rams_steps", 3)),
    )
    adaptation_optimizer = str(
        config.get("adaptation_optimizer", "adam_lbfgs")
    )
    _chonknoris_cfg = dict(config.get("chonknoris") or {})
    chonknoris_config = ReducedChonknorisConfig(
        iterations=int(_chonknoris_cfg.get("iterations", 4)),
        initial_relaxation=float(
            _chonknoris_cfg.get("initial_relaxation", 1.0e-2)
        ),
        initial_step_size=float(
            _chonknoris_cfg.get("initial_step_size", 1.0)
        ),
        relaxation_factors=tuple(
            float(value)
            for value in _chonknoris_cfg.get(
                "relaxation_factors", (0.5, 1.0, 2.0)
            )
        ),
        step_factors=tuple(
            float(value)
            for value in _chonknoris_cfg.get("step_factors", (0.5, 1.0, 2.0))
        ),
        minimum_relaxation=float(
            _chonknoris_cfg.get("minimum_relaxation", 1.0e-6)
        ),
        maximum_relaxation=float(
            _chonknoris_cfg.get("maximum_relaxation", 1.0e2)
        ),
        minimum_step_size=float(
            _chonknoris_cfg.get("minimum_step_size", 1.0e-3)
        ),
        maximum_step_size=float(
            _chonknoris_cfg.get("maximum_step_size", 2.0)
        ),
        residual_tolerance=float(
            _chonknoris_cfg.get("residual_tolerance", 1.0e-8)
        ),
        minimum_relative_improvement=float(
            _chonknoris_cfg.get("minimum_relative_improvement", 1.0e-6)
        ),
        maximum_condition_number=float(
            _chonknoris_cfg.get("maximum_condition_number", 1.0e8)
        ),
        exact_factor_fallback=bool(
            _chonknoris_cfg.get("exact_factor_fallback", True)
        ),
        residual_pool_size=int(
            _chonknoris_cfg.get("residual_pool_size", 4)
        ),
        physics_point_count=int(
            _chonknoris_cfg.get("physics_point_count", 128)
        ),
    )
    saved_time_parent = resolve_saved_time_parent_config(config)
    travel_time_h5 = config.get("travel_time_h5")
    if travel_time_h5 is None and saved_time_parent is not None:
        travel_time_h5 = saved_time_parent.get("travel_time_h5")
    dataset = GuardedOnsetDataset(
        config["source_h5"],
        manifest,
        split=selection_split,
        sample_ids=sample_ids,
        travel_time_h5=travel_time_h5,
    )
    reports = []
    try:
        for index in range(len(dataset)):
            record = dataset[index]
            source = record.source_parameters.to(device).unsqueeze(0)
            # In deployment-LoRA mode everything is fit in the normalizer's O(1)
            # space; the adapter sees a normalized parent field and normalized
            # onset frames, and the field is decoded only for sealed evaluation.
            parent_field = _predict_parent(
                model,
                normalizer,
                record,
                device,
                normalized=deployment_lora,
                background_provider=background_provider,
            )
            adapt_record = record
            if deployment_lora:
                adapt_record = dataclasses.replace(
                    record,
                    observed_wavefield=normalizer.encode_pressure(
                        record.observed_wavefield.to(device), source[:, 4]
                    ).squeeze(0).cpu(),
                )
            adapter = OnsetAdaptedV5(model, latent_dim=int(config.get("latent_dim", 32)), lora_rank=int(config.get("lora_rank", 4))).to(device)
            conditioner_info = None
            conditioner_checkpoint = config.get("conditioner_checkpoint")
            if conditioner_checkpoint:
                conditioner_info = _load_conditioner(
                    adapter,
                    conditioner_checkpoint,
                    device,
                    expected_manifest_digest=manifest.digest,
                    expected_parent_checkpoint=config.get("parent_checkpoint"),
                    require_parent_binding=bool(
                        config.get("require_parent_binding", False)
                    ),
                )
            result = adapt_instance(
                adapter,
                adapt_record,
                parent_field,
                adam_steps=int(config.get("adam_steps", 12)),
                lbfgs_steps=int(config.get("lbfgs_steps", 4)),
                learning_rate=float(config.get("learning_rate", 2.0e-4)),
                seed=int(config.get("seed", 17)) + index,
                deployment_lora=bool(deployment_lora),
                hard_project_observed=bool(config.get("hard_project_onset", False)),
                deployment_pde_weight=float(config.get("deployment_pde_weight", 0.0)),
                physics_sampling=physics_sampling,
                optimizer_name=adaptation_optimizer,
                chonknoris_config=chonknoris_config,
                minimum_physics_improvement=float(
                    config.get("minimum_physics_improvement", 0.01)
                ),
                minimum_energy_ratio=float(
                    config.get("minimum_energy_ratio", 0.8)
                ),
                maximum_energy_ratio=float(
                    config.get("maximum_energy_ratio", 1.25)
                ),
            )
            with torch.no_grad():
                adapted_field = adapter.raw_wavefield(
                    parent_field,
                    adapt_record.velocity_mps.to(device).unsqueeze(0),
                    adapt_record.source_parameters.to(device).unsqueeze(0),
                    adapt_record.observed_wavefield.to(device).unsqueeze(0),
                    adapt_record.time_s.to(device),
                )
                if deployment_lora:
                    # Decode both fields to physical pressure, THEN optionally
                    # hard-project the real (physical) onset frames.  Projection
                    # is off by default: on near-zero early frames it injects a
                    # discontinuity that hurts the energy-aggregated metric more
                    # than it helps the two observed frames.
                    adapted_field = normalizer.decode_pressure(adapted_field, source[:, 4])
                    parent_field = normalizer.decode_pressure(parent_field, source[:, 4])
                    if bool(config.get("hard_project_onset", False)):
                        adapted_field = adapter.hard_project(
                            adapted_field,
                            record.observed_wavefield.to(device).unsqueeze(0),
                            record.observed_indices,
                        )
                else:
                    adapted_field = adapter.hard_project(
                        adapted_field,
                        record.observed_wavefield.unsqueeze(0),
                        record.observed_indices,
                    )
            artifact = output / record.sample_id
            artifact.mkdir(parents=True, exist_ok=True)
            # The state and audit are sealed before the independent truth read below.
            sealed_payload = {"adaptation": result.to_dict()}
            if save_fields:
                sealed_payload.update(
                    {
                        "parent_field": parent_field.cpu(),
                        "adapted_field": adapted_field.cpu(),
                    }
                )
                torch.save(sealed_payload, artifact / "fields.pt")
            else:
                torch.save(sealed_payload, artifact / "adaptation.pt")
            truth = _read_future_truth(config["source_h5"], record.source_index)
            report = evaluate_after_adaptation(
                {"parent_field": parent_field.cpu(), "adapted_field": adapted_field.cpu()},
                truth.unsqueeze(0), observed_indices=(record.observed_indices,),
                families=(record.medium_type,), group_ids=(record.group_id,), sample_ids=(record.sample_id,), sealed=True,
            )
            report["sample_id"] = record.sample_id
            report["medium_type"] = record.medium_type
            report["adaptation"] = result.to_dict()
            report["conditioner"] = conditioner_info
            write_report(report, artifact / "evaluation.json")
            if write_plots:
                target_np = truth.numpy(); parent_np = parent_field[0].detach().cpu().numpy(); adapted_np = adapted_field[0].detach().cpu().numpy()
                snapshots = tuple(index for index in (record.observed_indices[1] + 1, len(record.time_s) // 2, len(record.time_s) - 1) if index < len(record.time_s))
                plot_wavefield_comparison(target_np, parent_np, adapted_np, snapshots, output=artifact / "wavefield_comparison", title=record.medium_type)
                plot_receiver_comparison(target_np, parent_np, adapted_np, record.time_s.numpy(), report["receiver_indices"], output=artifact / "receiver_waveforms", title=record.medium_type)
            reports.append(report)
    finally:
        dataset.close()
        if background_provider is not None:
            background_provider.close()
    write_report(
        {
            "records": reports,
            "selection_split": selection_split,
            "family_count": {
                family: sum(item["medium_type"] == family for item in reports)
                for family in ALLOWED_MEDIUM_TYPES
            },
        },
        output / "summary.json",
    )
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument(
        "--conditioner-checkpoint",
        help="Explicit meta-hypernetwork checkpoint to validate without editing config.",
    )
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument(
        "--selection-split",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--per-family", type=int)
    parser.add_argument("--exclude-manifest")
    parser.add_argument("--all-validation", action="store_true")
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--method", default="recommended")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--deployment-lora",
        action="store_true",
        help="Freeze the meta network and fine-tune only the per-instance latent (deployment mode).",
    )
    parser.add_argument(
        "--hard-project-onset",
        action="store_true",
        help="Project the two permitted onset snapshots exactly after adaptation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.parent_checkpoint:
        config["parent_checkpoint"] = args.parent_checkpoint
    manifest = build_manifest(config["source_h5"])
    excluded_group_ids = set()
    if args.exclude_manifest:
        excluded_rows = json.loads(Path(args.exclude_manifest).read_text())
        excluded_group_ids = {str(row["group_id"]) for row in excluded_rows}
    if args.all_validation:
        if args.selection_split != "validation":
            raise ValueError("--all-validation requires --selection-split validation")
        if args.sample_id or args.exclude_manifest:
            raise ValueError("complete validation cannot combine sample or exclusion filters")
        rows = build_all_validation_manifest(manifest)
    elif args.sample_id:
        by_sample = {
            row.sample_id: row
            for row in manifest.records
            if row.split == args.selection_split
            and row.group_id not in excluded_group_ids
        }
        missing = tuple(value for value in args.sample_id if value not in by_sample)
        if missing:
            raise ValueError(
                f"requested {args.selection_split} sample IDs are missing: {missing[:3]}"
            )
        rows = tuple(by_sample[value] for value in args.sample_id)
    else:
        rows = build_instance_manifest(
            manifest,
            seed=(
                int(config.get("seed", 17))
                if args.selection_seed is None
                else int(args.selection_seed)
            ),
            per_family=(
                (1 if args.pilot else 3)
                if args.per_family is None
                else int(args.per_family)
            ),
            excluded_group_ids=excluded_group_ids,
            split=args.selection_split,
        )
    if args.dry_run:
        print(json.dumps({
            "family_counts": {family: sum(row.medium_type == family for row in rows) for family in ALLOWED_MEDIUM_TYPES},
            "sample_ids": [row.sample_id for row in rows],
            "selection_split": args.selection_split,
            "future_truth_opened": False,
            "allowed_true_snapshot_count": 2,
            "adaptation_optimizer": str(
                config.get("adaptation_optimizer", "adam_lbfgs")
            ),
        }, sort_keys=True))
        return 0
    run(
        args.config,
        device_name=args.device,
        output_dir=args.output_dir,
        pilot=args.pilot,
        parent_checkpoint=args.parent_checkpoint,
        conditioner_checkpoint=args.conditioner_checkpoint,
        sample_ids=tuple(row.sample_id for row in rows),
        selection_split=args.selection_split,
        selection_seed=args.selection_seed,
        per_family=args.per_family,
        save_fields=not args.no_fields,
        write_plots=not args.no_plots,
        deployment_lora=args.deployment_lora,
        hard_project_onset=(True if args.hard_project_onset else None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
