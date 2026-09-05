#!/usr/bin/env python3
"""Leakage-safe train-only pilot for the direct WKB frequency surrogate.

The fit, calibration, and confirmation panels are mutually disjoint subsets of
the training split. Complete wavefields are read as training labels only for the
fit panel. Calibration labels select the checkpoint. Confirmation labels are
opened once, after that checkpoint is frozen; validation/test_id are not opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.diagnose_capacity_ladder_overfit import (
    _direct_frequency_coefficient_update,
    build_base_config,
    build_capacity_optimizer,
    build_probe_config,
    load_exact_model_initialization,
    load_continue_frequency_initialization,
    zero_initialize_direct_frequency_head,
)
from scripts.diagnose_saved_time_temporal_three_record_overfit import (
    FAMILIES,
    _dataset,
    _evaluate_triplet,
    restore_best_overfit_checkpoint,
    save_overfit_checkpoint,
)
from scripts.refine_saved_time_v4_lbfgs import _append_jsonl, _gpu_snapshot
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import (
    _gradient_report,
    clip_trainable_gradients,
    temporal_basis_gradient_norms,
)
from scripts.train_saved_time_v4_probe import _atomic_json, _digest, _model


def _sha256_file(path: Path, *, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def select_disjoint_family_panels(
    records,
    *,
    split: str,
    fit_per_family: int,
    calibration_per_family: int,
    confirm_per_family: int,
    skip_per_family: int = 1,
    families: tuple[str, ...] = FAMILIES,
    selection: str = "sequential",
    seed: int = 372,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return deterministic split-relative fit/calibration/confirmation indices."""

    selected = tuple(record for record in records if str(record.split) == str(split))
    fit: list[int] = []
    calibration: list[int] = []
    confirm: list[int] = []
    mode = str(selection)
    if mode not in {"sequential", "stratified_random"}:
        raise ValueError("panel selection must be sequential or stratified_random")
    for family_index, family in enumerate(families):
        available = [
            index
            for index, record in enumerate(selected)
            if str(record.medium_type) == family
        ]
        start = int(skip_per_family)
        if start < 0:
            raise ValueError("panel skip must be nonnegative")
        if mode == "stratified_random":
            eligible = available[start:]
            rng = np.random.default_rng(int(seed) + family_index * 1_000_003)
            available = [eligible[index] for index in rng.permutation(len(eligible))]
            start = 0
        fit_stop = start + int(fit_per_family)
        calibration_stop = fit_stop + int(calibration_per_family)
        confirm_stop = calibration_stop + int(confirm_per_family)
        if (
            start < 0
            or int(fit_per_family) <= 0
            or int(calibration_per_family) <= 0
            or int(confirm_per_family) <= 0
        ):
            raise ValueError("panel counts must be positive and skip nonnegative")
        if confirm_stop > len(available):
            raise ValueError(f"not enough {family} records for disjoint panels")
        fit.extend(available[start:fit_stop])
        calibration.extend(available[fit_stop:calibration_stop])
        confirm.extend(available[calibration_stop:confirm_stop])
    if (
        set(fit) & set(calibration)
        or set(fit) & set(confirm)
        or set(calibration) & set(confirm)
    ):
        raise RuntimeError("fit, calibration, and confirmation panels overlap")
    return tuple(fit), tuple(calibration), tuple(confirm)


def select_anchor_group_source_panels(
    records,
    *,
    split: str,
    anchor_sample_ids: tuple[str, ...],
    families: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split sources within each anchor medium into fit/calibration/confirmation.

    Every selected family must expose at least four records in the anchor group.
    The anchor and the earliest additional sources form the fit panel; the final
    two sources are held out one each for checkpoint selection and confirmation.
    """

    selected = tuple(record for record in records if str(record.split) == str(split))
    by_sample = {str(record.sample_id): index for index, record in enumerate(selected)}
    anchors: dict[str, int] = {}
    for sample_id in anchor_sample_ids:
        index = by_sample.get(str(sample_id))
        if index is None:
            continue
        family = str(selected[index].medium_type)
        if family in families:
            anchors[family] = index
    if set(anchors) != set(families):
        raise ValueError("source-holdout panels require one parent anchor per family")

    fit: list[int] = []
    calibration: list[int] = []
    confirm: list[int] = []
    for family in families:
        anchor_index = anchors[family]
        group_id = str(selected[anchor_index].group_id)
        group = [
            index
            for index, record in enumerate(selected)
            if str(record.medium_type) == family and str(record.group_id) == group_id
        ]
        if len(group) < 4:
            raise ValueError(f"{family} anchor group needs at least four sources")
        ordered = [anchor_index] + [index for index in group if index != anchor_index]
        fit.extend(ordered[:-2])
        calibration.append(ordered[-2])
        confirm.append(ordered[-1])
    if set(fit) & set(calibration + confirm) or set(calibration) & set(confirm):
        raise RuntimeError("source-holdout panels overlap")
    return tuple(fit), tuple(calibration), tuple(confirm)


def build_balanced_panel_schedule(
    records,
    indices: tuple[int, ...],
    *,
    split: str,
    updates: int,
    seed: int,
    families: tuple[str, ...] = FAMILIES,
) -> tuple[FullSupportStepSpec, ...]:
    """Draw one record from each family per update with deterministic reshuffles."""

    selected = tuple(record for record in records if str(record.split) == str(split))
    pools = {
        family: np.asarray(
            [index for index in indices if str(selected[index].medium_type) == family],
            dtype=np.int64,
        )
        for family in families
    }
    if any(not len(pool) for pool in pools.values()):
        raise ValueError("balanced schedule requires every family")
    streams: dict[str, list[int]] = {family: [] for family in families}
    for family_index, family in enumerate(families):
        cycle = 0
        while len(streams[family]) < int(updates):
            rng = np.random.default_rng(
                int(seed) + family_index * 1_000_003 + cycle * 104_729
            )
            streams[family].extend(int(value) for value in rng.permutation(pools[family]))
            cycle += 1
    appearances = {int(index): 0 for index in indices}
    schedule: list[FullSupportStepSpec] = []
    for update in range(int(updates)):
        chosen = tuple(streams[family][update] for family in families)
        current = tuple(appearances[index] for index in chosen)
        for index in chosen:
            appearances[index] += 1
        schedule.append(
            FullSupportStepSpec(
                step=1_100_000 + update,
                epoch=0,
                record_indices=chosen,
                appearance_indices=current,
            )
        )
    return tuple(schedule)


class StreamingFullTraceStore:
    """Read complete fit-panel labels lazily from the source HDF5/VDS."""

    def __init__(self, source_h5: str, records, *, expected_time_count: int):
        self.source_h5 = str(source_h5)
        self.source_index = {record.sample_id: int(record.source_index) for record in records}
        self.expected_time_count = int(expected_time_count)
        self._handle: h5py.File | None = None

    def __getitem__(self, sample_id: str) -> np.ndarray:
        if self._handle is None:
            self._handle = h5py.File(self.source_h5, "r", swmr=True)
        value = np.asarray(self._handle["wavefield"][self.source_index[str(sample_id)]])
        if value.shape != (self.expected_time_count, 201, 201):
            raise ValueError("streamed full-trace target shape changed")
        if not np.isfinite(value).all():
            raise FloatingPointError("non-finite streamed full-trace target")
        return value

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class BackgroundFullTraceStore:
    """Expose complete cached P_bg traces through the coefficient-target mapping API."""

    def __init__(self, provider: BackgroundFieldProvider):
        self.provider = provider

    def __getitem__(self, sample_id: str) -> np.ndarray:
        result = self.provider.full_physical((str(sample_id),), device="cpu")[0].numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite cached background target")
        return result

    def close(self) -> None:
        return None


class ScatteringFullTraceStore:
    """Return exact train truth minus the fixed source-aware P_bg field."""

    def __init__(self, truth: StreamingFullTraceStore, provider: BackgroundFieldProvider):
        self.truth = truth
        self.provider = provider

    def __getitem__(self, sample_id: str) -> np.ndarray:
        truth = self.truth[str(sample_id)]
        background = self.provider.full_physical((str(sample_id),), device="cpu")[0].numpy()
        if background.shape != truth.shape:
            raise ValueError("background and truth full traces do not align")
        residual = truth - background
        if not np.isfinite(residual).all():
            raise FloatingPointError("non-finite scattering full-trace target")
        return residual

    def close(self) -> None:
        self.truth.close()


def gate_score(*metrics: dict[str, object]) -> float:
    values: list[float] = []
    for item in metrics:
        values.append(float(item["aggregate_relative_l2"]))
        values.extend(float(value) for value in item["family_relative_l2"].values())
    return max(values)


def metrics_meet_panel_target(
    metrics: dict[str, object], *, families: tuple[str, ...], maximum: float
) -> bool:
    family_metrics = metrics.get("family_relative_l2", {})
    return float(metrics["aggregate_relative_l2"]) <= float(maximum) and all(
        float(family_metrics.get(family, math.inf)) <= float(maximum)
        for family in families
    )


def configure_trainable_scope(model, scope: str) -> dict[str, object]:
    mode = str(scope)
    if mode == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif mode == "helmholtz_head":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        synthesis = model.local_field.helmholtz_synthesis
        if synthesis is None or not hasattr(synthesis, "head"):
            raise ValueError("helmholtz_head scope requires an independent Helmholtz head")
        for parameter in synthesis.head.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"unsupported trainable scope: {mode!r}")
    names = tuple(name for name, p in model.named_parameters() if p.requires_grad)
    return {
        "scope": mode,
        "parameter_names": names,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }


def required_gradient_prefixes(
    *, zero_head: bool, update: int, trainable_scope: str = "all"
) -> tuple[str, ...]:
    if int(update) <= 0:
        raise ValueError("gradient-contract update must be positive")
    if str(trainable_scope) == "helmholtz_head":
        return ("local_field",)
    return (
        ("local_field",)
        if bool(zero_head) and int(update) == 1
        else ("local_field", "medium_encoder", "source_encoder")
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--parent-identity", required=True)
    parser.add_argument("--normalization-json", required=True)
    parser.add_argument("--travel-time-h5", required=True)
    parser.add_argument(
        "--panel-families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
    )
    parser.add_argument(
        "--target-background-cache-shard",
        action="append",
        default=[],
        help="use cached train-only P_bg as the supervision/evaluation target; repeat for shards",
    )
    parser.add_argument(
        "--background-cache-shard",
        action="append",
        default=[],
        help="fixed source-aware P_bg input; train on truth-P_bg and add P_bg for metrics",
    )
    parser.add_argument("--zero-initialize-helmholtz-head", action="store_true")
    parser.add_argument(
        "--trainable-scope",
        choices=("all", "helmholtz_head"),
        default="all",
    )
    parser.add_argument("--fit-records-per-family", type=int, default=12)
    parser.add_argument("--calibration-records-per-family", type=int, default=12)
    parser.add_argument("--confirm-records-per-family", type=int, default=12)
    parser.add_argument("--skip-records-per-family", type=int, default=1)
    parser.add_argument(
        "--panel-selection",
        choices=("sequential", "stratified_random", "anchor_group_source_holdout"),
        default="sequential",
    )
    parser.add_argument("--helmholtz-frequencies", type=int, default=64)
    parser.add_argument("--helmholtz-frequency-softmax", action="store_true")
    parser.add_argument("--helmholtz-source-onset-phase", action="store_true")
    parser.add_argument("--helmholtz-source-relative-coordinates", action="store_true")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--evaluate-every", type=int, default=250)
    parser.add_argument("--evaluation-frames", type=int, default=32)
    parser.add_argument("--evaluation-time-block", type=int, default=64)
    parser.add_argument("--evaluation-macro-records", type=int, default=3)
    parser.add_argument("--dense-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--local-field-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--family-gradient-weights", default="uniform:1,layered:1,marmousi:1")
    parser.add_argument("--target-relative-l2", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=372)
    args = parser.parse_args(argv)
    if (
        args.updates <= 0
        or args.evaluate_every <= 0
        or args.evaluation_time_block <= 0
        or args.evaluation_macro_records <= 0
    ):
        raise ValueError("updates and evaluation cadence must be positive")

    panel_families = tuple(dict.fromkeys(str(value) for value in args.panel_families))
    family_weights = {family: 1.0 for family in FAMILIES}
    for token in str(args.family_gradient_weights).split(","):
        family, separator, value = token.partition(":")
        if separator != ":" or family not in family_weights or float(value) <= 0.0:
            raise ValueError("invalid family gradient weight")
        family_weights[family] = float(value)

    root = Path(args.artifact_dir).resolve()
    preregistration_path = Path(args.preregistration).resolve()
    preregistration = json.loads(preregistration_path.read_text())
    if str(preregistration.get("status")) != "frozen_before_launch":
        raise ValueError("preregistration must be frozen before launch")
    if str(preregistration.get("candidate")) != root.name:
        raise ValueError("preregistration candidate does not match artifact directory")
    root.mkdir(parents=True, exist_ok=True)
    terminal_path = root / "terminal.json"
    if terminal_path.exists():
        print(terminal_path.read_text().strip())
        return 0

    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))

    base = build_base_config(128, base_config=str(args.base_config))
    import dataclasses
    base = dataclasses.replace(
        base,
        data=dataclasses.replace(base.data, normalization_json=str(args.normalization_json)),
    )
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": base.data.expected_train_records,
            "validation": base.data.expected_validation_records,
        },
    )
    normalizer = load_normalizer(base, manifest.digest)
    target_cache_paths = tuple(
        Path(value).expanduser().resolve()
        for value in args.target_background_cache_shard
    )
    target_provider = (
        BackgroundFieldProvider(target_cache_paths) if target_cache_paths else None
    )
    background_cache_paths = tuple(
        Path(value).expanduser().resolve() for value in args.background_cache_shard
    )
    if target_cache_paths and background_cache_paths:
        raise ValueError("target-background and residual-background caches are exclusive")
    background_provider = (
        BackgroundFieldProvider(background_cache_paths)
        if background_cache_paths
        else None
    )
    script_path = Path(__file__).resolve()
    diagnostic_path = script_path.with_name("diagnose_capacity_ladder_overfit.py")
    code_bindings = {
        "script": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "diagnostic": str(diagnostic_path),
        "diagnostic_sha256": _sha256_file(diagnostic_path),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": _sha256_file(preregistration_path),
        "base_config": str(Path(args.base_config).resolve()),
        "base_config_sha256": _sha256_file(Path(args.base_config).resolve()),
        "source_h5": str(Path(base.data.source_h5).resolve()),
        "source_h5_sha256": _sha256_file(Path(base.data.source_h5).resolve()),
        "normalization_json": str(Path(args.normalization_json).resolve()),
        "normalization_json_sha256": _sha256_file(
            Path(args.normalization_json).resolve()
        ),
        "travel_time_h5": str(Path(args.travel_time_h5).resolve()),
        "travel_time_h5_sha256": _sha256_file(Path(args.travel_time_h5).resolve()),
        "target_background_cache_shards": [str(path) for path in target_cache_paths],
        "target_background_cache_sha256": {
            str(path): _sha256_file(path) for path in target_cache_paths
        },
        "background_cache_shards": [str(path) for path in background_cache_paths],
        "background_cache_sha256": {
            str(path): _sha256_file(path) for path in background_cache_paths
        },
    }
    parent_identity_path = Path(args.parent_identity).resolve()
    parent_identity = json.loads(parent_identity_path.read_text())
    if str(args.panel_selection) == "anchor_group_source_holdout":
        fit_indices, calibration_indices, confirm_indices = select_anchor_group_source_panels(
            manifest.records,
            split="train",
            anchor_sample_ids=tuple(parent_identity["sample_ids"]),
            families=panel_families,
        )
    else:
        fit_indices, calibration_indices, confirm_indices = select_disjoint_family_panels(
            manifest.records,
            split="train",
            fit_per_family=int(args.fit_records_per_family),
            calibration_per_family=int(args.calibration_records_per_family),
            confirm_per_family=int(args.confirm_records_per_family),
            skip_per_family=int(args.skip_records_per_family),
            families=panel_families,
            selection=str(args.panel_selection),
            seed=int(args.seed),
        )
    train_records = tuple(record for record in manifest.records if record.split == "train")
    fit_sample_ids = tuple(train_records[index].sample_id for index in fit_indices)
    calibration_sample_ids = tuple(
        train_records[index].sample_id for index in calibration_indices
    )
    confirm_sample_ids = tuple(train_records[index].sample_id for index in confirm_indices)
    all_panel_sample_ids = fit_sample_ids + calibration_sample_ids + confirm_sample_ids
    if target_provider is not None and not target_provider.covers(all_panel_sample_ids):
        raise ValueError("target background cache does not cover every panel sample")
    if background_provider is not None and not background_provider.covers(
        all_panel_sample_ids, range(len(manifest.time_s))
    ):
        raise ValueError("residual background cache does not cover every panel sample/time")

    variant = ProbeVariant(
        depth=8,
        use_local_phase=True,
        spectral_rank=112,
        modes=32,
        temporal_basis_rank=0,
        family_expert_rank=0,
        local_field=True,
        local_field_channel_multipliers=(1, 1, 2, 2),
        local_field_causal_width_s=0.005,
        local_field_residual=False,
        local_field_helmholtz_synthesis=True,
        local_field_helmholtz_synthesis_frequencies=int(args.helmholtz_frequencies),
        local_field_helmholtz_synthesis_wkb_phase=True,
        local_field_helmholtz_synthesis_rank=0,
        local_field_helmholtz_synthesis_frequency_softmax=bool(
            args.helmholtz_frequency_softmax
        ),
        local_field_helmholtz_synthesis_source_onset_phase=bool(
            args.helmholtz_source_onset_phase
        ),
        local_field_helmholtz_source_relative_coordinates=bool(
            args.helmholtz_source_relative_coordinates
        ),
    )
    model = _model(base, manifest, variant).to(device)
    model.local_field.helmholtz_apply_causal_gate = False
    model.dense_apply_free_surface_factor = False
    transfer_loader = (
        load_continue_frequency_initialization
        if args.helmholtz_source_relative_coordinates
        else load_exact_model_initialization
    )
    transfer = transfer_loader(
        model,
        Path(args.parent_checkpoint).resolve(),
        Path(args.parent_identity).resolve(),
        manifest_digest=manifest.digest,
        device=device,
    )
    zero_head_report = (
        zero_initialize_direct_frequency_head(model)
        if args.zero_initialize_helmholtz_head
        else None
    )
    if args.trainable_scope == "helmholtz_head" and not args.zero_initialize_helmholtz_head:
        raise ValueError("helmholtz_head scope requires zero-initialize-helmholtz-head")
    trainable_report = configure_trainable_scope(model, str(args.trainable_scope))
    optimizer = build_capacity_optimizer(
        model,
        dense_lr=float(args.dense_learning_rate),
        backbone_lr=float(args.backbone_learning_rate),
        local_field_lr=float(args.local_field_learning_rate),
    )
    config = build_probe_config(
        dense_lr=float(args.dense_learning_rate),
        backbone_lr=float(args.backbone_learning_rate),
        temporal_lr=1.0e-5,
        seed=int(args.seed),
        travel_time_h5=str(args.travel_time_h5),
        family_gradient_weights=family_weights,
    )
    schedule = build_balanced_panel_schedule(
        manifest.records,
        fit_indices,
        split="train",
        updates=int(args.updates),
        seed=int(args.seed),
        families=panel_families,
    )
    train_data = _dataset(
        config,
        base,
        manifest,
        fit_indices,
        split="train",
        schedule=schedule,
        time_policy="appearance16",
        frames_per_record=4,
    )

    identity = {
        "schema": "wkb_frequency_train_panel_v1",
        "base_config": str(args.base_config),
        "manifest_digest": manifest.digest,
        "split": "train",
        "panel_families": panel_families,
        "panel_selection": str(args.panel_selection),
        "helmholtz_frequencies": int(args.helmholtz_frequencies),
        "helmholtz_frequency_softmax": bool(args.helmholtz_frequency_softmax),
        "helmholtz_source_onset_phase": bool(args.helmholtz_source_onset_phase),
        "helmholtz_source_relative_coordinates": bool(
            args.helmholtz_source_relative_coordinates
        ),
        "target_kind": (
            "cached_background_pbg"
            if target_provider is not None
            else "truth_minus_fixed_pbg"
            if background_provider is not None
            else "source_truth"
        ),
        "zero_head_initialization": zero_head_report,
        "trainable": trainable_report,
        "fit_indices": fit_indices,
        "fit_sample_ids": fit_sample_ids,
        "calibration_indices": calibration_indices,
        "calibration_sample_ids": calibration_sample_ids,
        "confirm_indices": confirm_indices,
        "confirm_sample_ids": confirm_sample_ids,
        "panels_disjoint": not bool(
            set(fit_indices) & set(calibration_indices)
            or set(fit_indices) & set(confirm_indices)
            or set(calibration_indices) & set(confirm_indices)
        ),
        "complete_trace_use": "fit_panel_train_labels_only",
        "calibration_access": "fixed-frame_checkpoint_selection_then_all_saved_report",
        "confirmation_access": "all_saved_once_after_checkpoint_freeze",
        "validation_opened": False,
        "test_id_opened": False,
        "updates": int(args.updates),
        "evaluate_every": int(args.evaluate_every),
        "evaluation_frames": int(args.evaluation_frames),
        "evaluation_time_block": int(args.evaluation_time_block),
        "evaluation_macro_records": int(args.evaluation_macro_records),
        "family_gradient_weights": family_weights,
        "target_relative_l2": float(args.target_relative_l2),
        "parent": transfer,
        "bindings": code_bindings,
        "seed": int(args.seed),
    }
    identity["run_digest"] = _digest(identity)
    _atomic_json(identity, root / "run_identity.json")

    def evaluate(indices: tuple[int, ...], frames: int) -> dict[str, object]:
        return _evaluate_triplet(
            model,
            base,
            manifest,
            normalizer,
            device,
            config,
            indices,
            split="train",
            time_policy="all_saved" if frames == len(manifest.time_s) else "validation_fixed",
            frames_per_record=frames,
            evaluation_macro_records=min(
                int(args.evaluation_macro_records),
                len(indices),
            ),
            apply_correction=False,
            evaluation_time_block=min(int(args.evaluation_time_block), frames),
            target_provider=target_provider,
            background_provider=background_provider,
        )

    fit_metrics = evaluate(fit_indices, int(args.evaluation_frames))
    calibration_metrics = evaluate(calibration_indices, int(args.evaluation_frames))
    baseline_score = gate_score(fit_metrics, calibration_metrics)
    _append_jsonl(
        root / "metrics.jsonl",
        {
            "event": "baseline",
            "update": 0,
            "fit": fit_metrics,
            "calibration": calibration_metrics,
            "gate_score": baseline_score,
        },
    )
    best_checkpoint = str(
        save_overfit_checkpoint(
            root,
            model=model,
            update=0,
            manifest_digest=manifest.digest,
            config_digest=identity["run_digest"],
            aggregate_relative_l2=baseline_score,
        )
    )
    best_score = baseline_score
    last_update = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if target_provider is not None:
        targets = BackgroundFullTraceStore(target_provider)
    else:
        truth_store = StreamingFullTraceStore(
            base.data.source_h5,
            train_records,
            expected_time_count=len(manifest.time_s),
        )
        targets = (
            ScatteringFullTraceStore(truth_store, background_provider)
            if background_provider is not None
            else truth_store
        )
    try:
        for update, batch in enumerate(train_data, start=1):
            model.train()
            components = _direct_frequency_coefficient_update(
                model,
                optimizer,
                batch,
                normalizer,
                device,
                full_targets=targets,
                frequency_count=int(args.helmholtz_frequencies),
                microbatch_records=1,
                saved_time_s=torch.as_tensor(manifest.time_s),
                frequency_energy_floor_fraction=0.0,
                family_gradient_weights=family_weights,
            )
            gradient_prefixes = required_gradient_prefixes(
                zero_head=bool(args.zero_initialize_helmholtz_head),
                update=update,
                trainable_scope=str(args.trainable_scope),
            )
            gradients = _gradient_report(model, gradient_prefixes)
            gradients.update(temporal_basis_gradient_norms(model))
            gradient_norm, clipping = clip_trainable_gradients(
                model,
                maximum_norm=float(config["optimizer"]["gradient_clip"]),
                mode=str(config["optimizer"]["gradient_clip_mode"]),
                prefix_limits=config["optimizer"]["gradient_clip_prefix_limits"],
                return_report=True,
            )
            optimizer.step()
            last_update = update
            _append_jsonl(
                root / "updates.jsonl",
                {
                    "event": "optimizer_update",
                    "update": update,
                    "loss_components": components,
                    "gradient_norm_before_clip": gradient_norm,
                    "gradient_clipping": clipping,
                    "gpu": _gpu_snapshot(),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            if update % int(args.evaluate_every) and update != int(args.updates):
                continue
            fit_metrics = evaluate(fit_indices, int(args.evaluation_frames))
            calibration_metrics = evaluate(
                calibration_indices, int(args.evaluation_frames)
            )
            score = gate_score(fit_metrics, calibration_metrics)
            checkpoint = save_overfit_checkpoint(
                root,
                model=model,
                update=update,
                manifest_digest=manifest.digest,
                config_digest=identity["run_digest"],
                aggregate_relative_l2=score,
            )
            if score <= best_score:
                best_score = score
                best_checkpoint = str(checkpoint)
            _append_jsonl(
                root / "metrics.jsonl",
                {
                    "event": "evaluation",
                    "update": update,
                    "fit": fit_metrics,
                    "calibration": calibration_metrics,
                    "gate_score": score,
                    "checkpoint": str(checkpoint),
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(json.dumps({"update": update, "gate_score": score}), flush=True)
            if metrics_meet_panel_target(
                fit_metrics,
                families=panel_families,
                maximum=float(args.target_relative_l2),
            ) and metrics_meet_panel_target(
                calibration_metrics,
                families=panel_families,
                maximum=float(args.target_relative_l2),
            ):
                break
    finally:
        targets.close()

    restore_best_overfit_checkpoint(
        best_checkpoint,
        model=model,
        manifest_digest=manifest.digest,
        config_digest=identity["run_digest"],
        map_location=device,
    )
    all_fit = evaluate(fit_indices, len(manifest.time_s))
    all_calibration = evaluate(calibration_indices, len(manifest.time_s))
    # Confirmation truth is opened only here, after best_checkpoint is frozen.
    all_confirm = evaluate(confirm_indices, len(manifest.time_s))
    passed = metrics_meet_panel_target(
        all_fit,
        families=panel_families,
        maximum=float(args.target_relative_l2),
    ) and metrics_meet_panel_target(
        all_calibration,
        families=panel_families,
        maximum=float(args.target_relative_l2),
    ) and metrics_meet_panel_target(
        all_confirm,
        families=panel_families,
        maximum=float(args.target_relative_l2),
    )
    terminal = {
        "status": "passed" if passed else "failed",
        "updates_completed": last_update,
        "best_checkpoint": best_checkpoint,
        "best_fixed_gate_score": best_score,
        "all_saved_fit": all_fit,
        "all_saved_calibration": all_calibration,
        "all_saved_confirm": all_confirm,
        "target_relative_l2": float(args.target_relative_l2),
        "train_only_gate_passed": passed,
        "confirmation_access": "all_saved_once_after_checkpoint_freeze",
        "validation_opened": False,
        "test_id_opened": False,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "bindings": {
            **code_bindings,
            "run_identity_sha256": _sha256_file(root / "run_identity.json"),
            "best_checkpoint_sha256": _sha256_file(Path(best_checkpoint)),
        },
    }
    _atomic_json(terminal, terminal_path)
    if target_provider is not None:
        target_provider.close()
    if background_provider is not None:
        background_provider.close()
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
