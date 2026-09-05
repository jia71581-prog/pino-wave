#!/usr/bin/env python3
"""G3 held-out generalization test for the low-rank Helmholtz-synthesis coarse field.

Question this answers
---------------------
G2 (``diagnose_capacity_ladder_overfit.py --helmholtz-synthesis --helmholtz-rank 8``)
proved the low-rank Helmholtz factorization can MEMORIZE three training records to
all_saved (401-frame, query-invariant) aggregate relative L2 = 0.206 -- the project's
best query-invariant G2 value, below Option-B LPF's 0.29 (also a G2 memorization
number).  But 0.206 is a fit to three records the model was trained on.  The whole
direction hinges on a different question, the one that killed the Kakeya arm:

    Can ``G_theta(c, x_s)`` PREDICT the smooth low-rank Helmholtz amplitude fields for
    records it never trained on, or is this another "target simple != mapping
    generalizable"?

This driver trains the identical Helmholtz r8 architecture on the FULL 2240-record
train pool and evaluates on a HELD-OUT validation triplet (uniform/layered/marmousi,
never seen in training) with the identical ``all_saved`` accumulator used for the
0.206 number.  The only difference from G2 is trained-on vs held-out records, so the
gap between this number and 0.206 is exactly the generalization gap.

Reuses the proven capacity-ladder primitives (``_model``, ``transfer_front_end``,
``build_capacity_optimizer`` with a dedicated ``local_field`` LR group, ``_train_update``,
``_evaluate_triplet``); only the data layer changes: full-pool ``build_full_support_schedule``
training instead of the 3-record repeat schedule, and validation-split evaluation.

Warm-start: the medium/source/travel/fusion front-end is transferred from the SAME
width-128 parent the Option-B LPF baseline used (warp_r1/best.pt), so the Helmholtz vs
LPF held-out comparison is controlled at the front-end.  The dense decoder and the
zero-init Helmholtz local_field generator are trained fresh.
"""
from __future__ import annotations

import argparse
import h5py
import json
import math
import time
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.data.pilot import make_pilot_loader
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
from saved_time_phase_operator_v4.probe import ProbeVariant
from saved_time_phase_operator_v4.full_support import (
    FullSupportStepSpec,
    build_full_support_schedule,
)
from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset
from saved_time_phase_operator_v4.evaluation import sha256_file
from saved_time_phase_operator_v4.multifidelity import fixed_teacher_time_indices
from saved_time_phase_operator_v4.vrba_sampling import vrba_record_pdf, vrba_frame_weights
from saved_time_phase_operator_v4.residual_activation import activate_residual_head

from scripts.train_saved_time_v4_probe import _model, _atomic_json, _digest
from scripts.train_saved_time_v4_full_support import (
    _train_update,
    _gradient_report,
    clip_trainable_gradients,
    temporal_basis_gradient_norms,
    distributed_context,
    distributed_barrier,
    distributed_cleanup,
    distributed_average_gradients,
    ddp_update_specs,
)
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer  # noqa: F401
from scripts.diagnose_capacity_ladder_overfit import (
    build_base_config,
    build_probe_config,
    build_capacity_optimizer,
)
from scripts.diagnose_saved_time_temporal_three_record_overfit import (
    FAMILIES,
    _evaluate_triplet,
    _retarget_batch_to_scattering,
    save_overfit_checkpoint,
    restore_best_overfit_checkpoint,
)
from scripts.refine_saved_time_v4_lbfgs import _append_jsonl, _gpu_snapshot


_FRONT_END_PREFIXES = (
    "coordinate_encoder.",
    "fusion.",
    "medium_encoder.",
    "source_encoder.",
    "travel_branch.",
)


def _distributed_average_components(
    components: Mapping[str, float],
    ddp: Mapping[str, object],
    device: torch.device,
) -> dict[str, float]:
    """Return component values on the same global-DDP scope as the gradients.

    ``_train_update`` runs on each rank's disjoint macro and therefore reports local
    component values.  Logging those values beside the already all-reduced gradient
    norm made a near-zero-scattering rank look like a global loss collapse.  Average
    the scalar report over ranks before writing ``updates.jsonl``; this does not take
    part in autograd and does not change the optimizer update.
    """

    ordered = tuple(sorted(str(name) for name in components))
    values = torch.tensor(
        [float(components[name]) for name in ordered],
        dtype=torch.float64,
        device=device,
    )
    if bool(ddp.get("enabled", False)):
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= float(ddp["world_size"])
    return {
        name: float(value)
        for name, value in zip(ordered, values.detach().cpu().tolist(), strict=True)
    }


def transfer_front_end_permissive(model, checkpoint_path: Path) -> dict:
    """Load only the true front-end encoders; leave decoder AND local_field fresh.

    The stock ``transfer_front_end`` forbids leaving any non-``dense_decoder`` tensor
    uninitialized, so it cannot warm-start a model whose fresh Helmholtz field lives
    under top-level ``local_field.*``.  Here we load exactly the medium/source/travel/
    fusion/coordinate encoders (shared with the LPF baseline's front-end) and leave the
    dense decoder and the zero-init Helmholtz local_field generator fresh -- the two
    components under test.
    """

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = payload["model_state"] if "model_state" in payload else payload
    target_keys = set(model.state_dict().keys())
    usable = {
        key: value
        for key, value in state.items()
        if key in target_keys and any(key.startswith(p) for p in _FRONT_END_PREFIXES)
    }
    result = model.load_state_dict(usable, strict=False)
    missing = [k for k in result.missing_keys]
    fresh_front_end = [
        k for k in missing if any(k.startswith(p) for p in _FRONT_END_PREFIXES)
    ]
    if fresh_front_end:
        raise ValueError(
            "permissive front-end transfer left front-end tensors uninitialized: "
            + ", ".join(fresh_front_end[:5])
        )
    return {
        "checkpoint": str(checkpoint_path),
        "transferred_tensor_count": len(usable),
        "decoder_tensors_left_fresh": sum(1 for k in missing if k.startswith("dense_decoder.")),
        "local_field_tensors_left_fresh": sum(1 for k in missing if k.startswith("local_field.")),
    }


def warmstart_full_helmholtz(model, checkpoint_path: Path) -> dict:
    """Continue-pretraining load: transfer every shape-matching tensor from a full A+1
    checkpoint, leaving ONLY the fresh late-rank head (``*.late_*``) new.

    The A+1 parent has no late-rank tensors, so those keys are missing on load; every
    OTHER tensor (front-end, dense decoder, rank-8 Helmholtz synthesis) must match by
    shape. Any non-``late_`` missing key, unexpected source key, or shape mismatch is a
    hard error (the architecture would not actually match the parent). Because the late
    mixing is zero-initialised, the warm-started model reproduces the A+1 parent exactly
    at step 0 -- the continue-pretraining contract.
    """
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    source = payload["model_state"] if "model_state" in payload else payload
    target = model.state_dict()
    loaded, shape_mismatch = {}, []
    for key, value in source.items():
        if key not in target:
            continue
        if tuple(target[key].shape) != tuple(value.shape):
            shape_mismatch.append(key)
            continue
        loaded[key] = value
    result = model.load_state_dict(loaded, strict=False)
    unexpected = sorted(k for k in source if k not in target)
    missing = sorted(result.missing_keys)
    # Allowed-fresh modules are exact zero-init residuals that the parent lacks.
    def _allowed_fresh(k):
        return (
            ".late_" in k
            or "helmholtz_spectral_bypass" in k
            or "helmholtz_background_conditioner" in k
        )
    non_late_missing = [k for k in missing if not _allowed_fresh(k)]
    if shape_mismatch:
        raise ValueError(f"warm-start shape mismatch (arch differs from A+1): {shape_mismatch[:5]}")
    if unexpected:
        raise ValueError(f"warm-start unexpected source keys: {unexpected[:5]}")
    if non_late_missing:
        raise ValueError(
            "warm-start left non-allowed tensors uninitialized (arch mismatch): "
            + ", ".join(non_late_missing[:5])
        )
    return {
        "checkpoint": str(checkpoint_path),
        "transferred_tensor_count": len(loaded),
        "fresh_tensors": sum(1 for k in missing if _allowed_fresh(k)),
        "mode": "continue_pretrain_full_helmholtz",
    }


def build_background_conditioner_stage2_optimizer(
    conditioner: torch.nn.Module,
    *,
    output_learning_rate: float,
    core_learning_rate: float,
    coupled_learning_rate: float | None = None,
    expert_learning_rate: float | None = None,
    expert_router_learning_rate: float | None = None,
    adam_epsilon: float = 1.0e-8,
    late_head: torch.nn.Module | None = None,
    late_learning_rate: float | None = None,
) -> tuple[torch.optim.Optimizer, dict]:
    """AdamW with a fast warmed output projection and a cautious fresh core.

    Stage 1 can safely wake the zero-initialised ``output_projection``, but keeping
    the input/global propagation stack frozen leaves a fixed random feature map and
    empirically predicts almost no scattered energy.  Stage 2 therefore unfreezes
    the complete conditioner while giving every non-output tensor a much smaller LR.
    """

    output_lr = float(output_learning_rate)
    core_lr = float(core_learning_rate)
    coupled_lr = (
        None if coupled_learning_rate is None else float(coupled_learning_rate)
    )
    expert_lr = None if expert_learning_rate is None else float(expert_learning_rate)
    expert_router_lr = (
        None
        if expert_router_learning_rate is None
        else float(expert_router_learning_rate)
    )
    epsilon = float(adam_epsilon)
    if not math.isfinite(output_lr) or output_lr <= 0.0:
        raise ValueError("background conditioner output learning rate must be positive")
    if not math.isfinite(core_lr) or core_lr <= 0.0:
        raise ValueError("background conditioner core learning rate must be positive")
    if coupled_lr is not None and (
        not math.isfinite(coupled_lr) or coupled_lr <= 0.0
    ):
        raise ValueError("background conditioner coupled learning rate must be positive")
    if expert_lr is not None and (
        not math.isfinite(expert_lr) or expert_lr <= 0.0
    ):
        raise ValueError("background conditioner expert learning rate must be positive")
    if expert_router_lr is not None and (
        not math.isfinite(expert_router_lr) or expert_router_lr <= 0.0
    ):
        raise ValueError(
            "background conditioner expert router learning rate must be positive"
        )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("background conditioner Adam epsilon must be positive")
    if (late_head is None) != (late_learning_rate is None):
        raise ValueError("late head and late learning rate must be provided together")

    output_parameters = []
    core_parameters = []
    coupled_parameters = []
    expert_parameters = []
    expert_router_parameters = []
    output_parameter_count = 0
    core_parameter_count = 0
    coupled_parameter_count = 0
    expert_parameter_count = 0
    expert_router_parameter_count = 0
    for name, parameter in conditioner.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("output_projection."):
            output_parameters.append(parameter)
            output_parameter_count += int(parameter.numel())
        elif coupled_lr is not None and name == "propagation_coupled_gates":
            coupled_parameters.append(parameter)
            coupled_parameter_count += int(parameter.numel())
        elif (
            expert_router_lr is not None
            and name.startswith(
                "direct_spectral_propagator.contrast_expert_gate."
            )
        ):
            expert_router_parameters.append(parameter)
            expert_router_parameter_count += int(parameter.numel())
        elif expert_lr is not None and name.startswith("direct_spectral_propagator."):
            expert_parameters.append(parameter)
            expert_parameter_count += int(parameter.numel())
        else:
            core_parameters.append(parameter)
            core_parameter_count += int(parameter.numel())
    if not output_parameters:
        raise ValueError("stage-2 conditioner has no trainable output projection")
    if not core_parameters:
        raise ValueError("stage-2 conditioner has no trainable propagation core")

    groups = [
        {"params": output_parameters, "lr": output_lr},
        {"params": core_parameters, "lr": core_lr},
    ]
    if coupled_lr is not None:
        if not coupled_parameters:
            raise ValueError("stage-2 conditioner has no trainable coupled propagation gates")
        groups.append({"params": coupled_parameters, "lr": coupled_lr})
    if expert_lr is not None:
        if not expert_parameters:
            raise ValueError("stage-2 conditioner has no trainable direct spectral experts")
        groups.append({"params": expert_parameters, "lr": expert_lr})
    if expert_router_lr is not None:
        if not expert_router_parameters:
            raise ValueError(
                "stage-2 conditioner has no trainable direct spectral expert router"
            )
        groups.append({"params": expert_router_parameters, "lr": expert_router_lr})
    late_parameter_count = 0
    if late_head is not None:
        late_lr = float(late_learning_rate)
        if not math.isfinite(late_lr) or late_lr <= 0.0:
            raise ValueError("Helmholtz late-head learning rate must be positive")
        late_parameters = [
            parameter
            for name, parameter in late_head.named_parameters()
            if parameter.requires_grad and name.startswith("late_")
        ]
        late_parameter_count = sum(
            int(parameter.numel()) for parameter in late_parameters
        )
        if not late_parameters:
            raise ValueError("stage-2 Helmholtz synthesis has no trainable late head")
        groups.append({"params": late_parameters, "lr": late_lr})

    optimizer = torch.optim.AdamW(
        groups,
        lr=output_lr,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        eps=epsilon,
    )
    report = {
        "kind": "background_conditioner_stage2_layered_adamw",
        "output_learning_rate": output_lr,
        "core_learning_rate": core_lr,
        "coupled_learning_rate": coupled_lr,
        "expert_learning_rate": expert_lr,
        "expert_router_learning_rate": expert_router_lr,
        "adam_epsilon": epsilon,
        "output_parameter_count": output_parameter_count,
        "core_parameter_count": core_parameter_count,
        "coupled_parameter_count": coupled_parameter_count,
        "expert_parameter_count": expert_parameter_count,
        "expert_router_parameter_count": expert_router_parameter_count,
        "late_learning_rate": (
            None if late_learning_rate is None else float(late_learning_rate)
        ),
        "late_parameter_count": late_parameter_count,
    }
    return optimizer, report


def resolve_best_warmstart(terminal_path: str | Path) -> tuple[Path, dict]:
    """Resolve and audit the checkpoint explicitly selected as best by a completed run."""

    terminal = Path(terminal_path).expanduser().resolve()
    if not terminal.is_file():
        raise FileNotFoundError(terminal)
    payload = json.loads(terminal.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"warm-start parent run is not complete: {terminal}")
    checkpoint = Path(str(payload.get("best_checkpoint", ""))).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"best checkpoint from {terminal} is missing: {checkpoint}")
    expected = float(payload["best_fixed_heldout_relative_l2"])
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    observed = float(
        checkpoint_payload.get("metrics", {}).get("triplet_aggregate_relative_l2", float("nan"))
    )
    if not math.isfinite(observed) or not math.isclose(
        observed, expected, rel_tol=1.0e-10, abs_tol=1.0e-12
    ):
        raise ValueError(
            "parent terminal/checkpoint best metric mismatch: "
            f"terminal={expected}, checkpoint={observed}"
        )
    return checkpoint, {
        "mode": "terminal_best_checkpoint",
        "terminal": str(terminal),
        "checkpoint": str(checkpoint),
        "selection_metric": "best_fixed_heldout_relative_l2",
        "selection_value": expected,
        "checkpoint_global_step": int(checkpoint_payload.get("global_step", -1)),
    }


def audit_source_replacement_gate(
    gate_path: str | Path,
    *,
    source_h5: str | Path,
    manifest_digest: str,
    source_manifest_sha256: str,
    sample_count: int,
) -> dict[str, object]:
    """Bind an audited train-only replacement VDS to this training run.

    The mixed v5 dataset keeps every validation/test row unchanged while replacing
    the invalid legacy train/Marmousi rows.  Parent normalization scales may only be
    reused when the replacement pipeline's PASS report cryptographically binds the
    exact VDS and manifest opened by this process.
    """

    report = Path(gate_path).expanduser().resolve()
    try:
        payload = json.loads(report.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid source replacement gate: {error}") from error
    if payload.get("status") != "PASS":
        raise ValueError("source replacement gate status must be PASS")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("source replacement gate has no bindings")
    source = Path(source_h5).expanduser().resolve()
    observed_source_sha256 = sha256_file(source)
    if str(bindings.get("hybrid_dataset_sha256", "")) != observed_source_sha256:
        raise ValueError("replacement gate hybrid dataset SHA256 does not match --source-h5")
    bound_manifest = str(bindings.get("hybrid_manifest_sha256", ""))
    if bound_manifest != str(source_manifest_sha256):
        raise ValueError("replacement gate manifest SHA256 does not match source manifest")
    if int(payload.get("sample_count", -1)) != int(sample_count):
        raise ValueError("replacement gate sample count does not match source dataset")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("source replacement gate has no protocol audit")
    if protocol.get("replacement_scope") != "train/Marmousi only":
        raise ValueError("source replacement gate scope must be train/Marmousi only")
    if not bool(protocol.get("coordinate_arrays_identical", False)):
        raise ValueError("source replacement gate did not preserve saved coordinates")
    return {
        "report": str(report),
        "source_sha256": observed_source_sha256,
        "manifest_digest": str(manifest_digest),
        "source_manifest_sha256": str(source_manifest_sha256),
        "sample_count": int(sample_count),
        "replacement_sample_count": int(payload.get("replacement_sample_count", 0)),
        "unchanged_nontrain_sample_count": int(
            payload.get("unchanged_nontrain_sample_count", 0)
        ),
        "replacement_scope": str(protocol["replacement_scope"]),
        "internal_dt_mixed": bool(protocol.get("internal_dt_mixed", False)),
    }


def select_family_balanced_pool(
    records,
    *,
    splits: Sequence[str],
    per_family: int | Mapping[str, int],
    excluded_sample_ids: Sequence[str] = (),
) -> tuple[int, ...]:
    """Selected-view indices of the first ``per_family`` records of each family.

    Returns indices into the split-filtered record list (the same convention the
    ExactStoredTimeBatchDataset schedule uses), family-blocked in FAMILIES order.
    A reduced training pool for the A+1 G3 test: the full 2240-record pool needs a
    2240-solve smoothed-background cache (~13h / ~80GB), out of scope on this host, so
    we test generalization on a family-balanced subsample whose P_bg cache is cheap.
    """

    selected_splits = set(str(value) for value in splits)
    split_records = [r for r in records if str(r.split) in selected_splits]
    excluded = {str(value) for value in excluded_sample_ids}
    requested = {
        family: int(
            per_family[family] if isinstance(per_family, Mapping) else per_family
        )
        for family in FAMILIES
    }
    pool: list[int] = []
    for family in FAMILIES:
        limit = requested[family]
        if limit <= 0:
            raise ValueError("per-family record limits must be positive")
        picked = [
            i
            for i, r in enumerate(split_records)
            if str(r.medium_type) == family
            and str(getattr(r, "sample_id", "")) not in excluded
        ]
        if len(picked) < limit:
            raise ValueError(
                f"splits {tuple(splits)!r} family {family!r} has {len(picked)} records < "
                f"requested per_family={limit}"
            )
        pool.extend(picked[:limit])
    return tuple(pool)


def select_one_index_per_family_after_skip(
    records: Sequence[object], *, split: str, skip_per_family: int = 0
) -> tuple[int, ...]:
    """Return deterministic split-relative held-out indices after a family-local skip."""

    selected_records = tuple(record for record in records if str(record.split) == str(split))
    skip = int(skip_per_family)
    if skip < 0:
        raise ValueError("held-out family skip must be nonnegative")
    result: list[int] = []
    for family in FAMILIES:
        matches = [
            index
            for index, record in enumerate(selected_records)
            if str(record.medium_type) == family
        ]
        if len(matches) <= skip:
            raise ValueError(
                f"split {split!r} family {family!r} has no record after skip={skip}"
            )
        result.append(int(matches[skip]))
    return tuple(result)


def _remap_schedule_to_pool(
    schedule: tuple[FullSupportStepSpec, ...], pool: tuple[int, ...]
) -> tuple[FullSupportStepSpec, ...]:
    """Rewrite a full_support schedule built over range(len(pool)) onto real
    split-relative indices ``pool``. ``build_full_support_schedule(len(pool))`` emits
    record_indices in 0..len(pool)-1; each is a position into ``pool``."""

    remapped: list[FullSupportStepSpec] = []
    for spec in schedule:
        remapped.append(
            dataclasses_replace_indices(spec, tuple(pool[i] for i in spec.record_indices))
        )
    return tuple(remapped)


def dataclasses_replace_indices(spec: FullSupportStepSpec, new_indices: tuple[int, ...]):
    import dataclasses
    return dataclasses.replace(spec, record_indices=new_indices)


def _train_schedule(
    train_record_count: int,
    *,
    epochs: int,
    macro_records: int,
    macros_per_update: int,
    seed: int,
    epoch_offset: int = 0,
    record_weights: np.ndarray | None = None,
    record_oversample: float = 1.0,
) -> tuple[FullSupportStepSpec, ...]:
    """Full-pool epoch-distinct schedule: one appearance per record per epoch.

    ``record_weights``/``record_oversample`` opt into residual-adaptive oversampling
    (defaults keep the uniform permutation bit-for-bit); ``epoch_offset`` replays the
    inherited appearance counter so a schedule rebuilt mid-run for a later epoch
    segment continues the ``appearance16`` time-window sequence seamlessly.
    """

    return build_full_support_schedule(
        train_record_count,
        epochs=int(epochs),
        macro_records=int(macro_records),
        macros_per_update=int(macros_per_update),
        seed=int(seed),
        epoch_offset=int(epoch_offset),
        record_weights=record_weights,
        record_oversample=float(record_oversample),
    )


def _train_dataset(
    config,
    base,
    manifest,
    schedule,
    *,
    frames_per_record: int,
    training_splits: Sequence[str] = ("train",),
    time_index_pool: Sequence[int] | None = None,
    time_index_probabilities: Sequence[float] | None = None,
):
    return ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split=tuple(training_splits),
        schedule=tuple(schedule),
        query_points=1,
        seed=int(config["seed"]) + 17_171,
        time_policy=(
            "appearance16" if time_index_pool is None else "numerical_teacher_pool"
        ),
        frames_per_record=int(frames_per_record),
        travel_time_h5=config.get("travel_time_h5"),
        allow_travel_source_path_mismatch=bool(
            config.get("allow_travel_source_path_mismatch", False)
        ),
        time_index_pool=time_index_pool,
        time_index_probabilities=time_index_probabilities,
    )


def fourier_training_design_report(
    time_values_s: Sequence[float],
    time_indices: Sequence[int],
    *,
    frequencies: int,
) -> dict[str, int | float]:
    """Audit identifiability of a real low-bin Fourier head on selected frames.

    The zero-frequency sine column is identically zero and is intentionally omitted
    from the effective design.  A direct frequency head should only be trained when
    the remaining ``2*K-1`` columns are full rank and well conditioned.
    """

    values = np.asarray(tuple(float(value) for value in time_values_s), dtype=np.float64)
    indices = np.asarray(tuple(int(value) for value in time_indices), dtype=np.int64)
    count = int(frequencies)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or np.any(np.diff(values) <= 0.0)
        or indices.ndim != 1
        or indices.size < 1
        or len(set(indices.tolist())) != indices.size
        or np.any(indices < 0)
        or np.any(indices >= values.size)
        or count <= 0
    ):
        raise ValueError("Fourier training design inputs are invalid")
    dt = (values[-1] - values[0]) / float(values.size - 1)
    omega = 2.0 * math.pi * np.arange(count, dtype=np.float64) / (
        float(values.size) * dt
    )
    phase = values[indices, None] * omega[None, :]
    design = np.concatenate((np.cos(phase), np.sin(phase)[:, 1:]), axis=1)
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    condition = (
        float("inf")
        if singular.size == 0 or float(singular[-1]) <= 0.0
        else float(singular[0] / singular[-1])
    )
    return {
        "frame_count": int(indices.size),
        "effective_coefficient_count": int(design.shape[1]),
        "rank": rank,
        "condition_number": condition,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--source-h5",
        help="explicit source dataset/VDS; overrides the legacy path in the base config",
    )
    parser.add_argument("--normalization-json")
    parser.add_argument(
        "--allow-filtered-invalidated-source",
        action="store_true",
        help="explicitly permit a retained VDS whose deleted/fill rows were filtered "
        "from the manifest; requires --source-invalidation-report",
    )
    parser.add_argument(
        "--source-invalidation-report",
        help="INVALIDATED audit JSON binding the retained original VDS",
    )
    parser.add_argument(
        "--source-replacement-gate",
        help="PASS audit JSON binding a train/Marmousi replacement VDS; permits an "
        "unchanged parent normalization only after dataset+manifest hash verification",
    )
    parser.add_argument(
        "--allow-parent-normalization-binding",
        action="store_true",
        help="reuse the parent normalization scales when the filtered retained VDS "
        "has a different manifest digest; both digests are recorded",
    )
    parser.add_argument(
        "--allow-travel-source-path-alias",
        action="store_true",
        help="allow the travel cache to name the pre-invalidation public VDS path; "
        "sample-ID lookup still enforces row alignment",
    )
    parser.add_argument("--warmstart-frontend", default=None,
                        help="width-128 parent best.pt; front-end transferred, decoder+local_field fresh. "
                        "Required unless --warmstart-helmholtz is given (continue-pretraining).")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dense-depth", type=int, default=8)
    parser.add_argument("--dense-spectral-rank", type=int, default=112)
    parser.add_argument("--dense-modes", type=int, default=32)
    parser.add_argument("--helmholtz-frequencies", type=int, default=64)
    parser.add_argument("--helmholtz-rank", type=int, default=8)
    parser.add_argument("--helmholtz-late-rank", type=int, default=0,
                        help="dedicated higher-rank late-multiple basis added to the lowest "
                        "late-frequencies bins (zero-init -> reproduces A+1 at start). 0 disables.")
    parser.add_argument("--helmholtz-late-frequencies", type=int, default=0,
                        help="number of lowest freq bins the late-rank basis feeds (0 -> min(24, nf))")
    parser.add_argument("--helmholtz-spectral-bypass", action="store_true",
                        help="multi-scale spectral bypass to break U-Net rank collapse (zero-init no-op on A+1)")
    parser.add_argument("--helmholtz-spectral-bypass-per-branch", action="store_true",
                        help="progressive-frequency: per-mode-scale learnable gates (freq-decaying init) so high modes enter gradually (incremental-FNO idea)")
    parser.add_argument(
        "--helmholtz-background-conditioning",
        action="store_true",
        help="feed WKB-demodulated Born source omega^2*(m-m_bg)*P_bg into the Helmholtz render",
    )
    parser.add_argument(
        "--helmholtz-background-sigma-cells",
        type=float,
        default=2.0,
        help="saved-grid Gaussian sigma used to reconstruct m_bg for the Born contrast",
    )
    parser.add_argument(
        "--helmholtz-background-global-propagator",
        action="store_true",
        help="propagate the Born source globally after the Helmholtz U-Net instead of "
        "injecting it only through the local pre-render path",
    )
    parser.add_argument(
        "--helmholtz-background-propagation-modes",
        type=int,
        default=48,
        help="maximum spatial Fourier modes used by the global Born propagator",
    )
    parser.add_argument(
        "--helmholtz-background-direct-frequency-head",
        action="store_true",
        help="render the globally propagated Born branch as raw complex scattering "
        "frequencies, bypassing the frozen direct-ray WKB/rank-8 synthesis bottleneck",
    )
    parser.add_argument(
        "--helmholtz-background-direct-frequencies",
        type=int,
        default=32,
        help="complex scattering bins in the direct head; 32 gives 64 real temporal "
        "coefficients, identifiable from the standard 64 training frames",
    )
    parser.add_argument(
        "--helmholtz-background-direct-spectral-experts",
        type=int,
        default=0,
        help="number of zero-init medium-conditioned, per-frequency full-2D Born "
        "transfer kernels; zero disables the diagnostic branch",
    )
    parser.add_argument(
        "--background-conditioner-only",
        action="store_true",
        help="freeze the validated parent and train only the new P_bg/Born conditioner",
    )
    parser.add_argument(
        "--background-conditioner-output-only",
        action="store_true",
        help="stage-1 stabilization: within --background-conditioner-only, train only "
        "the zero-init global output projection before unfreezing its propagator",
    )
    parser.add_argument("--warmstart-helmholtz",
                        help="continue-pretraining: load a full A+1 checkpoint (strict=False) so "
                        "the rank-8 synthesis + front-end + decoder all transfer and ONLY the "
                        "fresh late-rank head is left new. Mutually informative with --helmholtz-late-rank.")
    parser.add_argument(
        "--warmstart-best-terminal",
        help="resolve best_checkpoint from this completed parent terminal.json and verify "
        "that its stored selection metric matches the checkpoint payload",
    )
    parser.add_argument(
        "--require-best-warmstart",
        action="store_true",
        help="reject direct checkpoint warm-starts; continuation must use "
        "--warmstart-best-terminal",
    )
    parser.add_argument("--helmholtz-no-wkb", action="store_true")
    parser.add_argument("--local-field-learning-rate", type=float, default=5.0e-4)
    parser.add_argument(
        "--background-conditioner-output-learning-rate",
        type=float,
        help="stage-2 LR for the already-woken global output projection; must be paired "
        "with --background-conditioner-core-learning-rate",
    )
    parser.add_argument(
        "--background-conditioner-core-learning-rate",
        type=float,
        help="stage-2 LR for the newly-unfrozen input/global propagation stack",
    )
    parser.add_argument(
        "--background-conditioner-coupled-learning-rate",
        type=float,
        help="optional stage-2 LR for the zero-init joint x-z propagation gates; "
        "requires the paired output/core stage-2 learning rates",
    )
    parser.add_argument(
        "--background-conditioner-expert-learning-rate",
        type=float,
        help="optional stage-2 LR isolated to direct spectral expert parameters; "
        "requires a positive direct spectral expert count",
    )
    parser.add_argument(
        "--background-conditioner-expert-router-learning-rate",
        type=float,
        help="optional stage-2 LR isolated to the zero-init contrast expert router; "
        "requires a positive direct spectral expert count",
    )
    parser.add_argument(
        "--background-conditioner-adam-epsilon",
        type=float,
        default=1.0e-8,
        help="AdamW epsilon for the small-gradient background-conditioner stage",
    )
    parser.add_argument(
        "--background-conditioner-with-late-head",
        action="store_true",
        help="also train the zero-init Helmholtz late-rank head so reflected coda is "
        "not confined to the frozen rank-8 temporal/spatial basis",
    )
    parser.add_argument(
        "--helmholtz-late-learning-rate",
        type=float,
        help="learning rate for late_* synthesis tensors when the stage-2 late head is enabled",
    )
    parser.add_argument("--local-field-grad-clip", type=float, default=20.0)
    parser.add_argument("--dense-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--temporal-basis-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--optimizer", choices=["adamw", "soap"], default="adamw",
                        help="continue-pretraining optimizer; soap = second-order "
                        "Shampoo-style preconditioner (pytorch-optimizer).")
    parser.add_argument(
        "--residual-activation",
        choices=["preserve", "absorb", "rescale", "reset"],
        default="absorb",
        help="wake-up policy for the transferred dense residual head; absorb preserves "
        "the update-0 function while removing a collapsed scalar gradient gate",
    )
    parser.add_argument("--residual-correction-scale", type=float, default=1.0)
    parser.add_argument("--residual-output-std", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--macro-records", type=int, default=6)
    parser.add_argument("--macros-per-update", type=int, default=1)
    parser.add_argument("--microbatch-records", type=int, default=1)
    parser.add_argument("--training-frames", type=int, default=16)
    parser.add_argument("--validation-frames", type=int, default=32)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "train"),
        default="validation",
        help="fixed triplet split; train is only for an explicit capacity/overfit diagnostic",
    )
    parser.add_argument(
        "--evaluation-skip-records-per-family",
        type=int,
        default=0,
        help="select the fixed evaluation record after skipping this many records within each family",
    )
    parser.add_argument(
        "--exclude-evaluation-records-from-training",
        action="store_true",
        help="remove the fixed evaluation triplet from the optimization pool for a disjoint train-only gate",
    )
    parser.add_argument(
        "--per-frame-frame",
        action="store_true",
        help="normalize the data term per selected frame so low-energy late frames receive gradient",
    )
    parser.add_argument(
        "--frame-energy-floor-fraction",
        type=float,
        default=0.0,
        help="floor each frame denominator to this fraction of the record peak (requires --per-frame-frame)",
    )
    parser.add_argument(
        "--full-field-frame-weight",
        type=float,
        default=1.0,
        help="weight of the full-field frame loss; reflection-only recovery may set this to zero",
    )
    parser.add_argument(
        "--delta-loss-weight",
        type=float,
        default=0.5,
        help="weight of the predicted correction versus target-minus-coarse loss",
    )
    parser.add_argument(
        "--delta-reference",
        choices=("model_coarse", "target"),
        default="model_coarse",
        help="use target for background-retargeted scattering recovery; legacy recovery subtracts model coarse",
    )
    parser.add_argument(
        "--delta-reduction",
        choices=("per_record", "global", "global_squared"),
        default="per_record",
        help="global_squared streams the squared full-batch scattering L2 across microbatches",
    )
    parser.add_argument(
        "--delta-energy-floor-fraction",
        type=float,
        default=0.1,
        help="floor the correction denominator by this fraction of full-field energy",
    )
    parser.add_argument(
        "--spatial-gradient-loss-weight",
        type=float,
        default=0.1,
        help="weight of the spatial-gradient field loss",
    )
    parser.add_argument(
        "--late-frame-gain",
        type=float,
        default=0.0,
        help="additional linear late-time weight applied after --late-frame-start-fraction",
    )
    parser.add_argument("--late-frame-start-fraction", type=float, default=0.4)
    parser.add_argument("--uniform-gradient-weight", type=float, default=1.0)
    parser.add_argument("--layered-gradient-weight", type=float, default=1.0)
    parser.add_argument("--marmousi-gradient-weight", type=float, default=1.0)
    parser.add_argument("--uniform-sampling-weight", type=float, default=1.0)
    parser.add_argument("--layered-sampling-weight", type=float, default=1.0)
    parser.add_argument("--marmousi-sampling-weight", type=float, default=1.0)
    parser.add_argument("--evaluate-every", type=int, default=200,
                        help="optimizer updates between held-out validation_fixed evaluations")
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=["aggregate_relative_l2", "scattering_relative_l2"],
        default="aggregate_relative_l2",
        help="validation metric used to select the reproducible best checkpoint; "
        "reflection-recovery runs should use scattering_relative_l2",
    )
    parser.add_argument("--travel-time-h5",
                        default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5")
    parser.add_argument("--seed", type=int, default=372)
    parser.add_argument(
        "--background-cache",
        help="Direction A+1: NumericalTeacherCache-format smoothed-velocity background "
        "P_bg covering the training pool AND the held-out triplet. Training targets are "
        "retargeted to encode(scat)=encode(wf)-encode(P_bg); evaluation adds P_bg back. "
        "At held-out this P_bg is a fresh physical smoothed solve (no target leak), so a "
        "low gap here vs the 0.639 plain-Helmholtz G3 isolates the background's value.",
    )
    parser.add_argument(
        "--background-cache-shard",
        action="append",
        default=[],
        help="additional disjoint full/sparse P_bg cache; repeatable",
    )
    parser.add_argument(
        "--training-time-pool-from-background",
        action="store_true",
        help="sample exact training frames only from the common time-index pool exposed "
        "by the background cache set",
    )
    parser.add_argument(
        "--fourier-uniform-training-times",
        action="store_true",
        help="use --training-frames exact indices uniformly spanning the complete stored "
        "axis; required by the direct frequency head so its temporal design is full-rank "
        "and well-conditioned",
    )
    parser.add_argument(
        "--time-adaptive-sampling",
        action="store_true",
        help="rebuild the next segment's exact-time sampling PDF from residual EMA by "
        "actual saved-time index",
    )
    parser.add_argument(
        "--time-sampling-uniform-fraction",
        type=float,
        default=0.25,
        help="uniform coverage floor mixed into residual-adaptive time sampling",
    )
    parser.add_argument(
        "--time-residual-ema-momentum",
        type=float,
        default=0.3,
        help="EMA momentum for per-saved-time residuals between schedule rebuilds",
    )
    parser.add_argument(
        "--training-splits",
        nargs="+",
        default=["train"],
        choices=["train", "validation", "test_id", "ood_canonical"],
        help="source splits included in optimization; anomaly is already excluded by the manifest",
    )
    parser.add_argument(
        "--records-per-family",
        type=int,
        default=0,
        help="if >0, train on a family-balanced subsample of this many records per family "
        "instead of the full 2240 pool (required when the background cache only covers a "
        "reduced pool; the smoothed-solve cost makes the full pool infeasible on this host)",
    )
    parser.add_argument("--uniform-records", type=int, default=0)
    parser.add_argument("--layered-records", type=int, default=0)
    parser.add_argument("--marmousi-records", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--smoke-updates", type=int, default=0,
                        help="if >0, run only this many optimizer updates and one eval (fast wiring check)")
    # vRBA record-level residual-adaptive sampling (Phase 2). Default `none` keeps the
    # legacy uniform full-support schedule bit-for-bit; `rad`/`vrba` enable it (the
    # frame-level RBA half of `vrba` lands in Phase 3, so here they behave identically).
    parser.add_argument("--adaptive-sampling", choices=["none", "rad", "rba", "vrba"],
                        default="none",
                        help="record-level residual-adaptive sampling: none (uniform), "
                        "rad/vrba (oversample high-error records via vrba_record_pdf)")
    parser.add_argument("--rad-recompute-epochs", type=int, default=0,
                        help="rebuild the sampling schedule from the residual EMA every K "
                        "epochs (0 disables; required >0 when --adaptive-sampling is rad/vrba)")
    parser.add_argument("--record-oversample", type=float, default=1.0,
                        help="lengthen each epoch to (oversample x record_count) draws so "
                        "high-error records recur; 1.0 keeps one appearance per record")
    parser.add_argument("--rad-potential",
                        choices=["linear", "sublinear", "quadratic", "lp", "exponential", "logarithmic"],
                        default="quadratic",
                        help="vRBA potential mapping residual->sampling weight (quadratic=L2, gentle)")
    parser.add_argument("--rad-uniform-fraction", type=float, default=0.2,
                        help="convex mix toward uniform in the record PDF (coverage floor)")
    parser.add_argument("--rad-ema-momentum", type=float, default=0.3,
                        help="alpha of the per-record relative-L2 residual EMA across recompute cycles")
    # vRBA frame-level RBA (Phase 3): bounded-EMA temporal attention on the per-frame
    # loss weight, replacing the fixed late_frame_gain ramp. Requires --per-frame-frame.
    # `off` (default) keeps the frame loss unweighted bit-for-bit; the rba/vrba sampling
    # modes turn it on automatically unless overridden.
    parser.add_argument("--frame-rba", choices=["off", "on"], default="off",
                        help="bounded-EMA per-frame attention (vRBA RBA); requires --per-frame-frame")
    parser.add_argument("--rba-gamma", type=float, default=0.999,
                        help="frame-RBA EMA retention (bound = eta/(1-gamma))")
    parser.add_argument("--rba-eta", type=float, default=0.01,
                        help="frame-RBA EMA injection rate")
    parser.add_argument("--rba-phi", type=float, default=0.9,
                        help="fraction of the frame weight driven by the residual potential")
    parser.add_argument("--rba-potential",
                        choices=["linear", "sublinear", "quadratic", "lp", "exponential", "logarithmic"],
                        default="quadratic",
                        help="vRBA potential for frame RBA (quadratic=L2 gentle, exponential=L-inf aggressive)")
    args = parser.parse_args(argv)
    if args.allow_filtered_invalidated_source:
        if not args.source_h5 or not args.source_invalidation_report:
            parser.error(
                "--allow-filtered-invalidated-source requires --source-h5 and "
                "--source-invalidation-report"
            )
    elif args.source_invalidation_report:
        parser.error(
            "--source-invalidation-report requires --allow-filtered-invalidated-source"
        )
    if args.source_replacement_gate and args.allow_filtered_invalidated_source:
        parser.error(
            "use either --source-replacement-gate or --allow-filtered-invalidated-source"
        )
    if args.source_replacement_gate and not args.source_h5:
        parser.error("--source-replacement-gate requires --source-h5")
    if (
        args.allow_parent_normalization_binding
        or args.allow_travel_source_path_alias
    ) and not (
        args.allow_filtered_invalidated_source or args.source_replacement_gate
    ):
        parser.error(
            "parent normalization/travel aliases require an explicitly audited "
            "filtered or replacement source"
        )
    requested_family_limits = {
        "uniform": int(args.uniform_records),
        "layered": int(args.layered_records),
        "marmousi": int(args.marmousi_records),
    }
    if any(requested_family_limits.values()):
        if int(args.records_per_family) > 0:
            parser.error(
                "use either --records-per-family or the three family-specific limits"
            )
        if any(value <= 0 for value in requested_family_limits.values()):
            parser.error(
                "--uniform-records, --layered-records, and --marmousi-records "
                "must all be positive when any is set"
            )
        family_record_limits: dict[str, int] | None = requested_family_limits
    else:
        family_record_limits = None
    warmstart_selection = None
    if args.warmstart_best_terminal and args.warmstart_helmholtz:
        parser.error("use only one of --warmstart-best-terminal and --warmstart-helmholtz")
    if args.warmstart_best_terminal:
        try:
            best_path, warmstart_selection = resolve_best_warmstart(
                args.warmstart_best_terminal
            )
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        args.warmstart_helmholtz = str(best_path)
    if args.require_best_warmstart and warmstart_selection is None:
        parser.error("--require-best-warmstart requires --warmstart-best-terminal")
    if not args.warmstart_helmholtz and not args.warmstart_frontend:
        parser.error("one of --warmstart-frontend or --warmstart-helmholtz is required")
    if args.helmholtz_background_conditioning and not (
        args.background_cache or args.background_cache_shard
    ):
        parser.error("--helmholtz-background-conditioning requires --background-cache")
    if args.background_cache_shard and not args.background_cache:
        parser.error("--background-cache-shard requires a primary --background-cache")
    if (
        args.checkpoint_selection_metric == "scattering_relative_l2"
        and not args.background_cache
    ):
        parser.error(
            "--checkpoint-selection-metric scattering_relative_l2 requires "
            "--background-cache"
        )
    if args.training_time_pool_from_background and not args.background_cache:
        parser.error("--training-time-pool-from-background requires --background-cache")
    if args.fourier_uniform_training_times and args.training_time_pool_from_background:
        parser.error(
            "--fourier-uniform-training-times and "
            "--training-time-pool-from-background are mutually exclusive"
        )
    if args.time_adaptive_sampling and not args.training_time_pool_from_background:
        parser.error(
            "--time-adaptive-sampling requires --training-time-pool-from-background"
        )
    training_splits = tuple(str(value) for value in args.training_splits)
    if len(training_splits) != len(set(training_splits)):
        parser.error("--training-splits must not contain duplicates")
    if args.background_conditioner_only and not args.helmholtz_background_conditioning:
        parser.error("--background-conditioner-only requires --helmholtz-background-conditioning")
    if args.background_conditioner_output_only and not args.background_conditioner_only:
        parser.error(
            "--background-conditioner-output-only requires --background-conditioner-only"
        )
    if (
        args.background_conditioner_output_only
        and not args.helmholtz_background_global_propagator
    ):
        parser.error(
            "--background-conditioner-output-only requires the global propagator"
        )
    stage2_learning_rates = (
        args.background_conditioner_output_learning_rate,
        args.background_conditioner_core_learning_rate,
    )
    if any(value is not None for value in stage2_learning_rates):
        if not all(value is not None for value in stage2_learning_rates):
            parser.error(
                "stage-2 background conditioner output/core learning rates must be paired"
            )
        if not args.background_conditioner_only:
            parser.error("stage-2 background conditioner learning rates require --background-conditioner-only")
        if args.background_conditioner_output_only:
            parser.error("stage-2 background conditioner learning rates cannot use --background-conditioner-output-only")
        if not args.helmholtz_background_global_propagator:
            parser.error("stage-2 background conditioner learning rates require the global propagator")
        if args.optimizer != "adamw":
            parser.error("stage-2 layered background conditioner learning rates require --optimizer adamw")
        for value in stage2_learning_rates:
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                parser.error("stage-2 background conditioner learning rates must be positive")
        if (
            not math.isfinite(float(args.background_conditioner_adam_epsilon))
            or float(args.background_conditioner_adam_epsilon) <= 0.0
        ):
            parser.error("--background-conditioner-adam-epsilon must be positive")
    if args.background_conditioner_coupled_learning_rate is not None:
        if args.background_conditioner_output_learning_rate is None:
            parser.error(
                "--background-conditioner-coupled-learning-rate requires stage-2 "
                "output/core learning rates"
            )
        if (
            not math.isfinite(float(args.background_conditioner_coupled_learning_rate))
            or float(args.background_conditioner_coupled_learning_rate) <= 0.0
        ):
            parser.error(
                "--background-conditioner-coupled-learning-rate must be positive"
            )
    if args.background_conditioner_expert_learning_rate is not None:
        if args.background_conditioner_output_learning_rate is None:
            parser.error(
                "--background-conditioner-expert-learning-rate requires stage-2 "
                "output/core learning rates"
            )
        if int(args.helmholtz_background_direct_spectral_experts) <= 0:
            parser.error(
                "--background-conditioner-expert-learning-rate requires positive "
                "--helmholtz-background-direct-spectral-experts"
            )
        if (
            not math.isfinite(float(args.background_conditioner_expert_learning_rate))
            or float(args.background_conditioner_expert_learning_rate) <= 0.0
        ):
            parser.error(
                "--background-conditioner-expert-learning-rate must be positive"
            )
    if args.background_conditioner_expert_router_learning_rate is not None:
        if args.background_conditioner_output_learning_rate is None:
            parser.error(
                "--background-conditioner-expert-router-learning-rate requires "
                "stage-2 output/core learning rates"
            )
        if int(args.helmholtz_background_direct_spectral_experts) <= 0:
            parser.error(
                "--background-conditioner-expert-router-learning-rate requires "
                "positive --helmholtz-background-direct-spectral-experts"
            )
        if (
            not math.isfinite(
                float(args.background_conditioner_expert_router_learning_rate)
            )
            or float(args.background_conditioner_expert_router_learning_rate) <= 0.0
        ):
            parser.error(
                "--background-conditioner-expert-router-learning-rate must be positive"
            )
    if args.background_conditioner_with_late_head:
        if not args.background_conditioner_only:
            parser.error("--background-conditioner-with-late-head requires --background-conditioner-only")
        if int(args.helmholtz_late_rank) <= 0:
            parser.error("--background-conditioner-with-late-head requires --helmholtz-late-rank > 0")
        if args.helmholtz_late_learning_rate is None:
            parser.error("--background-conditioner-with-late-head requires --helmholtz-late-learning-rate")
        if args.background_conditioner_output_learning_rate is None:
            parser.error("--background-conditioner-with-late-head requires stage-2 layered learning rates")
        if (
            not math.isfinite(float(args.helmholtz_late_learning_rate))
            or float(args.helmholtz_late_learning_rate) <= 0.0
        ):
            parser.error("--helmholtz-late-learning-rate must be positive")
    elif args.helmholtz_late_learning_rate is not None:
        parser.error("--helmholtz-late-learning-rate requires --background-conditioner-with-late-head")
    if (
        args.helmholtz_background_global_propagator
        and not args.helmholtz_background_conditioning
    ):
        parser.error(
            "--helmholtz-background-global-propagator requires "
            "--helmholtz-background-conditioning"
        )
    if (
        args.helmholtz_background_direct_frequency_head
        and not args.helmholtz_background_global_propagator
    ):
        parser.error(
            "--helmholtz-background-direct-frequency-head requires the global propagator"
        )
    if (
        args.helmholtz_background_direct_frequency_head
        and not args.fourier_uniform_training_times
    ):
        parser.error(
            "--helmholtz-background-direct-frequency-head requires "
            "--fourier-uniform-training-times"
        )
    if int(args.helmholtz_background_direct_frequencies) <= 0:
        parser.error("--helmholtz-background-direct-frequencies must be positive")
    if int(args.helmholtz_background_direct_spectral_experts) < 0:
        parser.error(
            "--helmholtz-background-direct-spectral-experts must be nonnegative"
        )
    if (
        int(args.helmholtz_background_direct_spectral_experts) > 0
        and not args.helmholtz_background_direct_frequency_head
    ):
        parser.error(
            "--helmholtz-background-direct-spectral-experts requires the direct "
            "frequency head"
        )
    if int(args.helmholtz_background_direct_frequencies) > int(args.helmholtz_frequencies):
        parser.error(
            "direct scattering frequencies cannot exceed parent Helmholtz frequencies"
        )
    if (
        args.helmholtz_background_direct_frequency_head
        and 2 * int(args.helmholtz_background_direct_frequencies) - 1
        > int(args.training_frames)
    ):
        parser.error(
            "direct scattering head has more effective temporal coefficients than "
            "--training-frames"
        )
    if (
        args.helmholtz_background_direct_frequency_head
        and args.background_conditioner_with_late_head
    ):
        parser.error(
            "the direct Born frequency head and Helmholtz late head are alternative representations"
        )
    if float(args.helmholtz_background_sigma_cells) <= 0.0:
        parser.error("--helmholtz-background-sigma-cells must be positive")
    if int(args.helmholtz_background_propagation_modes) <= 0:
        parser.error("--helmholtz-background-propagation-modes must be positive")
    if (
        not math.isfinite(float(args.residual_correction_scale))
        or float(args.residual_correction_scale) <= 0.0
        or not math.isfinite(float(args.residual_output_std))
        or float(args.residual_output_std) <= 0.0
    ):
        parser.error("residual correction scale/output std must be finite and positive")
    if not 0.0 <= float(args.frame_energy_floor_fraction) <= 1.0:
        parser.error("--frame-energy-floor-fraction must lie in [0, 1]")
    if (
        not math.isfinite(float(args.full_field_frame_weight))
        or float(args.full_field_frame_weight) < 0.0
        or not math.isfinite(float(args.delta_loss_weight))
        or float(args.delta_loss_weight) < 0.0
        or not math.isfinite(float(args.spatial_gradient_loss_weight))
        or float(args.spatial_gradient_loss_weight) < 0.0
    ):
        parser.error("field/delta/gradient loss weights must be finite and nonnegative")
    if not 0.0 <= float(args.delta_energy_floor_fraction) <= 1.0:
        parser.error("--delta-energy-floor-fraction must lie in [0, 1]")
    if (
        float(args.full_field_frame_weight) == 0.0
        and float(args.delta_loss_weight) == 0.0
    ):
        parser.error("full-field and delta loss weights cannot both be zero")
    if float(args.late_frame_gain) < 0.0:
        parser.error("--late-frame-gain must be nonnegative")
    if not 0.0 <= float(args.late_frame_start_fraction) < 1.0:
        parser.error("--late-frame-start-fraction must lie in [0, 1)")
    family_gradient_weights = {
        "uniform": float(args.uniform_gradient_weight),
        "layered": float(args.layered_gradient_weight),
        "marmousi": float(args.marmousi_gradient_weight),
    }
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in family_gradient_weights.values()
    ):
        parser.error("all family gradient weights must be finite and positive")
    family_sampling_weights = {
        "uniform": float(args.uniform_sampling_weight),
        "layered": float(args.layered_sampling_weight),
        "marmousi": float(args.marmousi_sampling_weight),
    }
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in family_sampling_weights.values()
    ):
        parser.error("all family sampling weights must be finite and positive")
    if (
        float(args.frame_energy_floor_fraction) > 0.0
        or float(args.late_frame_gain) > 0.0
    ) and not bool(args.per_frame_frame):
        parser.error(
            "--frame-energy-floor-fraction/--late-frame-gain require --per-frame-frame"
        )

    adaptive_records = args.adaptive_sampling in {"rad", "vrba"}
    if adaptive_records:
        if int(args.rad_recompute_epochs) <= 0:
            parser.error(
                "--adaptive-sampling rad/vrba requires --rad-recompute-epochs > 0"
            )
        if float(args.record_oversample) <= 1.0:
            parser.error(
                "--adaptive-sampling rad/vrba requires --record-oversample > 1.0 "
                "(the base schedule has exactly one appearance per record, so extra "
                "weighted draws only appear when the epoch is oversampled)"
            )
    if not math.isfinite(float(args.record_oversample)) or float(args.record_oversample) < 1.0:
        parser.error("--record-oversample must be finite and >= 1.0")
    if not 0.0 <= float(args.rad_uniform_fraction) <= 1.0:
        parser.error("--rad-uniform-fraction must lie in [0, 1]")
    if not 0.0 <= float(args.rad_ema_momentum) <= 1.0:
        parser.error("--rad-ema-momentum must lie in [0, 1]")
    if not 0.0 <= float(args.time_sampling_uniform_fraction) <= 1.0:
        parser.error("--time-sampling-uniform-fraction must lie in [0, 1]")
    if not 0.0 <= float(args.time_residual_ema_momentum) <= 1.0:
        parser.error("--time-residual-ema-momentum must lie in [0, 1]")
    if args.time_adaptive_sampling and int(args.rad_recompute_epochs) <= 0:
        parser.error("--time-adaptive-sampling requires --rad-recompute-epochs > 0")
    # Frame-level RBA: on for the rba/vrba sampling modes, or when forced with
    # --frame-rba on. Requires the per-frame-normalized frame term (same contract as
    # late_frame_gain). rba mode = frame RBA only (no record oversampling), so it does
    # not require --record-oversample.
    frame_rba = (args.frame_rba == "on") or (args.adaptive_sampling in {"rba", "vrba"})
    if frame_rba and not bool(args.per_frame_frame):
        parser.error("--frame-rba / --adaptive-sampling rba|vrba require --per-frame-frame")
    if frame_rba and float(args.late_frame_gain) > 0.0:
        parser.error(
            "frame RBA replaces the late_frame_gain ramp; set --late-frame-gain 0"
        )
    if not (0.0 <= float(args.rba_gamma) < 1.0):
        parser.error("--rba-gamma must lie in [0, 1)")
    if float(args.rba_eta) < 0.0:
        parser.error("--rba-eta must be nonnegative")
    if not 0.0 <= float(args.rba_phi) <= 1.0:
        parser.error("--rba-phi must lie in [0, 1]")

    # DDP: each rank drives one GPU, trains on a disjoint slice of every update's macro
    # batch, and averages gradients. Only the main rank evaluates the held-out triplet,
    # writes metrics/checkpoints, and prints -- the held-out eval is tiny (3 records) so
    # replicating it across ranks would waste GPUs. Launch with torchrun --nproc_per_node.
    ddp = distributed_context()
    is_main = bool(ddp["is_main"])

    root = Path(args.artifact_dir).resolve()
    if is_main:
        root.mkdir(parents=True, exist_ok=True)
    terminal_path = root / "terminal.json"
    if terminal_path.exists():
        if is_main:
            print(terminal_path.read_text().strip())
        distributed_cleanup(ddp)
        return 0

    device = torch.device(f"cuda:{int(ddp['local_rank'])}" if ddp["enabled"] else "cuda")
    # CRITICAL for DDP correctness: all ranks MUST initialize the fresh dense-decoder +
    # local_field tensors identically, else averaging gradients over divergent weights is
    # wrong. So use the SAME seed on every rank (front-end is loaded from the shared
    # warmstart checkpoint; the fresh parts come from this seed). Per-rank data diversity
    # comes from ddp_update_specs slicing the schedule by rank, NOT from seed differences.
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    import dataclasses
    base = build_base_config(int(args.width))
    if args.source_h5 is not None or args.normalization_json is not None:
        data_cfg = dataclasses.replace(
            base.data,
            source_h5=(
                base.data.source_h5
                if args.source_h5 is None
                else str(Path(args.source_h5).expanduser().resolve())
            ),
            normalization_json=(
                base.data.normalization_json
                if args.normalization_json is None
                else str(Path(args.normalization_json).expanduser().resolve())
            ),
        )
        base = dataclasses.replace(base, data=data_cfg)
    manifest = build_manifest(base.data.source_h5)
    source_split_counts = {
        split: sum(record.split == split for record in manifest.records)
        for split in ("train", "validation", "test_id", "ood_canonical")
    }
    source_family_counts = {
        f"{split}/{family}": sum(
            record.split == split and record.medium_type == family
            for record in manifest.records
        )
        for split in ("train", "validation", "test_id", "ood_canonical")
        for family in FAMILIES
    }
    source_invalidation_audit = None
    source_replacement_audit = None
    if args.allow_filtered_invalidated_source:
        report_path = Path(args.source_invalidation_report).expanduser().resolve()
        try:
            invalidation = json.loads(report_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            parser.error(f"invalid source invalidation report: {error}")
        retained_path = Path(
            str(invalidation.get("retained_source_path", ""))
        ).expanduser().resolve()
        retained_sha256 = str(invalidation.get("retained_source_sha256", ""))
        source_path = Path(base.data.source_h5).expanduser().resolve()
        if invalidation.get("status") != "INVALIDATED":
            parser.error("source invalidation report status must be INVALIDATED")
        if retained_path != source_path:
            parser.error(
                "source invalidation report retained_source_path does not match --source-h5"
            )
        observed_sha256 = sha256_file(source_path)
        if retained_sha256 != observed_sha256:
            parser.error("retained source SHA256 does not match invalidation report")
        validate_expected_counts(
            manifest,
            {"validation": base.data.expected_validation_records},
        )
        if source_split_counts["train"] <= 0:
            parser.error("filtered retained source contains no train records")
        source_invalidation_audit = {
            "report": str(report_path),
            "reason": str(invalidation.get("reason", "")),
            "invalid_sample_count": int(invalidation.get("invalid_sample_count", 0)),
            "retained_source_sha256": observed_sha256,
            "filtered_manifest_digest": manifest.digest,
            "split_counts": source_split_counts,
            "family_counts": source_family_counts,
        }
    else:
        validate_expected_counts(
            manifest,
            {
                "train": base.data.expected_train_records,
                "validation": base.data.expected_validation_records,
            },
        )
        if args.source_replacement_gate:
            try:
                with h5py.File(base.data.source_h5, "r", swmr=True) as source_handle:
                    source_manifest_sha256 = str(
                        source_handle.attrs.get("manifest_sha256", "")
                    )
                    source_sample_count = int(source_handle["sample_id"].shape[0])
                source_replacement_audit = audit_source_replacement_gate(
                    args.source_replacement_gate,
                    source_h5=base.data.source_h5,
                    manifest_digest=manifest.digest,
                    source_manifest_sha256=source_manifest_sha256,
                    sample_count=source_sample_count,
                )
            except ValueError as error:
                parser.error(str(error))

    if args.allow_parent_normalization_binding:
        with Path(base.data.normalization_json).open(encoding="utf8") as handle:
            normalization_payload = json.load(handle)
        normalizer = PhysicalNormalizer.from_dict(normalization_payload)
        normalization_binding = {
            "mode": (
                "parent_binding_on_replacement_source"
                if source_replacement_audit is not None
                else "parent_binding_on_filtered_source"
            ),
            "normalization_manifest_digest": normalizer.metadata.train_manifest_sha256,
            "filtered_source_manifest_digest": manifest.digest,
            "normalization_record_count": int(normalizer.metadata.record_count),
        }
    else:
        normalizer = load_normalizer(base, manifest.digest)
        normalization_binding = {
            "mode": "strict_manifest_binding",
            "normalization_manifest_digest": normalizer.metadata.train_manifest_sha256,
            "filtered_source_manifest_digest": manifest.digest,
            "normalization_record_count": int(normalizer.metadata.record_count),
        }

    variant = ProbeVariant(
        depth=int(args.dense_depth),
        use_local_phase=True,
        spectral_rank=int(args.dense_spectral_rank),
        modes=int(args.dense_modes),
        temporal_basis_rank=0,
        family_expert_rank=0,
        local_field=True,
        local_field_residual=False,
        local_field_helmholtz_synthesis=True,
        local_field_helmholtz_synthesis_frequencies=int(args.helmholtz_frequencies),
        local_field_helmholtz_synthesis_wkb_phase=not bool(args.helmholtz_no_wkb),
        local_field_helmholtz_synthesis_rank=int(args.helmholtz_rank),
        local_field_helmholtz_synthesis_late_rank=int(args.helmholtz_late_rank),
        local_field_helmholtz_synthesis_late_frequencies=int(args.helmholtz_late_frequencies),
        local_field_helmholtz_spectral_bypass=bool(args.helmholtz_spectral_bypass),
        local_field_helmholtz_spectral_bypass_per_branch=bool(args.helmholtz_spectral_bypass_per_branch),
        local_field_helmholtz_background_conditioning=bool(args.helmholtz_background_conditioning),
        local_field_helmholtz_background_sigma_cells=float(args.helmholtz_background_sigma_cells),
        local_field_helmholtz_background_global_propagator=bool(
            args.helmholtz_background_global_propagator
        ),
        local_field_helmholtz_background_propagation_modes=int(
            args.helmholtz_background_propagation_modes
        ),
        local_field_helmholtz_background_direct_frequency_head=bool(
            args.helmholtz_background_direct_frequency_head
        ),
        local_field_helmholtz_background_direct_frequencies=int(
            args.helmholtz_background_direct_frequencies
        ),
        local_field_helmholtz_background_direct_spectral_experts=int(
            args.helmholtz_background_direct_spectral_experts
        ),
    )
    model = _model(base, manifest, variant).to(device)

    if args.warmstart_helmholtz:
        # Continue-pretraining: load a full A+1 checkpoint (rank-8 synthesis + front-end +
        # decoder) by shape, leaving ONLY the fresh late-rank head new. Zero-init late mix
        # means the model reproduces the A+1 parent exactly at step 0.
        transfer_report = warmstart_full_helmholtz(model, Path(args.warmstart_helmholtz))
    else:
        transfer_report = transfer_front_end_permissive(model, Path(args.warmstart_frontend))
    if args.background_conditioner_only:
        for param in model.parameters():
            param.requires_grad_(False)
        conditioner = model.local_field.helmholtz_background_conditioner
        selected_parameters = (
            conditioner.output_projection.parameters()
            if args.background_conditioner_output_only
            else conditioner.parameters()
        )
        for param in selected_parameters:
            param.requires_grad_(True)
        if args.background_conditioner_with_late_head:
            for name, param in model.local_field.helmholtz_synthesis.named_parameters():
                if name.startswith("late_"):
                    param.requires_grad_(True)
        active_prefixes = ("local_field.helmholtz_background_conditioner",)
        residual_activation_report = None
    else:
        for param in model.parameters():
            param.requires_grad_(True)
        residual_activation_report = activate_residual_head(
            model.dense_decoder,
            {
                "activation_mode": str(args.residual_activation),
                "correction_scale": float(args.residual_correction_scale),
                "output_std": float(args.residual_output_std),
            },
            seed=seed,
        )
        # REPLACE-mode local field: MIONet coarse path is bypassed. Active trainable
        # prefixes match the capacity-ladder replace-mode set.
        active_prefixes = ("dense_decoder", "local_field", "medium_encoder", "source_encoder")
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    trainable_parameter_count = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    config = build_probe_config(
        dense_lr=float(args.dense_learning_rate),
        backbone_lr=float(args.backbone_learning_rate),
        temporal_lr=float(args.temporal_basis_learning_rate),
        seed=seed,
        travel_time_h5=str(args.travel_time_h5),
        per_frame_frame=bool(args.per_frame_frame),
        frame_energy_floor_fraction=float(args.frame_energy_floor_fraction),
        late_frame_gain=float(args.late_frame_gain),
        late_frame_start_fraction=float(args.late_frame_start_fraction),
        full_field_frame_weight=float(args.full_field_frame_weight),
        delta_loss_weight=float(args.delta_loss_weight),
        delta_reference=str(args.delta_reference),
        delta_reduction=str(args.delta_reduction),
        delta_energy_floor_fraction=float(args.delta_energy_floor_fraction),
        spatial_gradient_loss_weight=float(args.spatial_gradient_loss_weight),
        family_gradient_weights=family_gradient_weights,
    )
    config["optimizer"]["gradient_clip_prefix_limits"]["local_field"] = float(args.local_field_grad_clip)
    config["allow_travel_source_path_mismatch"] = bool(
        args.allow_travel_source_path_alias
    )

    optimizer_report = {
        "kind": "capacity_optimizer",
        "local_field_learning_rate": float(args.local_field_learning_rate),
    }
    if args.background_conditioner_output_learning_rate is not None:
        optimizer, optimizer_report = build_background_conditioner_stage2_optimizer(
            conditioner,
            output_learning_rate=float(
                args.background_conditioner_output_learning_rate
            ),
            core_learning_rate=float(args.background_conditioner_core_learning_rate),
            coupled_learning_rate=(
                float(args.background_conditioner_coupled_learning_rate)
                if args.background_conditioner_coupled_learning_rate is not None
                else None
            ),
            expert_learning_rate=(
                float(args.background_conditioner_expert_learning_rate)
                if args.background_conditioner_expert_learning_rate is not None
                else None
            ),
            expert_router_learning_rate=(
                float(args.background_conditioner_expert_router_learning_rate)
                if args.background_conditioner_expert_router_learning_rate is not None
                else None
            ),
            adam_epsilon=float(args.background_conditioner_adam_epsilon),
            late_head=(
                model.local_field.helmholtz_synthesis
                if args.background_conditioner_with_late_head
                else None
            ),
            late_learning_rate=(
                float(args.helmholtz_late_learning_rate)
                if args.background_conditioner_with_late_head
                else None
            ),
        )
    else:
        optimizer = build_capacity_optimizer(
            model,
            dense_lr=float(args.dense_learning_rate),
            backbone_lr=float(args.backbone_learning_rate),
            local_field_lr=float(args.local_field_learning_rate),
            optimizer_name=str(args.optimizer),
        )

    # Fixed validation triplet. When validation is among --training-splits this panel is
    # explicitly in-sample and is not reported as a generalization gate.
    evaluation_split = str(args.evaluation_split)
    heldout_indices = select_one_index_per_family_after_skip(
        manifest.records,
        split=evaluation_split,
        skip_per_family=int(args.evaluation_skip_records_per_family),
    )
    heldout_records = tuple(
        r for r in manifest.records if r.split == evaluation_split
    )
    heldout_sample_ids = tuple(heldout_records[i].sample_id for i in heldout_indices)
    if args.exclude_evaluation_records_from_training and evaluation_split not in training_splits:
        parser.error(
            "--exclude-evaluation-records-from-training requires the evaluation split in --training-splits"
        )
    evaluation_excluded_from_training = bool(
        args.exclude_evaluation_records_from_training
        and evaluation_split in training_splits
    )
    evaluation_is_generalization = (
        evaluation_split not in training_splits or evaluation_excluded_from_training
    )
    validation_frame_count = int(args.validation_frames)
    if validation_frame_count > len(manifest.time_s):
        raise ValueError(
            f"validation frame count {validation_frame_count} exceeds stored axis "
            f"of {len(manifest.time_s)}"
        )
    validation_time_policy = (
        "all_saved"
        if validation_frame_count == len(manifest.time_s)
        else "validation_fixed"
    )

    # Direction A+1: smoothed-velocity background P_bg coupling. The provider must cover
    # both the reduced training pool (retargeted to encode(scat) during training) and the
    # held-out triplet (P_bg added back when scoring the full field).
    background_provider = None
    if args.background_cache:
        from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
        background_paths = (
            str(args.background_cache),
            *(str(value) for value in args.background_cache_shard),
        )
        background_provider = BackgroundFieldProvider(background_paths)
        if not background_provider.covers(heldout_sample_ids):
            raise ValueError(
                "background cache does not cover the held-out triplet: "
                f"{heldout_sample_ids}"
            )

    training_records = [
        record for record in manifest.records if str(record.split) in set(training_splits)
    ]
    if not training_records:
        raise ValueError(f"training split view is empty: {training_splits}")

    # Optional family-balanced reduced training pool. With records_per_family=0 the
    # complete selected split view is used (3203 records for all four non-anomaly splits).
    training_exclusion_sample_ids = (
        tuple(heldout_sample_ids) if evaluation_excluded_from_training else ()
    )
    train_pool = None
    if family_record_limits is not None or int(args.records_per_family) > 0:
        pool_limits: int | Mapping[str, int] = (
            family_record_limits
            if family_record_limits is not None
            else int(args.records_per_family)
        )
        train_pool = select_family_balanced_pool(
            manifest.records,
            splits=training_splits,
            per_family=pool_limits,
            excluded_sample_ids=training_exclusion_sample_ids,
        )
        train_record_count = len(train_pool)
        pool_records = [training_records[index] for index in train_pool]
    else:
        if training_exclusion_sample_ids:
            excluded = set(training_exclusion_sample_ids)
            train_pool = tuple(
                index
                for index, record in enumerate(training_records)
                if str(record.sample_id) not in excluded
            )
            train_record_count = len(train_pool)
            pool_records = [training_records[index] for index in train_pool]
        else:
            train_record_count = len(training_records)
            pool_records = training_records

    pool_sample_ids = tuple(record.sample_id for record in pool_records)
    if background_provider is not None and not background_provider.covers(pool_sample_ids):
        available = set(background_provider.sample_ids)
        missing = [sample_id for sample_id in pool_sample_ids if sample_id not in available]
        raise ValueError(
            f"background cache set misses {len(missing)} training-pool records, "
            f"first: {missing[:3]}"
        )
    if args.fourier_uniform_training_times:
        training_time_pool = fixed_teacher_time_indices(
            stored_time_count=len(manifest.time_s),
            count=int(args.training_frames),
        )
        if background_provider is not None and not set(training_time_pool).issubset(
            background_provider.time_indices
        ):
            raise ValueError(
                "background cache does not cover the Fourier-uniform training grid"
            )
    else:
        training_time_pool = (
            tuple(background_provider.time_indices)
            if args.training_time_pool_from_background and background_provider is not None
            else None
        )
    if training_time_pool is not None and int(args.training_frames) > len(training_time_pool):
        raise ValueError(
            f"--training-frames {args.training_frames} exceeds background common time pool "
            f"of {len(training_time_pool)}"
        )
    fourier_design = None
    if args.helmholtz_background_direct_frequency_head:
        assert training_time_pool is not None
        fourier_design = fourier_training_design_report(
            manifest.time_s,
            training_time_pool,
            frequencies=int(args.helmholtz_background_direct_frequencies),
        )
        if (
            int(fourier_design["rank"])
            != int(fourier_design["effective_coefficient_count"])
            or not math.isfinite(float(fourier_design["condition_number"]))
            or float(fourier_design["condition_number"]) > 100.0
        ):
            raise ValueError(
                "direct scattering temporal design is rank deficient or ill-conditioned: "
                f"{fourier_design}"
            )
    epochs = 1 if args.smoke_updates else int(args.epochs)
    # DDP data parallelism: macros_per_update macro-batches per optimizer update are split
    # across ranks (ddp_update_specs takes chunk[rank::world_size]), so macros_per_update
    # must be divisible by world_size. Every rank builds the SAME global schedule (shared
    # seed) so the per-rank slices tile the full macro batch without overlap or gaps.
    world_size = int(ddp["world_size"])
    macros_per_update = int(args.macros_per_update)
    if ddp["enabled"] and macros_per_update % world_size:
        raise ValueError(
            f"--macros-per-update ({macros_per_update}) must be divisible by the DDP "
            f"world size ({world_size}); pass e.g. --macros-per-update {world_size}"
        )
    schedule = _train_schedule(
        train_record_count,
        epochs=epochs,
        macro_records=int(args.macro_records),
        macros_per_update=macros_per_update,
        seed=seed,
    )
    if train_pool is not None:
        schedule = _remap_schedule_to_pool(schedule, train_pool)
    if args.smoke_updates:
        schedule = schedule[: int(args.smoke_updates) * max(macros_per_update, 1)]
    # Slice this rank's disjoint share of every update's macro batch.
    if ddp["enabled"]:
        schedule = ddp_update_specs(
            schedule,
            macros_per_update=macros_per_update,
            rank=int(ddp["rank"]),
            world_size=world_size,
        )
    train_ds = _train_dataset(
        config,
        base,
        manifest,
        schedule,
        frames_per_record=int(args.training_frames),
        training_splits=training_splits,
        time_index_pool=training_time_pool,
    )
    workers = 0 if args.smoke_updates else int(args.workers)
    train_data = make_pilot_loader(
        train_ds, workers=workers, prefetch_factor=int(args.prefetch_factor), pin_memory=True,
    )

    # ---- vRBA record-level residual-adaptive sampling (Phase 2) -------------------
    # `adaptive` opts into rebuilding the schedule from a per-record residual EMA every
    # `recompute_epochs` epochs; the run is then split into epoch segments, each with a
    # fresh loader whose `record_weights` oversample the current high-error records.
    # `adaptive=False` keeps the single-shot uniform loader above bit-for-bit.
    adaptive = args.adaptive_sampling in {"rad", "vrba"}
    adaptive_time = bool(args.time_adaptive_sampling)
    segmented_sampling = adaptive or adaptive_time
    recompute_epochs = int(args.rad_recompute_epochs)
    record_oversample = float(args.record_oversample)
    # Stable ordering: schedule record-index i (pre-remap) -> this sample_id. Weights and
    # the DDP all-reduce buffers are indexed by this order so every rank agrees.
    pool_ids = [record.sample_id for record in pool_records]
    pool_families = [record.medium_type for record in pool_records]
    if len(pool_ids) != train_record_count:
        raise ValueError(
            f"pool id ordering ({len(pool_ids)}) does not match train record count "
            f"({train_record_count})"
        )
    residual_ema: dict[str, float] = {}

    def _build_segment_loader(
        epochs_this: int,
        epoch_offset: int,
        weights,
        time_probabilities,
    ):
        seg = _train_schedule(
            train_record_count,
            epochs=int(epochs_this),
            macro_records=int(args.macro_records),
            macros_per_update=macros_per_update,
            seed=seed,
            epoch_offset=int(epoch_offset),
            record_weights=weights,
            record_oversample=record_oversample if adaptive else 1.0,
        )
        if train_pool is not None:
            seg = _remap_schedule_to_pool(seg, train_pool)
        if args.smoke_updates:
            seg = seg[: int(args.smoke_updates) * max(macros_per_update, 1)]
        if ddp["enabled"]:
            seg = ddp_update_specs(
                seg,
                macros_per_update=macros_per_update,
                rank=int(ddp["rank"]),
                world_size=world_size,
            )
        seg_ds = _train_dataset(
            config,
            base,
            manifest,
            seg,
            frames_per_record=int(args.training_frames),
            training_splits=training_splits,
            time_index_pool=training_time_pool,
            time_index_probabilities=time_probabilities,
        )
        seg_loader = make_pilot_loader(
            seg_ds, workers=workers, prefetch_factor=int(args.prefetch_factor), pin_memory=True,
        )
        return seg, seg_loader

    if segmented_sampling:
        # First segment samples uniformly (no residual signal yet) but is already
        # oversampled so the padded schedule length matches every later reweighted
        # segment, keeping updates-per-epoch constant across the whole run.
        uniform_weights = None
        if adaptive:
            uniform_weights = np.asarray(
                [family_sampling_weights[family] for family in pool_families],
                dtype=np.float64,
            )
            uniform_weights /= uniform_weights.sum()
        schedule, train_data = _build_segment_loader(
            min(recompute_epochs, epochs), 0, uniform_weights, None
        )
        updates_per_epoch = len(schedule) // max(min(recompute_epochs, epochs), 1)
        total_updates = updates_per_epoch * epochs
    else:
        updates_per_epoch = len(schedule) // max(epochs, 1)
        total_updates = len(schedule)

    identity = {
        "schema": "helmholtz_g3_heldout_v1",
        "width": int(args.width),
        "dense_depth": int(args.dense_depth),
        "dense_spectral_rank": int(args.dense_spectral_rank),
        "dense_modes": int(args.dense_modes),
        "helmholtz_frequencies": int(args.helmholtz_frequencies),
        "helmholtz_rank": int(args.helmholtz_rank),
        "helmholtz_late_rank": int(args.helmholtz_late_rank),
        "helmholtz_late_frequencies": int(args.helmholtz_late_frequencies),
        "helmholtz_spectral_bypass": bool(args.helmholtz_spectral_bypass),
        "helmholtz_spectral_bypass_per_branch": bool(args.helmholtz_spectral_bypass_per_branch),
        "helmholtz_background_conditioning": bool(args.helmholtz_background_conditioning),
        "helmholtz_background_sigma_cells": float(args.helmholtz_background_sigma_cells),
        "helmholtz_background_global_propagator": bool(
            args.helmholtz_background_global_propagator
        ),
        "helmholtz_background_propagation_modes": int(
            args.helmholtz_background_propagation_modes
        ),
        "helmholtz_background_direct_frequency_head": bool(
            args.helmholtz_background_direct_frequency_head
        ),
        "helmholtz_background_direct_frequencies": int(
            args.helmholtz_background_direct_frequencies
        ),
        "helmholtz_background_direct_spectral_experts": int(
            args.helmholtz_background_direct_spectral_experts
        ),
        "helmholtz_background_direct_fourier_design": fourier_design,
        "background_conditioner_only": bool(args.background_conditioner_only),
        "background_conditioner_output_only": bool(
            args.background_conditioner_output_only
        ),
        "background_conditioner_with_late_head": bool(
            args.background_conditioner_with_late_head
        ),
        "residual_activation": residual_activation_report,
        "helmholtz_wkb_phase": not bool(args.helmholtz_no_wkb),
        "local_field_learning_rate": float(args.local_field_learning_rate),
        "dense_learning_rate": float(args.dense_learning_rate),
        "backbone_learning_rate": float(args.backbone_learning_rate),
        "optimizer": str(args.optimizer),
        "optimizer_report": optimizer_report,
        "checkpoint_selection_metric": str(args.checkpoint_selection_metric),
        "warmstart_frontend": None if not args.warmstart_frontend else str(Path(args.warmstart_frontend).resolve()),
        "warmstart_helmholtz": None if not args.warmstart_helmholtz else str(Path(args.warmstart_helmholtz).resolve()),
        "warmstart_selection": warmstart_selection,
        "best_warmstart_required": bool(args.require_best_warmstart),
        "front_end_transfer": transfer_report,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "source_h5": str(Path(base.data.source_h5).resolve()),
        "manifest_digest": manifest.digest,
        "source_split_counts": source_split_counts,
        "source_family_counts": source_family_counts,
        "source_invalidation_audit": source_invalidation_audit,
        "source_replacement_audit": source_replacement_audit,
        "normalization_binding": normalization_binding,
        "allow_travel_source_path_alias": bool(
            args.allow_travel_source_path_alias
        ),
        "train_record_count": train_record_count,
        "training_pool_family_counts": {
            family: sum(value == family for value in pool_families)
            for family in FAMILIES
        },
        "requested_family_record_limits": family_record_limits,
        "training_splits": list(training_splits),
        "training_split_counts": {
            split: sum(record.split == split for record in training_records)
            for split in training_splits
        },
        "excluded_medium_types": list(manifest.excluded_medium_types),
        "anomaly_excluded": "anomaly" in manifest.excluded_medium_types,
        "evaluation_is_generalization": evaluation_is_generalization,
        "evaluation_split": evaluation_split,
        "evaluation_skip_records_per_family": int(
            args.evaluation_skip_records_per_family
        ),
        "evaluation_records_excluded_from_training": evaluation_excluded_from_training,
        "training_exclusion_sample_ids": list(training_exclusion_sample_ids),
        "heldout_indices": list(heldout_indices),
        "heldout_sample_ids": list(heldout_sample_ids),
        "families": FAMILIES,
        "epochs": epochs,
        "macro_records": int(args.macro_records),
        "macros_per_update": int(args.macros_per_update),
        "microbatch_records": int(args.microbatch_records),
        "training_frames_per_record": int(args.training_frames),
        "validation_frames_per_record": int(args.validation_frames),
        "validation_time_policy": validation_time_policy,
        "per_frame_frame": bool(args.per_frame_frame),
        "frame_energy_floor_fraction": float(args.frame_energy_floor_fraction),
        "full_field_frame_weight": float(args.full_field_frame_weight),
        "delta_loss_weight": float(args.delta_loss_weight),
        "delta_reference": str(args.delta_reference),
        "delta_reduction": str(args.delta_reduction),
        "delta_energy_floor_fraction": float(args.delta_energy_floor_fraction),
        "spatial_gradient_loss_weight": float(args.spatial_gradient_loss_weight),
        "late_frame_gain": float(args.late_frame_gain),
        "late_frame_start_fraction": float(args.late_frame_start_fraction),
        "family_gradient_weights": family_gradient_weights,
        "family_sampling_weights": family_sampling_weights,
        "evaluate_every": int(args.evaluate_every),
        "seed": seed,
        "smoke_updates": int(args.smoke_updates),
        "schedule_updates": len(schedule),
        "background_cache": None if not args.background_cache else str(Path(args.background_cache).resolve()),
        "background_cache_shards": [
            str(Path(value).resolve()) for value in args.background_cache_shard
        ],
        "training_time_policy": (
            "fourier_uniform"
            if args.fourier_uniform_training_times
            else (
                "appearance16"
                if training_time_pool is None
                else "numerical_teacher_pool"
            )
        ),
        "training_time_pool_count": (
            None if training_time_pool is None else len(training_time_pool)
        ),
        "training_time_pool": (
            None if training_time_pool is None else list(training_time_pool)
        ),
        "time_adaptive_sampling": bool(args.time_adaptive_sampling),
        "time_sampling_uniform_fraction": float(args.time_sampling_uniform_fraction),
        "time_residual_ema_momentum": float(args.time_residual_ema_momentum),
        "time_sampling_potential": str(args.rba_potential),
        "records_per_family": int(args.records_per_family),
        "adaptive_sampling": str(args.adaptive_sampling),
        "rad_recompute_epochs": int(args.rad_recompute_epochs),
        "record_oversample": float(args.record_oversample),
        "rad_potential": str(args.rad_potential),
        "rad_uniform_fraction": float(args.rad_uniform_fraction),
        "rad_ema_momentum": float(args.rad_ema_momentum),
        "frame_rba": bool(frame_rba),
        "rba_gamma": float(args.rba_gamma),
        "rba_eta": float(args.rba_eta),
        "rba_phi": float(args.rba_phi),
        "rba_potential": str(args.rba_potential),
        "rba_lambda_bound": float(args.rba_eta) / (1.0 - float(args.rba_gamma)),
        "updates_per_epoch": int(updates_per_epoch),
        "total_updates": int(total_updates),
    }
    identity["run_digest"] = _digest(identity)
    if is_main:
        _atomic_json(identity, root / "run_identity.json")

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    # Baseline fixed-panel eval at update 0. It is in-sample when validation participates
    # in optimization; identity/terminal make that distinction explicit. Main rank only;
    # weights are identical across ranks (shared seed + shared warmstart), so the main
    # rank's baseline is representative. best_score/best_checkpoint live only where they
    # are used (the main-rank eval branch in the loop and the final restore).
    selection_metric = str(args.checkpoint_selection_metric)
    best_score = float("inf")
    best_aggregate_score = float("inf")
    best_checkpoint = None
    if is_main:
        baseline = _evaluate_triplet(
            model, base, manifest, normalizer, device, config, heldout_indices,
            split=evaluation_split, time_policy=validation_time_policy,
            frames_per_record=validation_frame_count,
            background_provider=background_provider,
        )
        _append_jsonl(root / "metrics.jsonl", {"event": "baseline", "update": 0, "metrics": baseline})
        best_score = float(baseline[selection_metric])
        best_aggregate_score = float(baseline["aggregate_relative_l2"])
        best_checkpoint = str(save_overfit_checkpoint(
            root, model=model, update=0, manifest_digest=manifest.digest,
            config_digest=identity["run_digest"],
            aggregate_relative_l2=best_aggregate_score,
            extra_metrics={selection_metric: best_score},
        ))
        print(json.dumps({
            "update": 0,
            "agg": best_aggregate_score,
            "selection_metric": selection_metric,
            "selection_score": best_score,
            "family": baseline.get("family_relative_l2"),
        }, sort_keys=True), flush=True)
    else:
        baseline = None
    distributed_barrier(ddp)

    evaluate_every = 1 if args.smoke_updates else int(args.evaluate_every)
    last_update = 0

    # Frame-level RBA state: a persistent bounded-EMA weight per selected frame slot
    # (T = training frames). appearance16 returns time-sorted frames, so slot i is a
    # stable early->late position across updates. frame_lambda is cold-started at zero
    # (weight override = 1 + lambda, so the first step is unweighted). None when off.
    training_frames = int(args.training_frames)
    frame_lambda = np.zeros(training_frames, dtype=np.float64) if frame_rba else None
    frame_step = 0

    def _run_segment(loader, update0: int, sink, time_sink):
        """Run every update in one loader segment. Returns the last update index."""
        nonlocal best_score, best_aggregate_score, best_checkpoint, last_update, frame_lambda, frame_step
        update = update0
        for batch in loader:
            update += 1
            model.train()
            if background_provider is not None:
                batch = _retarget_batch_to_scattering(batch, background_provider)
            frame_sink: dict[int, tuple[float, float]] | None = {} if frame_rba else None
            weight_override = (
                None
                if frame_lambda is None
                else torch.as_tensor(1.0 + frame_lambda, dtype=torch.float32)
            )
            components = _train_update(
                model, optimizer, batch, normalizer, device, config,
                microbatch_records=int(args.microbatch_records),
                background_provider=background_provider,
                record_residual_sink=sink,
                frame_time_weights_override=weight_override,
                frame_residual_sink=frame_sink,
                time_index_residual_sink=time_sink,
            )
            logged_components = _distributed_average_components(
                components, ddp, device
            )
            # DDP: average gradients across ranks BEFORE clipping/step, matching the
            # full_support ordering (backward -> all_reduce -> clip -> step) so every rank
            # applies the identical averaged, clipped update and weights stay in lockstep.
            distributed_average_gradients(model, ddp)
            gradients = _gradient_report(model, active_prefixes)
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
            # Frame-level RBA EMA update: turn this step's per-frame-slot residual into
            # a relative-L2 per slot (DDP-reduced over the fixed slot order), then apply
            # the bounded vRBA update. Slots absent this step (den==0) keep their weight.
            if frame_lambda is not None:
                num = np.zeros(training_frames, dtype=np.float64)
                den = np.zeros(training_frames, dtype=np.float64)
                for slot, (fn, fd) in frame_sink.items():
                    if 0 <= int(slot) < training_frames:
                        num[int(slot)] = fn
                        den[int(slot)] = fd
                if ddp["enabled"]:
                    buffer = torch.from_numpy(np.concatenate([num, den])).to(device)
                    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
                    reduced = buffer.cpu().numpy()
                    num, den = reduced[:training_frames], reduced[training_frames:]
                seen = den > 0.0
                frame_residual = np.zeros(training_frames, dtype=np.float64)
                frame_residual[seen] = np.sqrt(num[seen] / den[seen])
                updated = vrba_frame_weights(
                    torch.from_numpy(frame_residual),
                    torch.from_numpy(frame_lambda),
                    gamma=float(args.rba_gamma),
                    eta=float(args.rba_eta),
                    phi=float(args.rba_phi),
                    potential=str(args.rba_potential),
                    iteration=frame_step,
                ).cpu().numpy()
                # Only advance slots that carried a residual this step; frozen slots keep
                # their prior weight (no decay toward zero on frames not sampled).
                frame_lambda = np.where(seen, updated, frame_lambda)
                frame_step += 1
            if is_main:
                _append_jsonl(root / "updates.jsonl", {
                    "event": "optimizer_update", "update": update,
                    "loss_components": logged_components,
                    "gradient_norm_before_clip": gradient_norm, "gpu": _gpu_snapshot(),
                    "elapsed_seconds": time.monotonic() - started,
                })
            if update % evaluate_every and update != total_updates:
                continue
            # Only the main rank evaluates the (tiny, 3-record) held-out triplet and writes
            # artifacts; other ranks wait at the barrier so weights do not drift before the
            # next update. Evaluation uses the local weights, identical across ranks post-step.
            if is_main:
                metrics = _evaluate_triplet(
                    model, base, manifest, normalizer, device, config, heldout_indices,
                    split=evaluation_split, time_policy=validation_time_policy,
                    frames_per_record=validation_frame_count,
                    background_provider=background_provider,
                )
                score = float(metrics[selection_metric])
                checkpoint_path = save_overfit_checkpoint(
                    root, model=model, update=update, manifest_digest=manifest.digest,
                    config_digest=identity["run_digest"],
                    aggregate_relative_l2=float(metrics["aggregate_relative_l2"]),
                    extra_metrics={selection_metric: score},
                )
                if score <= best_score:
                    best_score = score
                    best_aggregate_score = float(metrics["aggregate_relative_l2"])
                    best_checkpoint = str(checkpoint_path)
                _append_jsonl(root / "metrics.jsonl", {
                    "event": "evaluation", "update": update, "metrics": metrics,
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                    "elapsed_seconds": time.monotonic() - started, "checkpoint": str(checkpoint_path),
                    "frame_lambda": (None if frame_lambda is None else [round(float(v), 6) for v in frame_lambda]),
                })
                print(json.dumps({
                    "update": update,
                    "agg": float(metrics["aggregate_relative_l2"]),
                    "selection_metric": selection_metric,
                    "selection_score": score,
                    "family": metrics.get("family_relative_l2"),
                }, sort_keys=True), flush=True)
            distributed_barrier(ddp)
        return update

    def _reduce_and_update_ema(sink):
        """All-reduce this cycle's per-record squared num/den over the fixed pool order,
        fold into the residual EMA, and return residual-adaptive record weights."""
        num = np.zeros(train_record_count, dtype=np.float64)
        den = np.zeros(train_record_count, dtype=np.float64)
        for position, sid in enumerate(pool_ids):
            entry = sink.get(sid)
            if entry is not None:
                num[position] = entry[0]
                den[position] = entry[1]
        if ddp["enabled"]:
            buffer = torch.from_numpy(np.concatenate([num, den])).to(device)
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            reduced = buffer.cpu().numpy()
            num, den = reduced[:train_record_count], reduced[train_record_count:]
        # Per-record relative-L2 over every frame it appeared in this cycle. Records never
        # drawn this cycle (den==0) keep their previous EMA (no false-zero residual).
        alpha = float(args.rad_ema_momentum)
        for position, sid in enumerate(pool_ids):
            if den[position] <= 0.0:
                continue
            rel = math.sqrt(num[position] / den[position])
            if not math.isfinite(rel):
                continue
            prev = residual_ema.get(sid)
            residual_ema[sid] = rel if prev is None else (1.0 - alpha) * prev + alpha * rel
        default = float(np.mean([residual_ema[s] for s in residual_ema]) or 1.0) if residual_ema else 1.0
        scores = np.array([residual_ema.get(sid, default) for sid in pool_ids], dtype=np.float64)
        weights = vrba_record_pdf(
            torch.from_numpy(scores),
            potential=str(args.rad_potential),
            uniform_fraction=float(args.rad_uniform_fraction),
        )
        adaptive_weights = np.asarray(weights, dtype=np.float64)
        family_prior = np.asarray(
            [family_sampling_weights[family] for family in pool_families],
            dtype=np.float64,
        )
        combined = adaptive_weights * family_prior
        combined /= combined.sum()
        return combined

    time_residual_ema: dict[int, float] = {}

    def _reduce_and_update_time_pdf(sink):
        if training_time_pool is None:
            raise RuntimeError("adaptive time sampling requires a fixed time pool")
        count = len(training_time_pool)
        num = np.zeros(count, dtype=np.float64)
        den = np.zeros(count, dtype=np.float64)
        for position, time_index in enumerate(training_time_pool):
            entry = sink.get(int(time_index))
            if entry is not None:
                num[position], den[position] = entry
        if ddp["enabled"]:
            buffer = torch.from_numpy(np.concatenate([num, den])).to(device)
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            reduced = buffer.cpu().numpy()
            num, den = reduced[:count], reduced[count:]
        alpha = float(args.time_residual_ema_momentum)
        for position, time_index in enumerate(training_time_pool):
            if den[position] <= 0.0:
                continue
            residual = math.sqrt(num[position] / den[position])
            if not math.isfinite(residual):
                continue
            previous = time_residual_ema.get(int(time_index))
            time_residual_ema[int(time_index)] = (
                residual
                if previous is None
                else (1.0 - alpha) * previous + alpha * residual
            )
        default = (
            float(np.mean(tuple(time_residual_ema.values())))
            if time_residual_ema
            else 1.0
        )
        scores = np.asarray(
            [time_residual_ema.get(int(index), default) for index in training_time_pool],
            dtype=np.float64,
        )
        return vrba_record_pdf(
            torch.from_numpy(scores),
            potential=str(args.rba_potential),
            uniform_fraction=float(args.time_sampling_uniform_fraction),
            iteration=max(frame_step, 0),
        )

    if not segmented_sampling:
        _run_segment(train_data, 0, None, None)
    else:
        completed_epochs = 0
        current_loader = train_data
        while completed_epochs < epochs:
            segment_epochs = min(recompute_epochs, epochs - completed_epochs)
            sink: dict[str, tuple[float, float]] = {}
            time_sink: dict[int, tuple[float, float]] = {}
            update0 = completed_epochs * updates_per_epoch
            _run_segment(
                current_loader,
                update0,
                sink if adaptive else None,
                time_sink if adaptive_time else None,
            )
            completed_epochs += segment_epochs
            if completed_epochs >= epochs:
                break
            weights = _reduce_and_update_ema(sink) if adaptive else None
            time_probabilities = (
                _reduce_and_update_time_pdf(time_sink) if adaptive_time else None
            )
            if is_main:
                _append_jsonl(root / "metrics.jsonl", {
                    "event": "rad_recompute", "update": last_update,
                    "completed_epochs": completed_epochs,
                    "record_weight_max": None if weights is None else float(weights.max()),
                    "record_weight_min": None if weights is None else float(weights.min()),
                    "residual_ema_records": len(residual_ema),
                    "time_probability_max": (
                        None
                        if time_probabilities is None
                        else float(time_probabilities.max())
                    ),
                    "time_probability_min": (
                        None
                        if time_probabilities is None
                        else float(time_probabilities.min())
                    ),
                    "time_residual_ema_indices": len(time_residual_ema),
                })
            next_epochs = min(recompute_epochs, epochs - completed_epochs)
            _, current_loader = _build_segment_loader(
                next_epochs, completed_epochs, weights, time_probabilities
            )


    # Final decisive held-out all_saved (401-frame, query-invariant) eval on best
    # checkpoint. Main rank only (it holds best_checkpoint and does all artifact I/O);
    # other ranks skip straight to cleanup.
    if not is_main:
        distributed_cleanup(ddp)
        return 0

    restore_best_overfit_checkpoint(
        best_checkpoint, model=model, manifest_digest=manifest.digest,
        config_digest=identity["run_digest"], map_location=device,
    )
    full_metrics = _evaluate_triplet(
        model, base, manifest, normalizer, device, config, heldout_indices,
        split=evaluation_split, time_policy="all_saved", frames_per_record=len(manifest.time_s),
        background_provider=background_provider,
    )
    _append_jsonl(root / "metrics.jsonl", {
        "event": "all_saved_heldout_evaluation", "update": last_update,
        "checkpoint": best_checkpoint, "metrics": full_metrics,
    })
    terminal_payload = {
        "status": "complete",
        "schema": "helmholtz_g3_heldout_v1",
        "updates_completed": last_update,
        "baseline_heldout_relative_l2": None if baseline is None else float(baseline["aggregate_relative_l2"]),
        "best_fixed_heldout_relative_l2": best_aggregate_score,
        "checkpoint_selection_metric": selection_metric,
        "best_checkpoint_selection_score": best_score,
        "best_checkpoint": best_checkpoint,
        "evaluation_is_generalization": evaluation_is_generalization,
        "training_splits": list(training_splits),
        "train_record_count": train_record_count,
        "training_pool_family_counts": identity["training_pool_family_counts"],
        "source_h5": identity["source_h5"],
        "source_invalidation_audit": source_invalidation_audit,
        "source_replacement_audit": source_replacement_audit,
        "normalization_binding": normalization_binding,
        "allow_travel_source_path_alias": bool(
            args.allow_travel_source_path_alias
        ),
        "anomaly_excluded": "anomaly" in manifest.excluded_medium_types,
        "heldout_all_saved_metrics": full_metrics,
        "heldout_all_saved_aggregate_relative_l2": float(full_metrics["aggregate_relative_l2"]),
        "g2_memorization_all_saved_reference": 0.206,
        "lpf_g2_reference": 0.29,
        "plain_helmholtz_g3_reference": 0.639,
        "ddp_world_size": int(ddp["world_size"]),
        "background_cache": None if not args.background_cache else str(Path(args.background_cache).resolve()),
        "background_cache_shards": [
            str(Path(value).resolve()) for value in args.background_cache_shard
        ],
        "training_time_pool_count": (
            None if training_time_pool is None else len(training_time_pool)
        ),
        "time_adaptive_sampling": bool(args.time_adaptive_sampling),
        "time_sampling_uniform_fraction": float(args.time_sampling_uniform_fraction),
        "time_residual_ema_indices": len(time_residual_ema),
        "records_per_family": int(args.records_per_family),
        "adaptive_sampling": str(args.adaptive_sampling),
        "rad_recompute_epochs": int(args.rad_recompute_epochs),
        "record_oversample": float(args.record_oversample),
        "frame_rba": bool(frame_rba),
        "rba_potential": str(args.rba_potential),
        "frame_lambda_final": (None if frame_lambda is None else [round(float(v), 6) for v in frame_lambda]),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    _atomic_json(terminal_payload, terminal_path)
    print(json.dumps(terminal_payload, sort_keys=True), flush=True)
    distributed_cleanup(ddp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
