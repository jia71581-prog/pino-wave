"""Evidence-bound selection and configuration of long saved-time continuations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from saved_time_phase_operator_v4.family_gradients import (
    inverse_norm_family_weights,
)


@dataclass(frozen=True)
class PilotSelection:
    name: str
    config_path: Path
    config: dict[str, object]
    checkpoint: Path
    checkpoint_identity: Path
    epoch: int
    score: float
    family_relative_l2: dict[str, float]
    metrics: dict[str, object]
    loss_components: dict[str, float]


@dataclass(frozen=True)
class PilotHeadSelection:
    """One identity-bound coarse or corrected fixed-panel pilot head."""

    name: str
    config_path: Path
    config: dict[str, object]
    checkpoint: Path
    checkpoint_identity: Path
    epoch: int
    correction_scale: float
    score: float
    family_relative_l2: dict[str, float]
    metrics: dict[str, object]
    loss_components: dict[str, float]


@dataclass(frozen=True)
class FullTimeParentSelection:
    """One identity-bound coarse or corrected full-time parent."""

    name: str
    config_path: Path
    checkpoint: Path
    checkpoint_identity: Path
    correction_scale: float
    score: float
    family_relative_l2: dict[str, float]


def _pilot_rows(config_path: Path) -> tuple[dict[str, object], ...]:
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or not config.get("artifact_dir"):
        return ()
    source_artifact = Path(str(config["artifact_dir"])).resolve()
    rows: list[dict[str, object]] = []
    for mode, allowed_scopes in (
        ("pilot", {"pilot_fixed_panel"}),
        ("run", {"fixed_panel"}),
    ):
        root = source_artifact / mode
        metrics_path = root / "metrics.jsonl"
        identity_path = root / "run_identity.json"
        if not metrics_path.is_file() or not identity_path.is_file():
            continue
        identity = json.loads(identity_path.read_text())
        if not isinstance(identity, Mapping) or not identity.get("run_digest"):
            continue
        evidence_config = config
        recorded_config = identity.get("config")
        if isinstance(recorded_config, Mapping):
            evidence_config = dict(recorded_config)
            if (
                Path(str(evidence_config.get("artifact_dir", ""))).resolve()
                != source_artifact
            ):
                continue
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metrics = row.get("metrics", {})
            checkpoint = Path(str(row.get("checkpoint", "")))
            if (
                row.get("event") != "epoch"
                or row.get("validation_scope") not in allowed_scopes
                or int(metrics.get("frame_count", 0)) != 48 * 32
                or not checkpoint.is_file()
            ):
                continue
            score = float(metrics.get("aggregate_relative_l2", float("nan")))
            family = metrics.get("family_relative_l2", {})
            if (
                not math.isfinite(score)
                or set(family) != {"uniform", "layered", "marmousi"}
                or not all(math.isfinite(float(value)) for value in family.values())
            ):
                continue
            rows.append(
                {
                    "config": evidence_config,
                    "identity_path": identity_path,
                    "row": row,
                }
            )
    return tuple(rows)


def select_best_pilot_candidate(
    config_paths: Sequence[str | Path],
) -> PilotSelection:
    """Choose the lowest comparable fixed-panel epoch with complete provenance."""

    selections: list[PilotSelection] = []
    for raw_path in config_paths:
        config_path = Path(raw_path).resolve()
        for value in _pilot_rows(config_path):
            config = dict(value["config"])
            row = dict(value["row"])
            metrics = dict(row["metrics"])
            selections.append(
                PilotSelection(
                    name=config_path.stem,
                    config_path=config_path,
                    config=config,
                    checkpoint=Path(str(row["checkpoint"])).resolve(),
                    checkpoint_identity=Path(value["identity_path"]).resolve(),
                    epoch=int(row["epoch"]),
                    score=float(metrics["aggregate_relative_l2"]),
                    family_relative_l2={
                        str(name): float(score)
                        for name, score in dict(metrics["family_relative_l2"]).items()
                    },
                    metrics=deepcopy(metrics),
                    loss_components={
                        str(name): float(score)
                        for name, score in dict(
                            row.get("last_update_loss_components", {})
                        ).items()
                    },
                )
            )
    if not selections:
        raise ValueError("no comparable pilot evidence was found")
    return min(selections, key=lambda item: (item.score, item.name, item.epoch))


def select_best_pilot_head(
    config_paths: Sequence[str | Path],
) -> PilotHeadSelection:
    """Choose the best comparable coarse or corrected fixed-panel pilot head."""

    selections: list[PilotHeadSelection] = []
    required_families = {"uniform", "layered", "marmousi"}
    for raw_path in config_paths:
        config_path = Path(raw_path).resolve()
        for value in _pilot_rows(config_path):
            config = dict(value["config"])
            row = dict(value["row"])
            corrected = row.get("metrics")
            if not isinstance(corrected, Mapping):
                continue
            raw_losses = row.get("last_update_loss_components")
            if not isinstance(raw_losses, Mapping):
                continue
            try:
                delta = float(raw_losses["delta"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(delta):
                continue
            for correction_scale, raw_metrics in (
                (0.0, corrected.get("coarse_metrics")),
                (1.0, corrected),
            ):
                if not isinstance(raw_metrics, Mapping):
                    continue
                family = raw_metrics.get("family_relative_l2")
                spectrum = raw_metrics.get("spectrum_relative_l2")
                if (
                    int(raw_metrics.get("frame_count", 0)) != 48 * 32
                    or not isinstance(family, Mapping)
                    or set(family) != required_families
                    or not isinstance(spectrum, Mapping)
                ):
                    continue
                try:
                    score = float(raw_metrics["aggregate_relative_l2"])
                    high = float(spectrum["high"])
                    families = {
                        str(name): float(family[name]) for name in required_families
                    }
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(
                    math.isfinite(item)
                    for item in (score, high, *families.values())
                ):
                    continue
                selections.append(
                    PilotHeadSelection(
                        name=config_path.stem,
                        config_path=config_path,
                        config=config,
                        checkpoint=Path(str(row["checkpoint"])).resolve(),
                        checkpoint_identity=Path(value["identity_path"]).resolve(),
                        epoch=int(row["epoch"]),
                        correction_scale=correction_scale,
                        score=score,
                        family_relative_l2=families,
                        metrics=deepcopy(dict(raw_metrics)),
                        loss_components={
                            str(name): float(loss)
                            for name, loss in raw_losses.items()
                        },
                    )
                )
    if not selections:
        raise ValueError("no comparable pilot head evidence was found")
    return min(
        selections,
        key=lambda item: (
            item.score,
            item.name,
            item.epoch,
            item.correction_scale,
        ),
    )


def select_best_full_time_parent(
    config_paths: Sequence[str | Path],
) -> FullTimeParentSelection:
    """Choose the best provenance-complete 48x401 long-run head."""

    candidates: list[tuple[FullTimeParentSelection, int]] = []
    families = {"uniform", "layered", "marmousi"}
    for raw_path in config_paths:
        config_path = Path(raw_path).resolve()
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, Mapping) or not config.get("artifact_dir"):
            continue
        run = Path(str(config["artifact_dir"])).resolve() / "run"
        metrics_path = run / "metrics.jsonl"
        identity_path = run / "run_identity.json"
        if not metrics_path.is_file() or not identity_path.is_file():
            continue
        identity = json.loads(identity_path.read_text())
        if not isinstance(identity, Mapping) or not identity.get("run_digest"):
            continue
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            checkpoint = Path(str(row.get("checkpoint", ""))).resolve()
            metrics = row.get("metrics", {})
            if (
                row.get("event") != "epoch"
                or row.get("validation_scope") != "fixed_full_time_panel"
                or not checkpoint.is_file()
                or not isinstance(metrics, Mapping)
            ):
                continue
            for scale, head in (
                (0.0, metrics.get("coarse_metrics")),
                (1.0, metrics),
            ):
                if (
                    not isinstance(head, Mapping)
                    or int(head.get("frame_count", 0)) != 48 * 401
                ):
                    continue
                score = float(head.get("aggregate_relative_l2", float("nan")))
                family = head.get("family_relative_l2", {})
                if (
                    not math.isfinite(score)
                    or not isinstance(family, Mapping)
                    or set(family) != families
                    or not all(math.isfinite(float(value)) for value in family.values())
                ):
                    continue
                candidates.append(
                    (
                        FullTimeParentSelection(
                            name=config_path.stem,
                            config_path=config_path,
                            checkpoint=checkpoint,
                            checkpoint_identity=identity_path.resolve(),
                            correction_scale=scale,
                            score=score,
                            family_relative_l2={
                                str(name): float(value)
                                for name, value in family.items()
                            },
                        ),
                        int(row["epoch"]),
                    )
                )
    if not candidates:
        raise ValueError("no comparable full-time parent evidence was found")
    return min(
        candidates,
        key=lambda item: (
            item[0].score,
            item[0].name,
            item[1],
            item[0].correction_scale,
        ),
    )[0]


def build_long_continuation_config(
    selection: PilotSelection,
    *,
    artifact_dir: str | Path,
    epochs: int,
    physical_microbatch_records: int | None = None,
) -> dict[str, object]:
    """Transfer a selected epoch while continuing stage and time coverage."""

    epoch_count = int(epochs)
    if epoch_count <= 0:
        raise ValueError("long continuation epochs must be positive")
    config = deepcopy(selection.config)
    config["parent_checkpoint"] = str(selection.checkpoint)
    config["parent_checkpoint_identity"] = str(selection.checkpoint_identity)
    config["artifact_dir"] = str(Path(artifact_dir).resolve())
    config["epochs"] = epoch_count
    if physical_microbatch_records is not None:
        physical_microbatch = int(physical_microbatch_records)
        macro_records = int(config.get("macro_records", 0))
        if physical_microbatch <= 0 or physical_microbatch > macro_records:
            raise ValueError(
                "long continuation physical microbatch must be in [1, macro_records]"
            )
        config["microbatch_records"] = physical_microbatch
    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = True
    transfer.pop("allow_new_coupled_2d_parameters", None)
    transfer.pop("allow_new_temporal_basis_parameters", None)
    transfer.pop("allow_new_family_expert_parameters", None)
    config["checkpoint_transfer"] = transfer

    recovery = dict(config.get("residual_recovery", {}))
    recovery["activation_mode"] = "preserve"
    recovery["stage_epoch_offset"] = int(recovery.get("stage_epoch_offset", 0)) + int(
        selection.epoch
    )
    config["residual_recovery"] = recovery

    if config.get("family_curriculum") is not None:
        config["family_curriculum"] = {
            "stages": progressive_family_curriculum_stages(epoch_count)
        }
        config.pop("schedule_epoch_offset", None)
        config["time_appearance_offset"] = int(
            config.get("time_appearance_offset", 0)
        ) + int(selection.epoch)
    else:
        config["schedule_epoch_offset"] = int(
            config.get("schedule_epoch_offset", 0)
        ) + int(selection.epoch)

    family_experts = config.get("family_experts")
    if isinstance(family_experts, Mapping):
        family_experts = dict(family_experts)
        for name, unfreeze_epoch in (
            ("dense_unfreeze_epoch", 3),
            ("shared_unfreeze_epoch", 8),
            ("geometry_unfreeze_epoch", 13),
            ("backbone_unfreeze_epoch", 18),
        ):
            family_experts.setdefault(name, unfreeze_epoch)
        family_experts["stage_epoch_offset"] = int(
            family_experts.get("stage_epoch_offset", 0)
        ) + int(selection.epoch)
        config["family_experts"] = family_experts

    optimizer = dict(config.get("optimizer", {}))
    if optimizer.get("gradient_clip_mode") != "prefix_limits":
        optimizer["gradient_clip_mode"] = "prefix"
    optimizer["schedule_total_epochs"] = int(
        optimizer.get("schedule_total_epochs", selection.config["epochs"])
    )
    optimizer["schedule_epoch_offset"] = int(
        optimizer.get("schedule_epoch_offset", 0)
    ) + int(selection.epoch)
    config["optimizer"] = optimizer
    loss = dict(config.get("loss", {}))
    loss["hard_causality"] = True
    loss["hard_causality_lead_cycles"] = 1.0
    config["loss"] = loss
    return config


def build_all_modes_pilot_config(
    selection: PilotSelection,
    *,
    artifact_dir: str | Path,
) -> dict[str, object]:
    """Fork an exact-parent-equivalent 101-mode pilot from fixed-panel evidence."""

    config = deepcopy(selection.config)
    config["parent_checkpoint"] = str(selection.checkpoint)
    config["parent_checkpoint_identity"] = str(selection.checkpoint_identity)
    config["artifact_dir"] = str(Path(artifact_dir).resolve())

    overrides = dict(config.get("variant_overrides", {}))
    overrides["modes"] = 101
    config["variant_overrides"] = overrides

    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = False
    transfer["allow_spectral_mode_expansion"] = True
    config["checkpoint_transfer"] = transfer

    recovery = dict(config.get("residual_recovery", {}))
    recovery["activation_mode"] = "preserve"
    # The newly introduced high modes have no optimizer history.  Warm the
    # expanded dense decoder in isolation before unfreezing its conditioned
    # parent, while schedule_epoch_offset below still preserves time coverage.
    recovery["stage_epoch_offset"] = 0
    recovery["decoder_only_epochs"] = 2
    config["residual_recovery"] = recovery

    if config.pop("family_curriculum", None) is not None:
        config.pop("schedule_epoch_offset", None)
        config["time_appearance_offset"] = int(
            config.get("time_appearance_offset", 0)
        ) + int(selection.epoch)
    else:
        config["schedule_epoch_offset"] = int(
            config.get("schedule_epoch_offset", 0)
        ) + int(selection.epoch)

    optimizer = dict(config.get("optimizer", {}))
    optimizer["dense_learning_rate"] = max(
        float(optimizer.get("dense_learning_rate", 0.0)), 1.0e-4
    )
    optimizer["schedule_epoch_offset"] = int(
        optimizer.get("schedule_epoch_offset", 0)
    ) + int(selection.epoch)
    config["optimizer"] = optimizer
    gate = dict(config.get("gate", {}))
    gate["pilot_epochs"] = 2
    config["gate"] = gate
    return config


def build_architecture_pilot_config(
    selection: PilotSelection,
    *,
    artifact_dir: str | Path,
) -> dict[str, object]:
    """Fork a two-epoch local-differential pilot from fixed-panel evidence."""

    config = deepcopy(selection.config)
    config["parent_checkpoint"] = str(selection.checkpoint)
    config["parent_checkpoint_identity"] = str(selection.checkpoint_identity)
    config["artifact_dir"] = str(Path(artifact_dir).resolve())

    overrides = dict(config.get("variant_overrides", {}))
    overrides["local_differential_residual"] = True
    config["variant_overrides"] = overrides

    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = False
    transfer["allow_new_local_differential_parameters"] = True
    config["checkpoint_transfer"] = transfer

    recovery = dict(config.get("residual_recovery", {}))
    recovery["activation_mode"] = "preserve"
    recovery["stage_epoch_offset"] = 0
    recovery["decoder_only_epochs"] = 2
    config["residual_recovery"] = recovery
    config["microbatch_records"] = min(
        int(config.get("microbatch_records", 3)), 3
    )

    if config.get("family_curriculum") is not None:
        config.pop("schedule_epoch_offset", None)
        config["time_appearance_offset"] = int(
            config.get("time_appearance_offset", 0)
        ) + int(selection.epoch)
    else:
        config["schedule_epoch_offset"] = int(
            config.get("schedule_epoch_offset", 0)
        ) + int(selection.epoch)

    optimizer = dict(config.get("optimizer", {}))
    optimizer["schedule_total_epochs"] = int(
        optimizer.get("schedule_total_epochs", selection.config["epochs"])
    )
    optimizer["schedule_epoch_offset"] = int(
        optimizer.get("schedule_epoch_offset", 0)
    ) + int(selection.epoch)
    config["optimizer"] = optimizer

    gate = dict(config.get("gate", {}))
    gate["pilot_epochs"] = 2
    config["gate"] = gate
    return config


def local_differential_candidate_gate(
    *,
    parent_metrics: Mapping[str, object],
    parent_loss_components: Mapping[str, float],
    candidate_row: Mapping[str, object],
    family_tolerance: float,
    maximum_peak_cuda_gib: float,
) -> dict[str, object]:
    """Apply the V15 parent-relative accuracy, spectrum and memory gate."""

    tolerance = float(family_tolerance)
    maximum_gib = float(maximum_peak_cuda_gib)
    if tolerance < 0.0 or not math.isfinite(maximum_gib) or maximum_gib <= 0.0:
        raise ValueError("local differential gate thresholds are invalid")
    candidate_metrics = candidate_row.get("metrics", {})
    candidate_losses = candidate_row.get("last_update_loss_components", {})
    if not isinstance(candidate_metrics, Mapping) or not isinstance(
        candidate_losses, Mapping
    ):
        raise ValueError("local differential candidate metrics are malformed")
    families = ("uniform", "layered", "marmousi")
    parent_family = parent_metrics.get("family_relative_l2", {})
    candidate_family = candidate_metrics.get("family_relative_l2", {})
    parent_spectrum = parent_metrics.get("spectrum_relative_l2", {})
    candidate_spectrum = candidate_metrics.get("spectrum_relative_l2", {})
    if not all(
        isinstance(value, Mapping)
        for value in (
            parent_family,
            candidate_family,
            parent_spectrum,
            candidate_spectrum,
        )
    ):
        raise ValueError("local differential nested metrics are malformed")
    try:
        parent_aggregate = float(parent_metrics["aggregate_relative_l2"])
        candidate_aggregate = float(candidate_metrics["aggregate_relative_l2"])
        parent_high = float(parent_spectrum["high"])
        candidate_high = float(candidate_spectrum["high"])
        parent_delta = float(parent_loss_components["delta"])
        candidate_delta = float(candidate_losses["delta"])
        improvement_vs_coarse = float(
            candidate_metrics["relative_improvement_vs_coarse"]
        )
        peak_cuda_bytes = int(candidate_row["peak_cuda_bytes"])
        parent_family_values = {
            name: float(parent_family[name]) for name in families
        }
        candidate_family_values = {
            name: float(candidate_family[name]) for name in families
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("local differential gate is missing required metrics") from error
    scalar_values = (
        parent_aggregate,
        candidate_aggregate,
        parent_high,
        candidate_high,
        parent_delta,
        candidate_delta,
        improvement_vs_coarse,
        *parent_family_values.values(),
        *candidate_family_values.values(),
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("local differential gate metrics must be finite")
    checks = {
        "aggregate_improved": candidate_aggregate < parent_aggregate,
        "families_safe": all(
            candidate_family_values[name]
            <= parent_family_values[name] * (1.0 + tolerance)
            for name in families
        ),
        "high_band_improved": candidate_high < parent_high,
        "delta_improved": candidate_delta < parent_delta,
        "better_than_coarse": improvement_vs_coarse > 0.0,
        "cuda_peak_safe": peak_cuda_bytes < maximum_gib * 1024**3,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "parent": {
            "aggregate_relative_l2": parent_aggregate,
            "family_relative_l2": parent_family_values,
            "high_band_relative_l2": parent_high,
            "delta": parent_delta,
        },
        "candidate": {
            "aggregate_relative_l2": candidate_aggregate,
            "family_relative_l2": candidate_family_values,
            "high_band_relative_l2": candidate_high,
            "delta": candidate_delta,
            "relative_improvement_vs_coarse": improvement_vs_coarse,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
    }


def build_coupled_2d_pilot_config(
    selection: PilotSelection,
    *,
    artifact_dir: str | Path,
    coupling_rank: int = 16,
) -> dict[str, object]:
    """Fork a joint-frequency pilot from an already expanded all-mode parent."""

    rank = int(coupling_rank)
    if isinstance(coupling_rank, bool) or rank <= 0 or rank > 32:
        raise ValueError("coupled 2-D pilot rank must be an integer in [1, 32]")
    config = deepcopy(selection.config)
    overrides = dict(config.get("variant_overrides", {}))
    if int(overrides.get("modes", 0)) != 101:
        raise ValueError("coupled 2-D pilot requires a completed 101-mode parent")
    if int(overrides.get("coupled_2d_rank", 0)) != 0:
        raise ValueError("coupled 2-D pilot parent must not already contain the branch")
    overrides["coupled_2d_rank"] = rank
    config["variant_overrides"] = overrides
    config["parent_checkpoint"] = str(selection.checkpoint)
    config["parent_checkpoint_identity"] = str(selection.checkpoint_identity)
    config["artifact_dir"] = str(Path(artifact_dir).resolve())

    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = False
    transfer["allow_new_coupled_2d_parameters"] = True
    config["checkpoint_transfer"] = transfer

    recovery = dict(config.get("residual_recovery", {}))
    recovery["activation_mode"] = "preserve"
    recovery["stage_epoch_offset"] = 0
    recovery["decoder_only_epochs"] = 2
    config["residual_recovery"] = recovery
    # Four ranks receive 48 records per optimizer update.  Eight is the largest
    # divisor expected to remain below the registered 23 GiB CUDA gate; the
    # mandatory two-update smoke validates the estimate on the actual GPUs.
    config["microbatch_records"] = 8

    if config.get("family_curriculum") is not None:
        config.pop("schedule_epoch_offset", None)
        config["time_appearance_offset"] = int(
            config.get("time_appearance_offset", 0)
        ) + int(selection.epoch)
    else:
        config["schedule_epoch_offset"] = int(
            config.get("schedule_epoch_offset", 0)
        ) + int(selection.epoch)
    optimizer = dict(config.get("optimizer", {}))
    optimizer["schedule_total_epochs"] = int(
        optimizer.get("schedule_total_epochs", selection.config["epochs"])
    )
    optimizer["schedule_epoch_offset"] = int(
        optimizer.get("schedule_epoch_offset", 0)
    ) + int(selection.epoch)
    config["optimizer"] = optimizer
    gate = dict(config.get("gate", {}))
    gate["pilot_epochs"] = 2
    config["gate"] = gate
    return config


def progressive_family_curriculum_stages(epochs: int) -> list[dict[str, object]]:
    """Allocate a uniform, layered-with-replay, then Marmousi-with-replay curriculum."""

    epoch_count = int(epochs)
    if epoch_count < 3:
        raise ValueError("progressive family curriculum requires at least three epochs")
    uniform_epochs = max(1, epoch_count // 8)
    layered_epochs = max(1, epoch_count // 4)
    marmousi_epochs = epoch_count - uniform_epochs - layered_epochs
    if marmousi_epochs <= 0:
        raise ValueError("progressive family curriculum leaves no Marmousi stage")
    return [
        {"epochs": uniform_epochs, "macro_pattern": ["uniform"]},
        {
            "epochs": layered_epochs,
            "macro_pattern": ["layered", "layered", "layered", "uniform"],
        },
        {
            "epochs": marmousi_epochs,
            "macro_pattern": [
                "marmousi",
                "marmousi",
                "marmousi",
                "marmousi",
                "layered",
                "uniform",
            ],
        },
    ]


def build_family_expert_pilot_config(
    selection: PilotSelection,
    *,
    artifact_dir: str | Path,
    physical_microbatch_records: int = 3,
    family_expert_rank: int = 16,
    family_expert_learning_rate: float = 1.0e-4,
) -> dict[str, object]:
    """Expand a fixed-panel parent with balanced zero-output family experts."""

    config = deepcopy(selection.config)
    macro_records = int(config.get("macro_records", 0))
    physical_microbatch = int(physical_microbatch_records)
    rank = int(family_expert_rank)
    expert_lr = float(family_expert_learning_rate)
    if macro_records != 12:
        raise ValueError("family expert parent must use 12-record macros")
    if not 1 <= physical_microbatch <= macro_records:
        raise ValueError("family expert physical microbatch must lie within one macro")
    if rank <= 0 or rank > 64:
        raise ValueError("family expert rank must lie in [1, 64]")
    if not math.isfinite(expert_lr) or expert_lr <= 0.0:
        raise ValueError("family expert learning rate must be positive and finite")
    if str(config.get("time_policy", "")) != "appearance16":
        raise ValueError("family expert parent must use appearance16")
    validation = config.get("validation")
    if (
        not isinstance(validation, Mapping)
        or int(validation.get("panel_records", 0)) != 48
        or int(validation.get("frames_per_record", 0)) != 32
    ):
        raise ValueError("family expert parent must use the 48-by-32 panel")
    overrides = dict(config.get("variant_overrides", {}))
    if int(overrides.get("family_expert_rank", 0)) != 0:
        raise ValueError("family expert parent must not already contain experts")
    overrides["family_expert_rank"] = rank

    config["parent_checkpoint"] = str(selection.checkpoint)
    config["parent_checkpoint_identity"] = str(selection.checkpoint_identity)
    config["artifact_dir"] = str(Path(artifact_dir).resolve())
    config["variant_overrides"] = overrides
    config["macros_per_update"] = 16
    config["microbatch_records"] = physical_microbatch
    config["seed"] = int(config.get("seed", 0)) + 41

    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = False
    for consumed in (
        "allow_spectral_mode_expansion",
        "allow_new_coupled_2d_parameters",
        "allow_new_local_differential_parameters",
        "allow_new_temporal_basis_parameters",
        "allow_new_family_expert_parameters",
    ):
        transfer.pop(consumed, None)
    transfer["allow_new_family_expert_parameters"] = True
    config["checkpoint_transfer"] = transfer

    inherited_appearance = int(
        config.get("time_appearance_offset", config.get("schedule_epoch_offset", 0))
    ) + int(selection.epoch)
    config.pop("schedule_epoch_offset", None)
    config["time_appearance_offset"] = inherited_appearance
    config["family_curriculum"] = {
        "stages": progressive_family_curriculum_stages(3)
    }
    config["family_experts"] = {
        "router_loss_weight": 0.01,
        "head_only_epochs": 1,
        "minimum_router_accuracy": 0.95,
        "minimum_route_probability": 0.10,
        "teacher_forced_routing": True,
        "dense_unfreeze_epoch": 3,
        "shared_unfreeze_epoch": 8,
        "geometry_unfreeze_epoch": 13,
        "backbone_unfreeze_epoch": 18,
    }

    optimizer = dict(config.get("optimizer", {}))
    optimizer["family_expert_learning_rate"] = expert_lr
    optimizer["schedule_epoch_offset"] = int(
        optimizer.get("schedule_epoch_offset", 0)
    ) + int(selection.epoch)
    config["optimizer"] = optimizer
    gate = dict(config.get("gate", {}))
    gate["pilot_epochs"] = 3
    gate["external_evidence_gate"] = True
    config["gate"] = gate
    return config


def build_update_density_pilot_config(
    selection: PilotSelection,
    *,
    artifact_dir: str | Path,
    effective_batch: int = 96,
    pilot_epochs: int = 3,
    physical_microbatch_records: int = 3,
    family_curriculum: bool = False,
    balanced_families: bool = False,
    family_gradient_norms: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Increase update density without reducing the four-GPU microbatch."""

    config = deepcopy(selection.config)
    macro_records = int(config.get("macro_records", 0))
    batch = int(effective_batch)
    epochs = int(pilot_epochs)
    physical_microbatch = int(physical_microbatch_records)
    if batch <= 0:
        raise ValueError("effective batch must be positive")
    if macro_records <= 0 or batch % macro_records:
        raise ValueError("effective batch must be divisible by macro_records")
    if physical_microbatch <= 0 or physical_microbatch > macro_records:
        raise ValueError(
            "update-density physical microbatch must be in [1, macro_records]"
        )
    if batch % 32:
        raise ValueError(
            "effective batch must be divisible by the four-GPU instantaneous batch"
        )
    if epochs <= 0:
        raise ValueError("pilot epochs must be positive")
    if bool(family_curriculum) and bool(balanced_families):
        raise ValueError("family curriculum and balanced families are mutually exclusive")
    if family_gradient_norms is not None and not bool(balanced_families):
        raise ValueError("family gradient norms require balanced families")
    family_schedule = bool(family_curriculum) or bool(balanced_families)
    if family_schedule and epochs != 3:
        raise ValueError("family schedule requires exactly three epochs")
    if int(config.get("microbatch_records", 0)) <= 0:
        raise ValueError("update-density parent microbatch must be positive")
    if str(config.get("time_policy", "")) != "appearance16":
        raise ValueError("update-density parent must use appearance16")
    validation = config.get("validation")
    if (
        not isinstance(validation, Mapping)
        or int(validation.get("panel_records", 0)) != 48
        or int(validation.get("frames_per_record", 0)) != 32
    ):
        raise ValueError("update-density parent must use the 48-by-32 panel")

    config["parent_checkpoint"] = str(selection.checkpoint)
    config["parent_checkpoint_identity"] = str(selection.checkpoint_identity)
    config["artifact_dir"] = str(Path(artifact_dir).resolve())
    config["macros_per_update"] = batch // macro_records
    # Select this independently of the effective batch because later recovery
    # stages unfreeze shared branches and have a higher activation footprint.
    config["microbatch_records"] = physical_microbatch

    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = False
    for consumed in (
        "allow_spectral_mode_expansion",
        "allow_new_coupled_2d_parameters",
        "allow_new_local_differential_parameters",
        "allow_new_temporal_basis_parameters",
    ):
        transfer.pop(consumed, None)
    config["checkpoint_transfer"] = transfer

    recovery = dict(config.get("residual_recovery", {}))
    recovery["activation_mode"] = "preserve"
    recovery["stage_epoch_offset"] = int(
        recovery.get("stage_epoch_offset", 0)
    ) + int(selection.epoch)
    config["residual_recovery"] = recovery

    if config.get("family_curriculum") is not None:
        raise ValueError("update-density parent must not contain a family curriculum")
    inherited_appearance = int(config.get("schedule_epoch_offset", 0)) + int(
        selection.epoch
    )
    if family_schedule:
        config.pop("schedule_epoch_offset", None)
        config["time_appearance_offset"] = inherited_appearance
        if bool(balanced_families):
            stages = [
                {
                    "epochs": 3,
                    "macro_pattern": ["uniform", "layered", "marmousi"],
                }
            ]
        else:
            stages = [
                {"epochs": 1, "macro_pattern": ["uniform"]},
                {
                    "epochs": 1,
                    "macro_pattern": [
                        "layered",
                        "layered",
                        "layered",
                        "uniform",
                    ],
                },
                {
                    "epochs": 1,
                    "macro_pattern": [
                        "marmousi",
                        "marmousi",
                        "marmousi",
                        "uniform",
                        "layered",
                    ],
                },
            ]
        config["family_curriculum"] = {"stages": stages}
        if family_gradient_norms is not None:
            config["family_gradient_weights"] = inverse_norm_family_weights(
                family_gradient_norms
            )
    else:
        config["schedule_epoch_offset"] = inherited_appearance

    optimizer = dict(config.get("optimizer", {}))
    optimizer["schedule_epoch_offset"] = int(
        optimizer.get("schedule_epoch_offset", 0)
    ) + int(selection.epoch)
    config["optimizer"] = optimizer

    gate = dict(config.get("gate", {}))
    gate["pilot_epochs"] = epochs
    gate["external_evidence_gate"] = True
    config["gate"] = gate
    return config


def update_density_candidate_gate(
    *,
    parent_metrics: Mapping[str, object],
    parent_loss_components: Mapping[str, float],
    candidate_row: Mapping[str, object],
    family_tolerance: float,
    maximum_peak_cuda_gib: float,
    expected_global_macros_per_update: int = 8,
    expected_physical_microbatch: int = 3,
    minimum_optimizer_updates: int = 72,
    completed_optimizer_updates: int | None = None,
    minimum_relative_improvement: float = 1.0e-3,
) -> dict[str, object]:
    """Gate a denser-update pilot without conflating architecture changes."""

    tolerance = float(family_tolerance)
    maximum_gib = float(maximum_peak_cuda_gib)
    expected_macros = int(expected_global_macros_per_update)
    expected_microbatch = int(expected_physical_microbatch)
    minimum_updates = int(minimum_optimizer_updates)
    minimum_improvement = float(minimum_relative_improvement)
    if (
        tolerance < 0.0
        or not math.isfinite(maximum_gib)
        or maximum_gib <= 0.0
        or expected_macros <= 0
        or expected_microbatch <= 0
        or minimum_updates <= 0
        or not math.isfinite(minimum_improvement)
        or not 0.0 <= minimum_improvement < 1.0
    ):
        raise ValueError("update-density gate thresholds are invalid")

    candidate_metrics = candidate_row.get("metrics")
    candidate_losses = candidate_row.get("last_update_loss_components")
    gradients = candidate_row.get("gradient_norms")
    ddp = candidate_row.get("ddp")
    if not all(
        isinstance(value, Mapping)
        for value in (candidate_metrics, candidate_losses, gradients, ddp)
    ):
        raise ValueError("update-density candidate evidence is malformed")
    if not gradients:
        raise ValueError("update-density gradient evidence is missing")

    families = ("uniform", "layered", "marmousi")
    parent_family = parent_metrics.get("family_relative_l2")
    candidate_family = candidate_metrics.get("family_relative_l2")
    parent_spectrum = parent_metrics.get("spectrum_relative_l2")
    candidate_spectrum = candidate_metrics.get("spectrum_relative_l2")
    if not all(
        isinstance(value, Mapping)
        for value in (
            parent_family,
            candidate_family,
            parent_spectrum,
            candidate_spectrum,
        )
    ):
        raise ValueError("update-density nested metrics are malformed")
    try:
        parent_aggregate = float(parent_metrics["aggregate_relative_l2"])
        candidate_aggregate = float(candidate_metrics["aggregate_relative_l2"])
        parent_high = float(parent_spectrum["high"])
        candidate_high = float(candidate_spectrum["high"])
        parent_delta = float(parent_loss_components["delta"])
        candidate_delta = float(candidate_losses["delta"])
        improvement_vs_coarse = float(
            candidate_metrics["relative_improvement_vs_coarse"]
        )
        peak_cuda_bytes = int(candidate_row["peak_cuda_bytes"])
        global_macros = int(ddp["global_macros_per_update"])
        physical_microbatch = int(candidate_row["physical_microbatch_records"])
        selected_epoch_optimizer_updates = int(candidate_row["global_step"])
        optimizer_updates = (
            selected_epoch_optimizer_updates
            if completed_optimizer_updates is None
            else int(completed_optimizer_updates)
        )
        parent_family_values = {
            name: float(parent_family[name]) for name in families
        }
        candidate_family_values = {
            name: float(candidate_family[name]) for name in families
        }
        gradient_values = {str(name): float(value) for name, value in gradients.items()}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("update-density gate is missing required evidence") from error

    scalar_values = (
        parent_aggregate,
        candidate_aggregate,
        parent_high,
        candidate_high,
        parent_delta,
        candidate_delta,
        improvement_vs_coarse,
        *parent_family_values.values(),
        *candidate_family_values.values(),
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("update-density metrics must be finite")
    if (
        selected_epoch_optimizer_updates < 0
        or optimizer_updates < selected_epoch_optimizer_updates
    ):
        raise ValueError("completed optimizer updates precede the selected epoch")
    parent_relative_improvement = (
        parent_aggregate - candidate_aggregate
    ) / max(parent_aggregate, 1.0e-16)
    finite_gradients = all(math.isfinite(value) for value in gradient_values.values())
    active_gradients = finite_gradients and any(
        value > 0.0 for value in gradient_values.values()
    )
    checks = {
        "aggregate_improved": candidate_aggregate < parent_aggregate,
        "parent_relative_improvement_1e3": parent_relative_improvement
        >= minimum_improvement,
        "families_safe": all(
            candidate_family_values[name] <= parent_family_values[name] + tolerance
            for name in families
        ),
        "high_band_safe": candidate_high <= parent_high,
        "delta_improved": candidate_delta < parent_delta,
        "better_than_coarse": improvement_vs_coarse > 0.0,
        "coarse_relative_improvement_1e3": improvement_vs_coarse
        >= minimum_improvement,
        "macro_accumulation_8": global_macros == expected_macros,
        f"physical_microbatch_{expected_microbatch}": physical_microbatch
        == expected_microbatch,
        "optimizer_updates_72": optimizer_updates >= minimum_updates,
        "finite_gradients": active_gradients,
        "cuda_peak_safe": peak_cuda_bytes < maximum_gib * 1024**3,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_relative_improvement": minimum_improvement,
        },
        "parent": {
            "aggregate_relative_l2": parent_aggregate,
            "family_relative_l2": parent_family_values,
            "high_band_relative_l2": parent_high,
            "delta": parent_delta,
        },
        "candidate": {
            "aggregate_relative_l2": candidate_aggregate,
            "family_relative_l2": candidate_family_values,
            "high_band_relative_l2": candidate_high,
            "delta": candidate_delta,
            "relative_improvement_vs_coarse": improvement_vs_coarse,
            "relative_improvement_vs_parent": parent_relative_improvement,
            "global_macros_per_update": global_macros,
            "physical_microbatch_records": physical_microbatch,
            "effective_batch": global_macros * 12,
            "optimizer_updates": optimizer_updates,
            "selected_epoch_optimizer_updates": selected_epoch_optimizer_updates,
            "completed_optimizer_updates": optimizer_updates,
            "gradient_norms": gradient_values,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
    }


def build_temporal_basis_pilot_config(
    selection: PilotSelection,
    *,
    artifact_dir: str | Path,
    temporal_basis_rank: int = 96,
    effective_batch: int = 96,
    pilot_epochs: int = 3,
) -> dict[str, object]:
    """Build an identity-safe query-invariant temporal-basis pilot."""

    config = deepcopy(selection.config)
    rank = int(temporal_basis_rank)
    batch = int(effective_batch)
    epochs = int(pilot_epochs)
    macro_records = int(config.get("macro_records", 0))
    if not 1 <= rank <= 128:
        raise ValueError("temporal basis rank must be in [1, 128]")
    if batch <= 0 or macro_records <= 0 or batch % macro_records:
        raise ValueError("effective batch must be positive and divisible by macro_records")
    macros_per_update = batch // macro_records
    if batch % 32 or macros_per_update % 4:
        raise ValueError("temporal basis batch must divide across four GPUs")
    if epochs <= 0:
        raise ValueError("pilot epochs must be positive")
    if str(config.get("time_policy", "")) != "appearance16":
        raise ValueError("temporal basis parent must use appearance16")
    validation = config.get("validation")
    if (
        not isinstance(validation, Mapping)
        or int(validation.get("panel_records", 0)) != 48
        or int(validation.get("frames_per_record", 0)) != 32
    ):
        raise ValueError("temporal basis parent must use the 48-by-32 panel")
    overrides = dict(config.get("variant_overrides", {}))
    if int(overrides.get("temporal_basis_rank", 0)) != 0:
        raise ValueError("temporal basis parent rank must be rank zero")
    overrides["temporal_basis_rank"] = rank

    config["parent_checkpoint"] = str(selection.checkpoint)
    config["parent_checkpoint_identity"] = str(selection.checkpoint_identity)
    config["artifact_dir"] = str(Path(artifact_dir).resolve())
    config["variant_overrides"] = overrides
    config["macros_per_update"] = macros_per_update
    # Rank-96 temporal coefficients exhausted the production 24 GiB cards at
    # physical batches eight and six. Four is the largest remaining divisor of
    # each 12-record macro, while gradient accumulation preserves batch 96.
    config["microbatch_records"] = 4

    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = False
    for consumed in (
        "allow_spectral_mode_expansion",
        "allow_new_coupled_2d_parameters",
        "allow_new_local_differential_parameters",
        "allow_new_temporal_basis_parameters",
    ):
        transfer.pop(consumed, None)
    transfer["allow_new_temporal_basis_parameters"] = True
    config["checkpoint_transfer"] = transfer

    recovery = dict(config.get("residual_recovery", {}))
    recovery["activation_mode"] = "preserve"
    recovery["stage_epoch_offset"] = 0
    recovery["decoder_only_epochs"] = 2
    config["residual_recovery"] = recovery

    if config.pop("family_curriculum", None) is not None:
        config.pop("schedule_epoch_offset", None)
        config["time_appearance_offset"] = int(
            config.get("time_appearance_offset", 0)
        ) + int(selection.epoch)
        config["schedule_epoch_offset"] = 0
    else:
        config["schedule_epoch_offset"] = int(
            config.get("schedule_epoch_offset", 0)
        ) + int(selection.epoch)

    optimizer = dict(config.get("optimizer", {}))
    optimizer["schedule_epoch_offset"] = int(
        optimizer.get("schedule_epoch_offset", 0)
    ) + int(selection.epoch)
    config["optimizer"] = optimizer
    gate = dict(config.get("gate", {}))
    gate["pilot_epochs"] = epochs
    config["gate"] = gate
    return config


def build_temporal_delta_floor_pilot_config(
    selection: PilotSelection,
    *,
    artifact_dir: str | Path,
    delta_energy_floor_fraction: float = 0.1,
    dense_learning_rate_multiplier: float = 2.0,
    physical_microbatch_records: int = 5,
    pilot_epochs: int = 3,
) -> dict[str, object]:
    """Continue a trained temporal basis while removing tiny-delta domination."""

    floor = float(delta_energy_floor_fraction)
    multiplier = float(dense_learning_rate_multiplier)
    physical_microbatch = int(physical_microbatch_records)
    epochs = int(pilot_epochs)
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError("delta energy floor must be finite and lie in (0, 1]")
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("dense learning-rate multiplier must be positive and finite")
    if epochs <= 0:
        raise ValueError("pilot epochs must be positive")

    config = deepcopy(selection.config)
    overrides = config.get("variant_overrides", {})
    if (
        not isinstance(overrides, Mapping)
        or int(overrides.get("temporal_basis_rank", 0)) <= 0
    ):
        raise ValueError("delta-floor continuation requires a trained temporal basis")
    if str(config.get("time_policy", "")) != "appearance16":
        raise ValueError("delta-floor continuation requires appearance16 coverage")
    if config.get("family_curriculum") is not None:
        raise ValueError("delta-floor continuation requires the ordinary fixed schedule")
    macro_records = int(config.get("macro_records", 0))
    microbatch_records = int(config.get("microbatch_records", 0))
    macros_per_update = int(config.get("macros_per_update", 0))
    if (
        macro_records != 12
        or microbatch_records <= 0
        or microbatch_records > macro_records
        or macros_per_update != 8
    ):
        raise ValueError("delta-floor continuation requires batch96 within 12-record macros")
    validation = config.get("validation", {})
    if (
        not isinstance(validation, Mapping)
        or int(validation.get("panel_records", 0)) != 48
        or int(validation.get("frames_per_record", 0)) != 32
    ):
        raise ValueError("delta-floor continuation requires the 48-by-32 panel")

    config["parent_checkpoint"] = str(selection.checkpoint)
    config["parent_checkpoint_identity"] = str(selection.checkpoint_identity)
    config["artifact_dir"] = str(Path(artifact_dir).resolve())
    if not 1 <= physical_microbatch <= macro_records:
        raise ValueError("physical microbatch must lie within one macro")
    config["microbatch_records"] = physical_microbatch
    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = False
    for consumed in (
        "allow_spectral_mode_expansion",
        "allow_new_coupled_2d_parameters",
        "allow_new_local_differential_parameters",
        "allow_new_temporal_basis_parameters",
    ):
        transfer.pop(consumed, None)
    config["checkpoint_transfer"] = transfer

    recovery = dict(config.get("residual_recovery", {}))
    recovery["activation_mode"] = "preserve"
    recovery["stage_epoch_offset"] = 0
    recovery["decoder_only_epochs"] = epochs
    config["residual_recovery"] = recovery
    config["schedule_epoch_offset"] = int(config.get("schedule_epoch_offset", 0)) + int(
        selection.epoch
    )

    loss = dict(config.get("loss", {}))
    if "delta" not in loss:
        raise ValueError("delta-floor continuation requires residual delta loss")
    loss["delta_energy_floor_fraction"] = floor
    config["loss"] = loss
    optimizer = dict(config.get("optimizer", {}))
    dense_lr = float(optimizer.get("dense_learning_rate", 0.0))
    if not math.isfinite(dense_lr) or dense_lr <= 0.0:
        raise ValueError("delta-floor continuation requires a positive dense learning rate")
    optimizer["dense_learning_rate"] = dense_lr * multiplier
    optimizer["schedule_epoch_offset"] = int(
        optimizer.get("schedule_epoch_offset", 0)
    ) + int(selection.epoch)
    config["optimizer"] = optimizer
    gate = dict(config.get("gate", {}))
    gate["pilot_epochs"] = epochs
    config["gate"] = gate
    return config


def build_temporal_gate_absorption_pilot_config(
    selection: PilotSelection,
    *,
    artifact_dir: str | Path,
    temporal_basis_learning_rate: float = 1.0e-5,
    delta_energy_floor_fraction: float | None = None,
    dense_learning_rate: float | None = None,
    pilot_epochs: int = 3,
) -> dict[str, object]:
    """Reparameterize a trained temporal branch without changing its function."""

    loss = selection.config.get("loss", {})
    if not isinstance(loss, Mapping):
        raise ValueError("gate absorption requires residual loss configuration")
    floor = float(
        loss.get("delta_energy_floor_fraction", 0.0)
        if delta_energy_floor_fraction is None
        else delta_energy_floor_fraction
    )
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("gate absorption requires a positive delta energy floor")
    optimizer = selection.config.get("optimizer", {})
    if not isinstance(optimizer, Mapping):
        raise ValueError("gate absorption requires optimizer configuration")
    parent_dense_lr = float(optimizer.get("dense_learning_rate", 0.0))
    target_dense_lr = float(
        parent_dense_lr if dense_learning_rate is None else dense_learning_rate
    )
    if (
        not math.isfinite(parent_dense_lr)
        or parent_dense_lr <= 0.0
        or not math.isfinite(target_dense_lr)
        or target_dense_lr <= 0.0
    ):
        raise ValueError("gate absorption dense learning rates must be positive and finite")
    config = build_temporal_delta_floor_pilot_config(
        selection,
        artifact_dir=artifact_dir,
        delta_energy_floor_fraction=floor,
        dense_learning_rate_multiplier=target_dense_lr / parent_dense_lr,
        pilot_epochs=pilot_epochs,
    )
    recovery = dict(config["residual_recovery"])
    recovery["absorb_temporal_basis_gate"] = True
    config["residual_recovery"] = recovery
    temporal_lr = float(temporal_basis_learning_rate)
    if not math.isfinite(temporal_lr) or temporal_lr <= 0.0:
        raise ValueError("temporal basis learning rate must be positive and finite")
    optimizer = dict(config["optimizer"])
    optimizer["temporal_basis_learning_rate"] = temporal_lr
    config["optimizer"] = optimizer
    return config


def temporal_basis_candidate_gate(
    *,
    parent_metrics: Mapping[str, object],
    parent_loss_components: Mapping[str, float],
    candidate_row: Mapping[str, object],
    family_tolerance: float,
    maximum_peak_cuda_gib: float,
    expected_global_macros_per_update: int = 8,
    expected_physical_microbatch: int = 4,
    minimum_optimizer_updates: int = 72,
) -> dict[str, object]:
    """Gate the temporal-basis branch with accuracy and wake-up evidence."""

    tolerance = float(family_tolerance)
    maximum_gib = float(maximum_peak_cuda_gib)
    if tolerance < 0.0 or not math.isfinite(maximum_gib) or maximum_gib <= 0.0:
        raise ValueError("temporal basis gate thresholds are invalid")
    candidate_metrics = candidate_row.get("metrics")
    candidate_losses = candidate_row.get("last_update_loss_components")
    gradients = candidate_row.get("gradient_norms")
    ddp = candidate_row.get("ddp")
    if not all(
        isinstance(value, Mapping)
        for value in (candidate_metrics, candidate_losses, gradients, ddp)
    ):
        raise ValueError("temporal basis candidate evidence is malformed")
    families = ("uniform", "layered", "marmousi")
    parent_family = parent_metrics.get("family_relative_l2")
    candidate_family = candidate_metrics.get("family_relative_l2")
    parent_spectrum = parent_metrics.get("spectrum_relative_l2")
    candidate_spectrum = candidate_metrics.get("spectrum_relative_l2")
    parent_coarse_metrics = parent_metrics.get("coarse_metrics")
    candidate_coarse_metrics = candidate_metrics.get("coarse_metrics")
    if not all(
        isinstance(value, Mapping)
        for value in (
            parent_family,
            candidate_family,
            parent_spectrum,
            candidate_spectrum,
            parent_coarse_metrics,
            candidate_coarse_metrics,
        )
    ):
        raise ValueError("temporal basis nested metrics are malformed")
    try:
        parent_aggregate = float(parent_metrics["aggregate_relative_l2"])
        candidate_aggregate = float(candidate_metrics["aggregate_relative_l2"])
        parent_coarse = float(parent_coarse_metrics["aggregate_relative_l2"])
        candidate_coarse = float(candidate_coarse_metrics["aggregate_relative_l2"])
        parent_high = float(parent_spectrum["high"])
        candidate_high = float(candidate_spectrum["high"])
        parent_delta = float(parent_loss_components["delta"])
        candidate_delta = float(candidate_losses["delta"])
        improvement = float(candidate_metrics["relative_improvement_vs_coarse"])
        family_parent = {name: float(parent_family[name]) for name in families}
        family_candidate = {name: float(candidate_family[name]) for name in families}
        gate_gradient = float(gradients["temporal_basis_gate"])
        feature_gradient = float(gradients["temporal_basis_features"])
        global_macros = int(ddp["global_macros_per_update"])
        physical_microbatch = int(candidate_row["physical_microbatch_records"])
        optimizer_updates = int(candidate_row["global_step"])
        peak_cuda_bytes = int(candidate_row["peak_cuda_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("temporal basis gate is missing required evidence") from error
    scalars = (
        parent_aggregate,
        candidate_aggregate,
        parent_coarse,
        candidate_coarse,
        parent_high,
        candidate_high,
        parent_delta,
        candidate_delta,
        improvement,
        gate_gradient,
        feature_gradient,
        *family_parent.values(),
        *family_candidate.values(),
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("temporal basis metrics must be finite")
    checks = {
        "aggregate_improved": candidate_aggregate < parent_aggregate,
        "same_coarse_panel": math.isclose(
            candidate_coarse, parent_coarse, rel_tol=1.0e-6, abs_tol=1.0e-8
        ),
        "families_safe": all(
            family_candidate[name] <= family_parent[name] + tolerance
            for name in families
        ),
        "high_band_safe": candidate_high <= parent_high,
        "delta_improved": candidate_delta < parent_delta,
        "better_than_coarse": improvement > 0.0,
        "temporal_gate_active": gate_gradient > 0.0,
        "temporal_features_active": feature_gradient > 0.0,
        "macro_accumulation_8": global_macros
        == int(expected_global_macros_per_update),
        f"physical_microbatch_{int(expected_physical_microbatch)}": physical_microbatch
        == int(expected_physical_microbatch),
        "optimizer_updates_72": optimizer_updates >= int(minimum_optimizer_updates),
        "cuda_peak_safe": peak_cuda_bytes < maximum_gib * 1024**3,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "parent": {
            "aggregate_relative_l2": parent_aggregate,
            "family_relative_l2": family_parent,
            "high_band_relative_l2": parent_high,
            "delta": parent_delta,
        },
        "candidate": {
            "aggregate_relative_l2": candidate_aggregate,
            "family_relative_l2": family_candidate,
            "high_band_relative_l2": candidate_high,
            "delta": candidate_delta,
            "relative_improvement_vs_coarse": improvement,
            "temporal_basis_gate_gradient": gate_gradient,
            "temporal_basis_feature_gradient": feature_gradient,
            "global_macros_per_update": global_macros,
            "physical_microbatch_records": physical_microbatch,
            "effective_batch": global_macros * 12,
            "optimizer_updates": optimizer_updates,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
    }


def temporal_delta_floor_candidate_gate(
    *,
    parent_metrics: Mapping[str, object],
    parent_loss_components: Mapping[str, float],
    candidate_row: Mapping[str, object],
    family_tolerance: float,
    maximum_peak_cuda_gib: float,
    expected_global_macros_per_update: int = 8,
    expected_physical_microbatch: int = 4,
    minimum_optimizer_updates: int = 72,
    completed_optimizer_updates: int | None = None,
    minimum_relative_improvement: float = 1.0e-3,
) -> dict[str, object]:
    """Gate a changed delta denominator using only comparable loss evidence."""

    tolerance = float(family_tolerance)
    maximum_gib = float(maximum_peak_cuda_gib)
    minimum_improvement = float(minimum_relative_improvement)
    if (
        tolerance < 0.0
        or not math.isfinite(maximum_gib)
        or maximum_gib <= 0.0
        or not math.isfinite(minimum_improvement)
        or not 0.0 <= minimum_improvement < 1.0
    ):
        raise ValueError("temporal delta-floor gate thresholds are invalid")
    candidate_metrics = candidate_row.get("metrics")
    candidate_losses = candidate_row.get("last_update_loss_components")
    gradients = candidate_row.get("gradient_norms")
    ddp = candidate_row.get("ddp")
    if not all(
        isinstance(value, Mapping)
        for value in (candidate_metrics, candidate_losses, gradients, ddp)
    ):
        raise ValueError("temporal delta-floor candidate evidence is malformed")
    families = ("uniform", "layered", "marmousi")
    parent_family = parent_metrics.get("family_relative_l2")
    candidate_family = candidate_metrics.get("family_relative_l2")
    parent_spectrum = parent_metrics.get("spectrum_relative_l2")
    candidate_spectrum = candidate_metrics.get("spectrum_relative_l2")
    parent_coarse_metrics = parent_metrics.get("coarse_metrics")
    candidate_coarse_metrics = candidate_metrics.get("coarse_metrics")
    if not all(
        isinstance(value, Mapping)
        for value in (
            parent_family,
            candidate_family,
            parent_spectrum,
            candidate_spectrum,
            parent_coarse_metrics,
            candidate_coarse_metrics,
        )
    ):
        raise ValueError("temporal delta-floor nested metrics are malformed")
    try:
        parent_aggregate = float(parent_metrics["aggregate_relative_l2"])
        candidate_aggregate = float(candidate_metrics["aggregate_relative_l2"])
        parent_coarse = float(parent_coarse_metrics["aggregate_relative_l2"])
        candidate_coarse = float(candidate_coarse_metrics["aggregate_relative_l2"])
        parent_high = float(parent_spectrum["high"])
        candidate_high = float(candidate_spectrum["high"])
        parent_frame = float(parent_loss_components["frame"])
        candidate_frame = float(candidate_losses["frame"])
        candidate_delta = float(candidate_losses["delta"])
        improvement = float(candidate_metrics["relative_improvement_vs_coarse"])
        family_parent = {name: float(parent_family[name]) for name in families}
        family_candidate = {name: float(candidate_family[name]) for name in families}
        gate_gradient = float(gradients["temporal_basis_gate"])
        feature_gradient = float(gradients["temporal_basis_features"])
        global_macros = int(ddp["global_macros_per_update"])
        physical_microbatch = int(candidate_row["physical_microbatch_records"])
        selected_epoch_optimizer_updates = int(candidate_row["global_step"])
        optimizer_updates = (
            selected_epoch_optimizer_updates
            if completed_optimizer_updates is None
            else int(completed_optimizer_updates)
        )
        peak_cuda_bytes = int(candidate_row["peak_cuda_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("temporal delta-floor gate is missing required evidence") from error
    scalars = (
        parent_aggregate,
        candidate_aggregate,
        parent_coarse,
        candidate_coarse,
        parent_high,
        candidate_high,
        parent_frame,
        candidate_frame,
        candidate_delta,
        improvement,
        gate_gradient,
        feature_gradient,
        *family_parent.values(),
        *family_candidate.values(),
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("temporal delta-floor metrics must be finite")
    if (
        selected_epoch_optimizer_updates < 0
        or optimizer_updates < selected_epoch_optimizer_updates
    ):
        raise ValueError("completed optimizer updates precede the selected epoch")
    parent_relative_improvement = (
        parent_aggregate - candidate_aggregate
    ) / max(parent_aggregate, 1.0e-16)
    checks = {
        "aggregate_improved": candidate_aggregate < parent_aggregate,
        "parent_relative_improvement_1e3": parent_relative_improvement
        >= minimum_improvement,
        "same_coarse_panel": math.isclose(
            candidate_coarse, parent_coarse, rel_tol=1.0e-6, abs_tol=1.0e-8
        ),
        "families_safe": all(
            family_candidate[name] <= family_parent[name] + tolerance
            for name in families
        ),
        "high_band_safe": candidate_high <= parent_high + tolerance,
        "frame_improved": candidate_frame < parent_frame,
        "floored_delta_below_one": candidate_delta < 1.0,
        "better_than_coarse": improvement > 0.0,
        "coarse_relative_improvement_1e3": improvement >= minimum_improvement,
        "temporal_gate_active": gate_gradient > 0.0,
        "temporal_features_active": feature_gradient > 0.0,
        "macro_accumulation_8": global_macros
        == int(expected_global_macros_per_update),
        f"physical_microbatch_{int(expected_physical_microbatch)}": physical_microbatch
        == int(expected_physical_microbatch),
        "optimizer_updates_72": optimizer_updates >= int(minimum_optimizer_updates),
        "cuda_peak_safe": peak_cuda_bytes < maximum_gib * 1024**3,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_relative_improvement": minimum_improvement,
        },
        "parent": {
            "aggregate_relative_l2": parent_aggregate,
            "family_relative_l2": family_parent,
            "high_band_relative_l2": parent_high,
            "frame": parent_frame,
        },
        "candidate": {
            "aggregate_relative_l2": candidate_aggregate,
            "family_relative_l2": family_candidate,
            "high_band_relative_l2": candidate_high,
            "frame": candidate_frame,
            "floored_delta": candidate_delta,
            "relative_improvement_vs_coarse": improvement,
            "relative_improvement_vs_parent": parent_relative_improvement,
            "temporal_basis_gate_gradient": gate_gradient,
            "temporal_basis_feature_gradient": feature_gradient,
            "global_macros_per_update": global_macros,
            "physical_microbatch_records": physical_microbatch,
            "effective_batch": global_macros * 12,
            "optimizer_updates": optimizer_updates,
            "selected_epoch_optimizer_updates": selected_epoch_optimizer_updates,
            "completed_optimizer_updates": optimizer_updates,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
    }


def coupled_2d_candidate_gate(
    *,
    parent_metrics: Mapping[str, object],
    parent_loss_components: Mapping[str, float],
    candidate_row: Mapping[str, object],
    family_tolerance: float,
    maximum_peak_cuda_gib: float,
) -> dict[str, object]:
    """Require accuracy, high-band, gradient and memory evidence for 2-D coupling."""

    tolerance = float(family_tolerance)
    maximum_gib = float(maximum_peak_cuda_gib)
    if tolerance < 0.0 or not math.isfinite(maximum_gib) or maximum_gib <= 0.0:
        raise ValueError("coupled 2-D gate thresholds are invalid")
    candidate_metrics = candidate_row.get("metrics", {})
    candidate_losses = candidate_row.get("last_update_loss_components", {})
    gradients = candidate_row.get("gradient_norms", {})
    if not all(
        isinstance(value, Mapping)
        for value in (candidate_metrics, candidate_losses, gradients)
    ):
        raise ValueError("coupled 2-D candidate evidence is malformed")
    families = ("uniform", "layered", "marmousi")
    parent_family = parent_metrics.get("family_relative_l2", {})
    candidate_family = candidate_metrics.get("family_relative_l2", {})
    parent_spectrum = parent_metrics.get("spectrum_relative_l2", {})
    candidate_spectrum = candidate_metrics.get("spectrum_relative_l2", {})
    if not all(
        isinstance(value, Mapping)
        for value in (
            parent_family,
            candidate_family,
            parent_spectrum,
            candidate_spectrum,
        )
    ):
        raise ValueError("coupled 2-D nested metrics are malformed")
    try:
        parent_aggregate = float(parent_metrics["aggregate_relative_l2"])
        candidate_aggregate = float(candidate_metrics["aggregate_relative_l2"])
        parent_high = float(parent_spectrum["high"])
        candidate_high = float(candidate_spectrum["high"])
        parent_delta = float(parent_loss_components["delta"])
        candidate_delta = float(candidate_losses["delta"])
        improvement_vs_coarse = float(
            candidate_metrics["relative_improvement_vs_coarse"]
        )
        gate_gradient = float(gradients["coupled_2d_gate"])
        feature_gradient = float(gradients["coupled_2d_features"])
        peak_cuda_bytes = int(candidate_row["peak_cuda_bytes"])
        parent_family_values = {
            name: float(parent_family[name]) for name in families
        }
        candidate_family_values = {
            name: float(candidate_family[name]) for name in families
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("coupled 2-D gate is missing required evidence") from error
    scalar_values = (
        parent_aggregate,
        candidate_aggregate,
        parent_high,
        candidate_high,
        parent_delta,
        candidate_delta,
        improvement_vs_coarse,
        gate_gradient,
        feature_gradient,
        *parent_family_values.values(),
        *candidate_family_values.values(),
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("coupled 2-D gate evidence must be finite")
    checks = {
        "aggregate_improved": candidate_aggregate < parent_aggregate,
        "families_safe": all(
            candidate_family_values[name]
            <= parent_family_values[name] * (1.0 + tolerance)
            for name in families
        ),
        "high_band_improved": candidate_high < parent_high,
        "delta_improved": candidate_delta < parent_delta,
        "better_than_coarse": improvement_vs_coarse > 0.0,
        "coupled_gate_active": gate_gradient > 0.0,
        "coupled_features_active": feature_gradient > 0.0,
        "cuda_peak_safe": peak_cuda_bytes < maximum_gib * 1024**3,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "parent": {
            "aggregate_relative_l2": parent_aggregate,
            "family_relative_l2": parent_family_values,
            "high_band_relative_l2": parent_high,
            "delta": parent_delta,
        },
        "candidate": {
            "aggregate_relative_l2": candidate_aggregate,
            "family_relative_l2": candidate_family_values,
            "high_band_relative_l2": candidate_high,
            "delta": candidate_delta,
            "relative_improvement_vs_coarse": improvement_vs_coarse,
            "coupled_2d_gate_gradient": gate_gradient,
            "coupled_2d_feature_gradient": feature_gradient,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
    }


__all__ = [
    "FullTimeParentSelection",
    "PilotSelection",
    "build_architecture_pilot_config",
    "build_long_continuation_config",
    "build_all_modes_pilot_config",
    "build_coupled_2d_pilot_config",
    "build_family_expert_pilot_config",
    "progressive_family_curriculum_stages",
    "build_temporal_basis_pilot_config",
    "build_temporal_delta_floor_pilot_config",
    "build_temporal_gate_absorption_pilot_config",
    "coupled_2d_candidate_gate",
    "local_differential_candidate_gate",
    "select_best_full_time_parent",
    "select_best_pilot_candidate",
    "temporal_basis_candidate_gate",
    "temporal_delta_floor_candidate_gate",
]
