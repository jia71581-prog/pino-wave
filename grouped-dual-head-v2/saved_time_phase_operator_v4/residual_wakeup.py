"""Evidence selection for waking a collapsed saved-time residual head."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class WakeupDecision:
    mode: str
    scale: float
    output_std: float
    checks: dict[str, bool]


def _metrics(report: Mapping[str, object], scale: str) -> Mapping[str, object]:
    scales = report.get("scales")
    if not isinstance(scales, Mapping) or scale not in scales:
        raise ValueError(f"residual wake-up sweep is missing scale {scale}")
    value = scales[scale]
    if not isinstance(value, Mapping):
        raise ValueError("residual wake-up scale metrics are malformed")
    return value


def select_wakeup_decision(
    sweep: Mapping[str, object],
    *,
    minimum_relative_improvement: float = 0.01,
    family_tolerance: float = 0.03,
    minimum_correction_ratio: float = 0.03,
    maximum_correction_ratio: float = 0.30,
    reset_output_std: float = 1.0e-4,
) -> WakeupDecision:
    """Choose safe rescaling or fall back to deterministic output-head reset."""

    scales = sweep.get("scales")
    if not isinstance(scales, Mapping) or "1.0" not in scales:
        raise ValueError("residual wake-up sweep requires a unit scale baseline")
    baseline = _metrics(sweep, "1.0")
    baseline_family = baseline.get("family_relative_l2")
    baseline_spectrum = baseline.get("spectrum_relative_l2")
    if not isinstance(baseline_family, Mapping) or not isinstance(
        baseline_spectrum, Mapping
    ):
        raise ValueError("unit scale wake-up metrics are malformed")
    try:
        baseline_aggregate = float(baseline["aggregate_relative_l2"])
        baseline_high = float(baseline_spectrum["high"])
        baseline_families = {
            name: float(baseline_family[name])
            for name in ("uniform", "layered", "marmousi")
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("unit scale wake-up metrics are incomplete") from error
    if not all(
        math.isfinite(value)
        for value in (
            baseline_aggregate,
            baseline_high,
            *baseline_families.values(),
        )
    ):
        raise ValueError("unit scale wake-up metrics must be finite")

    eligible: list[tuple[float, float, dict[str, bool]]] = []
    for key, raw in scales.items():
        scale = float(key)
        if scale in {0.0, 1.0} or not isinstance(raw, Mapping):
            continue
        family = raw.get("family_relative_l2")
        spectrum = raw.get("spectrum_relative_l2")
        if not isinstance(family, Mapping) or not isinstance(spectrum, Mapping):
            continue
        try:
            aggregate = float(raw["aggregate_relative_l2"])
            high = float(spectrum["high"])
            ratio = float(raw["correction_to_coarse_l2_ratio"])
            families = {
                name: float(family[name]) for name in baseline_families
            }
        except (KeyError, TypeError, ValueError):
            continue
        checks = {
            "aggregate_improved": (
                baseline_aggregate - aggregate
            )
            / max(baseline_aggregate, 1.0e-16)
            >= float(minimum_relative_improvement),
            "families_safe": all(
                families[name]
                <= baseline_families[name] + float(family_tolerance)
                for name in baseline_families
            ),
            "high_band_safe": high <= baseline_high,
            "correction_bounded": float(minimum_correction_ratio)
            <= ratio
            <= float(maximum_correction_ratio),
        }
        if all(checks.values()):
            eligible.append((aggregate, scale, checks))
    if eligible:
        _, scale, checks = min(eligible)
        return WakeupDecision("rescale", scale, reset_output_std, checks)
    return WakeupDecision(
        "reset",
        1.0,
        float(reset_output_std),
        {
            "aggregate_improved": False,
            "families_safe": False,
            "high_band_safe": False,
            "correction_bounded": False,
        },
    )


def build_wakeup_config(
    selection,
    decision: WakeupDecision,
    *,
    artifact_dir: str | Path,
) -> dict[str, object]:
    """Fork a comparable V17 pilot from one identity-bound selected parent."""

    config = deepcopy(selection.config)
    config["parent_checkpoint"] = str(selection.checkpoint)
    config["parent_checkpoint_identity"] = str(selection.checkpoint_identity)
    config["artifact_dir"] = str(Path(artifact_dir).resolve())

    transfer = dict(config.get("checkpoint_transfer", {}))
    transfer["parent_residual_already_active"] = True
    transfer["parent_optimizer_state"] = False
    config["checkpoint_transfer"] = transfer

    recovery = dict(config.get("residual_recovery", {}))
    recovery["activation_mode"] = str(decision.mode)
    recovery["correction_scale"] = float(decision.scale)
    recovery["output_std"] = float(decision.output_std)
    recovery["stage_epoch_offset"] = int(
        recovery.get("stage_epoch_offset", 0)
    ) + int(selection.epoch)
    config["residual_recovery"] = recovery

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
    optimizer["schedule_epoch_offset"] = int(
        optimizer.get("schedule_epoch_offset", 0)
    ) + int(selection.epoch)
    optimizer["schedule_total_epochs"] = int(
        optimizer.get("schedule_total_epochs", config.get("epochs", 40))
    )
    optimizer["gradient_clip_mode"] = "prefix_limits"
    optimizer["gradient_clip_prefix_limits"] = {
        "dense_decoder": 20.0,
        "coordinate_encoder": 5.0,
        "source_encoder": 5.0,
        "fusion": 5.0,
        "travel_branch": 5.0,
        "medium_encoder": 2.0,
        "default": 1.0,
    }
    config["optimizer"] = optimizer

    loss = dict(config.get("loss", {}))
    loss["hard_causality"] = True
    loss["hard_causality_lead_cycles"] = 1.0
    config["loss"] = loss
    gate = dict(config.get("gate", {}))
    gate["pilot_epochs"] = 3
    config["gate"] = gate
    return config


def build_hybrid_wakeup_config(
    selection,
    *,
    artifact_dir: str | Path,
) -> dict[str, object]:
    """Reset the residual head on one selected coarse or corrected parent."""

    decision = WakeupDecision(
        mode="reset",
        scale=1.0,
        output_std=1.0e-4,
        checks={
            "aggregate_improved": False,
            "families_safe": True,
            "high_band_safe": True,
            "correction_bounded": False,
        },
    )
    return build_wakeup_config(
        selection,
        decision,
        artifact_dir=artifact_dir,
    )


def wakeup_candidate_gate(
    *,
    parent_metrics: Mapping[str, object],
    parent_loss_components: Mapping[str, float],
    candidate_row: Mapping[str, object],
    family_tolerance: float,
    maximum_peak_cuda_gib: float,
) -> dict[str, object]:
    """Apply parent-relative V17 accuracy, activation, and resource gates."""

    candidate_metrics = candidate_row.get("metrics")
    candidate_losses = candidate_row.get("last_update_loss_components")
    if not isinstance(candidate_metrics, Mapping) or not isinstance(
        candidate_losses, Mapping
    ):
        raise ValueError("residual wake-up candidate metrics are malformed")
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
        raise ValueError("residual wake-up nested metrics are malformed")
    families = ("uniform", "layered", "marmousi")
    try:
        parent_aggregate = float(parent_metrics["aggregate_relative_l2"])
        candidate_aggregate = float(candidate_metrics["aggregate_relative_l2"])
        parent_delta = float(parent_loss_components["delta"])
        candidate_delta = float(candidate_losses["delta"])
        parent_high = float(parent_spectrum["high"])
        candidate_high = float(candidate_spectrum["high"])
        correction_ratio = float(
            candidate_metrics["correction_to_coarse_l2_ratio"]
        )
        peak_cuda_bytes = int(candidate_row["peak_cuda_bytes"])
        parent_families = {name: float(parent_family[name]) for name in families}
        candidate_families = {
            name: float(candidate_family[name]) for name in families
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("residual wake-up gate is missing required metrics") from error
    finite = (
        parent_aggregate,
        candidate_aggregate,
        parent_delta,
        candidate_delta,
        parent_high,
        candidate_high,
        correction_ratio,
        *parent_families.values(),
        *candidate_families.values(),
    )
    if not all(math.isfinite(value) for value in finite):
        raise ValueError("residual wake-up gate metrics must be finite")
    tolerance = float(family_tolerance)
    maximum_bytes = float(maximum_peak_cuda_gib) * 1024**3
    checks = {
        "aggregate_improved": candidate_aggregate < parent_aggregate,
        "families_safe": all(
            candidate_families[name] <= parent_families[name] + tolerance
            for name in families
        ),
        "high_band_improved": candidate_high < parent_high,
        "delta_improved": candidate_delta < parent_delta,
        "correction_bounded": 0.03 <= correction_ratio <= 0.30,
        "cuda_peak_safe": peak_cuda_bytes < maximum_bytes,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "parent": {
            "aggregate_relative_l2": parent_aggregate,
            "family_relative_l2": parent_families,
            "high_band_relative_l2": parent_high,
            "delta": parent_delta,
        },
        "candidate": {
            "aggregate_relative_l2": candidate_aggregate,
            "family_relative_l2": candidate_families,
            "high_band_relative_l2": candidate_high,
            "delta": candidate_delta,
            "correction_to_coarse_l2_ratio": correction_ratio,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
    }


__all__ = [
    "WakeupDecision",
    "build_hybrid_wakeup_config",
    "build_wakeup_config",
    "select_wakeup_decision",
    "wakeup_candidate_gate",
]
