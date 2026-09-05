"""Controlled V4 capacity-by-propagation probe definitions and selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ProbeVariant:
    depth: int
    use_local_phase: bool
    spectral_rank: int = 112
    modes: int = 32
    coupled_axes: bool = False
    local_differential_residual: bool = False
    coupled_2d_rank: int = 0
    temporal_basis_rank: int = 0
    family_expert_rank: int = 0
    band_adapter_rank: int = 0
    band_adapter_architecture: str = "low_rank"
    band_adapter_spectral_rank: int = 32
    band_adapter_modes: int = 32
    band_adapter_full_depth: int = 4
    band_adapter_coarse_depth: int = 2
    band_adapter_activation_checkpointing: bool = True
    band_adapter_dropout: float = 0.0
    band_adapter_preserve_high_band: bool = True
    local_field: bool = False
    local_field_channel_multipliers: tuple[int, ...] = (1, 1, 2, 2)
    local_field_causal_width_s: float = 0.005
    local_field_residual: bool = False
    local_field_extended_late_features: bool = False
    local_field_temporal_operator_rank: int = 0
    local_field_temporal_operator_spatial_kernel: int = 1
    local_field_warp: bool = False
    local_field_warp_max_shift_cells: float = 8.0
    local_field_warp_shift_dilation: int = 1
    local_field_green_kernel: bool = False
    local_field_green_kernel_size: int = 5
    local_field_green_dilations: tuple[int, ...] = (1, 2, 4)
    local_field_temporal_latent_basis: bool = False
    local_field_temporal_latent_rank: int = 8
    local_field_temporal_latent_harmonics: int = 4
    local_field_multi_arrival: bool = False
    local_field_multi_arrival_paths: int = 3
    local_field_multi_arrival_max_shift_cells: float = 8.0
    local_field_multi_arrival_max_delay_frac: float = 0.5
    local_field_dispersive_modal: bool = False
    local_field_dispersive_modal_modes: int = 16
    local_field_dispersive_modal_max_frequency: float = 8.0
    local_field_windowed_propagation: bool = False
    local_field_windowed_propagation_window: int = 2
    local_field_windowed_propagation_stride: int = 1
    local_field_windowed_propagation_rank: int = 8
    local_field_windowed_propagation_max_advect_cells: float = 8.0
    local_field_helmholtz_synthesis: bool = False
    local_field_helmholtz_synthesis_frequencies: int = 64
    local_field_helmholtz_synthesis_wkb_phase: bool = True
    local_field_helmholtz_synthesis_rank: int = 0
    local_field_helmholtz_synthesis_late_rank: int = 0
    local_field_helmholtz_synthesis_late_frequencies: int = 0
    local_field_helmholtz_synthesis_frequency_softmax: bool = False
    local_field_helmholtz_synthesis_source_onset_phase: bool = False
    local_field_helmholtz_source_relative_coordinates: bool = False
    local_field_helmholtz_spectral_bypass: bool = False
    local_field_helmholtz_spectral_bypass_per_branch: bool = False
    local_field_helmholtz_background_conditioning: bool = False
    local_field_helmholtz_background_sigma_cells: float = 2.0
    local_field_helmholtz_background_global_propagator: bool = False
    local_field_helmholtz_background_propagation_modes: int = 48
    local_field_helmholtz_background_direct_frequency_head: bool = False
    local_field_helmholtz_background_direct_frequencies: int = 32
    local_field_helmholtz_background_direct_spectral_experts: int = 0
    local_field_adapter_gate_init: float = 0.0
    high_frequency_residual: bool = False
    high_frequency_hidden: int = 64
    high_frequency_depth: int = 3


def probe_variants() -> dict[str, ProbeVariant]:
    return {
        "shallow_no_phase": ProbeVariant(depth=2, use_local_phase=False),
        "deep_no_phase": ProbeVariant(depth=8, use_local_phase=False),
        "shallow_phase": ProbeVariant(depth=2, use_local_phase=True),
        "deep_phase": ProbeVariant(depth=8, use_local_phase=True),
    }


def select_probe_candidate(
    scores: Mapping[str, float],
    family_scores: Mapping[str, Mapping[str, float]],
    *,
    minimum_relative_improvement: float = 0.15,
    family_regression_tolerance: float = 0.05,
) -> str:
    control_name = "shallow_no_phase"
    if control_name not in scores or control_name not in family_scores:
        raise ValueError("V4 probe control metrics are missing")
    control = float(scores[control_name])
    if control <= 0 or not 0 <= minimum_relative_improvement < 1:
        raise ValueError("V4 probe selection thresholds are invalid")
    families = ("uniform", "layered", "marmousi")
    eligible: list[str] = []
    for name, score_value in scores.items():
        if name == control_name or name not in family_scores:
            continue
        improvement = (control - float(score_value)) / control
        family_ok = all(
            float(family_scores[name][family])
            <= (1.0 + family_regression_tolerance)
            * float(family_scores[control_name][family])
            for family in families
        )
        if improvement >= minimum_relative_improvement and family_ok:
            eligible.append(name)
    if not eligible:
        raise RuntimeError("no V4 probe variant passed the selection gate")
    return min(eligible, key=lambda name: float(scores[name]))


__all__ = ["ProbeVariant", "probe_variants", "select_probe_candidate"]
