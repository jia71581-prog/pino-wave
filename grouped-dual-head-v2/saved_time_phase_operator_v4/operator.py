"""V3-compatible stored-time, propagation-conditioned acoustic operator."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch

from grouped_ufno_mionet_v3.model.operator import (
    PhaseAlignedComplexFNOMIONet,
    PreparedV3State,
    free_surface_factor,
)

from .band_adapter import project_low_mid_increment
from .decoder import PropagationConditionedDenseDecoder
from .local_field import LocalPropagationFieldGenerator
from .time_grid import SavedTimeGrid


class SavedTimePhaseOperatorV4(PhaseAlignedComplexFNOMIONet):
    """Keep V3 medium/source/query branches and replace its dense decoder."""

    def __init__(
        self,
        *,
        saved_time_s: Sequence[float] | torch.Tensor,
        width: int = 64,
        rank: int = 128,
        spectral_rank: int = 40,
        modes: tuple[int, ...] = (20, 16, 12, 8),
        dense_spectral_rank: int = 112,
        dense_modes: int = 32,
        dense_depth: int = 8,
        dense_time_block: int = 1,
        use_local_phase: bool = True,
        activation_checkpointing: bool = True,
        dense_coupled_axes: bool = False,
        dense_local_differential_residual: bool = False,
        dense_coupled_2d_rank: int = 0,
        dense_temporal_basis_rank: int = 0,
        dense_family_expert_rank: int = 0,
        dense_band_adapter_rank: int = 0,
        dense_band_adapter_architecture: str = "low_rank",
        dense_band_adapter_spectral_rank: int = 32,
        dense_band_adapter_modes: int = 32,
        dense_band_adapter_full_depth: int = 4,
        dense_band_adapter_coarse_depth: int = 2,
        dense_band_adapter_activation_checkpointing: bool = True,
        dense_band_adapter_dropout: float = 0.0,
        dense_band_adapter_preserve_high_band: bool = True,
        dense_local_field: bool = False,
        dense_local_field_channel_multipliers: Sequence[int] = (1, 1, 2, 2),
        dense_local_field_causal_width_s: float = 0.005,
        dense_local_field_residual: bool = False,
        dense_local_field_extended_late_features: bool = False,
        dense_local_field_temporal_operator_rank: int = 0,
        dense_local_field_temporal_operator_spatial_kernel: int = 1,
        dense_local_field_warp: bool = False,
        dense_local_field_warp_max_shift_cells: float = 8.0,
        dense_local_field_warp_shift_dilation: int = 1,
        dense_local_field_green_kernel: bool = False,
        dense_local_field_green_kernel_size: int = 5,
        dense_local_field_green_dilations: Sequence[int] = (1, 2, 4),
        dense_local_field_temporal_latent_basis: bool = False,
        dense_local_field_temporal_latent_rank: int = 8,
        dense_local_field_temporal_latent_harmonics: int = 4,
        dense_local_field_multi_arrival: bool = False,
        dense_local_field_multi_arrival_paths: int = 3,
        dense_local_field_multi_arrival_max_shift_cells: float = 8.0,
        dense_local_field_multi_arrival_max_delay_frac: float = 0.5,
        dense_local_field_dispersive_modal: bool = False,
        dense_local_field_dispersive_modal_modes: int = 16,
        dense_local_field_dispersive_modal_max_frequency: float = 8.0,
        dense_local_field_windowed_propagation: bool = False,
        dense_local_field_windowed_propagation_window: int = 2,
        dense_local_field_windowed_propagation_stride: int = 1,
        dense_local_field_windowed_propagation_rank: int = 8,
        dense_local_field_windowed_propagation_max_advect_cells: float = 8.0,
        dense_local_field_helmholtz_synthesis: bool = False,
        dense_local_field_helmholtz_synthesis_frequencies: int = 64,
        dense_local_field_helmholtz_synthesis_wkb_phase: bool = True,
        dense_local_field_helmholtz_synthesis_rank: int = 0,
        dense_local_field_helmholtz_synthesis_late_rank: int = 0,
        dense_local_field_helmholtz_synthesis_late_frequencies: int = 0,
        dense_local_field_helmholtz_synthesis_frequency_softmax: bool = False,
        dense_local_field_helmholtz_synthesis_source_onset_phase: bool = False,
        dense_local_field_helmholtz_source_relative_coordinates: bool = False,
        dense_local_field_helmholtz_spectral_bypass: bool = False,
        dense_local_field_helmholtz_spectral_bypass_per_branch: bool = False,
        dense_local_field_helmholtz_background_conditioning: bool = False,
        dense_local_field_helmholtz_background_sigma_cells: float = 2.0,
        dense_local_field_helmholtz_background_global_propagator: bool = False,
        dense_local_field_helmholtz_background_propagation_modes: int = 48,
        dense_local_field_helmholtz_background_direct_frequency_head: bool = False,
        dense_local_field_helmholtz_background_direct_frequencies: int = 32,
        dense_local_field_helmholtz_background_direct_spectral_experts: int = 0,
        dense_local_field_adapter_gate_init: float = 0.0,
        dense_high_frequency_residual: bool = False,
        dense_high_frequency_hidden: int = 64,
        dense_high_frequency_depth: int = 3,
        heads: int = 4,
        token_grid: int = 8,
        position_bands: int = 4,
        fourier_bands: int = 6,
        gabor_scales_s: Sequence[float] = (0.01, 0.025, 0.05, 0.1),
        propagation_gabor_scales_s: Sequence[float] = (0.01, 0.025, 0.05, 0.1),
        ray_samples: int = 12,
        domain_x_m: float = 2000.0,
        domain_z_m: float = 2000.0,
        domain_t_s: float = 1.0,
    ) -> None:
        time_grid = SavedTimeGrid.from_values(saved_time_s)
        super().__init__(
            width=width,
            rank=rank,
            spectral_rank=spectral_rank,
            modes=tuple(modes),
            dense_modes=(1,),
            dense_time_block=dense_time_block,
            heads=heads,
            token_grid=token_grid,
            position_bands=position_bands,
            fourier_bands=fourier_bands,
            gabor_scales_s=gabor_scales_s,
            ray_samples=ray_samples,
            domain_x_m=domain_x_m,
            domain_z_m=domain_z_m,
            domain_t_s=domain_t_s,
        )
        self.saved_time_grid = time_grid
        self.register_buffer(
            "saved_time_values_s", time_grid.values_s.clone(), persistent=True
        )
        self.dense_decoder = PropagationConditionedDenseDecoder(
            width=width,
            spectral_rank=dense_spectral_rank,
            pyramid_levels=len(modes),
            modes=dense_modes,
            depth=dense_depth,
            saved_time_count=time_grid.count,
            time_block=dense_time_block,
            domain_t_s=domain_t_s,
            domain_diagonal_m=math.sqrt(domain_x_m**2 + domain_z_m**2),
            use_local_phase=use_local_phase,
            activation_checkpointing=activation_checkpointing,
            coupled_axes=bool(dense_coupled_axes),
            local_differential_residual=bool(dense_local_differential_residual),
            coupled_2d_rank=int(dense_coupled_2d_rank),
            temporal_basis_rank=int(dense_temporal_basis_rank),
            family_expert_rank=int(dense_family_expert_rank),
            band_adapter_rank=int(dense_band_adapter_rank),
            band_adapter_architecture=str(dense_band_adapter_architecture),
            band_adapter_spectral_rank=int(dense_band_adapter_spectral_rank),
            band_adapter_modes=int(dense_band_adapter_modes),
            band_adapter_full_depth=int(dense_band_adapter_full_depth),
            band_adapter_coarse_depth=int(dense_band_adapter_coarse_depth),
            band_adapter_activation_checkpointing=(
                dense_band_adapter_activation_checkpointing
            ),
            band_adapter_dropout=float(dense_band_adapter_dropout),
            gabor_scales_s=propagation_gabor_scales_s,
            high_frequency_residual=bool(dense_high_frequency_residual),
            high_frequency_hidden=int(dense_high_frequency_hidden),
            high_frequency_depth=int(dense_high_frequency_depth),
        )
        if not isinstance(dense_band_adapter_preserve_high_band, bool):
            raise ValueError("band adapter high-band preservation must be boolean")
        self.dense_band_adapter_preserve_high_band = bool(
            dense_band_adapter_preserve_high_band
        )
        self.local_field = (
            None
            if not dense_local_field
            else LocalPropagationFieldGenerator(
                width=width,
                pyramid_levels=len(modes),
                saved_time_count=time_grid.count,
                domain_t_s=domain_t_s,
                domain_diagonal_m=math.sqrt(domain_x_m**2 + domain_z_m**2),
                domain_x_m=domain_x_m,
                domain_z_m=domain_z_m,
                gabor_scales_s=propagation_gabor_scales_s,
                channel_multipliers=tuple(dense_local_field_channel_multipliers),
                causal_width_s=float(dense_local_field_causal_width_s),
                residual=bool(dense_local_field_residual),
                activation_checkpointing=activation_checkpointing,
                extended_late_features=bool(dense_local_field_extended_late_features),
                temporal_operator_rank=int(dense_local_field_temporal_operator_rank),
                temporal_operator_spatial_kernel=int(
                    dense_local_field_temporal_operator_spatial_kernel
                ),
                warp=bool(dense_local_field_warp),
                warp_max_shift_cells=float(dense_local_field_warp_max_shift_cells),
                warp_shift_dilation=int(dense_local_field_warp_shift_dilation),
                green_kernel=bool(dense_local_field_green_kernel),
                green_kernel_size=int(dense_local_field_green_kernel_size),
                green_dilations=tuple(dense_local_field_green_dilations),
                temporal_latent_basis=bool(dense_local_field_temporal_latent_basis),
                temporal_latent_rank=int(dense_local_field_temporal_latent_rank),
                temporal_latent_harmonics=int(dense_local_field_temporal_latent_harmonics),
                multi_arrival=bool(dense_local_field_multi_arrival),
                multi_arrival_paths=int(dense_local_field_multi_arrival_paths),
                multi_arrival_max_shift_cells=float(
                    dense_local_field_multi_arrival_max_shift_cells
                ),
                multi_arrival_max_delay_frac=float(
                    dense_local_field_multi_arrival_max_delay_frac
                ),
                dispersive_modal=bool(dense_local_field_dispersive_modal),
                dispersive_modal_modes=int(dense_local_field_dispersive_modal_modes),
                dispersive_modal_max_frequency=float(
                    dense_local_field_dispersive_modal_max_frequency
                ),
                windowed_propagation=bool(dense_local_field_windowed_propagation),
                windowed_propagation_window=int(
                    dense_local_field_windowed_propagation_window
                ),
                windowed_propagation_stride=int(
                    dense_local_field_windowed_propagation_stride
                ),
                windowed_propagation_rank=int(
                    dense_local_field_windowed_propagation_rank
                ),
                windowed_propagation_max_advect_cells=float(
                    dense_local_field_windowed_propagation_max_advect_cells
                ),
                helmholtz_synthesis=bool(dense_local_field_helmholtz_synthesis),
                helmholtz_synthesis_frequencies=int(
                    dense_local_field_helmholtz_synthesis_frequencies
                ),
                helmholtz_synthesis_wkb_phase=bool(
                    dense_local_field_helmholtz_synthesis_wkb_phase
                ),
                helmholtz_synthesis_rank=int(
                    dense_local_field_helmholtz_synthesis_rank
                ),
                helmholtz_synthesis_late_rank=int(
                    dense_local_field_helmholtz_synthesis_late_rank
                ),
                helmholtz_synthesis_late_frequencies=int(
                    dense_local_field_helmholtz_synthesis_late_frequencies
                ),
                helmholtz_synthesis_frequency_softmax=bool(
                    dense_local_field_helmholtz_synthesis_frequency_softmax
                ),
                helmholtz_synthesis_source_onset_phase=bool(
                    dense_local_field_helmholtz_synthesis_source_onset_phase
                ),
                helmholtz_source_relative_coordinates=bool(
                    dense_local_field_helmholtz_source_relative_coordinates
                ),
                helmholtz_spectral_bypass=bool(
                    dense_local_field_helmholtz_spectral_bypass
                ),
                helmholtz_spectral_bypass_per_branch=bool(
                    dense_local_field_helmholtz_spectral_bypass_per_branch
                ),
                helmholtz_background_conditioning=bool(
                    dense_local_field_helmholtz_background_conditioning
                ),
                helmholtz_background_sigma_cells=float(
                    dense_local_field_helmholtz_background_sigma_cells
                ),
                helmholtz_background_global_propagator=bool(
                    dense_local_field_helmholtz_background_global_propagator
                ),
                helmholtz_background_propagation_modes=int(
                    dense_local_field_helmholtz_background_propagation_modes
                ),
                helmholtz_background_direct_frequency_head=bool(
                    dense_local_field_helmholtz_background_direct_frequency_head
                ),
                helmholtz_background_direct_frequencies=int(
                    dense_local_field_helmholtz_background_direct_frequencies
                ),
                helmholtz_background_direct_spectral_experts=int(
                    dense_local_field_helmholtz_background_direct_spectral_experts
                ),
                adapter_gate_init=float(dense_local_field_adapter_gate_init),
            )
        )
        self.dense_local_field_residual = bool(dense_local_field_residual)
        # Historical MIONet outputs are softly projected to the free-surface
        # boundary with a 20 m taper.  A complete-field frequency head can instead
        # be supervised on the already boundary-consistent physical waveform; in
        # that case a second taper is a deterministic train/inference mismatch.
        # Keep the historical default and let the leakage-safe direct-coefficient
        # route disable it explicitly at runtime (no checkpoint-schema change).
        self.dense_apply_free_surface_factor = True
        if self.dense_local_field_residual and self.local_field is None:
            raise ValueError("dense_local_field_residual requires dense_local_field")

    def _expand_dense_times(
        self, prepared: PreparedV3State, time_s: torch.Tensor
    ) -> torch.Tensor:
        times = super()._expand_dense_times(prepared, time_s)
        self.saved_time_grid.indices(times)
        return times

    def _prepare_background_conditioning_features(
        self,
        prepared: PreparedV3State,
        dense_grid,
        background_normalized: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Build the query-independent Born features once per dense call."""

        conditioner = getattr(
            self.local_field, "helmholtz_background_conditioner", None
        )
        if conditioner is None:
            return None
        if background_normalized is None:
            raise ValueError(
                "Helmholtz background conditioning requires complete normalized P_bg"
            )
        height, width = int(dense_grid.height), int(dense_grid.width)
        arrival = torch.as_tensor(
            dense_grid.travel.seconds,
            dtype=torch.float32,
            device=prepared.source_parameters.device,
        ).reshape(prepared.source_parameters.shape[0], height, width)
        omega = self.local_field.helmholtz_synthesis.frequency_bank(
            self.saved_time_values_s
        )
        return conditioner(
            background_normalized,
            prepared.medium.velocity_mps,
            prepared.record_to_medium,
            arrival,
            omega,
        )

    def _dense_block_with_coarse(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        dense_grid,
        *,
        apply_correction: bool,
        background_normalized: torch.Tensor | None = None,
        background_conditioning_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        corrected, coarse, _ = self._dense_block_with_coarse_and_routing(
            prepared,
            time_s,
            dense_grid,
            apply_correction=apply_correction,
            background_normalized=background_normalized,
            background_conditioning_features=background_conditioning_features,
        )
        return corrected, coarse

    def _mionet_coarse(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        dense_grid,
        *,
        records: int,
        count: int,
        points: int,
    ) -> torch.Tensor:
        """Global low-rank MIONet coarse field (coordinate/travel/fusion path)."""

        xy = dense_grid.coords_xy_m[:, None].expand(-1, count, -1, -1)
        time_values = time_s[:, :, None].expand(-1, -1, points)
        coords = torch.cat((xy, time_values[..., None]), dim=-1).reshape(
            records, count * points, 3
        )
        travel = self._repeat_travel(dense_grid.travel, count)
        trunk_rank, bundle = self.coordinate_encoder(
            coords,
            prepared.source_parameters,
            travel,
            domain_x_m=self.domain_x_m,
            domain_z_m=self.domain_z_m,
            domain_t_s=self.domain_t_s,
        )
        travel_rank = self.travel_branch(
            travel,
            bundle,
            domain_t_s=self.domain_t_s,
            domain_diagonal_m=math.sqrt(self.domain_x_m**2 + self.domain_z_m**2),
        )
        normalized_xy = dense_grid.coords_xy_normalized[:, None].expand(
            -1, count, -1, -1
        )
        fused = self.fusion(
            prepared.medium.encoding,
            prepared.source_encoding,
            normalized_xy.reshape(records, count * points, 2),
            prepared.record_to_medium,
            travel_rank,
            trunk_rank,
        )
        return fused.normalized_pressure.reshape(
            records, count, dense_grid.height, dense_grid.width
        )

    def _dense_block_with_anchor_increment_and_routing(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        dense_grid,
        *,
        apply_correction: bool,
        route_override: torch.Tensor | None = None,
        background_normalized: torch.Tensor | None = None,
        background_conditioning_features: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        records, count = time_s.shape
        points = dense_grid.height * dense_grid.width
        saved_time_indices = self.saved_time_grid.indices(time_s)
        if self.local_field is not None and not self.dense_local_field_residual:
            coarse = self.local_field(
                prepared.medium.velocity_mps,
                prepared.medium.encoding,
                prepared.source_encoding,
                prepared.source_parameters,
                prepared.record_to_medium,
                time_s,
                dense_grid.travel,
                saved_time_indices,
                self.saved_time_values_s,
                background_normalized,
                background_conditioning_features,
            )
        else:
            coarse = self._mionet_coarse(
                prepared, time_s, dense_grid, records=records, count=count, points=points
            )
            if self.local_field is not None:
                # Residual mode: a zero-initialised local-propagation field adds
                # local, translation-equivariant structure on top of the global
                # MIONet coarse. At init it contributes nothing (zero output conv),
                # so training starts on the global-coarse trajectory rather than
                # paying the from-scratch warm-up the replace variant pays.
                coarse = coarse + self.local_field(
                    prepared.medium.velocity_mps,
                    prepared.medium.encoding,
                    prepared.source_encoding,
                    prepared.source_parameters,
                    prepared.record_to_medium,
                    time_s,
                    dense_grid.travel,
                    saved_time_indices,
                    self.saved_time_values_s,
                    background_normalized,
                    background_conditioning_features,
                )
        if apply_correction:
            anchor, raw_increment, router_logits = (
                self.dense_decoder.forward_with_anchor_increment_and_routing(
                    prepared.medium.velocity_mps,
                    prepared.medium.encoding,
                    prepared.source_encoding,
                    prepared.source_parameters,
                    prepared.source_normalized,
                    prepared.record_to_medium,
                    time_s,
                    coarse,
                    dense_grid.travel,
                    saved_time_indices,
                    route_override=route_override,
                )
            )
        else:
            anchor = coarse
            raw_increment = coarse.new_zeros(coarse.shape)
            router_logits = coarse.new_empty((0, 3))
        surface = (
            free_surface_factor(dense_grid.z_m)[None, None, :, None]
            if bool(self.dense_apply_free_surface_factor)
            else coarse.new_ones((1, 1, dense_grid.height, 1))
        )
        anchor_surface = anchor * surface
        increment_surface = raw_increment * surface
        projected_increment = (
            increment_surface
            if self.dense_decoder.band_limited_adapter is None
            or not self.dense_band_adapter_preserve_high_band
            else project_low_mid_increment(increment_surface)
        )
        if not self.dense_band_adapter_preserve_high_band:
            projected_increment = projected_increment.clone()
            projected_increment[..., 0, :] = 0.0
        return (
            anchor_surface + projected_increment,
            coarse * surface,
            anchor_surface,
            projected_increment,
            router_logits,
        )

    def _dense_block_with_coarse_and_routing(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        dense_grid,
        *,
        apply_correction: bool,
        route_override: torch.Tensor | None = None,
        background_normalized: torch.Tensor | None = None,
        background_conditioning_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        corrected, coarse, _, _, router_logits = (
            self._dense_block_with_anchor_increment_and_routing(
                prepared,
                time_s,
                dense_grid,
                apply_correction=apply_correction,
                route_override=route_override,
                background_normalized=background_normalized,
                background_conditioning_features=background_conditioning_features,
            )
        )
        return corrected, coarse, router_logits

    def _dense_block(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        dense_grid,
        *,
        apply_correction: bool,
        background_normalized: torch.Tensor | None = None,
        background_conditioning_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        corrected, _ = self._dense_block_with_coarse(
            prepared,
            time_s,
            dense_grid,
            apply_correction=apply_correction,
            background_normalized=background_normalized,
            background_conditioning_features=background_conditioning_features,
        )
        return corrected

    def dense_normalized_with_coarse(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        *,
        x_m: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        dense_grid=None,
        time_block: int | None = None,
        background_normalized: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return corrected and transferred coarse fields without duplicate encoding."""

        corrected, coarse, _ = self.dense_normalized_with_coarse_and_routing(
            prepared,
            time_s,
            x_m=x_m,
            z_m=z_m,
            dense_grid=dense_grid,
            time_block=time_block,
            background_normalized=background_normalized,
        )
        return corrected, coarse

    def dense_normalized(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        *,
        x_m: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        dense_grid=None,
        time_block: int | None = None,
        apply_correction: bool = True,
        background_normalized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense prediction with optional complete-axis ``P_bg`` conditioning."""

        times = self._expand_dense_times(prepared, time_s)
        if dense_grid is None:
            if x_m is None or z_m is None:
                raise ValueError("x_m and z_m are required when dense_grid is not supplied")
            dense_grid = self.prepare_dense_grid(prepared, x_m=x_m, z_m=z_m)
        background_features = self._prepare_background_conditioning_features(
            prepared, dense_grid, background_normalized
        )
        block = self.dense_decoder.time_block if time_block is None else int(time_block)
        if block <= 0:
            raise ValueError("time_block must be positive")
        return torch.cat(
            [
                self._dense_block(
                    prepared,
                    times[:, start : start + block],
                    dense_grid,
                    apply_correction=apply_correction,
                    background_normalized=background_normalized,
                    background_conditioning_features=background_features,
                )
                for start in range(0, times.shape[1], block)
            ],
            dim=1,
        )

    def dense_normalized_with_coarse_and_routing(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        *,
        x_m: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        dense_grid=None,
        time_block: int | None = None,
        route_override: torch.Tensor | None = None,
        background_normalized: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full fields and one velocity-derived routing row per medium."""

        times = self._expand_dense_times(prepared, time_s)
        if dense_grid is None:
            if x_m is None or z_m is None:
                raise ValueError("x_m and z_m are required when dense_grid is not supplied")
            dense_grid = self.prepare_dense_grid(prepared, x_m=x_m, z_m=z_m)
        background_features = self._prepare_background_conditioning_features(
            prepared, dense_grid, background_normalized
        )
        block = self.dense_decoder.time_block if time_block is None else int(time_block)
        if block <= 0:
            raise ValueError("time_block must be positive")
        corrected_blocks: list[torch.Tensor] = []
        coarse_blocks: list[torch.Tensor] = []
        router_logits: torch.Tensor | None = None
        for start in range(0, times.shape[1], block):
            corrected, coarse, block_router_logits = (
                self._dense_block_with_coarse_and_routing(
                prepared,
                times[:, start : start + block],
                    dense_grid,
                    apply_correction=True,
                    route_override=route_override,
                    background_normalized=background_normalized,
                    background_conditioning_features=background_features,
                )
            )
            corrected_blocks.append(corrected)
            coarse_blocks.append(coarse)
            if router_logits is None:
                router_logits = block_router_logits
        if router_logits is None:
            router_logits = times.new_empty((0, 3))
        return (
            torch.cat(corrected_blocks, dim=1),
            torch.cat(coarse_blocks, dim=1),
            router_logits,
        )

    def dense_normalized_with_anchor_increment_and_routing(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        *,
        x_m: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        dense_grid=None,
        time_block: int | None = None,
        route_override: torch.Tensor | None = None,
        background_normalized: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the high-band anchor and final projected adapter increment."""

        times = self._expand_dense_times(prepared, time_s)
        if dense_grid is None:
            if x_m is None or z_m is None:
                raise ValueError("x_m and z_m are required when dense_grid is not supplied")
            dense_grid = self.prepare_dense_grid(prepared, x_m=x_m, z_m=z_m)
        background_features = self._prepare_background_conditioning_features(
            prepared, dense_grid, background_normalized
        )
        block = self.dense_decoder.time_block if time_block is None else int(time_block)
        if block <= 0:
            raise ValueError("time_block must be positive")
        anchor_blocks: list[torch.Tensor] = []
        increment_blocks: list[torch.Tensor] = []
        router_logits: torch.Tensor | None = None
        for start in range(0, times.shape[1], block):
            _, _, anchor, increment, block_router_logits = (
                self._dense_block_with_anchor_increment_and_routing(
                    prepared,
                    times[:, start : start + block],
                    dense_grid,
                    apply_correction=True,
                    route_override=route_override,
                    background_normalized=background_normalized,
                    background_conditioning_features=background_features,
                )
            )
            anchor_blocks.append(anchor)
            increment_blocks.append(increment)
            if router_logits is None:
                router_logits = block_router_logits
        if router_logits is None:
            router_logits = times.new_empty((0, 3))
        return (
            torch.cat(anchor_blocks, dim=1),
            torch.cat(increment_blocks, dim=1),
            router_logits,
        )


def load_v3_backbone(
    model: SavedTimePhaseOperatorV4,
    checkpoint: str | Path,
) -> dict[str, int]:
    """Load every shape-compatible non-dense V3 tensor into V4."""

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError("V3 transfer checkpoint is missing model_state")
    state = {
        key: value
        for key, value in payload["model_state"].items()
        if not key.startswith("dense_decoder.")
    }
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = tuple(incompatible.unexpected_keys)
    forbidden_missing = tuple(
        key
        for key in incompatible.missing_keys
        if not key.startswith("dense_decoder.") and key != "saved_time_values_s"
    )
    if unexpected or forbidden_missing:
        raise ValueError(
            f"invalid V3 transfer: missing={forbidden_missing}, unexpected={unexpected}"
        )
    return {
        "loaded_parameter_tensors": len(state),
        "new_dense_parameter_tensors": sum(
            1 for key in incompatible.missing_keys if key.startswith("dense_decoder.")
        ),
    }


__all__ = ["SavedTimePhaseOperatorV4", "load_v3_backbone"]
