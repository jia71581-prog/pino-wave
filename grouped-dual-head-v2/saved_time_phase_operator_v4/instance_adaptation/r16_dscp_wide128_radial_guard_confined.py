"""Production-faithful v18 Wide128 DSCP with a radial high-band guard.

This candidate preserves the v18 deployment contract in two independent ways:

1. the sixteen predicted coefficient maps are projected onto the registered
   low/middle 2-D radial spatial-frequency subspace before materialization;
2. the materialized correction is confined by the frozen v18 parent-energy
   keep mask in addition to the C1 causal ramp and free-surface boundary.

All deployment inputs remain label-free.  The parent-energy mask consumes only
the frozen parent prediction and therefore does not open future target fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from saved_time_phase_operator_v4.band_adapter import project_low_mid_increment

from .r16_dscp import (
    CONDITION_MAXIMUM,
    DSCPContractError,
    HEIGHT,
    INPUT_CHANNELS,
    RANK,
    TIME_COUNT,
    WIDTH,
    RouteDecision,
    c1_causal_mask,
    deployment_features,
    predictor_parameter_count,
)
from .r16_dscp_training_v3 import (
    TAU,
    apply_confined_correction,
    full_time_keep_mask,
)


WIDE128_PARAMETERS = 25_266


class Wide128CoefficientHead(nn.Module):
    """Exact topology and state-dict keys used by the v18 Wide128Head."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                INPUT_CHANNELS,
                INPUT_CHANNELS,
                3,
                padding=1,
                groups=INPUT_CHANNELS,
                bias=True,
            ),
            nn.Conv2d(INPUT_CHANNELS, 128, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=True),
            nn.Conv2d(128, 128, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=True),
            nn.Conv2d(128, RANK, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        if predictor_parameter_count(self) != WIDE128_PARAMETERS:
            raise DSCPContractError("wide-128 head parameter count is not 25,266")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(features)
        if value.ndim != 4 or value.shape[1] != INPUT_CHANNELS:
            raise ValueError("features must have shape [batch,29,height,width]")
        return self.body(value).tanh()


class R16DSCPWide128RadialGuardConfined(nn.Module):
    """Deployable v18 Wide128 DSCP with spectral and temporal confinement."""

    def __init__(self, bases: torch.Tensor, coefficient_scales: torch.Tensor) -> None:
        super().__init__()
        basis_value = torch.as_tensor(bases, dtype=torch.float32).contiguous()
        scale_value = torch.as_tensor(coefficient_scales, dtype=torch.float32).contiguous()
        if basis_value.shape != (3, TIME_COUNT, RANK):
            raise ValueError("bases must have shape [3,401,16]")
        if scale_value.shape != (3, RANK) or bool((scale_value <= 0).any()):
            raise ValueError("coefficient_scales must be positive [3,16]")
        self.register_buffer("bases", basis_value, persistent=True)
        self.register_buffer("coefficient_scales", scale_value, persistent=True)
        self.head = Wide128CoefficientHead()
        if predictor_parameter_count(self) != WIDE128_PARAMETERS:
            raise DSCPContractError("guarded predictor parameter count is not 25,266")

    def load_v18_head_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load the unprefixed ``body.*`` state stored by v18 checkpoints."""

        self.head.load_state_dict(dict(state_dict), strict=True)

    @classmethod
    def from_v18_checkpoint(
        cls,
        bases: torch.Tensor,
        coefficient_scales: torch.Tensor,
        checkpoint: str | Path | Mapping[str, Any],
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["R16DSCPWide128RadialGuardConfined", Mapping[str, Any]]:
        payload: Mapping[str, Any]
        if isinstance(checkpoint, Mapping):
            payload = checkpoint
        else:
            payload = torch.load(
                Path(checkpoint), map_location=map_location, weights_only=False
            )
        if "model_state" not in payload or not isinstance(payload["model_state"], Mapping):
            raise DSCPContractError("v18 checkpoint lacks model_state")
        model = cls(bases, coefficient_scales)
        model.load_v18_head_state_dict(payload["model_state"])
        return model, payload

    def raw_coefficients_from_features(
        self,
        features: torch.Tensor,
        decisions: tuple[RouteDecision, ...],
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        unit = self.head(features)
        coefficients = torch.zeros_like(unit)
        for index, decision in enumerate(decisions):
            if not decision.abstain and float(conditions[index]) <= CONDITION_MAXIMUM:
                coefficients[index] = (
                    unit[index]
                    * self.coefficient_scales[decision.index, :, None, None].to(unit)
                )
        if not bool(torch.isfinite(coefficients).all()):
            raise FloatingPointError("raw wide-128 coefficients contain NaN or Inf")
        return coefficients

    def predict_coefficients(
        self,
        velocity_mps: torch.Tensor,
        source_map: torch.Tensor,
        travel_time_map: torch.Tensor,
        x_m: torch.Tensor,
        z_m: torch.Tensor,
        observed_k0: torch.Tensor,
        observed_k1: torch.Tensor,
        parent_wavefield: torch.Tensor,
        k0: torch.Tensor,
        k1: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[RouteDecision, ...], torch.Tensor]:
        features, decisions, conditions = deployment_features(
            velocity_mps,
            source_map,
            travel_time_map,
            x_m,
            z_m,
            observed_k0,
            observed_k1,
            parent_wavefield,
            self.bases,
            k0,
            k1,
        )
        raw = self.raw_coefficients_from_features(features, decisions, conditions)
        guarded = project_low_mid_increment(raw).to(dtype=raw.dtype)
        if guarded.shape != raw.shape or not bool(torch.isfinite(guarded).all()):
            raise FloatingPointError("guarded wide-128 coefficients are invalid")
        return guarded, decisions, conditions

    def forward(
        self,
        velocity_mps: torch.Tensor,
        source_map: torch.Tensor,
        travel_time_map: torch.Tensor,
        x_m: torch.Tensor,
        z_m: torch.Tensor,
        observed_k0: torch.Tensor,
        observed_k1: torch.Tensor,
        parent_wavefield: torch.Tensor,
        k0: torch.Tensor,
        k1: torch.Tensor,
    ) -> torch.Tensor:
        coefficients, decisions, conditions = self.predict_coefficients(
            velocity_mps,
            source_map,
            travel_time_map,
            x_m,
            z_m,
            observed_k0,
            observed_k1,
            parent_wavefield,
            k0,
            k1,
        )
        parent = torch.as_tensor(parent_wavefield)
        if parent.ndim != 4 or parent.shape[1:] != (TIME_COUNT, HEIGHT, WIDTH):
            raise ValueError("parent_wavefield must have shape [batch,401,201,201]")
        onset = torch.as_tensor(k1, dtype=torch.long, device=parent.device).flatten()
        if onset.numel() != parent.shape[0]:
            raise ValueError("k1 must provide one onset index per record")

        corrected_records: list[torch.Tensor] = []
        for index, decision in enumerate(decisions):
            if decision.abstain or float(conditions[index]) > CONDITION_MAXIMUM:
                corrected_records.append(parent[index])
                continue
            basis = self.bases[decision.index].to(parent)
            correction = torch.einsum("tr,rhw->thw", basis, coefficients[index].to(parent))
            correction = correction * c1_causal_mask(
                TIME_COUNT,
                int(onset[index]),
                device=parent.device,
                dtype=parent.dtype,
            )[:, None, None]
            correction = torch.cat(
                (torch.zeros_like(correction[:, :1]), correction[:, 1:]), dim=1
            )
            keep_mask, _ = full_time_keep_mask(
                parent[index], k1=int(onset[index]), tau=TAU
            )
            corrected_records.append(
                apply_confined_correction(parent[index], correction, keep_mask)
            )
        corrected = torch.stack(corrected_records, dim=0)
        if not bool(torch.isfinite(corrected).all()):
            raise FloatingPointError("guarded corrected output contains NaN or Inf")
        return corrected


__all__ = [
    "R16DSCPWide128RadialGuardConfined",
    "WIDE128_PARAMETERS",
    "Wide128CoefficientHead",
]
