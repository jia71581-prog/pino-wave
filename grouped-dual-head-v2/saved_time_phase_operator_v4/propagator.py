"""B2 -- recursive causal semigroup propagator (learned neural time-stepper).

Motivation (falsification-driven, see cc_research_supervisor/NEXT_ACTION.md R8+21/R8+22):
the pure query-invariant operator renders every saved-time frame INDEPENDENTLY from
(medium, source, t).  Its warp_r1 error is spread across ALL time bins
(early=0.119, middle=0.203, late=0.346) and the energy-weighted arithmetic proves the
hard goal aggregate_relative_l2 < 0.05 needs a ~uniform <0.05 across early+middle+late
(a 5-7x reduction in EVERY bin).  No additive LATE-window corrector (A3/A5/R5/B1/A4)
can do that -- its best conceivable aggregate is ~0.14.

This module trades the per-frame query-invariance contract for temporal RECURRENCE:
it learns a SHARED (time-invariant) step operator S_theta advancing a latent wavefield
state by one saved-time increment,

    h_{k+1} = h_k + step_scale * S_theta(h_k, cond),     p_k = gate * decode(h_k),

and rolls it out K steps from a single encode of the initial condition + static medium
conditioning.  Weight sharing across steps is the semigroup / time-invariance inductive
bias a per-frame regressor lacks; because accuracy propagates FORWARD from the early
frames, the propagator can reduce error across ALL time bins (the R8+22 requirement),
not just the late window.

Contract properties (pinned by tests/saved_time_phase_operator_v4/test_propagator.py):
  * CAUSAL by construction -- h_{k+1} depends only on h_k (past) and the static cond,
    never on future frames.  Hence rollout(K)[:, :j] == rollout(j) exactly (Markov).
  * SEMIGROUP -- the step operator is one shared module, so composing it twice equals a
    2-step rollout (S_theta^2 = rollout of length 2).
  * GATE-0 WARM-START NO-OP -- with gate_init=0 the emitted correction is exactly zero,
    so a warp_r1 warm-start  field = base + propagator  is bit-exact at initialization
    and the deadlock-free positive gate_init (0.03) opens it gradually, exactly as the
    A5 adapter does.
  * Bounded rollout -- step_scale is a learned scalar initialised small (0.1) and the
    inner FactorizedComplexResidualStack is pre-normalised, so a K~20 rollout stays
    finite without explosion.

The module is deliberately activation-checkpointed PER STEP (use_reentrant=False) so a
K-step rollout holds O(1) step activations rather than O(K); this is the memory control
that lets the propagator fit the <23.5 GiB budget over the warp base.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .boundary_consistency import project_pressure_free_surface
from .dg_interface import DGInterfaceFluxResidual2d, acoustic_interface_reflection_maps
from .spectral import FactorizedComplexResidualStack


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class CausalSemigroupPropagator(nn.Module):
    """Learned causal semigroup wavefield propagator (B2).

    forward(initial_state, cond, steps) -> pressure sequence [B, steps, 1, Z, X].

    Args:
        state_channels: channels of the initial-condition tensor (e.g. 2 = p_0, p_{-1},
            enough to make a 2nd-order wave state Markovian).
        cond_channels: channels of the static conditioning map (medium / source / geometry
            features), injected additively into the latent at every step.
        width: latent state width.
        spectral_rank / modes / depth: inner shared FNO step-stack capacity.
        gate_init: output gate initial value.  0.0 => exact warm-start no-op.
        activation_checkpointing: per-step recomputation (O(1) rollout activation memory).
    """

    def __init__(
        self,
        *,
        state_channels: int,
        cond_channels: int,
        width: int = 64,
        spectral_rank: int = 32,
        modes: int = 24,
        depth: int = 4,
        gate_init: float = 0.0,
        step_scale_init: float = 0.1,
        activation_checkpointing: bool = True,
        boundary_halo_radius: int = 0,
        hard_free_surface: bool = False,
        dg_interface_rank: int = 0,
        dg_cpml_margin: int = 20,
    ) -> None:
        super().__init__()
        if state_channels <= 0 or cond_channels <= 0 or width <= 0:
            raise ValueError("propagator channel/width dims must be positive")
        self.width = int(width)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.hard_free_surface = bool(hard_free_surface)
        self.dg_cpml_margin = int(dg_cpml_margin)

        # initial-condition encoder: (IC + cond) -> latent h0
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels + cond_channels, width, kernel_size=3, padding=1),
            _group_norm(width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        # static conditioning injected each step (medium is time-invariant)
        self.cond_projection = nn.Conv2d(cond_channels, width, kernel_size=1)
        # per-step warp-anchor projection (anchored mode only; see forward(base_seq=...))
        self.base_projection = nn.Conv2d(1, width, kernel_size=1)
        # SHARED time-invariant step operator (the semigroup generator)
        self.step_stack = FactorizedComplexResidualStack(
            width=int(width),
            spectral_rank=int(spectral_rank),
            modes=int(modes),
            depth=int(depth),
            activation_checkpointing=bool(activation_checkpointing),
            boundary_halo_radius=int(boundary_halo_radius),
        )
        self.step_norm = _group_norm(width)
        self.dg_interface = (
            DGInterfaceFluxResidual2d(width, rank=int(dg_interface_rank))
            if int(dg_interface_rank) > 0
            else None
        )
        # latent -> emitted pressure frame
        self.decoder = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 1, kernel_size=1),
        )
        # zero-init gate => exact no-op warm-start; step_scale bounds each increment
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self.step_scale = nn.Parameter(torch.tensor(float(step_scale_init), dtype=torch.float32))

    def _apply_output_boundary(self, field: torch.Tensor) -> torch.Tensor:
        if not self.hard_free_surface:
            return field
        return project_pressure_free_surface(field)

    def _interface_maps(self, cond: torch.Tensor) -> torch.Tensor | None:
        if self.dg_interface is None:
            return None
        return acoustic_interface_reflection_maps(
            cond, cpml_margin=self.dg_cpml_margin
        )

    def _step(
        self,
        hidden: torch.Tensor,
        cond_latent: torch.Tensor,
        interface_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One time-invariant increment h_k -> h_{k+1} (shared weights)."""
        driven = self.step_norm(hidden + cond_latent)
        update = self.step_stack(driven)
        if self.dg_interface is not None:
            if interface_maps is None:
                raise ValueError("DG interface maps are required by the active flux block")
            update = update + self.dg_interface(driven, interface_maps)
        return hidden + self.step_scale * update

    def _step_anchored(
        self,
        hidden: torch.Tensor,
        cond_latent: torch.Tensor,
        base_latent: torch.Tensor,
        interface_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Anchored increment: the current warp render is injected as an extra drive."""
        driven = self.step_norm(hidden + cond_latent + base_latent)
        update = self.step_stack(driven)
        if self.dg_interface is not None:
            if interface_maps is None:
                raise ValueError("DG interface maps are required by the active flux block")
            update = update + self.dg_interface(driven, interface_maps)
        return hidden + self.step_scale * update

    def forward(
        self,
        initial_state: torch.Tensor,
        cond: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """FREE-RUNNING rollout `steps` increments (pure propagator, no anchor).

        initial_state: [B, state_channels, Z, X]
        cond:          [B, cond_channels, Z, X]
        returns:       [B, steps, 1, Z, X] emitted pressure frames p_0..p_{steps-1}

        NOTE: over long horizons a free-running rollout can accumulate error; the
        drift-free training path is `forward_anchored` (every frame anchored to the
        warp render).  This free-running forward is kept as the pure-semigroup
        reference and for the contract tests.
        """
        if initial_state.ndim != 4 or cond.ndim != 4:
            raise ValueError("initial_state and cond must be [B,C,Z,X]")
        if steps <= 0:
            raise ValueError("steps must be positive")
        hidden = self.encoder(torch.cat((initial_state, cond), dim=1))
        cond_latent = self.cond_projection(cond)
        interface_maps = self._interface_maps(cond)
        frames = []
        for _ in range(int(steps)):
            frames.append(self._apply_output_boundary(self.gate * self.decoder(hidden)))
            if self.activation_checkpointing and self.training and hidden.requires_grad:
                if interface_maps is None:
                    hidden = checkpoint(
                        self._step, hidden, cond_latent, use_reentrant=False
                    )
                else:
                    hidden = checkpoint(
                        self._step,
                        hidden,
                        cond_latent,
                        interface_maps,
                        use_reentrant=False,
                    )
            else:
                hidden = self._step(hidden, cond_latent, interface_maps)
        return torch.stack(frames, dim=1)

    def forward_anchored(
        self,
        base_seq: torch.Tensor,
        cond: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """WARP-ANCHORED rollout (the drift-free TRAINING path).

        Every emitted frame is anchored to the frozen warp render, so there is no
        long-horizon free-running drift; a recurrent latent state carries the
        cross-frame temporal consistency the per-frame render lacks:

            p_k = base_k + gate * decode(h_k),   h_{k+1} = h_k + step(h_k, cond, base_k).

        Causal: p_k depends only on base_0..base_k (past + current), never future.
        gate == 0 => p_k == base_k exactly (bit-exact warp warm-start no-op).

        base_seq: [B, K, 1, Z, X] frozen warp renders on the saved-time grid.
        cond:     [B, cond_channels, Z, X] static medium/source/geometry features.
        initial_state: [B, state_channels, Z, X] IC for h_0; defaults to the first
            base frame broadcast to state_channels.
        returns:  [B, K, 1, Z, X] corrected pressure sequence.
        """
        if base_seq.ndim != 5 or base_seq.shape[2] != 1:
            raise ValueError("base_seq must be [B, K, 1, Z, X]")
        if cond.ndim != 4:
            raise ValueError("cond must be [B, cond_channels, Z, X]")
        b, k, _, z, x = base_seq.shape
        if initial_state is None:
            initial_state = base_seq[:, 0].expand(b, self.encoder[0].in_channels - cond.shape[1], z, x)
        hidden = self.encoder(torch.cat((initial_state, cond), dim=1))
        cond_latent = self.cond_projection(cond)
        interface_maps = self._interface_maps(cond)
        frames = []
        for j in range(int(k)):
            base_j = base_seq[:, j]
            frames.append(
                self._apply_output_boundary(base_j + self.gate * self.decoder(hidden))
            )
            base_latent = self.base_projection(base_j)
            if self.activation_checkpointing and self.training and hidden.requires_grad:
                if interface_maps is None:
                    hidden = checkpoint(
                        self._step_anchored,
                        hidden,
                        cond_latent,
                        base_latent,
                        use_reentrant=False,
                    )
                else:
                    hidden = checkpoint(
                        self._step_anchored,
                        hidden,
                        cond_latent,
                        base_latent,
                        interface_maps,
                        use_reentrant=False,
                    )
            else:
                hidden = self._step_anchored(
                    hidden, cond_latent, base_latent, interface_maps
                )
        return torch.stack(frames, dim=1)

    @torch.no_grad()
    def is_warmstart_noop(self) -> bool:
        """True iff the emitted correction is identically zero (gate == 0)."""
        return bool(self.gate.abs().item() == 0.0)
