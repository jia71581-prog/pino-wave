"""B2-H -- forced Hamiltonian-dissipative acoustic propagator (structure-preserving).

The CODEX deep-mathematics addendum's PRIMARY hypothesis: replace generic B2's
unidentifiable hidden-state recurrence with an EXPLICIT physical second-order state
and the audited discrete wave operator as a fixed core, correcting only a BOUNDED
neural residual acceleration -- with the KNOWN source waveform injected explicitly.

Update law (CFL-stable LWC-84 composition on the saved grid):

    p_{k+1} = Phi_LWC84^M(p_k, d_t p_k; c, s)
                + bounded learned closure,

State is the two consecutive pressure frames (p_{k-1}, p_k) == (p_k, v_k) -- a
genuine acoustic state, not a latent.  Contrast with generic B2:
  * gate==0 => a stable coarse-grid LWC-84 composition, not an arbitrary zero
    field.  A single 2.5 ms leapfrog step is forbidden because the 201-grid
    high-velocity cases violate its CFL limit.
  * the source is the actual Ricker waveform s(t_k) at the saved time, injected
    each step -- NOT a static feature broadcast identically (the addendum's
    essential correction).
  * the learned residual closes the audited discretization gap (20 internal
    substeps per saved frame + binomial5 restriction), the DGNet regime.

The neural residual is parameterized to be zero-initialized (gate) so the module is
an exact warm-start no-op over the physical step; frequency-window / symplectic
variants are registered SEPARATELY (one structural change at a time).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .spectral import FactorizedComplexResidualStack
from .wave_operators import wave_acceleration


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class PhysicalResidualPropagator(nn.Module):
    """B2-H: explicit (p_{k-1}, p_k) state + fixed wave core + bounded neural residual.

    forward(p0, p1, velocity, source_map, source_series, steps) -> [B, steps, 1, Z, X]

    p0, p1:         [B, 1, Z, X] two consecutive saved pressure frames (the IC state).
    velocity:       [B, 1, Z, X] medium c(x,z) in m/s (saved grid).
    source_map:     [B, 1, Z, X] spatial source footprint (bilinear point delta).
    source_series:  [B, steps] known source wavelet samples s(t_k) at saved times.
    Emits p_1..p_steps (the rollout); p_0/p_1 are the given IC.
    """

    def __init__(
        self,
        *,
        cond_channels: int = 0,
        memory_steps: int = 0,
        width: int = 48,
        spectral_rank: int = 24,
        modes: int = 16,
        depth: int = 3,
        dt_s: float = 0.0025,
        dx_m: float = 10.0,
        dz_m: float = 10.0,
        gate_init: float = 0.0,
        residual_scale_init: float = 1.0,
        substeps_per_saved_step: int = 2,
        pressure_scale: float = 1.0e-7,
        velocity_reference_mps: float = 3000.0,
        velocity_scale_mps: float = 1500.0,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.dt_s = float(dt_s)
        self.dx_m = float(dx_m)
        self.dz_m = float(dz_m)
        self.substeps_per_saved_step = int(substeps_per_saved_step)
        self.memory_steps = int(memory_steps)
        self.pressure_scale = float(pressure_scale)
        self.velocity_reference_mps = float(velocity_reference_mps)
        self.velocity_scale_mps = float(velocity_scale_mps)
        if self.substeps_per_saved_step <= 0:
            raise ValueError("substeps_per_saved_step must be positive")
        if self.memory_steps not in (0, 1):
            raise ValueError("memory_steps must be 0 (B2-H) or 1 (B2-HM)")
        if self.pressure_scale <= 0.0 or self.velocity_scale_mps <= 0.0:
            raise ValueError("pressure_scale and velocity_scale_mps must be positive")
        self.activation_checkpointing = bool(activation_checkpointing)
        # residual sees [p_k, p_{k-1}, c, source_map] + optional extra cond channels
        in_ch = 4 + self.memory_steps + int(cond_channels)
        self.lift = nn.Conv2d(in_ch, width, kernel_size=3, padding=1)
        self.norm = _group_norm(width)
        self.stack = FactorizedComplexResidualStack(
            width=int(width), spectral_rank=int(spectral_rank), modes=int(modes),
            depth=int(depth), activation_checkpointing=bool(activation_checkpointing),
        )
        self.project = nn.Conv2d(width, 1, kernel_size=1)
        # zero-init gate => exact physical-step warm-start no-op
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        if float(residual_scale_init) <= 0.0:
            raise ValueError("residual_scale_init must be positive")
        # Log-parameterization keeps the acceleration scale positive and makes
        # multiplicative adaptation gradual even when the gate has a larger LR.
        self.log_residual_scale = nn.Parameter(
            torch.tensor(math.log(float(residual_scale_init)), dtype=torch.float32)
        )

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.log_residual_scale.exp()

    def _neural_accel(
        self, p_cur, p_prev, velocity, source_map, memory, cond
    ):
        # The dataset pressure is O(1e-7) while velocity is O(1e3).  Feeding both
        # in physical units makes the residual closure effectively blind to the
        # wave state.  Normalize only the learned branch; the fixed wave operator
        # below continues to use physical SI units.
        p_cur_feature = p_cur / self.pressure_scale
        p_prev_feature = p_prev / self.pressure_scale
        velocity_feature = (
            velocity - self.velocity_reference_mps
        ) / self.velocity_scale_mps
        feats = [p_cur_feature, p_prev_feature, velocity_feature, source_map]
        if self.memory_steps:
            if memory is None:
                raise ValueError("B2-HM requires one history frame")
            feats.append(memory / self.pressure_scale)
        if cond is not None:
            feats.append(cond)
        h = self.lift(torch.cat(feats, dim=1))
        h = self.norm(h)
        h = self.stack(h)
        # The closure is explicitly bounded.  residual_scale retains acceleration
        # units while tanh prevents a newly-opened gate from destabilizing rollout.
        # The true homogeneous PDE preserves the zero state.  This activity
        # factor prevents medium/source-map features and network biases from
        # creating a spurious wave before the physical source arrives.
        activity = torch.tanh(
            (p_cur.abs() + p_prev.abs()) / self.pressure_scale
        )
        return self.residual_scale * activity * torch.tanh(self.project(h))

    def _physical_accel(self, p_cur, velocity):
        # c^2 * lap(p) using the audited 8th-order core (float compute in model dtype)
        return wave_acceleration(p_cur, velocity, dx_m=self.dx_m, dz_m=self.dz_m,
                                 free_surface_top=True)

    def step(self, p_prev, p_cur, velocity, source_map, source_scalar, cond=None):
        """One legacy coarse leapfrog step, retained as a unit-test reference."""
        # HDF5 source_map stores bilinear weights that sum to one.  The generator
        # injects the discrete delta_h = weights / (dx * dz), not the raw map.
        source_accel = source_scalar * source_map / (self.dx_m * self.dz_m)
        accel = self._physical_accel(p_cur, velocity) + source_accel
        if self.gate.abs().item() != 0.0 or self.training:
            accel = accel + torch.tanh(self.gate) * self._neural_accel(
                p_cur, p_prev, velocity, source_map, None, cond
            )
        return 2.0 * p_cur - p_prev + (self.dt_s ** 2) * accel

    def _source_terms(self, source_map, source_parameters, time_s):
        """Exact Ricker q and q_tt at per-example times.

        source_parameters is [B,3] ordered as (f0_hz, t0_s, amplitude).
        """
        frequency = source_parameters[:, 0].view(-1, 1, 1, 1)
        onset = source_parameters[:, 1].view(-1, 1, 1, 1)
        amplitude = source_parameters[:, 2].view(-1, 1, 1, 1)
        tau = time_s.view(-1, 1, 1, 1) - onset
        ricker_a = torch.pi**2 * frequency.square()
        exponential = torch.exp(-ricker_a * tau.square())
        wavelet = (1.0 - 2.0 * ricker_a * tau.square()) * exponential
        wavelet_tt = (
            -6.0 * ricker_a
            + 24.0 * ricker_a.square() * tau.square()
            - 8.0 * ricker_a.pow(3) * tau.pow(4)
        ) * exponential
        delta = source_map / (self.dx_m * self.dz_m)
        return amplitude * wavelet * delta, amplitude * wavelet_tt * delta

    def _advance_saved_interval(
        self,
        fine_prev,
        fine_cur,
        saved_history,
        saved_prev,
        saved_cur,
        velocity,
        source_map,
        source_scalar,
        source_parameters,
        interval_start_s,
        cond,
    ):
        """Compose stable LWC-84 substeps over one saved-frame interval."""
        if not self.training and self.gate.detach().abs().item() == 0.0:
            residual = torch.zeros_like(saved_cur)
        else:
            residual = torch.tanh(self.gate) * self._neural_accel(
                saved_cur,
                saved_prev,
                velocity,
                source_map,
                saved_history,
                cond,
            )
        dt_sub = self.dt_s / self.substeps_per_saved_step
        for substep in range(self.substeps_per_saved_step):
            if source_parameters is None:
                source = (
                    source_scalar * source_map / (self.dx_m * self.dz_m)
                )
                source_tt = torch.zeros_like(source)
            else:
                time_s = interval_start_s + float(substep) * dt_sub
                source, source_tt = self._source_terms(
                    source_map, source_parameters, time_s
                )
            acceleration = self._physical_accel(fine_cur, velocity) + source + residual
            fourth = self._physical_accel(acceleration, velocity) + source_tt
            fine_next = (
                2.0 * fine_cur
                - fine_prev
                + dt_sub**2 * acceleration
                + (dt_sub**4 / 12.0) * fourth
            )
            # The top node is the registered pressure-release free surface.
            fine_next = fine_next.clone()
            fine_next[..., 0, :] = 0.0
            fine_prev, fine_cur = fine_cur, fine_next
        return fine_prev, fine_cur

    def forward(
        self,
        p0,
        p1,
        velocity,
        source_map,
        source_series,
        steps=None,
        cond=None,
        source_parameters=None,
        initial_time_s=None,
        history=None,
    ):
        for t in (p0, p1, velocity, source_map):
            if t.ndim != 4 or t.shape[1] != 1:
                raise ValueError("p0,p1,velocity,source_map must be [B,1,Z,X]")
        if source_series.ndim != 2:
            raise ValueError("source_series must be [B, steps]")
        n = int(steps) if steps is not None else source_series.shape[1]
        if n <= 0:
            raise ValueError("steps must be positive")
        if source_parameters is not None:
            if source_parameters.ndim != 2 or source_parameters.shape[1] != 3:
                raise ValueError("source_parameters must be [B,3]")
            if initial_time_s is None or initial_time_s.ndim != 1:
                raise ValueError("initial_time_s must be [B] with source_parameters")
        if self.memory_steps:
            if (
                history is None
                or history.ndim != 4
                or history.shape != p0.shape
            ):
                raise ValueError("B2-HM history must match p0 [B,1,Z,X]")

        # Keep the original one-step branch as an exact regression oracle.
        if (
            self.substeps_per_saved_step == 1
            and source_parameters is None
            and not self.memory_steps
        ):
            p_prev, p_cur = p0, p1
            frames = []
            for k in range(n):
                s = source_series[:, k].view(-1, 1, 1, 1)
                p_next = self.step(
                    p_prev, p_cur, velocity, source_map, s, cond
                )
                frames.append(p_next)
                p_prev, p_cur = p_cur, p_next
            return torch.stack(frames, dim=1)

        dt_sub = self.dt_s / self.substeps_per_saved_step
        saved_history, saved_prev, saved_cur = history, p0, p1
        # Reconstruct the fine state at t_1-dt_sub from the two observed saved
        # frames.  Only the fixed physics is used here, so the learned closure
        # cannot corrupt its own initial condition.
        time_derivative = (p1 - p0) / self.dt_s
        if source_parameters is None:
            initial_source = (
                source_series[:, 0].view(-1, 1, 1, 1)
                * source_map
                / (self.dx_m * self.dz_m)
            )
        else:
            initial_source, _ = self._source_terms(
                source_map, source_parameters, initial_time_s
            )
        initial_acceleration = self._physical_accel(p1, velocity) + initial_source
        fine_prev = (
            p1
            - dt_sub * time_derivative
            + 0.5 * dt_sub**2 * initial_acceleration
        )
        fine_cur = p1
        frames = []
        for k in range(n):
            s = source_series[:, k].view(-1, 1, 1, 1)
            interval_start = (
                None
                if initial_time_s is None
                else initial_time_s + float(k) * self.dt_s
            )
            fine_prev, fine_cur = self._advance_saved_interval(
                fine_prev,
                fine_cur,
                saved_history,
                saved_prev,
                saved_cur,
                velocity,
                source_map,
                s,
                source_parameters,
                interval_start,
                cond,
            )
            frames.append(fine_cur)
            saved_history, saved_prev, saved_cur = (
                saved_prev,
                saved_cur,
                fine_cur,
            )
        return torch.stack(frames, dim=1)

    @torch.no_grad()
    def is_warmstart_noop(self) -> bool:
        return bool(self.gate.abs().item() == 0.0)
