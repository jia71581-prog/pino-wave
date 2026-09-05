from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from fno_acoustic.numerics.drp_coefficients import (
    DRPCoefficients,
    optimized_drp_second_derivative_coefficients,
)

from .cpml import CFSCPMLOperator, build_cfs_cpml_profiles
from .grid import AcousticGrid, BoundaryConfig
from .lwc84 import apply_l_operator, lwc84_startup
from .restriction import restrict_nodal_2x
from .ricker import ricker_triplet
from .source import bilinear_point_source


@dataclass(frozen=True)
class LWC84ExteriorAuxiliaryState:
    """Sparse saved-time snapshots on the complete exterior-CPML domain.

    Memory values at output step ``n`` are the causal ADE state after applying
    pressure step ``n-1`` and immediately before applying pressure step ``n``.
    """

    pressure_extended_saved: np.ndarray
    velocity_extended_saved_mps: np.ndarray
    psi_x_before_step: np.ndarray
    psi_z_before_step: np.ndarray
    phi_x_before_step: np.ndarray
    phi_z_before_step: np.ndarray
    physical_slice_zx: tuple[tuple[int, int], tuple[int, int]]
    restriction: str = "binomial5_lowpass_then_nodal_decimate2"
    memory_alignment: str = "after_pressure_step_n_minus_1_before_pressure_step_n"


@dataclass(frozen=True)
class LWC84SimulationResult:
    wavefield: np.ndarray
    velocity_saved_mps: np.ndarray
    source_map_saved: np.ndarray
    source_map_solver: np.ndarray
    source_wavelet: np.ndarray
    metrics: list[dict[str, float | int | bool]]
    exterior_auxiliary: LWC84ExteriorAuxiliaryState | None = None


def _batch_parameter(value, *, batch: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(batch, float(array), dtype=np.float64)
    array = array.reshape(-1)
    if array.size != batch:
        raise ValueError(f"{name} has {array.size} values for batch size {batch}")
    return array


class LWC84CPMLSolver:
    """Physical-domain LWC-84 core coupled to three-sided unsplit CFS-CPML."""

    def __init__(
        self,
        *,
        grid: AcousticGrid,
        boundaries: BoundaryConfig,
        dt_s: float,
        output_times_s: np.ndarray,
        c_ref_mps: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        kappa_max: float = 3.0,
        minimum_frequency_hz: float = 8.0,
        output_restriction_factor: int = 2,
        drp_max_nyquist_fraction: float | None = None,
    ) -> None:
        if (
            isinstance(output_restriction_factor, bool)
            or not isinstance(output_restriction_factor, (int, np.integer))
            or int(output_restriction_factor) not in (1, 2)
        ):
            raise ValueError("output_restriction_factor must be 1 or 2")
        if grid.centering != "node" or grid.nx % 2 != 1 or grid.nz % 2 != 1:
            raise ValueError("LWC84CPMLSolver requires odd node-centred physical dimensions")
        if float(dt_s) <= 0.0:
            raise ValueError("dt_s must be positive")
        self.grid = grid
        self.boundaries = boundaries
        self.dt_s = float(dt_s)
        self.output_times_s = np.asarray(output_times_s, dtype=np.float64)
        self.device = torch.device(device)
        self.dtype = dtype
        self.output_restriction_factor = int(output_restriction_factor)
        self.spatial_coefficients: DRPCoefficients | None = None
        if drp_max_nyquist_fraction is not None:
            self.spatial_coefficients = optimized_drp_second_derivative_coefficients(
                radius=4,
                max_nyquist_fraction=float(drp_max_nyquist_fraction),
            )
        if self.output_times_s.ndim != 1 or self.output_times_s.size < 1 or self.output_times_s[0] != 0.0:
            raise ValueError("output_times_s must be a one-dimensional vector starting at zero")
        steps = np.rint(self.output_times_s / self.dt_s).astype(np.int64)
        if not np.allclose(steps * self.dt_s, self.output_times_s, rtol=0.0, atol=1.0e-12):
            raise ValueError("all output times must align exactly to dt_s")
        if np.any(np.diff(steps) <= 0):
            raise ValueError("output times must be strictly increasing")
        self.output_steps = steps
        self.profiles = build_cfs_cpml_profiles(
            grid,
            boundaries,
            dt_s=self.dt_s,
            c_ref_mps=float(c_ref_mps),
            target_reflection=boundaries.cpml_target_reflection,
            polynomial_order=boundaries.cpml_polynomial_order,
            kappa_max=float(kappa_max),
            minimum_frequency_hz=float(minimum_frequency_hz),
            device=self.device,
            dtype=self.dtype,
        )

    def _extend_velocity(self, velocity: np.ndarray) -> torch.Tensor:
        npml = int(self.boundaries.npml)
        extended = np.pad(velocity, ((0, 0), (0, npml), (npml, npml)), mode="edge")
        return torch.as_tensor(extended, device=self.device, dtype=self.dtype)

    def _restrict_output(self, physical: torch.Tensor) -> torch.Tensor:
        if self.output_restriction_factor == 1:
            return physical.clone()
        return restrict_nodal_2x(physical)

    def simulate(
        self,
        velocity_mps,
        *,
        source_x_m,
        source_z_m,
        source_f0_hz,
        source_t0_s=None,
        source_amplitude=1.0,
        capture_exterior_auxiliary: bool = False,
    ) -> LWC84SimulationResult:
        compute_started = time.perf_counter()
        velocity_np = np.asarray(velocity_mps)
        if velocity_np.ndim == 2:
            velocity_np = velocity_np[None, ...]
        if velocity_np.ndim != 3 or velocity_np.shape[1:] != (self.grid.nz, self.grid.nx):
            raise ValueError(
                f"velocity must be [B,{self.grid.nz},{self.grid.nx}], got {velocity_np.shape}"
            )
        if not np.isfinite(velocity_np).all() or float(np.min(velocity_np)) <= 0.0:
            raise ValueError("velocity must be positive and finite before wavefield generation")
        batch = int(velocity_np.shape[0])
        source_x = _batch_parameter(source_x_m, batch=batch, name="source_x_m")
        source_z = _batch_parameter(source_z_m, batch=batch, name="source_z_m")
        f0 = _batch_parameter(source_f0_hz, batch=batch, name="source_f0_hz")
        amplitude = _batch_parameter(source_amplitude, batch=batch, name="source_amplitude")
        if source_t0_s is None:
            t0 = 1.5 / f0
        else:
            t0 = _batch_parameter(source_t0_s, batch=batch, name="source_t0_s")
        if np.any(f0 <= 0.0) or np.any(amplitude == 0.0):
            raise ValueError("source frequency must be positive and amplitude must be nonzero")

        f0_device = torch.as_tensor(f0, dtype=self.dtype, device=self.device).view(batch, 1, 1)
        t0_device = torch.as_tensor(t0, dtype=self.dtype, device=self.device).view(batch, 1, 1)
        amplitude_device = torch.as_tensor(
            amplitude, dtype=self.dtype, device=self.device
        ).view(batch, 1, 1)
        ricker_a = math.pi**2 * f0_device.square()

        velocity = self._extend_velocity(velocity_np.astype(np.float64, copy=False))
        shape = tuple(int(value) for value in velocity.shape)
        delta = torch.zeros(shape, dtype=self.dtype, device=self.device)
        source_solver = np.zeros((batch, self.grid.nz, self.grid.nx), dtype=np.float32)
        factor = self.output_restriction_factor
        saved_nz = (self.grid.nz - 1) // factor + 1
        saved_nx = (self.grid.nx - 1) // factor + 1
        source_saved = np.zeros((batch, saved_nz, saved_nx), dtype=np.float32)
        physical_z, physical_x = self.boundaries.physical_slices(self.grid)
        for index in range(batch):
            fine = bilinear_point_source(
                float(source_x[index]),
                float(source_z[index]),
                nx=self.grid.nx,
                nz=self.grid.nz,
                dx_m=self.grid.dx_m,
                dz_m=self.grid.dz_m,
                centering="node",
            )
            source_solver[index] = fine.source_map
            delta[index, physical_z, physical_x] = torch.as_tensor(
                fine.delta_h, device=self.device, dtype=self.dtype
            )
            saved_source = bilinear_point_source(
                float(source_x[index]),
                float(source_z[index]),
                nx=saved_nx,
                nz=saved_nz,
                dx_m=factor * self.grid.dx_m,
                dz_m=factor * self.grid.dz_m,
                centering="node",
            )
            source_saved[index] = saved_source.source_map

        maximum_step = int(self.output_steps[-1])
        internal_times = torch.arange(
            maximum_step + 1, dtype=self.dtype, device=self.device
        ).view(-1, 1, 1, 1) * self.dt_s
        tau = internal_times - t0_device.unsqueeze(0)
        exponential = torch.exp(-ricker_a.unsqueeze(0) * tau.square())
        source_series = (
            (1.0 - 2.0 * ricker_a.unsqueeze(0) * tau.square()) * exponential,
            (
                -6.0 * ricker_a.unsqueeze(0) * tau
                + 4.0 * ricker_a.square().unsqueeze(0) * tau.pow(3)
            )
            * exponential,
            (
                -6.0 * ricker_a.unsqueeze(0)
                + 24.0 * ricker_a.square().unsqueeze(0) * tau.square()
                - 8.0 * ricker_a.pow(3).unsqueeze(0) * tau.pow(4)
            )
            * exponential,
        )

        def source_terms(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return tuple(amplitude_device * values[int(step)] * delta for values in source_series)

        cpml = CFSCPMLOperator(
            self.profiles,
            dx_m=self.grid.dx_m,
            dz_m=self.grid.dz_m,
            physical_second_derivative_coefficients=self.spatial_coefficients,
        )
        p0 = torch.zeros_like(velocity)
        pt0 = torch.zeros_like(velocity)
        q0, qt0, qtt0 = source_terms(0)
        p1 = lwc84_startup(
            p0=p0,
            pt0=pt0,
            q0=q0,
            qt0=qt0,
            qtt0=qtt0,
            velocity_mps=velocity,
            dx_m=self.grid.dx_m,
            dz_m=self.grid.dz_m,
            dt_s=self.dt_s,
            boundary="free_surface",
            spatial_coefficients=self.spatial_coefficients,
        )
        p1[:, 0, :] = 0.0
        p1[:, -1, :] = 0.0
        p1[:, :, 0] = 0.0
        p1[:, :, -1] = 0.0
        output = torch.zeros(
            (batch, self.output_times_s.size, saved_nz, saved_nx),
            dtype=self.dtype,
            device=self.device,
        )
        exterior_output = None
        exterior_memories = None
        if bool(capture_exterior_auxiliary):
            extended_saved_nz = (shape[-2] - 1) // factor + 1
            extended_saved_nx = (shape[-1] - 1) // factor + 1
            exterior_shape = (
                batch,
                self.output_times_s.size,
                extended_saved_nz,
                extended_saved_nx,
            )
            exterior_output = torch.zeros(
                exterior_shape, dtype=self.dtype, device=self.device
            )
            exterior_memories = {
                name: torch.zeros(exterior_shape, dtype=self.dtype, device=self.device)
                for name in ("psi_x", "psi_z", "phi_x", "phi_z")
            }
        step_to_output = {int(step): index for index, step in enumerate(self.output_steps.tolist())}

        def save(step: int, field: torch.Tensor) -> None:
            output_index = step_to_output.get(int(step))
            if output_index is None:
                return
            physical = field[:, physical_z, physical_x]
            saved = self._restrict_output(physical)
            saved[:, 0, :] = 0.0
            output[:, output_index] = saved
            if exterior_output is None or exterior_memories is None:
                return
            extended_saved = self._restrict_output(field)
            extended_saved[:, 0, :] = 0.0
            extended_saved[:, -1, :] = 0.0
            extended_saved[:, :, 0] = 0.0
            extended_saved[:, :, -1] = 0.0
            exterior_output[:, output_index] = extended_saved
            for name, destination in exterior_memories.items():
                memory = getattr(cpml, name)
                if memory is not None:
                    destination[:, output_index] = self._restrict_output(memory)

        save(0, p0)
        save(1, p1)
        p_nm1, p_n = p0, p1
        for step in range(1, maximum_step):
            q_n, _, qtt_n = source_terms(step)
            acceleration = cpml.apply(p_n, velocity, update_memory=True) + q_n
            fourth = apply_l_operator(
                acceleration,
                velocity,
                dx_m=self.grid.dx_m,
                dz_m=self.grid.dz_m,
                boundary="free_surface",
                spatial_coefficients=self.spatial_coefficients,
            ) + qtt_n
            p_np1 = 2.0 * p_n - p_nm1 + self.dt_s**2 * acceleration + self.dt_s**4 * fourth / 12.0
            p_np1[:, 0, :] = 0.0
            p_np1[:, -1, :] = 0.0
            p_np1[:, :, 0] = 0.0
            p_np1[:, :, -1] = 0.0
            save(step + 1, p_np1)
            p_nm1, p_n = p_n, p_np1

        output_np = output.detach().cpu().numpy().astype(np.float32, copy=False)
        exterior_auxiliary = None
        if exterior_output is not None and exterior_memories is not None:
            exterior_np = (
                exterior_output.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            memory_np = {
                name: value.detach().cpu().numpy().astype(np.float32, copy=False)
                for name, value in exterior_memories.items()
            }
            extended_velocity_saved = self._restrict_output(velocity).detach().cpu()
            exterior_auxiliary = LWC84ExteriorAuxiliaryState(
                pressure_extended_saved=exterior_np,
                velocity_extended_saved_mps=extended_velocity_saved.numpy().astype(
                    np.float32, copy=False
                ),
                psi_x_before_step=memory_np["psi_x"],
                psi_z_before_step=memory_np["psi_z"],
                phi_x_before_step=memory_np["phi_x"],
                phi_z_before_step=memory_np["phi_z"],
                physical_slice_zx=(
                    (0, saved_nz),
                    (
                        self.boundaries.npml // factor,
                        self.boundaries.npml // factor + saved_nx,
                    ),
                ),
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        compute_elapsed_s = time.perf_counter() - compute_started
        if not np.isfinite(output_np).all():
            raise FloatingPointError("non-finite pressure in saved output")
        velocity_saved = self._restrict_output(
            torch.as_tensor(velocity_np, dtype=self.dtype, device=self.device)
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        source_wavelet = np.empty((batch, self.output_times_s.size), dtype=np.float32)
        metrics: list[dict[str, float | int | bool]] = []
        for index in range(batch):
            source_wavelet[index] = (
                amplitude[index]
                * ricker_triplet(
                    self.output_times_s, f0_hz=float(f0[index]), t0_s=float(t0[index])
                )[0]
            ).astype(np.float32)
            vmax = float(np.max(velocity_np[index]))
            cfl = vmax * self.dt_s * math.sqrt(
                1.0 / self.grid.dx_m**2 + 1.0 / self.grid.dz_m**2
            )
            axis_spectral_radius = 2048.0 / 315.0
            if self.spatial_coefficients is not None:
                theta = np.linspace(0.0, math.pi, 8193, dtype=np.float64)
                axis_spectral_radius = abs(
                    float(np.min(self.spatial_coefficients.symbol(theta)))
                )
            qmax = self.dt_s**2 * vmax**2 * axis_spectral_radius * (
                1.0 / self.grid.dx_m**2 + 1.0 / self.grid.dz_m**2
            )
            energies = np.sum(output_np[index].astype(np.float64) ** 2, axis=(1, 2))
            metrics.append(
                {
                    "vmin_mps": float(np.min(velocity_np[index])),
                    "vmax_mps": vmax,
                    "cfl_2d": cfl,
                    "lwc_qmax": qmax,
                    "dt_used_s": self.dt_s,
                    "qc_max_abs": float(np.max(np.abs(output_np[index]))),
                    "qc_final_energy_ratio": float(energies[-1] / max(float(np.max(energies)), 1.0e-30)),
                    "finite": bool(np.isfinite(output_np[index]).all()),
                    # Synchronized input-ready -> saved-output-materialized time.
                    # Subsequent finite/energy QC is intentionally outside this
                    # value so runtime comparisons score the numerical solve,
                    # not benchmark bookkeeping.
                    "compute_elapsed_s": float(compute_elapsed_s),
                }
            )
        return LWC84SimulationResult(
            wavefield=output_np,
            velocity_saved_mps=velocity_saved,
            source_map_saved=source_saved,
            source_map_solver=source_solver,
            source_wavelet=source_wavelet,
            metrics=metrics,
            exterior_auxiliary=exterior_auxiliary,
        )
