from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from fno_acoustic.numerics.drp_coefficients import DRPCoefficients, optimized_drp_second_derivative_coefficients

from .grid import AcousticGrid, BoundaryConfig, OutputTimeGrid
from .quality import cpml_damping_mask, cpml_memory_coefficients, enforce_free_surface
from .source import bilinear_point_source, source_time_function


@dataclass
class PreparedModel:
    velocity_zx: np.ndarray
    grid: AcousticGrid
    boundaries: BoundaryConfig
    dt_s: float


@dataclass
class SimulationResult:
    wavefield: np.ndarray
    metrics: dict[str, Any]


class AcousticTeacherBackend:
    backend_name = "abstract_acoustic_teacher"

    def prepare_velocity(self, velocity_zx, *, grid: AcousticGrid, boundaries: BoundaryConfig, dt_s: float) -> PreparedModel:
        velocity = np.asarray(velocity_zx, dtype=np.float64)
        if velocity.shape != (grid.nz, grid.nx):
            raise ValueError(f"velocity must be shaped {(grid.nz, grid.nx)}, got {velocity.shape}")
        extended = _extend_velocity_edge_replicated(velocity, grid=grid, boundaries=boundaries)
        return PreparedModel(velocity_zx=extended, grid=grid, boundaries=boundaries, dt_s=float(dt_s))


def _drp_radius4_coefficients() -> DRPCoefficients:
    return optimized_drp_second_derivative_coefficients(radius=4, max_nyquist_fraction=0.65)


def _extend_velocity_edge_replicated(velocity: np.ndarray, *, grid: AcousticGrid, boundaries: BoundaryConfig) -> np.ndarray:
    if velocity.shape != (grid.nz, grid.nx):
        raise ValueError(f"velocity must be shaped {(grid.nz, grid.nx)}, got {velocity.shape}")
    return np.pad(np.asarray(velocity), ((0, int(boundaries.npml)), (int(boundaries.npml), int(boundaries.npml))), mode="edge")


def _solver_order_metadata(coefficients: DRPCoefficients) -> dict[str, Any]:
    return {
        "spatial_radius": int(coefficients.radius),
        "space_order": int(2 * coefficients.radius),
        "time_order": 2,
        "spatial_method": coefficients.provenance["method"],
        "spatial_coefficients_sha256": coefficients.coefficient_sha256,
    }


def _shifted_slice(radius: int, offset: int) -> slice:
    start = int(radius + offset)
    stop = int(-radius + offset)
    return slice(start, None if stop == 0 else stop)


_STAGGERED_FIRST_DERIVATIVE_COEFFS = (
    1225.0 / 1024.0,
    -245.0 / 3072.0,
    49.0 / 5120.0,
    -5.0 / 7168.0,
)
_CPML_REFERENCE_VELOCITY_M_S = 5500.0


def _staggered_lwc8_solver_metadata() -> dict[str, Any]:
    return {
        "spatial_radius": 4,
        "space_order": 8,
        "time_order": 2,
        "spatial_method": "staggered_velocity_pressure_first_derivative",
        "temporal_method": "leapfrog_velocity_pressure",
        "lwc84_compatible": True,
        "staggered_first_derivative_coefficients": tuple(float(c) for c in _STAGGERED_FIRST_DERIVATIVE_COEFFS),
    }


def _pad_x_replicate(field: torch.Tensor, radius: int = 4) -> torch.Tensor:
    left = field[..., :1].expand(*field.shape[:-1], int(radius))
    right = field[..., -1:].expand(*field.shape[:-1], int(radius))
    return torch.cat((left, field, right), dim=-1)


def _pad_z_replicate(field: torch.Tensor, radius: int = 4) -> torch.Tensor:
    top = field[:, :1, :].expand(field.shape[0], int(radius), field.shape[-1])
    bottom = field[:, -1:, :].expand(field.shape[0], int(radius), field.shape[-1])
    return torch.cat((top, field, bottom), dim=1)


def _center_to_face_derivative_x(field: torch.Tensor, dx_m: float) -> torch.Tensor:
    radius = len(_STAGGERED_FIRST_DERIVATIVE_COEFFS)
    nx = int(field.shape[-1])
    padded = _pad_x_replicate(field, radius)
    out = torch.zeros_like(field)
    for offset, coeff in enumerate(_STAGGERED_FIRST_DERIVATIVE_COEFFS, start=1):
        right = padded[..., radius + offset : radius + offset + nx]
        left = padded[..., radius + 1 - offset : radius + 1 - offset + nx]
        out = out + float(coeff) * (right - left)
    return out / float(dx_m)


def _face_to_center_derivative_x(field: torch.Tensor, dx_m: float) -> torch.Tensor:
    radius = len(_STAGGERED_FIRST_DERIVATIVE_COEFFS)
    nx = int(field.shape[-1])
    padded = _pad_x_replicate(field, radius)
    out = torch.zeros_like(field)
    for offset, coeff in enumerate(_STAGGERED_FIRST_DERIVATIVE_COEFFS, start=1):
        right = padded[..., radius + offset - 1 : radius + offset - 1 + nx]
        left = padded[..., radius - offset : radius - offset + nx]
        out = out + float(coeff) * (right - left)
    return out / float(dx_m)


def _center_to_face_derivative_z(field: torch.Tensor, dz_m: float) -> torch.Tensor:
    radius = len(_STAGGERED_FIRST_DERIVATIVE_COEFFS)
    nz = int(field.shape[1])
    padded = _pad_z_replicate(field, radius)
    out = torch.zeros_like(field)
    for offset, coeff in enumerate(_STAGGERED_FIRST_DERIVATIVE_COEFFS, start=1):
        lower = padded[:, radius + offset : radius + offset + nz, :]
        upper = padded[:, radius + 1 - offset : radius + 1 - offset + nz, :]
        out = out + float(coeff) * (lower - upper)
    return out / float(dz_m)


def _face_to_center_derivative_z(field: torch.Tensor, dz_m: float) -> torch.Tensor:
    radius = len(_STAGGERED_FIRST_DERIVATIVE_COEFFS)
    nz = int(field.shape[1])
    padded = _pad_z_replicate(field, radius)
    out = torch.zeros_like(field)
    for offset, coeff in enumerate(_STAGGERED_FIRST_DERIVATIVE_COEFFS, start=1):
        lower = padded[:, radius + offset - 1 : radius + offset - 1 + nz, :]
        upper = padded[:, radius - offset : radius - offset + nz, :]
        out = out + float(coeff) * (lower - upper)
    return out / float(dz_m)


class CPUFloat64ReferenceBackend(AcousticTeacherBackend):
    backend_name = "cpu_float64_drp_radius4_extended_pml_reference"

    def __init__(self, *, grid: AcousticGrid, boundaries: BoundaryConfig, time: OutputTimeGrid, n_substeps: int) -> None:
        self.grid = grid
        self.boundaries = boundaries
        self.time = time
        self.n_substeps = int(n_substeps)
        self.coefficients = _drp_radius4_coefficients()
        self.solver_orders = _solver_order_metadata(self.coefficients)

    def simulate_batch(self, prepared_model: PreparedModel, *, source_xy_m, f0_hz, output_times_s) -> SimulationResult:
        start = time.perf_counter()
        grid = prepared_model.grid
        velocity = prepared_model.velocity_zx
        dt = float(prepared_model.dt_s)
        physical_z, physical_x = prepared_model.boundaries.physical_slices(grid)
        out_times = np.asarray(output_times_s, dtype=np.float64)
        sources = np.asarray(source_xy_m, dtype=np.float64)
        freqs = np.asarray(f0_hz, dtype=np.float64)
        batch = int(sources.shape[0])
        wavefield = np.zeros((batch, grid.nz, grid.nx, out_times.size), dtype=np.float32)
        damping = cpml_damping_mask(grid, prepared_model.boundaries)
        damping = damping.astype(np.float64)
        coeff_x = (velocity * dt / grid.dx_m) ** 2
        coeff_z = (velocity * dt / grid.dz_m) ** 2
        radius = int(self.coefficients.radius)
        if velocity.shape[0] <= 2 * radius or velocity.shape[1] <= 2 * radius:
            raise ValueError("grid is too small for the configured high-order stencil")
        core_z = slice(radius, -radius)
        core_x = slice(radius, -radius)
        n_total = int((out_times.size - 1) * self.n_substeps) + 1
        output_steps = set((np.arange(out_times.size) * self.n_substeps).tolist())
        for b in range(batch):
            p_nm1 = np.zeros(velocity.shape, dtype=np.float64)
            p_n = np.zeros_like(p_nm1)
            src = bilinear_point_source(
                float(sources[b, 0]),
                float(sources[b, 1]),
                nx=grid.nx,
                nz=grid.nz,
                dx_m=grid.dx_m,
                dz_m=grid.dz_m,
            )
            delta_h = np.zeros_like(p_n)
            delta_h[physical_z, physical_x] = src.delta_h
            out_index = 0
            for step in range(n_total):
                if step in output_steps:
                    snap = enforce_free_surface(p_n.copy()[None, ...])[0]
                    wavefield[b, :, :, out_index] = snap[physical_z, physical_x].astype(np.float32)
                    out_index += 1
                lap = np.zeros_like(p_n)
                core = p_n[core_z, core_x]
                sum_x = self.coefficients.center * core
                sum_z = self.coefficients.center * core
                for offset, value in enumerate(self.coefficients.positive_offsets, start=1):
                    xp = _shifted_slice(radius, offset)
                    xm = _shifted_slice(radius, -offset)
                    sum_x = sum_x + value * (p_n[core_z, xp] + p_n[core_z, xm])
                    sum_z = sum_z + value * (p_n[xp, core_x] + p_n[xm, core_x])
                lap[core_z, core_x] = coeff_x[core_z, core_x] * sum_x + coeff_z[core_z, core_x] * sum_z
                source_term = np.zeros_like(p_n)
                amp = source_time_function(np.asarray([step * dt]), float(freqs[b]))[0] * dt * dt
                source_term += (velocity**2) * amp * delta_h
                p_np1 = (2.0 * p_n - p_nm1 + lap + source_term) * damping
                p_np1[0, :] = 0.0
                p_nm1, p_n = p_n, p_np1
        elapsed = time.perf_counter() - start
        return SimulationResult(wavefield=wavefield, metrics={"backend_name": self.backend_name, "elapsed_s": elapsed})


class TorchCompiledAcousticBackend(AcousticTeacherBackend):
    backend_name = "torch_cuda_staggered_velocity_pressure_cpml_lrb_lwc8_compatible_v5"

    def __init__(self, *, grid: AcousticGrid, time: OutputTimeGrid, device: str = "cuda", n_substeps: int = 1) -> None:
        self.grid = grid
        self.time = time
        self.device = torch.device(device)
        self.n_substeps = int(n_substeps)
        if self.n_substeps < 1:
            raise ValueError("n_substeps must be positive")
        self.coefficients = _STAGGERED_FIRST_DERIVATIVE_COEFFS
        self.solver_orders = _staggered_lwc8_solver_metadata()

    def prepare_velocity(self, velocity_zx, *, grid: AcousticGrid, boundaries: BoundaryConfig, dt_s: float) -> dict[str, Any]:
        velocity_np = _extend_velocity_edge_replicated(np.asarray(velocity_zx, dtype=np.float32), grid=grid, boundaries=boundaries)
        cpml = cpml_memory_coefficients(grid, boundaries, dt_s=float(dt_s), c_ref_m_s=_CPML_REFERENCE_VELOCITY_M_S)
        velocity = torch.as_tensor(velocity_np, dtype=torch.float32, device=self.device)
        return {
            "velocity_zx": velocity,
            "a_x": torch.as_tensor(cpml["a_x"], dtype=torch.float32, device=self.device),
            "b_x": torch.as_tensor(cpml["b_x"], dtype=torch.float32, device=self.device),
            "inv_kappa_x": torch.as_tensor(cpml["inv_kappa_x"], dtype=torch.float32, device=self.device),
            "a_z": torch.as_tensor(cpml["a_z"], dtype=torch.float32, device=self.device),
            "b_z": torch.as_tensor(cpml["b_z"], dtype=torch.float32, device=self.device),
            "inv_kappa_z": torch.as_tensor(cpml["inv_kappa_z"], dtype=torch.float32, device=self.device),
            "grid": grid,
            "boundaries": boundaries,
            "dt_s": float(dt_s),
        }

    def prepare_velocity_batch(self, velocity_batch_zx, *, grid: AcousticGrid, boundaries: BoundaryConfig, dt_s: float) -> dict[str, Any]:
        velocities = np.asarray(velocity_batch_zx, dtype=np.float32)
        if velocities.ndim != 3 or velocities.shape[1:] != (grid.nz, grid.nx):
            raise ValueError(f"velocity batch must be shaped (B, {grid.nz}, {grid.nx}), got {velocities.shape}")
        extended = np.pad(velocities, ((0, 0), (0, int(boundaries.npml)), (int(boundaries.npml), int(boundaries.npml))), mode="edge")
        cpml = cpml_memory_coefficients(grid, boundaries, dt_s=float(dt_s), c_ref_m_s=_CPML_REFERENCE_VELOCITY_M_S)
        velocity = torch.as_tensor(extended, dtype=torch.float32, device=self.device)
        return {
            "velocity_zx": velocity,
            "a_x": torch.as_tensor(cpml["a_x"], dtype=torch.float32, device=self.device),
            "b_x": torch.as_tensor(cpml["b_x"], dtype=torch.float32, device=self.device),
            "inv_kappa_x": torch.as_tensor(cpml["inv_kappa_x"], dtype=torch.float32, device=self.device),
            "a_z": torch.as_tensor(cpml["a_z"], dtype=torch.float32, device=self.device),
            "b_z": torch.as_tensor(cpml["b_z"], dtype=torch.float32, device=self.device),
            "inv_kappa_z": torch.as_tensor(cpml["inv_kappa_z"], dtype=torch.float32, device=self.device),
            "grid": grid,
            "boundaries": boundaries,
            "dt_s": float(dt_s),
        }

    def simulate_batch(self, prepared_model, *, source_xy_m, f0_hz, output_times_s):
        start = time.perf_counter()
        grid = prepared_model["grid"]
        velocity = prepared_model["velocity_zx"]
        dt_s = float(prepared_model["dt_s"])
        physical_z, physical_x = prepared_model["boundaries"].physical_slices(grid)
        sources = np.asarray(source_xy_m, dtype=np.float64)
        freqs_np = np.asarray(f0_hz, dtype=np.float64)
        output_times = np.asarray(output_times_s, dtype=np.float64)
        batch = int(sources.shape[0])
        if int(freqs_np.shape[0]) != batch:
            raise ValueError(f"f0_hz length {freqs_np.shape[0]} does not match source batch {batch}")
        if velocity.ndim == 3 and int(velocity.shape[0]) != batch:
            raise ValueError(f"batched velocity size {velocity.shape[0]} does not match source batch {batch}")
        extended_shape = velocity.shape[-2:]
        if extended_shape[0] < 3 or extended_shape[1] < 3:
            raise ValueError("grid is too small for the configured CPML stencil")
        p_n = torch.zeros((batch, int(extended_shape[0]), int(extended_shape[1])), dtype=torch.float32, device=self.device)
        vx_h = torch.zeros_like(p_n)
        vz_h = torch.zeros_like(p_n)
        psi_vx = torch.zeros_like(p_n)
        psi_vz = torch.zeros_like(p_n)
        psi_px = torch.zeros_like(p_n)
        psi_pz = torch.zeros_like(p_n)
        out = torch.zeros((batch, grid.nz, grid.nx, output_times.size), dtype=torch.float32, device=self.device)
        delta = torch.zeros_like(p_n)
        for b in range(batch):
            src = bilinear_point_source(
                float(sources[b, 0]),
                float(sources[b, 1]),
                nx=grid.nx,
                nz=grid.nz,
                dx_m=grid.dx_m,
                dz_m=grid.dz_m,
            )
            for (z_idx, x_idx), weight in zip(src.indices.tolist(), src.weights.tolist()):
                value = float(weight) / (float(grid.dx_m) * float(grid.dz_m))
                delta[b, int(z_idx), int(x_idx) + int(prepared_model["boundaries"].npml)] = float(value)
        n_total = int((output_times.size - 1) * self.n_substeps) + 1
        amp = np.zeros((batch, n_total), dtype=np.float32)
        for b in range(batch):
            amp[b, :] = (source_time_function(np.arange(n_total, dtype=np.float64) * dt_s, float(freqs_np[b])) * dt_s).astype(np.float32)
        amp_t = torch.as_tensor(amp, dtype=torch.float32, device=self.device)
        out[:, :, :, 0] = p_n[:, physical_z, physical_x]
        velocity_batch = velocity if velocity.ndim == 3 else velocity.unsqueeze(0)
        a_x = prepared_model["a_x"].unsqueeze(0)
        b_x = prepared_model["b_x"].unsqueeze(0)
        inv_kappa_x = prepared_model["inv_kappa_x"].unsqueeze(0)
        a_z = prepared_model["a_z"].unsqueeze(0)
        b_z = prepared_model["b_z"].unsqueeze(0)
        inv_kappa_z = prepared_model["inv_kappa_z"].unsqueeze(0)
        for out_index in range(1, output_times.size):
            for substep in range(self.n_substeps):
                step = (out_index - 1) * self.n_substeps + substep
                dp_dx = _center_to_face_derivative_x(p_n, grid.dx_m)
                psi_vx = b_x * psi_vx + a_x * dp_dx
                vx_h = vx_h - float(dt_s) * (inv_kappa_x * dp_dx + psi_vx)

                dp_dz = _center_to_face_derivative_z(p_n, grid.dz_m)
                psi_vz = b_z * psi_vz + a_z * dp_dz
                vz_h = vz_h - float(dt_s) * (inv_kappa_z * dp_dz + psi_vz)
                vz_h[:, 0, :] = 0.0

                dvx_dx = _face_to_center_derivative_x(vx_h, grid.dx_m)
                psi_px = b_x * psi_px + a_x * dvx_dx
                div_x = inv_kappa_x * dvx_dx + psi_px

                dvz_dz = _face_to_center_derivative_z(vz_h, grid.dz_m)
                psi_pz = b_z * psi_pz + a_z * dvz_dz
                div_z = inv_kappa_z * dvz_dz + psi_pz

                source_term = velocity_batch.pow(2) * amp_t[:, step].view(batch, 1, 1) * delta
                p_n = p_n - velocity_batch.pow(2) * float(dt_s) * (div_x + div_z) + source_term
                p_n[:, 0, :] = 0.0
            out[:, :, :, out_index] = p_n[:, physical_z, physical_x]
        elapsed = time.perf_counter() - start
        return SimulationResult(wavefield=out.contiguous(), metrics={"backend_name": self.backend_name, "elapsed_s": elapsed})
