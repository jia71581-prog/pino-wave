from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .cpml import CFSCPMLOperator, build_cfs_cpml_profiles
from .grid import AcousticGrid, BoundaryConfig
from .ricker import ricker_triplet
from .source import bilinear_point_source
from .stencils import second_derivatives8


@dataclass(frozen=True)
class KSpaceSimulationResult:
    """Materialized pressure field and the deployment audit for one solve."""

    wavefield: np.ndarray
    velocity_saved_mps: np.ndarray
    source_map_saved: np.ndarray
    source_wavelet: np.ndarray
    metrics: list[dict[str, float | int | bool]]


def _batch_parameter(value, *, batch: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(batch, float(array), dtype=np.float64)
    array = array.reshape(-1)
    if array.size != batch:
        raise ValueError(f"{name} has {array.size} values for batch size {batch}")
    return array


class KSpacePSTDSolver:
    """Low-dispersion saved-grid acoustic propagator with three-sided sponge PML.

    The production truth is an eighth-order finite-difference solve on a 5 m grid
    followed by anti-aliased restriction to 10 m.  Re-running that stencil directly
    on the 10 m saved grid is fast but strongly dispersive at the shortest wavelengths.
    This candidate instead evaluates the Laplacian with a Fourier pseudospectral
    derivative and applies the standard homogeneous-reference k-space correction

        sinc(c_ref |k| dt / 2)^2.

    An odd vertical extension imposes the pressure-release top boundary exactly.
    Polynomial damping occupies only the left, right and bottom exterior layers; the
    physical 201-by-201 output is never tapered.  This is deliberately a go/no-go
    research kernel, not a replacement for the sealed LWC-84 reference solver.  It
    must pass same-protocol accuracy and runtime gates before it can be promoted.
    """

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
        target_reflection: float | None = None,
        damping_order: int | None = None,
        damping_c_ref_mps: float | None = None,
        kappa_max: float = 3.0,
        minimum_frequency_hz: float = 8.0,
        temporal_order: int = 2,
        match_fine_grid_restriction: bool = False,
    ) -> None:
        if grid.centering != "node" or grid.nx % 2 != 1 or grid.nz % 2 != 1:
            raise ValueError("KSpacePSTDSolver requires odd node-centred dimensions")
        if float(dt_s) <= 0.0 or not math.isfinite(float(dt_s)):
            raise ValueError("dt_s must be positive and finite")
        if float(c_ref_mps) <= 0.0 or not math.isfinite(float(c_ref_mps)):
            raise ValueError("c_ref_mps must be positive and finite")
        self.grid = grid
        self.boundaries = boundaries
        self.dt_s = float(dt_s)
        self.c_ref_mps = float(c_ref_mps)
        self.output_times_s = np.asarray(output_times_s, dtype=np.float64)
        self.device = torch.device(device)
        self.dtype = dtype
        self.temporal_order = int(temporal_order)
        if self.temporal_order not in (2, 4):
            raise ValueError("temporal_order must be 2 or 4")
        self.match_fine_grid_restriction = bool(match_fine_grid_restriction)
        if self.output_times_s.ndim != 1 or self.output_times_s.size < 1:
            raise ValueError("output_times_s must be a nonempty vector")
        if self.output_times_s[0] != 0.0:
            raise ValueError("output_times_s must start at zero")
        steps = np.rint(self.output_times_s / self.dt_s).astype(np.int64)
        if not np.allclose(
            steps * self.dt_s, self.output_times_s, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("all output times must align exactly to dt_s")
        if np.any(np.diff(steps) <= 0):
            raise ValueError("output times must be strictly increasing")
        self.output_steps = steps
        reflection = (
            float(boundaries.cpml_target_reflection)
            if target_reflection is None
            else float(target_reflection)
        )
        order = (
            int(boundaries.cpml_polynomial_order)
            if damping_order is None
            else int(damping_order)
        )
        if not 0.0 < reflection < 1.0:
            raise ValueError("target_reflection must lie between zero and one")
        if order < 1:
            raise ValueError("damping_order must be positive")
        self.target_reflection = reflection
        self.damping_order = order
        damping_reference = (
            self.c_ref_mps
            if damping_c_ref_mps is None
            else float(damping_c_ref_mps)
        )
        if not math.isfinite(damping_reference) or damping_reference <= 0.0:
            raise ValueError("damping_c_ref_mps must be positive and finite")

        npml = int(boundaries.npml)
        extended_nz, extended_nx = boundaries.extended_shape(grid)
        # The top and remote bottom endpoint are both pressure-release nodes in the
        # odd extension.  The bottom endpoint lies at the outside of the damping layer.
        odd_nz = 2 * (extended_nz - 1)
        kz = 2.0 * math.pi * torch.fft.fftfreq(
            odd_nz, d=float(grid.dz_m), device=self.device
        )
        kx = 2.0 * math.pi * torch.fft.rfftfreq(
            extended_nx, d=float(grid.dx_m), device=self.device
        )
        k_abs = torch.sqrt(kz[:, None].square() + kx[None, :].square())
        # torch.sinc(x) = sin(pi*x)/(pi*x).
        correction = torch.sinc(
            self.c_ref_mps * k_abs * self.dt_s / (2.0 * math.pi)
        ).square()
        self._second_x_multiplier = (-kx[None, :].square() * correction).to(
            self.dtype
        )
        self._second_z_multiplier = (-kz[:, None].square() * correction).to(
            self.dtype
        )
        # Reuse the production solver's exact CFS-CPML coefficients and memory
        # recursion.  Only the inactive physical-domain derivative is replaced by
        # the low-dispersion spectral derivative below.
        self._cpml_profiles = build_cfs_cpml_profiles(
            grid,
            boundaries,
            dt_s=self.dt_s,
            c_ref_mps=damping_reference,
            target_reflection=reflection,
            polynomial_order=order,
            kappa_max=float(kappa_max),
            minimum_frequency_hz=float(minimum_frequency_hz),
            device=self.device,
            dtype=self.dtype,
        )
        self._npml = npml

    def _spectral_second_derivatives(
        self, field: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if field.ndim != 3:
            raise ValueError("k-space field must be [batch,z,x]")
        # p(0)=p(L)=0 and p(-z)=-p(z) turn the FFT into a sine derivative in z.
        odd = torch.cat((field, -torch.flip(field[:, 1:-1], dims=(-2,))), dim=-2)
        spectrum = torch.fft.rfft2(odd, dim=(-2, -1))
        second_x = torch.fft.irfft2(
            spectrum * self._second_x_multiplier,
            s=odd.shape[-2:],
            dim=(-2, -1),
        )
        second_z = torch.fft.irfft2(
            spectrum * self._second_z_multiplier,
            s=odd.shape[-2:],
            dim=(-2, -1),
        )
        return (
            second_x[:, : field.shape[-2]],
            second_z[:, : field.shape[-2]],
        )

    def _spectral_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        second_x, second_z = self._spectral_second_derivatives(field)
        return second_x + second_z

    def _extend_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        npml = self._npml
        return F.pad(velocity[:, None], (npml, npml, 0, npml), mode="replicate")[:, 0]

    def simulate(
        self,
        velocity_mps,
        *,
        source_x_m,
        source_z_m,
        source_f0_hz,
        source_t0_s=None,
        source_amplitude=1.0,
    ) -> KSpaceSimulationResult:
        compute_started = time.perf_counter()
        velocity_np = np.asarray(velocity_mps)
        if velocity_np.ndim == 2:
            velocity_np = velocity_np[None]
        if velocity_np.ndim != 3 or velocity_np.shape[1:] != (
            self.grid.nz,
            self.grid.nx,
        ):
            raise ValueError(
                f"velocity must be [B,{self.grid.nz},{self.grid.nx}], got "
                f"{velocity_np.shape}"
            )
        if not np.isfinite(velocity_np).all() or float(velocity_np.min()) <= 0.0:
            raise ValueError("velocity must be positive and finite")
        batch = int(velocity_np.shape[0])
        source_x = _batch_parameter(source_x_m, batch=batch, name="source_x_m")
        source_z = _batch_parameter(source_z_m, batch=batch, name="source_z_m")
        f0 = _batch_parameter(source_f0_hz, batch=batch, name="source_f0_hz")
        amplitude = _batch_parameter(
            source_amplitude, batch=batch, name="source_amplitude"
        )
        t0 = (
            1.5 / f0
            if source_t0_s is None
            else _batch_parameter(source_t0_s, batch=batch, name="source_t0_s")
        )
        if np.any(f0 <= 0.0) or np.any(amplitude == 0.0):
            raise ValueError("source frequency must be positive and amplitude nonzero")

        physical_source = np.zeros(
            (batch, self.grid.nz, self.grid.nx), dtype=np.float32
        )
        delta = torch.zeros(
            self.boundaries.extended_shape(self.grid),
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0).expand(batch, -1, -1).clone()
        physical_z, physical_x = self.boundaries.physical_slices(self.grid)
        for index in range(batch):
            point = bilinear_point_source(
                float(source_x[index]),
                float(source_z[index]),
                nx=self.grid.nx,
                nz=self.grid.nz,
                dx_m=self.grid.dx_m,
                dz_m=self.grid.dz_m,
                centering="node",
            )
            physical_source[index] = point.source_map
            delta[index, physical_z, physical_x] = torch.as_tensor(
                point.delta_h, device=self.device, dtype=self.dtype
            )

        velocity = torch.as_tensor(
            velocity_np, device=self.device, dtype=self.dtype
        )
        velocity_extended = self._extend_velocity(velocity)
        maximum_step = int(self.output_steps[-1])
        internal_time = (
            torch.arange(
                maximum_step + 1, device=self.device, dtype=self.dtype
            )[:, None]
            * self.dt_s
        )
        f0_tensor = torch.as_tensor(f0, device=self.device, dtype=self.dtype)[None]
        t0_tensor = torch.as_tensor(t0, device=self.device, dtype=self.dtype)[None]
        amplitude_tensor = torch.as_tensor(
            amplitude, device=self.device, dtype=self.dtype
        )[None]
        a = math.pi**2 * f0_tensor.square()
        tau = internal_time - t0_tensor
        exponential = torch.exp(-a * tau.square())
        wavelet_internal = amplitude_tensor * (
            (1.0 - 2.0 * a * tau.square()) * exponential
        )
        wavelet_first = amplitude_tensor * (
            (-6.0 * a * tau + 4.0 * a.square() * tau.pow(3)) * exponential
        )
        wavelet_second = amplitude_tensor * (
            (
                -6.0 * a
                + 24.0 * a.square() * tau.square()
                - 8.0 * a.pow(3) * tau.pow(4)
            )
            * exponential
        )

        state_shape = velocity_extended.shape
        p0 = torch.zeros(state_shape, device=self.device, dtype=self.dtype)
        output = torch.zeros(
            (batch, self.output_times_s.size, self.grid.nz, self.grid.nx),
            device=self.device,
            dtype=self.dtype,
        )
        step_to_output = {
            int(step): position
            for position, step in enumerate(self.output_steps.tolist())
        }

        def save(step: int, field: torch.Tensor) -> None:
            position = step_to_output.get(int(step))
            if position is None:
                return
            physical = field[:, physical_z, physical_x]
            physical[:, 0, :] = 0.0
            output[:, position] = physical

        save(0, p0)
        cpml = CFSCPMLOperator(
            self._cpml_profiles,
            dx_m=self.grid.dx_m,
            dz_m=self.grid.dz_m,
        )
        dt2 = self.dt_s**2
        if self.temporal_order == 4:
            q0 = wavelet_internal[0, :, None, None] * delta
            qt0 = wavelet_first[0, :, None, None] * delta
            qtt0 = wavelet_second[0, :, None, None] * delta
            fourth0 = velocity_extended.square() * self._spectral_laplacian(q0) + qtt0
            p1 = (
                0.5 * dt2 * q0
                + self.dt_s**3 * qt0 / 6.0
                + self.dt_s**4 * fourth0 / 24.0
            )
        else:
            p1 = dt2 * wavelet_internal[0, :, None, None] * delta
        p1[:, 0, :] = 0.0
        p1[:, -1, :] = 0.0
        p1[:, :, 0] = 0.0
        p1[:, :, -1] = 0.0
        save(1, p1)
        p_nm1, p_n = p0, p1
        for step in range(1, maximum_step):
            spectral_x, spectral_z = self._spectral_second_derivatives(p_n)
            finite_x, finite_z = second_derivatives8(
                p_n,
                dx_m=self.grid.dx_m,
                dz_m=self.grid.dz_m,
                boundary="free_surface",
            )
            acceleration = cpml.apply(
                p_n, velocity_extended, update_memory=True
            )
            correction = torch.where(
                self._cpml_profiles.active_x,
                torch.zeros_like(spectral_x),
                spectral_x - finite_x,
            ) + torch.where(
                self._cpml_profiles.active_z,
                torch.zeros_like(spectral_z),
                spectral_z - finite_z,
            )
            acceleration = acceleration + velocity_extended.square() * correction
            acceleration = acceleration + wavelet_internal[step, :, None, None] * delta
            p_np1 = 2.0 * p_n - p_nm1 + dt2 * acceleration
            if self.temporal_order == 4:
                fourth = (
                    velocity_extended.square()
                    * self._spectral_laplacian(acceleration)
                    + wavelet_second[step, :, None, None] * delta
                )
                p_np1 = p_np1 + self.dt_s**4 * fourth / 12.0
            # The two outer endpoints close the odd/periodic spectral extension in a
            # region where the sponge has already attenuated outgoing energy.
            p_np1[:, 0, :] = 0.0
            p_np1[:, -1, :] = 0.0
            p_np1[:, :, 0] = 0.0
            p_np1[:, :, -1] = 0.0
            save(step + 1, p_np1)
            p_nm1, p_n = p_n, p_np1

        if self.match_fine_grid_restriction:
            # The sealed teacher applies [1,4,6,4,1]/16 on its 5 m grid before
            # retaining every second node.  On already-coincident 10 m nodes the
            # corresponding band-limiting transfer function is cos(kh/4)^4 per
            # axis (h=10 m).  Time is a leading batch dimension here.  Filtering in
            # modest chunks is mathematically identical to the old per-save version;
            # it avoids both 401 tiny launches and the poor CPU cache behaviour of a
            # single 401-frame transform.
            kz = 2.0 * math.pi * torch.fft.fftfreq(
                output.shape[-2],
                d=float(self.grid.dz_m),
                device=output.device,
            )[:, None]
            kx = 2.0 * math.pi * torch.fft.rfftfreq(
                output.shape[-1],
                d=float(self.grid.dx_m),
                device=output.device,
            )[None, :]
            response = torch.cos(
                kz * float(self.grid.dz_m) / 4.0
            ).pow(4) * torch.cos(
                kx * float(self.grid.dx_m) / 4.0
            ).pow(4)
            chunk_size = 32
            for start in range(0, output.shape[1], chunk_size):
                stop = min(start + chunk_size, output.shape[1])
                spectrum = torch.fft.rfft2(
                    output[:, start:stop], dim=(-2, -1)
                )
                output[:, start:stop] = torch.fft.irfft2(
                    spectrum * response,
                    s=output.shape[-2:],
                    dim=(-2, -1),
                )
            output[:, :, 0, :] = 0.0

        output_np = output.detach().cpu().numpy().astype(np.float32, copy=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - compute_started
        if not np.isfinite(output_np).all():
            raise FloatingPointError("non-finite pressure in k-space output")
        saved_wavelet = np.stack(
            [
                amplitude[index]
                * ricker_triplet(
                    self.output_times_s,
                    f0_hz=float(f0[index]),
                    t0_s=float(t0[index]),
                )[0]
                for index in range(batch)
            ],
            axis=0,
        ).astype(np.float32)
        metrics = []
        for index in range(batch):
            energies = np.sum(output_np[index].astype(np.float64) ** 2, axis=(1, 2))
            metrics.append(
                {
                    "finite": True,
                    "dt_used_s": self.dt_s,
                    "internal_steps": maximum_step,
                    "temporal_order": self.temporal_order,
                    "match_fine_grid_restriction": self.match_fine_grid_restriction,
                    "compute_elapsed_s": float(elapsed),
                    "vmin_mps": float(velocity_np[index].min()),
                    "vmax_mps": float(velocity_np[index].max()),
                    "qc_max_abs": float(np.abs(output_np[index]).max()),
                    "qc_final_energy_ratio": float(
                        energies[-1] / max(float(energies.max()), 1.0e-30)
                    ),
                }
            )
        return KSpaceSimulationResult(
            wavefield=output_np,
            velocity_saved_mps=velocity_np.astype(np.float32, copy=False),
            source_map_saved=physical_source,
            source_wavelet=saved_wavelet,
            metrics=metrics,
        )


__all__ = ["KSpacePSTDSolver", "KSpaceSimulationResult"]
