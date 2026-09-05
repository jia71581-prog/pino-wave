"""Opt-in device-resident output for the fused LWC84 solver.

The legacy :meth:`FusedLWC84CPMLSolver.simulate` API is inherited unchanged.
Only :meth:`simulate_device` returns a CUDA tensor and deliberately avoids
materializing the parent wavefield on the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from .lwc84 import lwc84_startup
from .ricker import ricker_triplet
from .solver_lwc84 import _batch_parameter
from .solver_lwc84_fused import FusedLWC84CPMLSolver
from .source import bilinear_point_source


@dataclass(frozen=True)
class DeviceLWC84SimulationResult:
    wavefield_device: torch.Tensor
    source_map_saved: np.ndarray
    source_map_solver: np.ndarray
    source_wavelet: np.ndarray
    metrics: list[dict[str, float | int | bool]]
    transfer_ledger: dict[str, int | bool]


class DeviceResidentFusedLWC84CPMLSolver(FusedLWC84CPMLSolver):
    """Fused solver with an explicit device-resident result method."""

    def simulate_device(
        self,
        velocity_mps,
        *,
        source_x_m,
        source_z_m,
        source_f0_hz,
        source_t0_s=None,
        source_amplitude=1.0,
    ) -> DeviceLWC84SimulationResult:
        if self.device.type != "cuda" or self.dtype != torch.float32:
            raise RuntimeError("device-resident LWC84 requires CUDA float32")
        velocity_np = np.asarray(velocity_mps)
        if velocity_np.ndim == 2:
            velocity_np = velocity_np[None, ...]
        if velocity_np.ndim != 3 or velocity_np.shape[1:] != (self.grid.nz, self.grid.nx):
            raise ValueError(f"velocity must be [B,{self.grid.nz},{self.grid.nx}], got {velocity_np.shape}")
        if not np.isfinite(velocity_np).all() or float(np.min(velocity_np)) <= 0.0:
            raise ValueError("velocity must be positive and finite")
        batch = int(velocity_np.shape[0])
        source_x = _batch_parameter(source_x_m, batch=batch, name="source_x_m")
        source_z = _batch_parameter(source_z_m, batch=batch, name="source_z_m")
        f0 = _batch_parameter(source_f0_hz, batch=batch, name="source_f0_hz")
        amplitude = _batch_parameter(source_amplitude, batch=batch, name="source_amplitude")
        t0 = 1.5 / f0 if source_t0_s is None else _batch_parameter(source_t0_s, batch=batch, name="source_t0_s")
        if np.any(f0 <= 0.0) or np.any(amplitude == 0.0):
            raise ValueError("source frequency must be positive and amplitude must be nonzero")

        f0_device = torch.as_tensor(f0, dtype=self.dtype, device=self.device).view(batch, 1, 1)
        t0_device = torch.as_tensor(t0, dtype=self.dtype, device=self.device).view(batch, 1, 1)
        amplitude_device = torch.as_tensor(amplitude, dtype=self.dtype, device=self.device).view(batch, 1, 1)
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
        delta_transfer_bytes = 0
        for index in range(batch):
            fine = bilinear_point_source(
                float(source_x[index]), float(source_z[index]), nx=self.grid.nx, nz=self.grid.nz,
                dx_m=self.grid.dx_m, dz_m=self.grid.dz_m, centering="node",
            )
            source_solver[index] = fine.source_map
            delta_tensor = torch.as_tensor(fine.delta_h, device=self.device, dtype=self.dtype)
            delta_transfer_bytes += int(delta_tensor.numel() * delta_tensor.element_size())
            delta[index, physical_z, physical_x] = delta_tensor
            saved_source = bilinear_point_source(
                float(source_x[index]), float(source_z[index]), nx=saved_nx, nz=saved_nz,
                dx_m=factor * self.grid.dx_m, dz_m=factor * self.grid.dz_m, centering="node",
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
            (-6.0 * ricker_a.unsqueeze(0) * tau + 4.0 * ricker_a.square().unsqueeze(0) * tau.pow(3)) * exponential,
            (-6.0 * ricker_a.unsqueeze(0) + 24.0 * ricker_a.square().unsqueeze(0) * tau.square()
             - 8.0 * ricker_a.pow(3).unsqueeze(0) * tau.pow(4)) * exponential,
        )

        def source_terms(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return tuple(amplitude_device * values[int(step)] * delta for values in source_series)

        def source_sequences(start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
            scale = amplitude_device.unsqueeze(0) * delta.unsqueeze(0)
            return (source_series[0][int(start):int(stop)] * scale,
                    source_series[2][int(start):int(stop)] * scale)

        p0 = torch.zeros_like(velocity)
        pt0 = torch.zeros_like(velocity)
        q0, qt0, qtt0 = source_terms(0)
        p1 = lwc84_startup(
            p0=p0, pt0=pt0, q0=q0, qt0=qt0, qtt0=qtt0, velocity_mps=velocity,
            dx_m=self.grid.dx_m, dz_m=self.grid.dz_m, dt_s=self.dt_s,
            boundary="free_surface", spatial_coefficients=self.spatial_coefficients,
        )
        p1[:, 0, :] = 0.0
        p1[:, -1, :] = 0.0
        p1[:, :, 0] = 0.0
        p1[:, :, -1] = 0.0
        output = torch.zeros(
            (batch, self.output_times_s.size, saved_nz, saved_nx),
            dtype=self.dtype, device=self.device,
        )
        step_to_output = {int(step): index for index, step in enumerate(self.output_steps.tolist())}

        def save(step: int, field: torch.Tensor) -> None:
            output_index = step_to_output.get(int(step))
            if output_index is not None:
                saved = self._restrict_output(field[:, physical_z, physical_x])
                saved[:, 0, :] = 0.0
                output[:, output_index] = saved

        save(0, p0)
        if maximum_step >= 1:
            save(1, p1)
        p_nm1, p_n = p0, p1
        memories = tuple(torch.zeros_like(velocity) for _ in range(4))
        graph_runner = self._graph_runner(shape) if self.cuda_graphs else None
        graph_state_loaded = False
        if self._saved_interval_block is None or maximum_step <= 1:
            for step in range(1, maximum_step):
                q_n, _, qtt_n = source_terms(step)
                p_nm1, p_n, *next_memories = self._time_step(
                    p_nm1, p_n, *memories, q_n, qtt_n, velocity,
                )
                memories = tuple(next_memories)
                save(step + 1, p_n)
        else:
            assert self._saved_interval_steps is not None
            assert self._block_step_count is not None
            block_steps = self._block_step_count
            first_output_step = int(self.output_steps[1])
            previous_output_step = 0
            for output_step in self.output_steps[1:].tolist():
                stop = int(output_step)
                start = 1 if stop == first_output_step else int(previous_output_step)
                while start + block_steps <= stop:
                    block_stop = start + block_steps
                    q_sequence, qtt_sequence = source_sequences(start, block_stop)
                    if graph_runner is None:
                        p_nm1, p_n, *next_memories = self._saved_interval_block(
                            p_nm1, p_n, *memories, q_sequence, qtt_sequence, velocity,
                        )
                    else:
                        if not graph_state_loaded:
                            graph_runner.load_state((p_nm1, p_n, *memories), velocity)
                            graph_state_loaded = True
                        p_nm1, p_n, *next_memories = graph_runner.replay(q_sequence, qtt_sequence)
                    memories = tuple(next_memories)
                    start = block_stop
                while start < stop:
                    q_n, _, qtt_n = source_terms(start)
                    p_nm1, p_n, *next_memories = self._time_step(
                        p_nm1, p_n, *memories, q_n, qtt_n, velocity,
                    )
                    memories = tuple(next_memories)
                    start += 1
                save(stop, p_n)
                previous_output_step = stop

        owned_output = output.detach().contiguous().clone()
        torch._assert_async(torch.isfinite(owned_output).all(), "non-finite device-resident parent")
        source_wavelet = np.empty((batch, self.output_times_s.size), dtype=np.float32)
        metrics: list[dict[str, float | int | bool]] = []
        for index in range(batch):
            source_wavelet[index] = (
                amplitude[index] * ricker_triplet(
                    self.output_times_s, f0_hz=float(f0[index]), t0_s=float(t0[index]),
                )[0]
            ).astype(np.float32)
            vmax = float(np.max(velocity_np[index]))
            cfl = vmax * self.dt_s * math.sqrt(1.0 / self.grid.dx_m**2 + 1.0 / self.grid.dz_m**2)
            axis_spectral_radius = 2048.0 / 315.0
            if self.spatial_coefficients is not None:
                theta = np.linspace(0.0, math.pi, 8193, dtype=np.float64)
                axis_spectral_radius = abs(float(np.min(self.spatial_coefficients.symbol(theta))))
            qmax = self.dt_s**2 * vmax**2 * axis_spectral_radius * (
                1.0 / self.grid.dx_m**2 + 1.0 / self.grid.dz_m**2
            )
            metrics.append({
                "vmin_mps": float(np.min(velocity_np[index])), "vmax_mps": vmax,
                "cfl_2d": cfl, "lwc_qmax": qmax, "dt_used_s": self.dt_s,
                "finite_asserted_on_device": True, "output_frame_count": int(owned_output.shape[1]),
            })
        h2d_bytes = int(velocity.numel() * velocity.element_size()) + delta_transfer_bytes
        h2d_bytes += int((f0_device.numel() + t0_device.numel() + amplitude_device.numel()) * 4)
        return DeviceLWC84SimulationResult(
            wavefield_device=owned_output,
            source_map_saved=source_saved,
            source_map_solver=source_solver,
            source_wavelet=source_wavelet,
            metrics=metrics,
            transfer_ledger={
                "solver_h2d_call_count": 4 + batch,
                "solver_h2d_bytes": h2d_bytes,
                "parent_d2h_call_count": 0,
                "parent_d2h_bytes": 0,
                "output_owned_contiguous_cuda_float32": True,
            },
        )


__all__ = ["DeviceLWC84SimulationResult", "DeviceResidentFusedLWC84CPMLSolver"]
