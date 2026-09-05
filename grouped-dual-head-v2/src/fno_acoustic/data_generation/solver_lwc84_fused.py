from __future__ import annotations

import math
import time

import numpy as np
import torch

from .fused_lwc84 import (
    make_cfs_cpml_lwc84_block,
    make_cfs_cpml_lwc84_step,
)
from .lwc84 import lwc84_startup
from .restriction import restrict_nodal_2x
from .ricker import ricker_triplet
from .solver_lwc84 import (
    LWC84CPMLSolver,
    LWC84SimulationResult,
    _batch_parameter,
)
from .source import bilinear_point_source


class _CUDAGraphedLWC84Block:
    """Replay one fixed-shape functional block with recurrent static state."""

    def __init__(
        self,
        block,
        *,
        state_shape: tuple[int, ...],
        step_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA-graph LWC84 execution requires a CUDA device")
        self.state_shape = tuple(int(value) for value in state_shape)
        self.step_count = int(step_count)
        self.state = tuple(
            torch.zeros(self.state_shape, device=device, dtype=dtype)
            for _ in range(6)
        )
        self.q_sequence = torch.zeros(
            (self.step_count, *self.state_shape), device=device, dtype=dtype
        )
        self.qtt_sequence = torch.zeros_like(self.q_sequence)
        self.velocity = torch.ones(self.state_shape, device=device, dtype=dtype)

        # Allocate all lazy CUDA-library state before capture. The captured copy
        # closes the recurrence by writing each block output back to its static
        # input buffer for the next replay.
        block(
            *self.state,
            self.q_sequence,
            self.qtt_sequence,
            self.velocity,
        )
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            outputs = block(
                *self.state,
                self.q_sequence,
                self.qtt_sequence,
                self.velocity,
            )
            for destination, source in zip(self.state, outputs, strict=True):
                destination.copy_(source)

    def load_state(
        self,
        state: tuple[torch.Tensor, ...],
        velocity: torch.Tensor,
    ) -> None:
        if len(state) != 6 or tuple(velocity.shape) != self.state_shape:
            raise ValueError("CUDA-graph LWC84 state shape changed")
        for destination, source in zip(self.state, state, strict=True):
            if tuple(source.shape) != self.state_shape:
                raise ValueError("CUDA-graph LWC84 state shape changed")
            destination.copy_(source)
        self.velocity.copy_(velocity)

    def replay(
        self,
        q_sequence: torch.Tensor,
        qtt_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if q_sequence.shape != self.q_sequence.shape:
            raise ValueError("CUDA-graph source block shape changed")
        if qtt_sequence.shape != self.qtt_sequence.shape:
            raise ValueError("CUDA-graph source-second-derivative block shape changed")
        self.q_sequence.copy_(q_sequence)
        self.qtt_sequence.copy_(qtt_sequence)
        self.graph.replay()
        return self.state


class FusedLWC84CPMLSolver(LWC84CPMLSolver):
    """Drop-in LWC84 solver with a functional, optionally compiled time loop.

    The public setup, source convention, startup, saving, restriction, QC, and
    result schema remain identical to :class:`LWC84CPMLSolver`. Only the
    production CPML/LWC84 inner step is fused. Compilation is deliberately
    opt-in and can be warmed once for a fixed deployment batch shape before a
    synchronized per-instance runtime measurement.
    """

    def __init__(
        self,
        *,
        compile_graph: bool = False,
        compile_mode: str = "reduce-overhead",
        fuse_saved_interval: bool = False,
        block_step_count: int | None = None,
        cuda_graphs: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.compile_graph = bool(compile_graph)
        self.compile_mode = str(compile_mode)
        self.cuda_graphs = bool(cuda_graphs)
        self.fuse_saved_interval = bool(fuse_saved_interval) or self.cuda_graphs
        if self.cuda_graphs and self.device.type != "cuda":
            raise ValueError("CUDA-graph LWC84 execution requires a CUDA device")
        if self.cuda_graphs and self.compile_graph:
            raise ValueError("CUDA graphs and torch.compile are mutually exclusive")
        if block_step_count is not None and int(block_step_count) <= 0:
            raise ValueError("block_step_count must be positive when provided")
        self._time_step = make_cfs_cpml_lwc84_step(
            profiles=self.profiles,
            dx_m=self.grid.dx_m,
            dz_m=self.grid.dz_m,
            dt_s=self.dt_s,
            spatial_coefficients=self.spatial_coefficients,
            compile_graph=self.compile_graph,
            compile_mode=self.compile_mode,
        )
        self._saved_interval_steps: int | None = None
        self._block_step_count: int | None = None
        self._saved_interval_block = None
        self._cuda_graph_runner: _CUDAGraphedLWC84Block | None = None
        if self.fuse_saved_interval or block_step_count is not None:
            differences = np.diff(self.output_steps)
            if differences.size == 0 or np.any(differences != differences[0]):
                raise ValueError(
                    "blocked execution requires uniformly spaced output steps"
                )
            self._saved_interval_steps = int(differences[0])
            self._block_step_count = (
                self._saved_interval_steps
                if block_step_count is None
                else int(block_step_count)
            )
            if self._saved_interval_steps % self._block_step_count != 0:
                raise ValueError(
                    "block_step_count must divide the saved-frame interval"
                )
            self._saved_interval_block = make_cfs_cpml_lwc84_block(
                profiles=self.profiles,
                dx_m=self.grid.dx_m,
                dz_m=self.grid.dz_m,
                dt_s=self.dt_s,
                step_count=self._block_step_count,
                spatial_coefficients=self.spatial_coefficients,
                compile_graph=self.compile_graph,
                compile_mode=self.compile_mode,
            )

    def _graph_runner(
        self, state_shape: tuple[int, ...]
    ) -> _CUDAGraphedLWC84Block:
        if not self.cuda_graphs or self._saved_interval_block is None:
            raise RuntimeError("CUDA-graph LWC84 block is not enabled")
        assert self._block_step_count is not None
        if (
            self._cuda_graph_runner is None
            or self._cuda_graph_runner.state_shape != tuple(state_shape)
        ):
            self._cuda_graph_runner = _CUDAGraphedLWC84Block(
                self._saved_interval_block,
                state_shape=tuple(state_shape),
                step_count=self._block_step_count,
                device=self.device,
                dtype=self.dtype,
            )
        return self._cuda_graph_runner

    def warmup(self, *, batch: int = 1) -> float:
        """Materialize the fixed-shape compiled graph without retaining state."""

        count = int(batch)
        if count <= 0:
            raise ValueError("warmup batch must be positive")
        shape = (count, *self.boundaries.extended_shape(self.grid))
        zero = torch.zeros(shape, dtype=self.dtype, device=self.device)
        velocity = torch.full_like(zero, 2000.0)
        started = time.perf_counter()
        self._time_step(
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            velocity,
        )
        if self._saved_interval_block is not None:
            assert self._block_step_count is not None
            sequence = torch.zeros(
                (self._block_step_count, *shape),
                dtype=self.dtype,
                device=self.device,
            )
            self._saved_interval_block(
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                sequence,
                sequence,
                velocity,
            )
            if self.cuda_graphs:
                self._graph_runner(shape)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return float(time.perf_counter() - started)

    def simulate(
        self,
        velocity_mps,
        *,
        source_x_m,
        source_z_m,
        source_f0_hz,
        source_t0_s=None,
        source_amplitude=1.0,
    ) -> LWC84SimulationResult:
        compute_started = time.perf_counter()
        velocity_np = np.asarray(velocity_mps)
        if velocity_np.ndim == 2:
            velocity_np = velocity_np[None, ...]
        if velocity_np.ndim != 3 or velocity_np.shape[1:] != (
            self.grid.nz,
            self.grid.nx,
        ):
            raise ValueError(
                f"velocity must be [B,{self.grid.nz},{self.grid.nx}], got {velocity_np.shape}"
            )
        if not np.isfinite(velocity_np).all() or float(np.min(velocity_np)) <= 0.0:
            raise ValueError("velocity must be positive and finite before wavefield generation")
        batch = int(velocity_np.shape[0])
        source_x = _batch_parameter(source_x_m, batch=batch, name="source_x_m")
        source_z = _batch_parameter(source_z_m, batch=batch, name="source_z_m")
        f0 = _batch_parameter(source_f0_hz, batch=batch, name="source_f0_hz")
        amplitude = _batch_parameter(
            source_amplitude, batch=batch, name="source_amplitude"
        )
        if source_t0_s is None:
            t0 = 1.5 / f0
        else:
            t0 = _batch_parameter(source_t0_s, batch=batch, name="source_t0_s")
        if np.any(f0 <= 0.0) or np.any(amplitude == 0.0):
            raise ValueError("source frequency must be positive and amplitude must be nonzero")

        f0_device = torch.as_tensor(
            f0, dtype=self.dtype, device=self.device
        ).view(batch, 1, 1)
        t0_device = torch.as_tensor(
            t0, dtype=self.dtype, device=self.device
        ).view(batch, 1, 1)
        amplitude_device = torch.as_tensor(
            amplitude, dtype=self.dtype, device=self.device
        ).view(batch, 1, 1)
        ricker_a = math.pi**2 * f0_device.square()

        velocity = self._extend_velocity(velocity_np.astype(np.float64, copy=False))
        shape = tuple(int(value) for value in velocity.shape)
        delta = torch.zeros(shape, dtype=self.dtype, device=self.device)
        source_solver = np.zeros(
            (batch, self.grid.nz, self.grid.nx), dtype=np.float32
        )
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
        internal_times = (
            torch.arange(
                maximum_step + 1, dtype=self.dtype, device=self.device
            ).view(-1, 1, 1, 1)
            * self.dt_s
        )
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

        def source_terms(
            step: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return tuple(
                amplitude_device * values[int(step)] * delta
                for values in source_series
            )

        def source_sequences(
            start: int,
            stop: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            scale = amplitude_device.unsqueeze(0) * delta.unsqueeze(0)
            return (
                source_series[0][int(start) : int(stop)] * scale,
                source_series[2][int(start) : int(stop)] * scale,
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
        step_to_output = {
            int(step): index for index, step in enumerate(self.output_steps.tolist())
        }

        def save(step: int, field: torch.Tensor) -> None:
            output_index = step_to_output.get(int(step))
            if output_index is None:
                return
            physical = field[:, physical_z, physical_x]
            saved = self._restrict_output(physical)
            saved[:, 0, :] = 0.0
            output[:, output_index] = saved

        save(0, p0)
        if maximum_step >= 1:
            save(1, p1)
        p_nm1, p_n = p0, p1
        memories = tuple(torch.zeros_like(velocity) for _ in range(4))
        graph_runner = (
            self._graph_runner(shape) if self.cuda_graphs else None
        )
        graph_state_loaded = False
        if self._saved_interval_block is None or maximum_step <= 1:
            for step in range(1, maximum_step):
                q_n, _, qtt_n = source_terms(step)
                p_nm1, p_n, *next_memories = self._time_step(
                    p_nm1,
                    p_n,
                    *memories,
                    q_n,
                    qtt_n,
                    velocity,
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
                # Startup already materializes p_1, so only the first output
                # interval begins at recurrence index one. Later intervals
                # begin at the preceding saved step.
                start = 1 if stop == first_output_step else int(previous_output_step)
                while start + block_steps <= stop:
                    block_stop = start + block_steps
                    q_sequence, qtt_sequence = source_sequences(start, block_stop)
                    if graph_runner is None:
                        p_nm1, p_n, *next_memories = self._saved_interval_block(
                            p_nm1,
                            p_n,
                            *memories,
                            q_sequence,
                            qtt_sequence,
                            velocity,
                        )
                    else:
                        if not graph_state_loaded:
                            graph_runner.load_state(
                                (p_nm1, p_n, *memories), velocity
                            )
                            graph_state_loaded = True
                        p_nm1, p_n, *next_memories = graph_runner.replay(
                            q_sequence, qtt_sequence
                        )
                    memories = tuple(next_memories)
                    start = block_stop
                while start < stop:
                    q_n, _, qtt_n = source_terms(start)
                    p_nm1, p_n, *next_memories = self._time_step(
                        p_nm1,
                        p_n,
                        *memories,
                        q_n,
                        qtt_n,
                        velocity,
                    )
                    memories = tuple(next_memories)
                    start += 1
                save(stop, p_n)
                previous_output_step = stop

        output_np = output.detach().cpu().numpy().astype(np.float32, copy=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        compute_elapsed_s = time.perf_counter() - compute_started
        if not np.isfinite(output_np).all():
            raise FloatingPointError("non-finite pressure in saved output")
        velocity_saved = (
            self._restrict_output(
                torch.as_tensor(
                    velocity_np, dtype=self.dtype, device=self.device
                )
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        source_wavelet = np.empty(
            (batch, self.output_times_s.size), dtype=np.float32
        )
        metrics: list[dict[str, float | int | bool]] = []
        for index in range(batch):
            source_wavelet[index] = (
                amplitude[index]
                * ricker_triplet(
                    self.output_times_s,
                    f0_hz=float(f0[index]),
                    t0_s=float(t0[index]),
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
            energies = np.sum(
                output_np[index].astype(np.float64) ** 2, axis=(1, 2)
            )
            metrics.append(
                {
                    "vmin_mps": float(np.min(velocity_np[index])),
                    "vmax_mps": vmax,
                    "cfl_2d": cfl,
                    "lwc_qmax": qmax,
                    "dt_used_s": self.dt_s,
                    "qc_max_abs": float(np.max(np.abs(output_np[index]))),
                    "qc_final_energy_ratio": float(
                        energies[-1] / max(float(np.max(energies)), 1.0e-30)
                    ),
                    "finite": bool(np.isfinite(output_np[index]).all()),
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
        )


__all__ = ["FusedLWC84CPMLSolver"]
