#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.lwc84 import apply_l_operator, lwc84_startup, lwc84_step
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from fno_acoustic.data_generation.stencils import laplacian8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/lwc84_numerical_validation_401to201_801/convergence.json")
    args = parser.parse_args()
    spatial_errors, spacings = [], []
    for n in (16, 20, 24, 28):
        h = 2.0 * math.pi / n
        coordinate = torch.arange(n, dtype=torch.float64) * h
        zz, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        field = torch.sin(xx) + torch.cos(2.0 * zz)
        exact = -torch.sin(xx) - 4.0 * torch.cos(2.0 * zz)
        numeric = laplacian8(field, dx_m=h, dz_m=h, boundary="periodic")
        spatial_errors.append(float(torch.linalg.vector_norm(numeric - exact) / torch.linalg.vector_norm(exact)))
        spacings.append(h)
    spatial_orders = [
        math.log(spatial_errors[i] / spatial_errors[i + 1]) / math.log(spacings[i] / spacings[i + 1])
        for i in range(3)
    ]

    n, t_end = 32, 0.8
    h = 2.0 * math.pi / n
    coordinate = torch.arange(n, dtype=torch.float64) * h
    zz, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    p0, velocity = torch.sin(xx), torch.ones_like(xx)
    zero = torch.zeros_like(p0)
    lp0 = apply_l_operator(p0, velocity, dx_m=h, dz_m=h, boundary="periodic")
    omega = math.sqrt(-float((lp0 * p0).sum() / (p0 * p0).sum()))
    temporal_errors, time_steps = [], []
    for steps in (10, 20, 40, 80):
        dt = t_end / steps
        p_nm1 = p0
        p_n = lwc84_startup(
            p0=p0, pt0=zero, q0=zero, qt0=zero, qtt0=zero, velocity_mps=velocity,
            dx_m=h, dz_m=h, dt_s=dt, boundary="periodic"
        )
        for _ in range(1, steps):
            p_nm1, p_n = p_n, lwc84_step(
                p_nm1=p_nm1, p_n=p_n, q_n=zero, qtt_n=zero, velocity_mps=velocity,
                dx_m=h, dz_m=h, dt_s=dt, boundary="periodic"
            )
        exact = p0 * math.cos(omega * t_end)
        temporal_errors.append(float(torch.linalg.vector_norm(p_n - exact) / torch.linalg.vector_norm(exact)))
        time_steps.append(dt)
    temporal_orders = [math.log(temporal_errors[i] / temporal_errors[i + 1]) / math.log(2.0) for i in range(3)]

    cpu_gpu = None
    if torch.cuda.is_available():
        grid = AcousticGrid(nx=41, nz=41, dx_m=5.0, dz_m=5.0, lx_m=200.0, lz_m=200.0, centering="node")
        common = dict(
            grid=grid, boundaries=BoundaryConfig(npml=8), dt_s=0.0002,
            output_times_s=np.asarray([0.0, 0.01, 0.02]), c_ref_mps=3000.0
        )
        velocity_np = np.full((41, 41), 2000.0)
        source = dict(source_x_m=103.25, source_z_m=52.75, source_f0_hz=80.0, source_amplitude=1.0)
        cpu = LWC84CPMLSolver(**common, device="cpu", dtype=torch.float64).simulate(velocity_np, **source).wavefield
        gpu = LWC84CPMLSolver(**common, device="cuda", dtype=torch.float32).simulate(velocity_np.astype(np.float32), **source).wavefield
        cpu_gpu = float(np.linalg.norm(cpu - gpu) / np.linalg.norm(cpu))
    payload = {
        "spatial_relative_errors": spatial_errors,
        "observed_spatial_orders": spatial_orders,
        "temporal_relative_errors": temporal_errors,
        "observed_temporal_orders": temporal_orders,
        "cpu_float64_vs_gpu_float32_relative_l2": cpu_gpu,
        "passed": min(spatial_orders) > 7.5 and min(temporal_orders) > 3.7 and (cpu_gpu is None or cpu_gpu < 3.0e-3),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
