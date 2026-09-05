from __future__ import annotations

import numpy as np
import torch

from .analytic_halfspace import analytic_halfspace_reference
from .grid import AcousticGrid, OutputTimeGrid
from .source import bilinear_point_source


def simulate_uniform_gpu_analytic_compatible(
    grid: AcousticGrid,
    time: OutputTimeGrid,
    *,
    c_mps: float,
    source_xy_m: tuple[float, float],
    f0_hz: float,
    device: str = "cuda",
) -> np.ndarray:
    if str(device).lower() == "cpu":
        return analytic_halfspace_reference(grid, time, c_mps=c_mps, source_xy_m=source_xy_m, f0_hz=f0_hz)
    torch_device = torch.device(device)
    src = bilinear_point_source(source_xy_m[0], source_xy_m[1], nx=grid.nx, nz=grid.nz, dx_m=grid.dx_m, dz_m=grid.dz_m)
    x = torch.as_tensor(grid.x_m, dtype=torch.float64, device=torch_device)
    z = torch.as_tensor(grid.z_m, dtype=torch.float64, device=torch_device)
    xx, zz = torch.meshgrid(x, z, indexing="xy")
    t = torch.as_tensor(time.t_s, dtype=torch.float64, device=torch_device)
    out = torch.zeros((grid.nz, grid.nx, time.nt_out), dtype=torch.float64, device=torch_device)
    for (z_idx, x_idx), weight in zip(src.indices.tolist(), src.weights.tolist()):
        sx = float(grid.x_m[int(x_idx)])
        sz = float(grid.z_m[int(z_idx)])
        rd = torch.sqrt((xx - sx) ** 2 + (zz - sz) ** 2)
        ri = torch.sqrt((xx - sx) ** 2 + (zz + sz) ** 2)
        for distance, sign in ((rd, 1.0), (ri, -1.0)):
            tau = t.view(1, 1, -1) - distance.unsqueeze(-1) / float(c_mps)
            active = tau >= 0.0
            tau_pos = torch.clamp(tau, min=0.0)
            a = 0.6 * float(f0_hz) * tau_pos - 1.0
            pulse = -9.6 * float(f0_hz) * a * torch.exp(-8.0 * a**2)
            pulse = torch.where(active, pulse, torch.zeros_like(pulse))
            out = out + float(weight) * sign * pulse / torch.sqrt(torch.clamp(distance.unsqueeze(-1), min=0.5))
    out[0, :, :] = 0.0
    return out.to(dtype=torch.float32).detach().cpu().numpy()
