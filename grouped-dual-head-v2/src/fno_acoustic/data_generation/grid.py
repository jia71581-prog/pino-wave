from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcousticGrid:
    nx: int = 400
    nz: int = 400
    dx_m: float = 5.0
    dz_m: float = 5.0
    lx_m: float | None = None
    lz_m: float | None = None
    centering: str = "cell"

    def __post_init__(self) -> None:
        if self.centering not in {"cell", "node"}:
            raise ValueError("centering must be 'cell' or 'node'")
        x_intervals = self.nx if self.centering == "cell" else self.nx - 1
        z_intervals = self.nz if self.centering == "cell" else self.nz - 1
        object.__setattr__(self, "lx_m", float(x_intervals * self.dx_m) if self.lx_m is None else float(self.lx_m))
        object.__setattr__(self, "lz_m", float(z_intervals * self.dz_m) if self.lz_m is None else float(self.lz_m))
        if abs(float(self.lx_m) - x_intervals * self.dx_m) > 1.0e-9:
            raise ValueError("lx_m is inconsistent with nx, dx_m, and centering")
        if abs(float(self.lz_m) - z_intervals * self.dz_m) > 1.0e-9:
            raise ValueError("lz_m is inconsistent with nz, dz_m, and centering")

    @property
    def x_m(self) -> np.ndarray:
        offset = 0.5 if self.centering == "cell" else 0.0
        return (np.arange(self.nx, dtype=np.float64) + offset) * float(self.dx_m)

    @property
    def z_m(self) -> np.ndarray:
        offset = 0.5 if self.centering == "cell" else 0.0
        return (np.arange(self.nz, dtype=np.float64) + offset) * float(self.dz_m)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "nx": int(self.nx),
            "nz": int(self.nz),
            "dx_m": float(self.dx_m),
            "dz_m": float(self.dz_m),
            "lx_m": float(self.lx_m),
            "lz_m": float(self.lz_m),
            "centering": self.centering,
        }


@dataclass(frozen=True)
class BoundaryConfig:
    top: str = "free_surface_dirichlet"
    left: str = "cpml"
    right: str = "cpml"
    bottom: str = "cpml"
    npml: int = 40
    cpml_target_reflection: float = 1.0e-8
    cpml_polynomial_order: int = 3
    cpml_outside_physical_domain: bool = True

    def __post_init__(self) -> None:
        if self.top != "free_surface_dirichlet":
            raise ValueError("v3 requires free-surface top boundary")
        if (self.left, self.right, self.bottom) != ("cpml", "cpml", "cpml"):
            raise ValueError("v3 requires CPML on left, right, and bottom only")
        if self.npml < 1:
            raise ValueError("npml must be positive")
        if not self.cpml_outside_physical_domain:
            raise ValueError("v3 CPML must be outside the physical 400x400 domain")

    def extended_shape(self, grid: AcousticGrid) -> tuple[int, int]:
        return (int(grid.nz + self.npml), int(grid.nx + 2 * self.npml))

    def physical_slices(self, grid: AcousticGrid) -> tuple[slice, slice]:
        return (slice(0, int(grid.nz)), slice(int(self.npml), int(self.npml + grid.nx)))

    def as_dict(self) -> dict[str, object]:
        return {
            "top": self.top,
            "left": self.left,
            "right": self.right,
            "bottom": self.bottom,
            "npml": int(self.npml),
            "cpml_target_reflection": float(self.cpml_target_reflection),
            "cpml_polynomial_order": int(self.cpml_polynomial_order),
            "cpml_outside_physical_domain": bool(self.cpml_outside_physical_domain),
        }


@dataclass(frozen=True)
class OutputTimeGrid:
    nt_out: int = 61
    dt_out_s: float = 0.01

    @property
    def t_s(self) -> np.ndarray:
        return np.arange(self.nt_out, dtype=np.float64) * float(self.dt_out_s)

    @property
    def t_end_s(self) -> float:
        return float((self.nt_out - 1) * self.dt_out_s)

    def aligned_internal_dt(self, n_substeps: int) -> float:
        if int(n_substeps) <= 0:
            raise ValueError("n_substeps must be positive")
        return float(self.dt_out_s) / int(n_substeps)

    def output_step_numbers(self, n_substeps: int) -> np.ndarray:
        return np.arange(self.nt_out, dtype=np.int64) * int(n_substeps)

    def as_dict(self) -> dict[str, float | int]:
        return {"nt_out": int(self.nt_out), "dt_out_s": float(self.dt_out_s), "t_end_s": self.t_end_s}
