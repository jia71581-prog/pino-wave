"""Saved-grid contract for the teacher's three-sided exterior CFS-CPML.

The registered wavefield is stored only on the 201 x 201 physical domain.
At 10 m resolution the absorbing layer occupies 20 nodes outside the left,
right and bottom boundaries; the top is a pressure-release free surface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch

from fno_acoustic.data_generation.cpml import CFSCPMLProfiles, build_cfs_cpml_profiles
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig


PROFILE_FIELDS = (
    "sigma_x",
    "sigma_z",
    "kappa_x",
    "kappa_z",
    "alpha_x",
    "alpha_z",
    "a_x",
    "a_z",
    "b_x",
    "b_z",
    "inv_kappa_x",
    "inv_kappa_z",
    "active_x",
    "active_z",
)


@dataclass(frozen=True)
class ExteriorCPMLContract:
    physical_nz: int = 201
    physical_nx: int = 201
    spacing_m: float = 10.0
    cpml_layers: int = 20
    internal_dt_s: float = 1.25e-4
    saved_dt_s: float = 2.5e-3
    internal_substeps_per_saved_frame: int = 20
    target_reflection: float = 1.0e-8
    polynomial_order: int = 3
    kappa_max: float = 3.0
    minimum_frequency_hz: float = 8.0
    top: str = "free_surface_dirichlet"
    left: str = "cpml"
    right: str = "cpml"
    bottom: str = "cpml"
    cpml_outside_physical_domain: bool = True

    def __post_init__(self) -> None:
        if self.physical_nz < 5 or self.physical_nx < 5:
            raise ValueError("physical grid must contain at least five nodes per axis")
        if self.physical_nz % 2 != 1 or self.physical_nx % 2 != 1:
            raise ValueError("physical grid must be odd and node centred")
        if self.cpml_layers <= 0 or self.spacing_m <= 0.0:
            raise ValueError("CPML layers and spacing must be positive")
        if self.top != "free_surface_dirichlet":
            raise ValueError("the top boundary must be a free surface")
        if (self.left, self.right, self.bottom) != ("cpml", "cpml", "cpml"):
            raise ValueError("CPML must be present on left, right and bottom only")
        if not self.cpml_outside_physical_domain:
            raise ValueError("the registered CPML must be outside the physical domain")
        if self.internal_substeps_per_saved_frame <= 0:
            raise ValueError("internal substeps must be positive")
        if not np.isclose(
            self.internal_dt_s * self.internal_substeps_per_saved_frame,
            self.saved_dt_s,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("CPML internal and saved time steps are inconsistent")

    @property
    def extended_nz(self) -> int:
        return self.physical_nz + self.cpml_layers

    @property
    def extended_nx(self) -> int:
        return self.physical_nx + 2 * self.cpml_layers

    @property
    def physical_z_slice(self) -> slice:
        return slice(0, self.physical_nz)

    @property
    def physical_x_slice(self) -> slice:
        return slice(self.cpml_layers, self.cpml_layers + self.physical_nx)

    @property
    def physical_thickness_m(self) -> float:
        return self.cpml_layers * self.spacing_m

    def grid(self) -> AcousticGrid:
        return AcousticGrid(
            nx=self.physical_nx,
            nz=self.physical_nz,
            dx_m=self.spacing_m,
            dz_m=self.spacing_m,
            lx_m=(self.physical_nx - 1) * self.spacing_m,
            lz_m=(self.physical_nz - 1) * self.spacing_m,
            centering="node",
        )

    def boundaries(self) -> BoundaryConfig:
        return BoundaryConfig(
            top=self.top,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            npml=self.cpml_layers,
            cpml_target_reflection=self.target_reflection,
            cpml_polynomial_order=self.polynomial_order,
            cpml_outside_physical_domain=True,
        )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "physical_shape_zx": [self.physical_nz, self.physical_nx],
                "extended_shape_zx": [self.extended_nz, self.extended_nx],
                "physical_slice_zx": [
                    [self.physical_z_slice.start, self.physical_z_slice.stop],
                    [self.physical_x_slice.start, self.physical_x_slice.stop],
                ],
                "physical_thickness_m": self.physical_thickness_m,
            }
        )
        return payload


def build_saved_exterior_cpml_profiles(
    contract: ExteriorCPMLContract,
    *,
    c_ref_mps: float,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CFSCPMLProfiles:
    return build_cfs_cpml_profiles(
        contract.grid(),
        contract.boundaries(),
        dt_s=contract.internal_dt_s,
        c_ref_mps=float(c_ref_mps),
        target_reflection=contract.target_reflection,
        polynomial_order=contract.polynomial_order,
        kappa_max=contract.kappa_max,
        minimum_frequency_hz=contract.minimum_frequency_hz,
        device=device,
        dtype=dtype,
    )


def extend_velocity_to_exterior(
    velocity: np.ndarray | torch.Tensor,
    contract: ExteriorCPMLContract,
) -> np.ndarray | torch.Tensor:
    """Replicate physical edge velocity into left/right/bottom exterior strips."""

    value = velocity
    if value.ndim < 2 or tuple(value.shape[-2:]) != (
        contract.physical_nz,
        contract.physical_nx,
    ):
        raise ValueError("velocity must end in the registered physical shape")
    margin = contract.cpml_layers
    if isinstance(value, torch.Tensor):
        left = value[..., :, :1].expand(*value.shape[:-1], margin)
        right = value[..., :, -1:].expand(*value.shape[:-1], margin)
        horizontal = torch.cat((left, value, right), dim=-1)
        bottom = horizontal[..., -1:, :].expand(
            *horizontal.shape[:-2], margin, horizontal.shape[-1]
        )
        return torch.cat((horizontal, bottom), dim=-2)
    array = np.asarray(value)
    padding = [(0, 0)] * array.ndim
    padding[-2] = (0, margin)
    padding[-1] = (margin, margin)
    return np.pad(array, padding, mode="edge")


def crop_physical_domain(
    extended: np.ndarray | torch.Tensor,
    contract: ExteriorCPMLContract,
) -> np.ndarray | torch.Tensor:
    if extended.ndim < 2 or tuple(extended.shape[-2:]) != (
        contract.extended_nz,
        contract.extended_nx,
    ):
        raise ValueError("extended field does not match the exterior CPML contract")
    return extended[..., contract.physical_z_slice, contract.physical_x_slice]


def profile_sampling_report(
    saved: CFSCPMLProfiles,
    fine: CFSCPMLProfiles,
    *,
    factor: int = 2,
) -> dict[str, float]:
    """Compare saved profiles with exact nodal samples of the fine profiles."""

    if factor <= 0:
        raise ValueError("sampling factor must be positive")
    result: dict[str, float] = {}
    for name in PROFILE_FIELDS:
        coarse_value = getattr(saved, name)
        sampled = getattr(fine, name)[::factor, ::factor]
        if sampled.shape != coarse_value.shape:
            raise ValueError(f"sampled profile shape mismatch for {name}")
        if coarse_value.dtype == torch.bool:
            result[name] = float(torch.count_nonzero(coarse_value != sampled))
        else:
            result[name] = float((coarse_value - sampled).abs().max())
    return result


def contract_from_dataset_config(config: Mapping[str, object]) -> ExteriorCPMLContract:
    grid = dict(config.get("storage_grid", {}) or {})
    fine_grid = dict(config.get("grid", {}) or {})
    time = dict(config.get("time", {}) or {})
    boundary = dict(config.get("boundaries", {}) or {})
    fine_npml = int(boundary.get("npml", -1))
    if fine_npml % 2:
        raise ValueError("fine CPML layer count must be divisible by two")
    contract = ExteriorCPMLContract(
        physical_nz=int(grid.get("nz", -1)),
        physical_nx=int(grid.get("nx", -1)),
        spacing_m=float(grid.get("dx_m", float("nan"))),
        cpml_layers=fine_npml // 2,
        internal_dt_s=float(time.get("dt_used_s", float("nan"))),
        saved_dt_s=float(time.get("dt_out_s", float("nan"))),
        internal_substeps_per_saved_frame=int(time.get("snapshot_stride", -1)),
        target_reflection=float(boundary.get("cpml_target_reflection", float("nan"))),
        polynomial_order=int(boundary.get("cpml_polynomial_order", -1)),
        kappa_max=float(boundary.get("kappa_max", float("nan"))),
        minimum_frequency_hz=float(boundary.get("minimum_frequency_hz", float("nan"))),
        top=str(boundary.get("top", "")),
        left=str(boundary.get("left", "")),
        right=str(boundary.get("right", "")),
        bottom=str(boundary.get("bottom", "")),
        cpml_outside_physical_domain=bool(
            boundary.get("cpml_outside_physical_domain", False)
        ),
    )
    if float(grid.get("dz_m", float("nan"))) != contract.spacing_m:
        raise ValueError("saved grid must use equal x/z spacing")
    if (
        int(fine_grid.get("nz", -1)),
        int(fine_grid.get("nx", -1)),
    ) != (2 * contract.physical_nz - 1, 2 * contract.physical_nx - 1):
        raise ValueError("fine and saved physical grids are not factor-two nodal pairs")
    if float(fine_grid.get("dx_m", float("nan"))) * 2.0 != contract.spacing_m:
        raise ValueError("fine and saved grid spacing are not factor-two pairs")
    return contract


__all__ = [
    "ExteriorCPMLContract",
    "PROFILE_FIELDS",
    "build_saved_exterior_cpml_profiles",
    "contract_from_dataset_config",
    "crop_physical_domain",
    "extend_velocity_to_exterior",
    "profile_sampling_report",
]
