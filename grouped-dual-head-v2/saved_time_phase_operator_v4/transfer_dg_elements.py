"""Reusable local trace-to-flux operators for Transfer DG neural elements."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


EDGE_ORDER = ("top", "right", "bottom", "left")


def cartesian_element_origins(
    height: int, width: int, *, element_intervals: int = 20
) -> torch.Tensor:
    """Return ``[elements,2]`` node origins for a conforming Cartesian partition."""
    nz, nx, size = int(height), int(width), int(element_intervals)
    if min(nz, nx) < 2 or size <= 0:
        raise ValueError("grid dimensions and element_intervals must be positive")
    if (nz - 1) % size or (nx - 1) % size:
        raise ValueError("element_intervals must divide both grid interval counts")
    return torch.tensor(
        [(z0, x0) for z0 in range(0, nz - 1, size) for x0 in range(0, nx - 1, size)],
        dtype=torch.int64,
    )


def orthonormal_cosine_trace_basis(
    points: int, modes: int, *, device=None, dtype=torch.float32
) -> torch.Tensor:
    """DCT-II basis ``[points,modes]`` with orthonormal columns."""
    count, retained = int(points), int(modes)
    if count <= 0 or not 0 < retained <= count:
        raise ValueError("trace modes must lie in [1, points]")
    coordinate = torch.arange(count, device=device, dtype=dtype)[:, None] + 0.5
    frequency = torch.arange(retained, device=device, dtype=dtype)[None]
    basis = torch.cos(math.pi * coordinate * frequency / float(count))
    basis[:, 0] *= math.sqrt(1.0 / float(count))
    if retained > 1:
        basis[:, 1:] *= math.sqrt(2.0 / float(count))
    return basis


def extract_element_traces(
    field: torch.Tensor,
    origins: torch.Tensor,
    *,
    element_intervals: int = 20,
) -> torch.Tensor:
    """Extract oriented edge traces as ``[...,elements,4,intervals+1]``."""
    value = torch.as_tensor(field)
    if value.ndim < 2:
        raise ValueError("field must have trailing [z,x] dimensions")
    size = int(element_intervals)
    rows = []
    for origin in torch.as_tensor(origins, dtype=torch.int64).tolist():
        z0, x0 = (int(item) for item in origin)
        patch = value[..., z0 : z0 + size + 1, x0 : x0 + size + 1]
        if patch.shape[-2:] != (size + 1, size + 1):
            raise ValueError("element origin lies outside the field")
        rows.append(
            torch.stack(
                (
                    patch[..., 0, :],
                    patch[..., :, -1],
                    torch.flip(patch[..., -1, :], dims=(-1,)),
                    torch.flip(patch[..., :, 0], dims=(-1,)),
                ),
                dim=-2,
            )
        )
    if not rows:
        raise ValueError("at least one element origin is required")
    return torch.stack(rows, dim=-3)


def extract_element_normal_flux_traces(
    derivative_x: torch.Tensor,
    derivative_z: torch.Tensor,
    origins: torch.Tensor,
    *,
    element_intervals: int = 20,
) -> torch.Tensor:
    """Extract outward-normal derivative traces in ``EDGE_ORDER``."""
    dx, dz = torch.as_tensor(derivative_x), torch.as_tensor(derivative_z)
    if dx.shape != dz.shape or dx.ndim < 2:
        raise ValueError("derivative fields must have equal trailing [z,x] shapes")
    size = int(element_intervals)
    rows = []
    for origin in torch.as_tensor(origins, dtype=torch.int64).tolist():
        z0, x0 = (int(item) for item in origin)
        x_patch = dx[..., z0 : z0 + size + 1, x0 : x0 + size + 1]
        z_patch = dz[..., z0 : z0 + size + 1, x0 : x0 + size + 1]
        if x_patch.shape[-2:] != (size + 1, size + 1):
            raise ValueError("element origin lies outside the derivative field")
        rows.append(
            torch.stack(
                (
                    -z_patch[..., 0, :],
                    x_patch[..., :, -1],
                    torch.flip(z_patch[..., -1, :], dims=(-1,)),
                    torch.flip(-x_patch[..., :, 0], dims=(-1,)),
                ),
                dim=-2,
            )
        )
    if not rows:
        raise ValueError("at least one element origin is required")
    return torch.stack(rows, dim=-3)


def project_trace_modes(trace: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    value, transform = torch.as_tensor(trace), torch.as_tensor(basis)
    if value.shape[-1] != transform.shape[0] or transform.ndim != 2:
        raise ValueError("trace and basis point dimensions do not match")
    return torch.einsum("...n,nm->...m", value, transform.to(value))


def reconstruct_trace_modes(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    value, transform = torch.as_tensor(coefficients), torch.as_tensor(basis)
    if value.shape[-1] != transform.shape[1] or transform.ndim != 2:
        raise ValueError("coefficient and basis mode dimensions do not match")
    return torch.einsum("...m,nm->...n", value, transform.to(value))


class PassiveComplexDtN(nn.Module):
    """Context-conditioned complex DtN map with a PSD dissipative component.

    The complex map is ``K = S + iD``. ``S`` is symmetric but may be
    indefinite, while ``D`` is positive definite.  For the chosen Fourier sign
    convention this constrains the outgoing/radiative part to be passive.
    """

    def __init__(self, context_dim: int, trace_dofs: int, *, rank: int = 8) -> None:
        super().__init__()
        context, dofs, factor_rank = int(context_dim), int(trace_dofs), int(rank)
        if min(context, dofs, factor_rank) <= 0:
            raise ValueError("DtN dimensions must be positive")
        self.context_dim = context
        self.trace_dofs = dofs
        self.rank = factor_rank
        # A, B define an indefinite symmetric part A B^T + B A^T.
        # C C^T + positive diagonal defines the dissipative part.
        output_dim = 3 * dofs * factor_rank + 2 * dofs + 2 * dofs
        self.hyper = nn.Linear(context, output_dim)

    def matrices_and_source(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = torch.as_tensor(context)
        if value.ndim != 2 or value.shape[1] != self.context_dim:
            raise ValueError(f"context must be [B,{self.context_dim}]")
        raw = self.hyper(value)
        b = value.shape[0]
        width = self.trace_dofs * self.rank
        a = raw[:, :width].reshape(b, self.trace_dofs, self.rank)
        left = raw[:, width : 2 * width].reshape(b, self.trace_dofs, self.rank)
        dissipative_factor = raw[:, 2 * width : 3 * width].reshape(
            b, self.trace_dofs, self.rank
        )
        cursor = 3 * width
        symmetric_diagonal = raw[:, cursor : cursor + self.trace_dofs]
        cursor += self.trace_dofs
        dissipative_diagonal = F.softplus(
            raw[:, cursor : cursor + self.trace_dofs]
        ) + 1.0e-6
        cursor += self.trace_dofs
        source = raw[:, cursor:].reshape(b, 2, self.trace_dofs)
        symmetric = torch.bmm(a, left.transpose(1, 2))
        symmetric = symmetric + symmetric.transpose(1, 2)
        symmetric = symmetric + torch.diag_embed(symmetric_diagonal)
        dissipative = torch.bmm(
            dissipative_factor, dissipative_factor.transpose(1, 2)
        ) + torch.diag_embed(dissipative_diagonal)
        return symmetric, dissipative, source

    def forward(self, context: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        boundary = torch.as_tensor(trace)
        if boundary.ndim != 3 or boundary.shape[1:] != (2, self.trace_dofs):
            raise ValueError(f"trace must be [B,2,{self.trace_dofs}]")
        symmetric, dissipative, source = self.matrices_and_source(context)
        real, imaginary = boundary[:, 0], boundary[:, 1]
        flux_real = torch.bmm(symmetric, real[..., None])[..., 0]
        flux_real -= torch.bmm(dissipative, imaginary[..., None])[..., 0]
        flux_imag = torch.bmm(symmetric, imaginary[..., None])[..., 0]
        flux_imag += torch.bmm(dissipative, real[..., None])[..., 0]
        return torch.stack((flux_real, flux_imag), dim=1) + source

    def dissipation_quadratic(
        self, context: torch.Tensor, trace: torch.Tensor
    ) -> torch.Tensor:
        _, dissipative, _ = self.matrices_and_source(context)
        boundary = torch.as_tensor(trace)
        real, imaginary = boundary[:, 0], boundary[:, 1]
        real_energy = torch.einsum("bi,bij,bj->b", real, dissipative, real)
        imaginary_energy = torch.einsum(
            "bi,bij,bj->b", imaginary, dissipative, imaginary
        )
        return real_energy + imaginary_energy


class TransferDGLocalElementOperator(nn.Module):
    """Shared neural element mapping complex pressure traces to normal fluxes."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        trace_modes: int = 8,
        context_dim: int = 64,
        dtn_rank: int = 8,
        frequency_dim: int = 4,
    ) -> None:
        super().__init__()
        if min(input_channels, trace_modes, context_dim, dtn_rank, frequency_dim) <= 0:
            raise ValueError("local element dimensions must be positive")
        self.input_channels = int(input_channels)
        self.trace_modes = int(trace_modes)
        self.trace_dofs = 4 * self.trace_modes
        self.frequency_dim = int(frequency_dim)
        self.medium_encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.context = nn.Sequential(
            nn.Linear(64 + self.frequency_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.dtn = PassiveComplexDtN(
            context_dim, self.trace_dofs, rank=int(dtn_rank)
        )

    def encode_context(
        self, element_features: torch.Tensor, frequency_features: torch.Tensor
    ) -> torch.Tensor:
        features = torch.as_tensor(element_features)
        frequency = torch.as_tensor(frequency_features)
        if features.ndim != 4 or features.shape[1] != self.input_channels:
            raise ValueError(
                f"element_features must be [B,{self.input_channels},Z,X]"
            )
        if frequency.shape != (features.shape[0], self.frequency_dim):
            raise ValueError(
                f"frequency_features must be [B,{self.frequency_dim}]"
            )
        return self.context(torch.cat((self.medium_encoder(features), frequency), dim=1))

    def forward(
        self,
        element_features: torch.Tensor,
        frequency_features: torch.Tensor,
        trace_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        trace = torch.as_tensor(trace_coefficients)
        expected = (element_features.shape[0], 2, 4, self.trace_modes)
        if tuple(trace.shape) != expected:
            raise ValueError(f"trace_coefficients must have shape {expected}")
        context = self.encode_context(element_features, frequency_features)
        flux = self.dtn(context, trace.flatten(2))
        return flux.reshape(*expected)


__all__ = [
    "EDGE_ORDER",
    "PassiveComplexDtN",
    "TransferDGLocalElementOperator",
    "cartesian_element_origins",
    "extract_element_normal_flux_traces",
    "extract_element_traces",
    "orthonormal_cosine_trace_basis",
    "project_trace_modes",
    "reconstruct_trace_modes",
]
