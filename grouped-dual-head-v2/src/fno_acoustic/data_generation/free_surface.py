from __future__ import annotations

import torch


def pad_pressure_halo(field: torch.Tensor, *, radius: int = 4, side_mode: str = "zero") -> torch.Tensor:
    """Pad [..,z,x] pressure with an odd top extension and outer-side halos.

    The physical grid is cell centred.  The pressure-release boundary lies
    halfway between the first physical row and the nearest top ghost row, so
    the odd extension makes the interpolated pressure at z=0 exactly zero.
    """

    if not isinstance(field, torch.Tensor) or field.ndim < 2:
        raise TypeError("field must be a torch tensor with trailing [z,x] dimensions")
    radius = int(radius)
    if radius < 1 or min(field.shape[-2:]) < radius:
        raise ValueError("radius must be positive and fit inside the field")
    if side_mode == "replicate":
        left = field[..., :, :1].expand(*field.shape[:-1], radius)
        right = field[..., :, -1:].expand(*field.shape[:-1], radius)
    elif side_mode == "zero":
        side_shape = (*field.shape[:-1], radius)
        left = torch.zeros(side_shape, dtype=field.dtype, device=field.device)
        right = torch.zeros_like(left)
    else:
        raise ValueError("side_mode must be 'zero' or 'replicate'")
    interior = field.clone()
    interior[..., 0, :] = 0.0
    if side_mode == "replicate":
        left = interior[..., :, :1].expand(*interior.shape[:-1], radius)
        right = interior[..., :, -1:].expand(*interior.shape[:-1], radius)
    else:
        side_shape = (*interior.shape[:-1], radius)
        left = torch.zeros(side_shape, dtype=interior.dtype, device=interior.device)
        right = torch.zeros_like(left)
    x_padded = torch.cat((left, interior, right), dim=-1)
    top = -torch.flip(x_padded[..., 1 : radius + 1, :], dims=(-2,))
    if side_mode == "replicate":
        bottom = x_padded[..., -1:, :].expand(*x_padded.shape[:-2], radius, x_padded.shape[-1])
    else:
        bottom = torch.zeros(
            (*x_padded.shape[:-2], radius, x_padded.shape[-1]),
            dtype=field.dtype,
            device=field.device,
        )
    return torch.cat((top, x_padded, bottom), dim=-2)


def free_surface_boundary_error(field: torch.Tensor, *, radius: int = 4) -> torch.Tensor:
    padded = pad_pressure_halo(field, radius=radius, side_mode="replicate")
    return padded[..., radius, radius:-radius].abs().amax()
