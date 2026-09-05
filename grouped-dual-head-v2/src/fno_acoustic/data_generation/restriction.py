from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


_BINOMIAL5 = (1.0, 4.0, 6.0, 4.0, 1.0)


def _restrict_torch(field: torch.Tensor) -> torch.Tensor:
    if field.ndim < 2:
        raise ValueError("field must have trailing [z,x] dimensions")
    nz, nx = int(field.shape[-2]), int(field.shape[-1])
    if nz < 5 or nx < 5 or nz % 2 != 1 or nx % 2 != 1:
        raise ValueError("nodal 2x restriction requires odd dimensions of at least five")
    original_shape = field.shape
    flat = field.reshape(-1, 1, nz, nx)
    weights = torch.tensor(_BINOMIAL5, dtype=field.dtype, device=field.device) / 16.0
    kernel_x = weights.reshape(1, 1, 1, 5)
    kernel_z = weights.reshape(1, 1, 5, 1)
    filtered = F.conv2d(F.pad(flat, (2, 2, 0, 0), mode="reflect"), kernel_x)
    filtered = F.conv2d(F.pad(filtered, (0, 0, 2, 2), mode="reflect"), kernel_z)
    coarse = filtered[..., ::2, ::2]
    return coarse.reshape(*original_shape[:-2], (nz + 1) // 2, (nx + 1) // 2)


def restrict_nodal_2x(field):
    """Anti-alias an odd nodal grid and retain every second filtered node."""

    if isinstance(field, torch.Tensor):
        return _restrict_torch(field)
    array = np.asarray(field)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("restriction input must be floating point")
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return _restrict_torch(tensor).numpy()
