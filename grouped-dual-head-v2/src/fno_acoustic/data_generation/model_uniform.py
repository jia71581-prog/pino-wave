from __future__ import annotations

import numpy as np

from .grid import AcousticGrid


def uniform_velocity(grid: AcousticGrid, velocity_mps: float) -> np.ndarray:
    return np.full((int(grid.nz), int(grid.nx)), float(velocity_mps), dtype=np.float32)
