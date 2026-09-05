from __future__ import annotations

import numpy as np

from .grid import AcousticGrid


def generate_layered_model(grid: AcousticGrid, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    n_layers = 2
    min_cells = max(1, int(round(50.0 / grid.dz_m)))
    interface = int(rng.integers(min_cells, grid.nz - min_cells + 1))
    edges = [0, interface, grid.nz]
    monotonic = bool(rng.random() < 0.70)
    if monotonic:
        layer_values = np.sort(rng.uniform(1500.0, 5200.0, size=n_layers))
    else:
        layer_values = rng.uniform(1500.0, 5200.0, size=n_layers)
    if rng.random() < 0.25:
        layer_values[0] = rng.uniform(1500.0, 1700.0)
    profile = np.zeros(grid.nz, dtype=np.float64)
    for layer_index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        profile[start:stop] = layer_values[layer_index]
    velocity = np.repeat(profile[:, None], grid.nx, axis=1).astype(np.float32)
    return velocity, {
        "model_type": "layered",
        "seed": int(seed),
        "n_layers": int(n_layers),
        "layer_edges_z_index": [int(v) for v in edges],
        "layer_values_mps": [float(v) for v in layer_values.tolist()],
        "min_layer_thickness_m": float(min(np.diff(edges)) * grid.dz_m),
        "monotonic_trend": monotonic,
        "smoothed_transition": False,
        "smoothing_sigma_m": 0.0,
    }
