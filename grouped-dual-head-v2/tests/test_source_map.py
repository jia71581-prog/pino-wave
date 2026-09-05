from __future__ import annotations

import torch

from fno_acoustic.data import make_gaussian_source_map


def test_source_position_to_map_peak():
    source = make_gaussian_source_map(16, 12, 5, 4, sigma_grid=1.5)
    assert torch.isclose(source.max(), torch.tensor(1.0))
    assert torch.argmax(source).item() == 5 * 12 + 4
