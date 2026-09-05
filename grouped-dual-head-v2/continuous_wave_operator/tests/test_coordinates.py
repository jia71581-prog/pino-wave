from __future__ import annotations

import torch

from continuous_wave_operator.config import DomainConfig
from continuous_wave_operator.coordinates import normalize_query_coordinates, physical_output_gate


def test_physical_coordinates_and_hard_gates() -> None:
    domain = DomainConfig(lx_m=2000.0, lz_m=2000.0, t_end_s=1.0)
    query = torch.tensor([[[[1000.0, 500.0, 0.5]]]])

    normalized = normalize_query_coordinates(query, domain)

    assert torch.allclose(normalized, torch.tensor([[[[0.0, -0.5, 0.0]]]]))

    initial = torch.tensor([[[[1000.0, 500.0, 0.0]]]], requires_grad=True)
    initial_gate = physical_output_gate(initial, domain)
    gradient = torch.autograd.grad(initial_gate.sum(), initial, create_graph=True)[0]
    assert initial_gate.item() == 0.0
    assert gradient[..., 2].item() == 0.0

    top = torch.tensor([[[[1000.0, 0.0, 0.5]]]])
    assert physical_output_gate(top, domain).item() == 0.0
