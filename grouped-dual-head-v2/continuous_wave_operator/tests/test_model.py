from __future__ import annotations

import torch

from continuous_wave_operator.config import DomainConfig, ModelConfig
from continuous_wave_operator.model import ContinuousWaveOperator


def _inputs(amplitude: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity = torch.full((1, 1, 33, 33), 2800.0, requires_grad=True)
    source_map = torch.zeros(1, 2, 1, 33, 33)
    source_map[:, 0, :, 3, 8] = 1.0
    source_map[:, 1, :, 5, 24] = 1.0
    source_parameters = torch.tensor(
        [[[500.0, 180.0, 12.0, 0.1, amplitude], [1500.0, 300.0, 18.0, 0.08, amplitude]]]
    )
    query = torch.tensor(
        [[
            [[200.0, 100.0, 0.1], [1000.0, 500.0, 0.4], [1700.0, 1200.0, 0.8]],
            [[300.0, 150.0, 0.2], [900.0, 800.0, 0.5], [1800.0, 1600.0, 0.9]],
        ]],
        requires_grad=True,
    )
    return velocity, source_map, source_parameters, query


def test_continuous_model_reuses_medium_and_chunks_queries() -> None:
    config = ModelConfig(
        width=8,
        decoder_width=16,
        attention_heads=2,
        token_grid_size=4,
        spectral_modes=(4, 3, 2, 1),
    )
    model = ContinuousWaveOperator(config, DomainConfig(2000.0, 2000.0, 1.0))
    velocity, source_map, source_parameters, query = _inputs()
    calls = 0

    def count_call(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], _output: object) -> None:
        nonlocal calls
        calls += 1

    handle = model.medium_encoder.register_forward_hook(count_call)
    full = model(velocity, source_map, source_parameters, query)
    handle.remove()
    medium = model.encode_medium(velocity)
    sources = model.encode_sources(medium, source_map, source_parameters)
    chunked = model.query(medium, sources, query, chunk_size=2)

    assert full.shape == (1, 2, 3)
    assert calls == 1
    assert torch.allclose(full, chunked, atol=1.0e-6, rtol=1.0e-5)


def test_amplitude_gates_and_input_gradients() -> None:
    torch.manual_seed(7)
    config = ModelConfig(
        width=8,
        decoder_width=16,
        attention_heads=2,
        token_grid_size=4,
        spectral_modes=(4, 3, 2, 1),
    )
    model = ContinuousWaveOperator(config, DomainConfig(2000.0, 2000.0, 1.0))
    velocity, source_map, source_parameters, query = _inputs(amplitude=1.0)
    pressure = model(velocity, source_map, source_parameters, query)
    doubled_parameters = source_parameters.clone()
    doubled_parameters[..., 4] = 2.0
    doubled = model(velocity, source_map, doubled_parameters, query)

    assert torch.allclose(doubled, 2.0 * pressure, atol=1.0e-6, rtol=1.0e-5)
    assert pressure.abs().max() < 1.0e-6

    boundary_query = query.detach().clone()
    boundary_query[:, :, 0, 2] = 0.0
    boundary_query[:, :, 1, 1] = 0.0
    boundary_pressure = model(velocity, source_map, source_parameters, boundary_query)
    assert torch.all(boundary_pressure[:, :, 0] == 0.0)
    assert torch.all(boundary_pressure[:, :, 1] == 0.0)

    pressure.sum().backward()
    assert velocity.grad is not None and torch.isfinite(velocity.grad).all()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert torch.count_nonzero(velocity.grad) > 0
    assert torch.count_nonzero(query.grad) > 0
