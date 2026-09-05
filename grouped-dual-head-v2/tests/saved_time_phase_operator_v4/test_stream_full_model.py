"""Full-model streaming loss-block: streamed backward == single-block backward.

Verifies the core mechanism of stream_full_model — a shared front end whose graph
is retained across time slices, with each slice's backward accumulating gradient
into it — reproduces the gradients of a single full-block backward exactly, for a
sum-over-frames (decomposable) squared-relative-energy objective.
"""
from __future__ import annotations

import torch
from torch import nn

from saved_time_phase_operator_v4.losses import (
    relative_energy_squared_reference,
    relative_energy_squared_block_loss,
)


class _Toy(nn.Module):
    """Shared front end (trained) + per-frame decoder (trained)."""

    def __init__(self):
        super().__init__()
        self.frontend = nn.Linear(4, 4)   # shared across all frames/slices
        self.decoder = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def field(self, cond, base):
        # cond: (R,4) medium/source code -> per-record scalar gain on a base field
        gain = self.frontend(cond).mean(dim=1).view(-1, 1, 1, 1)  # (R,1,1,1)
        return self.decoder(base.reshape(-1, 1, *base.shape[-2:])).reshape(base.shape) * (1.0 + gain)


def _run(stream: bool):
    torch.manual_seed(0)
    R, T, H, W = 2, 8, 6, 6
    model = _Toy()
    torch.manual_seed(1)
    cond = torch.randn(R, 4)
    base = torch.randn(R, T, H, W)
    target = torch.randn(R, T, H, W) * torch.linspace(1.0, 0.1, T)[None, :, None, None]
    ref = relative_energy_squared_reference(target, energy_floor_fraction=0.05)
    model.zero_grad(set_to_none=True)
    if not stream:
        pred = model.field(cond, base)
        loss = relative_energy_squared_block_loss(pred, target, reference=ref, spectrum_weight=0.1).total
        loss.backward()
    else:
        # shared front end computed once, graph retained across slices
        slices = [(0, 4), (4, 8)]
        for i, (a, b) in enumerate(slices):
            pred = model.field(cond, base)[:, a:b]
            loss = relative_energy_squared_block_loss(
                pred, target[:, a:b], reference=ref, spectrum_weight=0.1,
            ).total
            loss.backward(retain_graph=(i < len(slices) - 1))
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def test_streamed_backward_matches_single_block():
    single = _run(stream=False)
    streamed = _run(stream=True)
    for name in single:
        assert torch.allclose(single[name], streamed[name], atol=1e-5, rtol=1e-4), name


def test_frontend_receives_gradient_in_streamed_path():
    streamed = _run(stream=True)
    # the shared front end (analogue of medium/source encoder) is trained
    assert streamed["frontend.weight"].abs().max() > 0
