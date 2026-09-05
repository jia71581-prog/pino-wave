from __future__ import annotations

import pytest
import torch

from fno_acoustic.model import AcousticFNO3D


def test_fno_forward_shape():
    model = AcousticFNO3D(in_features=3, modes_x=2, modes_z=2, modes_t=2, width=4, n_layers=1)
    x = torch.randn(2, 8, 8, 5, 3)
    y = model(x)
    assert y.shape == (2, 8, 8, 5)
    assert torch.isfinite(y).all()


def test_fno_backward_gradients_finite():
    model = AcousticFNO3D(in_features=3, modes_x=2, modes_z=2, modes_t=2, width=4, n_layers=1)
    x = torch.randn(1, 8, 8, 5, 3)
    target = torch.randn(1, 8, 8, 5)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_modes_out_of_range_errors():
    model = AcousticFNO3D(in_features=3, modes_x=9, modes_z=2, modes_t=2, width=4, n_layers=1)
    with pytest.raises(ValueError, match="modes_x"):
        model(torch.randn(1, 8, 8, 5, 3))
