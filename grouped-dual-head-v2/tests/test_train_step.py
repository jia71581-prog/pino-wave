from __future__ import annotations

import torch

from fno_acoustic.data import PinoHDF5Dataset, collate_pino
from fno_acoustic.losses import relative_l2
from fno_acoustic.model import AcousticFNO3D
from fno_acoustic.train import compute_normalization


def test_relative_l2_near_zero_target_finite():
    pred = torch.ones(2, 4, 4, 3)
    target = torch.zeros_like(pred)
    loss = relative_l2(pred, target)
    assert torch.isfinite(loss)


def test_tiny_realistic_batch_forward_backward(tiny_config):
    stats = compute_normalization(tiny_config, [0, 1])
    dataset = PinoHDF5Dataset(tiny_config, [0], normalization_stats=stats, return_normalized=True)
    batch = collate_pino([dataset[0]])
    model = AcousticFNO3D(**tiny_config["model"])
    pred = model(batch["input"])
    loss = torch.nn.functional.mse_loss(pred, batch["target"])
    loss.backward()
    assert pred.shape == batch["target"].shape
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
