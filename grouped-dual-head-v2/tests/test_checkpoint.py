from __future__ import annotations

import torch

from fno_acoustic.checkpoint import load_checkpoint, save_checkpoint
from fno_acoustic.model import AcousticFNO3D


def test_checkpoint_save_load_prediction_consistent(tmp_path):
    torch.manual_seed(0)
    model = AcousticFNO3D(in_features=3, modes_x=2, modes_z=2, modes_t=2, width=4, n_layers=1)
    model.eval()
    x = torch.randn(1, 8, 8, 5, 3)
    pred = model(x)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, {"model_state_dict": model.state_dict(), "model_config": {"in_features": 3, "modes_x": 2, "modes_z": 2, "modes_t": 2, "width": 4, "n_layers": 1}})
    ckpt = load_checkpoint(path)
    restored = AcousticFNO3D(**ckpt["model_config"])
    restored.load_state_dict(ckpt["model_state_dict"])
    restored.eval()
    assert torch.allclose(pred, restored(x), atol=1e-6)
