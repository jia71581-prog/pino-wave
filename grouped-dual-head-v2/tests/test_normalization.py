from __future__ import annotations

import torch

from fno_acoustic.normalization import decode_standard, encode_standard
from fno_acoustic.train import compute_normalization


def test_encode_decode_round_trip():
    x = torch.linspace(-1, 1, 8)
    stats = {"mean": 0.25, "std": 2.0}
    assert torch.allclose(decode_standard(encode_standard(x, stats), stats), x)


def test_normalization_uses_train_indices(tiny_config):
    stats = compute_normalization(tiny_config, [0, 1])
    assert stats["computed_from_split"] == "train"
    assert stats["train_sample_count"] == 2
    assert stats["velocity"]["count"] == 2 * 1 * 8 * 8
    assert stats["wavefield"]["std"] > 0
