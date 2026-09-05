from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from continuous_wave_operator.training.adaptive_sampling import HierarchicalAIS
from continuous_wave_operator.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_complete_state_and_rng(tmp_path: Path) -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    ais = HierarchicalAIS(sample_count=2, time_bins=3, spatial_shape=(2, 2), seed=14)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ais=ais,
        normalization={"pressure_std": 2.0},
        config_digest="config-a",
        dataset_digest="data-a",
        epoch=2,
        global_step=9,
    )
    expected = (random.random(), np.random.rand(), torch.rand(1))
    with torch.no_grad():
        model.weight.zero_()

    state = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_config_digest="config-a",
        expected_dataset_digest="data-a",
    )

    replay = (random.random(), np.random.rand(), torch.rand(1))
    assert state.epoch == 2 and state.global_step == 9
    assert state.normalization == {"pressure_std": 2.0}
    assert torch.count_nonzero(model.weight) > 0
    assert expected[0] == replay[0] and expected[1] == replay[1]
    assert torch.equal(expected[2], replay[2])
    with pytest.raises(ValueError, match="config"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config_digest="changed",
            expected_dataset_digest="data-a",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_checkpoint_restores_cpu_rng_state_when_model_maps_to_cuda(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    ais = HierarchicalAIS(sample_count=1, time_bins=2, spatial_shape=(2, 2), seed=1)
    path = tmp_path / "cuda.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ais=ais,
        normalization={},
        config_digest="c",
        dataset_digest="d",
        epoch=0,
        global_step=0,
    )

    restored = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_config_digest="c",
        expected_dataset_digest="d",
        map_location="cuda",
    )

    assert restored.ais.step == 0
