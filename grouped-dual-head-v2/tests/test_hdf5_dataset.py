from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from fno_acoustic.data import PinoHDF5Dataset, collate_pino, create_splits, sample_count_from_config


def test_dataset_lazy_and_shapes(tiny_config):
    dataset = PinoHDF5Dataset(tiny_config, [0], normalization_stats=None, return_normalized=False)
    assert dataset._file is None
    sample = dataset[0]
    assert dataset._file is not None
    assert sample["input"].shape == (8, 8, 5, 3)
    assert sample["target"].shape == (8, 8, 5)
    assert sample["velocity"].shape == (1, 8, 8)
    assert sample["source_map"].shape == (1, 8, 8)
    assert torch.isfinite(sample["input"]).all()
    assert torch.isfinite(sample["target"]).all()
    assert sample["target"].abs().max() > 0
    dataset.close()


def test_num_workers_loader(tiny_config):
    dataset = PinoHDF5Dataset(tiny_config, [0, 1], normalization_stats=None, return_normalized=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=collate_pino)
    batch = next(iter(loader))
    assert batch["input"].shape[0] == 1
    assert torch.isfinite(batch["input"]).all()


def test_split_complete_samples(tiny_config):
    splits = create_splits(sample_count_from_config(tiny_config), [0.5, 0.25, 0.25], seed=2026)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
