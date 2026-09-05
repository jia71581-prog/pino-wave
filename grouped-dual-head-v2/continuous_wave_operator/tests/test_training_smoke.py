from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from continuous_wave_operator.config import DomainConfig, ModelConfig, TrainingConfig
from continuous_wave_operator.data.dataset import ContinuousQueryBatch
from continuous_wave_operator.model import ContinuousWaveOperator
from continuous_wave_operator.training.trainer import Trainer


class TinyDataset:
    medium_types = ("uniform", "uniform", "layered", "marmousi")

    def __init__(self) -> None:
        self.calls = 0

    def __len__(self) -> int:
        return 4

    def sample_queries(
        self,
        local_indices: list[int] | np.ndarray,
        _sampler: object,
        sampling_probabilities: object = None,
    ) -> ContinuousQueryBatch:
        self.calls += 1
        count = len(local_indices)
        velocity = torch.full((count, 1, 17, 17), 2500.0)
        source_map = torch.zeros(count, 1, 1, 17, 17)
        source_map[:, :, :, 2, 8] = 1.0
        source_parameters = torch.tensor([500.0, 200.0, 15.0, 0.1, 1.0]).repeat(count, 1, 1)
        query = torch.rand(count, 1, 16, 3)
        query[..., 0] *= 2000.0
        query[..., 1] *= 2000.0
        target = torch.sin(query[..., 0] / 300.0 - query[..., 2] * 10.0)
        return ContinuousQueryBatch(
            velocity_mps=velocity,
            source_map=source_map,
            source_parameters=source_parameters,
            query_coords=query,
            target_pressure=target,
            sample_indices=torch.tensor(local_indices),
            sample_ids=tuple(f"sample-{index}" for index in local_indices),
            group_ids=tuple(f"group-{index}" for index in local_indices),
            medium_types=tuple("uniform" for _ in local_indices),
        )


def test_two_step_training_writes_finite_resumable_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = ContinuousWaveOperator(
        ModelConfig(
            width=8,
            decoder_width=16,
            decoder_layers=1,
            attention_heads=2,
            token_grid_size=2,
            spectral_modes=(3, 2, 1, 1),
        ),
        DomainConfig(2000.0, 2000.0, 1.0),
    )
    config = TrainingConfig(
        max_steps=2,
        batch_size=2,
        time_frames=2,
        points_per_frame=8,
        prefetch_batches=3,
        stage1_steps=1,
        physics_ramp_steps=1,
        spectral_weight=0.01,
        pde_weight=1.0e-6,
        pde_queries=2,
    )
    train_dataset = TinyDataset()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=TinyDataset(),
        config=config,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config_digest="smoke",
        dataset_digest="tiny",
    )
    initial = model.query_decoder.output[-1].weight.detach().clone()

    assert trainer.ais.family_ids.tolist() == [2, 2, 0, 1]

    metrics = trainer.fit()

    assert np.isfinite(metrics["loss"])
    assert trainer.global_step == 2 and trainer.ais.step == 2
    assert train_dataset.calls >= 4
    assert not torch.equal(initial, model.query_decoder.output[-1].weight)
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
