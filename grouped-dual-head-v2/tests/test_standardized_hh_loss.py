from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from fno_acoustic.ais_sampler import AdaptiveSpatialSampler
from fno_acoustic.ais_normalization import AISNormalizationBinding
from fno_acoustic.query_data import QueryScene
from fno_acoustic.query_losses import standardized_hh_field_loss
from fno_acoustic.query_training import train_query_epoch, validate_query_guard


def fake_binding(*, wavefield_mean: float, wavefield_std: float) -> AISNormalizationBinding:
    return AISNormalizationBinding(
        path=Path("unused.json"),
        stats_sha256="a" * 64,
        velocity_mean=3000.0,
        velocity_std=500.0,
        wavefield_mean=wavefield_mean,
        wavefield_std=wavefield_std,
        eps=1.0e-8,
    )


def test_standardized_hh_uses_normalized_mse_and_physical_relative_l2() -> None:
    target = torch.stack((torch.zeros(160), torch.ones(160)))[None]
    prediction = target + 0.2
    binding = fake_binding(wavefield_mean=0.1, wavefield_std=0.5)
    target_hat = binding.encode_wavefield(target)
    prediction_hat = binding.encode_wavefield(prediction)

    result = standardized_hh_field_loss(
        prediction_hat,
        target_hat,
        torch.full((1, 2), 0.5),
        2,
        binding,
    )

    assert torch.allclose(result.mse, (prediction_hat - target_hat).square().mean())
    assert result.relative_l2.item() == pytest.approx(
        (
            torch.linalg.vector_norm(prediction - target)
            / torch.linalg.vector_norm(target)
        ).item()
    )
    assert result.loss == result.mse + result.relative_l2


class MemoryStore:
    def __init__(self, scene: QueryScene) -> None:
        self.scene = scene

    def read_scene(self, sample_id: int) -> QueryScene:
        assert sample_id == self.scene.sample_id
        return self.scene


class FixedEncodedPredictionModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))
        self.encoder_scale = nn.Parameter(torch.tensor(0.0))

    def encode_global(self, global_inputs: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
        return global_inputs.mean(dim=(1, 2, 4)) * self.encoder_scale

    def decode_queries(
        self,
        context: torch.Tensor,
        native_static: torch.Tensor,
        query_xz: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        return (self.value + 0.0 * context[:, None, :]).expand(
            context.shape[0], query_xz.shape[1], time_s.numel()
        )


class RecordingSampler(AdaptiveSpatialSampler):
    last_residual: torch.Tensor
    last_sites: torch.Tensor

    def update_residual_tiles(
        self,
        sample_ids: torch.Tensor,
        site_indices: torch.Tensor,
        squared_error: torch.Tensor,
    ) -> None:
        self.last_residual = squared_error.clone()
        self.last_sites = site_indices.clone()
        super().update_residual_tiles(sample_ids, site_indices, squared_error)


def make_scene() -> QueryScene:
    height, width = 5, 5
    return QueryScene(
        sample_id=0,
        target_cpu=torch.ones(height, width, 160),
        velocity_cpu=torch.full((height, width), 3000.0),
        source_cpu=torch.nn.functional.pad(torch.ones(1, 1), (2, 2, 2, 2)),
        time_s=torch.linspace(0.0, 1.0, 160, dtype=torch.float64),
        x_m=torch.linspace(0.0, 400.0, height, dtype=torch.float64),
        z_m=torch.linspace(0.0, 400.0, width, dtype=torch.float64),
        metadata={},
    )


def query_config() -> dict[str, object]:
    return {
        "seed": 7,
        "sampling": {"target_height": 5, "target_width": 5, "global_size": 3},
        "receiver": {
            "x_start_m": 0.0,
            "x_stop_m": 400.0,
            "x_stride_m": 200.0,
            "z_m": [0.0],
        },
        "train": {
            "query_sites_per_scene": 3,
            "query_chunk_size": 2,
            "patch_size": 3,
            "patch_centers_per_scene": 1,
            "grad_clip": 10.0,
            "amp": False,
        },
        "loss": {
            "hh_reweight": True,
            "receiver_weight": 0.0,
            "phase_weight": 0.0,
            "local_spectrum_weight": 0.0,
            "energy_weight": 0.0,
        },
    }


def test_train_epoch_updates_sampler_with_decoded_physical_residual() -> None:
    scene = make_scene()
    binding = fake_binding(wavefield_mean=0.2, wavefield_std=0.5)
    sampler = RecordingSampler(
        5, 5, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], seed=7, tile_size=2
    )
    model = FixedEncodedPredictionModel(0.8)

    train_query_epoch(
        model,
        MemoryStore(scene),
        [0],
        sampler,
        torch.optim.SGD(model.parameters(), lr=1.0e-3),
        query_config(),
        torch.device("cpu"),
        epoch=0,
        global_step=0,
        normalization=binding,
    )

    physical_prediction = binding.decode_wavefield(torch.full((1, 3, 160), 0.8))
    physical_target = scene.target_cpu.reshape(-1, 160).index_select(
        0, sampler.last_sites[0]
    )[None]
    assert torch.allclose(
        sampler.last_residual,
        (physical_prediction - physical_target).square().mean(-1),
    )


def test_validation_decodes_prediction_before_relative_l2() -> None:
    scene = make_scene()
    binding = fake_binding(wavefield_mean=0.2, wavefield_std=0.5)
    exact_encoded_target = float(binding.encode_wavefield(torch.tensor(1.0)))

    metrics = validate_query_guard(
        FixedEncodedPredictionModel(exact_encoded_target),
        MemoryStore(scene),
        [0],
        {0: torch.tensor([0, 2, 4])},
        query_config(),
        torch.device("cpu"),
        normalization=binding,
    )

    assert metrics["relative_l2"] == pytest.approx(0.0, abs=1.0e-7)
    assert metrics["relative_l2_q4"] == pytest.approx(0.0, abs=1.0e-7)
