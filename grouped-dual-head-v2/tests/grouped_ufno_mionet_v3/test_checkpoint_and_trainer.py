from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from grouped_ufno_mionet_v3.training.checkpoint import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    save_checkpoint_atomic,
)
from grouped_ufno_mionet_v3.training.trainer import (
    GuardedV3Trainer,
    PlateauDetector,
    fresh_full_batch_lbfgs,
)
from scripts.train_grouped_v3 import main as train_main


class TinyOperator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

    def required_gradient_groups(self):
        return {"tiny": tuple(self.parameters())}


def test_checkpoint_round_trip_is_v3_versioned_and_atomic(tmp_path: Path):
    torch.manual_seed(1)
    model = TinyOperator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    path = tmp_path / "epoch.pt"
    save_checkpoint_atomic(
        path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=17,
        manifest_digest="manifest",
        config_digest="config",
        metrics={"loss": 0.25},
    )
    raw = torch.load(path, map_location="cpu", weights_only=False)
    assert raw["format"] == CHECKPOINT_FORMAT
    assert raw["epoch"] == 3 and raw["global_step"] == 17
    assert not list(tmp_path.glob("*.partial.*"))

    restored = TinyOperator()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-3)
    metadata = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        expected_manifest_digest="manifest",
        expected_config_digest="config",
    )
    assert metadata.epoch == 3 and metadata.global_step == 17
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_checkpoint_rejects_v2_and_identity_mismatches(tmp_path: Path):
    model = TinyOperator()
    v2 = tmp_path / "v2.pt"
    torch.save({"format": "grouped_dual_head_v2", "model": model.state_dict()}, v2)
    with pytest.raises(ValueError, match="V2|format"):
        load_checkpoint(
            v2,
            model=model,
            expected_manifest_digest="manifest",
            expected_config_digest="config",
        )

    v3 = tmp_path / "v3.pt"
    save_checkpoint_atomic(
        v3,
        model=model,
        optimizer=None,
        epoch=0,
        global_step=0,
        manifest_digest="actual-manifest",
        config_digest="actual-config",
        metrics={},
    )
    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint(
            v3,
            model=model,
            expected_manifest_digest="different",
            expected_config_digest="actual-config",
        )
    with pytest.raises(ValueError, match="config"):
        load_checkpoint(
            v3,
            model=model,
            expected_manifest_digest="actual-manifest",
            expected_config_digest="different",
        )


def test_checkpoint_restores_python_numpy_and_torch_rng(tmp_path: Path):
    random.seed(22)
    np.random.seed(22)
    torch.manual_seed(22)
    path = tmp_path / "rng.pt"
    model = TinyOperator()
    save_checkpoint_atomic(
        path,
        model=model,
        optimizer=None,
        epoch=1,
        global_step=2,
        manifest_digest="m",
        config_digest="c",
        metrics={},
    )
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    load_checkpoint(
        path,
        model=model,
        expected_manifest_digest="m",
        expected_config_digest="c",
        restore_rng=True,
    )
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_restores_cpu_rng_when_payload_is_mapped_to_cuda(tmp_path: Path):
    path = tmp_path / "rng_cuda.pt"
    model = TinyOperator().cuda()
    save_checkpoint_atomic(
        path,
        model=model,
        optimizer=None,
        epoch=1,
        global_step=2,
        manifest_digest="m",
        config_digest="c",
        metrics={},
    )

    metadata = load_checkpoint(
        path,
        model=model,
        expected_manifest_digest="m",
        expected_config_digest="c",
        restore_rng=True,
        map_location="cuda",
    )

    assert metadata.epoch == 1


def test_guarded_trainer_steps_and_saves_every_epoch(tmp_path: Path):
    torch.manual_seed(3)
    model = TinyOperator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
    trainer = GuardedV3Trainer(
        model,
        optimizer,
        checkpoint_dir=tmp_path,
        manifest_digest="manifest",
        config_digest="config",
        gradient_clip=10.0,
    )
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    y = torch.tensor([[1.0], [2.0]])
    loss = trainer.train_step(lambda: (model(x) - y).square().mean())
    assert torch.isfinite(loss)
    checkpoint = trainer.save_epoch(1, metrics={"loss": float(loss)})
    assert checkpoint.name == "checkpoint_epoch_0001.pt"
    assert checkpoint.exists()
    assert trainer.global_step == 1


def test_guarded_trainer_stops_nonfinite_loss_and_missing_gradients(tmp_path: Path):
    model = TinyOperator()
    trainer = GuardedV3Trainer(
        model,
        torch.optim.AdamW(model.parameters()),
        checkpoint_dir=tmp_path,
        manifest_digest="m",
        config_digest="c",
    )
    with pytest.raises(RuntimeError, match="nonfinite"):
        trainer.train_step(lambda: model.linear.weight.sum() * torch.tensor(float("nan")))
    with pytest.raises(RuntimeError, match="gradient"):
        trainer.train_step(lambda: model.linear.weight.square().mean())


def test_plateau_and_lbfgs_use_fresh_curvature_history():
    detector = PlateauDetector(patience=2, min_delta=1.0e-3)
    assert not detector.update(1.0)
    assert not detector.update(0.9)
    assert not detector.update(0.9005)
    assert detector.update(0.9004)
    model = TinyOperator()
    optimizer = fresh_full_batch_lbfgs(
        model,
        learning_rate=0.5,
        max_iter=5,
        history_size=7,
    )
    assert isinstance(optimizer, torch.optim.LBFGS)
    assert optimizer.state == {}
    assert optimizer.param_groups[0]["history_size"] == 7


def test_train_cli_dry_run_binds_real_manifest_and_v3_format(capsys):
    result = train_main(["--config", "configs/grouped_v3/smoke.yaml", "--dry-run"])
    assert result == 0
    output = capsys.readouterr().out
    assert CHECKPOINT_FORMAT in output
    assert '"train": 2240' in output
    assert '"validation": 480' in output
