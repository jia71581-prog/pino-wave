from pathlib import Path

import pytest
import torch

from grouped_ufno_mionet_v2.config import LossConfig, V2Config
from grouped_ufno_mionet_v2.data.batch import V2MacroBatch
from grouped_ufno_mionet_v2.model.operator import DualHeadWaveOperator
from grouped_ufno_mionet_v2.normalization import PhysicalNormalizer, ScaleMetadata
from grouped_ufno_mionet_v2.training.audit import LossDominanceMonitor, require_gradients
from grouped_ufno_mionet_v2.training.checkpoint import AtomicCheckpointManager, load_state
from grouped_ufno_mionet_v2.training.trainer import JointDualHeadTrainer, compose_objective
from scripts.overfit_grouped_v2 import restore_for_finetune


def make_model():
    return DualHeadWaveOperator(width=8, rank=4, modes=(3, 3, 3, 3), heads=2, dense_time_block=4)


def make_normalizer():
    return PhysicalNormalizer(ScaleMetadata(2500, 1000, 1e-8, (2000, 2000, 50, 1.2, 1), "train"))


def test_gradient_audit_rejects_unused_dense_head():
    model = make_model()
    velocity = torch.full((1, 1, 17, 17), 2200.0)
    source = torch.tensor([[500., 500., 10., .05, 1.]])
    source_map = torch.zeros(1, 1, 17, 17); source_map[0, 0, 5, 5] = 1
    coords = torch.tensor([[[500., 500., .2], [700., 600., .3]]])
    model.query_normalized(velocity, source, source_map, coords, make_normalizer()).sum().backward()
    with pytest.raises(RuntimeError, match="dense_decoder"):
        require_gradients(model, required_prefixes=("medium_encoder", "source_encoder", "query_head", "dense_decoder"))


def test_dominance_monitor_requires_five_consecutive_bad_steps():
    monitor = LossDominanceMonitor(max_ratio=10, patience=5)
    for _ in range(4):
        monitor.update(data_loss=1.0, auxiliary_loss=11.0)
    with pytest.raises(RuntimeError, match="dominated"):
        monitor.update(data_loss=1.0, auxiliary_loss=11.0)
    monitor = LossDominanceMonitor(max_ratio=10, patience=5)
    for _ in range(4):
        monitor.update(1.0, 11.0)
    monitor.update(1.0, 1.0)
    assert monitor.bad_steps == 0


def test_receiver_data_is_an_independently_weighted_auxiliary_target():
    config = LossConfig(query=4.0, dense=2.0, trace=0.0, consistency=0.0,
                        gradient=0.0, spatial_fft=0.0, trace_fft=0.0)
    total, data, auxiliary = compose_objective(
        config,
        query=torch.tensor(3.0), dense=torch.tensor(5.0), trace=torch.tensor(1000.0),
        consistency=torch.tensor(0.0), gradient=torch.tensor(0.0),
        spatial_fft=torch.tensor(0.0), trace_fft=torch.tensor(0.0),
    )
    assert data.item() == pytest.approx(22.0)
    assert auxiliary.item() == pytest.approx(0.0)
    assert total.item() == pytest.approx(22.0)


def test_best_checkpoint_requires_validation_improvement_and_resume(tmp_path: Path):
    torch.manual_seed(2)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = AtomicCheckpointManager(tmp_path)
    metadata = dict(normalizer=make_normalizer().metadata.to_dict(), dataset_digests={"train": "abc"},
                    config_digest="cfg", zero_baseline={"query": 1.0, "dense": 1.0})
    manager.save_epoch(model=model, optimizer=optimizer, epoch=1, step=3,
                       validation_score=.4, validation_metrics={"query": .3}, **metadata)
    first_best = load_state(tmp_path / "best.pt")
    manager.save_epoch(model=model, optimizer=optimizer, epoch=2, step=6,
                       validation_score=.6, validation_metrics={"query": .5}, **metadata)
    assert load_state(tmp_path / "best.pt")["validation_score"] == pytest.approx(.4)
    assert load_state(tmp_path / "last.pt")["epoch"] == 2
    restored = make_model(); restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = manager.restore(tmp_path / "best.pt", restored, restored_opt)
    assert state["step"] == 3 and first_best["config_digest"] == "cfg"
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_finetune_resume_keeps_optimizer_state_but_resets_lr_and_step(tmp_path: Path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.query_normalized(
        torch.full((1, 1, 17, 17), 2200.0),
        torch.tensor([[500., 500., 10., .05, 1.]]),
        torch.nn.functional.one_hot(torch.tensor([5 * 17 + 5]), 17 * 17).reshape(1, 1, 17, 17).float(),
        torch.tensor([[[500., 500., .2]]]), make_normalizer(),
    ).sum().backward()
    optimizer.step()
    manager = AtomicCheckpointManager(tmp_path)
    manager.save_epoch(
        model=model, optimizer=optimizer, epoch=0, step=4000, validation_score=.7,
        validation_metrics={}, normalizer=make_normalizer().metadata.to_dict(),
        dataset_digests={"train": "abc"}, config_digest="old",
        zero_baseline={"query": 1.0, "dense": 1.0},
    )
    restored_model = make_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=2e-4)
    trainer = JointDualHeadTrainer(restored_model, make_normalizer(), restored_optimizer,
                                   LossConfig())
    state = restore_for_finetune(manager, tmp_path / "best.pt", restored_model,
                                 restored_optimizer, trainer, learning_rate=2e-4)
    assert state["step"] == trainer.step == 4000
    assert all(group["lr"] == pytest.approx(2e-4) for group in restored_optimizer.param_groups)
    assert restored_optimizer.state


def test_lbfgs_resume_restores_weights_without_loading_adam_state(tmp_path: Path):
    model = make_model()
    adam = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = AtomicCheckpointManager(tmp_path)
    manager.save_epoch(
        model=model, optimizer=adam, epoch=0, step=6125, validation_score=.5,
        validation_metrics={}, normalizer=make_normalizer().metadata.to_dict(),
        dataset_digests={"train": "abc"}, config_digest="adam",
        zero_baseline={"query": 1.0, "dense": 1.0},
    )
    restored_model = make_model()
    lbfgs = torch.optim.LBFGS(restored_model.parameters(), lr=.5, max_iter=2,
                              history_size=5, line_search_fn="strong_wolfe")
    trainer = JointDualHeadTrainer(restored_model, make_normalizer(), lbfgs, LossConfig())
    state = restore_for_finetune(manager, tmp_path / "best.pt", restored_model,
                                 lbfgs, trainer, learning_rate=.5,
                                 restore_optimizer=False)
    assert state["step"] == trainer.step == 6125
    assert not lbfgs.state


def test_lbfgs_config_is_strict_and_explicit():
    config = V2Config.from_mapping({"train": {
        "optimizer": "lbfgs", "lbfgs_max_iter": 7,
        "lbfgs_history_size": 25, "lbfgs_line_search_fn": "strong_wolfe",
    }})
    assert config.train.optimizer == "lbfgs"
    assert config.train.lbfgs_max_iter == 7
    with pytest.raises(ValueError, match="optimizer"):
        V2Config.from_mapping({"train": {"optimizer": "not-an-optimizer"}})


def test_joint_step_trains_both_heads_and_passes_source_map():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    source_map = torch.zeros(1, 1, 17, 17); source_map[0, 0, 5, 6] = 1
    time_s = torch.tensor([.2, .3, .4, .5, .6])
    batch = V2MacroBatch(
        velocity_mps=torch.full((1, 1, 17, 17), 2200.0),
        record_to_medium=torch.zeros(1, dtype=torch.long),
        source_parameters=torch.tensor([[750., 625., 10., .05, 1.]]),
        source_map=source_map,
        dense_time_indices=torch.tensor([[0, 3]]),
        dense_target=torch.randn(1, 2, 17, 17) * 1e-8,
        receiver_zx_indices=torch.tensor([[[2, 4], [2, 12]]]),
        receiver_target=torch.randn(1, 2, 5) * 1e-8,
        query_coords=torch.tensor([[[500., 500., .2], [1000., 700., .3], [1400., 900., .5], [900., 1000., .6]]]),
        query_target=torch.randn(1, 4) * 1e-8,
        sample_probability=torch.full((1, 4), .25), time_s=time_s,
        sample_id=("a",), group_id=("g",), medium_type=("uniform",),
    )
    trainer = JointDualHeadTrainer(model, make_normalizer(), optimizer, LossConfig(), audit_every=1)
    metrics = trainer.run_step(batch)
    assert metrics["query_relative_l2"] > 0 and metrics["dense_relative_l2"] > 0
    assert "gradient_norm/query_head" in metrics and "gradient_norm/dense_decoder" in metrics

    lbfgs_model = make_model()
    lbfgs = torch.optim.LBFGS(lbfgs_model.parameters(), lr=.1, max_iter=2,
                              history_size=5, line_search_fn="strong_wolfe")
    lbfgs_trainer = JointDualHeadTrainer(lbfgs_model, make_normalizer(), lbfgs,
                                         LossConfig(), audit_every=1)
    lbfgs_metrics = lbfgs_trainer.run_lbfgs_step([batch, batch])
    assert lbfgs_metrics["closure_evaluations"] >= 2
    assert lbfgs_trainer.step == 1
    assert "gradient_norm/query_head" in lbfgs_metrics
