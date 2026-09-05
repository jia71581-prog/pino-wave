from __future__ import annotations

import copy
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import frozen_loss
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import (
    BASIS_FILE_SHA256,
    BindingRefusal,
    FROZEN_OPTIMIZATION,
    PARENT_PATH,
    atomic_json_exclusive,
    checkpoint_payload,
    cosine_learning_rate,
    exact_microblocked_loss,
    load_checkpoint,
    make_optimizer,
    require_cuda_environment,
    save_best_last,
    selection_metric,
    space_gate,
    validate_checkpoint_payload,
    weighted_coefficient_target,
)


def _model_optimizer():
    model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2))
    return model, make_optimizer(list(model.parameters()))


def _step(model, optimizer):
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), FROZEN_OPTIMIZATION.grad_clip_norm)
    optimizer.step()
    return float(loss)


def test_frozen_hyperparameters_and_schedule() -> None:
    cfg = FROZEN_OPTIMIZATION
    assert (cfg.learning_rate, cfg.betas, cfg.epsilon, cfg.weight_decay) == (
        3e-3, (0.9, 0.99), 1e-8, 1e-4
    )
    assert (cfg.grad_clip_norm, cfg.batch_records, cfg.time_microblock, cfg.seed) == (1.0, 1, 32, 372)
    assert cosine_learning_rate(0) == pytest.approx(3e-3)
    assert cosine_learning_rate(19) == pytest.approx(3e-5)


def test_microblocked_loss_matches_direct_full_future() -> None:
    generator = torch.Generator().manual_seed(372)
    truth = torch.randn(1, 65, 5, 7, generator=generator)
    prediction = truth + 0.03 * torch.randn(truth.shape, generator=generator)
    coefficients = torch.randn(1, 16, 5, 7, generator=generator)
    streamed = exact_microblocked_loss(prediction, truth, coefficients)
    direct = frozen_loss(prediction, truth, coefficients)
    for key in direct:
        assert torch.allclose(streamed[key], direct[key], rtol=2e-6, atol=1e-8)


def test_weighted_coefficient_target_is_finite_and_not_serialized(tmp_path) -> None:
    generator = torch.Generator().manual_seed(372)
    q, _ = torch.linalg.qr(torch.randn(401, 16, generator=generator, dtype=torch.float64))
    parent = torch.randn(401, 2, 3, generator=generator)
    truth = parent + 0.01 * torch.randn(parent.shape, generator=generator)
    coefficients = weighted_coefficient_target(q, parent, truth, k1=12)
    assert coefficients.shape == (16, 2, 3)
    assert torch.isfinite(coefficients).all()
    assert list(tmp_path.iterdir()) == []


def test_cpu_resume_equivalence_and_rng_restoration(tmp_path) -> None:
    random.seed(372); np.random.seed(372); torch.manual_seed(372)
    reference, reference_optimizer = _model_optimizer()
    initial = copy.deepcopy(reference.state_dict())
    for _ in range(4):
        _step(reference, reference_optimizer)

    random.seed(372); np.random.seed(372); torch.manual_seed(372)
    interrupted, interrupted_optimizer = _model_optimizer()
    interrupted.load_state_dict(initial)
    for _ in range(2):
        _step(interrupted, interrupted_optimizer)
    identity = {"candidate": "r16_dscp_v2", "run_digest": "toy"}
    payload = checkpoint_payload(
        interrupted, interrupted_optimizer, run_identity=identity,
        sampler_order=[0, 1, 2], progress={"update": 2, "epoch": 0},
    )
    paths = save_best_last(payload, tmp_path / "run", is_best=True)
    assert paths["hardlinked"] is True
    resumed, resumed_optimizer = _model_optimizer()
    load_checkpoint(paths["last"], resumed, resumed_optimizer, expected_run_identity=identity)
    for _ in range(2):
        _step(resumed, resumed_optimizer)
    for left, right in zip(reference.parameters(), resumed.parameters()):
        assert torch.equal(left, right)


def test_corrupt_and_binding_refusal(tmp_path) -> None:
    model, optimizer = _model_optimizer()
    identity = {"candidate": "r16_dscp_v2", "run_digest": "toy"}
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not-a-checkpoint")
    with pytest.raises(BindingRefusal):
        load_checkpoint(corrupt, model, optimizer, expected_run_identity=identity)
    payload = checkpoint_payload(model, optimizer, run_identity=identity, sampler_order=[0], progress={})
    payload["basis"]["file_sha256"] = "0" * 64
    with pytest.raises(BindingRefusal):
        validate_checkpoint_payload(payload, expected_run_identity=identity)


def test_checkpoint_never_embeds_parent_or_basis_tensor() -> None:
    model, optimizer = _model_optimizer()
    identity = {"candidate": "r16_dscp_v2", "run_digest": "toy"}
    payload = checkpoint_payload(model, optimizer, run_identity=identity, sampler_order=[0], progress={})
    assert payload["parent"] == {"path": PARENT_PATH, "sha256": payload["parent"]["sha256"]}
    assert payload["basis"]["file_sha256"] == BASIS_FILE_SHA256
    assert "basis_tensor" not in payload
    assert "parent_state" not in payload and "parent_model" not in payload


def test_best_last_retention_and_refuse_overwrite(tmp_path) -> None:
    model, optimizer = _model_optimizer()
    identity = {"candidate": "r16_dscp_v2", "run_digest": "toy"}
    payload = checkpoint_payload(model, optimizer, run_identity=identity, sampler_order=[0], progress={})
    paths = save_best_last(payload, tmp_path / "run", is_best=True)
    assert {path.name for path in (tmp_path / "run").glob("*.pt")} == {"best.pt", "last.pt"}
    assert paths["hardlinked"]
    terminal = tmp_path / "terminal.json"
    atomic_json_exclusive({"status": "failed"}, terminal)
    with pytest.raises(FileExistsError):
        atomic_json_exclusive({"status": "passed"}, terminal)


def test_cuda_environment_fails_closed(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(BindingRefusal):
        require_cuda_environment(visible_devices="0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require_cuda_environment(visible_devices="0")


def test_selection_metric_and_space_formula(tmp_path) -> None:
    truth = [torch.ones(3, 2), torch.ones(3, 2) * 2]
    prediction = [truth[0] * 1.1, truth[1] * 0.9]
    assert selection_metric(prediction, truth) == pytest.approx(0.1)
    gate = space_gate(12345, tmp_path)
    assert gate["required_bytes"] == 2 * 1024**3 + 3 * 12345 + 64 * 1024**2
