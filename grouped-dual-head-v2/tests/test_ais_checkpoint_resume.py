from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import fno_acoustic.query_training as query_training
from fno_acoustic.ais_sampler import AdaptiveSpatialSampler, SpatialSamplingFeatures
from fno_acoustic.query_training import (
    QueryResumeMetadata,
    load_query_checkpoint,
    save_query_checkpoint,
)


CONFIG_HASH = "a" * 64
SPLIT_HASH = "b" * 64
BOUNDARY = "after_optimizer_step_before_query_draw"
EXPECTED_FIELDS = {
    "schema_version",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "sampler_state_dict",
    "torch_rng_state",
    "cuda_rng_state_all",
    "numpy_rng_state",
    "python_rng_state",
    "epoch",
    "epoch_batch_cursor",
    "epoch_scene_order",
    "data_generator_state",
    "global_step",
    "config_sha256",
    "split_manifest_sha256",
}


def _write_attack_marker(path: str) -> None:
    Path(path).write_text("unsafe pickle executed", encoding="utf-8")


class _MaliciousCheckpoint:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return _write_attack_marker, (str(self.marker_path),)


class _ApplyAbort(BaseException):
    pass


class _TwinParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.right = torch.nn.Parameter(torch.tensor([3.0, 4.0]))

    def forward(self) -> torch.Tensor:
        return self.left.sum() + 2.0 * self.right.sum()


def _objects(seed: int = 23):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    sampler = AdaptiveSpatialSampler(
        4, 5, [0.2, 0.2, 0.2, 0.2, 0.1, 0.1], seed=seed, tile_size=2
    )
    data_generator = torch.Generator().manual_seed(seed + 17)
    return model, optimizer, scheduler, sampler, data_generator


def _features() -> SpatialSamplingFeatures:
    base = torch.arange(1, 21, dtype=torch.float64)
    return SpatialSamplingFeatures(
        base, base.flip(0), torch.ones(20), base.square(), base.sqrt()
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def _step(model, optimizer, scheduler, sampler, data_generator, step: int):
    draw = sampler.draw([3], [_features()], count=6)
    x = torch.randn(2, 3) + float(np.random.random()) + random.random()
    x = x + torch.rand((), generator=data_generator)
    optimizer.zero_grad(set_to_none=True)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    sampler.update_residual_tiles(
        draw.sample_ids,
        draw.site_indices,
        torch.full(draw.site_indices.shape, float(step + 1)),
    )
    return draw


def _save(path: Path, objects, **overrides) -> None:
    model, optimizer, scheduler, sampler, data_generator = objects
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        epoch=2,
        epoch_batch_cursor=2,
        epoch_scene_order=[3, 1, 4, 2],
        data_generator=data_generator,
        global_step=2,
        config_sha256=CONFIG_HASH,
        split_manifest_sha256=SPLIT_HASH,
        checkpoint_boundary=BOUNDARY,
    )
    kwargs.update(overrides)
    save_query_checkpoint(path, **kwargs)


def _load(path: Path, objects, **overrides) -> QueryResumeMetadata:
    model, optimizer, scheduler, sampler, data_generator = objects
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        data_generator=data_generator,
        expected_config_sha256=CONFIG_HASH,
        expected_split_manifest_sha256=SPLIT_HASH,
        map_location="cpu",
    )
    kwargs.update(overrides)
    return load_query_checkpoint(path, **kwargs)


def test_interrupted_training_reproduces_all_states_and_next_random_draws(
    tmp_path,
) -> None:
    continuous = _objects()
    for step in range(4):
        _step(*continuous, step)
    continuous_next = continuous[3].draw([3], [_features()], count=7)
    continuous_probe = (
        torch.rand(5),
        np.random.random(5),
        [random.random() for _ in range(5)],
        torch.rand(5, generator=continuous[4]),
    )

    interrupted = _objects()
    for step in range(2):
        _step(*interrupted, step)
    path = tmp_path / "nested" / "query.pt"
    _save(path, interrupted)

    resumed = _objects(seed=999)
    metadata = _load(path, resumed)
    assert metadata == QueryResumeMetadata(2, 2, (3, 1, 4, 2), 2)
    for step in range(2, 4):
        _step(*resumed, step)
    resumed_next = resumed[3].draw([3], [_features()], count=7)
    resumed_probe = (
        torch.rand(5),
        np.random.random(5),
        [random.random() for _ in range(5)],
        torch.rand(5, generator=resumed[4]),
    )

    assert torch.equal(resumed_next.site_indices, continuous_next.site_indices)
    _assert_nested_equal(resumed[0].state_dict(), continuous[0].state_dict())
    _assert_nested_equal(resumed[1].state_dict(), continuous[1].state_dict())
    _assert_nested_equal(resumed[2].state_dict(), continuous[2].state_dict())
    _assert_nested_equal(resumed[3].state_dict(), continuous[3].state_dict())
    _assert_nested_equal(resumed_probe, continuous_probe)


def test_payload_has_stable_exact_schema_and_loads_on_cpu(tmp_path) -> None:
    objects = _objects()
    path = tmp_path / "query.pt"
    _save(path, objects)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == EXPECTED_FIELDS
    assert payload["schema_version"] == 1
    for value in payload["model_state_dict"].values():
        assert value.device.type == "cpu"
    restored = _objects(seed=99)
    assert _load(path, restored).epoch_scene_order == (3, 1, 4, 2)


def test_schema_v3_cryptographically_binds_runtime_seed_on_resume(tmp_path) -> None:
    objects = _objects()
    path = tmp_path / "query-v3.pt"
    _save(path, objects, phase_index=0, phase_update=2, runtime_seed=20260714)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == 3
    assert payload["runtime_seed"] == 20260714
    assert _load(
        path, _objects(seed=99), expected_runtime_seed=20260714
    ).phase_update == 2
    with pytest.raises(ValueError, match="runtime seed"):
        _load(path, _objects(seed=100), expected_runtime_seed=20260715)


@pytest.mark.parametrize(
    ("payload_update", "load_update", "message"),
    [
        ({"schema_version": 2}, {}, "schema"),
        ({}, {"expected_config_sha256": "c" * 64}, "config"),
        ({}, {"expected_split_manifest_sha256": "d" * 64}, "split"),
        ({"epoch_batch_cursor": 5}, {}, "cursor"),
    ],
)
def test_load_rejects_mismatch_before_mutating_objects(
    tmp_path, payload_update, load_update, message
) -> None:
    source = _objects()
    path = tmp_path / "query.pt"
    _save(path, source)
    payload = torch.load(path, weights_only=True)
    payload.update(payload_update)
    torch.save(payload, path)
    target = _objects(seed=77)
    before = [copy.deepcopy(item.state_dict()) for item in target[:4]]
    generator_before = target[4].get_state().clone()

    with pytest.raises(ValueError, match=message):
        _load(path, target, **load_update)

    for item, state in zip(target[:4], before, strict=True):
        _assert_nested_equal(item.state_dict(), state)
    assert torch.equal(target[4].get_state(), generator_before)


def test_load_rejects_missing_field_and_corrupt_file(tmp_path) -> None:
    objects = _objects()
    path = tmp_path / "query.pt"
    _save(path, objects)
    payload = torch.load(path, weights_only=True)
    del payload["python_rng_state"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="fields"):
        _load(path, _objects(seed=98))

    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(Exception):
        _load(path, _objects(seed=98))


def test_load_rejects_malicious_pickle_without_executing_side_effect(tmp_path) -> None:
    path = tmp_path / "malicious.pt"
    marker = tmp_path / "executed.txt"
    torch.save({"payload": _MaliciousCheckpoint(marker)}, path)

    with pytest.raises(Exception):
        _load(path, _objects(seed=97))

    assert not marker.exists()


def test_scheduler_none_contract_is_strict(tmp_path) -> None:
    objects = list(_objects())
    objects[2] = None
    path = tmp_path / "query.pt"
    _save(path, objects)
    with pytest.raises(ValueError, match="scheduler"):
        _load(path, _objects(seed=41))

    with_scheduler = _objects(seed=42)
    _save(path, with_scheduler)
    target = list(_objects(seed=43))
    target[2] = None
    with pytest.raises(ValueError, match="scheduler"):
        _load(path, target)


def test_malformed_scheduler_state_is_rejected_without_any_state_pollution(
    tmp_path,
) -> None:
    path = tmp_path / "query.pt"
    _save(path, _objects())
    payload = torch.load(path, weights_only=True)
    payload["scheduler_state_dict"] = {}
    torch.save(payload, path)
    target = _objects(seed=47)
    component_before = [copy.deepcopy(item.state_dict()) for item in target[:4]]
    data_generator_before = target[4].get_state().clone()
    torch_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    with pytest.raises(ValueError, match="scheduler.*structure"):
        _load(path, target)

    for item, state in zip(target[:4], component_before, strict=True):
        _assert_nested_equal(item.state_dict(), state)
    assert torch.equal(target[4].get_state(), data_generator_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
    _assert_nested_equal(torch.cuda.get_rng_state_all(), cuda_before)
    _assert_nested_equal(np.random.get_state(), numpy_before)
    assert random.getstate() == python_before


@pytest.mark.parametrize(
    "overrides",
    [
        {"checkpoint_boundary": "during_query_draw"},
        {"epoch": -1},
        {"global_step": True},
        {"epoch_batch_cursor": 3, "epoch_scene_order": [1, 2]},
        {"epoch_scene_order": [1, 1]},
    ],
)
def test_save_rejects_ambiguous_boundary_and_invalid_resume_metadata(
    tmp_path, overrides
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _save(tmp_path / "query.pt", _objects(), **overrides)


def test_atomic_replace_preserves_old_checkpoint_on_save_failure_and_cleans_tmp(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "query.pt"
    path.write_bytes(b"old checkpoint")
    original_save = query_training.torch.save

    def failing_save(payload, file_object):
        file_object.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(query_training.torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        _save(path, _objects())
    assert path.read_bytes() == b"old checkpoint"
    assert list(tmp_path.glob("*.tmp")) == []

    monkeypatch.setattr(query_training.torch, "save", original_save)
    replaced = []
    original_replace = os.replace

    def recording_replace(source, destination):
        replaced.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(query_training.os, "replace", recording_replace)
    _save(path, _objects())
    assert len(replaced) == 1
    assert replaced[0][0].parent == path.parent
    assert replaced[0][0].name.endswith(".tmp")
    assert replaced[0][1] == path
    assert list(tmp_path.glob("*.tmp")) == []


def test_replace_failure_preserves_old_checkpoint_and_cleans_tmp(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "query.pt"
    path.write_bytes(b"old checkpoint")

    def failing_replace(source, destination):
        raise OSError("rename failed")

    monkeypatch.setattr(query_training.os, "replace", failing_replace)
    with pytest.raises(OSError, match="rename failed"):
        _save(path, _objects())

    assert path.read_bytes() == b"old checkpoint"
    assert list(tmp_path.glob("*.tmp")) == []


def test_file_fsync_failure_preserves_old_checkpoint_and_cleans_tmp(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "query.pt"
    path.write_bytes(b"old checkpoint")

    def failing_fsync(file_descriptor):
        raise OSError("file fsync failed")

    monkeypatch.setattr(query_training.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="file fsync failed"):
        _save(path, _objects())

    assert path.read_bytes() == b"old checkpoint"
    assert list(tmp_path.glob("*.tmp")) == []


def test_directory_fsync_failure_after_replace_reports_committed_success(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "query.pt"
    path.write_bytes(b"old checkpoint")
    original_fsync = os.fsync
    fsync_calls = 0

    def fail_second_fsync(file_descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("directory fsync failed after commit")
        return original_fsync(file_descriptor)

    monkeypatch.setattr(query_training.os, "fsync", fail_second_fsync)
    _save(path, _objects())

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == EXPECTED_FIELDS
    assert payload["schema_version"] == 1
    assert fsync_calls == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_rolls_back_when_component_restore_fails(tmp_path, monkeypatch) -> None:
    source = _objects()
    path = tmp_path / "query.pt"
    _save(path, source)
    target = _objects(seed=81)
    before = copy.deepcopy(target[0].state_dict())
    original_load = target[3].load_state_dict
    calls = 0

    def fail_once(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("sampler restore failed")
        return original_load(state)

    monkeypatch.setattr(target[3], "load_state_dict", fail_once)
    with pytest.raises(ValueError, match="sampler restore failed"):
        _load(path, target)
    _assert_nested_equal(target[0].state_dict(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rollback_model_snapshot_is_stored_on_cpu(tmp_path, monkeypatch) -> None:
    path = tmp_path / "query.pt"
    _save(path, _objects())
    target = _objects(seed=86)
    target[0].cuda()
    model_load_states = []
    original_model_load = target[0].load_state_dict
    original_sampler_load = target[3].load_state_dict
    sampler_calls = 0

    def recording_model_load(state):
        model_load_states.append(state)
        return original_model_load(state)

    def fail_sampler_apply_once(state):
        nonlocal sampler_calls
        sampler_calls += 1
        if sampler_calls == 1:
            raise ValueError("apply failed")
        return original_sampler_load(state)

    monkeypatch.setattr(target[0], "load_state_dict", recording_model_load)
    monkeypatch.setattr(target[3], "load_state_dict", fail_sampler_apply_once)
    with pytest.raises(ValueError, match="apply failed"):
        _load(path, target)

    assert len(model_load_states) == 2
    assert all(
        tensor.device.type == "cpu"
        for tensor in model_load_states[1].values()
        if isinstance(tensor, torch.Tensor)
    )


def test_baseexception_rollback_failure_preserves_original_and_continues_rng_restore(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "query.pt"
    _save(path, _objects())
    target = _objects(seed=87)
    original_model_load = target[0].load_state_dict
    original_torch_set = torch.set_rng_state
    original_python_set = random.setstate
    model_calls = 0
    torch_set_calls = 0
    python_set_calls = 0

    def fail_model_rollback(state):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 2:
            raise RuntimeError("model rollback failed")
        return original_model_load(state)

    def record_torch_set(state):
        nonlocal torch_set_calls
        torch_set_calls += 1
        return original_torch_set(state)

    def fail_python_apply_once(state):
        nonlocal python_set_calls
        python_set_calls += 1
        if python_set_calls == 1:
            raise _ApplyAbort("original apply abort")
        return original_python_set(state)

    monkeypatch.setattr(target[0], "load_state_dict", fail_model_rollback)
    monkeypatch.setattr(query_training.torch, "set_rng_state", record_torch_set)
    monkeypatch.setattr(query_training.random, "setstate", fail_python_apply_once)

    with pytest.raises(_ApplyAbort, match="original apply abort") as error:
        _load(path, target)

    assert model_calls == 2
    assert torch_set_calls == 2
    assert python_set_calls == 2
    rollback_notes = " ".join(getattr(error.value, "__notes__", ()))
    rollback_notes += " ".join(getattr(error.value, "rollback_failures", ()))
    assert "model rollback failed" in rollback_notes


def test_model_dtype_mismatch_is_rejected_before_component_apply(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "query.pt"
    _save(path, _objects())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    first_key = next(iter(payload["model_state_dict"]))
    payload["model_state_dict"][first_key] = payload["model_state_dict"][
        first_key
    ].double()
    torch.save(payload, path)
    target = _objects(seed=83)
    apply_calls = 0
    original_load = target[0].load_state_dict

    def recording_load(state):
        nonlocal apply_calls
        apply_calls += 1
        return original_load(state)

    monkeypatch.setattr(target[0], "load_state_dict", recording_load)
    with pytest.raises(ValueError, match="model.*dtype"):
        _load(path, target)
    assert apply_calls == 0


@pytest.mark.parametrize(
    "corruption",
    ["lr_string", "lr_nan", "param_order", "state_shape", "state_dtype"],
)
def test_optimizer_corruption_is_rejected_before_component_apply(
    tmp_path, monkeypatch, corruption
) -> None:
    source = _objects()
    _step(*source, 0)
    path = tmp_path / "query.pt"
    _save(path, source)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    optimizer_state = payload["optimizer_state_dict"]
    group = optimizer_state["param_groups"][0]
    first_parameter_id = group["params"][0]
    if corruption == "lr_string":
        group["lr"] = "bad"
    elif corruption == "lr_nan":
        group["lr"] = float("nan")
    elif corruption == "param_order":
        group["params"] = list(reversed(group["params"]))
    elif corruption == "state_shape":
        optimizer_state["state"][first_parameter_id]["momentum_buffer"] = torch.zeros(
            2, 2
        )
    else:
        optimizer_state["state"][first_parameter_id]["momentum_buffer"] = (
            optimizer_state["state"][first_parameter_id]["momentum_buffer"].double()
        )
    torch.save(payload, path)
    target = _objects(seed=84)
    model_before = copy.deepcopy(target[0].state_dict())
    torch_before = torch.get_rng_state().clone()
    apply_calls = 0
    original_load = target[0].load_state_dict

    def recording_load(state):
        nonlocal apply_calls
        apply_calls += 1
        return original_load(state)

    monkeypatch.setattr(target[0], "load_state_dict", recording_load)
    with pytest.raises(ValueError, match="optimizer"):
        _load(path, target)
    assert apply_calls == 0
    _assert_nested_equal(target[0].state_dict(), model_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_valid_adam_parameter_state_restores(tmp_path) -> None:
    source = list(_objects())
    source[1] = torch.optim.Adam(source[0].parameters(), lr=0.004)
    source[2] = torch.optim.lr_scheduler.StepLR(source[1], step_size=1, gamma=0.9)
    _step(*source, 0)
    path = tmp_path / "adam.pt"
    _save(path, source)

    target = list(_objects(seed=85))
    target[1] = torch.optim.Adam(target[0].parameters(), lr=0.004)
    target[2] = torch.optim.lr_scheduler.StepLR(target[1], step_size=1, gamma=0.9)
    _load(path, target)

    _assert_nested_equal(target[0].state_dict(), source[0].state_dict())
    _assert_nested_equal(target[1].state_dict(), source[1].state_dict())


def test_same_shape_parameter_id_reversal_is_rejected_before_apply(
    tmp_path, monkeypatch
) -> None:
    source_model = _TwinParameterModel()
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.03, momentum=0.8)
    source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=1)
    source_model().backward()
    source_optimizer.step()
    source_scheduler.step()
    source = (
        source_model,
        source_optimizer,
        source_scheduler,
        _objects()[3],
        torch.Generator().manual_seed(40),
    )
    path = tmp_path / "same-shape.pt"
    _save(path, source)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    group = payload["optimizer_state_dict"]["param_groups"][0]
    first_id, second_id = group["params"]
    first_momentum = payload["optimizer_state_dict"]["state"][first_id][
        "momentum_buffer"
    ]
    second_momentum = payload["optimizer_state_dict"]["state"][second_id][
        "momentum_buffer"
    ]
    assert not torch.equal(first_momentum, second_momentum)
    group["params"] = [second_id, first_id]
    torch.save(payload, path)

    target_model = _TwinParameterModel()
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.03, momentum=0.8)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)
    target = (
        target_model,
        target_optimizer,
        target_scheduler,
        _objects(seed=90)[3],
        torch.Generator().manual_seed(41),
    )
    apply_calls = 0
    original_load = target_model.load_state_dict

    def recording_load(state):
        nonlocal apply_calls
        apply_calls += 1
        return original_load(state)

    monkeypatch.setattr(target_model, "load_state_dict", recording_load)
    with pytest.raises(ValueError, match="optimizer parameter ID.*order"):
        _load(path, target)
    assert apply_calls == 0


def test_lbfgs_is_rejected_at_save_and_load_entrypoints(tmp_path) -> None:
    model = torch.nn.Linear(3, 1)
    unsupported = torch.optim.LBFGS(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(unsupported, step_size=1)
    objects = (
        model,
        unsupported,
        scheduler,
        _objects()[3],
        torch.Generator().manual_seed(42),
    )
    with pytest.raises(TypeError, match="SGD.*Adam.*AdamW"):
        _save(tmp_path / "lbfgs.pt", objects)

    path = tmp_path / "supported.pt"
    _save(path, _objects())
    with pytest.raises(TypeError, match="SGD.*Adam.*AdamW"):
        _load(path, objects)


def test_valid_adamw_parameter_state_restores(tmp_path) -> None:
    source = list(_objects())
    source[1] = torch.optim.AdamW(source[0].parameters(), lr=0.004)
    source[2] = torch.optim.lr_scheduler.StepLR(source[1], step_size=1, gamma=0.9)
    _step(*source, 0)
    path = tmp_path / "adamw.pt"
    _save(path, source)

    target = list(_objects(seed=91))
    target[1] = torch.optim.AdamW(target[0].parameters(), lr=0.004)
    target[2] = torch.optim.lr_scheduler.StepLR(target[1], step_size=1, gamma=0.9)
    _load(path, target)

    _assert_nested_equal(target[0].state_dict(), source[0].state_dict())
    _assert_nested_equal(target[1].state_dict(), source[1].state_dict())


@pytest.mark.parametrize(("saved_count", "device_count"), [(0, 1), (1, 0), (1, 2)])
def test_cuda_rng_topology_mismatch_is_rejected_before_component_apply(
    tmp_path, monkeypatch, saved_count, device_count
) -> None:
    path = tmp_path / "query.pt"
    _save(path, _objects())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["cuda_rng_state_all"] = [
        payload["torch_rng_state"].clone() for _ in range(saved_count)
    ]
    torch.save(payload, path)
    target = _objects(seed=82)
    model_before = copy.deepcopy(target[0].state_dict())
    torch_before = torch.get_rng_state().clone()
    apply_calls = 0
    original_model_load = target[0].load_state_dict

    def recording_model_load(state):
        nonlocal apply_calls
        apply_calls += 1
        return original_model_load(state)

    monkeypatch.setattr(target[0], "load_state_dict", recording_model_load)
    monkeypatch.setattr(query_training.torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(query_training.torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA RNG.*count"):
        _load(path, target)

    assert apply_calls == 0
    _assert_nested_equal(target[0].state_dict(), model_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_state_is_exactly_restored(tmp_path) -> None:
    objects = _objects()
    torch.cuda.manual_seed_all(314)
    path = tmp_path / "query.pt"
    _save(path, objects)
    expected = torch.rand(8, device="cuda")
    torch.cuda.manual_seed_all(999)
    _load(path, _objects(seed=88))
    actual = torch.rand(8, device="cuda")
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_map_location_keeps_rng_and_sampler_metadata_restorable(tmp_path) -> None:
    source = _objects()
    source[3].draw([3], [_features()], count=2)
    path = tmp_path / "query.pt"
    _save(path, source)

    target = _objects(seed=91)
    metadata = _load(path, target, map_location="cuda")

    assert metadata.global_step == 2
    _assert_nested_equal(target[3].state_dict(), source[3].state_dict())
