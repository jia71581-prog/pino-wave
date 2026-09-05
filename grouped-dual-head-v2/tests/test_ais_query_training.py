from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch import nn

from fno_acoustic.ais_sampler import AdaptiveSpatialSampler, SpatialSamplingFeatures
from fno_acoustic.query_data import QueryScene
from fno_acoustic.query_training import (
    QUERY_CHECKPOINT_BOUNDARY,
    _memory_bounded_field_backward,
    build_scene_model_inputs,
    build_spatial_sampling_features,
    curriculum_phase_events,
    normalize_physical_xz,
    query_field_loss_chunked,
    receiver_site_indices,
    load_query_checkpoint,
    save_query_checkpoint,
    centered_site_offsets,
    decode_halo_patches,
    gather_receiver_targets,
    train_query_epoch,
    validate_query_guard,
)
from fno_acoustic.query_losses import auxiliary_query_losses, hansen_hurwitz_field_loss


class TinyQueryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7))
        self.bias = nn.Parameter(torch.tensor(-0.1))

    def encode_global(self, global_inputs, time_s):
        return global_inputs.mean(dim=(1, 2, 4)) * self.weight

    def decode_queries(self, context, native_static, query_xz, time_s):
        base = context[:, None, :] + query_xz.sum(-1, keepdim=True)
        return base * self.weight + self.bias


class MemoryStore:
    def __init__(self, scenes):
        self.scenes = {scene.sample_id: scene for scene in scenes}

    def read_scene(self, sample_id):
        return self.scenes[int(sample_id)]


def make_scene(sample_id=0, height=7, width=6):
    time = torch.linspace(0.0, 1.0, 160, dtype=torch.float64)
    x = torch.linspace(0.0, 600.0, height, dtype=torch.float64)
    z = torch.linspace(0.0, 500.0, width, dtype=torch.float64)
    xx, zz, tt = torch.meshgrid(x.float() / 600, z.float() / 500, time.float(), indexing="ij")
    target = torch.sin(2 * torch.pi * tt) + 0.1 * xx + 0.2 * zz
    velocity = 1500.0 + 100.0 * xx[..., 0] + 50.0 * zz[..., 0]
    source = torch.zeros(height, width)
    source[height // 2, width // 2] = 1.0
    return QueryScene(sample_id, target, velocity, source, time, x, z, {})


def config(height=7, width=6, query_count=11, chunk=4):
    return {
        "seed": 29,
        "sampling": {"target_height": height, "target_width": width, "global_size": 4},
        "receiver": {"x_start_m": 0.0, "x_stop_m": 600.0, "x_stride_m": 300.0, "z_m": [0.0]},
        "train": {
            "query_sites_per_scene": query_count,
            "query_chunk_size": chunk,
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


def make_sampler(height=7, width=6, seed=29):
    return AdaptiveSpatialSampler(height, width, [0.3, 0.2, 0.2, 0.1, 0.1, 0.1], seed, tile_size=3)


def test_scene_model_inputs_record_uniform_positive_grid_spacing() -> None:
    scene = make_scene()
    inputs = build_scene_model_inputs(scene, 4, torch.device("cpu"))
    assert inputs.dx_m == pytest.approx(100.0)
    assert inputs.dz_m == pytest.approx(100.0)

    nonuniform = QueryScene(
        scene.sample_id,
        scene.target_cpu,
        scene.velocity_cpu,
        scene.source_cpu,
        scene.time_s,
        torch.tensor([0.0, 100.0, 210.0, 300.0, 400.0, 500.0, 600.0]),
        scene.z_m,
        scene.metadata,
    )
    with pytest.raises(ValueError, match="uniform.*x_m|x_m.*uniform"):
        build_scene_model_inputs(nonuniform, 4, torch.device("cpu"))


def test_scene_model_inputs_own_a_sealed_time_snapshot() -> None:
    scene = make_scene()
    original = scene.time_s.clone()
    inputs = build_scene_model_inputs(scene, 4, torch.device("cpu"))

    scene.time_s[80] = scene.time_s[79]

    assert torch.equal(inputs.time_s, original)
    assert inputs.time_s.data_ptr() != scene.time_s.data_ptr()
    inputs.validated_time_grid.assert_current()


def test_chunked_query_backward_matches_unchunked_with_remainder():
    torch.manual_seed(4)
    scene = make_scene()
    inputs = build_scene_model_inputs(scene, 4, torch.device("cpu"))
    indices = torch.arange(19) % (scene.target_cpu.shape[0] * scene.target_cpu.shape[1])
    physical = torch.stack((scene.x_m[indices // 6], scene.z_m[indices % 6]), -1)
    query = normalize_physical_xz(physical, scene.x_m, scene.z_m).float()[None]
    target = scene.target_cpu.reshape(-1, 160)[indices][None]
    probability = torch.linspace(0.01, 0.1, 19)[None]
    left, right = TinyQueryModel(), TinyQueryModel()
    right.load_state_dict(left.state_dict())

    loss_left, _ = query_field_loss_chunked(left, inputs, query, target, probability, 42, 7, True)
    loss_left.backward()
    loss_right, _ = query_field_loss_chunked(right, inputs, query, target, probability, 42, 19, True)
    loss_right.backward()

    assert float(loss_left.detach()) == pytest.approx(float(loss_right.detach()), abs=1e-7)
    for a, b in zip(left.parameters(), right.parameters(), strict=True):
        assert torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-6)


def test_memory_bounded_train_update_matches_monolithic_global_objective():
    torch.manual_seed(17)
    scene = make_scene()
    bounded, reference = TinyQueryModel(), TinyQueryModel()
    reference.load_state_dict(bounded.state_dict())
    bounded_optimizer = torch.optim.SGD(bounded.parameters(), lr=1.0e-3)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1.0e-3)
    bounded_sampler = make_sampler()
    reference_sampler = make_sampler()
    cfg = config(query_count=19, chunk=7)

    train_query_epoch(
        bounded,
        MemoryStore([scene]),
        [0],
        bounded_sampler,
        bounded_optimizer,
        cfg,
        torch.device("cpu"),
        0,
        0,
    )

    receivers = receiver_site_indices(scene, cfg["receiver"])
    features = build_spatial_sampling_features(scene, receivers)
    draw = reference_sampler.draw([0], [features], count=19)
    sites = scene.target_cpu.reshape(-1, 160).index_select(0, draw.site_indices[0])[None]
    flat = draw.site_indices[0]
    physical = torch.stack((scene.x_m[flat // 6], scene.z_m[flat % 6]), dim=-1)
    query = normalize_physical_xz(physical, scene.x_m, scene.z_m).float()[None]
    inputs = build_scene_model_inputs(scene, 4, torch.device("cpu"))
    reference_optimizer.zero_grad(set_to_none=True)
    loss, _ = query_field_loss_chunked(
        reference, inputs, query, sites, draw.draw_probability, 42, 19, True
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 10.0, error_if_nonfinite=True)
    reference_optimizer.step()

    for actual, expected in zip(bounded.parameters(), reference.parameters(), strict=True):
        assert torch.allclose(actual, expected, atol=1.0e-7, rtol=1.0e-6)


def test_memory_bounded_backward_matches_all_auxiliary_gradients():
    class PartitionedQueryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_only = nn.Parameter(torch.tensor(0.6))
            self.decoder_only = nn.Parameter(torch.tensor(-0.2))
            self.shared = nn.Parameter(torch.tensor(0.4))

        def encode_global(self, global_inputs, time_s):
            base = global_inputs.mean(dim=(1, 2, 4))
            return base * self.encoder_only + self.shared

        def decode_queries(self, context, native_static, query_xz, time_s):
            local = query_xz.sum(-1, keepdim=True)
            return context[:, None, :] * self.shared + local * self.decoder_only

    torch.manual_seed(23)
    scene = make_scene()
    inputs = build_scene_model_inputs(scene, 4, torch.device("cpu"))
    indices = torch.arange(19) % 42
    physical = torch.stack((scene.x_m[indices // 6], scene.z_m[indices % 6]), dim=-1)
    query = normalize_physical_xz(physical, scene.x_m, scene.z_m).float()[None]
    target = scene.target_cpu.reshape(-1, 160).index_select(0, indices)[None]
    probability = torch.linspace(0.01, 0.1, 19)[None]
    receivers = receiver_site_indices(scene, config()["receiver"])
    receiver_xz, receiver_target = gather_receiver_targets(scene, receivers)
    patch_centers = torch.tensor([15, 20], dtype=torch.long)
    weights = (0.3, 0.2, 0.1, 0.4)
    bounded, monolithic = PartitionedQueryModel(), PartitionedQueryModel()
    monolithic.load_state_dict(bounded.state_dict())

    bounded.zero_grad(set_to_none=True)
    bounded_context = bounded.encode_global(inputs.global_inputs, inputs.time_s)
    bounded_field, _, context_leaf = _memory_bounded_field_backward(
        bounded,
        bounded_context,
        inputs,
        query,
        target,
        probability,
        42,
        7,
        True,
        None,
    )
    bounded_receiver = bounded.decode_queries(
        context_leaf, inputs.native_static, receiver_xz.float(), inputs.time_s
    )
    bounded_patch, patch_target = decode_halo_patches(
        bounded, inputs, scene, patch_centers, 3, context=context_leaf
    )
    bounded_aux = auxiliary_query_losses(
        bounded_receiver,
        receiver_target,
        bounded_patch,
        patch_target,
        inputs.time_s,
    )
    bounded_aux_total = sum(
        weight * value
        for weight, value in zip(weights, bounded_aux.__dict__.values(), strict=True)
    )
    bounded_total = bounded_field.loss.detach() + bounded_aux_total.detach()
    bounded_aux_total.backward()
    bounded_context.backward(context_leaf.grad)

    monolithic.zero_grad(set_to_none=True)
    monolithic_context = monolithic.encode_global(inputs.global_inputs, inputs.time_s)
    monolithic_prediction = monolithic.decode_queries(
        monolithic_context, inputs.native_static, query, inputs.time_s
    )
    monolithic_field = hansen_hurwitz_field_loss(
        monolithic_prediction, target, probability, population_size=42
    )
    monolithic_receiver = monolithic.decode_queries(
        monolithic_context, inputs.native_static, receiver_xz.float(), inputs.time_s
    )
    monolithic_patch, _ = decode_halo_patches(
        monolithic, inputs, scene, patch_centers, 3, context=monolithic_context
    )
    monolithic_aux = auxiliary_query_losses(
        monolithic_receiver,
        receiver_target,
        monolithic_patch,
        patch_target,
        inputs.time_s,
    )
    monolithic_total = monolithic_field.loss + sum(
        weight * value
        for weight, value in zip(weights, monolithic_aux.__dict__.values(), strict=True)
    )
    monolithic_total.backward()

    assert float(bounded_total) == pytest.approx(float(monolithic_total.detach()), abs=1.0e-6)
    bounded_parameters = dict(bounded.named_parameters())
    monolithic_parameters = dict(monolithic.named_parameters())
    assert set(bounded_parameters) == {"encoder_only", "decoder_only", "shared"}
    for name in bounded_parameters:
        assert torch.allclose(
            bounded_parameters[name].grad,
            monolithic_parameters[name].grad,
            atol=2.0e-5,
            rtol=2.0e-5,
        ), name


def test_train_epoch_moves_only_full_trace_queries_from_dense_labels(monkeypatch):
    scene = make_scene(height=400, width=400)
    store = MemoryStore([scene])
    seen = []
    original = torch.Tensor.to

    def audited(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            seen.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", audited)
    model = TinyQueryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    result = train_query_epoch(model, store, [0], make_sampler(400, 400), optimizer,
                               config(400, 400, 11, 4), torch.device("cpu"), 0, 0)
    assert result.full160_label_sites == 11
    assert (400, 400, 160) not in seen
    assert any(shape[-2:] == (11, 160) or shape[-2:] == (4, 160) for shape in seen)


def test_train_scene_encodes_once_and_sampler_residual_uses_pre_step_prediction():
    class SpyModel(TinyQueryModel):
        def __init__(self):
            super().__init__()
            self.encode_calls = 0
            self.decoded = []

        def encode_global(self, global_inputs, time_s):
            self.encode_calls += 1
            return super().encode_global(global_inputs, time_s)

        def decode_queries(self, context, native_static, query_xz, time_s):
            value = super().decode_queries(context, native_static, query_xz, time_s)
            self.decoded.append(value.detach().cpu().clone())
            return value

    class SpySampler(AdaptiveSpatialSampler):
        def update_residual_tiles(self, sample_ids, site_indices, squared_error):
            self.last_error = squared_error.clone()
            self.last_sites = site_indices.clone()
            super().update_residual_tiles(sample_ids, site_indices, squared_error)

    scene = make_scene()
    model = SpyModel()
    sampler = SpySampler(7, 6, [0.3, 0.2, 0.2, 0.1, 0.1, 0.1], 29, tile_size=3)
    train_query_epoch(model, MemoryStore([scene]), [0], sampler,
                      torch.optim.SGD(model.parameters(), lr=1.0), config(query_count=11, chunk=4),
                      torch.device("cpu"), 0, 0)
    assert model.encode_calls == 1
    assert len(model.decoded) == 6
    prediction = torch.cat(model.decoded[:3], dim=1)
    target = scene.target_cpu.reshape(-1, 160).index_select(0, sampler.last_sites[0])[None]
    assert torch.allclose(sampler.last_error, (prediction - target).square().mean(-1))


def test_validation_does_not_mutate_training_sampler_state():
    scene = make_scene()
    store = MemoryStore([scene])
    sampler = make_sampler()
    features = build_spatial_sampling_features(scene, torch.tensor([0]))
    sampler.draw([0], [features], 3)
    before = copy.deepcopy(sampler.state_dict())
    metrics = validate_query_guard(TinyQueryModel(), store, [0], {0: torch.tensor([0, 5, 11])},
                                   config(), torch.device("cpu"))
    assert metrics["physical_samples"] == 1.0
    assert_nested_equal(sampler.state_dict(), before)


def test_fixed_validation_sites_cover_resized_grid_and_invalid_manifest_rejected():
    from scripts.train_ais_mqfno import _fixed_validation_sites

    sites = _fixed_validation_sites([7], 64, 64, 2048)[7]
    assert sites.numel() == torch.unique(sites).numel() == 2048
    assert int(sites.min()) == 0
    assert int(sites.max()) >= 4094
    scene = make_scene(7, 400, 400)
    for invalid in (torch.tensor([0.0]), torch.tensor([-1]), torch.tensor([4096]), torch.tensor([1, 1])):
        with pytest.raises((TypeError, ValueError), match="validation|site|unique|bounds"):
            validate_query_guard(TinyQueryModel(), MemoryStore([scene]), [7], {7: invalid},
                                 config(64, 64), torch.device("cpu"))


def test_partial_epoch_cursor_resumes_at_next_scene_and_does_not_commit():
    scenes = [make_scene(i) for i in range(3)]
    sampler = make_sampler()
    model = TinyQueryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    first = train_query_epoch(model, MemoryStore(scenes), [0, 1, 2], sampler, optimizer,
                              config(), torch.device("cpu"), 0, 0, max_scene_draws=1)
    assert first.next_epoch_batch_cursor == 1
    assert sampler.ema_version == 0
    second = train_query_epoch(model, MemoryStore(scenes), [0, 1, 2], sampler, optimizer,
                               config(), torch.device("cpu"), 0, first.global_step,
                               epoch_batch_cursor=first.next_epoch_batch_cursor,
                               max_scene_draws=1)
    assert second.next_epoch_batch_cursor == 2
    assert set(sampler.generators) == {0, 1}


@pytest.mark.parametrize("query_count,chunk", [(0, 3), (3, 0), (3, -1)])
def test_invalid_query_count_or_chunk_is_rejected(query_count, chunk):
    cfg = config(query_count=query_count, chunk=chunk)
    with pytest.raises((TypeError, ValueError), match="query|chunk"):
        train_query_epoch(TinyQueryModel(), MemoryStore([make_scene()]), [0], make_sampler(),
                          torch.optim.SGD(TinyQueryModel().parameters(), lr=1e-3), cfg,
                          torch.device("cpu"), 0, 0)


def test_nonfinite_loss_is_rejected_before_optimizer_step():
    scene = make_scene()
    scene.target_cpu[0, 0, 0] = float("nan")
    model = TinyQueryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(FloatingPointError, match="nonfinite"):
        train_query_epoch(model, MemoryStore([scene]), [0], make_sampler(), optimizer,
                          config(query_count=42, chunk=13), torch.device("cpu"), 0, 0)
    assert_nested_equal(model.state_dict(), before)


def test_two_phase_curriculum_events_are_exact_and_reject_bad_init():
    phases = [
        {"name": "field_pretrain", "init_from": "random"},
        {"name": "selected_auxiliary", "init_from": "phase_best"},
    ]
    assert curriculum_phase_events(phases) == [
        "field_pretrain:new_optimizer", "field_pretrain:best",
        "selected_auxiliary:load_phase_best", "selected_auxiliary:new_optimizer",
    ]
    with pytest.raises(ValueError, match="random"):
        curriculum_phase_events([{"name": "a", "init_from": "random"},
                                 {"name": "b", "init_from": "random"}])


def test_first_curriculum_phase_must_be_field_only():
    from scripts.train_ais_mqfno import _phase_config

    cfg = config()
    cfg["loss"]["phase_weight"] = 0.2
    with pytest.raises(ValueError, match="field-only"):
        _phase_config(cfg, {"name": "field_pretrain", "init_from": "random"}, phase_index=0)


def test_resume_keeps_prior_phase_best_metric(tmp_path):
    from scripts.train_ais_mqfno import _existing_phase_best

    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '\n'.join((
            json.dumps({"phase": "field_pretrain", "val_relative_l2": 0.4}),
            json.dumps({"phase": "field_pretrain", "val_relative_l2": 0.05,
                        "best_eligible": False}),
            json.dumps({"phase": "field_pretrain", "val_relative_l2": 0.2,
                        "best_eligible": True}),
            json.dumps({"phase": "selected_auxiliary", "val_relative_l2": 0.1}),
        )) + '\n'
    )
    assert _existing_phase_best(metrics, "field_pretrain", "relative_l2") == 0.2


def test_full_validation_is_sparse_but_forced_at_phase_and_smoke_boundaries():
    from scripts.train_ais_mqfno import _validation_schedule

    assert _validation_schedule(1, 3000, 1, None) == (None, False)
    assert _validation_schedule(99, 3000, 99, None) == (None, False)
    assert _validation_schedule(100, 3000, 100, None) == ("cadence", True)
    assert _validation_schedule(3000, 3000, 3000, None) == ("phase_end", True)
    assert _validation_schedule(1, 3000, 1, 1) == ("budget_stop", False)
    with pytest.raises(ValueError, match="exceeds"):
        _validation_schedule(3001, 3000, 3001, None)


def test_formal_common_stage_config_resolves_model_sampler_and_updates():
    from scripts.train_ais_mqfno import (
        _model_kwargs,
        _phase_updates,
        _sampler_kwargs,
        _validate_configuration,
    )

    cfg = config(64, 64)
    cfg["model"] = {
        "name": "ais_mqfno", "global_in_features": 6, "native_in_channels": 5,
        "global_max_size": 100, "spatial_width": 4, "spatial_modes": 2,
        "spatial_layers": 1, "temporal_modes": 2, "local_dim": 2,
        "fusion_dim": 2, "local_patch_size": 3, "activation_checkpointing": False,
    }
    cfg["sampling"].update({"max_time_steps": 160, "learning_rate": 2e-4})
    cfg["sampler"] = {
        "mixture": {"uniform": 0.3, "interface": 0.2, "source_wavefront": 0.2,
                    "residual": 0.1, "edge": 0.1, "receiver": 0.1},
        "tile_grid": [16, 16],
        "ema_momentum": 0.9, "lag_epochs": 1, "residual_power": 1.0,
        "min_probability": 1e-12,
    }
    cfg["train"]["phases"] = [{"name": "selected_64", "optimizer_updates": 3,
                                  "init_from": "external_checkpoint"}]
    _validate_configuration(cfg)
    assert _phase_updates(cfg["train"]["phases"][0]) == 3
    assert _model_kwargs(cfg)["halo_size"] == 3
    assert "name" not in _model_kwargs(cfg)
    assert _sampler_kwargs(cfg)["tile_size"] == 4
    from fno_acoustic.model_ais_mqfno import AISMQFNO

    AISMQFNO(**_model_kwargs(cfg))
    AdaptiveSpatialSampler(64, 64, seed=29, **_sampler_kwargs(cfg))


def test_named_sampler_mixture_rejects_missing_or_extra_components():
    from scripts.train_ais_mqfno import _sampler_kwargs

    cfg = config(64, 64)
    base = {"uniform": 0.3, "interface": 0.2, "source_wavefront": 0.2,
            "residual": 0.1, "edge": 0.1, "receiver": 0.1}
    cfg["sampler"] = {"mixture": base, "tile_grid": [16, 16]}
    assert _sampler_kwargs(cfg)["mixture"] == [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
    for invalid in ({key: value for key, value in base.items() if key != "edge"},
                    {**base, "unknown": 0.0}):
        cfg["sampler"]["mixture"] = invalid
        with pytest.raises(ValueError, match="mixture"):
            _sampler_kwargs(cfg)


def test_phase_update_alias_conflict_is_rejected():
    from scripts.train_ais_mqfno import _phase_updates

    assert _phase_updates({"optimizer_updates": 3, "updates": 3}) == 3
    with pytest.raises(ValueError, match="updates"):
        _phase_updates({"optimizer_updates": 3, "updates": 2})


@pytest.mark.parametrize(
    ("stage", "query_chunk_size"),
    [(64, 256), (128, 512), (400, 2048)],
)
def test_stage_sampling_query_chunk_size_overrides_common_train(stage, query_chunk_size):
    from scripts.train_ais_mqfno import _phase_config

    cfg = config(stage, stage, chunk=256)
    cfg["sampling"]["query_chunk_size"] = query_chunk_size
    phase = {"name": f"selected_{stage}", "optimizer_updates": 1,
             "init_from": "external_checkpoint"}
    assert _phase_config(cfg, phase, phase_index=0)["train"]["query_chunk_size"] == query_chunk_size


def test_late_quarter_weights_change_q4_loss_and_gradient():
    scene = make_scene()
    inputs = build_scene_model_inputs(scene, 4, torch.device("cpu"))
    query = torch.tensor([[[0.5, 0.5]]])
    target = torch.zeros(1, 1, 160)
    probabilities = torch.ones(1, 1)
    low, high = TinyQueryModel(), TinyQueryModel()
    high.load_state_dict(low.state_dict())
    low_loss, _ = query_field_loss_chunked(
        low, inputs, query, target, probabilities, 1, 1, False,
        late_time_weights=torch.ones(160),
    )
    weights = torch.ones(160)
    weights[120:] = 8.0
    high_loss, _ = query_field_loss_chunked(
        high, inputs, query, target, probabilities, 1, 1, False,
        late_time_weights=weights,
    )
    low_loss.backward()
    high_loss.backward()
    assert not torch.allclose(low_loss, high_loss)
    assert not torch.allclose(low.weight.grad, high.weight.grad)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_grad_clip_must_be_positive_finite_real(value):
    cfg = config()
    cfg["train"]["grad_clip"] = value
    model = TinyQueryModel()
    with pytest.raises((TypeError, ValueError), match="grad_clip"):
        train_query_epoch(model, MemoryStore([make_scene()]), [0], make_sampler(),
                          torch.optim.SGD(model.parameters(), lr=1e-3), cfg,
                          torch.device("cpu"), 0, 0)


def test_nonfinite_clipped_gradient_norm_never_steps_optimizer(monkeypatch):
    class CountingSGD(torch.optim.SGD):
        steps = 0

        def step(self, closure=None):
            self.steps += 1
            return super().step(closure)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args, **kwargs: torch.tensor(float("inf")))
    model = TinyQueryModel()
    optimizer = CountingSGD(model.parameters(), lr=1e-3)
    with pytest.raises(FloatingPointError, match="gradient norm"):
        train_query_epoch(model, MemoryStore([make_scene()]), [0], make_sampler(), optimizer,
                          config(), torch.device("cpu"), 0, 0)
    assert optimizer.steps == 0


def test_receiver_requests_must_be_finite_and_inside_domain():
    scene = make_scene()
    for bad in (
        {"x_start_m": -1.0, "x_stop_m": 10.0, "x_stride_m": 1.0, "z_m": [0.0]},
        {"x_start_m": 0.0, "x_stop_m": 10.0, "x_stride_m": float("nan"), "z_m": [0.0]},
        {"x_start_m": 10.0, "x_stop_m": 0.0, "x_stride_m": 1.0, "z_m": [0.0]},
        {"x_start_m": 0.0, "x_stop_m": 10.0, "x_stride_m": 1.0, "z_m": [float("nan")]},
        {"x_start_m": True, "x_stop_m": 10.0, "x_stride_m": 1.0, "z_m": [0.0]},
    ):
        with pytest.raises((TypeError, ValueError), match="receiver|finite|domain|start"):
            receiver_site_indices(scene, bad)


def test_receiver_stride_never_generates_a_point_past_stop():
    scene = make_scene(height=11, width=6)
    scene = QueryScene(
        scene.sample_id, scene.target_cpu, scene.velocity_cpu, scene.source_cpu,
        scene.time_s, torch.linspace(0.0, 10.0, 11, dtype=torch.float64), scene.z_m,
        scene.metadata,
    )
    indices = receiver_site_indices(
        scene, {"x_start_m": 0.0, "x_stop_m": 10.0, "x_stride_m": 6.0, "z_m": [0.0]}
    )
    assert torch.equal(indices, torch.tensor([0, 36]))


def test_receiver_geometry_alias_is_rejected():
    with pytest.raises(ValueError, match="aliases"):
        receiver_site_indices(make_scene(height=3, width=3), {
            "x_start_m": 0.0, "x_stop_m": 10.0, "x_stride_m": 5.0, "z_m": [0.0]
        })


@pytest.mark.parametrize(
    "centers",
    [
        torch.tensor([-1]),
        torch.tensor([99]),
        torch.tensor([7.0]),
        torch.tensor([7.5]),
        torch.tensor([True]),
    ],
)
def test_centered_site_offsets_rejects_invalid_center_types_and_bounds(centers):
    with pytest.raises((TypeError, ValueError), match="center|integer|bounds"):
        centered_site_offsets(centers, 3, (5, 5))


def test_v2_checkpoint_stores_independent_data_epoch_and_phase_state(tmp_path):
    model = TinyQueryModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    sampler = make_sampler()
    generator = torch.Generator().manual_seed(9)
    path = tmp_path / "v2.pt"
    save_query_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
        epoch=3, epoch_batch_cursor=1, epoch_scene_order=[0, 1], data_generator=generator,
        global_step=5, config_sha256="a" * 64, split_manifest_sha256="b" * 64,
        checkpoint_boundary=QUERY_CHECKPOINT_BOUNDARY, phase_index=1, phase_update=2,
    )
    payload = torch.load(path, weights_only=True)
    assert payload["schema_version"] == 2
    assert len(payload) == 18
    assert payload["epoch"] == 3
    assert payload["phase_index"] == 1
    assert payload["phase_update"] == 2
    restored_model = TinyQueryModel()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=2)
    restored = load_query_checkpoint(
        path, model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler, sampler=make_sampler(), data_generator=torch.Generator().manual_seed(1),
        expected_config_sha256="a" * 64, expected_split_manifest_sha256="b" * 64,
    )
    assert (restored.epoch, restored.phase_index, restored.phase_update) == (3, 1, 2)


def test_screen_checkpoint_atomically_binds_candidate_gate_and_parent(tmp_path):
    model = TinyQueryModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "screen.pt"
    save_query_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        sampler=make_sampler(),
        epoch=0,
        epoch_batch_cursor=0,
        epoch_scene_order=[0],
        data_generator=torch.Generator().manual_seed(9),
        global_step=400,
        config_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        checkpoint_boundary=QUERY_CHECKPOINT_BOUNDARY,
        phase_index=0,
        phase_update=400,
        runtime_seed=2026,
        normalization_stats_sha256="c" * 64,
        normalization_contract="ais_normalization_v2",
        screen_candidate_id="N0",
        screen_gate="O",
        parent_checkpoint_sha256=None,
        dataset_binding_sha256="d" * 64,
        execution_binding_sha256="e" * 64,
    )

    payload = torch.load(path, weights_only=True)
    assert payload["schema_version"] == 5
    assert payload["screen_candidate_id"] == "N0"
    assert payload["screen_gate"] == "O"
    assert payload["parent_checkpoint_sha256"] is None


def write_train_only_normalization(tmp_path):
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps({
        "computed_from_split": "train",
        "velocity": {"mean": 3000.0, "std": 500.0},
        "wavefield": {"mean": 0.0, "std": 1.0},
        "eps": 1.0e-6,
    }))
    return path


def test_cpu_cli_smoke_writes_full160_checkpoint_metrics_and_hashes(tmp_path):
    h5_path = tmp_path / "tiny.h5"
    with h5py.File(h5_path, "w") as h5:
        h5["tensor"] = np.stack([make_scene(i, 5, 5).target_cpu.permute(2, 0, 1).numpy() for i in range(2)])
        h5["nu"] = np.stack([make_scene(i, 5, 5).velocity_cpu.numpy() for i in range(2)])
        h5["source_mask"] = np.stack([make_scene(i, 5, 5).source_cpu.numpy() for i in range(2)])
        h5["t-coordinate"] = np.linspace(0, 1, 160)
        h5["x-coordinate"] = np.linspace(0, 0.004, 5)
        h5["y-coordinate"] = np.linspace(0, 0.004, 5)
        h5["model_type"] = np.array([b"uniform", b"layered"])
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train": [0], "val": [1], "test": []}))
    normalization = write_train_only_normalization(tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""
seed: 5
data: {{path: {h5_path}, split_manifest: {manifest}}}
normalization: {{contract: ais_normalization_v2, stats_path: {normalization}}}
sampling: {{target_height: 5, target_width: 5, global_size: 3, mixture: [0.3, 0.2, 0.2, 0.1, 0.1, 0.1], tile_size: 2}}
model: {{global_in_features: 6, native_in_channels: 5, spatial_width: 2, spatial_modes: 1, temporal_modes: 2, local_dim: 2, fusion_dim: 2, halo_size: 3, spatial_layers: 1}}
receiver: {{x_start_m: 0, x_stop_m: 4, x_stride_m: 2, z_m: [0]}}
train: {{query_sites_per_scene: 3, query_chunk_size: 2, patch_size: 3, patch_centers_per_scene: 1, grad_clip: 1.0, amp: false, learning_rate: 0.001, weight_decay: 0.0, phases: [{{name: field_pretrain, updates: 2, init_from: random}}]}}
loss: {{hh_reweight: true, receiver_weight: 0.0, phase_weight: 0.0, local_spectrum_weight: 0.0, energy_weight: 0.0}}
""")
    output = tmp_path / "run"
    process = subprocess.run([
        sys.executable, "scripts/train_ais_mqfno.py", "--config", str(cfg),
        "--output-dir", str(output), "--device", "cpu", "--max-train-batches", "1",
        "--max-val-batches", "1", "--seed", "20260716",
    ], cwd=Path(__file__).parents[1], text=True, capture_output=True, timeout=120)
    assert process.returncode == 0, process.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["time_steps"] == 160
    assert summary["query_unit"] == "spatial_site_full_trace"
    assert len(summary["config_sha256"]) == len(summary["split_manifest_sha256"]) == 64
    assert summary["global_step"] == 1
    assert summary["status"] == "partial"
    assert (output / "last.pt").exists()
    assert (output / "best.pt").exists()
    assert (output / "checkpoints/last.pt").exists()
    assert (output / "checkpoints/best.pt").exists()
    assert json.loads((output / "checkpoints/runtime_seed.json").read_text()) == {
        "runtime_seed": 20260716
    }
    from scripts.train_ais_mqfno import _config_sha256
    assert summary["config_sha256"] == _config_sha256(yaml.safe_load(cfg.read_text()))
    assert (output / "metrics.jsonl").exists()

    external_cfg = tmp_path / "external.yaml"
    external_cfg.write_text(cfg.read_text().replace("init_from: random", "init_from: external_checkpoint"))
    smoke_output = tmp_path / "external-smoke"
    smoke = subprocess.run([
        sys.executable, "scripts/train_ais_mqfno.py", "--config", str(external_cfg),
        "--output-dir", str(smoke_output), "--device", "cpu",
        "--max-train-batches", "1", "--max-val-batches", "1",
    ], cwd=Path(__file__).parents[1], text=True, capture_output=True, timeout=120)
    assert smoke.returncode == 0, smoke.stderr
    assert "smoke_random_init_without_external_checkpoint" in json.loads(
        (smoke_output / "summary.json").read_text()
    )["events"]
    formal = subprocess.run([
        sys.executable, "scripts/train_ais_mqfno.py", "--config", str(external_cfg),
        "--output-dir", str(tmp_path / "external-formal"), "--device", "cpu",
    ], cwd=Path(__file__).parents[1], text=True, capture_output=True, timeout=120)
    assert formal.returncode != 0
    assert "requires --init-checkpoint" in formal.stderr


def test_two_phase_cli_preserves_data_cursor_across_phase_boundary(tmp_path):
    h5_path = tmp_path / "three.h5"
    scenes = [make_scene(i, 5, 5) for i in range(3)]
    with h5py.File(h5_path, "w") as h5:
        h5["tensor"] = np.stack([scene.target_cpu.permute(2, 0, 1).numpy() for scene in scenes])
        h5["nu"] = np.stack([scene.velocity_cpu.numpy() for scene in scenes])
        h5["source_mask"] = np.stack([scene.source_cpu.numpy() for scene in scenes])
        h5["t-coordinate"] = np.linspace(0, 1, 160)
        h5["x-coordinate"] = np.linspace(0, 0.004, 5)
        h5["y-coordinate"] = np.linspace(0, 0.004, 5)
        h5["model_type"] = np.array([b"uniform", b"layered", b"marmousi"])
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train": [0, 1], "val": [2], "test": []}))
    normalization = write_train_only_normalization(tmp_path)
    cfg = tmp_path / "two_phase.yaml"
    cfg.write_text(f"""
seed: 17
data: {{path: {h5_path}, split_manifest: {manifest}}}
normalization: {{contract: ais_normalization_v2, stats_path: {normalization}}}
sampling: {{target_height: 5, target_width: 5, global_size: 3, mixture: [0.3, 0.2, 0.2, 0.1, 0.1, 0.1], tile_size: 2}}
model: {{global_in_features: 6, native_in_channels: 5, spatial_width: 2, spatial_modes: 1, temporal_modes: 2, local_dim: 2, fusion_dim: 2, halo_size: 3, spatial_layers: 1}}
receiver: {{x_start_m: 0, x_stop_m: 4, x_stride_m: 2, z_m: [0]}}
train: {{query_sites_per_scene: 3, query_chunk_size: 2, patch_size: 3, patch_centers_per_scene: 1, grad_clip: 1.0, amp: false, learning_rate: 0.001, weight_decay: 0.0, phases: [{{name: field_pretrain, updates: 1, init_from: random}}, {{name: selected_auxiliary, updates: 1, init_from: phase_best}}]}}
loss: {{hh_reweight: true, receiver_weight: 0.0, phase_weight: 0.0, local_spectrum_weight: 0.0, energy_weight: 0.0}}
""")
    output = tmp_path / "run"
    process = subprocess.run([
        sys.executable, "scripts/train_ais_mqfno.py", "--config", str(cfg),
        "--output-dir", str(output), "--device", "cpu", "--max-val-batches", "1",
        "--max-train-batches", "1",
    ], cwd=Path(__file__).parents[1], text=True, capture_output=True, timeout=120)
    assert process.returncode == 0, process.stderr
    interrupted = torch.load(output / "last.pt", weights_only=True)
    assert interrupted["epoch"] == 0
    assert interrupted["epoch_batch_cursor"] == 1
    assert interrupted["phase_index"] == 0
    assert interrupted["phase_update"] == 1
    process = subprocess.run([
        sys.executable, "scripts/train_ais_mqfno.py", "--config", str(cfg),
        "--output-dir", str(output), "--device", "cpu", "--max-val-batches", "1",
        "--max-train-batches", "1", "--resume", str(output / "last.pt"),
    ], cwd=Path(__file__).parents[1], text=True, capture_output=True, timeout=120)
    assert process.returncode == 0, process.stderr
    payload = torch.load(output / "last.pt", weights_only=True)
    assert payload["epoch"] == 1
    assert payload["epoch_batch_cursor"] == 0
    assert payload["phase_index"] == 1
    assert payload["phase_update"] == 1
    sampler_state = payload["sampler_state_dict"]
    assert set(sampler_state["generator_states"]) == {0, 1}
    reference = AdaptiveSpatialSampler(
        5, 5, [0.3, 0.2, 0.2, 0.1, 0.1, 0.1], 17, tile_size=2
    )
    feature = SpatialSamplingFeatures(*(torch.ones(25) for _ in range(5)))
    reference.draw([0], [feature], 3)
    reference.draw([1], [feature], 3)
    for sample_id in (0, 1):
        assert torch.equal(
            sampler_state["generator_states"][sample_id],
            reference.generator_for(sample_id).get_state(),
        )
    assert sampler_state["ema_version"] == 1
    assert sampler_state["pending_updates"] == []


def test_non_cadence_stop_resume_matches_uninterrupted_across_phase(tmp_path):
    h5_path = tmp_path / "three.h5"
    scenes = [make_scene(i, 5, 5) for i in range(3)]
    with h5py.File(h5_path, "w") as h5:
        h5["tensor"] = np.stack(
            [scene.target_cpu.permute(2, 0, 1).numpy() for scene in scenes]
        )
        h5["nu"] = np.stack([scene.velocity_cpu.numpy() for scene in scenes])
        h5["source_mask"] = np.stack([scene.source_cpu.numpy() for scene in scenes])
        h5["t-coordinate"] = np.linspace(0, 1, 160)
        h5["x-coordinate"] = np.linspace(0, 0.004, 5)
        h5["y-coordinate"] = np.linspace(0, 0.004, 5)
        h5["model_type"] = np.array([b"uniform", b"layered", b"marmousi"])
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train": [0, 1], "val": [2], "test": []}))
    normalization = write_train_only_normalization(tmp_path)
    cfg = tmp_path / "two_phase.yaml"
    cfg.write_text(f"""
seed: 17
data: {{path: {h5_path}, split_manifest: {manifest}}}
normalization: {{contract: ais_normalization_v2, stats_path: {normalization}}}
sampling: {{target_height: 5, target_width: 5, global_size: 3, mixture: [0.3, 0.2, 0.2, 0.1, 0.1, 0.1], tile_size: 2}}
model: {{global_in_features: 6, native_in_channels: 5, spatial_width: 2, spatial_modes: 1, temporal_modes: 2, local_dim: 2, fusion_dim: 2, halo_size: 3, spatial_layers: 1}}
receiver: {{x_start_m: 0, x_stop_m: 4, x_stride_m: 2, z_m: [0]}}
train: {{query_sites_per_scene: 3, query_chunk_size: 2, patch_size: 3, patch_centers_per_scene: 1, grad_clip: 1.0, amp: false, learning_rate: 0.001, weight_decay: 0.0, phases: [{{name: field_pretrain, updates: 2, init_from: random}}, {{name: selected_auxiliary, updates: 1, init_from: phase_best}}]}}
loss: {{hh_reweight: true, receiver_weight: 0.0, phase_weight: 0.0, local_spectrum_weight: 0.0, energy_weight: 0.0}}
""")
    root = Path(__file__).parents[1]
    common = [
        sys.executable,
        "scripts/train_ais_mqfno.py",
        "--config",
        str(cfg),
        "--device",
        "cpu",
        "--max-val-batches",
        "1",
    ]
    uninterrupted = tmp_path / "uninterrupted"
    process = subprocess.run(
        [*common, "--output-dir", str(uninterrupted)],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr

    segmented = tmp_path / "segmented"
    process = subprocess.run(
        [*common, "--output-dir", str(segmented), "--max-train-batches", "1"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    first_rows = [json.loads(line) for line in (segmented / "metrics.jsonl").read_text().splitlines()]
    assert first_rows[-1]["validation_reason"] == "budget_stop"
    assert first_rows[-1]["best_eligible"] is False
    process = subprocess.run(
        [
            *common,
            "--output-dir",
            str(segmented),
            "--resume",
            str(segmented / "checkpoints/last.pt"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr

    expected = torch.load(uninterrupted / "checkpoints/last.pt", weights_only=True)
    actual = torch.load(segmented / "checkpoints/last.pt", weights_only=True)
    assert actual["global_step"] == expected["global_step"] == 3
    assert actual["phase_index"] == expected["phase_index"] == 1
    for name, tensor in expected["model_state_dict"].items():
        assert torch.equal(actual["model_state_dict"][name], tensor), name


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right
