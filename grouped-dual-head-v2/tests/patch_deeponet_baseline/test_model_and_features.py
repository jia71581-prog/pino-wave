from __future__ import annotations

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

import torch

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer, ScaleMetadata
from patch_deeponet_baseline.features import (
    build_query_descriptors,
    build_static_features,
)
from patch_deeponet_baseline.model import (
    PatchDeepONet,
    PatchDeepONetConfig,
    resolve_parameter_matched_width,
)
from patch_deeponet_baseline.training import (
    dense_training_pair,
    patch_deeponet_loss,
    pi_deeponet_lwc84_residual_loss,
)
from scripts.predict_patch_deeponet_marmousi_position_control import (
    CHECKPOINT_SCHEMA,
    IDENTITY_SCHEMA,
    _dense_prediction,
    _load_baseline,
)
from scripts.train_patch_deeponet_baseline import _model_from_config


def _inputs(*, records: int = 2, height: int = 17, width: int = 19, queries: int = 23):
    velocity = 1800.0 + 200.0 * torch.rand(records, 1, height, width)
    velocity_n = (velocity - 1900.0) / 200.0
    source_map = torch.zeros(records, 1, height, width)
    source_map[:, 0, 3, 5] = 1.0
    travel = torch.rand(records, height, width) * 0.8
    static = build_static_features(
        velocity_n, source_map, travel, velocity, domain_t_s=1.0
    )
    physical = torch.tensor(
        [[500.0, 100.0, 19.0, 1.5 / 19.0, 1.0]] * records
    )
    normalized = physical / torch.tensor([2000.0, 2000.0, 50.0, 1.0, 1.0])
    query = torch.rand(records, queries, 3)
    query[..., 0] *= 2000.0
    query[..., 1] *= 2000.0
    query_travel = torch.rand(records, queries) * 0.8
    descriptors = build_query_descriptors(
        query,
        normalized,
        physical,
        query_travel,
        domain_x_m=2000.0,
        domain_z_m=2000.0,
        domain_t_s=1.0,
    )
    return static, descriptors


def _tiny_config() -> PatchDeepONetConfig:
    return PatchDeepONetConfig(
        branch_width=16,
        branch_blocks=2,
        branch_global_width=48,
        branch_bottleneck_width=8,
        latent_dim=32,
        trunk_width=48,
        trunk_depth=4,
        fourier_bands=4,
        local_residual_width=16,
        target_parameters=100_000,
        parameter_tolerance_fraction=0.01,
    )


def test_features_are_target_free_and_have_registered_shapes() -> None:
    static, descriptors = _inputs()
    assert static.shape == (2, 4, 17, 19)
    assert descriptors.shape == (2, 23, 10)
    assert torch.isfinite(static).all()
    assert torch.isfinite(descriptors).all()


def test_query_chunking_is_functionally_identical() -> None:
    torch.manual_seed(3)
    model = PatchDeepONet(_tiny_config()).eval()
    static, descriptors = _inputs()
    whole = model(static, descriptors)
    chunked = model(static, descriptors, query_chunk=5)
    torch.testing.assert_close(chunked, whole, atol=2.0e-6, rtol=2.0e-6)


def test_all_learned_paths_receive_gradient() -> None:
    torch.manual_seed(4)
    model = PatchDeepONet(_tiny_config()).train()
    static, descriptors = _inputs(records=1, queries=11)
    model(static, descriptors).square().mean().backward()
    groups = {
        "branch": tuple(model.static_stem.parameters())
        + tuple(model.static_blocks.parameters())
        + tuple(model.branch_bottleneck.parameters())
        + tuple(model.branch_global.parameters())
        + tuple(model.branch_projection.parameters()),
        "trunk": tuple(model.trunk_input.parameters())
        + tuple(model.trunk_blocks.parameters())
        + tuple(model.trunk_output.parameters()),
        "local": tuple(model.local_residual.parameters()),
        "ricker": (model.ricker_bias,),
    }
    for parameters in groups.values():
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in parameters
        )


def test_default_model_matches_current_r5b_capacity() -> None:
    model = PatchDeepONet()
    report = model.parameter_match()
    assert report["target_parameters"] == 32_420_564
    assert report["parameter_count"] == 32_420_465
    assert report["within_tolerance"] is True
    assert report["relative_difference"] < 0.000004


def test_downsampling_depth_changes_resolution_without_changing_capacity() -> None:
    torch.manual_seed(9)
    config = _tiny_config()
    downsampled = PatchDeepONet(config).eval()
    full_resolution = PatchDeepONet(replace(config, branch_downsample_stages=0)).eval()
    static, _ = _inputs(records=1, height=17, width=19)
    assert downsampled.encode_static(static).shape[-2:] == (2, 2)
    assert full_resolution.encode_static(static).shape[-2:] == (17, 19)
    assert downsampled.parameter_count() == full_resolution.parameter_count()


def test_training_config_builds_registered_spatial_resolution() -> None:
    model = _model_from_config({"model": {"branch_downsample_stages": 0}})
    assert model.config.branch_downsample_stages == 0
    assert model.parameter_match()["within_tolerance"] is True


def test_spatial_fourier_override_is_finite_and_parameter_matched() -> None:
    config = replace(
        _tiny_config(),
        branch_downsample_stages=0,
        fourier_coordinate_indices=(0, 1, 2, 9),
    )
    model = PatchDeepONet(config).eval()
    static, descriptors = _inputs(records=1)
    prediction = model(static, descriptors)
    assert torch.isfinite(prediction).all()
    default = PatchDeepONet(_tiny_config())
    assert model.parameter_count() - default.parameter_count() == 2 * 2 * 4 * 48


def test_bounded_width_search_finds_same_parameter_match() -> None:
    start = replace(PatchDeepONetConfig(), trunk_width=52)
    resolved = resolve_parameter_matched_width(start, search_radius=8)
    assert resolved.trunk_width == 48
    assert PatchDeepONet(resolved).parameter_match()["within_tolerance"] is True


def _write_checkpoint_pair(tmp_path, *, selection_split: str = "train"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _tiny_config()
    model = PatchDeepONet(config)
    identity = {
        "schema": IDENTITY_SCHEMA,
        "manifest_digest": "locked-manifest",
        "selection_split": selection_split,
        "model_config": asdict(config),
        "parameter_count": model.parameter_count(),
        "run_digest": "locked-run",
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "manifest_digest": "locked-manifest",
        "selection_split": selection_split,
        "run_digest": "locked-run",
        "epoch": 2,
        "global_step": 17,
        "model_state": model.state_dict(),
    }
    identity_path = tmp_path / "run_identity.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    identity_path.write_text(json.dumps(identity), encoding="utf8")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, identity_path


def test_checkpoint_loader_enforces_train_only_manifest_binding(tmp_path) -> None:
    checkpoint_path, identity_path = _write_checkpoint_pair(tmp_path)
    model, identity, checkpoint = _load_baseline(
        checkpoint_path,
        identity_path,
        expected_manifest_digest="locked-manifest",
        device=torch.device("cpu"),
    )
    assert model.training is False
    assert identity["selection_split"] == checkpoint["selection_split"] == "train"

    bad_checkpoint_path, bad_identity_path = _write_checkpoint_pair(
        tmp_path / "bad", selection_split="validation"
    )
    with pytest.raises(ValueError, match="train only"):
        _load_baseline(
            bad_checkpoint_path,
            bad_identity_path,
            expected_manifest_digest="locked-manifest",
            device=torch.device("cpu"),
        )


def test_small_dense_prediction_is_finite_and_honours_surface() -> None:
    torch.manual_seed(8)
    model = PatchDeepONet(_tiny_config()).eval()
    normalizer = PhysicalNormalizer(
        ScaleMetadata(
            velocity_center_mps=1900.0,
            velocity_scale_mps=200.0,
            pressure_scale_pa=3.0,
            source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
            train_manifest_sha256="locked-manifest",
            allowed_medium_types=ALLOWED_MEDIUM_TYPES,
            record_count=1,
            algorithm="test",
        )
    )
    x_m = np.linspace(0.0, 2000.0, 11, dtype=np.float32)
    z_m = np.linspace(0.0, 2000.0, 9, dtype=np.float32)
    time_s = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    velocity = np.full((len(z_m), len(x_m)), 1900.0, dtype=np.float32)
    source_map = np.zeros_like(velocity)
    source_map[1, 3] = 1.0
    prediction = _dense_prediction(
        model,
        normalizer,
        velocity,
        [600.0, 250.0, 19.0, 1.5 / 19.0, 1.0],
        source_map,
        time_s,
        x_m,
        z_m,
        device=torch.device("cpu"),
        time_block=2,
        query_chunk=37,
    )
    assert prediction.shape == (5, 9, 11)
    assert np.isfinite(prediction).all()
    assert np.array_equal(prediction[:, 0], np.zeros((5, 11), dtype=np.float32))


def test_dense_training_pair_and_composite_loss_are_finite() -> None:
    torch.manual_seed(9)
    model = PatchDeepONet(_tiny_config()).eval()
    normalizer = PhysicalNormalizer(
        ScaleMetadata(
            velocity_center_mps=1900.0,
            velocity_scale_mps=200.0,
            pressure_scale_pa=3.0,
            source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
            train_manifest_sha256="locked-manifest",
            allowed_medium_types=ALLOWED_MEDIUM_TYPES,
            record_count=1,
            algorithm="test",
        )
    )
    height, width, frames = 9, 11, 5
    batch = SimpleNamespace(
        sample_id=("train_uniform_test",),
        group_id=("train:uniform:test",),
        medium_type=("uniform",),
        dense_travel_time_s=torch.linspace(0.0, 0.8, height * width).reshape(1, height, width),
        velocity_mps=torch.full((1, 1, height, width), 1900.0),
        record_to_medium=torch.zeros(1, dtype=torch.long),
        source_parameters=torch.tensor([[600.0, 250.0, 19.0, 1.5 / 19.0, 1.0]]),
        source_map=torch.nn.functional.one_hot(
            torch.tensor([1 * width + 3]), num_classes=height * width
        ).float().reshape(1, 1, height, width),
        requested_time_s=torch.linspace(0.0, 1.0, frames)[None],
        dense_target_physical=torch.randn(1, frames, height, width),
        left_index=torch.arange(frames)[None],
        x_m=torch.linspace(0.0, 2000.0, width),
        z_m=torch.linspace(0.0, 2000.0, height),
    )
    prediction, target, indices = dense_training_pair(
        model,
        batch,
        normalizer,
        torch.device("cpu"),
        time_block=2,
        query_chunk=37,
    )
    loss = patch_deeponet_loss(prediction, target, indices)
    assert prediction.shape == target.shape == (1, frames, height, width)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_dense_training_pair_supports_physical_record_microbatches() -> None:
    torch.manual_seed(11)
    model = PatchDeepONet(_tiny_config()).eval()
    normalizer = PhysicalNormalizer(
        ScaleMetadata(
            velocity_center_mps=1900.0,
            velocity_scale_mps=200.0,
            pressure_scale_pa=3.0,
            source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
            train_manifest_sha256="locked-manifest",
            allowed_medium_types=ALLOWED_MEDIUM_TYPES,
            record_count=2,
            algorithm="test",
        )
    )
    records, height, width, frames = 2, 9, 11, 5
    source_map = torch.zeros(records, 1, height, width)
    source_map[:, 0, 1, 3] = 1.0
    batch = SimpleNamespace(
        sample_id=("train_uniform_a", "train_layered_b"),
        group_id=("train:uniform:a", "train:layered:b"),
        medium_type=("uniform", "layered"),
        dense_travel_time_s=torch.linspace(0.0, 0.8, records * height * width).reshape(
            records, height, width
        ),
        velocity_mps=torch.full((records, 1, height, width), 1900.0),
        record_to_medium=torch.arange(records),
        source_parameters=torch.tensor(
            [
                [600.0, 250.0, 19.0, 1.5 / 19.0, 1.0],
                [800.0, 350.0, 19.0, 1.5 / 19.0, 1.0],
            ]
        ),
        source_map=source_map,
        requested_time_s=torch.stack(
            (torch.linspace(0.0, 1.0, frames), torch.tensor([0.0, 0.2, 0.55, 0.8, 1.0]))
        ),
        dense_target_physical=torch.randn(records, frames, height, width),
        left_index=torch.arange(frames)[None].expand(records, -1),
        x_m=torch.linspace(0.0, 2000.0, width),
        z_m=torch.linspace(0.0, 2000.0, height),
    )
    prediction, target, indices = dense_training_pair(
        model,
        batch,
        normalizer,
        torch.device("cpu"),
        time_block=2,
        query_chunk=37,
    )
    assert prediction.shape == target.shape == (records, frames, height, width)
    loss = patch_deeponet_loss(prediction, target, indices)
    assert torch.isfinite(loss.total)
    loss.total.backward()


def test_pi_deeponet_residual_uses_only_consecutive_saved_triplets() -> None:
    torch.manual_seed(17)
    height = width = 19
    field = torch.randn(1, 5, height, width, requires_grad=True)
    velocity = torch.full((1, 1, height, width), 1900.0)
    source = torch.tensor([[600.0, 250.0, 19.0, 0.05, 1.0]])
    source_map = torch.zeros(1, 1, height, width)
    source_map[0, 0, 9, 9] = 1.0
    indices = torch.tensor([[0, 1, 2, 10, 11]])
    time_s = indices.float() * 0.0025
    residual = pi_deeponet_lwc84_residual_loss(
        field,
        velocity,
        source,
        source_map,
        time_s,
        indices,
        pressure_scale_pa=3.0,
        dt_s=0.0025,
        dx_m=10.0,
        dz_m=10.0,
        maximum_triplets_per_record=2,
    )
    assert residual.triplet_count == 1
    assert torch.isfinite(residual.loss)
    residual.loss.backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()
