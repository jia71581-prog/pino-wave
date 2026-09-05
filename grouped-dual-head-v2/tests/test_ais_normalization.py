from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fno_acoustic.ais_normalization import (
    AISNormalizationBinding,
    build_normalized_static_features,
    load_ais_normalization,
)
from fno_acoustic.query_data import QueryScene
from fno_acoustic.query_training import build_scene_model_inputs


def make_binding(
    tmp_path: Path,
    *,
    velocity: tuple[float, float],
    wavefield: tuple[float, float],
    eps: float = 1.0e-6,
) -> AISNormalizationBinding:
    path = tmp_path / "normalization_stats.json"
    path.write_text(
        json.dumps(
            {
                "computed_from_split": "train",
                "train_sample_count": 4,
                "velocity": {"mean": velocity[0], "std": velocity[1]},
                "wavefield": {"mean": wavefield[0], "std": wavefield[1]},
                "eps": eps,
            }
        )
    )
    return load_ais_normalization(path)


def valid_stats() -> dict[str, object]:
    return {
        "computed_from_split": "train",
        "train_sample_count": 4,
        "velocity": {"mean": 3000.0, "std": 500.0},
        "wavefield": {"mean": 0.0, "std": 0.2},
        "eps": 1.0e-6,
    }


def set_stats_field(
    stats: dict[str, object], field: tuple[str, ...], value: object
) -> None:
    target = stats
    for name in field[:-1]:
        target = target[name]  # type: ignore[assignment,index]
    target[field[-1]] = value


_INVALID_STATS_CASES = [
    (("computed_from_split",), "val", ValueError),
    *[
        ((section, statistic), value, ValueError)
        for section in ("velocity", "wavefield")
        for statistic in ("mean", "std")
        for value in (float("nan"), float("inf"))
    ],
    *[
        ((section, "std"), value, ValueError)
        for section in ("velocity", "wavefield")
        for value in (0.0, -1.0)
    ],
    (("eps",), 0.0, ValueError),
    (("eps",), -1.0, ValueError),
    *[
        (field, True, TypeError)
        for field in (
            ("velocity", "mean"),
            ("velocity", "std"),
            ("wavefield", "mean"),
            ("wavefield", "std"),
            ("eps",),
        )
    ],
]


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    _INVALID_STATS_CASES,
    ids=[f"{'.'.join(field)}={value!r}" for field, value, _ in _INVALID_STATS_CASES],
)
def test_loader_rejects_invalid_train_statistics(
    tmp_path: Path,
    field: tuple[str, ...],
    value: object,
    error_type: type[Exception],
) -> None:
    stats = valid_stats()
    set_stats_field(stats, field, value)
    path = tmp_path / "invalid_stats.json"
    path.write_text(json.dumps(stats))

    with pytest.raises(error_type):
        load_ais_normalization(path)


def test_wavefield_standardization_round_trips_physical_values(tmp_path: Path) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.1, 0.25)
    )
    physical = torch.tensor([-0.4, 0.1, 0.6])

    encoded = binding.encode_wavefield(physical)

    assert torch.allclose(encoded, torch.tensor([-2.0, 0.0, 2.0]))
    assert torch.allclose(binding.decode_wavefield(encoded), physical)


def test_float32_tiny_stats_stay_finite_and_round_trip(tmp_path: Path) -> None:
    binding = make_binding(
        tmp_path,
        velocity=(3000.0, 1.0e-300),
        wavefield=(0.0, 1.0e-300),
        eps=1.0e-300,
    )
    tiny = torch.finfo(torch.float32).tiny
    physical = torch.tensor([-tiny, 0.0, tiny], dtype=torch.float32)

    encoded = binding.encode_wavefield(physical)
    decoded = binding.decode_wavefield(encoded)

    assert torch.isfinite(encoded).all()
    assert torch.isfinite(decoded).all()
    assert torch.equal(encoded, torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(decoded, physical)


def test_binding_hashes_exact_stats_bytes(tmp_path: Path) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2)
    )

    assert binding.contract_id == "ais_normalization_v2"
    assert binding.stats_sha256 == hashlib.sha256(binding.path.read_bytes()).hexdigest()


def make_scene(height: int = 5, width: int = 7, *, zero_source: bool = False) -> QueryScene:
    x_m = torch.linspace(0.0, 400.0, height, dtype=torch.float64)
    z_m = torch.linspace(0.0, 600.0, width, dtype=torch.float64)
    xi = torch.linspace(-1.0, 1.0, height)[:, None]
    zeta = torch.linspace(-1.0, 1.0, width)[None, :]
    velocity = 3000.0 + 500.0 * (0.4 * xi - 0.25 * zeta)
    source = torch.zeros(height, width) if zero_source else 2.0 * xi + 3.0 * zeta
    return QueryScene(
        sample_id=0,
        target_cpu=torch.zeros(height, width, 160),
        velocity_cpu=velocity,
        source_cpu=source,
        time_s=torch.linspace(0.0, 1.0, 160, dtype=torch.float64),
        x_m=x_m,
        z_m=z_m,
        metadata={},
    )


def expected_normalized_native(scene: QueryScene) -> torch.Tensor:
    height, width = scene.velocity_cpu.shape
    xi = torch.linspace(-1.0, 1.0, height)[:, None]
    zeta = torch.linspace(-1.0, 1.0, width)[None, :]
    velocity_hat = 0.4 * xi - 0.25 * zeta
    source_scale = scene.source_cpu.abs().max().clamp_min(1.0e-6)
    return torch.stack(
        (
            velocity_hat,
            scene.source_cpu / source_scale,
            torch.full((height, width), 0.4),
            torch.full((height, width), -0.25),
            (3000.0 / scene.velocity_cpu).square() - 1.0,
        )
    )[None]


def test_static_features_have_strict_dimensionless_channel_contract(
    tmp_path: Path,
) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2)
    )
    scene = make_scene(height=5, width=7)

    features = build_normalized_static_features(scene, binding)

    assert features.shape == (1, 5, 5, 7)
    expected = expected_normalized_native(scene)
    for channel in range(5):
        assert torch.allclose(features[:, channel], expected[:, channel], atol=1.0e-6)
    assert torch.isfinite(features).all()
    assert float(features.abs().max()) < 20.0


def test_zero_source_normalizes_to_finite_zeros(tmp_path: Path) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2)
    )
    features = build_normalized_static_features(
        make_scene(zero_source=True), binding
    )

    assert torch.equal(features[:, 1], torch.zeros_like(features[:, 1]))
    assert torch.isfinite(features[:, 1]).all()


def test_tiny_eps_keeps_zero_source_finite(tmp_path: Path) -> None:
    binding = make_binding(
        tmp_path,
        velocity=(3000.0, 1.0e-300),
        wavefield=(0.0, 1.0e-300),
        eps=1.0e-300,
    )

    scene = make_scene(zero_source=True)
    scene = replace(
        scene,
        velocity_cpu=torch.full_like(scene.velocity_cpu, binding.velocity_mean),
    )
    features = build_normalized_static_features(scene, binding)

    assert torch.equal(features[:, 1], torch.zeros_like(features[:, 1]))
    assert torch.isfinite(features).all()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("velocity_cpu", float("nan")),
        ("velocity_cpu", float("inf")),
        ("source_cpu", float("nan")),
        ("source_cpu", float("inf")),
    ],
)
def test_static_features_reject_nonfinite_scene_tensors(
    tmp_path: Path, field: str, value: float
) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2)
    )
    scene = make_scene()
    invalid = getattr(scene, field).clone()
    invalid[0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        build_normalized_static_features(scene=replace(scene, **{field: invalid}), binding=binding)


@pytest.mark.parametrize("field", ["velocity_cpu", "source_cpu"])
def test_static_features_reject_integer_scene_tensors(
    tmp_path: Path, field: str
) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2)
    )
    scene = make_scene()
    invalid = getattr(scene, field).to(dtype=torch.int64)

    with pytest.raises(TypeError, match="floating"):
        build_normalized_static_features(scene=replace(scene, **{field: invalid}), binding=binding)


@pytest.mark.parametrize("velocity", [0.0, -1.0])
def test_static_features_reject_nonpositive_velocity(
    tmp_path: Path, velocity: float
) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2)
    )
    scene = make_scene()
    invalid = scene.velocity_cpu.clone()
    invalid[0, 0] = velocity

    with pytest.raises(ValueError, match="positive"):
        build_normalized_static_features(
            replace(scene, velocity_cpu=invalid), binding
        )


def test_static_features_reject_slow_contrast_overflow(tmp_path: Path) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2)
    )
    scene = make_scene()
    velocity = scene.velocity_cpu.clone()
    velocity[0, 0] = torch.finfo(velocity.dtype).tiny

    with pytest.raises(ValueError, match="features.*finite"):
        build_normalized_static_features(
            replace(scene, velocity_cpu=velocity), binding
        )


def test_static_features_reject_mean_outside_scene_dtype(tmp_path: Path) -> None:
    binding = make_binding(
        tmp_path, velocity=(1.0e300, 500.0), wavefield=(0.0, 0.2)
    )

    with pytest.raises(ValueError, match="features.*finite"):
        build_normalized_static_features(make_scene(), binding)


def test_static_features_align_source_to_velocity_dtype(tmp_path: Path) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2)
    )
    scene = make_scene()
    mixed = replace(scene, source_cpu=scene.source_cpu.to(dtype=torch.float64))

    features = build_normalized_static_features(mixed, binding)

    expected_source = mixed.source_cpu.to(dtype=scene.velocity_cpu.dtype)
    expected_source = expected_source / expected_source.abs().max()
    assert features.dtype == scene.velocity_cpu.dtype
    assert features.device == scene.velocity_cpu.device
    assert torch.allclose(features[:, 1], expected_source[None])


def test_scene_model_inputs_preserve_normalized_channel_order(tmp_path: Path) -> None:
    binding = make_binding(
        tmp_path, velocity=(3000.0, 500.0), wavefield=(0.0, 0.2)
    )
    scene = make_scene()

    inputs = build_scene_model_inputs(
        scene, global_size=3, device=torch.device("cpu"), normalization=binding
    )

    expected_native = expected_normalized_native(scene)
    expected_static = F.interpolate(
        expected_native,
        size=(3, 3),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    ).permute(0, 2, 3, 1)

    assert torch.allclose(inputs.native_static, expected_native, atol=1.0e-6)
    assert torch.allclose(
        inputs.global_inputs[..., :5],
        expected_static[:, :, :, None].expand(-1, -1, -1, 160, -1),
        atol=1.0e-6,
    )


def test_scene_model_inputs_none_exactly_preserves_legacy_formulas() -> None:
    scene = make_scene()
    velocity = scene.velocity_cpu[None, None]
    source = scene.source_cpu[None, None]
    grad_x, grad_z = torch.gradient(
        velocity,
        spacing=(scene.x_m.float(), scene.z_m.float()),
        dim=(2, 3),
        edge_order=2,
    )
    expected_native = torch.cat(
        (
            velocity,
            source,
            grad_x,
            grad_z,
            velocity.clamp_min(1.0).reciprocal().square(),
        ),
        dim=1,
    )
    expected_static = F.interpolate(
        expected_native,
        size=(3, 3),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    ).permute(0, 2, 3, 1)
    tau = ((scene.time_s - scene.time_s[0]) / (scene.time_s[-1] - scene.time_s[0])).float()
    expected_global = torch.cat(
        (
            expected_static[:, :, :, None].expand(-1, -1, -1, 160, -1),
            tau[None, None, None, :, None].expand(1, 3, 3, -1, -1),
        ),
        dim=-1,
    )

    inputs = build_scene_model_inputs(
        scene, global_size=3, device=torch.device("cpu"), normalization=None
    )

    assert torch.equal(inputs.native_static, expected_native)
    assert torch.equal(inputs.global_inputs, expected_global)
    assert torch.equal(inputs.time_s, scene.time_s)
