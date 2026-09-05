"""Contract tests for the full-160 AIS-MQFNO query model."""

from __future__ import annotations

import pytest
import torch

from fno_acoustic.model_ais_mqfno import (
    AISMQFNO,
    DilatedTemporalResidual,
    LocalQueryEncoder,
)


def full160_time(batch: int | None = None) -> torch.Tensor:
    early = torch.linspace(0.0, 0.3, 97, dtype=torch.float64)[:-1]
    late = torch.linspace(0.3, 1.2, 64, dtype=torch.float64)
    time_s = torch.cat((early, late))
    return time_s if batch is None else time_s.expand(batch, -1).clone()


def tiny_model(**overrides: object) -> AISMQFNO:
    kwargs = dict(
        global_in_features=2,
        native_in_channels=3,
        spatial_width=4,
        spatial_modes=2,
        spatial_layers=1,
        temporal_modes=4,
        local_dim=3,
        fusion_dim=5,
        halo_size=17,
    )
    kwargs.update(overrides)
    return AISMQFNO(**kwargs)


def tiny_inputs(batch: int = 2, queries: int = 3):
    global_inputs = torch.randn(batch, 6, 6, 160, 2)
    native_static = torch.randn(batch, 3, 11, 13)
    query_xz = torch.rand(batch, queries, 2)
    return global_inputs, native_static, query_xz, full160_time(batch)


def test_model_outputs_complete_160_trace_per_native_site() -> None:
    output = tiny_model()(*tiny_inputs(batch=2, queries=5))

    assert output.shape == (2, 5, 160)


def test_different_scenes_may_use_different_query_coordinates() -> None:
    inputs = list(tiny_inputs(batch=2, queries=2))
    inputs[2] = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0]], [[0.2, 0.8], [0.7, 0.3]]]
    )

    assert tiny_model()(*inputs).shape == (2, 2, 160)


def test_model_backward_is_finite_and_uses_bounded_native_halo() -> None:
    model = tiny_model()
    inputs = list(tiny_inputs(batch=1, queries=2))
    inputs[0].requires_grad_()
    inputs[1].requires_grad_()

    model(*inputs).square().mean().backward()

    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert inputs[0].grad is not None and torch.isfinite(inputs[0].grad).all()
    assert inputs[1].grad is not None and torch.isfinite(inputs[1].grad).all()
    assert model.last_native_query_shape == (17, 17)


def test_spatial_execution_options_are_forwarded_to_encoder(monkeypatch) -> None:
    model = tiny_model(spatial_chunk_size=40, activation_checkpointing=True)
    calls: list[tuple[int, bool]] = []
    original = model.spatial_encoder.forward

    def spy(x, spatial_chunk_size=0, use_checkpointing=False):
        calls.append((spatial_chunk_size, use_checkpointing))
        return original(x, spatial_chunk_size, use_checkpointing)

    monkeypatch.setattr(model.spatial_encoder, "forward", spy)
    inputs = list(tiny_inputs(batch=1, queries=1))
    inputs[0].requires_grad_()

    model(*inputs).square().mean().backward()

    assert calls == [(40, True)]
    assert inputs[0].grad is not None and torch.isfinite(inputs[0].grad).all()


def test_complete_model_supports_double_forward_and_backward() -> None:
    model = tiny_model(spatial_chunk_size=80).double()
    global_inputs, native_static, query_xz, time_s = tiny_inputs(batch=1, queries=1)
    global_inputs = global_inputs.double().requires_grad_()
    native_static = native_static.double().requires_grad_()

    output = model(global_inputs, native_static, query_xz.double(), time_s)
    output.square().mean().backward()

    assert output.dtype == torch.float64
    assert global_inputs.grad is not None and torch.isfinite(global_inputs.grad).all()
    assert native_static.grad is not None and torch.isfinite(native_static.grad).all()


def test_encode_once_then_decode_matches_forward_in_eval_mode() -> None:
    model = tiny_model().eval()
    global_inputs, native_static, query_xz, time_s = tiny_inputs(batch=1, queries=3)

    with torch.no_grad():
        direct = model(global_inputs, native_static, query_xz, time_s)
        context = model.encode_global(global_inputs, time_s)
        reused = model.decode_queries(context, native_static, query_xz, time_s)

    assert torch.equal(direct, reused)


def test_sample_global_context_maps_physical_xz_to_vertical_horizontal_grid() -> None:
    # Value 10*x + z makes swapping x/z observable.
    x = torch.linspace(0.0, 1.0, 3)
    z = torch.linspace(0.0, 1.0, 4)
    field = (10 * x[:, None] + z[None, :])[None, :, :, None, None].expand(
        1, 3, 4, 160, 1
    )
    queries = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.5, 1.0 / 3.0]]])

    sampled = AISMQFNO.sample_global_context(field, queries)

    assert sampled.shape == (1, 3, 160, 1)
    assert torch.allclose(sampled[0, :, 0, 0], torch.tensor([1.0, 10.0, 5.0 + 1 / 3]))


def test_float64_coordinates_are_safely_sampled_from_float32_features() -> None:
    global_inputs, native_static, query_xz, time_s = tiny_inputs(batch=1, queries=2)

    output = tiny_model()(global_inputs, native_static, query_xz.double(), time_s)

    assert output.dtype == torch.float32


def test_local_encoder_extracts_per_scene_patches_and_can_respond_after_unfreezing() -> None:
    encoder = LocalQueryEncoder(in_channels=1, local_dim=2, halo_size=3)
    native = torch.stack((torch.zeros(1, 5, 5), torch.ones(1, 5, 5)))
    query = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]])

    patches = encoder.extract_patches(native, query)
    assert patches.shape == (2, 1, 1, 3, 3)
    assert not torch.equal(patches[0], patches[1])
    assert torch.equal(encoder(native, query)[0], encoder(native, query)[1])

    with torch.no_grad():
        encoder.encoder[-1].weight.fill_(1.0)
    assert not torch.equal(encoder(native, query)[0], encoder(native, query)[1])


def test_local_encoder_uses_explicit_border_policy_at_domain_edges() -> None:
    encoder = LocalQueryEncoder(in_channels=1, local_dim=2, halo_size=3)
    native = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)

    patches = encoder.extract_patches(native, torch.tensor([[[0.0, 0.0]]]))

    assert patches[0, 0, 0, 0, 0] == native[0, 0, 0, 0]


@pytest.mark.parametrize("halo_size", [0, 2, -3])
def test_local_encoder_rejects_nonpositive_or_even_halo(halo_size: int) -> None:
    with pytest.raises(ValueError, match="positive odd"):
        LocalQueryEncoder(1, 2, halo_size)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda values: values.__setitem__(2, torch.rand(2, 3)), r"\[B,Q,2\]"),
        (lambda values: values.__setitem__(2, torch.full((2, 3, 2), 1.01)), r"\[0,1\]"),
        (lambda values: values.__setitem__(1, torch.randn(1, 3, 11, 13)), "batch"),
        (lambda values: values.__setitem__(1, torch.randn(2, 4, 11, 13)), "channels"),
        (lambda values: values.__setitem__(0, torch.randn(1, 6, 6, 160, 2)), "batch"),
        (lambda values: values.__setitem__(0, torch.randn(2, 6, 6, 160, 3)), "features"),
    ],
)
def test_forward_rejects_bad_shapes_batches_channels_and_coordinates(mutate, message) -> None:
    values = list(tiny_inputs())
    mutate(values)

    with pytest.raises(ValueError, match=message):
        tiny_model()(*values)


def test_forward_requires_shared_strictly_increasing_full160_time() -> None:
    values = list(tiny_inputs())
    inconsistent = values[3].clone()
    inconsistent[1, 80] += 1e-4
    values[3] = inconsistent
    with pytest.raises(ValueError, match="shared"):
        tiny_model()(*values)

    values = list(tiny_inputs())
    values[3][0, 80] = values[3][0, 79]
    values[3][1] = values[3][0]
    with pytest.raises(ValueError, match="strictly increasing"):
        tiny_model()(*values)

    values = list(tiny_inputs())
    values[3] = values[3][:, :-1]
    with pytest.raises(ValueError, match="160"):
        tiny_model()(*values)


def test_dilated_temporal_residual_is_zero_at_initialization() -> None:
    layer = DilatedTemporalResidual(channels=3)
    x = torch.randn(1, 2, 2, 160, 3)

    assert torch.equal(layer(x), torch.zeros_like(x))


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("global_in_features", True, "global_in_features"),
        ("native_in_channels", 2.5, "native_in_channels"),
        ("spatial_width", 0, "spatial_width"),
        ("spatial_modes", 1.5, "spatial_modes"),
        ("temporal_modes", -1, "temporal_modes"),
        ("local_dim", False, "local_dim"),
        ("fusion_dim", 0, "fusion_dim"),
        ("halo_size", 17.0, "halo_size"),
        ("spatial_layers", 0, "spatial_layers"),
        ("spatial_chunk_size", -1, "spatial_chunk_size"),
        ("spatial_chunk_size", 2.5, "spatial_chunk_size"),
        ("activation_checkpointing", 1, "activation_checkpointing"),
    ],
)
def test_constructor_rejects_noninteger_bool_and_nonpositive_hyperparameters(
    name: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tiny_model(**{name: value})
