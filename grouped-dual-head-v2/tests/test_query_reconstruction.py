from __future__ import annotations

import pickle
import weakref

import numpy as np
import pytest
import torch

from fno_acoustic.model_ais_mqfno import AISMQFNO
from fno_acoustic.query_reconstruction import (
    QueryInferenceScene,
    _resolve_scene_device,
    iter_spatial_query_chunks,
    reconstruct_native,
)


class DeterministicQueryModel:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.query_counts: list[int] = []
        self.query_devices: list[torch.device] = []

    def encode_global(
        self, global_inputs: torch.Tensor, time_s: torch.Tensor
    ) -> torch.Tensor:
        self.encode_calls += 1
        return global_inputs[:, :1, :1, :1, :1]

    def decode_queries(
        self,
        context: torch.Tensor,
        native_static: torch.Tensor,
        query_xz: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        del context
        self.query_counts.append(query_xz.shape[1])
        self.query_devices.append(query_xz.device)
        time = time_s if time_s.ndim == 1 else time_s[0]
        batch_offset = native_static[:, 0, 0, 0, None, None]
        space = query_xz[..., 0, None] + 2.0 * query_xz[..., 1, None]
        return space + time[None, None, :] + batch_offset


class LifetimeRecordingModel(DeterministicQueryModel):
    def __init__(self) -> None:
        super().__init__()
        self.previous_output: weakref.ReferenceType[torch.Tensor] | None = None

    def decode_queries(
        self,
        context: torch.Tensor,
        native_static: torch.Tensor,
        query_xz: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        if self.previous_output is not None:
            assert self.previous_output() is None, "previous device chunk is still live"
        values = super().decode_queries(
            context, native_static, query_xz, time_s
        )
        self.previous_output = weakref.ref(values)
        return values


class FailingSecondChunkModel(DeterministicQueryModel):
    def decode_queries(
        self,
        context: torch.Tensor,
        native_static: torch.Tensor,
        query_xz: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        if self.query_counts:
            raise RuntimeError("injected decode failure")
        return super().decode_queries(
            context, native_static, query_xz, time_s
        )


class NonfiniteQueryModel(DeterministicQueryModel):
    def decode_queries(
        self,
        context: torch.Tensor,
        native_static: torch.Tensor,
        query_xz: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        values = super().decode_queries(
            context, native_static, query_xz, time_s
        )
        values[0, 0, 0] = torch.nan
        return values


class SpacingAwareQueryModel(DeterministicQueryModel):
    supports_physical_spacing = True

    def decode_queries(
        self,
        context: torch.Tensor,
        native_static: torch.Tensor,
        query_xz: torch.Tensor,
        time_s: torch.Tensor,
        *,
        dx_m: float | None = None,
        dz_m: float | None = None,
    ) -> torch.Tensor:
        self.spacings = getattr(self, "spacings", []) + [(dx_m, dz_m)]
        return super().decode_queries(context, native_static, query_xz, time_s)


def make_scene(
    height: int = 8,
    width: int = 6,
    batch: int = 1,
    device: str | torch.device = "cpu",
) -> QueryInferenceScene:
    device = torch.device(device)
    global_inputs = torch.zeros(batch, 2, 3, 160, 1, device=device)
    native_static = torch.zeros(batch, 1, height, width, device=device)
    native_static[:, 0, 0, 0] = torch.arange(batch, device=device)
    time_s = torch.linspace(0.0, 1.0, 160, device=device)
    return QueryInferenceScene(
        global_inputs=global_inputs,
        native_static=native_static,
        time_s=time_s,
        height=height,
        width=width,
        batch=batch,
        device=device,
    )


def test_spatial_chunks_cover_every_site_exactly_once_in_row_major_order():
    chunks = list(iter_spatial_query_chunks(7, 9, chunk_sites=16))
    indices = torch.cat([chunk.site_indices for chunk in chunks])

    assert torch.equal(indices, torch.arange(63))
    assert torch.bincount(indices, minlength=63).eq(1).all()
    assert torch.equal(chunks[0].query_xz[0], torch.tensor([0.0, 0.0]))
    assert torch.equal(chunks[-1].query_xz[-1], torch.tensor([1.0, 1.0]))


def test_tiled_reconstruction_matches_monolithic_on_small_grid():
    scene = make_scene()
    tiled = reconstruct_native(DeterministicQueryModel(), scene, chunk_sites=7)
    full = reconstruct_native(DeterministicQueryModel(), scene, chunk_sites=48)

    assert isinstance(tiled.field, torch.Tensor)
    assert tiled.field.shape == (1, 8, 6, 160)
    assert tiled.field.dtype == torch.float32
    assert tiled.field.device.type == "cpu"
    assert torch.allclose(tiled.field, full.field, atol=1e-6, rtol=1e-6)
    assert tiled.coverage.device.type == "cpu"
    assert tiled.coverage.min() == tiled.coverage.max() == 1


def test_reconstruction_passes_spacing_only_to_capable_models() -> None:
    scene = make_scene()
    scene = QueryInferenceScene(
        scene.global_inputs,
        scene.native_static,
        scene.time_s,
        scene.height,
        scene.width,
        scene.batch,
        scene.device,
        dx_m=30.0,
        dz_m=40.0,
    )
    capable = SpacingAwareQueryModel()

    reconstruct_native(capable, scene, chunk_sites=7)
    reconstruct_native(DeterministicQueryModel(), scene, chunk_sites=7)

    assert capable.spacings
    assert set(capable.spacings) == {(30.0, 40.0)}


def test_real_ais_mqfno_reconstructs_full160_tiled_like_monolithic():
    torch.manual_seed(17)
    model = AISMQFNO(
        global_in_features=1,
        native_in_channels=1,
        spatial_width=2,
        spatial_modes=1,
        temporal_modes=2,
        local_dim=2,
        fusion_dim=2,
        halo_size=1,
        spatial_layers=1,
    ).eval()
    scene = QueryInferenceScene(
        global_inputs=torch.randn(1, 2, 2, 160, 1),
        native_static=torch.randn(1, 1, 3, 2),
        time_s=torch.linspace(0.0, 1.0, 160),
        height=3,
        width=2,
        batch=1,
        device=torch.device("cpu"),
    )

    tiled = reconstruct_native(model, scene, chunk_sites=2)
    monolithic = reconstruct_native(model, scene, chunk_sites=6)

    assert isinstance(tiled.field, torch.Tensor)
    assert tiled.field.shape == (1, 3, 2, 160)
    assert tiled.field.device.type == "cpu"
    assert tiled.field.dtype == torch.float32
    assert torch.isfinite(tiled.field).all()
    assert tiled.coverage.eq(1).all()
    assert torch.allclose(tiled.field, monolithic.field, atol=1e-6, rtol=1e-6)


def test_reconstruction_reuses_scene_time_validation_across_chunks(monkeypatch) -> None:
    import fno_acoustic.model_ais_mqfno as model_module

    model = AISMQFNO(
        global_in_features=1,
        native_in_channels=1,
        spatial_width=2,
        spatial_modes=1,
        temporal_modes=1,
        local_dim=2,
        fusion_dim=2,
        halo_size=1,
        spatial_layers=1,
    ).eval()
    scene = QueryInferenceScene(
        global_inputs=torch.zeros(1, 2, 2, 160, 1),
        native_static=torch.zeros(1, 1, 3, 2),
        time_s=torch.linspace(0.0, 1.0, 160),
        height=3,
        width=2,
        batch=1,
        device=torch.device("cpu"),
        dx_m=30.0,
        dz_m=40.0,
    )
    calls = 0
    original = model_module._shared_full160_time

    def counted(time_s: torch.Tensor, batch: int) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(time_s, batch)

    monkeypatch.setattr(model_module, "_shared_full160_time", counted)
    reconstruct_native(model, scene, chunk_sites=2)

    assert calls == 0


def test_reconstruction_skips_temporal_revalidation_per_scene_and_chunk(
    monkeypatch,
) -> None:
    import fno_acoustic.temporal_operator as temporal_module

    model = AISMQFNO(
        global_in_features=1,
        native_in_channels=1,
        spatial_width=2,
        spatial_modes=1,
        temporal_modes=1,
        local_dim=2,
        fusion_dim=2,
        halo_size=1,
        spatial_layers=1,
    ).eval()

    def scene_with_end(end: float) -> QueryInferenceScene:
        return QueryInferenceScene(
            global_inputs=torch.zeros(1, 2, 2, 160, 1),
            native_static=torch.zeros(1, 1, 3, 2),
            time_s=torch.linspace(0.0, end, 160),
            height=3,
            width=2,
            batch=1,
            device=torch.device("cpu"),
            dx_m=30.0,
            dz_m=40.0,
        )

    first = scene_with_end(1.0)
    second = scene_with_end(2.0)
    calls = 0
    original = temporal_module._validate_time_vector

    def counted(time_s: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        original(time_s)

    monkeypatch.setattr(temporal_module, "_validate_time_vector", counted)
    reconstruct_native(model, first, chunk_sites=2)
    reconstruct_native(model, second, chunk_sites=3)

    assert calls == 0
    assert not torch.equal(
        first.validated_time_grid.time_s, second.validated_time_grid.time_s
    )


def test_scene_owns_time_snapshot_and_ignores_later_caller_mutation() -> None:
    time_s = torch.linspace(0.0, 1.0, 160)
    expected = time_s.clone()
    scene = QueryInferenceScene(
        global_inputs=torch.zeros(1, 2, 2, 160, 1),
        native_static=torch.zeros(1, 1, 2, 2),
        time_s=time_s,
        height=2,
        width=2,
        batch=1,
        device=torch.device("cpu"),
    )

    time_s[80] = time_s[79]
    result = reconstruct_native(DeterministicQueryModel(), scene, chunk_sites=2)

    assert torch.equal(scene.time_s, expected)
    assert scene.time_s.data_ptr() != time_s.data_ptr()
    assert torch.isfinite(result.field).all()


def test_reconstruction_rejects_mutated_scene_time_capability() -> None:
    model = AISMQFNO(
        global_in_features=1,
        native_in_channels=1,
        spatial_width=2,
        spatial_modes=1,
        temporal_modes=1,
        local_dim=2,
        fusion_dim=2,
        halo_size=1,
        spatial_layers=1,
    ).eval()
    scene = QueryInferenceScene(
        global_inputs=torch.zeros(1, 2, 2, 160, 1),
        native_static=torch.zeros(1, 1, 2, 2),
        time_s=torch.linspace(0.0, 1.0, 160),
        height=2,
        width=2,
        batch=1,
        device=torch.device("cpu"),
    )
    scene.validated_time_grid._time_s.add_(1.0)

    with pytest.raises(ValueError, match="mutated"):
        reconstruct_native(model, scene, chunk_sites=2)


def test_pickled_scene_restores_validated_time_and_reconstructs() -> None:
    scene = make_scene(height=2, width=3)
    model = AISMQFNO(
        global_in_features=scene.global_inputs.shape[-1],
        native_in_channels=scene.native_static.shape[1],
        spatial_width=2,
        spatial_modes=1,
        temporal_modes=1,
        local_dim=2,
        fusion_dim=2,
        halo_size=1,
        spatial_layers=1,
    ).eval()

    restored = pickle.loads(pickle.dumps(scene))
    restored.validated_time_grid.assert_current()
    result = reconstruct_native(model, restored, chunk_sites=2)

    assert torch.equal(restored.time_s, scene.time_s)
    assert torch.isfinite(result.field).all()


def test_reconstruction_bounds_each_device_chunk_and_encodes_once():
    model = DeterministicQueryModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = make_scene(device=device)

    result = reconstruct_native(model, scene, chunk_sites=11)

    assert model.encode_calls == 1
    assert max(model.query_counts) <= 11
    assert all(device == scene.device for device in model.query_devices)
    assert isinstance(result.field, torch.Tensor)
    assert result.field.device.type == "cpu"


def test_reconstruction_releases_previous_device_output_before_next_decode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LifetimeRecordingModel()

    reconstruct_native(model, make_scene(device=device), chunk_sites=11)

    assert len(model.query_counts) > 1


def test_scene_device_resolution_rejects_inputs_on_different_gpu_indices():
    with pytest.raises(ValueError, match="same device"):
        _resolve_scene_device(
            torch.device("cuda"),
            (
                torch.device("cuda:0"),
                torch.device("cuda:1"),
                torch.device("cuda:0"),
            ),
        )


def test_scene_device_resolution_resolves_unindexed_cuda_to_actual_device():
    assert _resolve_scene_device(
        torch.device("cuda"),
        (torch.device("cuda:1"),) * 3,
    ) == torch.device("cuda:1")
    with pytest.raises(ValueError, match="scene.device"):
        _resolve_scene_device(
            torch.device("cuda:0"),
            (torch.device("cuda:1"),) * 3,
        )


def test_scene_rejects_nonshared_batched_time_coordinates():
    scene = make_scene(height=3, width=4, batch=2)
    batched_time = scene.time_s.expand(2, -1).clone()
    batched_time[1] += 0.01

    with pytest.raises(ValueError):
        QueryInferenceScene(
            global_inputs=scene.global_inputs,
            native_static=scene.native_static,
            time_s=batched_time,
            height=scene.height,
            width=scene.width,
            batch=scene.batch,
            device=scene.device,
        )


def test_reconstruction_supports_batch_greater_than_one():
    scene = make_scene(height=3, width=4, batch=3)
    result = reconstruct_native(DeterministicQueryModel(), scene, chunk_sites=5)

    assert result.field.shape == (3, 3, 4, 160)
    assert torch.allclose(result.field[1] - result.field[0], torch.ones_like(result.field[0]))
    assert torch.allclose(result.field[2] - result.field[1], torch.ones_like(result.field[0]))


def test_reconstruction_writes_user_cpu_tensor():
    scene = make_scene(height=3, width=5)
    output = torch.full((1, 3, 5, 160), torch.nan, dtype=torch.float32)

    result = reconstruct_native(
        DeterministicQueryModel(), scene, chunk_sites=4, output=output
    )

    assert result.field is output
    assert torch.isfinite(output).all()


def test_reconstruction_writes_float32_memmap(tmp_path):
    scene = make_scene(height=3, width=5)
    path = tmp_path / "native.dat"
    output = np.memmap(path, mode="w+", dtype=np.float32, shape=(1, 3, 5, 160))

    result = reconstruct_native(
        DeterministicQueryModel(), scene, chunk_sites=4, output=output
    )

    assert result.field is output
    output.flush()
    reopened = np.memmap(path, mode="r", dtype=np.float32, shape=output.shape)
    assert np.isfinite(reopened).all()
    assert np.allclose(reopened[0, -1, -1], np.linspace(0, 1, 160) + 3.0)


def test_failed_reconstruction_invalidates_entire_external_tensor():
    scene = make_scene(height=3, width=5)
    output = torch.full((1, 3, 5, 160), 7.0, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="injected decode failure"):
        reconstruct_native(
            FailingSecondChunkModel(), scene, chunk_sites=4, output=output
        )

    assert torch.isnan(output).all()


def test_failed_reconstruction_invalidates_and_flushes_entire_memmap(tmp_path):
    scene = make_scene(height=3, width=5)
    path = tmp_path / "failed-native.dat"
    shape = (1, 3, 5, 160)
    output = np.memmap(path, mode="w+", dtype=np.float32, shape=shape)
    output[:] = 7.0
    output.flush()

    with pytest.raises(RuntimeError, match="injected decode failure"):
        reconstruct_native(
            FailingSecondChunkModel(), scene, chunk_sites=4, output=output
        )

    reopened = np.memmap(path, mode="r", dtype=np.float32, shape=shape)
    assert np.isnan(reopened).all()


def test_nonfinite_decode_output_fails_and_invalidates_destination():
    scene = make_scene(height=3, width=5)
    output = torch.full((1, 3, 5, 160), 7.0, dtype=torch.float32)

    with pytest.raises(ValueError, match="finite"):
        reconstruct_native(
            NonfiniteQueryModel(), scene, chunk_sites=4, output=output
        )

    assert torch.isnan(output).all()


@pytest.mark.parametrize(
    ("height", "width", "chunk_sites"),
    [(0, 4, 2), (3, 0, 2), (3, 4, 0), (True, 4, 2), (3.5, 4, 2)],
)
def test_spatial_chunk_iterator_rejects_nonpositive_or_noninteger_sizes(
    height, width, chunk_sites
):
    with pytest.raises(ValueError):
        list(iter_spatial_query_chunks(height, width, chunk_sites))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values.update(batch=0),
        lambda values: values.update(height=0),
        lambda values: values.update(width=True),
        lambda values: values.update(global_inputs=values["global_inputs"][:, :, :, :-1]),
        lambda values: values.update(global_inputs=values["global_inputs"][..., :0]),
        lambda values: values.update(native_static=values["native_static"][:, :, :-1]),
        lambda values: values.update(time_s=values["time_s"][:-1]),
        lambda values: values.update(batch=2),
        lambda values: values.update(device=torch.device("meta")),
    ],
)
def test_scene_rejects_invalid_dimensions_shapes_batch_or_device(mutate):
    scene = make_scene(height=3, width=4)
    values = {
        "global_inputs": scene.global_inputs,
        "native_static": scene.native_static,
        "time_s": scene.time_s,
        "height": scene.height,
        "width": scene.width,
        "batch": scene.batch,
        "device": scene.device,
    }
    mutate(values)

    with pytest.raises(ValueError):
        QueryInferenceScene(**values)


@pytest.mark.parametrize(
    "output",
    [
        torch.empty(1, 3, 4, 159),
        torch.empty(1, 3, 4, 160, dtype=torch.float64),
        np.empty((1, 3, 4, 160), dtype=np.float32),
    ],
)
def test_reconstruction_rejects_invalid_output_shape_dtype_or_type(output):
    with pytest.raises((TypeError, ValueError)):
        reconstruct_native(
            DeterministicQueryModel(), make_scene(height=3, width=4), 5, output
        )


def test_native_400_chunk_iterator_covers_all_sites_without_model_inference():
    chunks = list(iter_spatial_query_chunks(400, 400, chunk_sites=2048))
    indices = torch.cat([chunk.site_indices for chunk in chunks])

    assert indices.numel() == 160_000
    assert torch.equal(indices, torch.arange(160_000))
    assert max(chunk.site_indices.numel() for chunk in chunks) <= 2048
