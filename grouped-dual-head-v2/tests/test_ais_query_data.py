from __future__ import annotations

from pathlib import Path
import os

import h5py
import numpy as np
import pytest
import torch

import fno_acoustic.query_data as query_data_module
from fno_acoustic.ais_dataset_binding import (
    canonical_json_bytes,
    compute_dataset_manifest,
    dataset_content_root,
    merkle_root,
    verified_ids_sha256,
)

from fno_acoustic.query_data import (
    DenseCPUQueryStore,
    Full160GridContract,
    collate_query_sites,
    resize_query_scene,
)


@pytest.fixture()
def full160_h5(tmp_path: Path) -> Path:
    path = tmp_path / "full160.h5"
    n, nt, nx, nz = 2, 160, 4, 5
    increments = np.resize(np.asarray([0.0028, 0.0032], dtype=np.float64), nt - 1)
    time_s = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(increments)))
    x_km = np.linspace(0.25, 1.75, nx, dtype=np.float64)
    z_km = np.linspace(0.1, 1.1, nz, dtype=np.float64)
    tensor = np.arange(n * nt * nx * nz, dtype=np.float32).reshape(n, nt, nx, nz)
    velocity = np.arange(n * nx * nz, dtype=np.float32).reshape(n, nx, nz) + 1500.0
    source = np.zeros((n, nx, nz), dtype=np.float32)
    source[:, 1, 2] = 1.0

    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=tensor)
        h5.create_dataset("nu", data=velocity)
        h5.create_dataset("source_mask", data=source)
        h5.create_dataset("t-coordinate", data=time_s)
        h5.create_dataset("x-coordinate", data=x_km)
        h5.create_dataset("y-coordinate", data=z_km)
        h5.create_dataset("model_type", data=np.asarray([b"uniform", b"layered"]))
    return path


def test_full160_contract_preserves_nonuniform_physical_time(full160_h5: Path):
    scene = DenseCPUQueryStore(full160_h5).read_scene(0)

    assert scene.target_cpu.shape == (4, 5, 160)
    assert scene.target_cpu.dtype == torch.float32
    assert scene.target_cpu.device.type == "cpu"
    assert scene.velocity_cpu.dtype == torch.float32
    assert scene.source_cpu.dtype == torch.float32
    assert scene.time_s.dtype == torch.float64
    assert scene.x_m.dtype == torch.float64
    assert scene.z_m.dtype == torch.float64
    assert scene.time_s.numel() == 160
    assert set(torch.diff(scene.time_s).mul(10_000).round().int().tolist()) == {28, 32}
    assert torch.equal(scene.x_m, torch.linspace(250.0, 1750.0, 4, dtype=torch.float64))
    assert torch.equal(scene.z_m, torch.linspace(100.0, 1100.0, 5, dtype=torch.float64))
    assert scene.metadata == {"model_type": "uniform"}

    with pytest.raises(ValueError, match="exactly 160"):
        Full160GridContract(4, 5, scene.time_s[:-1], scene.x_m, scene.z_m).validate()
    with pytest.raises(ValueError, match="strictly increasing"):
        Full160GridContract(4, 5, scene.time_s.flip(0), scene.x_m, scene.z_m).validate()
    with pytest.raises(ValueError, match="coordinate lengths"):
        Full160GridContract(4, 5, scene.time_s, scene.x_m[:-1], scene.z_m).validate()


def _binding_manifest(dataset: Path, path: Path) -> Path:
    path.write_bytes(canonical_json_bytes(compute_dataset_manifest(dataset)))
    return path


def test_bound_store_verifies_leaf_on_every_read_and_reports_process_unique_ids(
    full160_h5: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _binding_manifest(full160_h5, tmp_path / "manifest.json")
    original = query_data_module.sample_digest
    calls = 0

    def counted(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(query_data_module, "sample_digest", counted)
    store = DenseCPUQueryStore(full160_h5, dataset_content_manifest=manifest)

    store.read_scene(0)
    store.read_scene(0)

    assert calls == 2
    assert store.binding_summary() == {
        "dataset_content_root": compute_dataset_manifest(full160_h5)[
            "dataset_content_root"
        ],
        "verified_sample_count": 1,
        "verified_sample_ids_sha256": verified_ids_sha256([0]),
        "verified_sample_scope": "process_unique",
    }


def test_bound_store_rejects_leaf_mutation_after_manifest_mtime_is_restored(
    full160_h5: Path, tmp_path: Path,
) -> None:
    manifest = _binding_manifest(full160_h5, tmp_path / "manifest.json")
    store = DenseCPUQueryStore(full160_h5, dataset_content_manifest=manifest)
    store.read_scene(0)
    original_stat = full160_h5.stat()
    with h5py.File(full160_h5, "r+") as h5:
        h5["tensor"][0, 0, 0, 0] += 1.0
    os.utime(full160_h5, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(ValueError, match="sample leaf"):
        store.read_scene(0)


def test_bound_store_rejects_global_mutation_after_manifest_mtime_is_restored(
    full160_h5: Path, tmp_path: Path,
) -> None:
    manifest = _binding_manifest(full160_h5, tmp_path / "manifest.json")
    store = DenseCPUQueryStore(full160_h5, dataset_content_manifest=manifest)
    original_stat = full160_h5.stat()
    with h5py.File(full160_h5, "r+") as h5:
        h5["x-coordinate"][0] += 0.01
    os.utime(full160_h5, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(ValueError, match="global"):
        store.read_scene(0)


def test_bound_store_post_stat_failure_does_not_record_verified_id(
    full160_h5: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _binding_manifest(full160_h5, tmp_path / "manifest.json")
    store = DenseCPUQueryStore(full160_h5, dataset_content_manifest=manifest)
    calls = 0
    original = store._assert_bound_stat

    def fail_post_stat(stat_result: os.stat_result) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic post stat failure")
        original(stat_result)

    monkeypatch.setattr(store, "_assert_bound_stat", fail_post_stat)
    with pytest.raises(ValueError, match="post stat"):
        store.read_scene(0)

    assert store.binding_summary()["verified_sample_count"] == 0


def test_bound_store_rejects_self_consistent_wrong_leaf_before_torch_conversion(
    full160_h5: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = compute_dataset_manifest(full160_h5)
    payload["sample_digests"][0] = "0" * 64
    payload["sample_merkle_root"] = merkle_root(payload["sample_digests"])
    payload["dataset_content_root"] = dataset_content_root(
        payload["global_sha256"], payload["sample_merkle_root"], payload["sample_count"]
    )
    manifest = tmp_path / "forged.json"
    manifest.write_bytes(canonical_json_bytes(payload))
    store = DenseCPUQueryStore(full160_h5, dataset_content_manifest=manifest)
    monkeypatch.setattr(
        torch,
        "from_numpy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Torch conversion happened before leaf verification")
        ),
    )

    with pytest.raises(ValueError, match="sample leaf"):
        store.read_scene(0)


def test_bound_store_rejects_stat_drift_before_read(full160_h5: Path, tmp_path: Path) -> None:
    manifest = _binding_manifest(full160_h5, tmp_path / "manifest.json")
    store = DenseCPUQueryStore(full160_h5, dataset_content_manifest=manifest)
    current = full160_h5.stat()
    os.utime(
        full160_h5,
        ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
    )

    with pytest.raises(ValueError, match="stat"):
        store.read_scene(0)


def test_read_sites_returns_complete_trace_per_spatial_site(full160_h5: Path):
    query = DenseCPUQueryStore(full160_h5).read_sites(0, torch.tensor([0, 7, 19]))

    assert query.target_cpu.shape == (3, 160)
    assert query.physical_xz.shape == (3, 2)
    assert query.target_cpu.device.type == "cpu"
    assert torch.equal(query.physical_xz[1], torch.tensor([750.0, 600.0], dtype=torch.float64))


def test_query_collate_never_materializes_dense_gpu_target(full160_h5: Path):
    store = DenseCPUQueryStore(full160_h5)
    batch = collate_query_sites(
        [
            store.read_sites(0, torch.tensor([0, 3])),
            store.read_sites(1, torch.tensor([1, 4])),
        ]
    )

    assert batch.target_cpu.shape == (2, 2, 160)
    assert not hasattr(batch, "dense_target")


def test_spatial_curriculum_changes_only_space_and_keeps_all_times(full160_h5: Path):
    native = DenseCPUQueryStore(full160_h5).read_scene(0)
    stage64 = resize_query_scene(native, 64, 64)
    stage128 = resize_query_scene(native, 128, 128)

    assert stage64.target_cpu.shape == (64, 64, 160)
    assert stage128.target_cpu.shape == (128, 128, 160)
    assert torch.equal(stage64.time_s, native.time_s)
    assert torch.equal(stage128.time_s, native.time_s)
    assert stage64.x_m[[0, -1]].tolist() == native.x_m[[0, -1]].tolist()
    assert stage64.z_m[[0, -1]].tolist() == native.z_m[[0, -1]].tolist()
    assert torch.allclose(stage64.source_cpu.sum(), native.source_cpu.sum())


def test_gather_loaded_scene_uses_resized_stage_grid(full160_h5: Path):
    store = DenseCPUQueryStore(full160_h5)
    stage = resize_query_scene(store.read_scene(0), 8, 6)
    indices = torch.tensor([0, 17, 47])
    sites = store.gather_loaded_scene(stage, indices)

    assert sites.target_cpu.shape == (3, 160)
    assert torch.equal(sites.target_cpu, stage.target_cpu.reshape(48, 160)[indices])
    with pytest.raises(ValueError, match="nonempty"):
        store.gather_loaded_scene(stage, torch.tensor([], dtype=torch.long))
    with pytest.raises(ValueError, match="inside"):
        store.gather_loaded_scene(stage, torch.tensor([48]))


def test_read_scene_rejects_159_frame_and_wrong_spatial_field_shapes(full160_h5: Path):
    with h5py.File(full160_h5, "r+") as h5:
        tensor = h5["tensor"][:]
        del h5["tensor"]
        h5.create_dataset("tensor", data=tensor[:, :-1])
    with pytest.raises(ValueError, match=r"target must have shape \[H, W, 160\]"):
        DenseCPUQueryStore(full160_h5).read_scene(0)

    with h5py.File(full160_h5, "r+") as h5:
        del h5["tensor"]
        h5.create_dataset("tensor", data=tensor)
        velocity = h5["nu"][:, :, :-1]
        del h5["nu"]
        h5.create_dataset("nu", data=velocity)
    with pytest.raises(ValueError, match=r"velocity and source must have shape \[H, W\]"):
        DenseCPUQueryStore(full160_h5).read_scene(0)


@pytest.mark.parametrize(
    ("coordinate_key", "replacement", "message"),
    [
        ("x-coordinate", np.asarray([0.25, 0.75, 0.75, 1.75]), "strictly increasing"),
        ("y-coordinate", np.asarray([0.1, 0.35, np.nan, 0.85, 1.1]), "must be finite"),
        (
            "t-coordinate",
            np.concatenate((np.asarray([0.0, np.inf]), np.arange(158, dtype=np.float64))),
            "must be finite",
        ),
    ],
)
def test_read_scene_rejects_invalid_physical_coordinates(
    full160_h5: Path,
    coordinate_key: str,
    replacement: np.ndarray,
    message: str,
):
    with h5py.File(full160_h5, "r+") as h5:
        del h5[coordinate_key]
        h5.create_dataset(coordinate_key, data=replacement)

    with pytest.raises(ValueError, match=message):
        DenseCPUQueryStore(full160_h5).read_scene(0)


def test_gather_requires_one_dimensional_integer_indices_and_owns_a_clone(full160_h5: Path):
    store = DenseCPUQueryStore(full160_h5)
    scene = store.read_scene(0)

    with pytest.raises(ValueError, match="one-dimensional integer tensor"):
        store.gather_loaded_scene(scene, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="one-dimensional integer tensor"):
        store.gather_loaded_scene(scene, torch.tensor([[0, 1]]))

    caller_indices = torch.tensor([0, 7, 19])
    sites = store.gather_loaded_scene(scene, caller_indices)
    caller_indices[0] = 4

    assert torch.equal(sites.site_indices, torch.tensor([0, 7, 19]))
    assert torch.equal(sites.target_cpu, scene.target_cpu.reshape(20, 160)[sites.site_indices])
