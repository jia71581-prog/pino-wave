from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from grouped_ufno_mionet_v3.data.index import build_manifest
from grouped_ufno_mionet_v3.data.curriculum import (
    CurriculumMicrobatchSpec,
    CurriculumStepDataset,
    CurriculumStepSpec,
    make_curriculum_loader,
)
from grouped_ufno_mionet_v3.data.pilot import (
    PilotBatchDataset,
    build_pilot_schedule,
    make_pilot_loader,
)


@pytest.fixture()
def grouped_pilot_h5(tmp_path: Path) -> Path:
    path = tmp_path / "pilot.h5"
    families = ["uniform"] * 4 + ["layered"] * 4 + ["marmousi"] * 5
    groups = [f"u{i}" for i in range(4)] + ["layered-0"] * 4 + ["marmousi-0"] * 5
    n, nt, nz, nx = len(families), 13, 5, 6
    time_s = np.linspace(0.0, 1.2, nt, dtype=np.float64)
    z, x = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    velocity_by_group: dict[str, np.ndarray] = {}
    velocity = []
    wavefield = []
    source_map = np.zeros((n, nz, nx), np.float32)
    for record, group in enumerate(groups):
        if group not in velocity_by_group:
            velocity_by_group[group] = np.full((nz, nx), 1800.0 + 10 * record, np.float32)
        velocity.append(velocity_by_group[group])
        frames = np.stack(
            [record * 1000.0 + frame * 100.0 + 10.0 * z + x for frame in range(nt)]
        ).astype(np.float32)
        wavefield.append(frames)
        source_map[record, record % nz, (record * 2) % nx] = 1.0
    with h5py.File(path, "w") as h5:
        h5.attrs.update(manifest_sha256="manifest", config_sha256="config")
        text = h5py.string_dtype()
        h5["medium_type"] = np.asarray(families, dtype=text)
        h5["split"] = np.asarray(["train"] * n, dtype=text)
        h5["sample_id"] = np.asarray([f"sample-{i}" for i in range(n)], dtype=text)
        h5["group_id"] = np.asarray(groups, dtype=text)
        h5["sample_sha256"] = np.asarray([f"sha-{i}" for i in range(n)], dtype=text)
        h5["split_id"] = np.zeros(n, np.uint8)
        h5["time_s"] = time_s
        h5["x_m"] = np.linspace(0.0, 50.0, nx)
        h5["z_m"] = np.linspace(0.0, 40.0, nz)
        h5["velocity_mps"] = np.stack(velocity)
        h5["wavefield"] = np.stack(wavefield)
        h5["source_map"] = source_map
        h5["source_x_m"] = np.linspace(5.0, 45.0, n)
        h5["source_z_m"] = np.linspace(5.0, 35.0, n)
        h5["source_f0_hz"] = np.linspace(8.0, 20.0, n)
        h5["source_t0_s"] = np.full(n, 0.1)
        h5["source_amplitude"] = np.ones(n)
    return path


def test_schedule_is_balanced_unique_and_reuses_grouped_media(grouped_pilot_h5: Path):
    manifest = build_manifest(grouped_pilot_h5)
    first = build_pilot_schedule(manifest, split="train", steps=6, seed=17)
    second = build_pilot_schedule(manifest, split="train", steps=6, seed=17)
    assert first == second
    records = tuple(record for record in manifest.records if record.split == "train")
    for spec in first:
        chosen = [records[index] for index in spec.record_indices]
        assert len(chosen) == len({record.sample_id for record in chosen}) == 12
        assert Counter(record.medium_type for record in chosen) == {
            "uniform": 4,
            "layered": 4,
            "marmousi": 4,
        }
        assert len({record.group_id for record in chosen}) == 6
    marmousi_ids = {
        records[index].sample_id
        for spec in first
        for index in spec.record_indices
        if records[index].medium_type == "marmousi"
    }
    assert len(marmousi_ids) == 5


def test_pilot_batch_has_exact_interpolated_targets_and_matching_queries(
    grouped_pilot_h5: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = build_manifest(grouped_pilot_h5)
    schedule = build_pilot_schedule(manifest, split="train", steps=2, seed=17)
    dataset = PilotBatchDataset(
        grouped_pilot_h5,
        manifest,
        split="train",
        schedule=schedule,
        continuous_fraction=0.25,
        query_points=24,
        seed=17,
    )
    original_read_wavefield = dataset.records.read_wavefield
    wavefield_reads: list[tuple[int, torch.Tensor]] = []

    def tracked_read_wavefield(record_index: int, requested_time_s: torch.Tensor):
        wavefield_reads.append((record_index, requested_time_s.clone()))
        return original_read_wavefield(record_index, requested_time_s)

    monkeypatch.setattr(dataset.records, "read_wavefield", tracked_read_wavefield)
    batch = dataset[0]
    assert len(wavefield_reads) == 12
    assert all(len(requested_time_s) == 3 for _, requested_time_s in wavefield_reads)
    assert batch.velocity_mps.shape == (6, 1, 5, 6)
    assert batch.source_parameters.shape == (12, 5)
    assert batch.dense_target_physical.shape == (12, 3, 5, 6)
    assert batch.query_coords.shape == (12, 24, 3)
    assert Counter(batch.medium_type) == {"uniform": 4, "layered": 4, "marmousi": 4}
    assert int((~batch.target_exact).sum()) == 9
    assert torch.all((batch.interpolation_alpha > 0) == (~batch.target_exact))
    torch.testing.assert_close(
        batch.interpolation_alpha[~batch.target_exact],
        torch.full((9,), 0.5),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert torch.isfinite(batch.dense_target_physical).all()
    assert torch.isfinite(batch.query_target_physical).all()
    assert torch.all(batch.query_probability > 0)

    nx, nz = len(batch.x_m), len(batch.z_m)
    dx, dz = float(batch.x_m[1] - batch.x_m[0]), float(batch.z_m[1] - batch.z_m[0])
    for record in range(12):
        x_index = torch.round((batch.query_coords[record, :, 0] - batch.x_m[0]) / dx).long()
        z_index = torch.round((batch.query_coords[record, :, 1] - batch.z_m[0]) / dz).long()
        time_difference = (
            batch.query_coords[record, :, 2, None] - batch.requested_time_s[record, None]
        ).abs()
        time_index = time_difference.argmin(dim=1)
        expected = batch.dense_target_physical[record, time_index, z_index, x_index]
        torch.testing.assert_close(batch.query_target_physical[record], expected)
        assert x_index.min() >= 0 and x_index.max() < nx
        assert z_index.min() >= 0 and z_index.max() < nz


def test_pilot_loader_prefetches_whole_macro_batches(grouped_pilot_h5: Path):
    manifest = build_manifest(grouped_pilot_h5)
    dataset = PilotBatchDataset(
        grouped_pilot_h5,
        manifest,
        split="train",
        schedule=build_pilot_schedule(manifest, split="train", steps=2, seed=19),
        continuous_fraction=0.25,
        query_points=12,
        seed=19,
    )
    loader = make_pilot_loader(dataset, workers=1, prefetch_factor=2, pin_memory=False)
    batches = list(loader)
    assert [batch.step for batch in batches] == [0, 1]
    assert all(batch.dense_target_physical.shape[0] == 12 for batch in batches)


def test_curriculum_step_materializes_variable_microbatch_with_single_wavefield_read_per_record(
    grouped_pilot_h5: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = build_manifest(grouped_pilot_h5)
    spec = CurriculumStepSpec(
        optimizer_step=0,
        stage="uniform",
        microbatches=(
            CurriculumMicrobatchSpec(
                record_indices=(0, 1, 2, 3),
                family="uniform",
                replay=False,
            ),
        ),
    )
    dataset = CurriculumStepDataset(
        grouped_pilot_h5,
        manifest,
        split="train",
        schedule=(spec,),
        continuous_fraction=0.25,
        query_points=16,
        seed=29,
    )
    original_read_wavefield = dataset.materializer.records.read_wavefield
    reads: list[int] = []

    def tracked_read_wavefield(record_index: int, requested_time_s: torch.Tensor):
        reads.append(record_index)
        return original_read_wavefield(record_index, requested_time_s)

    monkeypatch.setattr(dataset.materializer.records, "read_wavefield", tracked_read_wavefield)
    step = dataset[0]
    assert step.stage == "uniform"
    assert step.optimizer_step == 0
    assert step.replay == (False,)
    assert step.loss_scale == (1.0,)
    assert len(step.microbatches) == 1
    assert step.microbatches[0].dense_target_physical.shape == (4, 3, 5, 6)
    assert reads == [0, 1, 2, 3]


def test_curriculum_loader_prefetches_whole_optimizer_steps(grouped_pilot_h5: Path):
    manifest = build_manifest(grouped_pilot_h5)
    schedule = tuple(
        CurriculumStepSpec(
            optimizer_step=step,
            stage="uniform",
            microbatches=(
                CurriculumMicrobatchSpec(
                    record_indices=(0, 1, 2, 3),
                    family="uniform",
                    replay=False,
                ),
            ),
        )
        for step in range(2)
    )
    dataset = CurriculumStepDataset(
        grouped_pilot_h5,
        manifest,
        split="train",
        schedule=schedule,
        continuous_fraction=0.25,
        query_points=8,
        seed=29,
    )
    loader = make_curriculum_loader(dataset, workers=0, prefetch_factor=2, pin_memory=False)
    steps = list(loader)
    assert [step.optimizer_step for step in steps] == [0, 1]
    assert all(len(step.microbatches) == 1 for step in steps)
