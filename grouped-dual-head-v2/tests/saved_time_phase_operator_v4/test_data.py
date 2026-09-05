from pathlib import Path

import pytest
import numpy as np
import torch
import yaml

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest
from grouped_ufno_mionet_v3.data.pilot import build_pilot_schedule
from saved_time_phase_operator_v4.data import (
    ExactStoredTimeBatchDataset,
    merge_pilot_batches,
    split_pilot_batch,
)
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec
from saved_time_phase_operator_v4.time_grid import SavedTimeGrid
from saved_time_phase_operator_v4.sampling import appearance_time_indices
from saved_time_phase_operator_v4.multifidelity import (
    fixed_teacher_time_indices,
    numerical_teacher_pool_indices,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = (
    REPOSITORY_ROOT
    / "configs/saved_time_v4/generated/"
    "local_field_w128_temporal_latent_a3_rank32_r5c_shared_dynamic_adapter_train_diag.yaml"
)
_fixture_values = yaml.safe_load(FIXTURE_CONFIG.read_text())
_base_config_path = Path(str(_fixture_values["base_config"]))
if not _base_config_path.is_absolute():
    _base_config_path = REPOSITORY_ROOT / _base_config_path
FIXTURE_BASE_CONFIG = V3Config.from_yaml(_base_config_path)
SOURCE_H5 = Path(FIXTURE_BASE_CONFIG.data.source_h5)
TRAVEL_TIME_H5 = Path(str(_fixture_values["travel_time_h5"]))
FIXTURE_SPLIT = "train"


@pytest.fixture(scope="module")
def exact_batch():
    if not SOURCE_H5.is_file():
        pytest.skip(f"configured production acoustic HDF5 is unavailable: {SOURCE_H5}")
    manifest = build_manifest(SOURCE_H5)
    schedule = build_pilot_schedule(manifest, split=FIXTURE_SPLIT, steps=1, seed=29)
    dataset = ExactStoredTimeBatchDataset(
        SOURCE_H5,
        manifest,
        split=FIXTURE_SPLIT,
        schedule=schedule,
        query_points=8,
        seed=29,
    )
    return dataset[0], manifest


def test_v4_batch_contains_only_exact_stored_targets(exact_batch):
    batch, manifest = exact_batch
    grid = SavedTimeGrid.from_values(manifest.time_s)

    assert batch.requested_time_s.shape == (12, 4)
    assert batch.target_exact.all()
    assert torch.equal(batch.left_index, batch.right_index)
    assert torch.count_nonzero(batch.interpolation_alpha) == 0
    assert grid.indices(batch.requested_time_s).shape == (12, 4)
    assert batch.requested_time_s[:, -1].min() >= 0.7


def test_microbatch_split_preserves_all_records_and_valid_medium_maps(exact_batch):
    batch, _ = exact_batch

    pieces = split_pilot_batch(batch, microbatch_records=6)

    assert len(pieces) == 2
    assert sum(len(piece.sample_id) for piece in pieces) == 12
    assert tuple(value for piece in pieces for value in piece.sample_id) == batch.sample_id
    for piece in pieces:
        assert piece.velocity_mps.shape[0] == int(piece.record_to_medium.max()) + 1
        assert piece.source_parameters.shape[0] == 6
        assert piece.dense_target_physical.shape[1] == 4


def test_microbatch_split_supports_nondivisible_sizes(exact_batch):
    batch, _ = exact_batch
    pieces = split_pilot_batch(batch, microbatch_records=5)

    assert tuple(len(piece.sample_id) for piece in pieces) == (5, 5, 2)
    assert tuple(value for piece in pieces for value in piece.sample_id) == batch.sample_id


def test_exact_dataset_loads_and_splits_source_eikonal_fields(exact_batch):
    if not TRAVEL_TIME_H5.is_file():
        pytest.skip(f"configured travel-time cache is unavailable: {TRAVEL_TIME_H5}")
    _, manifest = exact_batch
    schedule = build_pilot_schedule(manifest, split=FIXTURE_SPLIT, steps=1, seed=29)
    dataset = ExactStoredTimeBatchDataset(
        SOURCE_H5,
        manifest,
        split=FIXTURE_SPLIT,
        schedule=schedule,
        query_points=1,
        seed=29,
        travel_time_h5=TRAVEL_TIME_H5,
    )

    batch = dataset[0]
    pieces = split_pilot_batch(batch, microbatch_records=6)

    assert batch.dense_travel_time_s.shape == (12, 201, 201)
    assert torch.isfinite(batch.dense_travel_time_s).all()
    assert torch.all(batch.dense_travel_time_s >= 0.0)
    source_cells = batch.source_map[:, 0].flatten(1).argmax(dim=1)
    travel_time_cells = batch.dense_travel_time_s.flatten(1)
    minimum_cells = travel_time_cells.argmin(dim=1)
    # A ray-based cache can assign a small positive source-to-cell-centre time;
    # the stable physical invariant is that the source cell is the minimum.
    assert torch.equal(minimum_cells, source_cells)
    torch.testing.assert_close(
        travel_time_cells.gather(1, source_cells[:, None]).squeeze(1),
        travel_time_cells.amin(dim=1),
    )
    assert torch.equal(
        torch.cat([piece.dense_travel_time_s for piece in pieces]),
        batch.dense_travel_time_s,
    )


def test_four_macros_merge_and_split_into_physical_microbatches_of_sixteen(exact_batch):
    batch, _ = exact_batch

    merged = merge_pilot_batches((batch, batch, batch, batch))
    pieces = split_pilot_batch(merged, microbatch_records=16)

    assert len(merged.sample_id) == 48
    assert len(pieces) == 3
    assert all(len(piece.sample_id) == 16 for piece in pieces)
    assert merged.velocity_mps.shape[0] == 4 * batch.velocity_mps.shape[0]
    assert int(merged.record_to_medium.max()) + 1 == merged.velocity_mps.shape[0]


def test_full_support_spec_uses_appearance_indices(exact_batch):
    _, manifest = exact_batch
    spec = FullSupportStepSpec(
        step=0,
        epoch=0,
        record_indices=tuple(range(12)),
        appearance_indices=(7,) * 12,
    )
    dataset = ExactStoredTimeBatchDataset(
        SOURCE_H5,
        manifest,
        split=FIXTURE_SPLIT,
        schedule=(spec,),
        query_points=1,
        seed=307,
        time_policy="appearance4",
    )

    batch = dataset[0]

    assert batch.requested_time_s.shape == (12, 4)
    assert batch.target_exact.all()
    assert torch.equal(batch.left_index, batch.right_index)


def test_validation_policy_materializes_sixteen_exact_frames(exact_batch):
    _, manifest = exact_batch
    spec = FullSupportStepSpec(
        step=19,
        epoch=2,
        record_indices=tuple(range(12)),
        appearance_indices=(2,) * 12,
    )
    dataset = ExactStoredTimeBatchDataset(
        SOURCE_H5,
        manifest,
        split=FIXTURE_SPLIT,
        schedule=(spec,),
        query_points=1,
        seed=307,
        time_policy="validation16",
    )

    batch = dataset[0]

    assert batch.requested_time_s.shape == (12, 16)
    assert batch.target_exact.all()
    assert torch.equal(batch.left_index, batch.right_index)
    assert torch.count_nonzero(batch.interpolation_alpha) == 0


def test_recovery_policy_selects_sixteen_unique_frames_with_onset_pair():
    axis = torch.linspace(0.0, 1.0, 401, dtype=torch.float64).numpy()
    onset = 40

    indices = appearance_time_indices(
        axis,
        source_t0_s=float(axis[onset]),
        sample_id="recovery-sample",
        appearance=7,
        seed=307,
        count=16,
    )

    assert indices.shape == (16,)
    assert len(set(indices.tolist())) == 16
    assert onset in indices and onset + 1 in indices
    assert (indices[1:] > indices[:-1]).all()
    assert ((indices[1:] - indices[:-1]) == 1).sum() >= 8


def test_recovery_onset_pair_uses_ricker_start_instead_of_wavelet_center():
    axis = torch.linspace(0.0, 1.0, 401, dtype=torch.float64).numpy()
    source_t0_s = 0.15
    source_f0_hz = 10.0
    start = int(np.searchsorted(axis, source_t0_s - 1.0 / source_f0_hz))

    indices = appearance_time_indices(
        axis,
        source_t0_s=source_t0_s,
        source_f0_hz=source_f0_hz,
        sample_id="ricker-start-sample",
        appearance=0,
        seed=307,
        count=16,
    )

    assert start in indices and start + 1 in indices


def test_all_saved_policy_materializes_every_exact_frame(exact_batch):
    _, manifest = exact_batch
    spec = FullSupportStepSpec(
        step=31,
        epoch=0,
        record_indices=tuple(range(12)),
        appearance_indices=(0,) * 12,
    )
    dataset = ExactStoredTimeBatchDataset(
        SOURCE_H5,
        manifest,
        split=FIXTURE_SPLIT,
        schedule=(spec,),
        query_points=1,
        seed=307,
        time_policy="all_saved",
    )

    batch = dataset[0]

    assert batch.requested_time_s.shape == (12, 401)
    assert batch.target_exact.all()
    assert torch.equal(batch.left_index, batch.right_index)
    assert torch.equal(batch.left_index[0], torch.arange(401))


def test_numerical_teacher_policy_materializes_only_registered_pool(exact_batch):
    _, manifest = exact_batch
    pool = fixed_teacher_time_indices(stored_time_count=401, count=64)
    spec = FullSupportStepSpec(
        step=37,
        epoch=0,
        record_indices=tuple(range(12)),
        appearance_indices=tuple(range(12)),
    )
    dataset = ExactStoredTimeBatchDataset(
        SOURCE_H5,
        manifest,
        split=FIXTURE_SPLIT,
        schedule=(spec,),
        query_points=1,
        seed=372,
        time_policy="numerical_teacher_pool",
        frames_per_record=16,
        time_index_pool=pool,
    )

    batch = dataset[0]

    assert batch.left_index.shape == (12, 16)
    assert batch.target_exact.all()
    assert torch.equal(batch.left_index, batch.right_index)
    for offset, sample_id in enumerate(batch.sample_id):
        expected = numerical_teacher_pool_indices(
            pool,
            sample_id=sample_id,
            appearance=offset,
            seed=372,
            count=16,
        )
        assert tuple(batch.left_index[offset].tolist()) == expected
