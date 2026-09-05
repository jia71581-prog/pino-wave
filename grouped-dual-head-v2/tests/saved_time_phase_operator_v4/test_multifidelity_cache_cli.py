from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from saved_time_phase_operator_v4.multifidelity import NumericalTeacherCache
from scripts.build_lwc84_multifidelity_cache import (
    TrainingRecord,
    initialize_or_validate_shard,
    load_exact_training_numerical_contract,
    load_manifest_index,
    reconstruct_generation_inputs,
    select_training_records,
    shard_training_records,
    source_identity,
    write_completed_batch,
)
from fno_acoustic.data_generation.config import load_config
from scripts.merge_lwc84_multifidelity_cache import (
    build_source_teacher_vds,
    merge_cache_shards,
)


SOURCE_IDENTITY = {
    "path": "/data/source.h5",
    "manifest_sha256": "source-manifest",
    "config_sha256": "source-config",
}
NUMERICAL_CONTRACT = {
    "dx_m": 10.0,
    "dz_m": 10.0,
    "internal_dt_s": 0.00025,
    "npml": 20,
    "top_boundary": "free_surface_dirichlet",
    "other_boundaries": "cpml",
}


def _records() -> tuple[TrainingRecord, ...]:
    return tuple(
        TrainingRecord(
            source_index=100 + index,
            sample_id=f"train_uniform_{index:05d}",
            medium_type="uniform",
        )
        for index in range(6)
    )


def _complete_shard(path: Path, records: tuple[TrainingRecord, ...], offset: int):
    pending = initialize_or_validate_shard(
        path,
        records=records,
        source_identity=SOURCE_IDENTITY,
        time_indices=(0, 2),
        time_s=(0.0, 0.005),
        spatial_shape=(3, 2),
        numerical_contract=NUMERICAL_CONTRACT,
    )
    values = (
        np.arange(len(records) * 2 * 3 * 2, dtype=np.float32).reshape(
            len(records), 2, 3, 2
        )
        + offset
    )
    values[:, :, 0, :] = 0.0
    write_completed_batch(
        path,
        positions=pending,
        wavefield=values,
        lwc_qmax=np.full(len(records), 0.5, dtype=np.float64),
    )
    return values


def test_training_records_use_deterministic_stride_shards():
    records = _records()

    assert shard_training_records(records, shard_index=0, shard_count=2) == records[::2]
    assert shard_training_records(records, shard_index=1, shard_count=2) == records[1::2]


def test_numerical_teacher_reproduces_exact_training_grid_and_cpml_contract(tmp_path):
    config = tmp_path / "frozen_config.yaml"
    config.write_text(
        """
grid:
  nx: 401
  nz: 401
  lx_m: 2000.0
  lz_m: 2000.0
  dx_m: 5.0
  dz_m: 5.0
  centering: node
storage_grid:
  nx: 201
  nz: 201
  dx_m: 10.0
  dz_m: 10.0
  centering: node
  restriction: binomial5_lowpass_then_decimate2
time:
  dt_used_s: 0.000125
  nt_out: 401
  dt_out_s: 0.0025
boundaries:
  top: free_surface_dirichlet
  left: cpml
  right: cpml
  bottom: cpml
  npml: 40
  cpml_target_reflection: 1.0e-8
  cpml_polynomial_order: 3
  cpml_outside_physical_domain: true
  kappa_max: 3.0
  minimum_frequency_hz: 8.0
  alpha_max_definition: pi_times_minimum_frequency
""".strip()
        + "\n"
    )

    contract = load_exact_training_numerical_contract(config)

    assert contract["solver_grid_shape"] == [401, 401]
    assert contract["saved_grid_shape"] == [201, 201]
    assert contract["solver_dx_m"] == 5.0
    assert contract["saved_dx_m"] == 10.0
    assert contract["npml"] == 40
    assert contract["cpml_physical_thickness_m"] == 200.0
    assert contract["cpml_target_reflection"] == 1.0e-8
    assert contract["cpml_polynomial_order"] == 3
    assert contract["kappa_max"] == 3.0
    assert contract["minimum_frequency_hz"] == 8.0
    assert contract["internal_dt_s"] == 0.000125
    assert contract["output_restriction_factor"] == 2
    assert contract["restriction"] == "binomial5_lowpass_then_decimate2"
    assert contract["top_boundary"] == "free_surface_dirichlet"
    assert contract["other_boundaries"] == "cpml"


PRODUCTION_DATASET = Path(
    "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1"
)


@pytest.mark.skipif(
    not (PRODUCTION_DATASET / "dataset_v1.h5").is_file(),
    reason="production LWC-84 dataset is not mounted",
)
def test_three_family_401_velocity_reconstruction_is_bitwise_identical_to_hdf5():
    source = PRODUCTION_DATASET / "dataset_v1.h5"
    config_path = PRODUCTION_DATASET / "frozen_config.yaml"
    manifest_path = PRODUCTION_DATASET / "manifest.jsonl"
    identity = source_identity(
        source, manifest_path=manifest_path, dataset_config=config_path
    )
    manifest = load_manifest_index(
        manifest_path, expected_sha256=str(identity["manifest_sha256"])
    )
    records = select_training_records(source)
    selected = tuple(
        next(record for record in records if record.medium_type == family)
        for family in ("uniform", "layered", "marmousi")
    )

    velocity, _, saved, _ = reconstruct_generation_inputs(
        source,
        selected,
        config=load_config(config_path),
        manifest_rows=manifest,
    )

    assert velocity.shape == (3, 401, 401)
    assert saved.shape == (3, 201, 201)
    assert float(velocity[1, 0].mean()) > float(velocity[1, -1].mean())


def test_cache_shard_resume_marks_data_before_completion(tmp_path):
    path = tmp_path / "shard_00.h5"
    records = _records()[:3]
    pending = initialize_or_validate_shard(
        path,
        records=records,
        source_identity=SOURCE_IDENTITY,
        time_indices=(0, 2),
        time_s=(0.0, 0.005),
        spatial_shape=(3, 2),
        numerical_contract=NUMERICAL_CONTRACT,
    )

    assert pending == (0, 1, 2)
    values = np.ones((2, 2, 3, 2), dtype=np.float32)
    values[:, :, 0, :] = 0.0
    write_completed_batch(
        path,
        positions=(0, 1),
        wavefield=values,
        lwc_qmax=np.asarray([0.4, 0.5]),
    )

    resumed = initialize_or_validate_shard(
        path,
        records=records,
        source_identity=SOURCE_IDENTITY,
        time_indices=(0, 2),
        time_s=(0.0, 0.005),
        spatial_shape=(3, 2),
        numerical_contract=NUMERICAL_CONTRACT,
    )
    assert resumed == (2,)
    with h5py.File(path, "r") as handle:
        assert handle["completed"][:].tolist() == [True, True, False]
        assert np.array_equal(handle["wavefield"][:2], values)

    with pytest.raises(ValueError, match="identity"):
        initialize_or_validate_shard(
            path,
            records=records,
            source_identity={**SOURCE_IDENTITY, "manifest_sha256": "wrong"},
            time_indices=(0, 2),
            time_s=(0.0, 0.005),
            spatial_shape=(3, 2),
            numerical_contract=NUMERICAL_CONTRACT,
        )


def test_merge_builds_global_order_vds_without_copying_wavefields(tmp_path):
    records = _records()
    shard_paths = (tmp_path / "shard_00.h5", tmp_path / "shard_01.h5")
    shard_values = tuple(
        _complete_shard(
            path,
            shard_training_records(records, shard_index=index, shard_count=2),
            offset=1000 * index,
        )
        for index, path in enumerate(shard_paths)
    )
    output = tmp_path / "teacher_vds.h5"

    report = merge_cache_shards(
        shard_paths,
        output,
        expected_records=records,
        source_identity=SOURCE_IDENTITY,
        numerical_contract=NUMERICAL_CONTRACT,
    )

    assert report["record_count"] == 6
    assert report["time_count"] == 2
    assert output.stat().st_size < sum(path.stat().st_size for path in shard_paths)
    cache = NumericalTeacherCache(
        output,
        expected_source_manifest_sha256="source-manifest",
        expected_sample_ids=tuple(record.sample_id for record in records),
    )
    values = cache.read(
        tuple(record.sample_id for record in records),
        torch.tensor([[0, 2]] * len(records)),
    ).numpy()
    expected = np.empty_like(values)
    expected[::2] = shard_values[0]
    expected[1::2] = shard_values[1]
    assert np.array_equal(values, expected)


def test_direct_teacher_vds_reuses_exact_source_solver_outputs(tmp_path):
    source = tmp_path / "source.h5"
    wavefield = np.arange(6 * 5 * 3 * 2, dtype=np.float32).reshape(6, 5, 3, 2)
    with h5py.File(source, "w", libver="latest") as handle:
        handle.create_dataset("wavefield", data=wavefield)
        handle.create_dataset("time_s", data=np.arange(5, dtype=np.float64) * 0.1)
    records = tuple(
        TrainingRecord(index, f"train_uniform_{ordinal:05d}", "uniform")
        for ordinal, index in enumerate((0, 1, 4, 5))
    )
    output = tmp_path / "teacher_vds.h5"
    contract = {"saved_grid_shape": [3, 2], "output_time_count": 5}

    report = build_source_teacher_vds(
        source,
        output,
        expected_records=records,
        source_identity=SOURCE_IDENTITY,
        numerical_contract=contract,
        time_indices=(0, 2, 4),
    )

    assert report["zero_copy"] is True
    assert report["mapping_run_count"] == 2
    cache = NumericalTeacherCache(
        output,
        expected_source_manifest_sha256="source-manifest",
        expected_sample_ids=tuple(record.sample_id for record in records),
    )
    actual = cache.read(
        tuple(record.sample_id for record in records),
        torch.tensor([[0, 2, 4]] * len(records)),
    ).numpy()
    assert np.array_equal(actual, wavefield[[0, 1, 4, 5]][:, [0, 2, 4]])
