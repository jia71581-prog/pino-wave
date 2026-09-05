from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import h5py

from fno_acoustic.data_generation.config import grid_from_config, load_config, time_from_config
from fno_acoustic.data_generation.lwc84_manifest import (
    build_lwc84_manifest,
    configured_lwc84_counts,
    validate_lwc84_manifest,
)
from fno_acoustic.data_generation.holdout_audit import (
    audit_candidate_split_disjointness,
    audit_generated_sample_sha256_disjointness,
    audit_historical_manifest_disjointness,
)
from fno_acoustic.data_generation.model_marmousi import (
    MarmousiInventoryRow,
    inventory_marmousi,
    select_full_marmousi_candidate,
)
from fno_acoustic.data_generation.pipeline_lwc84 import (
    _production_velocity,
    _smoke_problem,
    group_partitions_for_batch,
    prepare_production_batch,
    production_batch_size,
    storage_budget,
    validated_lwc84_time_plan,
)
from scripts.launch_gpu_dataset_workers import (
    _dataset_vds_path,
    _normalized_worker_exit_code,
)


CONFIG = "configs/datasets/acoustic_lwc84_2km_401x401_to_201_v1.yaml"
HIRES_MARMOUSI_CONFIG = (
    "configs/datasets/acoustic_lwc84_2km_401x401_to_201_marmousi1_v2.yaml"
)


def test_frozen_config_uses_401_solver_grid_and_201_storage_grid() -> None:
    config = load_config(CONFIG)
    grid = grid_from_config(config)
    time = time_from_config(config)
    assert (grid.nz, grid.nx, grid.centering) == (401, 401, "node")
    assert (grid.lz_m, grid.lx_m, grid.dz_m, grid.dx_m) == (2000.0, 2000.0, 5.0, 5.0)
    assert config["storage_grid"] == {
        "nx": 201,
        "nz": 201,
        "dx_m": 10.0,
        "dz_m": 10.0,
        "centering": "node",
        "restriction": "binomial5_lowpass_then_decimate2",
    }
    assert (time.nt_out, time.dt_out_s, time.t_end_s) == (33, 0.03125, 1.0)
    assert config["time"]["dt_requested_s"] == 1.0e-5
    assert config["time"]["dt_used_s"] == 1.0e-5
    assert config["time"]["snapshot_stride"] == 3125
    assert config["production"]["confirm_token"] == "RUN_4000"
    assert config["production"]["cpu_fallback"] is False
    assert production_batch_size(config, requested=128) == 128


def test_hires_config_uses_exact_original_lwc84_time_protocol() -> None:
    config = load_config(HIRES_MARMOUSI_CONFIG)
    plan = validated_lwc84_time_plan(config)

    assert plan.dt_requested_s == 1.0e-5
    assert plan.dt_used_s == 1.0e-5
    assert plan.snapshot_stride == 3125
    assert plan.output_interval_s == 0.03125
    assert plan.reduced is False


def test_lwc84_time_plan_rejects_a_drifting_frozen_dt() -> None:
    config = load_config(HIRES_MARMOUSI_CONFIG)
    config["time"]["dt_used_s"] = 1.25e-4

    with pytest.raises(ValueError, match="differs from the aligned LWC plan"):
        validated_lwc84_time_plan(config)


def test_smoke_problem_uses_the_configured_output_cadence() -> None:
    config = load_config(HIRES_MARMOUSI_CONFIG)
    _, _, output_times, t_end = _smoke_problem(config, "unit")

    np.testing.assert_allclose(
        output_times,
        np.arange(7, dtype=np.float64) * 0.03125,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert t_end == pytest.approx(0.1875)


def test_batch_128_groups_sixteen_eight_sample_shards() -> None:
    partitions = [("train", index, [{}] * 8) for index in range(17)]
    groups = group_partitions_for_batch(partitions, batch_size=128)
    assert [len(group) for group in groups] == [16, 1]
    assert [sum(len(rows) for _, _, rows in group) for group in groups] == [128, 8]


def test_prepare_production_batch_stacks_all_samples_for_one_gpu_call() -> None:
    rows = [
        {
            "source_x_m": 200.0 + index,
            "source_z_m": 100.0 + index,
            "source_f0_hz": 20.0 + index,
            "source_t0_s": 0.1 + 0.01 * index,
            "source_amplitude": 1.0,
        }
        for index in range(3)
    ]
    velocities = [np.full((5, 5), 2000.0 + index, dtype=np.float32) for index in range(3)]

    velocity_batch, source_parameters = prepare_production_batch(rows, velocities)

    assert velocity_batch.shape == (3, 5, 5)
    assert velocity_batch.dtype == np.float32
    assert source_parameters["source_x_m"].tolist() == [200.0, 201.0, 202.0]
    assert source_parameters["source_f0_hz"].tolist() == [20.0, 21.0, 22.0]


def test_original_marmousi_inventory_still_fails_full_2km_extent_gate() -> None:
    config = load_config(CONFIG)
    old_path = Path("/synthetic/legacy_marmousi_bl.bin")
    old_sha256 = str(config["marmousi"]["source_original_sha256"])
    rows = [
        MarmousiInventoryRow(
            path=str(old_path),
            sha256=old_sha256,
            size_bytes=1,
            shape=[116, 227],
            dtype="float32",
            velocity_min_mps=1500.0,
            velocity_max_mps=5500.0,
            velocity_mean_mps=3000.0,
            dx_m=5.0,
            dz_m=5.0,
            shape_source="synthetic_legacy_contract",
            extractable_2km_crop_count=0,
            physical_width_m=1130.0,
            physical_depth_m=575.0,
        )
    ]
    original = {
        **config["marmousi"],
        "velocity_file": str(old_path),
        "sha256": old_sha256,
    }
    with pytest.raises(ValueError, match="cannot provide a 2000.0x2000.0 m crop"):
        select_full_marmousi_candidate(rows, original)


def test_user_authorized_interpolated_marmousi_passes_extent_gate() -> None:
    config = load_config(CONFIG)
    rows = inventory_marmousi(Path(config["marmousi"]["root_dir"]))
    selected = select_full_marmousi_candidate(rows, config["marmousi"])
    assert selected.shape == [801, 2401]
    assert selected.sha256 == config["marmousi"]["sha256"]
    assert config["marmousi"]["coordinate_transform"] == "normalized_extent_stretch"


def test_original_resolution_marmousi_is_provenance_bound_and_full_extent() -> None:
    config = load_config(HIRES_MARMOUSI_CONFIG)
    rows = inventory_marmousi(Path(config["marmousi"]["root_dir"]))

    selected = select_full_marmousi_candidate(rows, config["marmousi"])

    assert selected.shape == [751, 2301]
    assert (selected.dz_m, selected.dx_m) == (4.0, 4.0)
    assert (selected.physical_depth_m, selected.physical_width_m) == (3000.0, 9200.0)
    assert selected.sha256 == config["marmousi"]["sha256"]
    assert config["marmousi"]["coordinate_transform"] == "transpose_only_no_geological_interpolation"


def test_hires_marmousi_manifest_spreads_no_water_crop_origins_across_model() -> None:
    config = load_config(HIRES_MARMOUSI_CONFIG)
    rows = build_lwc84_manifest(
        config,
        marmousi_geometry={
            "shape": [751, 2301],
            "dx_m": 4.0,
            "dz_m": 4.0,
            "sha256": config["marmousi"]["sha256"],
        },
    )
    validate_lwc84_manifest(rows, config)
    marmousi = [
        row for row in rows if row["medium_type"] == "marmousi" and row["split"] != "ood_canonical"
    ]
    origins = {(float(row["crop_x0_m"]), float(row["crop_z0_m"])) for row in marmousi}
    origins_by_split = {
        split: {
            (float(row["crop_x0_m"]), float(row["crop_z0_m"]))
            for row in marmousi
            if row["split"] == split
        }
        for split in ("train", "validation", "test_id")
    }

    assert len(origins) == 200
    assert min(x for x, _ in origins) == 0.0
    assert max(x for x, _ in origins) == 4900.0
    assert min(z for _, z in origins) == 50.0
    assert max(z for _, z in origins) == 1000.0
    assert all(float(row["crop_z0_m"]) > 40.0 for row in marmousi)
    assert [len(origins_by_split[split]) for split in ("train", "validation", "test_id")] == [140, 30, 30]
    for left, right in (("train", "validation"), ("train", "test_id"), ("validation", "test_id")):
        assert all(
            x1 + 2000.0 <= x2 or x2 + 2000.0 <= x1
            for x1, _ in origins_by_split[left]
            for x2, _ in origins_by_split[right]
        )


def test_manifest_validator_rejects_cross_split_marmousi_overlap() -> None:
    config = load_config(HIRES_MARMOUSI_CONFIG)
    rows = build_lwc84_manifest(
        config,
        marmousi_geometry={
            "shape": [751, 2301],
            "dx_m": 4.0,
            "dz_m": 4.0,
            "sha256": config["marmousi"]["sha256"],
        },
    )
    train_row = next(
        row for row in rows if row["split"] == "train" and row["medium_type"] == "marmousi"
    )
    validation_row = next(
        row
        for row in rows
        if row["split"] == "validation" and row["medium_type"] == "marmousi"
    )
    validation_row["crop_x0_m"] = train_row["crop_x0_m"]
    validation_row["crop_z0_m"] = train_row["crop_z0_m"]

    with pytest.raises(ValueError, match="Marmousi spatial leakage"):
        validate_lwc84_manifest(rows, config)


def test_negative_worker_signal_cannot_be_reported_as_success() -> None:
    assert _normalized_worker_exit_code([-9, 0, 0, 0]) == 1
    assert _normalized_worker_exit_code([0, 0, 0, 0]) == 0


def test_split_scoped_vds_keeps_validation_and_test_artifacts_separate(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["storage"]["vds_filename"] = "frozen.h5"
    config["storage"]["split_scoped_vds"] = True

    validation = _dataset_vds_path(config, tmp_path, [{"split": "validation"}])
    test_id = _dataset_vds_path(config, tmp_path, [{"split": "test_id"}])

    assert validation.name == "frozen_validation.h5"
    assert test_id.name == "frozen_test_id.h5"


def test_manifest_has_exact_4000_plus_three_contract_without_leakage() -> None:
    config = load_config(CONFIG)
    # Synthetic geometry only exercises planning logic; production uses the real inventory gate above.
    rows = build_lwc84_manifest(
        config,
        marmousi_geometry={"shape": [801, 2401], "dx_m": 5.0, "dz_m": 5.0, "sha256": "a" * 64},
    )
    summary = validate_lwc84_manifest(rows, config)
    assert len(rows) == 4003
    assert summary["split_counts"] == {"train": 2800, "validation": 600, "test_id": 600, "ood_canonical": 3}
    assert summary["medium_counts"] == {"uniform": 600, "layered": 1600, "anomaly": 800, "marmousi": 1000}
    assert summary["ood_case_ids"] == [
        "ood_uniform_c4000_f15",
        "ood_layered_c3000_c5000_f15",
        "ood_marmousi_holdout_f10",
    ]
    non_ood = [row for row in rows if row["split"] != "ood_canonical"]
    assert all(not (9.75 <= row["source_f0_hz"] <= 10.25) for row in non_ood)
    assert all(not (14.75 <= row["source_f0_hz"] <= 15.25) for row in non_ood)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_group[row["group_id"]].add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert Counter(row["medium_type"] for row in non_ood) == Counter(
        {"uniform": 600, "layered": 1600, "anomaly": 800, "marmousi": 1000}
    )


def test_manifest_supports_namespaced_frozen_validation_and_test_only_contract() -> None:
    config = copy.deepcopy(load_config(HIRES_MARMOUSI_CONFIG))
    config["seed"] = 20260812
    config["dataset"]["sample_namespace"] = "target5_frozen_r1"
    config["dataset"]["splits"] = {"train": 0, "validation": 480, "test_id": 480}
    config["dataset"]["composition"] = {
        "uniform": {"train": 0, "validation": 90, "test_id": 90},
        "layered": {"train": 0, "validation": 240, "test_id": 240},
        "anomaly": {"train": 0, "validation": 0, "test_id": 0},
        "marmousi": {"train": 0, "validation": 150, "test_id": 150},
    }
    config["dataset"]["ood_canonical"] = []
    config["marmousi"]["require_cross_split_nonoverlap"] = False

    counts = configured_lwc84_counts(config)
    rows = build_lwc84_manifest(
        config,
        marmousi_geometry={
            "shape": [751, 2301],
            "dx_m": 4.0,
            "dz_m": 4.0,
            "sha256": config["marmousi"]["sha256"],
        },
    )
    summary = validate_lwc84_manifest(rows, config)

    assert counts["sample_count"] == 960
    assert summary["split_counts"] == {
        "train": 0,
        "validation": 480,
        "test_id": 480,
        "ood_canonical": 0,
    }
    assert summary["medium_counts"] == {
        "uniform": 180,
        "layered": 480,
        "anomaly": 0,
        "marmousi": 300,
    }
    assert all(str(row["sample_id"]).startswith("target5_frozen_r1__") for row in rows)
    assert all(str(row["group_id"]).startswith("target5_frozen_r1__") for row in rows)


def test_configured_counts_reject_split_composition_mismatch() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["dataset"]["splits"]["validation"] = 599

    with pytest.raises(ValueError, match="composition sums to 600"):
        configured_lwc84_counts(config)


def test_pretruth_holdout_audit_detects_semantic_group_overlap(tmp_path: Path) -> None:
    historical = [
        {
            "sample_id": "old-a",
            "group_id": "old-group",
            "medium_type": "uniform",
            "medium_parameters": {"seed": 1, "velocity_mps": 2400.0},
            "source_x_m": 500.0,
            "source_z_m": 100.0,
            "source_f0_hz": 20.0,
            "source_t0_s": 0.075,
            "source_amplitude": 1.0,
            "crop_x0_m": None,
            "crop_z0_m": None,
        }
    ]
    candidate = copy.deepcopy(historical)
    candidate[0]["sample_id"] = "new-a"
    candidate[0]["group_id"] = "new-group"
    candidate[0]["medium_parameters"]["seed"] = 99
    manifest = tmp_path / "historical.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in historical))

    audit = audit_historical_manifest_disjointness(candidate, [manifest])

    assert audit["passed"] is False
    assert audit["intersection_counts"] == {
        "sample_id": 0,
        "group_id": 0,
        "semantic_group_sha256": 1,
        "problem_sha256": 1,
    }


def test_postgeneration_sample_hash_audit_blocks_exact_content_overlap(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.h5"
    historical = tmp_path / "historical.h5"
    text = h5py.string_dtype("ascii", 64)
    with h5py.File(candidate, "w") as handle:
        handle.create_dataset("sample_sha256", data=np.asarray(["a" * 64, "b" * 64], dtype=text))
    with h5py.File(historical, "w") as handle:
        handle.create_dataset("sample_sha256", data=np.asarray(["b" * 64, "c" * 64], dtype=text))

    audit = audit_generated_sample_sha256_disjointness(candidate, [historical])

    assert audit["passed"] is False
    assert audit["intersection_count"] == 1
    assert audit["intersection_examples"] == ["b" * 64]


def test_internal_split_audit_blocks_renamed_semantic_group_overlap() -> None:
    rows = [
        {
            "sample_id": f"{split}-sample",
            "group_id": f"{split}-group",
            "split": split,
            "medium_type": "uniform",
            "medium_parameters": {"seed": seed, "velocity_mps": 2400.0},
            "source_x_m": 500.0,
            "source_z_m": 100.0,
            "source_f0_hz": 20.0,
            "source_t0_s": 0.075,
            "source_amplitude": 1.0,
            "crop_x0_m": None,
            "crop_z0_m": None,
        }
        for split, seed in (("validation", 1), ("test_id", 2))
    ]

    audit = audit_candidate_split_disjointness(rows)

    assert audit["passed"] is False
    assert audit["pairwise_intersection_counts"]["validation__vs__test_id"][
        "semantic_group_sha256"
    ] == 1


def test_fixed_ood_layer_uses_configured_velocities_without_seed() -> None:
    config = load_config(CONFIG)
    grid = grid_from_config(config)
    rows = build_lwc84_manifest(
        config,
        marmousi_geometry={"shape": [801, 2401], "dx_m": 5.0, "dz_m": 5.0, "sha256": "a" * 64},
    )
    row = next(item for item in rows if item["sample_id"] == "ood_layered_c3000_c5000_f15")

    velocity = _production_velocity(config, grid, row)

    assert velocity.shape == (401, 401)
    assert np.all(velocity[:200] == np.float32(3000.0))
    assert np.all(velocity[200:] == np.float32(5000.0))


def test_storage_budget_counts_only_concurrent_partial_shards_as_temporary_space() -> None:
    config = load_config(CONFIG)
    budget = storage_budget(config)
    bytes_per_sample = budget["total_uncompressed_bytes"] / budget["sample_count_including_ood"]
    one_shard = bytes_per_sample * config["storage"]["samples_per_shard"]
    assert budget["estimated_temporary_bytes"] <= one_shard * 1.01
    assert budget["required_free_bytes"] < budget["total_uncompressed_bytes"] * 1.2
