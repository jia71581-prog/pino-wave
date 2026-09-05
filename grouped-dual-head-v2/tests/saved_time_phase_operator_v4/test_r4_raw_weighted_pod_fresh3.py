from __future__ import annotations

import ast
from collections import Counter

import pytest
import torch

from scripts import confirm_r4_raw_weighted_pod_fresh3 as confirm


def test_winner_and_gates_are_frozen() -> None:
    assert confirm.WINNER_ARM == "raw+weighted"
    assert confirm.RANKS == (8, 16, 32)
    assert confirm.RANK32_REDUCTION_MINIMUM == 0.20
    assert confirm.TIME_COUNT == 401
    assert confirm.TIME_BLOCK == 16
    assert confirm.GPU_SECONDS_MAXIMUM == 360.0


def test_fresh_stored_records_are_train_group_disjoint_and_unexposed() -> None:
    manifest = confirm.parent_runtime.load_manifest_payload()
    registry = confirm.exposure_registry(manifest)
    assert registry["group_count"] == 649
    assert registry["by_family"] == {
        "uniform": 252,
        "layered": 257,
        "marmousi": 140,
    }
    assert registry["selected_stored_group_hits"] == {
        "uniform": False,
        "layered": False,
    }
    rows = confirm.fresh_confirmation_census(manifest)
    assert Counter(row["family"] for row in rows) == Counter(
        {family: 1 for family in confirm.FAMILIES}
    )
    assert len({row["group_id"] for row in rows}) == 3
    assert all(row["future_truth_opened_before_freeze"] is False for row in rows)


def test_synthetic_spec_and_parent_inputs_are_deterministic() -> None:
    record, loaded, fine, metadata = confirm.synthetic_input()
    assert confirm.canonical_sha256(confirm.SYNTHETIC_SPEC) == (
        confirm.EXPECTED_SYNTHETIC_SPEC_SHA256
    )
    assert record.split == "train"
    assert record.source_index == -1
    assert metadata["crop_sha256"] == confirm.EXPECTED_SYNTHETIC_CROP_SHA256
    assert loaded.nontruth_input_sha256 == confirm.EXPECTED_SYNTHETIC_NONTRUTH_SHA256
    assert fine.shape == (401, 401)
    assert loaded.velocity_mps.shape == (201, 201)
    assert loaded.source_map.shape == (201, 201)
    assert float(loaded.source_map.sum()) == pytest.approx(1.0)
    assert torch.isfinite(torch.from_numpy(fine)).all()


def test_generator_contract_is_free_surface_and_three_sided_cpml() -> None:
    config = confirm.generator_config()
    assert config["boundaries"] == {
        "top": "free_surface_dirichlet",
        "left": "cpml",
        "right": "cpml",
        "bottom": "cpml",
        "npml": 40,
        "cpml_target_reflection": 1.0e-8,
        "cpml_polynomial_order": 3,
        "cpml_outside_physical_domain": True,
        "kappa_max": 3.0,
        "minimum_frequency_hz": 8.0,
        "alpha_max_definition": "pi_times_minimum_frequency",
    }
    assert config["time"]["dt_used_s"] == 0.000125
    assert config["time"]["dt_out_s"] == 0.0025
    assert config["time"]["nt_out"] == 401


def test_synthetic_crop_and_source_do_not_collide_with_manifest() -> None:
    import h5py
    import numpy as np

    spec = confirm.SYNTHETIC_SPEC
    with h5py.File(confirm.parent_runtime.SOURCE_H5_PATH, "r", swmr=True) as handle:
        groups = set(handle["group_id"].asstr()[:].tolist())
        samples = set(handle["sample_id"].asstr()[:].tolist())
        triples = set(
            zip(
                np.asarray(handle["source_x_m"][:]).round(12),
                np.asarray(handle["source_z_m"][:]).round(12),
                np.asarray(handle["source_f0_hz"][:]).round(12),
            )
        )
        crop_pairs = set(
            zip(
                np.asarray(handle["crop_x0_m"][:]).round(12),
                np.asarray(handle["crop_z0_m"][:]).round(12),
            )
        )
    assert spec["group_id"] not in groups
    assert spec["sample_id"] not in samples
    assert (
        spec["source_x_m"],
        spec["source_z_m"],
        spec["source_f0_hz"],
    ) not in triples
    assert (spec["crop_x0_m"], spec["crop_z0_m"]) not in crop_pairs


def test_no_field_or_checkpoint_serialization_api() -> None:
    source = confirm.SCRIPT_PATH.read_text(encoding="utf8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "save" not in calls
    assert "savez" not in calls
    assert "savez_compressed" not in calls
    assert "memmap" not in calls
    assert "tofile" not in calls
    assert "torch.save(" not in source
    assert "save_checkpoint_atomic" not in source


def test_only_declared_terminal_paths_and_resources() -> None:
    assert confirm.PREREGISTRATION_PATH.name.endswith(
        "fresh3_confirmation_v1_preregistration_20260826.json"
    )
    assert confirm.RESULT_PATH.name.endswith(
        "fresh3_confirmation_v1_20260826.json"
    )
    assert confirm.PREREGISTRATION_PATH.parent == confirm.PROJECT_ROOT / "results"
    assert confirm.RESULT_PATH.parent == confirm.PROJECT_ROOT / "results"
    assert confirm.MIN_FREE_BYTES == 2 * 1024**3
    assert confirm.MAX_OUTPUT_BYTES == 10 * 1024**2
    assert confirm.PEAK_CUDA_BYTES_MAXIMUM == int(23.5 * 1024**3)
