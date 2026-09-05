"""Contracts for the eight registered AIS normalization-v2 candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from fno_acoustic.model_ais_mqfno import AISMQFNO
from scripts.generate_ais_v2_configs import build_v2_configs, main
from scripts.train_ais_mqfno import (
    _load_configuration,
    _load_normalization_binding,
    _model_kwargs,
)


EXPECTED_FILENAMES = {
    "n0_norm.yaml",
    "n1_wide.yaml",
    "n2_spatial.yaml",
    "n3_temporal.yaml",
    "n4_large_local.yaml",
    "n5_multi_local.yaml",
    "n6_dispersion.yaml",
    "n7_multi_dispersion.yaml",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_FILENAMES = {
    "gate_o_sample_0002_sites_2048.json",
    "fixed_validation_sites_2048.json",
}

EXPECTED_MODELS = {
    "n0_norm.yaml": {},
    "n1_wide.yaml": {"spatial_width": 32, "local_dim": 24, "fusion_dim": 48},
    "n2_spatial.yaml": {"spatial_width": 32, "spatial_modes": 32},
    "n3_temporal.yaml": {"temporal_modes": 48, "fusion_dim": 48},
    "n4_large_local.yaml": {"halo_size": 25, "local_dim": 24},
    "n5_multi_local.yaml": {
        "local_encoder_kind": "multiscale_9_25",
        "halo_size": 25,
        "local_dim": 24,
        "fusion_dim": 48,
    },
    "n6_dispersion.yaml": {"dispersion_head": "phase_residual_24"},
    "n7_multi_dispersion.yaml": {
        "local_encoder_kind": "multiscale_9_25",
        "halo_size": 25,
        "local_dim": 24,
        "fusion_dim": 48,
        "dispersion_head": "phase_residual_24",
    },
}

BASE_MODEL = {
    "spatial_width": 24,
    "spatial_modes": 24,
    "temporal_modes": 32,
    "local_dim": 16,
    "fusion_dim": 32,
    "halo_size": 17,
    "local_encoder_kind": "single",
    "dispersion_head": "none",
}


def test_generator_materializes_exactly_eight_candidates() -> None:
    configs = build_v2_configs()

    assert set(configs) == EXPECTED_FILENAMES
    assert len(configs) == 8
    assert all(
        config["normalization"]["contract"] == "ais_normalization_v2"
        for config in configs.values()
    )


def test_candidate_models_match_the_eight_registered_structures_exactly() -> None:
    configs = build_v2_configs()

    for filename, overrides in EXPECTED_MODELS.items():
        expected = {**BASE_MODEL, **overrides}
        actual = configs[filename]["model"]
        for key, value in expected.items():
            assert actual[key] == value, (filename, key)
        assert "velocity_mean" not in actual
        assert "velocity_std" not in actual

    canonical = {
        yaml.safe_dump(config["model"], sort_keys=True) for config in configs.values()
    }
    assert len(canonical) == 8


def test_candidates_share_frozen_data_normalization_and_training_contracts() -> None:
    configs = build_v2_configs()
    first = next(iter(configs.values()))
    shared = {
        "seed": first["seed"],
        "data": first["data"],
        "normalization": first["normalization"],
        "sampling": first["sampling"],
        "train": first["train"],
        "loss": first["loss"],
        "sampler": first["sampler"],
    }

    for config in configs.values():
        assert {key: config[key] for key in shared} == shared
        assert config["sampling"]["max_time_steps"] == 160
        assert config["train"]["query_sites_per_scene"] == 2048
        assert config["train"]["optimizer"] == "adamw"
        assert config["train"]["scheduler"] == "cosine_by_update"
        assert config["train"]["phases"][0]["optimizer_updates"] == 3000
        assert config["normalization"]["static_features"] == [
            "v_hat",
            "source_hat",
            "dv_dxi",
            "dv_dzeta",
            "slow_contrast",
        ]
        assert config["normalization"]["time_feature"] == "tau"
        assert config["normalization"]["target"] == "u_hat"


def test_screen_registration_binds_train_sample_sites_order_and_halving_budgets() -> None:
    configs = build_v2_configs()
    first_screen = next(iter(configs.values()))["screen"]
    common_keys = {
        "gate_o_sample_id",
        "gate_o_site_manifest",
        "gate_o_site_manifest_sha256",
        "gate_o_fixed_unique_sites",
        "validation_site_manifest",
        "validation_site_manifest_sha256",
        "validation_sites_per_scene",
        "scene_order",
        "budgets",
    }
    common = {key: first_screen[key] for key in common_keys}

    split_path = Path(next(iter(configs.values()))["data"]["split_manifest"])
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    assert common["gate_o_sample_id"] in splits["train"]
    assert common["gate_o_sample_id"] not in splits["val"] + splits["test"]
    assert common["gate_o_fixed_unique_sites"] == 2048
    assert common["validation_sites_per_scene"] == 2048
    for key in ("gate_o_site_manifest_sha256", "validation_site_manifest_sha256"):
        assert len(common[key]) == 64
        int(common[key], 16)

    budgets = common["budgets"]
    assert [budgets[name]["total_optimizer_updates"] for name in ("gate_o", "h1", "h2", "h3")] == [400, 600, 1500, 3000]
    assert budgets["gate_o"]["initialization"] == "restart_from_registered_seed"
    assert budgets["gate_o"]["separate_output"] is True
    assert budgets["h1"]["initialization"] == "restart_from_registered_seed"
    assert budgets["h1"]["reuse_gate_o_weights"] is False
    assert budgets["h2"]["resume_from"] == "h1_update_600_last"
    assert budgets["h2"]["additional_optimizer_updates"] == 900
    assert budgets["h3"]["resume_from"] == "h2_update_1500_last"
    assert budgets["h3"]["additional_optimizer_updates"] == 1500

    for filename, config in configs.items():
        assert config["screen"]["candidate_id"] == filename.split("_", 1)[0].upper()
        assert {key: config["screen"][key] for key in common_keys} == common


def test_materialized_site_manifests_match_hashes_splits_and_exact_site_contracts() -> None:
    config = next(iter(build_v2_configs().values()))
    split_path = PROJECT_ROOT / config["data"]["split_manifest"]
    splits = json.loads(split_path.read_text(encoding="utf-8"))

    registrations = {
        "gate_o": (
            config["screen"]["gate_o_site_manifest"],
            config["screen"]["gate_o_site_manifest_sha256"],
        ),
        "validation": (
            config["screen"]["validation_site_manifest"],
            config["screen"]["validation_site_manifest_sha256"],
        ),
    }
    payloads: dict[str, dict[str, object]] = {}
    for name, (relative_path, expected_sha) in registrations.items():
        path = PROJECT_ROOT / relative_path
        assert path.is_file(), name
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha
        assert raw.endswith(b"\n")
        payload = json.loads(raw)
        payloads[name] = payload
        assert payload["schema"] == "ais_fixed_spatial_sites"
        assert payload["schema_version"] == 1
        assert payload["split_manifest_sha256"] == config["data"]["split_manifest_sha256"]
        assert payload["grid"] == {
            "height": 64,
            "index_order": "row_major_x_then_z",
            "width": 64,
        }
        sites = payload["site_indices"]
        assert isinstance(sites, list) and len(sites) == 2048
        assert len(set(sites)) == 2048
        assert all(type(site) is int and 0 <= site < 64 * 64 for site in sites)

    gate_o = payloads["gate_o"]
    assert gate_o["purpose"] == "gate_o_train_representability"
    assert gate_o["split"] == "train"
    assert gate_o["sample_id"] == config["screen"]["gate_o_sample_id"] == 2
    assert gate_o["sample_id"] in splits["train"]

    validation = payloads["validation"]
    assert validation["purpose"] == "fixed_validation_sites"
    assert validation["split"] == "val"
    assert validation["sites_shared_across_scenes"] is True
    assert validation["sample_ids"] == splits["val"]
    assert set(validation["sample_ids"]) == set(splits["val"])


def test_registered_split_and_normalization_hashes_match_frozen_files() -> None:
    config = next(iter(build_v2_configs().values()))
    for section, path_key, hash_key in (
        ("data", "split_manifest", "split_manifest_sha256"),
        ("normalization", "stats_path", "stats_sha256"),
    ):
        path = Path(config[section][path_key])
        assert config[section][hash_key] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_configs_are_deterministic_atomic_checkable_and_cli_valid(
    tmp_path: Path,
) -> None:
    first = build_v2_configs()
    second = build_v2_configs()
    assert first == second
    assert first is not second

    assert main(["--output-dir", str(tmp_path)]) == 0
    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_FILENAMES | {
        "registration"
    }
    assert {
        path.name for path in (tmp_path / "registration").iterdir()
    } == REGISTRATION_FILENAMES
    before_check = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert main(["--output-dir", str(tmp_path), "--check"]) == 0
    after_check = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after_check == before_check

    for path in sorted(tmp_path.glob("*.yaml")):
        loaded = _load_configuration(path)
        binding = _load_normalization_binding(loaded)
        AISMQFNO(**_model_kwargs(loaded, binding))

    drifted = tmp_path / "n0_norm.yaml"
    drifted.write_bytes(drifted.read_bytes() + b"\n")
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path), "--check"])


def test_check_is_read_only_and_generator_rejects_a_ninth_yaml(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(SystemExit):
        main(["--output-dir", str(missing), "--check"])
    assert not missing.exists()

    output = tmp_path / "generated"
    assert main(["--output-dir", str(output)]) == 0
    (output / "n8_forbidden.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(["--output-dir", str(output)])

    (output / "n8_forbidden.yaml").unlink()
    (output / "registration" / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(["--output-dir", str(output), "--check"])
