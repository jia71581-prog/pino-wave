from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts.freeze_ais_recipe import InjectedFreezeFailure, freeze_recipe
from scripts.generate_ais_configs import (
    CONFIG_FILENAMES,
    LOSS_ARMS,
    LOSS_NAMES,
    build_configs,
)
from scripts.train_ais_mqfno import _load_configuration, _phase_config
from fno_acoustic.train import build_training_model
from fno_acoustic.train import (
    _sample_spatial_indices,
    _spatial_importance_sampling_enabled,
    _spatial_importance_loss,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_ais_configs.py"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison(cfg: dict) -> dict:
    return {
        "data_path": cfg["data"]["path"],
        "split_manifest": cfg["data"]["split_manifest"],
        "normalization_stats": cfg["normalization"]["stats_path"],
        "time_steps": cfg["sampling"]["max_time_steps"],
        "receiver": cfg["receiver"],
        "optimizer_updates": cfg["experiment"]["optimizer_updates"],
        "physical_scene_draws": cfg["experiment"]["physical_scene_draws"],
        "gpu_hour_budget": cfg["experiment"]["gpu_hour_budget"],
    }


def test_generator_materializes_exact_declared_configs_from_isolated_copies(tmp_path: Path) -> None:
    configs = build_configs()
    assert tuple(configs) == CONFIG_FILENAMES
    first = deepcopy(configs[CONFIG_FILENAMES[0]])
    configs[CONFIG_FILENAMES[1]]["data"]["path"] = "mutated"
    assert configs[CONFIG_FILENAMES[0]] == first
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--output-dir", str(tmp_path)],
        cwd="/", text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert {path.name for path in tmp_path.glob("*.yaml")} == set(CONFIG_FILENAMES)


def test_checked_in_configs_match_generator_and_check_detects_drift(tmp_path: Path) -> None:
    assert subprocess.run(
        [sys.executable, str(GENERATOR), "--output-dir", str(ROOT / "configs"), "--check"],
        cwd="/", capture_output=True,
    ).returncode == 0
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output-dir", str(tmp_path)],
        cwd="/", check=True,
    )
    target = tmp_path / CONFIG_FILENAMES[0]
    target.write_text(target.read_text() + "# drift\n")
    assert subprocess.run(
        [sys.executable, str(GENERATOR), "--output-dir", str(tmp_path), "--check"],
        cwd="/", capture_output=True,
    ).returncode == 2


def test_exploration_configs_share_contract_and_record_actual_label_cost() -> None:
    configs = build_configs()
    names = (
        "ais_mqfno_64x160_b1_uniform.yaml", "ais_mqfno_64x160_s_static_hh.yaml",
        "ais_mqfno_64x160_a_residual_uncorrected.yaml",
        "ais_mqfno_64x160_ai_adaptive_hh.yaml",
    )
    arms = [configs[name] for name in names]
    expected = _comparison(arms[0])
    assert all(_comparison(cfg) == expected for cfg in arms)
    assert all(cfg["sampling"]["max_time_steps"] == 160 for cfg in configs.values())
    for cfg in configs.values():
        schedule = cfg["experiment"]["label_site_schedule"]
        assert cfg["experiment"]["full160_label_sites"] == sum(u * q for u, q in schedule)
    diagnostic = arms[2]
    assert diagnostic["experiment"]["diagnostic_only"] is True
    assert diagnostic["loss"]["hh_reweight"] is False


def test_all_candidate_configs_satisfy_the_task8_training_schema(tmp_path: Path) -> None:
    for name, payload in build_configs().items():
        if name.startswith("factorized_fno"):
            continue
        path = tmp_path / name
        path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        assert _load_configuration(path) == payload


def test_every_generated_and_frozen_candidate_phase_resolves_its_loss_profile(
    tmp_path: Path,
) -> None:
    for name, config in build_configs().items():
        if name.startswith("factorized_fno"):
            continue
        for profile_name, weights in LOSS_ARMS.items():
            assert config["loss_profiles"][profile_name] == {
                f"{loss_name}_weight": weight
                for loss_name, weight in zip(LOSS_NAMES, weights, strict=True)
            }
        assert config["loss_profiles"]["selected_frozen_loss"] == config["loss"]
        for phase_index, phase in enumerate(config["train"]["phases"]):
            _phase_config(config, phase, phase_index=phase_index)

    selection, _, _ = _selection(tmp_path)
    frozen = freeze_recipe(selection, tmp_path / "frozen")
    for path in (frozen.config_64, frozen.config_128, frozen.config_400):
        config = _load(path)
        assert config["loss_profiles"]["selected_frozen_loss"] == config["loss"]
        for phase_index, phase in enumerate(config["train"]["phases"]):
            resolved = _phase_config(config, phase, phase_index=phase_index)
            expected = config["loss_profiles"][phase["loss_profile"]]
            assert all(resolved["loss"][key] == value for key, value in expected.items())


def test_b0_configs_are_consumable_by_legacy_train_pino_model_schema() -> None:
    configs = build_configs()
    for name in ("factorized_fno_400x400x160_b0.yaml",
                 "factorized_fno_400x400x160_b0_smoke.yaml"):
        cfg = configs[name]
        assert "loss_profiles" not in cfg
        assert all(key in cfg["train"] for key in (
            "epochs", "checkpoint_dir", "log_dir", "learning_rate", "weight_decay"
        ))
        assert all(key in cfg["data"] for key in (
            "raw_hdf5", "velocity_key", "wavefield_key", "input_features",
            "wavefield_axes", "velocity_axes",
        ))
        model, model_cfg = build_training_model(cfg)
        assert model_cfg["name"] == "factorized_fno"
        assert model.in_features == 3
        sampling = cfg["train"]["spatial_importance_sampling"]
        assert sampling == {
            "enabled": True,
            "pixels_per_sample": 2048,
            "uniform_fraction": 1.0,
            "reweight_loss": True,
        }
        assert _spatial_importance_sampling_enabled(cfg)
    formal = configs["factorized_fno_400x400x160_b0.yaml"]
    assert formal["train"]["epochs"] == 6
    assert formal["train"]["max_train_batches"] is None
    assert formal["train"]["checkpoint_dir"] == (
        "artifacts/ais_mqfno_full160_native400_20260714/baselines/b0/checkpoints"
    )


def test_named_mixture_and_auxiliary_order_are_exact() -> None:
    configs = build_configs()
    adaptive = configs["ais_mqfno_64x160_ai_adaptive_hh.yaml"]
    assert adaptive["sampler"]["mixture"] == {
        "uniform": 0.30, "interface": 0.20, "source_wavefront": 0.15,
        "residual": 0.20, "edge": 0.05, "receiver": 0.10,
    }
    ordered = [configs[f"ais_mqfno_64x160_ai_{name}.yaml"] for name in
               ("receiver", "phase", "local_spectrum", "energy")]
    fields = ("receiver", "phase", "local_spectrum", "energy")
    for index, cfg in enumerate(ordered, 1):
        enabled = tuple(name for name in fields if cfg["loss"][f"{name}_weight"] > 0)
        assert enabled == fields[:index]
        assert cfg["loss"]["pde_weight"] == cfg["loss"]["drp_weight"] == 0.0


def _selection(tmp_path: Path) -> tuple[Path, Path, Path]:
    selected = deepcopy(build_configs()["ais_mqfno_64x160_ai_energy.yaml"])
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"train": [300], "val": [301, 302],
                                 "test": list(range(250))}), encoding="utf-8")
    selected["data"]["split_manifest"] = str(split)
    config = tmp_path / "selected.yaml"
    config.write_text(yaml.safe_dump(selected, sort_keys=True), encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({
        "selected_config": str(config), "selected_checkpoint": str(checkpoint),
        "selected_config_sha256": _sha(config),
        "selected_checkpoint_sha256": _sha(checkpoint),
        "diagnostic_only": False,
    }), encoding="utf-8")
    return manifest, config, checkpoint


def test_freeze_snapshots_sources_and_materializes_fair_stage_budget(tmp_path: Path) -> None:
    selection, config, checkpoint = _selection(tmp_path)
    before = checkpoint.read_bytes()
    outputs = freeze_recipe(selection, tmp_path / "frozen")
    assert outputs.manifest["selected_config_sha256"] == _sha(config)
    assert outputs.manifest["selected_checkpoint_sha256"] == _sha(checkpoint)
    assert outputs.manifest["split_manifest"]["sha256"] == _sha(tmp_path / "splits.json")
    assert outputs.manifest["files"]["split_manifest.json"] == _sha(
        outputs.root / "split_manifest.json"
    )
    assert checkpoint.read_bytes() == before
    cfg64, cfg128, cfg400, b0 = map(_load, (
        outputs.config_64, outputs.config_128, outputs.config_400, outputs.config_b0))
    assert cfg128["loss"] == cfg400["loss"] == cfg64["loss"]
    assert [phase["optimizer_updates"] for phase in cfg64["train"]["phases"]] == [2000, 1000]
    assert cfg64["train"]["phases"][1]["init_from"] == "phase_best"
    assert cfg128["train"]["phases"][0]["init_from"] == "external_checkpoint"
    assert cfg400["train"]["phases"][0]["optimizer_updates"] == 6000
    assert b0["model"]["name"] == "factorized_fno"
    assert "loss_profiles" not in b0
    assert b0["train"]["epochs"] == 12000
    assert b0["experiment"]["label_site_schedule"] == cfg400["experiment"]["label_site_schedule"]


def test_b0_uniform_hh_uses_actual_sampling_and_loss_helpers() -> None:
    import torch

    cfg = build_configs()["factorized_fno_400x400x160_b0.yaml"]
    indices, probabilities = _sample_spatial_indices(
        400, 400, None, epoch=0, global_step=0, config=cfg, device=torch.device("cpu")
    )
    assert indices.shape == probabilities.shape == (2048,)
    assert torch.allclose(probabilities, torch.full_like(probabilities, 1.0 / 160000))
    pred = torch.zeros(1, 400, 400, 2)
    target = torch.ones_like(pred)
    loss, _ = _spatial_importance_loss(
        pred, target, indices, probabilities, 160000, cfg["loss"], reweight=True
    )
    assert torch.isfinite(loss)


def test_freeze_rejects_train_count_that_cannot_exactly_reach_12000_updates(tmp_path: Path) -> None:
    selection, selected_path, _ = _selection(tmp_path)
    selected = yaml.safe_load(selected_path.read_text())
    split = Path(selected["data"]["split_manifest"])
    payload = json.loads(split.read_text())
    payload["train"] = list(range(7001))
    split.write_text(json.dumps(payload))
    selected_path.write_text(yaml.safe_dump(selected, sort_keys=True))
    manifest = json.loads(selection.read_text())
    manifest["selected_config_sha256"] = _sha(selected_path)
    selection.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="12000.*divisible|divide.*12000"):
        freeze_recipe(selection, tmp_path / "frozen")


def test_freeze_failure_cleans_staging_and_existing_recipe_rehashes_every_file(tmp_path: Path) -> None:
    selection, _, checkpoint = _selection(tmp_path)
    original = _sha(checkpoint)
    with pytest.raises(InjectedFreezeFailure):
        freeze_recipe(selection, tmp_path / "failed", fail_before_publish=True)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.*.tmp"))
    assert _sha(checkpoint) == original
    outputs = freeze_recipe(selection, tmp_path / "frozen")
    assert freeze_recipe(selection, tmp_path / "frozen").manifest == outputs.manifest
    outputs.config_128.write_text(outputs.config_128.read_text() + "# tamper\n")
    with pytest.raises(ValueError, match="hash"):
        freeze_recipe(selection, tmp_path / "frozen")


def test_freeze_does_not_trust_a_rewritten_recipe_inventory(tmp_path: Path) -> None:
    selection, _, _ = _selection(tmp_path)
    outputs = freeze_recipe(selection, tmp_path / "frozen")
    outputs.config_128.write_text(outputs.config_128.read_text() + "# tamper\n")
    recipe = json.loads(outputs.manifest_path.read_text())
    recipe["files"]["config_128.yaml"] = _sha(outputs.config_128)
    outputs.manifest_path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="hash|identical"):
        freeze_recipe(selection, tmp_path / "frozen")


def test_frozen_configs_use_immutable_split_and_normalization_copies(tmp_path: Path) -> None:
    selection, selected_path, _ = _selection(tmp_path)
    selected = yaml.safe_load(selected_path.read_text())
    source_split = Path(selected["data"]["split_manifest"])
    source_norm = Path(selected["normalization"]["stats_path"])
    original_norm = source_norm.read_bytes()
    outputs = freeze_recipe(selection, tmp_path / "frozen")
    source_split.write_text(json.dumps({"train": [999], "val": [], "test": []}))
    frozen = _load(outputs.config_400)
    assert Path(frozen["data"]["split_manifest"]) == outputs.root / "split_manifest.json"
    assert Path(frozen["normalization"]["stats_path"]) == outputs.root / "normalization_stats.json"
    assert json.loads((outputs.root / "split_manifest.json").read_text())["test"] == list(range(250))
    assert (outputs.root / "normalization_stats.json").read_bytes() == original_norm
