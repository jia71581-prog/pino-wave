from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/saved_time_v4/generated"
RESULTS_DIR = ROOT / "results"
CONFIGS = {
    "r5e": CONFIG_DIR
    / "local_field_w128_temporal_latent_a3_rank32_r5e_true_metric_aligned_time_train_diag.yaml",
    "r5f": CONFIG_DIR
    / "local_field_w128_temporal_latent_a3_rank32_r5f_uniform_random_aligned_time_train_diag.yaml",
    "r5g": CONFIG_DIR
    / "local_field_w128_temporal_latent_a3_rank32_r5g_dropout005_aligned_time_train_diag.yaml",
}
PREREGISTRATIONS = {
    "r5e": RESULTS_DIR / "r5e_true_metric_aligned_time_train_preregistration_20260813.json",
    "r5f": RESULTS_DIR / "r5f_uniform_random_aligned_time_train_preregistration_20260813.json",
    "r5g": RESULTS_DIR / "r5g_dropout005_aligned_time_train_preregistration_20260813.json",
}


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            nested = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(child, nested))
        return output
    return {prefix: value}


def _config_differences(control, candidate):
    left = dict(control)
    right = dict(candidate)
    left.pop("artifact_dir")
    right.pop("artifact_dir")
    left = _flatten(left)
    right = _flatten(right)
    return {key for key in set(left) | set(right) if left.get(key) != right.get(key)}


def test_aligned_time_ablations_are_train_only_and_share_the_parent():
    configs = {name: yaml.safe_load(path.read_text()) for name, path in CONFIGS.items()}
    parents = {config["parent_checkpoint"] for config in configs.values()}
    assert len(parents) == 1

    for config in configs.values():
        assert config["parent_checkpoint_selection_split"] == "train"
        assert config["time_policy"] == "fixed_train_gate"
        assert config["training_frames_per_record"] == 24
        assert config["validation"]["frames_per_record"] == 32
        assert config["epoch_validation_control"]["evaluation_split"] == "train"
        assert config["epoch_validation_control"]["time_selector_seed_offset"] == 0


def test_r5f_and_r5g_each_change_exactly_one_conceptual_variable():
    configs = {name: yaml.safe_load(path.read_text()) for name, path in CONFIGS.items()}
    assert _config_differences(configs["r5e"], configs["r5f"]) == {
        "adaptive_sampling.record_axis_strategy"
    }
    assert configs["r5f"]["adaptive_sampling"]["record_axis_strategy"] == "uniform"

    assert _config_differences(configs["r5e"], configs["r5g"]) == {
        "variant_overrides.band_adapter_dropout"
    }
    assert configs["r5g"]["variant_overrides"]["band_adapter_dropout"] == 0.05


def test_preregistered_config_digests_match():
    for name, config_path in CONFIGS.items():
        preregistration = json.loads(PREREGISTRATIONS[name].read_text())
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert preregistration["bindings"]["config"] == str(
            config_path.relative_to(ROOT)
        )
        assert preregistration["bindings"]["config_sha256"] == digest
