from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from patch_deeponet_baseline.model import PatchDeepONet, PatchDeepONetConfig
from scripts.train_patch_deeponet_baseline import (
    CHECKPOINT_SCHEMA,
    ENTRY_SCHEMA,
    EVALUATION_ENERGY_FLOOR_FRACTION,
    IDENTITY_SCHEMA,
    _load_resume_checkpoint,
    _resume_position,
    _require_full_authorization,
    _require_appearance_time_policy,
    audit_config,
)


CONFIG = Path("configs/baselines/patch_deeponet_fixed_train_gate_20260813.yaml")


def test_evaluation_energy_floor_is_valid() -> None:
    assert 0.0 < EVALUATION_ENERGY_FLOOR_FRACTION <= 1.0


def test_training_rejects_fixed_repeated_time_panel() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    with pytest.raises(ValueError, match="appearance16"):
        _require_appearance_time_policy(config)
    config["schedule"]["time_policy"] = "appearance16"
    _require_appearance_time_policy(config)


def test_registered_training_config_is_train_only_and_launch_locked() -> None:
    report = audit_config(CONFIG)
    assert report["status"] == "pass"
    assert report["launch_authorized"] is False
    assert report["selection_split"] == "train"
    assert report["frequency_generalization_permitted"] is False
    assert report["parameter_match"]["within_tolerance"] is True
    assert all(row["record_count"] == 2240 for row in report["epoch_audits"])


def test_full_launch_fails_closed_without_authorization(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    with pytest.raises(PermissionError, match="locked"):
        _require_full_authorization(config, None)

    config["launch_authorized"] = True
    report = tmp_path / "entry_gate.json"
    report.write_text(
        json.dumps({"schema": ENTRY_SCHEMA, "status": "fail"}), encoding="utf8"
    )
    with pytest.raises(ValueError, match="has not passed"):
        _require_full_authorization(config, report)

    report.write_text(
        json.dumps(
            {
                "schema": ENTRY_SCHEMA,
                "status": "pass",
                "selection_split": "train",
                "config_sha256": "a" * 64,
                "manifest_digest": "b" * 64,
                "run_digest": "c" * 64,
            }
        ),
        encoding="utf8",
    )
    _require_full_authorization(
        config,
        report,
        expected_config_sha256="a" * 64,
        expected_manifest_digest="b" * 64,
    )
    with pytest.raises(ValueError, match="manifest binding"):
        _require_full_authorization(
            config,
            report,
            expected_config_sha256="a" * 64,
            expected_manifest_digest="d" * 64,
        )


def test_failed_entry_requires_explicit_comparison_override(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    config["launch_authorized"] = True
    report = tmp_path / "failed_entry_gate.json"
    report.write_text(
        json.dumps(
            {
                "schema": ENTRY_SCHEMA,
                "status": "fail",
                "selection_split": "train",
                "config_sha256": "a" * 64,
                "manifest_digest": "b" * 64,
                "run_digest": "c" * 64,
            }
        ),
        encoding="utf8",
    )
    with pytest.raises(ValueError, match="has not passed"):
        _require_full_authorization(
            config,
            report,
            expected_config_sha256="a" * 64,
            expected_manifest_digest="b" * 64,
        )
    accepted = _require_full_authorization(
        config,
        report,
        expected_config_sha256="a" * 64,
        expected_manifest_digest="b" * 64,
        allow_failed_comparison=True,
    )
    assert accepted["status"] == "fail"


def test_resume_checkpoint_requires_matching_run_identity(tmp_path: Path) -> None:
    config = PatchDeepONetConfig(
        branch_width=16,
        branch_blocks=1,
        branch_global_width=32,
        branch_bottleneck_width=8,
        latent_dim=16,
        trunk_width=16,
        trunk_depth=3,
        fourier_bands=2,
        local_residual_width=8,
        target_parameters=20_000,
        parameter_tolerance_fraction=0.01,
    )
    source_model = PatchDeepONet(config)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=3.0e-4)
    identity = {
        "schema": IDENTITY_SCHEMA,
        "config_sha256": "a" * 64,
        "manifest_digest": "b" * 64,
        "selection_split": "train",
        "run_digest": "c" * 64,
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "manifest_digest": "b" * 64,
        "selection_split": "train",
        "run_digest": "c" * 64,
        "epoch": 1,
        "global_step": 140,
        "selection_metrics": {"aggregate_relative_l2": 0.7},
        "model_state": source_model.state_dict(),
        "optimizer_state": source_optimizer.state_dict(),
    }
    (tmp_path / "run_identity.json").write_text(json.dumps(identity), encoding="utf8")
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    target_model = PatchDeepONet(config)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=3.0e-4)
    loaded, parent = _load_resume_checkpoint(
        checkpoint_path,
        model=target_model,
        optimizer=target_optimizer,
        expected_config_sha256="a" * 64,
        expected_manifest_digest="b" * 64,
        device=torch.device("cpu"),
    )
    assert loaded["global_step"] == 140
    assert parent["run_digest"] == "c" * 64
    assert _resume_position(loaded) == (1, 0)


def test_rolling_resume_restores_mid_epoch_cursor_and_incumbent(
    tmp_path: Path,
) -> None:
    config = PatchDeepONetConfig(
        branch_width=16,
        branch_blocks=1,
        branch_global_width=32,
        branch_bottleneck_width=8,
        latent_dim=16,
        trunk_width=16,
        trunk_depth=3,
        fourier_bands=2,
        local_residual_width=8,
        target_parameters=20_000,
        parameter_tolerance_fraction=0.01,
    )
    source_model = PatchDeepONet(config)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=3.0e-4)
    identity = {
        "schema": IDENTITY_SCHEMA,
        "config_sha256": "a" * 64,
        "manifest_digest": "b" * 64,
        "selection_split": "train",
        "run_digest": "c" * 64,
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_kind": "rolling",
        "manifest_digest": "b" * 64,
        "selection_split": "train",
        "run_digest": "c" * 64,
        "epoch": 3,
        "update_in_epoch": 20,
        "global_step": 300,
        "model_state": source_model.state_dict(),
        "optimizer_state": source_optimizer.state_dict(),
    }
    (tmp_path / "run_identity.json").write_text(json.dumps(identity), encoding="utf8")
    (tmp_path / "best.json").write_text(
        json.dumps(
            {
                "epoch": 2,
                "score": 0.6,
                "metrics": {"aggregate_relative_l2": 0.6},
            }
        ),
        encoding="utf8",
    )
    checkpoint_path = tmp_path / "latest.pt"
    torch.save(checkpoint, checkpoint_path)
    target_model = PatchDeepONet(config)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=3.0e-4)
    loaded, _ = _load_resume_checkpoint(
        checkpoint_path,
        model=target_model,
        optimizer=target_optimizer,
        expected_config_sha256="a" * 64,
        expected_manifest_digest="b" * 64,
        device=torch.device("cpu"),
    )
    assert loaded["selection_metrics"]["aggregate_relative_l2"] == 0.6
    assert loaded["incumbent_best_epoch"] == 2
    assert len(loaded["incumbent_best_metadata_sha256"]) == 64
    assert _resume_position(loaded) == (2, 20)
