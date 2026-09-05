from __future__ import annotations

from types import SimpleNamespace
import subprocess
import sys

import torch
from torch import nn

from saved_time_phase_operator_v4.instance_adaptation.adapters import (
    ADAPTER_SCHEMA_VERSION,
    OnsetAdaptedV5,
)
from scripts.run_v5_instance_adaptation import (
    _apply_parent_correction_policy,
    _load_conditioner,
    build_instance_manifest,
)


def _manifest():
    rows = []
    for split in ("train", "validation"):
        for family in ("uniform", "layered", "marmousi"):
            for index in range(3):
                rows.append(
                    SimpleNamespace(
                        split=split,
                        medium_type=family,
                        group_id=f"{split}-{family}-{index}",
                        sample_id=f"{split}-{family}-{index}",
                    )
                )
    return SimpleNamespace(records=tuple(rows))


def test_cli_manifest_has_three_families_and_nine_instances():
    manifest = build_instance_manifest(_manifest(), seed=17)
    assert {row.medium_type for row in manifest} == {"uniform", "layered", "marmousi"}
    counts = {family: sum(row.medium_type == family for row in manifest) for family in ("uniform", "layered", "marmousi")}
    assert counts == {"uniform": 3, "layered": 3, "marmousi": 3}


def test_cli_manifest_order_is_reproducible():
    left = build_instance_manifest(_manifest(), seed=17)
    right = build_instance_manifest(_manifest(), seed=17)
    assert tuple(row.sample_id for row in left) == tuple(row.sample_id for row in right)


def test_cli_manifest_defaults_to_validation_and_can_select_train():
    validation = build_instance_manifest(_manifest(), seed=17)
    train = build_instance_manifest(_manifest(), seed=17, split="train")
    assert {row.split for row in validation} == {"validation"}
    assert {row.split for row in train} == {"train"}
    assert {row.sample_id for row in validation}.isdisjoint(
        row.sample_id for row in train
    )


def test_cli_manifest_rejects_test_split_for_tuning():
    try:
        build_instance_manifest(_manifest(), split="test_id")
    except ValueError as error:
        assert "train or validation" in str(error)
    else:
        raise AssertionError("test_id must never be available for adapter tuning")


def test_conditioner_checkpoint_is_loaded_and_manifest_guarded(tmp_path):
    parent = nn.Conv2d(1, 1, kernel_size=1)
    source = OnsetAdaptedV5(parent, latent_dim=8, lora_rank=2)
    checkpoint = tmp_path / "conditioner.pt"
    torch.save(
        {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "conditioner_state": source.conditioner.state_dict(),
            "chonknoris_state": source.chonknoris.state_dict(),
            "manifest_digest": "expected",
            "parent_checkpoint": "/tmp/phase4b-parent.pt",
            "future_truth_used_only_for_train_episode": True,
        },
        checkpoint,
    )
    target = OnsetAdaptedV5(nn.Conv2d(1, 1, kernel_size=1), latent_dim=8, lora_rank=2)
    info = _load_conditioner(target, checkpoint, torch.device("cpu"), expected_manifest_digest="expected")
    assert info["future_truth_used_only_for_train_episode"] is True
    assert info["chonknoris_state_loaded"] is True
    assert target._chonknoris_pretrained is True
    bound = _load_conditioner(
        target,
        checkpoint,
        torch.device("cpu"),
        expected_manifest_digest="expected",
        expected_parent_checkpoint="/tmp/phase4b-parent.pt",
        require_parent_binding=True,
    )
    assert bound["parent_checkpoint"] == "/tmp/phase4b-parent.pt"
    try:
        _load_conditioner(
            target,
            checkpoint,
            torch.device("cpu"),
            expected_manifest_digest="expected",
            expected_parent_checkpoint="/tmp/different-parent.pt",
            require_parent_binding=True,
        )
    except ValueError as error:
        assert "different parent" in str(error)
    else:
        raise AssertionError("parent binding mismatch must be rejected")
    try:
        _load_conditioner(target, checkpoint, torch.device("cpu"), expected_manifest_digest="other")
    except ValueError as error:
        assert "manifest digest" in str(error)
    else:
        raise AssertionError("manifest mismatch must be rejected")


def _parent_with_correction_scale(value: float):
    return SimpleNamespace(
        dense_decoder=SimpleNamespace(
            correction_scale=nn.Parameter(torch.tensor(float(value)))
        )
    )


def test_enabled_parent_correction_preserves_checkpoint_amplitude():
    parent = _parent_with_correction_scale(0.02854149043560028)
    info = _apply_parent_correction_policy(
        parent, {"parent_correction_scale": 1.0}
    )
    assert float(parent.dense_decoder.correction_scale) == 0.02854149043560028
    assert info["policy"] == "enabled_preserve_checkpoint"
    assert info["effective_scale"] == info["checkpoint_scale"]


def test_disabled_parent_correction_zeros_checkpoint_amplitude():
    parent = _parent_with_correction_scale(0.02854149043560028)
    info = _apply_parent_correction_policy(
        parent, {"parent_correction_scale": 0.0}
    )
    assert float(parent.dense_decoder.correction_scale) == 0.0
    assert info["policy"] == "disabled_coarse_only"
    assert info["checkpoint_scale"] > 0.0


def test_parent_correction_policy_rejects_nonbinary_legacy_value():
    parent = _parent_with_correction_scale(0.02854149043560028)
    try:
        _apply_parent_correction_policy(
            parent, {"parent_correction_scale": 0.5}
        )
    except ValueError as error:
        assert "must be 0 or 1" in str(error)
    else:
        raise AssertionError("nonbinary correction policy must be rejected")
