import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import scripts.diagnose_saved_time_temporal_three_record_overfit as overfit_diagnostic
from saved_time_phase_operator_v4.band_adapter import BandLimitedFamilyAdapter
from scripts.diagnose_saved_time_temporal_three_record_overfit import (
    _candidate_config,
    apply_diagnostic_overrides,
    build_repeat_schedule,
    configure_overfit_trainable_stage,
    localize_remote_paths,
    resolve_microbatch_records,
    resolve_training_sampling,
    resolve_warmstart_paths,
    select_one_index_per_family,
)


def test_selects_one_split_relative_record_from_each_family():
    records = (
        SimpleNamespace(split="validation", medium_type="uniform"),
        SimpleNamespace(split="train", medium_type="layered"),
        SimpleNamespace(split="train", medium_type="uniform"),
        SimpleNamespace(split="train", medium_type="marmousi"),
        SimpleNamespace(split="train", medium_type="uniform"),
    )

    assert select_one_index_per_family(records, split="train") == (1, 0, 2)


def test_record_selection_rejects_a_missing_family():
    records = (
        SimpleNamespace(split="train", medium_type="uniform"),
        SimpleNamespace(split="train", medium_type="layered"),
    )

    with pytest.raises(ValueError, match="marmousi"):
        select_one_index_per_family(records, split="train")


def test_repeat_schedule_advances_exact_time_appearance():
    schedule = build_repeat_schedule((1, 0, 2), updates=3)

    assert tuple(spec.record_indices for spec in schedule) == ((1, 0, 2),) * 3
    assert tuple(spec.appearance_indices for spec in schedule) == (
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
    )


def test_remote_artifact_paths_are_localized_recursively():
    value = {
        "checkpoint": "/root/autodl-tmp/home/jiayh/Data/FNO/checkpoint.pt",
        "nested": ["/data/jiayh/identity.json", 3],
    }

    assert localize_remote_paths(value) == {
        "checkpoint": "/home/jiayh/Data/FNO/checkpoint.pt",
        "nested": ["/data/jiayh/identity.json", 3],
    }


def test_candidate_registers_discriminative_temporal_learning_rate(tmp_path):
    identity = tmp_path / "run_identity.json"
    identity.write_text(
        json.dumps(
            {
                "config": {
                    "checkpoint_transfer": {},
                    "residual_recovery": {},
                    "loss": {"delta": 0.5},
                    "optimizer": {"weight_decay": 1.0e-6},
                }
            }
        )
    )

    config = _candidate_config(
        identity,
        tmp_path / "parent.pt",
        tmp_path / "artifact",
        dense_learning_rate=1.0e-4,
        temporal_basis_learning_rate=1.0e-5,
        delta_weight=0.0,
        family_gradient_weights={
            "uniform": 0.5,
            "layered": 0.25,
            "marmousi": 2.25,
        },
    )

    assert config["optimizer"]["dense_learning_rate"] == pytest.approx(1.0e-4)
    assert config["optimizer"]["temporal_basis_learning_rate"] == pytest.approx(
        1.0e-5
    )
    assert config["loss"]["delta"] == pytest.approx(0.0)
    assert config["family_gradient_weights"] == pytest.approx(
        {"uniform": 0.5, "layered": 0.25, "marmousi": 2.25}
    )


def test_candidate_config_overrides_are_applied_to_loaded_yaml():
    config = {
        "loss": {"delta": 0.5, "delta_energy_floor_fraction": 0.1},
        "optimizer": {
            "dense_learning_rate": 1.0e-5,
            "temporal_basis_learning_rate": 1.0e-5,
            "gradient_clip_mode": "prefix_limits",
            "gradient_clip_prefix_limits": {"default": 1.0, "dense_decoder": 20.0},
            "weight_decay": 1.0e-6,
        },
    }

    result = apply_diagnostic_overrides(
        config,
        dense_learning_rate=3.0e-5,
        temporal_basis_learning_rate=2.0e-5,
        delta_weight=0.0,
        dense_gradient_clip_limit=100.0,
    )

    assert result["optimizer"]["dense_learning_rate"] == pytest.approx(3.0e-5)
    assert result["optimizer"]["temporal_basis_learning_rate"] == pytest.approx(
        2.0e-5
    )
    assert result["optimizer"]["gradient_clip_prefix_limits"][
        "dense_decoder"
    ] == pytest.approx(100.0)
    assert result["optimizer"]["weight_decay"] == pytest.approx(0.0)
    assert result["loss"]["delta"] == pytest.approx(0.0)


def test_candidate_config_registers_split_band_adapter_learning_rates():
    config = {
        "loss": {"delta": 0.5},
        "optimizer": {"weight_decay": 1.0e-6},
    }

    result = apply_diagnostic_overrides(
        config,
        dense_learning_rate=3.0e-4,
        temporal_basis_learning_rate=1.0e-5,
        delta_weight=0.0,
        band_adapter_feature_learning_rate=3.0e-4,
        band_adapter_output_learning_rate=1.0e-5,
    )

    assert result["optimizer"][
        "band_adapter_feature_learning_rate"
    ] == pytest.approx(3.0e-4)
    assert result["optimizer"][
        "band_adapter_output_learning_rate"
    ] == pytest.approx(1.0e-5)


@pytest.mark.parametrize(
    ("feature_lr", "output_lr"),
    (
        (1.0e-4, None),
        (None, 1.0e-5),
        (0.0, 1.0e-5),
        (1.0e-4, float("nan")),
    ),
)
def test_candidate_config_rejects_incomplete_or_invalid_band_adapter_rates(
    feature_lr, output_lr
):
    config = {
        "loss": {"delta": 0.5},
        "optimizer": {"weight_decay": 1.0e-6},
    }

    with pytest.raises(ValueError, match="band adapter"):
        apply_diagnostic_overrides(
            config,
            dense_learning_rate=3.0e-4,
            temporal_basis_learning_rate=1.0e-5,
            delta_weight=0.0,
            band_adapter_feature_learning_rate=feature_lr,
            band_adapter_output_learning_rate=output_lr,
        )


def test_pure_metric_override_disables_auxiliaries_and_family_reweighting():
    config = {
        "family_gradient_weights": {
            "uniform": 0.5,
            "layered": 0.25,
            "marmousi": 2.25,
        },
        "loss": {
            "delta": 0.5,
            "temporal_difference": 0.1,
            "spatial_gradient": 0.2,
            "spectrum": 0.3,
        },
        "optimizer": {"weight_decay": 1.0e-6},
    }

    result = apply_diagnostic_overrides(
        config,
        dense_learning_rate=3.0e-5,
        temporal_basis_learning_rate=2.0e-5,
        delta_weight=0.0,
        auxiliary_loss_scale=0.0,
        equal_family_weights=True,
    )

    assert result["loss"] == pytest.approx(
        {
            "delta": 0.0,
            "delta_energy_floor_fraction": 0.1,
            "temporal_difference": 0.0,
            "spatial_gradient": 0.0,
            "spectrum": 0.0,
        }
    )
    assert result["family_gradient_weights"] == pytest.approx(
        {"uniform": 1.0, "layered": 1.0, "marmousi": 1.0}
    )


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan")))
def test_candidate_config_rejects_invalid_dense_clip_limit(value):
    config = {
        "loss": {"delta": 0.5},
        "optimizer": {
            "gradient_clip_mode": "prefix_limits",
            "gradient_clip_prefix_limits": {"dense_decoder": 20.0},
        },
    }

    with pytest.raises(ValueError, match="clip"):
        apply_diagnostic_overrides(
            config,
            dense_learning_rate=1.0e-4,
            temporal_basis_learning_rate=1.0e-5,
            delta_weight=0.5,
            dense_gradient_clip_limit=value,
        )


def test_fixed_training_sampling_reuses_validation_panel_size():
    assert resolve_training_sampling(
        "validation_fixed", training_frames=16, validation_frames=32
    ) == ("validation_fixed", 32)


def test_appearance_training_sampling_keeps_requested_frame_count():
    assert resolve_training_sampling(
        "appearance16", training_frames=24, validation_frames=32
    ) == ("appearance16", 24)


@pytest.mark.parametrize(
    ("policy", "training_frames", "validation_frames"),
    (("unknown", 16, 32), ("appearance16", 0, 32), ("validation_fixed", 16, 0)),
)
def test_training_sampling_rejects_invalid_inputs(
    policy, training_frames, validation_frames
):
    with pytest.raises(ValueError):
        resolve_training_sampling(
            policy,
            training_frames=training_frames,
            validation_frames=validation_frames,
        )


def test_microbatch_record_count_must_be_positive():
    assert resolve_microbatch_records(1) == 1
    with pytest.raises(ValueError, match="microbatch"):
        resolve_microbatch_records(0)


def test_overfit_stage_honors_dense_decoder_unfreeze_epoch():
    assert overfit_diagnostic.resolve_family_expert_overfit_stage(
        {"head_only_epochs": 1, "dense_unfreeze_epoch": 3}
    ) == {
        "epoch": 3,
        "head_only_epochs": 1,
        "dense_unfreeze_epoch": 3,
    }


def test_overfit_stage_uses_most_expressive_registered_shared_stage():
    assert overfit_diagnostic.resolve_family_expert_overfit_stage(
        {
            "head_only_epochs": 1,
            "dense_unfreeze_epoch": 3,
            "shared_unfreeze_epoch": 8,
            "geometry_unfreeze_epoch": 13,
            "backbone_unfreeze_epoch": 18,
        }
    ) == {
        "epoch": 18,
        "head_only_epochs": 1,
        "dense_unfreeze_epoch": 3,
        "shared_unfreeze_epoch": 8,
        "geometry_unfreeze_epoch": 13,
        "backbone_unfreeze_epoch": 18,
    }


def test_overfit_band_adapter_stage_precedes_family_unfreezing():
    model = nn.Module()
    model.parent = nn.Linear(4, 4)
    model.dense_decoder = nn.Module()
    model.dense_decoder.band_limited_adapter = BandLimitedFamilyAdapter(
        width=8,
        rank=4,
    )
    config = {
        "band_limited_adapter": {"adapter_only": True},
        "family_experts": {
            "head_only_epochs": 1,
            "dense_unfreeze_epoch": 2,
        },
    }

    stage = configure_overfit_trainable_stage(model, config)

    assert stage.trainable_prefixes == ("dense_decoder.band_limited_adapter",)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith(
            "dense_decoder.band_limited_adapter."
        )


def test_overfit_warmstart_requires_checkpoint_and_bound_identity(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    identity = tmp_path / "run_identity.json"
    checkpoint.write_bytes(b"checkpoint")
    identity.write_text(json.dumps({"run_digest": "digest"}))

    assert resolve_warmstart_paths(checkpoint, identity) == (
        checkpoint.resolve(),
        identity.resolve(),
    )
    assert resolve_warmstart_paths(None, None) == (None, None)

    with pytest.raises(ValueError, match="together"):
        resolve_warmstart_paths(checkpoint, None)
    identity.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        resolve_warmstart_paths(checkpoint, identity)


def test_best_checkpoint_roundtrip_restores_the_selected_weights(tmp_path: Path):
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.25)

    checkpoint = overfit_diagnostic.save_overfit_checkpoint(
        tmp_path,
        model=model,
        update=0,
        manifest_digest="manifest",
        config_digest="run",
        aggregate_relative_l2=0.40,
        extra_metrics={"scattering_relative_l2": 0.75},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with torch.no_grad():
        model.weight.fill_(9.0)

    overfit_diagnostic.restore_best_overfit_checkpoint(
        checkpoint,
        model=model,
        manifest_digest="manifest",
        config_digest="run",
        map_location="cpu",
    )

    assert checkpoint.name == "update_0000.pt"
    assert payload["metrics"]["scattering_relative_l2"] == pytest.approx(0.75)
    assert torch.equal(model.weight, torch.full_like(model.weight, 1.25))
    assert (tmp_path / "latest.pt").samefile(checkpoint)


def test_terminal_report_keeps_stage_and_anchor_reductions_distinct():
    full_metrics = {
        "aggregate_relative_l2": 0.38,
        "family_relative_l2": {
            "uniform": 0.40,
            "layered": 0.20,
            "marmousi": 0.45,
        },
        "relative_improvement_vs_coarse": 0.25,
    }

    terminal = overfit_diagnostic.build_overfit_terminal_report(
        updates_completed=100,
        baseline_relative_l2=0.40,
        best_fixed_relative_l2=0.39,
        best_checkpoint="/artifact/checkpoints/update_0090.pt",
        all_saved_metrics=full_metrics,
    )

    assert terminal["relative_reduction"] == pytest.approx(0.025)
    assert terminal["anchor_relative_reduction"] == pytest.approx(0.25)
    assert terminal["best_checkpoint"] == "/artifact/checkpoints/update_0090.pt"
    assert terminal["all_saved_metrics"] is full_metrics
