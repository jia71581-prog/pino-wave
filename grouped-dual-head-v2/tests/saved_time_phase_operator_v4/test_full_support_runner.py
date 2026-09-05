import json

import pytest
import torch
import yaml
from types import SimpleNamespace

import scripts.train_saved_time_v4_full_support as full_support_runner

from scripts.train_saved_time_v4_full_support import (
    _coverage_update,
    _set_epoch_learning_rates,
    adaptive_record_schedule_parameters,
    backed_off_learning_rate_multiplier,
    checkpoint_is_eligible,
    build_update_report,
    clip_trainable_gradients,
    epoch_step_ranges,
    epoch_gate_validation_plan,
    evaluation_time_selector_seed,
    epoch_metric_evaluation_split,
    epoch_retry_exhausted,
    family_expert_missing_prefixes,
    family_expert_identity_report,
    family_expert_stage_epoch,
    backward_with_isolated_family_weighting,
    parent_manifest_transfer_metadata,
    parent_checkpoint_expected_manifest_digest,
    microbatch_records_for_epoch,
    coupled_2d_gradient_norms,
    coupled_2d_missing_prefixes,
    local_differential_gradient_norms,
    recovery_stage_epoch,
    resolve_epoch_validation_control,
    run_epochs_for_mode,
    pilot_gate,
    pilot_gate_on_main,
    pilot_terminal_exit_code,
    probe_variant_for_config,
    validation_panel_indices,
    validation_scope,
    validation_plan,
    validation_score_improved,
    recovery_time_indices,
    restore_parent_optimizer_state,
    training_dense_time_block,
    coverage_frames_per_appearance,
    training_frames_per_record,
    velocity_fields_per_record,
)
from saved_time_phase_operator_v4.full_support import (
    FullSupportStepSpec,
    warmup_cosine_factor,
)
from saved_time_phase_operator_v4.sampling import validation_time_indices
from saved_time_phase_operator_v4.spectral import FactorizedComplexResidualStack
from saved_time_phase_operator_v4.experts import FamilyRoutedResidualExperts
from scripts.supervise_saved_time_v5_recovery import pilot_allows_production


def test_velocity_fields_per_record_expands_deduplicated_media():
    velocity = torch.stack(
        (torch.full((3, 4), 1500.0), torch.full((3, 4), 2400.0))
    )
    expanded = velocity_fields_per_record(
        velocity, torch.tensor([0, 0, 1, 1, 0])
    )

    assert expanded.shape == (5, 3, 4)
    torch.testing.assert_close(expanded[0], velocity[0])
    torch.testing.assert_close(expanded[2], velocity[1])
    expanded_channel = velocity_fields_per_record(
        velocity[:, None], torch.tensor([1, 0, 1])
    )
    assert expanded_channel.shape == (3, 3, 4)
    torch.testing.assert_close(expanded_channel[0], velocity[1])
    with pytest.raises(ValueError, match="outside"):
        velocity_fields_per_record(velocity, torch.tensor([0, 2]))


def test_smoke_uses_pilot_epoch_horizon_for_three_epoch_family_schedule():
    assert run_epochs_for_mode(
        registered_epochs=40,
        pilot_epochs=3,
        pilot=False,
        smoke_updates=2,
    ) == 3
    assert run_epochs_for_mode(
        registered_epochs=40,
        pilot_epochs=3,
        pilot=True,
        smoke_updates=0,
    ) == 3
    assert run_epochs_for_mode(
        registered_epochs=40,
        pilot_epochs=3,
        pilot=False,
        smoke_updates=0,
    ) == 40


def test_train_bound_epoch_metric_stays_train_only_in_smoke():
    config = {
        "epoch_validation_control": {
            "enabled": True,
            "evaluation_split": "train",
        }
    }
    disabled_for_smoke = resolve_epoch_validation_control(
        config, pilot_or_smoke=True
    )

    assert disabled_for_smoke["enabled"] is False
    assert epoch_metric_evaluation_split(
        config, disabled_for_smoke, smoke=True
    ) == "train"
    assert epoch_metric_evaluation_split(
        config, disabled_for_smoke, smoke=False
    ) == "validation"


def test_training_dense_time_block_defaults_to_one_and_requires_positive_integer():
    assert training_dense_time_block({"optimizer": {}}) == 1
    assert training_dense_time_block(
        {"optimizer": {"training_time_block": 2}}
    ) == 2
    with pytest.raises(ValueError, match="training_time_block"):
        training_dense_time_block(
            {"optimizer": {"training_time_block": 0}}
        )


def test_training_frames_per_record_is_optional_and_requires_positive_integer():
    assert training_frames_per_record({}) is None
    assert training_frames_per_record({"training_frames_per_record": 32}) == 32
    with pytest.raises(ValueError, match="training_frames_per_record"):
        training_frames_per_record({"training_frames_per_record": 0})


def test_fixed_train_gate_policy_preserves_registered_frame_compute():
    assert coverage_frames_per_appearance(
        {"time_policy": "fixed_train_gate", "training_frames_per_record": 24}
    ) == 24


def test_fixed_train_gate_coverage_uses_same_zero_offset_time_selector():
    axis = torch.linspace(0.0, 1.0, 401).numpy()
    coverage = __import__("numpy").zeros((1, len(axis)), dtype=bool)
    spec = FullSupportStepSpec(
        step=7,
        epoch=2,
        record_indices=(0,),
        appearance_indices=(19,),
    )

    _coverage_update(
        coverage,
        (spec,),
        time_s=axis,
        source_t0_s=(0.15,),
        source_f0_hz=(10.0,),
        sample_ids=("train:gate-aligned",),
        seed=372,
        frames_per_appearance=24,
        time_policy="fixed_train_gate",
    )
    expected = validation_time_indices(
        axis,
        source_t0_s=0.15,
        source_f0_hz=10.0,
        sample_id="train:gate-aligned",
        panel_offset=0,
        seed=372,
        count=24,
    )

    assert coverage.sum() == 24
    assert coverage[0, expected].all()


def test_fixed_train_gate_requires_explicit_zero_gate_seed_offset():
    misaligned = {
        "seed": 372,
        "time_policy": "fixed_train_gate",
        "epoch_validation_control": {
            "enabled": True,
            "evaluation_split": "train",
        },
    }
    with pytest.raises(ValueError, match="time_selector_seed_offset=0"):
        resolve_epoch_validation_control(misaligned, pilot_or_smoke=False)

    aligned = {
        **misaligned,
        "epoch_validation_control": {
            **misaligned["epoch_validation_control"],
            "time_selector_seed_offset": 0,
        },
    }
    control = resolve_epoch_validation_control(aligned, pilot_or_smoke=False)
    assert control["time_selector_seed_offset"] == 0
    assert evaluation_time_selector_seed(aligned) == 372


def test_zero_gate_seed_offset_makes_24_training_frames_subset_of_32_gate_frames():
    axis = torch.linspace(0.0, 1.0, 401).numpy()
    config = {
        "seed": 372,
        "epoch_validation_control": {"time_selector_seed_offset": 0},
    }
    common = {
        "time_s": axis,
        "source_t0_s": 0.15,
        "source_f0_hz": 19.0,
        "sample_id": "train:gate-aligned",
        "panel_offset": 0,
    }
    training = validation_time_indices(seed=372, count=24, **common)
    gate = validation_time_indices(
        seed=evaluation_time_selector_seed(config), count=32, **common
    )

    assert len(set(training) & set(gate)) == 24


def test_uniform_random_record_oversampling_preserves_compute_without_rad_bias():
    weights, oversample = adaptive_record_schedule_parameters(
        {
            "enabled": True,
            "record_axis": True,
            "record_axis_strategy": "uniform",
            "record_oversample": 2.0,
        },
        ("uniform", "layered", "marmousi", "layered"),
    )

    assert oversample == pytest.approx(2.0)
    assert weights.tolist() == pytest.approx([0.25] * 4)


def test_rad_record_sampling_remains_the_default_strategy():
    config = {
        "enabled": True,
        "record_axis": True,
        "record_oversample": 2.0,
        "family_errors": {"uniform": 0.1, "layered": 0.3, "marmousi": 0.6},
    }

    implicit, implicit_oversample = adaptive_record_schedule_parameters(
        config, ("uniform", "layered", "marmousi")
    )
    explicit, explicit_oversample = adaptive_record_schedule_parameters(
        {**config, "record_axis_strategy": "rad"},
        ("uniform", "layered", "marmousi"),
    )

    assert implicit.tolist() == pytest.approx(explicit.tolist())
    assert implicit_oversample == explicit_oversample == pytest.approx(2.0)
    assert implicit[2] > implicit[1] > implicit[0]


@pytest.mark.parametrize("strategy", ("sequential", "", "RAD"))
def test_record_sampling_rejects_unknown_strategy(strategy):
    with pytest.raises(ValueError, match="record_axis_strategy"):
        adaptive_record_schedule_parameters(
            {
                "enabled": True,
                "record_axis": True,
                "record_axis_strategy": strategy,
            },
            ("uniform",),
        )


def test_epoch_ranges_align_four_macros_per_update():
    assert epoch_step_ranges(total_macros=376, macros_per_epoch=188) == (
        (0, 188),
        (188, 376),
    )
    with pytest.raises(ValueError, match="divide"):
        epoch_step_ranges(total_macros=377, macros_per_epoch=188)


def test_update_report_records_loss_gradient_learning_rate_and_gpu_metrics():
    report = build_update_report(
        epoch=2,
        update_index=3,
        updates_per_epoch=47,
        global_step=50,
        loss_components={"total": 1.2, "delta": 0.4},
        gradient_norm=2.5,
        gradient_norms={"dense_decoder": 2.0},
        gradient_clipping={
            "total_before": 40.0,
            "total_after": 20.0,
            "prefixes": {
                "dense_decoder": {
                    "before": 40.0,
                    "after": 20.0,
                    "limit": 20.0,
                    "scale": 0.5,
                }
            },
        },
        learning_rates={"dense_decay": 5.0e-5},
        physical_microbatch_records=8,
        gpu={"utilization_percent": 100.0, "power_w": 345.0, "memory_mib": 21400.0},
        elapsed_seconds=123.0,
    )

    assert report["event"] == "optimizer_update"
    assert report["epoch"] == 2
    assert report["update"] == 3
    assert report["updates_per_epoch"] == 47
    assert report["global_step"] == 50
    assert report["loss_components"]["total"] == pytest.approx(1.2)
    assert report["gradient_norm_before_clip"] == pytest.approx(2.5)
    assert report["gradient_clipping"]["total_after"] == pytest.approx(20.0)
    assert report["gradient_clipping"]["prefixes"]["dense_decoder"][
        "scale"
    ] == pytest.approx(0.5)
    assert report["learning_rates"]["dense_decay"] == pytest.approx(5.0e-5)
    assert report["gpu"]["power_w"] == pytest.approx(345.0)


def test_local_differential_gradient_audit_tracks_gate_then_features():
    stack = FactorizedComplexResidualStack(
        width=8,
        spectral_rank=8,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        local_differential_residual=True,
    )
    model = SimpleNamespace(dense_decoder=SimpleNamespace(stack=stack))
    value = torch.randn(1, 8, 17, 19)
    stack(value).square().mean().backward()

    first = local_differential_gradient_norms(model)

    assert first["local_differential_gate"] > 0.0
    assert first["local_differential_features"] == 0.0
    stack.zero_grad(set_to_none=True)
    stack.blocks[0].local_differential.scale.data.fill_(1.0e-3)
    stack(value).square().mean().backward()
    second = local_differential_gradient_norms(model)
    assert second["local_differential_gate"] > 0.0
    assert second["local_differential_features"] > 0.0


def test_coupled_2d_gradient_audit_tracks_gate_then_features():
    stack = FactorizedComplexResidualStack(
        width=4,
        spectral_rank=4,
        modes=6,
        depth=1,
        activation_checkpointing=False,
        coupled_2d_rank=2,
    )
    model = SimpleNamespace(dense_decoder=SimpleNamespace(stack=stack))
    value = torch.randn(1, 4, 201, 201)
    stack(value).square().mean().backward()

    first = coupled_2d_gradient_norms(model)

    assert first["coupled_2d_gate"] > 0.0
    assert first["coupled_2d_features"] == 0.0
    stack.zero_grad(set_to_none=True)
    stack.blocks[0].coupled_2d.scale.data.fill_(1.0e-3)
    stack(value).square().mean().backward()
    second = coupled_2d_gradient_norms(model)
    assert second["coupled_2d_gate"] > 0.0
    assert second["coupled_2d_features"] > 0.0


def test_learning_rate_schedule_continues_parent_epoch_and_clamps_at_floor():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 2.0, "initial_lr": 2.0}]
    )
    config = {
        "optimizer": {
            "warmup_epochs": 2,
            "minimum_factor": 0.05,
            "schedule_epoch_offset": 4,
            "schedule_total_epochs": 40,
        }
    }

    factor = _set_epoch_learning_rates(
        optimizer, config, epoch_index=0, total_epochs=2
    )
    expected = warmup_cosine_factor(
        4, total_epochs=40, warmup_epochs=2, minimum_factor=0.05
    )
    assert factor == pytest.approx(expected)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0 * expected)

    floor = _set_epoch_learning_rates(
        optimizer, config, epoch_index=100, total_epochs=2
    )
    assert floor == pytest.approx(0.05)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_epoch_validation_backoff_scales_the_registered_schedule():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 2.0, "initial_lr": 2.0}]
    )
    config = {
        "optimizer": {
            "warmup_epochs": 1,
            "minimum_factor": 0.1,
            "schedule_total_epochs": 40,
        }
    }

    factor = _set_epoch_learning_rates(
        optimizer,
        config,
        epoch_index=0,
        total_epochs=40,
        control_multiplier=0.5,
    )

    assert factor == pytest.approx(0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_epoch_validation_controller_is_strict_and_backoff_is_bounded():
    assert validation_score_improved(0.5, 0.49)
    assert not validation_score_improved(0.5, 0.5)
    assert not validation_score_improved(0.5, 0.51)
    assert not validation_score_improved(float("nan"), 0.49)
    assert not validation_score_improved(
        0.5, 0.4995, minimum_absolute_improvement=1.0e-3
    )
    assert backed_off_learning_rate_multiplier(
        1.0, backoff=0.5, minimum=0.125
    ) == pytest.approx(0.5)
    assert backed_off_learning_rate_multiplier(
        0.125, backoff=0.5, minimum=0.125
    ) == pytest.approx(0.125)


def test_epoch_retry_stops_when_the_distinct_learning_rate_ladder_ends():
    assert not epoch_retry_exhausted(
        attempt=3,
        maximum_attempts=8,
        current_multiplier=0.125,
        next_multiplier=0.0625,
    )
    assert epoch_retry_exhausted(
        attempt=7,
        maximum_attempts=8,
        current_multiplier=0.0078125,
        next_multiplier=0.0078125,
    )
    assert epoch_retry_exhausted(
        attempt=8,
        maximum_attempts=8,
        current_multiplier=0.015625,
        next_multiplier=0.0078125,
    )


def test_epoch_validation_control_is_disabled_for_pilot_and_validates_policy():
    config = {
        "epoch_validation_control": {
            "enabled": True,
            "metric": "aggregate_relative_l2",
            "learning_rate_backoff": 0.5,
            "minimum_learning_rate_multiplier": 1.0 / 256.0,
            "maximum_attempts_per_epoch": 8,
        }
    }

    assert resolve_epoch_validation_control(
        config, pilot_or_smoke=False
    )["enabled"]
    assert not resolve_epoch_validation_control(
        config, pilot_or_smoke=True
    )["enabled"]
    assert resolve_epoch_validation_control(
        config, pilot_or_smoke=False
    )["evaluation_split"] == "validation"


def test_epoch_control_can_use_train_truth_without_changing_the_default():
    config = {
        "epoch_validation_control": {
            "enabled": True,
            "evaluation_split": "train",
        }
    }

    assert resolve_epoch_validation_control(
        config, pilot_or_smoke=False
    )["evaluation_split"] == "train"
    with pytest.raises(ValueError, match="evaluation_split"):
        resolve_epoch_validation_control(
            {
                "epoch_validation_control": {
                    "enabled": True,
                    "evaluation_split": "test_id",
                }
            },
            pilot_or_smoke=False,
        )


def test_parent_optimizer_restore_is_identity_bound_and_opt_in(tmp_path, monkeypatch):
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(
        json.dumps({"run_digest": "parent-run", "manifest_digest": "old-manifest"})
    )
    calls = []

    def record_load(path, **kwargs):
        calls.append((path, kwargs))
        kwargs["optimizer"].param_groups[0]["lr"] = 2.0e-6
        kwargs["optimizer"].param_groups[0]["initial_lr"] = 2.0e-6

    monkeypatch.setattr(full_support_runner, "load_checkpoint", record_load)
    model = object()
    optimizer = SimpleNamespace(
        param_groups=[
            {
                "group_name": "backbone_decay",
                "lr": 1.0e-5,
                "initial_lr": 1.0e-5,
            }
        ]
    )
    base_identity = {"run_digest": "base", "manifest_digest": "old-manifest"}
    config = {
        "parent_checkpoint": str(tmp_path / "parent.pt"),
        "parent_checkpoint_identity": str(identity_path),
        "checkpoint_transfer": {
            "allow_parent_manifest_mismatch": True,
            "parent_optimizer_state": True,
        },
    }

    assert restore_parent_optimizer_state(
        config,
        model=model,
        optimizer=optimizer,
        active_manifest_digest="new-manifest",
        parent_identity=base_identity,
        device=torch.device("cpu"),
    )
    assert calls[0][0] == config["parent_checkpoint"]
    assert calls[0][1]["model"] is model
    assert calls[0][1]["optimizer"] is optimizer
    assert calls[0][1]["expected_manifest_digest"] == "old-manifest"
    assert calls[0][1]["expected_config_digest"] == "parent-run"
    assert calls[0][1]["restore_rng"] is False
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(1.0e-5)

    calls.clear()
    config["checkpoint_transfer"]["parent_optimizer_state"] = False
    assert not restore_parent_optimizer_state(
        config,
        model=model,
        optimizer=optimizer,
        active_manifest_digest="new-manifest",
        parent_identity=base_identity,
        device=torch.device("cpu"),
    )
    assert calls == []


def test_prefix_gradient_clipping_does_not_suppress_small_conditioning_modules():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense_decoder = torch.nn.Linear(1, 1, bias=False)
            self.fusion = torch.nn.Linear(1, 1, bias=False)

    model = ToyModel()
    model.dense_decoder.weight.grad = torch.tensor([[40.0]])
    model.fusion.weight.grad = torch.tensor([[0.5]])

    norm = clip_trainable_gradients(model, maximum_norm=1.0, mode="prefix")

    assert norm == pytest.approx((40.0**2 + 0.5**2) ** 0.5)
    assert model.dense_decoder.weight.grad.item() == pytest.approx(1.0)
    assert model.fusion.weight.grad.item() == pytest.approx(0.5)


def test_prefix_limits_mode_uses_explicit_optimizer_limits():
    model = torch.nn.Module()
    model.dense_decoder = torch.nn.Linear(1, 1, bias=False)
    model.dense_decoder.weight.grad = torch.tensor([[40.0]])

    total, telemetry = clip_trainable_gradients(
        model,
        maximum_norm=1.0,
        mode="prefix_limits",
        prefix_limits={"dense_decoder": 20.0, "default": 1.0},
        return_report=True,
    )

    assert total == pytest.approx(40.0)
    assert telemetry["prefixes"]["dense_decoder"]["after"] == pytest.approx(
        20.0
    )


def test_global_gradient_clipping_preserves_legacy_shared_scaling():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    model[0].weight.grad = torch.tensor([[4.0]])

    norm = clip_trainable_gradients(model, maximum_norm=2.0, mode="global")

    assert norm == pytest.approx(4.0)
    assert model[0].weight.grad.item() == pytest.approx(2.0)


def test_gradient_clipping_rejects_unknown_mode():
    with pytest.raises(ValueError, match="gradient clip mode"):
        clip_trainable_gradients(torch.nn.Linear(1, 1), maximum_norm=1.0, mode="mixed")


def test_recovery_time_indices_are_moved_from_batch_metadata():
    batch = SimpleNamespace(left_index=torch.tensor([[3, 4, 8, 9]]))

    result = recovery_time_indices(batch, torch.device("cpu"))

    assert torch.equal(result, batch.left_index)
    assert result.device.type == "cpu"


def test_parent_manifest_transfer_is_explicit_and_audited():
    parent = {"manifest_digest": "old-manifest"}
    config = {"checkpoint_transfer": {"allow_parent_manifest_mismatch": True}}

    metadata = parent_manifest_transfer_metadata(
        config,
        parent_identity=parent,
        active_manifest_digest="new-manifest",
    )

    assert metadata == {
        "allowed": True,
        "parent_manifest_digest": "old-manifest",
        "active_manifest_digest": "new-manifest",
        "reason": "explicit_checkpoint_weight_transfer_after_dataset_repair",
    }
    assert (
        parent_checkpoint_expected_manifest_digest(
            config,
            active_manifest_digest="new-manifest",
            checkpoint_identity=parent,
        )
        == "old-manifest"
    )


def test_config_can_override_checkpoint_compatible_probe_architecture():
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }

    legacy = probe_variant_for_config({}, identity)
    coupled = probe_variant_for_config(
        {"variant_overrides": {"coupled_axes": True}}, identity
    )

    assert not legacy.coupled_axes
    assert coupled.coupled_axes
    assert coupled.depth == legacy.depth == 8


def test_config_can_enable_local_differential_architecture_override():
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }

    candidate = probe_variant_for_config(
        {"variant_overrides": {"local_differential_residual": True}}, identity
    )

    assert candidate.local_differential_residual


def test_config_can_enable_rank16_coupled_2d_architecture_override():
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }

    candidate = probe_variant_for_config(
        {"variant_overrides": {"coupled_2d_rank": 16}}, identity
    )

    assert candidate.coupled_2d_rank == 16


def test_multiscale_band_adapter_overrides_are_registered():
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }
    config = {
        "variant_overrides": {
            "band_adapter_rank": 32,
            "band_adapter_architecture": "multiscale_spectral",
            "band_adapter_spectral_rank": 32,
            "band_adapter_modes": 32,
            "band_adapter_full_depth": 4,
            "band_adapter_coarse_depth": 2,
            "band_adapter_activation_checkpointing": True,
            "band_adapter_dropout": 0.05,
            "band_adapter_preserve_high_band": False,
        }
    }

    variant = probe_variant_for_config(config, identity)

    assert variant.band_adapter_architecture == "multiscale_spectral"
    assert variant.band_adapter_rank == 32
    assert variant.band_adapter_spectral_rank == 32
    assert variant.band_adapter_modes == 32
    assert variant.band_adapter_full_depth == 4
    assert variant.band_adapter_coarse_depth == 2
    assert variant.band_adapter_activation_checkpointing is True
    assert variant.band_adapter_dropout == pytest.approx(0.05)
    assert variant.band_adapter_preserve_high_band is False


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("band_adapter_architecture", "unknown"),
        ("band_adapter_spectral_rank", 0),
        ("band_adapter_modes", 102),
        ("band_adapter_full_depth", 0),
        ("band_adapter_coarse_depth", 0),
        ("band_adapter_activation_checkpointing", 1),
        ("band_adapter_dropout", -0.01),
        ("band_adapter_dropout", 1.0),
        ("band_adapter_dropout", "0.05"),
        ("band_adapter_preserve_high_band", 1),
    ),
)
def test_multiscale_band_adapter_rejects_invalid_overrides(name, value):
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }
    overrides = {
        "band_adapter_rank": 32,
        "band_adapter_architecture": "multiscale_spectral",
        "band_adapter_spectral_rank": 32,
        "band_adapter_modes": 32,
        "band_adapter_full_depth": 4,
        "band_adapter_coarse_depth": 2,
        "band_adapter_activation_checkpointing": True,
    }
    overrides[name] = value

    with pytest.raises(ValueError, match="band adapter"):
        probe_variant_for_config({"variant_overrides": overrides}, identity)


def test_multiscale_band_adapter_requires_enabled_adapter_rank():
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }

    with pytest.raises(ValueError, match="band adapter rank"):
        probe_variant_for_config(
            {
                "variant_overrides": {
                    "band_adapter_architecture": "multiscale_spectral",
                }
            },
            identity,
        )


@pytest.mark.parametrize("invalid", [True, -1, 33, 4.0, "16"])
def test_coupled_2d_rank_rejects_invalid_values(invalid):
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }

    with pytest.raises(ValueError, match="coupled_2d_rank"):
        probe_variant_for_config(
            {"variant_overrides": {"coupled_2d_rank": invalid}}, identity
        )


def test_coupled_2d_checkpoint_expansion_is_explicit_and_prefix_bounded():
    parent = full_support_runner.ProbeVariant(
        depth=8, use_local_phase=True, modes=101, coupled_2d_rank=0
    )
    candidate = full_support_runner.ProbeVariant(
        depth=8, use_local_phase=True, modes=101, coupled_2d_rank=16
    )
    config = {
        "checkpoint_transfer": {
            "allow_new_coupled_2d_parameters": True,
            "parent_optimizer_state": False,
        }
    }

    prefixes = coupled_2d_missing_prefixes(
        config,
        parent_variant=parent,
        candidate_variant=candidate,
        block_count=2,
    )

    assert prefixes == (
        "dense_decoder.stack.blocks.0.coupled_2d.",
        "dense_decoder.stack.blocks.1.coupled_2d.",
    )


def test_family_expert_checkpoint_expansion_is_explicit_and_prefix_bounded():
    parent = full_support_runner.ProbeVariant(
        depth=8, use_local_phase=True, modes=101, family_expert_rank=0
    )
    candidate = full_support_runner.ProbeVariant(
        depth=8, use_local_phase=True, modes=101, family_expert_rank=16
    )
    config = {
        "checkpoint_transfer": {
            "allow_new_family_expert_parameters": True,
            "parent_optimizer_state": False,
        }
    }

    prefixes = family_expert_missing_prefixes(
        config,
        parent_variant=parent,
        candidate_variant=candidate,
    )

    assert prefixes == ("dense_decoder.family_experts.",)


def test_family_expert_checkpoint_expansion_rejects_unsafe_transfer():
    parent = full_support_runner.ProbeVariant(
        depth=8, use_local_phase=True, modes=101, family_expert_rank=0
    )
    candidate = full_support_runner.ProbeVariant(
        depth=8, use_local_phase=True, modes=101, family_expert_rank=16
    )

    with pytest.raises(ValueError, match="explicit transfer permission"):
        family_expert_missing_prefixes(
            {}, parent_variant=parent, candidate_variant=candidate
        )
    with pytest.raises(ValueError, match="optimizer"):
        family_expert_missing_prefixes(
            {
                "checkpoint_transfer": {
                    "allow_new_family_expert_parameters": True,
                    "parent_optimizer_state": True,
                }
            },
            parent_variant=parent,
            candidate_variant=candidate,
        )


def test_family_expert_identity_report_requires_zero_output_heads():
    module = FamilyRoutedResidualExperts(width=8, rank=4)
    model = SimpleNamespace(
        dense_decoder=SimpleNamespace(family_experts=module)
    )

    report = family_expert_identity_report(model, expected_rank=4)

    assert report == {
        "family_expert_rank": 4,
        "expert_count": 3,
        "exact_parent_identity": True,
    }
    module.experts[0].output.weight.data.fill_(1.0e-3)
    with pytest.raises(ValueError, match="exact parent identity"):
        family_expert_identity_report(model, expected_rank=4)


def test_expert_gradients_are_not_double_weighted_by_shared_family_scale():
    shared = torch.nn.Parameter(torch.tensor(1.0))
    expert = torch.nn.Parameter(torch.tensor(1.0))
    field_loss = shared + expert
    router_loss = 3.0 * expert

    backward_with_isolated_family_weighting(
        field_loss,
        router_loss=router_loss,
        expert_parameters=(expert,),
        piece_weight=0.5,
        family_scale=2.0,
        router_weight=0.01,
    )

    assert shared.grad.item() == pytest.approx(1.0)
    assert expert.grad.item() == pytest.approx(0.5 + 0.015)


def test_family_expert_memory_probe_can_force_the_fully_unfrozen_stage():
    config = {"family_experts": {"stage_epoch_offset": 1}}

    assert family_expert_stage_epoch(config, epoch=1) == 2
    assert family_expert_stage_epoch({"family_experts": {}}, epoch=1) == 1

    with pytest.raises(ValueError, match="stage epoch offset"):
        family_expert_stage_epoch(
            {"family_experts": {"stage_epoch_offset": -1}}, epoch=1
        )


def test_family_expert_microbatch_shrinks_as_shared_paths_unfreeze():
    config = {
        "microbatch_records": 4,
        "macro_records": 12,
        "macros_per_update": 16,
        "family_experts": {
            "stage_epoch_offset": 2,
            "shared_unfreeze_epoch": 5,
            "geometry_unfreeze_epoch": 7,
            "backbone_unfreeze_epoch": 9,
        },
    }

    assert microbatch_records_for_epoch(config, epoch=1) == 4
    assert microbatch_records_for_epoch(config, epoch=3) == 3
    assert microbatch_records_for_epoch(config, epoch=5) == 2
    assert microbatch_records_for_epoch(config, epoch=7) == 2


def test_family_expert_microbatch_shrinks_when_dense_decoder_unfreezes():
    config = {
        "microbatch_records": 12,
        "macro_records": 12,
        "macros_per_update": 8,
        "family_experts": {
            "head_only_epochs": 1,
            "dense_unfreeze_epoch": 3,
            "shared_unfreeze_epoch": 8,
            "geometry_unfreeze_epoch": 13,
            "backbone_unfreeze_epoch": 18,
            "stage_epoch_offset": 2,
        },
    }

    assert microbatch_records_for_epoch(config, epoch=1) == 3


def test_coupled_2d_checkpoint_expansion_rejects_unsafe_transfer_modes():
    parent = full_support_runner.ProbeVariant(
        depth=8, use_local_phase=True, modes=32, coupled_2d_rank=0
    )
    candidate = full_support_runner.ProbeVariant(
        depth=8, use_local_phase=True, modes=101, coupled_2d_rank=16
    )
    base = {"checkpoint_transfer": {"parent_optimizer_state": False}}

    with pytest.raises(ValueError, match="allow_new_coupled_2d_parameters"):
        coupled_2d_missing_prefixes(
            base,
            parent_variant=parent,
            candidate_variant=full_support_runner.ProbeVariant(
                depth=8, use_local_phase=True, modes=32, coupled_2d_rank=16
            ),
            block_count=8,
        )
    with pytest.raises(ValueError, match="staged"):
        coupled_2d_missing_prefixes(
            {
                "checkpoint_transfer": {
                    "allow_new_coupled_2d_parameters": True,
                    "parent_optimizer_state": False,
                }
            },
            parent_variant=parent,
            candidate_variant=candidate,
            block_count=8,
        )
    with pytest.raises(ValueError, match="optimizer"):
        coupled_2d_missing_prefixes(
            {
                "checkpoint_transfer": {
                    "allow_new_coupled_2d_parameters": True,
                    "parent_optimizer_state": True,
                }
            },
            parent_variant=full_support_runner.ProbeVariant(
                depth=8, use_local_phase=True, modes=101, coupled_2d_rank=0
            ),
            candidate_variant=full_support_runner.ProbeVariant(
                depth=8, use_local_phase=True, modes=101, coupled_2d_rank=16
            ),
            block_count=8,
        )


def test_config_can_expand_to_all_201_grid_rfft_modes():
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }

    candidate = probe_variant_for_config(
        {"variant_overrides": {"modes": 101}}, identity
    )

    assert candidate.modes == 101


@pytest.mark.parametrize("invalid", [True, 32.0, 0, 102])
def test_all_mode_override_rejects_invalid_values(invalid):
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }

    with pytest.raises(ValueError, match="modes"):
        probe_variant_for_config(
            {"variant_overrides": {"modes": invalid}}, identity
        )


def test_spectral_mode_expansion_requires_explicit_weight_only_transfer():
    parent_identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }
    checkpoint_identity = {"config": {}}
    config = {
        "variant_overrides": {"modes": 101},
        "checkpoint_transfer": {
            "allow_spectral_mode_expansion": True,
            "parent_optimizer_state": False,
        },
    }

    parent, candidate, expands = full_support_runner.resolve_spectral_mode_transfer(
        config,
        parent_identity=parent_identity,
        checkpoint_identity=checkpoint_identity,
    )

    assert parent.modes == 32
    assert candidate.modes == 101
    assert expands


def test_spectral_mode_expansion_rejects_missing_permission_or_optimizer_restore():
    parent_identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }
    checkpoint_identity = {"config": {}}
    base = {"variant_overrides": {"modes": 101}}

    with pytest.raises(ValueError, match="allow_spectral_mode_expansion"):
        full_support_runner.resolve_spectral_mode_transfer(
            base,
            parent_identity=parent_identity,
            checkpoint_identity=checkpoint_identity,
        )
    with pytest.raises(ValueError, match="optimizer"):
        full_support_runner.resolve_spectral_mode_transfer(
            {
                **base,
                "checkpoint_transfer": {
                    "allow_spectral_mode_expansion": True,
                    "parent_optimizer_state": True,
                },
            },
            parent_identity=parent_identity,
            checkpoint_identity=checkpoint_identity,
        )


def test_completed_all_mode_checkpoint_needs_no_second_expansion():
    parent_identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }
    checkpoint_identity = {
        "config": {"variant_overrides": {"modes": 101}}
    }
    config = {
        "variant_overrides": {"modes": 101},
        "checkpoint_transfer": {"parent_optimizer_state": True},
    }

    parent, candidate, expands = full_support_runner.resolve_spectral_mode_transfer(
        config,
        parent_identity=parent_identity,
        checkpoint_identity=checkpoint_identity,
    )

    assert parent.modes == candidate.modes == 101
    assert not expands


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_variant_override_rejects_non_boolean_local_differential(invalid):
    identity = {
        "variant_config": {
            "depth": 8,
            "use_local_phase": True,
            "spectral_rank": 112,
            "modes": 32,
        }
    }

    with pytest.raises(ValueError, match="must be boolean"):
        probe_variant_for_config(
            {"variant_overrides": {"local_differential_residual": invalid}},
            identity,
        )


def test_parent_manifest_transfer_is_rejected_by_default():
    parent = {"manifest_digest": "old-manifest"}

    with pytest.raises(ValueError, match="parent manifest identity mismatch"):
        parent_manifest_transfer_metadata(
            {},
            parent_identity=parent,
            active_manifest_digest="new-manifest",
        )


def test_validation_scope_rotates_and_expands_every_fifth_epoch():
    assert validation_scope(epoch=1, validation_records=480) == ("panel", 48)
    assert validation_scope(epoch=5, validation_records=480) == (
        "all_records",
        480,
    )


def test_recovery_validation_plan_is_fixed_and_uses_all_401_frames_every_fifth_epoch():
    config = {
        "seed": 307,
        "validation": {
            "panel_records": 48,
            "frames_per_record": 32,
            "full_panel_records": 48,
            "all_records_every": 5,
            "final_frames_per_record": 401,
        },
    }

    first = validation_plan(config, epoch=1, validation_records=480, stored_time_count=401)
    second = validation_plan(config, epoch=2, validation_records=480, stored_time_count=401)
    fifth = validation_plan(config, epoch=5, validation_records=480, stored_time_count=401)

    assert first == second
    assert first[0] == "fixed_panel"
    assert len(first[1]) == 48
    assert first[2:] == ("validation_fixed", 32)
    assert fifth[0] == "fixed_full_time_panel"
    assert fifth[1] == first[1]
    assert fifth[2:] == ("all_saved", 401)


def test_epoch_gate_validation_plan_never_changes_protocol():
    config = {
        "seed": 307,
        "validation": {"panel_records": 48, "frames_per_record": 32},
    }

    plan = epoch_gate_validation_plan(config, validation_records=480)

    assert plan[0] == "fixed_epoch_gate"
    assert len(plan[1]) == 48
    assert plan[2:] == ("validation_fixed", 32)


def test_validation_panels_are_deterministic_complete_and_epoch_distinct():
    first = validation_panel_indices(
        validation_records=480, panel_records=48, epoch=1, seed=307
    )
    repeated = validation_panel_indices(
        validation_records=480, panel_records=48, epoch=1, seed=307
    )
    second = validation_panel_indices(
        validation_records=480, panel_records=48, epoch=2, seed=307
    )
    assert first == repeated
    assert first != second
    assert len(first) == len(set(first)) == 48
    assert min(first) >= 0 and max(first) < 480


def test_three_epoch_gate_requires_best_nonincrease_and_family_safety():
    reports = [
        {"score": 0.56, "family": {"uniform": 0.55, "layered": 0.58}},
        {"score": 0.54, "family": {"uniform": 0.54, "layered": 0.56}},
        {"score": 0.53, "family": {"uniform": 0.53, "layered": 0.55}},
    ]
    assert pilot_gate(reports, family_tolerance=0.03)["passed"]

    regressed = list(reports)
    regressed[-1] = {
        "score": 0.53,
        "family": {"uniform": 0.53, "layered": 0.61},
    }
    result = pilot_gate(regressed, family_tolerance=0.03)
    assert not result["passed"]
    assert not result["family_safe"]


def test_non_main_ddp_rank_does_not_evaluate_main_only_pilot_reports():
    assert pilot_gate_on_main((), is_main=False, family_tolerance=0.03) is None


def test_external_evidence_gate_owns_exit_code_after_complete_pilot_metrics():
    assert pilot_terminal_exit_code("complete", external_evidence_gate=True) == 0
    assert pilot_terminal_exit_code(
        "pilot_gate_failed", external_evidence_gate=True
    ) == 0
    assert pilot_terminal_exit_code(
        "pilot_gate_failed", external_evidence_gate=False
    ) == 2


def test_checkpoint_selection_does_not_compare_moving_family_coarse_anchor():
    metrics = {"relative_improvement_vs_coarse": -0.01}

    assert checkpoint_is_eligible(
        {"residual_recovery": {}, "family_experts": {"head_only_epochs": 1}},
        metrics=metrics,
        validation_scope="pilot_fixed_panel",
        pilot_or_smoke=True,
    )
    assert not checkpoint_is_eligible(
        {"residual_recovery": {"decoder_only_epochs": 2}},
        metrics=metrics,
        validation_scope="pilot_fixed_panel",
        pilot_or_smoke=True,
    )
    assert checkpoint_is_eligible(
        {},
        metrics=metrics,
        validation_scope="fixed_full_time_panel",
        pilot_or_smoke=False,
    )
    assert not checkpoint_is_eligible(
        {},
        metrics=metrics,
        validation_scope="rotating_panel",
        pilot_or_smoke=False,
    )


def test_recovery_gate_requires_improvement_over_coarse_and_bounded_correction():
    reports = [
        {
            "score": 0.56,
            "family": {"uniform": 0.55, "layered": 0.58},
            "improvement_vs_coarse": -0.01,
            "correction_ratio": 0.04,
        },
        {
            "score": 0.54,
            "family": {"uniform": 0.53, "layered": 0.56},
            "improvement_vs_coarse": 0.03,
            "correction_ratio": 0.08,
        },
    ]
    assert pilot_gate(reports, family_tolerance=0.03)["passed"]

    reports[-1]["improvement_vs_coarse"] = -0.02
    result = pilot_gate(reports, family_tolerance=0.03)
    assert not result["passed"]
    assert not result["better_than_coarse"]

    reports[-1]["improvement_vs_coarse"] = 0.03
    reports[-1]["correction_ratio"] = 0.5
    result = pilot_gate(reports, family_tolerance=0.03)
    assert not result["passed"]
    assert not result["correction_bounded"]


def test_registered_full_support_config_matches_the_production_contract():
    with open(
        "configs/saved_time_v4/full_support_adamw_batch48.yaml", encoding="utf8"
    ) as handle:
        config = yaml.safe_load(handle)

    assert config["epochs"] == 50
    assert config["macro_records"] == 12
    assert config["macros_per_update"] == 4
    assert config["microbatch_records"] == 12
    assert config["workers"] == 8
    assert config["prefetch_factor"] == 4
    assert config["validation"]["panel_records"] == 48
    assert config["validation"]["microbatch_records"] == 4
    assert config["validation"]["all_records_every"] == 5
    assert config["validation"]["final_frames_per_record"] == 401
    assert config["optimizer"]["full_forward_checkpointing"] is True


def test_residual_recovery_config_uses_dense_time_supervision_and_fixed_validation():
    with open(
        "configs/saved_time_v4/v6_residual_recovery.yaml", encoding="utf8"
    ) as handle:
        config = yaml.safe_load(handle)

    assert config["time_policy"] == "appearance16"
    assert config["microbatch_records"] == 8
    assert config["validation"]["frames_per_record"] == 32
    assert config["validation"]["final_frames_per_record"] == 401
    assert config["loss"]["delta"] == pytest.approx(0.5)
    assert config["loss"]["temporal_difference"] == pytest.approx(0.1)
    assert config["loss"]["hard_causality"] is False
    assert config["residual_recovery"]["correction_scale"] == pytest.approx(1.0)
    assert config["residual_recovery"]["output_std"] == pytest.approx(1.0e-4)
    assert config["residual_recovery"]["decoder_only_epochs"] == 2


def test_supervisor_launches_production_only_after_a_passing_pilot_gate():
    assert pilot_allows_production(
        {"status": "complete", "pilot_gate": {"passed": True}}
    )
    assert not pilot_allows_production(
        {"status": "pilot_gate_failed", "pilot_gate": {"passed": False}}
    )
    assert not pilot_allows_production(
        {"status": "complete", "pilot_gate": {"passed": False}}
    )


def test_physical_microbatch_shrinks_as_more_backbone_stages_unfreeze():
    config = {"microbatch_records": 12, "macro_records": 12, "macros_per_update": 4}

    assert microbatch_records_for_epoch(config, epoch=1) == 12
    assert microbatch_records_for_epoch(config, epoch=2) == 12
    assert microbatch_records_for_epoch(config, epoch=3) == 8
    assert microbatch_records_for_epoch(config, epoch=5) == 8
    assert microbatch_records_for_epoch(config, epoch=6) == 4


def test_recovery_microbatch_drops_when_parent_dense_modules_unfreeze():
    config = {
        "microbatch_records": 8,
        "macro_records": 12,
        "macros_per_update": 4,
        "residual_recovery": {"decoder_only_epochs": 2},
    }

    assert microbatch_records_for_epoch(config, epoch=1) == 8
    assert microbatch_records_for_epoch(config, epoch=2) == 8
    # Rank-96 temporal decoding reached 23.41 GiB and failed its next 140 MiB
    # allocation when source/fusion first unfroze at physical batch four.
    assert microbatch_records_for_epoch(config, epoch=3) == 3
    assert microbatch_records_for_epoch(config, epoch=4) == 3
    assert microbatch_records_for_epoch(config, epoch=5) == 2
    assert microbatch_records_for_epoch(config, epoch=7) == 2
    assert microbatch_records_for_epoch(config, epoch=8) == 2


def test_recovery_stage_offset_continues_the_parent_checkpoint_schedule():
    config = {
        "microbatch_records": 8,
        "macro_records": 12,
        "macros_per_update": 4,
        "residual_recovery": {
            "decoder_only_epochs": 2,
            "stage_epoch_offset": 3,
        },
    }

    assert recovery_stage_epoch(config, epoch=1) == 4
    assert recovery_stage_epoch(config, epoch=2) == 5
    # A four-record physical batch left less than 100 MiB free on the
    # production 24 GiB GPU and already caused a recovery-stage OOM.  Three
    # records sustains full utilization while preserving safe headroom.
    assert microbatch_records_for_epoch(config, epoch=1) == 3
    # The first geometry-unfrozen update at physical batch three consumed
    # 23.43 GiB and failed its next 104 MiB FFT allocation on a 24 GiB card.
    assert microbatch_records_for_epoch(config, epoch=2) == 2
    assert microbatch_records_for_epoch(config, epoch=5) == 2


def test_recovery_stage_offset_rejects_negative_or_nonpositive_epochs():
    config = {"residual_recovery": {"stage_epoch_offset": -1}}

    with pytest.raises(ValueError, match="offset"):
        recovery_stage_epoch(config, epoch=1)
    with pytest.raises(ValueError, match="epoch"):
        recovery_stage_epoch({"residual_recovery": {}}, epoch=0)
