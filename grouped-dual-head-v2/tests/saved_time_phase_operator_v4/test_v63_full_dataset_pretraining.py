from pathlib import Path

import yaml

from saved_time_phase_operator_v4.full_support import (
    audit_family_curriculum_epoch_schedule,
    build_family_curriculum_schedule,
)
from saved_time_phase_operator_v4.losses import (
    relative_energy_squared_block_loss,
    relative_energy_squared_reference,
)
from scripts.train_saved_time_v4_full_support import (
    coverage_frames_per_appearance,
    training_loss_time_slices,
)
from scripts.gate_saved_time_v63_full_dataset import v63_candidate_gate


ROOT = Path(__file__).resolve().parents[2]
PILOT_CONFIG = (
    ROOT
    / "configs/saved_time_v4/generated/v63_full_dataset_allband_pilot_4gpu.yaml"
)
LONG_CONFIG = (
    ROOT
    / "configs/saved_time_v4/generated/v64_full_dataset_allband_long_4gpu.yaml"
)


def test_coverage_uses_actual_registered_training_frame_count():
    config = {
        "time_policy": "appearance16",
        "training_frames_per_record": 24,
    }

    assert coverage_frames_per_appearance(config) == 24


def test_training_loss_time_slices_cover_every_frame_once():
    config = {"optimizer": {"training_loss_time_block": 4}}

    slices = training_loss_time_slices(config, frame_count=22)

    assert [(start, stop) for start, stop, _ in slices] == [
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 22),
    ]
    assert sum(weight for _, _, weight in slices) == 1.0


def test_relative_energy_squared_loss_is_exactly_time_decomposable():
    import torch

    generator = torch.Generator().manual_seed(17)
    target = torch.randn(2, 5, 8, 8, generator=generator)
    full_prediction = torch.randn(2, 5, 8, 8, generator=generator, requires_grad=True)
    block_prediction = full_prediction.detach().clone().requires_grad_(True)
    reference = relative_energy_squared_reference(target)

    full = relative_energy_squared_block_loss(
        full_prediction, target, reference=reference, spectrum_weight=0.05
    )
    blocked = sum(
        relative_energy_squared_block_loss(
            block_prediction[:, start:stop],
            target[:, start:stop],
            reference=reference,
            spectrum_weight=0.05,
        ).total
        for start, stop in ((0, 2), (2, 4), (4, 5))
    )

    assert torch.allclose(blocked, full.total, rtol=1.0e-6, atol=1.0e-7)
    full.total.backward()
    blocked.backward()
    assert torch.allclose(
        block_prediction.grad, full_prediction.grad, rtol=1.0e-5, atol=1.0e-6
    )


def test_v63_pilot_registers_full_dataset_exact_time_gpu_contract():
    config = yaml.safe_load(PILOT_CONFIG.read_text())

    assert "saved_time_v49_family_experts" in config["parent_checkpoint"]
    assert "/1/pretraining/" in config["artifact_dir"]
    assert config["epochs"] == config["gate"]["pilot_epochs"] == 6
    assert config["time_policy"] == "appearance16"
    assert config["training_frames_per_record"] == 22
    assert config["macro_records"] == 24
    assert config["macros_per_update"] == 4
    assert config["microbatch_records"] == 24
    assert config["macro_records"] * config["macros_per_update"] == 96
    assert config["workers"] == 16
    assert config["prefetch_factor"] == 2
    assert config["optimizer"]["adamw_implementation"] == "fused"
    assert config["optimizer"]["full_forward_checkpointing"] is False
    assert config["optimizer"]["training_loss_time_block"] == 4
    assert config["optimizer"]["training_loss_objective"] == "squared_relative_energy"
    assert config["variant_overrides"] == {
        "temporal_basis_rank": 96,
        "family_expert_rank": 16,
        "band_adapter_rank": 32,
        "band_adapter_preserve_high_band": False,
        "band_adapter_architecture": "multiscale_spectral",
        "band_adapter_spectral_rank": 32,
        "band_adapter_modes": 32,
        "band_adapter_full_depth": 4,
        "band_adapter_coarse_depth": 2,
        "band_adapter_activation_checkpointing": True,
    }
    assert config["checkpoint_transfer"]["allow_new_band_adapter_parameters"] is True
    assert config["band_limited_adapter"]["preserve_high_band"] is False


def test_v63_curriculum_reaches_complete_three_family_support():
    config = yaml.safe_load(PILOT_CONFIG.read_text())
    families = ("uniform",) * 420 + ("layered",) * 1120 + ("marmousi",) * 700

    schedule = build_family_curriculum_schedule(
        families,
        stages=config["family_curriculum"]["stages"],
        epochs=config["epochs"],
        macro_records=config["macro_records"],
        macros_per_update=config["macros_per_update"],
        seed=config["seed"],
        appearance_offset=config["time_appearance_offset"],
    )
    audits = [
        audit_family_curriculum_epoch_schedule(
            schedule,
            epoch=epoch,
            record_families=families,
            macros_per_update=config["macros_per_update"],
        )
        for epoch in range(config["epochs"])
    ]

    assert [audit.record_count for audit in audits] == [420, 1540, 2240, 2240, 2240, 2240]
    assert all(audit.appearances == 2304 for audit in audits)
    assert audits[2].family_appearances == {
        "layered": 1152,
        "marmousi": 720,
        "uniform": 432,
    }
    assert all(audit.optimizer_updates == 24 for audit in audits)


def test_v64_long_continues_from_pilot_best_without_second_expansion():
    config = yaml.safe_load(LONG_CONFIG.read_text())

    assert config["epochs"] == 40
    assert config["parent_checkpoint"].endswith("/pilot/best.pt")
    assert config["parent_checkpoint_identity"].endswith("/pilot/run_identity.json")
    assert "allow_new_band_adapter_parameters" not in config["checkpoint_transfer"]
    assert config["optimizer"]["training_loss_time_block"] == 4
    assert config["family_curriculum"]["stages"] == [
        {
            "epochs": 40,
            "macro_pattern": [
                "uniform",
                "uniform",
                "uniform",
                "layered",
                "layered",
                "layered",
                "layered",
                "layered",
                "layered",
                "layered",
                "layered",
                "marmousi",
                "marmousi",
                "marmousi",
                "marmousi",
                "marmousi",
            ],
        }
    ]


def _metric_row(aggregate, *, uniform, layered, marmousi, epoch=1):
    return {
        "event": "epoch",
        "epoch": epoch,
        "global_step": 24 * epoch,
        "validation_scope": "pilot_fixed_panel",
        "metrics": {
            "aggregate_relative_l2": aggregate,
            "family_relative_l2": {
                "uniform": uniform,
                "layered": layered,
                "marmousi": marmousi,
            },
            "record_count": 48,
            "frame_count": 48 * 32,
        },
    }


def _smoke_row():
    return {
        "event": "epoch",
        "epoch": 1,
        "global_step": 2,
        "validation_scope": "smoke",
        "train_loss": 0.3,
        "gradient_norms": {"dense_decoder.band_limited_adapter": 1.0},
        "physical_microbatch_records": 24,
        "peak_cuda_bytes": 22 * 1024**3,
        "ddp": {"world_size": 4, "global_macros_per_update": 4},
    }


def test_v63_gate_requires_resource_safe_strict_heldout_improvement():
    report = v63_candidate_gate(
        parent_row=_metric_row(
            0.4691, uniform=0.3967, layered=0.4373, marmousi=0.5391
        ),
        candidate_row=_metric_row(
            0.42, uniform=0.38, layered=0.41, marmousi=0.51, epoch=6
        ),
        smoke_row=_smoke_row(),
        smoke_terminal={"status": "complete", "global_step": 2},
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert report["passes"] is True
    assert all(report["checks"].values())
    assert report["candidate"]["epoch"] == 6


def test_v63_gate_rejects_training_panel_regression_or_wrong_batch():
    candidate = _metric_row(
        0.47, uniform=0.38, layered=0.41, marmousi=0.58, epoch=6
    )
    smoke = _smoke_row()
    smoke["physical_microbatch_records"] = 8

    report = v63_candidate_gate(
        parent_row=_metric_row(
            0.4691, uniform=0.3967, layered=0.4373, marmousi=0.5391
        ),
        candidate_row=candidate,
        smoke_row=smoke,
        smoke_terminal={"status": "complete", "global_step": 2},
        family_tolerance=0.03,
        maximum_peak_cuda_gib=23.0,
    )

    assert report["passes"] is False
    assert report["checks"]["aggregate_improved"] is False
    assert report["checks"]["families_safe"] is False
    assert report["checks"]["physical_microbatch_24"] is False
