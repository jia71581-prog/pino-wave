import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.evaluate_saved_time_high_band_splice import (
    checkpoint_run_identity_path,
    high_band_anchor_delta,
    splice_low_mid_with_high_anchor,
)
from saved_time_phase_operator_v4.band_adapter import registered_high_band_mask
from scripts.prepare_saved_time_band_adapter_candidate import (
    V49_ARTIFACT_NAME,
    V49_RUN_DIGEST,
    build_band_adapter_pilot_config,
    validate_dataset_contract,
    validate_v49_parent_identity,
)


def test_splice_keeps_candidate_low_mid_and_anchor_high_modes():
    generator = torch.Generator().manual_seed(20260720)
    candidate = torch.randn(2, 3, 41, 41, generator=generator)
    anchor = torch.randn(2, 3, 41, 41, generator=generator)

    spliced = splice_low_mid_with_high_anchor(candidate, anchor)

    candidate_fft = torch.fft.rfft2(candidate.float(), norm="ortho")
    anchor_fft = torch.fft.rfft2(anchor.float(), norm="ortho")
    spliced_fft = torch.fft.rfft2(spliced.float(), norm="ortho")
    high = registered_high_band_mask(41, 41, spliced.device)
    assert (spliced_fft[..., high] - anchor_fft[..., high]).abs().max() < 2.0e-5
    assert (spliced_fft[..., ~high] - candidate_fft[..., ~high]).abs().max() < 2.0e-5
    assert high_band_anchor_delta(spliced, anchor) < 2.0e-5


@pytest.mark.parametrize(
    ("candidate", "anchor", "message"),
    (
        (torch.zeros(2, 41, 41), torch.zeros(2, 41, 41), "record,time,z,x"),
        (torch.zeros(1, 2, 41, 41), torch.zeros(1, 3, 41, 41), "same shape"),
        (
            torch.full((1, 2, 41, 41), float("nan")),
            torch.zeros(1, 2, 41, 41),
            "finite",
        ),
    ),
)
def test_splice_rejects_invalid_inputs(candidate, anchor, message):
    with pytest.raises(ValueError, match=message):
        splice_low_mid_with_high_anchor(candidate, anchor)


def test_checkpoint_identity_is_bound_to_checkpoint_run_directory(tmp_path: Path):
    checkpoint = tmp_path / "pilot" / "checkpoints" / "epoch_0001.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    identity = checkpoint.parent.parent / "run_identity.json"
    identity.write_text(json.dumps({"run_digest": "digest"}))

    assert checkpoint_run_identity_path(checkpoint) == identity.resolve()

    identity.unlink()
    with pytest.raises(FileNotFoundError, match="run identity"):
        checkpoint_run_identity_path(checkpoint)


def _v49_identity() -> dict[str, object]:
    return {
        "schema": "saved_time_v5_training_contract_recovery_v1",
        "run_digest": V49_RUN_DIGEST,
        "manifest_digest": "manifest",
        "time_axis_sha256": "time-digest",
        "config": {
            "artifact_dir": f"/remote/artifacts/{V49_ARTIFACT_NAME}",
            "base_config": "configs/grouped_v3/continuous_pilot.yaml",
            "checkpoint_transfer": {
                "allow_new_family_expert_parameters": True,
                "parent_optimizer_state": False,
            },
            "family_curriculum": {"stages": [{"epochs": 3}]},
            "family_experts": {
                "head_only_epochs": 1,
                "router_loss_weight": 0.01,
                "teacher_forced_routing": True,
            },
            "family_gradient_weights": {
                "uniform": 0.5,
                "layered": 0.25,
                "marmousi": 2.25,
            },
            "gate": {
                "pilot_epochs": 3,
                "target_aggregate_relative_l2": 0.1,
                "target_family_relative_l2": 0.12,
            },
            "loss": {
                "delta": 0.5,
                "temporal_difference": 0.1,
                "spatial_gradient": 0.1,
                "spectrum": 0.1,
                "hard_causality": True,
                "hard_causality_lead_cycles": 1.0,
            },
            "macro_records": 12,
            "macros_per_update": 8,
            "microbatch_records": 12,
            "optimizer": {
                "dense_learning_rate": 1.0e-5,
                "geometry_learning_rate": 2.0e-5,
                "backbone_learning_rate": 2.0e-6,
                "temporal_basis_learning_rate": 1.0e-5,
                "family_expert_learning_rate": 1.0e-4,
                "weight_decay": 1.0e-6,
                "schedule_epoch_offset": 8,
                "schedule_total_epochs": 40,
                "warmup_epochs": 2,
            },
            "residual_recovery": {
                "activation_mode": "preserve",
                "absorb_temporal_basis_gate": True,
            },
            "time_appearance_offset": 14,
            "time_policy": "appearance16",
            "validation": {
                "panel_records": 48,
                "frames_per_record": 32,
                "microbatch_records": 1,
            },
            "variant_overrides": {
                "temporal_basis_rank": 96,
                "family_expert_rank": 16,
            },
        },
    }


def test_build_band_adapter_candidate_changes_only_registered_branch(tmp_path: Path):
    identity = _v49_identity()

    config = build_band_adapter_pilot_config(
        identity,
        parent_checkpoint=tmp_path / "epoch_0001.pt",
        parent_checkpoint_identity=tmp_path / "run_identity.json",
        artifact_dir=tmp_path / "v60",
        band_adapter_rank=16,
        physical_microbatch_records=12,
    )

    assert config["variant_overrides"] == {
        "temporal_basis_rank": 96,
        "family_expert_rank": 16,
        "band_adapter_rank": 16,
    }
    assert config["checkpoint_transfer"]["parent_optimizer_state"] is False
    assert config["checkpoint_transfer"]["allow_new_band_adapter_parameters"] is True
    assert "allow_new_family_expert_parameters" not in config["checkpoint_transfer"]
    assert config["band_limited_adapter"] == {
        "adapter_only": True,
        "cutoff_normalized_radius": pytest.approx(2.0 / 3.0),
    }
    assert config["family_curriculum"]["stages"] == [
        {
            "epochs": 6,
            "macro_pattern": ["uniform", "layered", "marmousi"],
        }
    ]
    assert config["optimizer"]["dense_learning_rate"] == pytest.approx(1.0e-4)
    assert config["optimizer"]["weight_decay"] == pytest.approx(1.0e-6)
    assert config["family_experts"]["router_loss_weight"] == pytest.approx(0.0)
    assert config["macros_per_update"] * config["macro_records"] == 96
    assert config["epochs"] == 6
    assert config["gate"]["pilot_epochs"] == 6


def test_build_v61_candidate_registers_multiscale_fields(tmp_path: Path):
    config = build_band_adapter_pilot_config(
        _v49_identity(),
        parent_checkpoint=tmp_path / "epoch_0001.pt",
        parent_checkpoint_identity=tmp_path / "run_identity.json",
        artifact_dir=tmp_path / "v61",
        band_adapter_rank=32,
        band_adapter_architecture="multiscale_spectral",
        band_adapter_spectral_rank=32,
        band_adapter_modes=32,
        band_adapter_full_depth=4,
        band_adapter_coarse_depth=2,
        band_adapter_activation_checkpointing=True,
    )

    assert config["variant_overrides"] == {
        "temporal_basis_rank": 96,
        "family_expert_rank": 16,
        "band_adapter_rank": 32,
        "band_adapter_architecture": "multiscale_spectral",
        "band_adapter_spectral_rank": 32,
        "band_adapter_modes": 32,
        "band_adapter_full_depth": 4,
        "band_adapter_coarse_depth": 2,
        "band_adapter_activation_checkpointing": True,
    }
    assert config["band_limited_adapter"] == {
        "adapter_only": True,
        "architecture": "multiscale_spectral",
        "cutoff_normalized_radius": pytest.approx(2.0 / 3.0),
    }
    assert config["optimizer"][
        "band_adapter_feature_learning_rate"
    ] == pytest.approx(3.0e-4)
    assert config["optimizer"][
        "band_adapter_output_learning_rate"
    ] == pytest.approx(1.0e-5)


def test_build_v68_candidate_registers_dynamic_physical_conditioning(tmp_path: Path):
    config = build_band_adapter_pilot_config(
        _v49_identity(),
        parent_checkpoint=tmp_path / "epoch_0001.pt",
        parent_checkpoint_identity=tmp_path / "run_identity.json",
        artifact_dir=tmp_path / "v68",
        band_adapter_rank=32,
        band_adapter_architecture="dynamic_multiscale_spectral",
        band_adapter_preserve_high_band=False,
    )

    assert config["variant_overrides"]["band_adapter_architecture"] == (
        "dynamic_multiscale_spectral"
    )
    assert config["band_limited_adapter"]["architecture"] == (
        "dynamic_multiscale_spectral"
    )
    assert config["band_limited_adapter"]["physical_conditioning"] == {
        "medium": "raw_velocity_encoded_once",
        "source": "normalized_parameters",
    }
    assert "band_adapter_feature_learning_rate" in config["optimizer"]


def test_build_v62_candidate_can_disable_high_band_preservation(tmp_path: Path):
    config = build_band_adapter_pilot_config(
        _v49_identity(),
        parent_checkpoint=tmp_path / "epoch_0001.pt",
        parent_checkpoint_identity=tmp_path / "run_identity.json",
        artifact_dir=tmp_path / "v62",
        band_adapter_rank=32,
        band_adapter_architecture="multiscale_spectral",
        band_adapter_preserve_high_band=False,
    )

    assert config["variant_overrides"]["band_adapter_preserve_high_band"] is False
    assert config["band_limited_adapter"]["preserve_high_band"] is False


def test_legacy_v60_candidate_does_not_change_optimizer_schema(tmp_path: Path):
    config = build_band_adapter_pilot_config(
        _v49_identity(),
        parent_checkpoint=tmp_path / "epoch_0001.pt",
        parent_checkpoint_identity=tmp_path / "run_identity.json",
        artifact_dir=tmp_path / "v60",
        band_adapter_architecture="low_rank",
    )

    assert "band_adapter_feature_learning_rate" not in config["optimizer"]
    assert "band_adapter_output_learning_rate" not in config["optimizer"]


@pytest.mark.parametrize(
    ("feature_lr", "output_lr"),
    ((0.0, 1.0e-5), (3.0e-4, float("nan"))),
)
def test_v61_candidate_rejects_invalid_split_learning_rates(
    tmp_path: Path, feature_lr: float, output_lr: float
):
    with pytest.raises(ValueError, match="learning rate"):
        build_band_adapter_pilot_config(
            _v49_identity(),
            parent_checkpoint=tmp_path / "epoch_0001.pt",
            parent_checkpoint_identity=tmp_path / "run_identity.json",
            artifact_dir=tmp_path / "v61",
            band_adapter_architecture="multiscale_spectral",
            band_adapter_feature_learning_rate=feature_lr,
            band_adapter_output_learning_rate=output_lr,
        )


def test_build_band_adapter_candidate_registers_exact_training_panel_and_lr(tmp_path: Path):
    config = build_band_adapter_pilot_config(
        _v49_identity(),
        parent_checkpoint=tmp_path / "epoch_0001.pt",
        parent_checkpoint_identity=tmp_path / "run_identity.json",
        artifact_dir=tmp_path / "v60",
        training_frames_per_record=38,
        dense_learning_rate=3.0e-4,
    )

    assert config["training_frames_per_record"] == 38
    assert config["optimizer"]["dense_learning_rate"] == pytest.approx(3.0e-4)


def test_build_band_adapter_candidate_allows_full_per_rank_batch_of_twenty_four(
    tmp_path: Path,
):
    config = build_band_adapter_pilot_config(
        _v49_identity(),
        parent_checkpoint=tmp_path / "epoch_0001.pt",
        parent_checkpoint_identity=tmp_path / "run_identity.json",
        artifact_dir=tmp_path / "v60",
        physical_microbatch_records=24,
        macro_records=24,
        effective_batch_records=96,
    )

    assert config["microbatch_records"] == 24
    assert config["macro_records"] == 24
    assert config["macros_per_update"] == 4


def test_build_band_adapter_candidate_rejects_microbatch_larger_than_macro(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="within one macro"):
        build_band_adapter_pilot_config(
            _v49_identity(),
            parent_checkpoint=tmp_path / "epoch_0001.pt",
            parent_checkpoint_identity=tmp_path / "run_identity.json",
            artifact_dir=tmp_path / "v60",
            physical_microbatch_records=24,
            macro_records=12,
            effective_batch_records=96,
        )


@pytest.mark.parametrize(
    ("macro_records", "effective_batch"),
    ((0, 96), (24, 95), (24, 48)),
)
def test_build_band_adapter_candidate_rejects_invalid_four_gpu_batch_geometry(
    tmp_path: Path, macro_records: int, effective_batch: int
):
    with pytest.raises(ValueError, match="macro|effective batch|four-GPU"):
        build_band_adapter_pilot_config(
            _v49_identity(),
            parent_checkpoint=tmp_path / "epoch_0001.pt",
            parent_checkpoint_identity=tmp_path / "run_identity.json",
            artifact_dir=tmp_path / "v60",
            physical_microbatch_records=min(max(macro_records, 1), 24),
            macro_records=macro_records,
            effective_batch_records=effective_batch,
        )


@pytest.mark.parametrize(
    ("training_frames", "dense_lr"),
    ((15, 1.0e-4), (402, 1.0e-4), (38, 0.0), (38, float("nan"))),
)
def test_build_band_adapter_candidate_rejects_invalid_panel_or_lr(
    tmp_path: Path, training_frames: int, dense_lr: float
):
    with pytest.raises(ValueError, match="training frames|learning rate"):
        build_band_adapter_pilot_config(
            _v49_identity(),
            parent_checkpoint=tmp_path / "epoch_0001.pt",
            parent_checkpoint_identity=tmp_path / "run_identity.json",
            artifact_dir=tmp_path / "v60",
            training_frames_per_record=training_frames,
            dense_learning_rate=dense_lr,
        )


@pytest.mark.parametrize("physical", (0, 25))
def test_build_band_adapter_candidate_rejects_invalid_physical_batch(
    tmp_path: Path, physical: int
):
    with pytest.raises(ValueError, match="physical microbatch"):
        build_band_adapter_pilot_config(
            _v49_identity(),
            parent_checkpoint=tmp_path / "epoch_0001.pt",
            parent_checkpoint_identity=tmp_path / "run_identity.json",
            artifact_dir=tmp_path / "v60",
            physical_microbatch_records=physical,
        )


@pytest.mark.parametrize("rank", (0, 65))
def test_band_adapter_candidate_rejects_invalid_rank(tmp_path: Path, rank: int):
    with pytest.raises(ValueError, match="rank"):
        build_band_adapter_pilot_config(
            _v49_identity(),
            parent_checkpoint=tmp_path / "epoch_0001.pt",
            parent_checkpoint_identity=tmp_path / "run_identity.json",
            artifact_dir=tmp_path / "v60",
            band_adapter_rank=rank,
        )


def test_band_adapter_candidate_rejects_non_v49_or_simultaneous_expansion(tmp_path: Path):
    wrong = _v49_identity()
    wrong["run_digest"] = "other"
    with pytest.raises(ValueError, match="V49"):
        validate_v49_parent_identity(wrong)

    with pytest.raises(ValueError, match="staged"):
        build_band_adapter_pilot_config(
            _v49_identity(),
            parent_checkpoint=tmp_path / "epoch_0001.pt",
            parent_checkpoint_identity=tmp_path / "run_identity.json",
            artifact_dir=tmp_path / "v60",
            extra_variant_overrides={"modes": 101},
        )

    with pytest.raises(ValueError, match="optimizer"):
        build_band_adapter_pilot_config(
            _v49_identity(),
            parent_checkpoint=tmp_path / "epoch_0001.pt",
            parent_checkpoint_identity=tmp_path / "run_identity.json",
            artifact_dir=tmp_path / "v60",
            restore_parent_optimizer=True,
        )


def _write_contract(path: Path, *, saved_shape: str = "[201, 201]") -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "acoustic_lwc84_401_to_201_v1"
        handle.attrs["solver_grid_shape"] = "[401, 401]"
        handle.attrs["saved_grid_shape"] = saved_shape
        handle.attrs["free_surface"] = "node-centred p(z=0)=0"
        handle.attrs["cpml"] = "left,right,bottom; no top CPML"
        handle.create_dataset("time_s", data=np.linspace(0.0, 1.0, 5))
        handle.create_dataset("source_x_m", data=np.zeros(2))
        handle.create_dataset("source_z_m", data=np.zeros(2))
        handle.create_dataset("source_f0_hz", data=np.ones(2))
        handle.create_dataset("source_t0_s", data=np.zeros(2))
        handle.create_dataset("source_amplitude", data=np.ones(2))
        handle.create_dataset("source_map", shape=(2, 201, 201), dtype="f4")
        handle.create_dataset("velocity_mps", shape=(2, 201, 201), dtype="f4")
        handle.create_dataset("wavefield", shape=(2, 5, 201, 201), dtype="f4")


def test_dataset_contract_binds_single_source_boundary_and_output_shape(tmp_path: Path):
    source = tmp_path / "dataset.h5"
    _write_contract(source)

    report = validate_dataset_contract(source)

    assert report["solver_grid_shape"] == [401, 401]
    assert report["wavefield_shape"] == [201, 201]
    assert report["one_source_per_record"] is True
    assert report["receiver_input"] is False
    assert report["stored_time_count"] == 5
    assert len(report["time_axis_sha256"]) == 64


def test_dataset_contract_rejects_non_201_wavefields(tmp_path: Path):
    source = tmp_path / "dataset.h5"
    _write_contract(source, saved_shape="[101, 101]")

    with pytest.raises(ValueError, match="201"):
        validate_dataset_contract(source)
